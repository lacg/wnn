"""
Per-Cluster Connectivity Optimization.

REPLACES the global GA/TS approach which was essentially a random walk.

Instead of optimizing all neurons together (which mixes unrelated clusters),
this optimizes each cluster's neurons independently with discriminative fitness.

Key insight: Each cluster is an independent optimization problem.
- Cluster 0's neurons should fire HIGH for "the" and LOW for other tokens
- Cluster 1's neurons should fire HIGH for "of" and LOW for other tokens
- etc.

Discriminative fitness (PENALIZE_HIGH_VOTES mode, default):
- Positive signal: How strongly does this cluster vote when it SHOULD win?
- Negative signal: How strongly does this cluster vote when it should NOT win?
- Fitness = positive_strength - negative_strength

This is analogous to margin-based loss in neural networks.

Usage:
    optimizer = PerClusterOptimizer(
        tier_configs=[
            TierOptConfig(tier=0, ga_gens=100, ts_iters=50),  # Most effort on frequent
            TierOptConfig(tier=1, ga_gens=30, ts_iters=20),   # Medium effort
            TierOptConfig(tier=2, ga_gens=10, ts_iters=5),    # Minimal effort on rare
        ],
        fitness_mode=FitnessMode.PENALIZE_HIGH_VOTES,  # Default, smoothest gradient
        cluster_order="random",  # Avoid ordering bias
    )
    result = optimizer.optimize_all_tiers()

Tier budgets follow Pareto principle:
- Tier 0 (frequent tokens, ~40% of predictions): Most optimization effort
- Tier 1 (medium tokens, ~15% of predictions): Medium effort
- Tier 2+ (rare tokens, <5% of predictions): Minimal effort
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Optional
from enum import IntEnum
import multiprocessing as mp
import random

import torch
from torch import Tensor


class FitnessMode(IntEnum):
    """
    How to compute discriminative fitness for a cluster.

    Higher values = more sophisticated discrimination.
    Default is PENALIZE_HIGH_VOTES (smoothest gradient for optimization).
    """
    POSITIVE_ONLY = 1       # Only reward correct predictions for this cluster
    PENALIZE_WINS = 2       # Also penalize when cluster wrongly wins
    PENALIZE_HIGH_VOTES = 3 # Penalize high votes even if doesn't win (smoothest)
    CROSS_ENTROPY = 4       # Use actual cross-entropy with global softmax
    SIMPLE = 5              # Simple: +vote when correct, -vote when wrong (intuitive)
    ACCURACY = 6            # Count wins when should win (directly optimizes top-1)


@dataclass
class TierOptConfig:
    """Configuration for optimizing one tier."""
    tier: int
    ga_gens: int = 50           # GA generations for this tier
    ga_population: int = 30     # GA population size
    ts_iters: int = 20          # TS iterations for this tier
    ts_neighbors: int = 30      # TS neighbors per iteration
    mutation_rate: float = 0.01
    enabled: bool = True        # Can disable optimization for specific tiers

    def __repr__(self):
        if not self.enabled:
            return f"Tier{self.tier}(disabled)"
        return f"Tier{self.tier}(GA:{self.ga_gens}gen, TS:{self.ts_iters}iter)"


@dataclass
class ClusterOptResult:
    """Result of optimizing a single cluster."""
    cluster_id: int
    tier: int
    initial_accuracy: float
    final_accuracy: float
    initial_connectivity: Tensor
    final_connectivity: Tensor
    improvement_pct: float
    generations_run: int


@dataclass
class PerClusterResult:
    """Result of optimizing all clusters."""
    cluster_results: list[ClusterOptResult]
    initial_ppl: float
    final_ppl: float
    improvement_pct: float
    total_clusters_optimized: int
    tier_summaries: dict[int, dict]  # tier -> {clusters, avg_improvement, ...}


class IncrementalEvaluator:
    """
    Efficient evaluator with batch acceleration for per-cluster optimization.

    Instead of re-training the entire model for each connectivity variant,
    we cache all clusters' votes and only update the changed cluster.

    ## Batch Acceleration

    Key optimization: evaluate MANY connectivity variants in parallel:
    - GA population: 30 variants evaluated together
    - Metal GPU: Parallel training and vote computation
    - Vectorized fitness: NumPy/Torch batch operations

    ## Rust/Metal Extension Points

    When Rust accelerator is extended, these methods will use native code:
    - `_train_cluster_batch_rust()` - Train N variants in parallel (rayon)
    - `_compute_votes_batch_metal()` - GPU-accelerated vote computation
    - `_discriminative_fitness_batch()` - Vectorized fitness calculation
    """

    def __init__(
        self,
        train_contexts: Tensor,      # [N, context_bits] - binary context vectors
        train_targets: Tensor,       # [N] - target cluster IDs
        eval_contexts: Tensor,       # [M, context_bits] - binary eval contexts
        eval_targets: Tensor,        # [M] - eval target cluster IDs
        context_bits: int,           # Total input bits
        cluster_to_neurons: dict[int, tuple[int, int]],  # cluster_id -> (start_neuron, end_neuron)
        cluster_to_bits: dict[int, int],  # cluster_id -> bits_per_neuron
        num_clusters: int,           # Total number of clusters
        logger: Optional[Callable[[str], None]] = None,
    ):
        self.train_contexts = train_contexts
        self.train_targets = train_targets
        self.eval_contexts = eval_contexts
        self.eval_targets = eval_targets
        self.context_bits = context_bits
        self.cluster_to_neurons = cluster_to_neurons
        self.cluster_to_bits = cluster_to_bits
        self.num_clusters = num_clusters
        self._log = logger or (lambda x: None)

        # Precompute indices per cluster for O(1) lookup
        self._train_indices: dict[int, Tensor] = {}
        self._eval_indices: dict[int, Tensor] = {}
        self._precompute_indices()

        # Baseline state (populated by precompute_baseline)
        self._baseline_votes: Optional[Tensor] = None  # [M, num_clusters]
        self._baseline_rams: dict[int, list[dict[int, int]]] = {}  # cluster -> [neuron_rams]

        # Acceleration
        self._rust_available = self._check_rust()
        self._metal_available = self._check_metal()
        if self._rust_available:
            self._log(f"  Rust accelerator: available")
        if self._metal_available:
            self._log(f"  Metal GPU: available")

    def _check_rust(self) -> bool:
        try:
            import ram_accelerator
            return True
        except ImportError:
            return False

    def _check_metal(self) -> bool:
        try:
            import ram_accelerator
            return ram_accelerator.metal_available()
        except:
            return False

    def _precompute_indices(self) -> None:
        """Precompute example indices per cluster for fast lookup."""
        for cluster_id in range(self.num_clusters):
            train_mask = self.train_targets == cluster_id
            eval_mask = self.eval_targets == cluster_id
            self._train_indices[cluster_id] = torch.where(train_mask)[0]
            self._eval_indices[cluster_id] = torch.where(eval_mask)[0]

    # =========================================================================
    # Baseline Computation
    # =========================================================================

    def precompute_baseline(self, full_connectivity: Tensor) -> None:
        """
        Precompute votes for clusters in cluster_to_neurons mapping.
        Only computes for clusters that will be optimized (not all vocab).
        """
        self._log("Precomputing baseline votes...")
        num_eval = len(self.eval_contexts)
        # Only allocate for clusters being optimized
        self._baseline_votes = torch.zeros(num_eval, self.num_clusters)
        self._baseline_rams = {}

        # Only iterate over clusters in the mapping (the ones being optimized)
        clusters_to_process = sorted(self.cluster_to_neurons.keys())
        total = len(clusters_to_process)

        for i, cluster_id in enumerate(clusters_to_process):
            cluster_conn = self._extract_cluster_connectivity(full_connectivity, cluster_id)
            bits = self.cluster_to_bits.get(cluster_id, 8)

            # Train and get votes
            rams, votes = self._train_and_vote_single(cluster_id, cluster_conn, bits)
            self._baseline_rams[cluster_id] = rams
            self._baseline_votes[:, cluster_id] = votes

            if (i + 1) % 50 == 0 or (i + 1) == total:
                self._log(f"    Baseline: {i + 1}/{total} clusters")

        self._log(f"  Baseline complete for {total} clusters")

    def _extract_cluster_connectivity(self, full_conn: Tensor, cluster_id: int) -> Tensor:
        """Extract connectivity for a single cluster."""
        start, end = self.cluster_to_neurons.get(cluster_id, (cluster_id, cluster_id + 1))
        bits = self.cluster_to_bits.get(cluster_id, 8)
        num_neurons = end - start

        if full_conn.dim() == 1:
            # Flattened - compute offset (simplified: assume uniform within tier)
            offset = start * bits
            return full_conn[offset:offset + num_neurons * bits].view(num_neurons, bits)
        else:
            return full_conn[start:end]

    def _train_and_vote_single(
        self,
        cluster_id: int,
        connectivity: Tensor,  # [num_neurons, bits_per_neuron]
        bits_per_neuron: int,
    ) -> tuple[list[dict[int, int]], Tensor]:
        """Train one cluster's RAMs and compute votes on eval set."""
        num_neurons = connectivity.shape[0]
        train_idx = self._train_indices.get(cluster_id, torch.tensor([]))

        # Initialize RAMs (one dict per neuron: address -> count)
        rams: list[dict[int, int]] = [{} for _ in range(num_neurons)]

        # Train on positive examples
        if len(train_idx) > 0:
            train_ctx = self.train_contexts[train_idx]
            for neuron_idx in range(num_neurons):
                conn = connectivity[neuron_idx].long()
                for ctx in train_ctx:
                    addr = self._compute_address(ctx, conn)
                    rams[neuron_idx][addr] = rams[neuron_idx].get(addr, 0) + 1

        # Compute votes on eval set
        votes = self._compute_votes(connectivity, rams, self.eval_contexts)
        return rams, votes

    def _compute_address(self, context: Tensor, connectivity: Tensor) -> int:
        """Compute RAM address from context and connectivity."""
        addr = 0
        for bit_idx, conn_bit in enumerate(connectivity):
            if conn_bit < len(context) and context[conn_bit]:
                addr |= (1 << bit_idx)
        return addr

    def _compute_votes(
        self,
        connectivity: Tensor,
        rams: list[dict[int, int]],
        contexts: Tensor,
    ) -> Tensor:
        """Compute vote strengths for all contexts."""
        num_contexts = len(contexts)
        num_neurons = connectivity.shape[0]
        votes = torch.zeros(num_contexts)

        for ex_idx, ctx in enumerate(contexts):
            vote_sum = 0.0
            for neuron_idx in range(num_neurons):
                conn = connectivity[neuron_idx].long()
                addr = self._compute_address(ctx, conn)
                vote_sum += rams[neuron_idx].get(addr, 0)
            votes[ex_idx] = vote_sum / max(num_neurons, 1)

        return votes

    # =========================================================================
    # Single Variant Evaluation
    # =========================================================================

    def evaluate_cluster_variant(
        self,
        cluster_id: int,
        new_connectivity: Tensor,
        fitness_mode: FitnessMode = FitnessMode.PENALIZE_HIGH_VOTES,
    ) -> float:
        """Evaluate a single connectivity variant for a cluster."""
        if self._baseline_votes is None:
            raise RuntimeError("Call precompute_baseline() first")

        bits = self.cluster_to_bits.get(cluster_id, 8)
        _, new_votes = self._train_and_vote_single(cluster_id, new_connectivity, bits)

        return self._compute_fitness(cluster_id, new_votes, fitness_mode)

    # =========================================================================
    # BATCH Evaluation (Key Acceleration)
    # =========================================================================

    def evaluate_cluster_variants_batch(
        self,
        cluster_id: int,
        connectivity_variants: list[Tensor],
        fitness_mode: FitnessMode = FitnessMode.PENALIZE_HIGH_VOTES,
    ) -> list[float]:
        """
        Evaluate MULTIPLE connectivity variants in batch.

        This is the key acceleration point:
        - Vectorized training across variants
        - Parallel vote computation
        - Batch fitness calculation

        Args:
            cluster_id: Cluster to evaluate
            connectivity_variants: List of [num_neurons, bits] tensors
            fitness_mode: Fitness calculation mode

        Returns:
            List of fitness scores (one per variant)
        """
        if self._baseline_votes is None:
            raise RuntimeError("Call precompute_baseline() first")

        num_variants = len(connectivity_variants)
        if num_variants == 0:
            return []

        bits = self.cluster_to_bits.get(cluster_id, 8)
        train_idx = self._train_indices.get(cluster_id, torch.tensor([]))
        train_ctx = self.train_contexts[train_idx] if len(train_idx) > 0 else None

        # Batch train and vote
        all_votes = self._train_and_vote_batch(
            cluster_id, connectivity_variants, bits, train_ctx
        )

        # Batch fitness calculation
        return self._compute_fitness_batch(cluster_id, all_votes, fitness_mode)

    def _train_and_vote_batch(
        self,
        cluster_id: int,
        variants: list[Tensor],
        bits_per_neuron: int,
        train_contexts: Optional[Tensor],
    ) -> Tensor:
        """
        Train and vote for MULTIPLE connectivity variants.

        Returns:
            Tensor of shape [num_variants, num_eval] with vote strengths
        """
        num_variants = len(variants)
        num_eval = len(self.eval_contexts)
        all_votes = torch.zeros(num_variants, num_eval)

        # Process each variant (can be parallelized with joblib/multiprocessing)
        for v_idx, conn in enumerate(variants):
            num_neurons = conn.shape[0]
            rams: list[dict[int, int]] = [{} for _ in range(num_neurons)]

            # Train
            if train_contexts is not None:
                for neuron_idx in range(num_neurons):
                    neuron_conn = conn[neuron_idx].long()
                    for ctx in train_contexts:
                        addr = self._compute_address(ctx, neuron_conn)
                        rams[neuron_idx][addr] = rams[neuron_idx].get(addr, 0) + 1

            # Vote
            all_votes[v_idx] = self._compute_votes(conn, rams, self.eval_contexts)

        return all_votes

    def _compute_fitness_batch(
        self,
        cluster_id: int,
        all_votes: Tensor,  # [num_variants, num_eval]
        fitness_mode: FitnessMode,
    ) -> list[float]:
        """
        Compute fitness for multiple variants in batch.

        Uses vectorized operations for efficiency.
        """
        num_variants = all_votes.shape[0]
        pos_idx = self._eval_indices.get(cluster_id, torch.tensor([]))
        neg_mask = self.eval_targets != cluster_id
        neg_idx = torch.where(neg_mask)[0]

        fitness_scores = []

        for v_idx in range(num_variants):
            votes = all_votes[v_idx]
            score = self._compute_fitness(cluster_id, votes, fitness_mode, pos_idx, neg_idx)
            fitness_scores.append(score)

        return fitness_scores

    def _compute_fitness(
        self,
        cluster_id: int,
        votes: Tensor,
        fitness_mode: FitnessMode,
        pos_idx: Optional[Tensor] = None,
        neg_idx: Optional[Tensor] = None,
    ) -> float:
        """Compute discriminative fitness for a single vote vector."""
        if pos_idx is None:
            pos_idx = self._eval_indices.get(cluster_id, torch.tensor([]))
        if neg_idx is None:
            neg_mask = self.eval_targets != cluster_id
            neg_idx = torch.where(neg_mask)[0]

        if fitness_mode == FitnessMode.POSITIVE_ONLY:
            if len(pos_idx) == 0:
                return 0.0
            return votes[pos_idx].mean().item()

        elif fitness_mode == FitnessMode.PENALIZE_WINS:
            # Positive: how often does this cluster win when it should?
            if len(pos_idx) == 0:
                pos_score = 0.0
            else:
                wins = 0
                for idx in pos_idx:
                    all_v = self._baseline_votes[idx].clone()
                    all_v[cluster_id] = votes[idx]
                    if all_v.argmax() == cluster_id:
                        wins += 1
                pos_score = wins / len(pos_idx)

            # Negative: how often does this cluster wrongly win?
            if len(neg_idx) == 0:
                neg_score = 0.0
            else:
                sample = neg_idx[:1000]  # Sample for efficiency
                wrong_wins = 0
                for idx in sample:
                    all_v = self._baseline_votes[idx].clone()
                    all_v[cluster_id] = votes[idx]
                    if all_v.argmax() == cluster_id:
                        wrong_wins += 1
                neg_score = wrong_wins / len(sample)

            return pos_score - neg_score

        else:  # PENALIZE_HIGH_VOTES (default)
            # Positive: average vote when should fire
            pos_strength = votes[pos_idx].mean().item() if len(pos_idx) > 0 else 0.0

            # Negative: average vote when should NOT fire (sample)
            if len(neg_idx) == 0:
                neg_strength = 0.0
            else:
                sample_size = min(len(neg_idx), 2000)
                sample = neg_idx[torch.randperm(len(neg_idx))[:sample_size]]
                neg_strength = votes[sample].mean().item()

            return pos_strength - neg_strength

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_cluster_examples(
        self,
        cluster_id: int,
        split: str = "train",
    ) -> tuple[Tensor, Tensor]:
        """Get examples where target = cluster_id."""
        if split == "train":
            idx = self._train_indices.get(cluster_id, torch.tensor([]))
            if len(idx) == 0:
                return torch.tensor([]), torch.tensor([])
            return self.train_contexts[idx], self.train_targets[idx]
        else:
            idx = self._eval_indices.get(cluster_id, torch.tensor([]))
            if len(idx) == 0:
                return torch.tensor([]), torch.tensor([])
            return self.eval_contexts[idx], self.eval_targets[idx]

    def get_example_count(self, cluster_id: int, split: str = "train") -> int:
        """Get number of examples for a cluster."""
        idx = self._train_indices if split == "train" else self._eval_indices
        return len(idx.get(cluster_id, []))


class ClusterOptimizer:
    """
    Optimizes a single cluster's connectivity using GA or TS.

    This is the core reusable component - it handles:
    1. Extracting examples for this cluster
    2. Computing discriminative fitness
    3. Running GA/TS optimization with BATCH evaluation
    4. Returning optimized connectivity

    Key difference from global optimization:
    - Only operates on this cluster's neurons
    - Uses discriminative fitness (fire for correct token, don't fire for others)
    - Batch evaluation of population for efficiency
    """

    def __init__(
        self,
        evaluator: IncrementalEvaluator,
        fitness_mode: FitnessMode = FitnessMode.PENALIZE_HIGH_VOTES,
        seed: Optional[int] = None,
        logger: Optional[Callable[[str], None]] = None,
    ):
        self.evaluator = evaluator
        self.fitness_mode = fitness_mode
        self.seed = seed
        self._log = logger or (lambda x: None)

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

    def optimize_cluster_ga(
        self,
        cluster_id: int,
        initial_connectivity: Tensor,
        config: TierOptConfig,
    ) -> ClusterOptResult:
        """
        Optimize a cluster's connectivity using Genetic Algorithm.

        The GA operates ONLY on this cluster's neurons:
        - Population: N variants of this cluster's connectivity
        - Crossover: Swap neurons within this cluster only
        - Mutation: Mutate connections within this cluster only
        - Fitness: Discriminative fitness for this cluster's token
        - BATCH evaluation: All population members evaluated together
        """
        num_neurons, bits_per_neuron = initial_connectivity.shape
        total_bits = self.evaluator.context_bits
        pop_size = config.ga_population
        elitism = max(1, pop_size // 10)

        # Initialize population
        population = [initial_connectivity.clone()]
        for _ in range(pop_size - 1):
            variant = self._random_connectivity(num_neurons, bits_per_neuron, total_bits)
            population.append(variant)

        # Evaluate initial population (BATCH)
        fitness = self.evaluator.evaluate_cluster_variants_batch(
            cluster_id, population, self.fitness_mode
        )

        # Track best
        best_idx = max(range(len(fitness)), key=lambda i: fitness[i])
        best_conn = population[best_idx].clone()
        best_fitness = fitness[best_idx]
        initial_fitness = self.evaluator.evaluate_cluster_variant(
            cluster_id, initial_connectivity, self.fitness_mode
        )

        # Evolution loop
        for gen in range(config.ga_gens):
            # Sort by fitness (descending - higher is better)
            sorted_indices = sorted(range(len(fitness)), key=lambda i: fitness[i], reverse=True)

            # Elitism: keep top individuals
            new_population = [population[sorted_indices[i]].clone() for i in range(elitism)]

            # Generate offspring
            while len(new_population) < pop_size:
                # Tournament selection
                p1 = self._tournament_select(population, fitness)
                p2 = self._tournament_select(population, fitness)

                # Crossover
                if random.random() < 0.7:
                    child = self._crossover(p1, p2)
                else:
                    child = p1.clone()

                # Mutation
                child = self._mutate(child, config.mutation_rate, total_bits)
                new_population.append(child)

            population = new_population[:pop_size]

            # Batch evaluate new population
            fitness = self.evaluator.evaluate_cluster_variants_batch(
                cluster_id, population, self.fitness_mode
            )

            # Update best
            gen_best_idx = max(range(len(fitness)), key=lambda i: fitness[i])
            if fitness[gen_best_idx] > best_fitness:
                best_conn = population[gen_best_idx].clone()
                best_fitness = fitness[gen_best_idx]

        # Get tier from evaluator
        start, end = self.evaluator.cluster_to_neurons.get(cluster_id, (0, 1))
        tier = 0  # Default, should be computed from cluster_id

        improvement = ((best_fitness - initial_fitness) / abs(initial_fitness) * 100) if initial_fitness != 0 else 0

        return ClusterOptResult(
            cluster_id=cluster_id,
            tier=tier,
            initial_accuracy=initial_fitness,
            final_accuracy=best_fitness,
            initial_connectivity=initial_connectivity,
            final_connectivity=best_conn,
            improvement_pct=improvement,
            generations_run=config.ga_gens,
        )

    def optimize_cluster_ts(
        self,
        cluster_id: int,
        initial_connectivity: Tensor,
        config: TierOptConfig,
    ) -> ClusterOptResult:
        """
        Optimize a cluster's connectivity using Tabu Search.

        TS operates ONLY on this cluster's neurons:
        - Generate N neighbors by mutating only this cluster's connections
        - Tabu list tracks recent moves to avoid cycling
        - Always move to best non-tabu neighbor
        - BATCH evaluation: All neighbors evaluated together
        """
        num_neurons, bits_per_neuron = initial_connectivity.shape
        total_bits = self.evaluator.context_bits

        # Current solution
        current = initial_connectivity.clone()
        current_fitness = self.evaluator.evaluate_cluster_variant(
            cluster_id, current, self.fitness_mode
        )
        initial_fitness = current_fitness

        # Best solution
        best = current.clone()
        best_fitness = current_fitness

        # Tabu list: stores (neuron_idx, old_value, new_value) tuples
        from collections import deque
        tabu_list: deque = deque(maxlen=5)

        # TS iterations
        for iteration in range(config.ts_iters):
            # Generate neighbors
            neighbors = []
            moves = []
            for _ in range(config.ts_neighbors):
                neighbor, move = self._generate_neighbor(
                    current, config.mutation_rate, total_bits
                )
                # Skip if move is tabu
                if not self._is_tabu(move, tabu_list):
                    neighbors.append(neighbor)
                    moves.append(move)

            if len(neighbors) == 0:
                # All moves are tabu, reduce tabu size or break
                continue

            # Batch evaluate neighbors
            neighbor_fitness = self.evaluator.evaluate_cluster_variants_batch(
                cluster_id, neighbors, self.fitness_mode
            )

            # Pick best neighbor (even if worse - TS characteristic)
            best_neighbor_idx = max(range(len(neighbor_fitness)), key=lambda i: neighbor_fitness[i])
            best_neighbor = neighbors[best_neighbor_idx]
            best_neighbor_fit = neighbor_fitness[best_neighbor_idx]
            best_move = moves[best_neighbor_idx]

            # Move to best neighbor
            current = best_neighbor
            current_fitness = best_neighbor_fit
            tabu_list.append(best_move)

            # Update global best
            if current_fitness > best_fitness:
                best = current.clone()
                best_fitness = current_fitness

        # Get tier
        tier = 0  # Default

        improvement = ((best_fitness - initial_fitness) / abs(initial_fitness) * 100) if initial_fitness != 0 else 0

        return ClusterOptResult(
            cluster_id=cluster_id,
            tier=tier,
            initial_accuracy=initial_fitness,
            final_accuracy=best_fitness,
            initial_connectivity=initial_connectivity,
            final_connectivity=best,
            improvement_pct=improvement,
            generations_run=config.ts_iters,
        )

    # =========================================================================
    # GA/TS Helper Methods
    # =========================================================================

    def _random_connectivity(
        self,
        num_neurons: int,
        bits_per_neuron: int,
        total_bits: int,
    ) -> Tensor:
        """Generate random connectivity for a cluster."""
        return torch.randint(0, total_bits, (num_neurons, bits_per_neuron))

    def _tournament_select(
        self,
        population: list[Tensor],
        fitness: list[float],
        tournament_size: int = 3,
    ) -> Tensor:
        """Tournament selection: pick best from random subset."""
        indices = random.sample(range(len(population)), min(tournament_size, len(population)))
        best_idx = max(indices, key=lambda i: fitness[i])
        return population[best_idx]

    def _crossover(self, parent1: Tensor, parent2: Tensor) -> Tensor:
        """Single-point crossover at neuron level."""
        num_neurons = parent1.shape[0]
        if num_neurons <= 1:
            return parent1.clone()

        crossover_point = random.randint(1, num_neurons - 1)
        child = parent1.clone()
        child[crossover_point:] = parent2[crossover_point:].clone()
        return child

    def _mutate(
        self,
        connectivity: Tensor,
        mutation_rate: float,
        total_bits: int,
    ) -> Tensor:
        """Mutate connectivity with given probability per connection."""
        mutated = connectivity.clone()
        num_neurons, bits_per_neuron = mutated.shape

        for n in range(num_neurons):
            for b in range(bits_per_neuron):
                if random.random() < mutation_rate:
                    mutated[n, b] = random.randint(0, total_bits - 1)

        return mutated

    def _generate_neighbor(
        self,
        connectivity: Tensor,
        mutation_rate: float,
        total_bits: int,
    ) -> tuple[Tensor, tuple[int, int, int]]:
        """Generate a neighbor by mutation. Returns (neighbor, move)."""
        neighbor = connectivity.clone()
        num_neurons, bits_per_neuron = neighbor.shape

        # Make at least one mutation
        mutations_made = 0
        last_move = (0, 0, 0)

        for n in range(num_neurons):
            for b in range(bits_per_neuron):
                if random.random() < mutation_rate or mutations_made == 0:
                    old_val = int(neighbor[n, b].item())
                    new_val = random.randint(0, total_bits - 1)
                    neighbor[n, b] = new_val
                    last_move = (n * bits_per_neuron + b, old_val, new_val)
                    mutations_made += 1

        return neighbor, last_move

    def _is_tabu(
        self,
        move: tuple[int, int, int],
        tabu_list: deque,
    ) -> bool:
        """Check if move reverses a recent tabu move."""
        # move = (flat_idx, old_val, new_val)
        for tabu_move in tabu_list:
            # Tabu if same index and reverses the change
            if tabu_move[0] == move[0] and tabu_move[2] == move[1]:
                return True
        return False

    def compute_discriminative_fitness(
        self,
        cluster_id: int,
        connectivity: Tensor,
    ) -> float:
        """
        Compute discriminative fitness for a cluster.

        This is the KEY function that determines optimization quality.

        Args:
            cluster_id: The cluster being optimized
            connectivity: Proposed connectivity for this cluster's neurons

        Returns:
            Fitness score (higher = better)

        TODO: User to implement the fitness logic based on chosen mode
        """
        return self.evaluator.evaluate_cluster_variant(
            cluster_id,
            connectivity,
            self.fitness_mode,
        )


class PerClusterOptimizer:
    """
    Main orchestrator for per-cluster optimization across all tiers.

    This handles:
    1. Iterating through tiers in priority order
    2. Applying tier-specific optimization budgets
    3. Tracking per-cluster and aggregate results
    4. Optional parallelization within tiers
    """

    def __init__(
        self,
        tier_configs: list[TierOptConfig],
        evaluator: IncrementalEvaluator,
        fitness_mode: FitnessMode = FitnessMode.PENALIZE_HIGH_VOTES,
        cluster_order: str = "random",  # "random", "frequency", "sequential"
        seed: Optional[int] = None,
        logger: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ):
        """
        Args:
            tier_configs: List of TierOptConfig, one per tier
            evaluator: IncrementalEvaluator instance
            fitness_mode: How to compute discriminative fitness
            cluster_order: How to order clusters within a tier
            seed: Random seed for reproducibility
            logger: Logging function
            progress_callback: Called with (clusters_done, total_clusters, current_ppl)
        """
        self.tier_configs = {tc.tier: tc for tc in tier_configs}
        self.evaluator = evaluator
        self.fitness_mode = fitness_mode
        self.cluster_order = cluster_order
        self.seed = seed
        self._log = logger or print
        self._progress_callback = progress_callback

        self._cluster_optimizer = ClusterOptimizer(
            evaluator=evaluator,
            fitness_mode=fitness_mode,
            seed=seed,
            logger=logger,
        )

    def _optimize_single_cluster(
        self,
        cluster_id: int,
        cluster_conn: Tensor,
        config: TierOptConfig,
    ) -> ClusterOptResult:
        """Optimize a single cluster (can be called in parallel)."""
        # Run GA then TS
        if config.ga_gens > 0:
            result = self._cluster_optimizer.optimize_cluster_ga(
                cluster_id, cluster_conn, config
            )
            cluster_conn = result.final_connectivity

        if config.ts_iters > 0:
            result = self._cluster_optimizer.optimize_cluster_ts(
                cluster_id, cluster_conn, config
            )

        return result

    def optimize_tier(
        self,
        tier: int,
        cluster_ids: list[int],
        current_connectivity: Tensor,
        parallel: bool = True,
        max_workers: Optional[int] = None,
    ) -> tuple[Tensor, list[ClusterOptResult]]:
        """
        Optimize all clusters in a tier.

        Args:
            tier: Tier index (0, 1, 2, ...)
            cluster_ids: List of cluster IDs in this tier
            current_connectivity: Current full connectivity tensor
            parallel: Use parallel processing for cluster optimization
            max_workers: Max parallel workers (default: CPU count)

        Returns:
            (updated_connectivity, list of ClusterOptResult)
        """
        config = self.tier_configs.get(tier)
        if config is None or not config.enabled:
            self._log(f"  Tier {tier}: Skipped (disabled)")
            return current_connectivity, []

        num_clusters = len(cluster_ids)
        n_workers = max_workers or min(mp.cpu_count(), num_clusters)

        if parallel and num_clusters > 1:
            self._log(f"  Tier {tier}: Optimizing {num_clusters} clusters in parallel ({n_workers} workers) ({config})")
        else:
            self._log(f"  Tier {tier}: Optimizing {num_clusters} clusters ({config})")

        # Order clusters
        ordered_clusters = self._order_clusters(cluster_ids)

        # Extract all cluster connectivities upfront
        cluster_conns = {}
        for cluster_id in ordered_clusters:
            cluster_conns[cluster_id] = self._extract_cluster_connectivity(
                current_connectivity, cluster_id
            )

        results = []
        updated_connectivity = current_connectivity.clone()

        if parallel and num_clusters > 1:
            # Parallel optimization - clusters are independent!
            # We use ThreadPoolExecutor because the GIL is released during
            # torch tensor operations, giving good parallelism
            completed = 0
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(
                        self._optimize_single_cluster,
                        cluster_id,
                        cluster_conns[cluster_id],
                        config
                    ): cluster_id
                    for cluster_id in ordered_clusters
                }

                for future in as_completed(futures):
                    cluster_id = futures[future]
                    try:
                        result = future.result()
                        results.append(result)

                        # Update connectivity
                        updated_connectivity = self._update_cluster_connectivity(
                            updated_connectivity, cluster_id, result.final_connectivity
                        )

                        completed += 1
                        if completed % 10 == 0 or completed == num_clusters:
                            self._log(f"    Progress: {completed}/{num_clusters} clusters optimized")

                    except Exception as e:
                        self._log(f"    ERROR optimizing cluster {cluster_id}: {e}")

        else:
            # Sequential optimization
            for i, cluster_id in enumerate(ordered_clusters):
                cluster_conn = cluster_conns[cluster_id]
                result = self._optimize_single_cluster(cluster_id, cluster_conn, config)

                # Update full connectivity with optimized cluster
                updated_connectivity = self._update_cluster_connectivity(
                    updated_connectivity, cluster_id, result.final_connectivity
                )

                results.append(result)

                # Progress callback
                if self._progress_callback and (i + 1) % 10 == 0:
                    self._progress_callback(i + 1, num_clusters, 0.0)

        return updated_connectivity, results

    def optimize_all_tiers(
        self,
        initial_connectivity: Tensor,
        tier_to_clusters: dict[int, list[int]],
    ) -> PerClusterResult:
        """
        Optimize all tiers in order (tier0 first, then tier1, etc.).

        Args:
            initial_connectivity: Starting connectivity tensor
            tier_to_clusters: Mapping from tier index to list of cluster IDs

        Returns:
            PerClusterResult with all optimization results
        """
        self._log("=" * 70)
        self._log("PER-CLUSTER OPTIMIZATION")
        self._log("=" * 70)

        # Precompute baseline
        self._log("Precomputing baseline votes...")
        self.evaluator.precompute_baseline(initial_connectivity)

        # TODO: Compute initial PPL
        initial_ppl = 0.0

        current_connectivity = initial_connectivity.clone()
        all_results = []
        tier_summaries = {}

        # Process tiers in order
        for tier in sorted(tier_to_clusters.keys()):
            cluster_ids = tier_to_clusters[tier]

            updated_conn, tier_results = self.optimize_tier(
                tier, cluster_ids, current_connectivity
            )

            current_connectivity = updated_conn
            all_results.extend(tier_results)

            # Summarize tier results
            if tier_results:
                tier_summaries[tier] = {
                    "clusters": len(tier_results),
                    "avg_improvement": sum(r.improvement_pct for r in tier_results) / len(tier_results),
                    "total_improvement": sum(r.improvement_pct for r in tier_results),
                }

        # TODO: Compute final PPL
        final_ppl = 0.0
        improvement = ((initial_ppl - final_ppl) / initial_ppl * 100) if initial_ppl > 0 else 0.0

        return PerClusterResult(
            cluster_results=all_results,
            initial_ppl=initial_ppl,
            final_ppl=final_ppl,
            improvement_pct=improvement,
            total_clusters_optimized=len(all_results),
            tier_summaries=tier_summaries,
        )

    def _order_clusters(self, cluster_ids: list[int]) -> list[int]:
        """Order clusters based on configured strategy."""
        if self.cluster_order == "sequential":
            return sorted(cluster_ids)
        elif self.cluster_order == "random":
            shuffled = list(cluster_ids)
            random.shuffle(shuffled)
            return shuffled
        else:  # "frequency" - already ordered by frequency in tier
            return cluster_ids

    def _extract_cluster_connectivity(
        self,
        full_connectivity: Tensor,
        cluster_id: int,
    ) -> Tensor:
        """Extract just this cluster's neurons' connectivity."""
        start, end = self.evaluator.cluster_to_neurons[cluster_id]
        bits = self.evaluator.cluster_to_bits[cluster_id]

        if full_connectivity.dim() == 1:
            # Flattened - need to compute offset
            # TODO: Use neuron_offsets here
            raise NotImplementedError("Flattened tensor extraction")
        else:
            return full_connectivity[start:end].clone()

    def _update_cluster_connectivity(
        self,
        full_connectivity: Tensor,
        cluster_id: int,
        new_cluster_conn: Tensor,
    ) -> Tensor:
        """Update full connectivity with new cluster connectivity."""
        start, end = self.evaluator.cluster_to_neurons[cluster_id]

        updated = full_connectivity.clone()
        if updated.dim() == 1:
            # TODO: Handle flattened tensor
            raise NotImplementedError("Flattened tensor update")
        else:
            updated[start:end] = new_cluster_conn

        return updated


# =============================================================================
# Rust Accelerator Integration
# =============================================================================

def _check_rust_per_cluster() -> bool:
    """Check if Rust per-cluster acceleration is available."""
    try:
        import ram_accelerator
        return hasattr(ram_accelerator, 'per_cluster_create_evaluator')
    except ImportError:
        return False


class RustPerClusterOptimizer:
    """
    Rust-accelerated per-cluster optimizer.

    This class wraps the Rust accelerator functions for maximum performance.
    Falls back to Python implementation if Rust is not available.

    Usage:
        optimizer = RustPerClusterOptimizer(
            train_contexts=train_ctx,  # [N, context_bits] bool tensor
            train_targets=train_tgt,   # [N] int tensor
            eval_contexts=eval_ctx,    # [M, context_bits] bool tensor
            eval_targets=eval_tgt,     # [M] int tensor
            context_bits=48,
            cluster_to_neurons={0: (0, 15), 1: (15, 30), ...},
            cluster_to_bits={0: 20, 1: 20, ...},
            num_clusters=50257,
        )
        results = optimizer.optimize_tier(
            tier=0,
            cluster_ids=[0, 1, 2, ...],
            current_connectivity=conn_tensor,
            config=TierOptConfig(tier=0, ga_gens=50, ts_iters=20),
        )
    """

    def __init__(
        self,
        train_contexts: Tensor,      # [N, context_bits] bool
        train_targets: Tensor,       # [N] int
        eval_contexts: Tensor,       # [M, context_bits] bool
        eval_targets: Tensor,        # [M] int
        context_bits: int,
        cluster_to_neurons: dict[int, tuple[int, int]],
        cluster_to_bits: dict[int, int],
        num_clusters: int,
        fitness_mode: FitnessMode = FitnessMode.SIMPLE,
        logger: Optional[Callable[[str], None]] = None,
    ):
        self.context_bits = context_bits
        self.cluster_to_neurons = cluster_to_neurons
        self.cluster_to_bits = cluster_to_bits
        self.num_clusters = num_clusters
        self.fitness_mode = fitness_mode
        self._log = logger or print

        self._rust_available = _check_rust_per_cluster()
        self._evaluator_id: Optional[int] = None

        if self._rust_available:
            self._log("  Rust per-cluster accelerator: creating evaluator...")
            import ram_accelerator

            # Flatten contexts to 1D bool list
            train_flat = train_contexts.flatten().tolist()
            eval_flat = eval_contexts.flatten().tolist()

            # Convert targets to list of usize
            train_tgt = train_targets.tolist()
            eval_tgt = eval_targets.tolist()

            # Convert mappings to lists of tuples
            cluster_neurons_list = [
                (cid, start, end)
                for cid, (start, end) in cluster_to_neurons.items()
            ]
            cluster_bits_list = [
                (cid, bits)
                for cid, bits in cluster_to_bits.items()
            ]

            self._evaluator_id = ram_accelerator.per_cluster_create_evaluator(
                train_contexts_flat=train_flat,
                train_targets=train_tgt,
                eval_contexts_flat=eval_flat,
                eval_targets=eval_tgt,
                context_bits=context_bits,
                cluster_neurons=cluster_neurons_list,
                cluster_bits=cluster_bits_list,
                num_clusters=num_clusters,
            )
            self._log(f"  Rust evaluator created (ID={self._evaluator_id})")
            self._global_baseline_computed = False
        else:
            self._log("  Rust per-cluster accelerator: NOT available (using Python fallback)")
            self._global_baseline_computed = False

    def precompute_global_baseline(
        self,
        all_connectivities: dict[int, list[int]],
    ) -> None:
        """
        Precompute baseline votes for ALL clusters (enables true global CE).

        This enables exact global CE computation during optimization.
        Memory: num_eval * num_clusters * 8 bytes (e.g., 4K * 50K * 8 = 1.6GB)
        Time: ~3s for 50K clusters (parallelized with rayon)

        Call this ONCE before optimization starts. After this, all CE fitness
        computations will use true global softmax over all 50K clusters.

        Args:
            all_connectivities: Dict mapping cluster_id -> connectivity for ALL clusters
        """
        if not self._rust_available or self._evaluator_id is None:
            raise RuntimeError("Rust accelerator not available")

        import ram_accelerator
        import time

        self._log(f"  Precomputing global baseline for {len(all_connectivities)} clusters...")
        t0 = time.time()

        ram_accelerator.per_cluster_precompute_global_baseline(
            evaluator_id=self._evaluator_id,
            all_connectivities=all_connectivities,
        )

        elapsed = time.time() - t0
        memory_gb = len(all_connectivities) * 4000 * 8 / (1024**3)  # Rough estimate
        self._log(f"  Global baseline computed in {elapsed:.1f}s (~{memory_gb:.1f}GB)")
        self._global_baseline_computed = True

    def update_global_baseline(
        self,
        cluster_id: int,
        new_connectivity: list[int],
    ) -> None:
        """
        Update global baseline for a specific cluster after optimization.

        Call this after optimizing a cluster to keep the cache accurate
        for subsequent cluster optimizations.
        """
        if not self._rust_available or self._evaluator_id is None:
            return
        if not self._global_baseline_computed:
            return

        import ram_accelerator
        ram_accelerator.per_cluster_update_global_baseline(
            evaluator_id=self._evaluator_id,
            cluster_id=cluster_id,
            new_connectivity=new_connectivity,
        )

    def optimize_tier(
        self,
        tier: int,
        cluster_ids: list[int],
        current_connectivity: Tensor,
        config: TierOptConfig,
        seed: Optional[int] = None,  # None = time-based
    ) -> list[ClusterOptResult]:
        """
        Optimize all clusters in a tier using Rust acceleration.

        Args:
            tier: Tier index
            cluster_ids: List of cluster IDs in this tier
            current_connectivity: Current connectivity tensor
            config: Optimization configuration
            seed: Random seed

        Returns:
            List of ClusterOptResult for each optimized cluster
        """
        if not self._rust_available or self._evaluator_id is None:
            raise RuntimeError("Rust accelerator not available")

        import ram_accelerator
        import time

        self._log(f"  Tier {tier}: Optimizing {len(cluster_ids)} clusters via Rust")
        self._log(f"    Config: GA {config.ga_gens} gens × {config.ga_population} pop, TS {config.ts_iters} iters × {config.ts_neighbors} neighbors")
        fitness_mode_name = self.fitness_mode.name
        baseline_info = " (with global baseline)" if self._global_baseline_computed else ""
        self._log(f"    Fitness: {fitness_mode_name}{baseline_info}")

        # Extract connectivity for each cluster
        t0 = time.time()
        initial_connectivities: dict[int, list[int]] = {}
        for cluster_id in cluster_ids:
            start, end = self.cluster_to_neurons[cluster_id]
            cluster_conn = current_connectivity[start:end]  # [num_neurons, bits]
            # Flatten to 1D list
            initial_connectivities[cluster_id] = cluster_conn.flatten().long().tolist()
        prep_time = time.time() - t0

        # Call Rust optimizer (includes baseline precomputation + GA/TS)
        self._log(f"    Starting optimization (prep took {prep_time:.1f}s)...")
        t0 = time.time()
        rust_results = ram_accelerator.per_cluster_optimize_tier(
            evaluator_id=self._evaluator_id,
            tier=tier,
            cluster_ids=cluster_ids,
            initial_connectivities=initial_connectivities,
            ga_gens=config.ga_gens,
            ga_population=config.ga_population,
            ts_iters=config.ts_iters,
            ts_neighbors=config.ts_neighbors,
            mutation_rate=config.mutation_rate,
            seed=seed,
            fitness_mode=int(self.fitness_mode),
        )
        opt_time = time.time() - t0

        # Convert Rust results to ClusterOptResult
        results = []
        total_init_fit = 0.0
        total_final_fit = 0.0
        improved_count = 0

        for cluster_id, final_conn, initial_fit, final_fit, improvement in rust_results:
            start, end = self.cluster_to_neurons[cluster_id]
            num_neurons = end - start
            bits = self.cluster_to_bits[cluster_id]

            # Reshape connectivity back to tensor
            initial_tensor = torch.tensor(
                initial_connectivities[cluster_id]
            ).view(num_neurons, bits)
            final_tensor = torch.tensor(final_conn).view(num_neurons, bits)

            results.append(ClusterOptResult(
                cluster_id=cluster_id,
                tier=tier,
                initial_accuracy=initial_fit,
                final_accuracy=final_fit,
                initial_connectivity=initial_tensor,
                final_connectivity=final_tensor,
                improvement_pct=improvement,
                generations_run=config.ga_gens + config.ts_iters,
            ))

            total_init_fit += initial_fit
            total_final_fit += final_fit
            if improvement > 0:
                improved_count += 1

        # Log summary stats
        n = len(results)
        avg_init = total_init_fit / n if n > 0 else 0
        avg_final = total_final_fit / n if n > 0 else 0
        avg_improvement = ((avg_final - avg_init) / abs(avg_init) * 100) if avg_init != 0 else 0

        self._log(f"    Done in {opt_time:.1f}s ({opt_time/n*1000:.1f}ms/cluster)")
        self._log(f"    CE fitness: {avg_init:.4f} → {avg_final:.4f} (avg {avg_improvement:+.1f}%)")
        self._log(f"    Improved: {improved_count}/{n} clusters ({improved_count/n*100:.0f}%)")
        return results

    def optimize_tier_grouped(
        self,
        tier: int,
        cluster_ids: list[int],
        current_connectivity: Tensor,
        config: TierOptConfig,
        group_size: int = 10,
        seed: Optional[int] = None,  # None = time-based
    ) -> list[ClusterOptResult]:
        """
        Optimize all clusters in a tier using joint group optimization.

        Instead of optimizing each cluster independently, clusters are grouped
        and optimized jointly. This captures inter-cluster competition:
        if cluster A gets stronger, cluster B must adapt.

        Args:
            tier: Tier index
            cluster_ids: List of cluster IDs in this tier
            current_connectivity: Current connectivity tensor
            config: Optimization configuration
            group_size: Number of clusters per group (e.g., 10 for top-k tokens)
            seed: Random seed

        Returns:
            List of ClusterOptResult for each optimized cluster
        """
        if not self._rust_available or self._evaluator_id is None:
            raise RuntimeError("Rust accelerator not available")

        if not self._global_baseline_computed:
            raise RuntimeError("Global baseline must be computed before grouped optimization")

        import ram_accelerator
        import time

        num_groups = (len(cluster_ids) + group_size - 1) // group_size
        self._log(f"  Tier {tier}: Group optimization ({len(cluster_ids)} clusters in {num_groups} groups of {group_size})")
        self._log(f"    Config: GA {config.ga_gens} gens × {config.ga_population} pop")
        self._log(f"    Fitness: {self.fitness_mode.name} (joint group)")

        # Extract connectivity for each cluster
        t0 = time.time()
        initial_connectivities: dict[int, list[int]] = {}
        for cluster_id in cluster_ids:
            start, end = self.cluster_to_neurons[cluster_id]
            cluster_conn = current_connectivity[start:end]  # [num_neurons, bits]
            initial_connectivities[cluster_id] = cluster_conn.flatten().long().tolist()
        prep_time = time.time() - t0

        # Call Rust grouped optimizer
        self._log(f"    Starting optimization (prep took {prep_time:.1f}s)...")
        t0 = time.time()
        rust_results = ram_accelerator.per_cluster_optimize_tier_grouped(
            evaluator_id=self._evaluator_id,
            tier=tier,
            cluster_ids=cluster_ids,
            initial_connectivities=initial_connectivities,
            group_size=group_size,
            ga_gens=config.ga_gens,
            ga_population=config.ga_population,
            ts_iters=config.ts_iters,
            ts_neighbors=config.ts_neighbors,
            mutation_rate=config.mutation_rate,
            seed=seed,
            fitness_mode=int(self.fitness_mode),
        )
        opt_time = time.time() - t0

        # Convert Rust results to ClusterOptResult
        results = []
        total_init_fit = 0.0
        total_final_fit = 0.0
        improved_count = 0

        for cluster_id, final_conn, initial_fit, final_fit, improvement in rust_results:
            start, end = self.cluster_to_neurons[cluster_id]
            num_neurons = end - start
            bits = self.cluster_to_bits[cluster_id]

            # Reshape connectivity back to tensor
            initial_tensor = torch.tensor(
                initial_connectivities[cluster_id]
            ).view(num_neurons, bits)
            final_tensor = torch.tensor(final_conn).view(num_neurons, bits)

            results.append(ClusterOptResult(
                cluster_id=cluster_id,
                tier=tier,
                initial_accuracy=initial_fit,
                final_accuracy=final_fit,
                initial_connectivity=initial_tensor,
                final_connectivity=final_tensor,
                improvement_pct=improvement,
                generations_run=config.ga_gens,
            ))

            total_init_fit += initial_fit
            total_final_fit += final_fit
            if improvement > 0:
                improved_count += 1

        # Log summary stats
        n = len(results)
        avg_init = total_init_fit / n if n > 0 else 0
        avg_final = total_final_fit / n if n > 0 else 0
        avg_improvement = ((avg_final - avg_init) / abs(avg_init) * 100) if avg_init != 0 else 0

        self._log(f"    Done in {opt_time:.1f}s ({opt_time/n*1000:.1f}ms/cluster)")
        self._log(f"    CE fitness: {avg_init:.4f} → {avg_final:.4f} (avg {avg_improvement:+.1f}%)")
        self._log(f"    Improved: {improved_count}/{n} clusters ({improved_count/n*100:.0f}%)")
        return results

    def update_connectivity(
        self,
        full_connectivity: Tensor,
        results: list[ClusterOptResult],
    ) -> Tensor:
        """Update full connectivity tensor with optimized cluster connectivities."""
        updated = full_connectivity.clone()

        for result in results:
            start, end = self.cluster_to_neurons[result.cluster_id]
            updated[start:end] = result.final_connectivity

        return updated

    @property
    def rust_available(self) -> bool:
        return self._rust_available


# =============================================================================
# Factory function for easy creation
# =============================================================================

def create_per_cluster_optimizer(
    model,  # TieredRAMLM or similar
    train_tokens: list[str],
    eval_tokens: list[str],
    tier_budgets: Optional[dict[int, tuple[int, int]]] = None,  # tier -> (ga_gens, ts_iters)
    fitness_mode: FitnessMode = FitnessMode.PENALIZE_HIGH_VOTES,
    seed: int = 42,
    logger: Optional[Callable[[str], None]] = None,
) -> PerClusterOptimizer:
    """
    Factory function to create a PerClusterOptimizer from a model.

    Args:
        model: The RAM language model
        train_tokens: Training token sequence
        eval_tokens: Evaluation token sequence
        tier_budgets: Optional dict mapping tier -> (ga_gens, ts_iters)
                     Defaults to Pareto-style: {0: (100, 50), 1: (30, 20), 2: (10, 5)}
        fitness_mode: How to compute discriminative fitness
        seed: Random seed
        logger: Logging function

    Returns:
        Configured PerClusterOptimizer
    """
    # Default tier budgets (Pareto principle)
    if tier_budgets is None:
        tier_budgets = {
            0: (100, 50),   # Most effort on frequent tokens
            1: (30, 20),    # Medium effort
            2: (10, 5),     # Minimal effort on rare tokens
        }

    # Create tier configs
    tier_configs = [
        TierOptConfig(tier=t, ga_gens=gg, ts_iters=ti)
        for t, (gg, ti) in tier_budgets.items()
    ]

    # TODO: Create IncrementalEvaluator from model
    # This requires extracting cluster->neuron mappings from the model

    raise NotImplementedError("Factory function - needs model integration")


def create_rust_optimizer(
    train_contexts: Tensor,
    train_targets: Tensor,
    eval_contexts: Tensor,
    eval_targets: Tensor,
    context_bits: int,
    cluster_to_neurons: dict[int, tuple[int, int]],
    cluster_to_bits: dict[int, int],
    num_clusters: int,
    fitness_mode: FitnessMode = FitnessMode.SIMPLE,
    logger: Optional[Callable[[str], None]] = None,
) -> Optional[RustPerClusterOptimizer]:
    """
    Create a Rust-accelerated per-cluster optimizer if available.

    Args:
        train_contexts: Training context bit vectors [N, context_bits]
        train_targets: Training target cluster IDs [N]
        eval_contexts: Eval context bit vectors [M, context_bits]
        eval_targets: Eval target cluster IDs [M]
        context_bits: Total context bits
        cluster_to_neurons: Mapping from cluster ID to (start, end) neuron indices
        cluster_to_bits: Mapping from cluster ID to bits_per_neuron
        num_clusters: Total number of clusters (vocab size)
        fitness_mode: Fitness calculation mode
        logger: Optional logging function

    Returns:
        RustPerClusterOptimizer if Rust is available, None otherwise
    """
    if not _check_rust_per_cluster():
        if logger:
            logger("Rust per-cluster accelerator not available")
        return None

    return RustPerClusterOptimizer(
        train_contexts=train_contexts,
        train_targets=train_targets,
        eval_contexts=eval_contexts,
        eval_targets=eval_targets,
        context_bits=context_bits,
        cluster_to_neurons=cluster_to_neurons,
        cluster_to_bits=cluster_to_bits,
        num_clusters=num_clusters,
        fitness_mode=fitness_mode,
        logger=logger,
    )
