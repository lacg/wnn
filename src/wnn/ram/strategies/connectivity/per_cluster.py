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

from dataclasses import dataclass, field
from typing import Callable, Optional, List, Dict, Tuple
from enum import IntEnum
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
    PENALIZE_HIGH_VOTES = 3 # Penalize high votes even if doesn't win (smoothest, DEFAULT)


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
    cluster_results: List[ClusterOptResult]
    initial_ppl: float
    final_ppl: float
    improvement_pct: float
    total_clusters_optimized: int
    tier_summaries: Dict[int, dict]  # tier -> {clusters, avg_improvement, ...}


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
        cluster_to_neurons: Dict[int, Tuple[int, int]],  # cluster_id -> (start_neuron, end_neuron)
        cluster_to_bits: Dict[int, int],  # cluster_id -> bits_per_neuron
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
        self._train_indices: Dict[int, Tensor] = {}
        self._eval_indices: Dict[int, Tensor] = {}
        self._precompute_indices()

        # Baseline state (populated by precompute_baseline)
        self._baseline_votes: Optional[Tensor] = None  # [M, num_clusters]
        self._baseline_rams: Dict[int, List[Dict[int, int]]] = {}  # cluster -> [neuron_rams]

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
        Precompute votes for all clusters with current connectivity.
        Called once before per-cluster optimization starts.
        """
        self._log("Precomputing baseline votes...")
        num_eval = len(self.eval_contexts)
        self._baseline_votes = torch.zeros(num_eval, self.num_clusters)
        self._baseline_rams = {}

        for cluster_id in range(self.num_clusters):
            cluster_conn = self._extract_cluster_connectivity(full_connectivity, cluster_id)
            bits = self.cluster_to_bits.get(cluster_id, 8)

            # Train and get votes
            rams, votes = self._train_and_vote_single(cluster_id, cluster_conn, bits)
            self._baseline_rams[cluster_id] = rams
            self._baseline_votes[:, cluster_id] = votes

            if (cluster_id + 1) % 1000 == 0:
                self._log(f"    Baseline: {cluster_id + 1}/{self.num_clusters} clusters")

        self._log(f"  Baseline complete: {self._baseline_votes.shape}")

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
    ) -> Tuple[List[Dict[int, int]], Tensor]:
        """Train one cluster's RAMs and compute votes on eval set."""
        num_neurons = connectivity.shape[0]
        train_idx = self._train_indices.get(cluster_id, torch.tensor([]))

        # Initialize RAMs (one dict per neuron: address -> count)
        rams: List[Dict[int, int]] = [{} for _ in range(num_neurons)]

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
        rams: List[Dict[int, int]],
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
        connectivity_variants: List[Tensor],
        fitness_mode: FitnessMode = FitnessMode.PENALIZE_HIGH_VOTES,
    ) -> List[float]:
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
        variants: List[Tensor],
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
            rams: List[Dict[int, int]] = [{} for _ in range(num_neurons)]

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
    ) -> List[float]:
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
    ) -> Tuple[Tensor, Tensor]:
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
    3. Running GA/TS optimization
    4. Returning optimized connectivity
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
        self._log = logger or print

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
        - Population: 30 variants of this cluster's connectivity
        - Crossover: Swap neurons within this cluster only
        - Mutation: Mutate connections within this cluster only
        - Fitness: Discriminative accuracy for this cluster's token
        """
        # TODO: Implement per-cluster GA
        # Key difference from global GA:
        # - Only mutate/crossover this cluster's neurons
        # - Fitness = discriminative_fitness(cluster_id, connectivity)
        raise NotImplementedError("Per-cluster GA - to be implemented")

    def optimize_cluster_ts(
        self,
        cluster_id: int,
        initial_connectivity: Tensor,
        config: TierOptConfig,
    ) -> ClusterOptResult:
        """
        Optimize a cluster's connectivity using Tabu Search.

        TS operates ONLY on this cluster's neurons:
        - Generate neighbors by mutating only this cluster's connections
        - Tabu list tracks moves within this cluster
        - Fitness: Discriminative accuracy for this cluster's token
        """
        # TODO: Implement per-cluster TS
        raise NotImplementedError("Per-cluster TS - to be implemented")

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
        tier_configs: List[TierOptConfig],
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

    def optimize_tier(
        self,
        tier: int,
        cluster_ids: List[int],
        current_connectivity: Tensor,
    ) -> Tuple[Tensor, List[ClusterOptResult]]:
        """
        Optimize all clusters in a tier.

        Args:
            tier: Tier index (0, 1, 2, ...)
            cluster_ids: List of cluster IDs in this tier
            current_connectivity: Current full connectivity tensor

        Returns:
            (updated_connectivity, list of ClusterOptResult)
        """
        config = self.tier_configs.get(tier)
        if config is None or not config.enabled:
            self._log(f"  Tier {tier}: Skipped (disabled)")
            return current_connectivity, []

        self._log(f"  Tier {tier}: Optimizing {len(cluster_ids)} clusters ({config})")

        # Order clusters
        ordered_clusters = self._order_clusters(cluster_ids)

        results = []
        updated_connectivity = current_connectivity.clone()

        for i, cluster_id in enumerate(ordered_clusters):
            # Extract this cluster's connectivity slice
            cluster_conn = self._extract_cluster_connectivity(
                updated_connectivity, cluster_id
            )

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

            # Update full connectivity with optimized cluster
            updated_connectivity = self._update_cluster_connectivity(
                updated_connectivity, cluster_id, result.final_connectivity
            )

            results.append(result)

            # Progress callback
            if self._progress_callback and (i + 1) % 10 == 0:
                # TODO: Compute current PPL
                self._progress_callback(i + 1, len(ordered_clusters), 0.0)

        return updated_connectivity, results

    def optimize_all_tiers(
        self,
        initial_connectivity: Tensor,
        tier_to_clusters: Dict[int, List[int]],
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

    def _order_clusters(self, cluster_ids: List[int]) -> List[int]:
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
# Factory function for easy creation
# =============================================================================

def create_per_cluster_optimizer(
    model,  # TieredRAMLM or similar
    train_tokens: List[str],
    eval_tokens: List[str],
    tier_budgets: Optional[Dict[int, Tuple[int, int]]] = None,  # tier -> (ga_gens, ts_iters)
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
