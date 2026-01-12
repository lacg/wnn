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
    Efficient evaluator that caches baseline votes and only recomputes changed clusters.

    Instead of re-training the entire model for each connectivity variant,
    we cache all clusters' votes and only update the changed cluster.

    This makes per-cluster evaluation O(cluster_size) instead of O(all_neurons).
    """

    def __init__(
        self,
        train_contexts: Tensor,      # [N, context_size] - context token IDs
        train_targets: Tensor,       # [N] - target token IDs
        eval_contexts: Tensor,       # [M, context_size]
        eval_targets: Tensor,        # [M]
        context_bits: int,           # Total input bits (context_size * bits_per_token)
        cluster_to_neurons: Dict[int, Tuple[int, int]],  # cluster_id -> (start_neuron, end_neuron)
        cluster_to_bits: Dict[int, int],  # cluster_id -> bits_per_neuron for this cluster
        logger: Optional[Callable[[str], None]] = None,
    ):
        self.train_contexts = train_contexts
        self.train_targets = train_targets
        self.eval_contexts = eval_contexts
        self.eval_targets = eval_targets
        self.context_bits = context_bits
        self.cluster_to_neurons = cluster_to_neurons
        self.cluster_to_bits = cluster_to_bits
        self._log = logger or print

        # Will be populated by precompute_baseline()
        self._baseline_votes: Optional[Tensor] = None  # [N, num_clusters] vote counts
        self._cluster_rams: Dict[int, dict] = {}  # cluster_id -> trained RAM state

    def precompute_baseline(self, full_connectivity: Tensor) -> None:
        """
        Precompute votes for all clusters with current connectivity.
        This is called once before per-cluster optimization starts.
        """
        # TODO: Implement baseline computation
        # This will train all RAMs and cache their votes
        raise NotImplementedError("Baseline computation - to be implemented")

    def evaluate_cluster_variant(
        self,
        cluster_id: int,
        new_connectivity: Tensor,
        fitness_mode: FitnessMode = FitnessMode.PENALIZE_HIGH_VOTES,
    ) -> float:
        """
        Evaluate a connectivity variant for a single cluster.

        Only recomputes this cluster's votes, uses cached baseline for others.

        Args:
            cluster_id: Which cluster to evaluate
            new_connectivity: New connectivity for this cluster's neurons
            fitness_mode: How to compute discriminative fitness

        Returns:
            Fitness score (higher is better)
        """
        # TODO: Implement incremental evaluation
        raise NotImplementedError("Incremental evaluation - to be implemented")

    def get_cluster_examples(
        self,
        cluster_id: int,
        split: str = "train",
    ) -> Tuple[Tensor, Tensor]:
        """
        Get examples where target = cluster_id (positive examples for this cluster).

        Args:
            cluster_id: The cluster/token ID
            split: "train" or "eval"

        Returns:
            (contexts, targets) tensors filtered to this cluster
        """
        if split == "train":
            mask = self.train_targets == cluster_id
            return self.train_contexts[mask], self.train_targets[mask]
        else:
            mask = self.eval_targets == cluster_id
            return self.eval_contexts[mask], self.eval_targets[mask]


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
