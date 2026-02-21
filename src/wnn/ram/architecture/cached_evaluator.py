"""
Cached Evaluator - Uses Rust TokenCache for zero-copy genome evaluation.

Holds all tokens in Rust memory for the entire session. Provides per-iteration
subset rotation with zero Python→Rust data transfer overhead.

Usage:
    # Create cache once at session start
    evaluator = CachedEvaluator(
        train_tokens=train_tokens,
        eval_tokens=eval_tokens,
        vocab_size=50257,
        context_size=4,
        cluster_order=cluster_order,
        num_parts=3,  # Divide into thirds
    )

    # Per iteration: get subset indices and evaluate
    train_idx = evaluator.next_train_idx()
    eval_idx = evaluator.next_eval_idx()
    results = evaluator.evaluate_batch(genomes, train_idx, eval_idx)

    # Final evaluation with full data
    final_results = evaluator.evaluate_batch_full(genomes)
"""

import os
from dataclasses import dataclass
from typing import Optional, Callable, NamedTuple

from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
from wnn.ram.architecture.genome_log import (
    GenomeLogType,
    format_genome_log,
    format_gen_prefix,
    format_completion_log,
)
from wnn.ram.strategies.connectivity.generic_strategies import OptimizationLogger
from wnn.ram.core import RAMClusterLayer, GatingModel, create_gating
from wnn.ram.architecture.base_evaluator import BaseEvaluator, EvalResult


class OffspringSearchResult(NamedTuple):
    """Result of offspring search with counts for tracking.

    Attributes:
        genomes: List of ClusterGenome objects (viable + fallback if return_best_n)
        evaluated: Total candidates evaluated
        viable: Candidates that passed accuracy threshold (before fallback)
    """
    genomes: list[ClusterGenome]
    evaluated: int
    viable: int


@dataclass
class CachedEvaluatorConfig:
    """Configuration for CachedEvaluator."""
    vocab_size: int = 50257
    context_size: int = 4
    num_parts: int = 3  # Default to thirds
    num_negatives: int = 5
    empty_value: float = 0.0
    seed: Optional[int] = None  # None = time-based
    use_hybrid: bool = True  # Use hybrid CPU+GPU evaluation (4-8x faster)
    # Gating configuration (optional)
    gating_enabled: bool = False  # Enable RAM-based gating
    gating_neurons_per_cluster: int = 8  # Neurons per gate (majority voting)
    gating_bits_per_neuron: int = 12  # Address space for gate neurons
    gating_threshold: float = 0.5  # Threshold for majority voting


class CachedEvaluator(BaseEvaluator):
    """
    Rust-backed evaluator with persistent token storage.

    All tokens are stored in Rust memory once. Per-iteration evaluations
    use zero-copy subset selection via indices.
    """

    def __init__(
        self,
        train_tokens: list[int],
        eval_tokens: list[int],
        vocab_size: int,
        context_size: int,
        cluster_order: list[int],
        num_parts: int = 3,
        num_negatives: int = 5,
        empty_value: float = 0.0,
        seed: Optional[int] = None,  # None = time-based
        log_path: Optional[str] = None,
        use_hybrid: bool = True,  # Use hybrid CPU+GPU evaluation (4-8x faster)
    ):
        """
        Create a cached evaluator with all tokens pre-encoded in Rust.

        Args:
            train_tokens: Full training token sequence
            eval_tokens: Full evaluation token sequence
            vocab_size: Vocabulary size
            context_size: Context window size
            cluster_order: Token IDs sorted by frequency (most frequent first)
            num_parts: Number of subsets to divide train data into (3=thirds)
            num_negatives: Number of negative samples per example
            empty_value: Value for EMPTY cells (0.0 recommended)
            seed: Random seed for subset rotation
            use_hybrid: If True, use hybrid CPU+GPU evaluation (4-8x faster, default True)
        """
        try:
            import ram_accelerator
        except ImportError:
            raise ImportError(
                "ram_accelerator not available. Build with: "
                "cd src/wnn/ram/strategies/accelerator && maturin develop --release"
            )

        super().__init__(
            train_tokens=train_tokens,
            eval_tokens=eval_tokens,
            vocab_size=vocab_size,
            context_size=context_size,
            num_parts=num_parts,
            seed=seed,
        )

        # CachedEvaluator-specific fields
        self._empty_value = empty_value
        self._log_path = log_path
        self._use_hybrid = use_hybrid

        # Create the Rust TokenCache
        self._cache = ram_accelerator.TokenCacheWrapper(
            train_tokens=[int(t) for t in train_tokens],
            eval_tokens=[int(t) for t in eval_tokens],
            test_tokens=[],  # Not used currently
            vocab_size=vocab_size,
            context_size=context_size,
            cluster_order=cluster_order,
            num_parts=num_parts,
            num_negatives=num_negatives,
            seed=self._seed,
        )

        # Track call counts for logging
        self._train_call_count = 0
        self._eval_call_count = 0

    def next_train_idx(self) -> int:
        """Get the next train subset index (advances rotator)."""
        self._train_call_count += 1
        return self._cache.next_train_idx()

    def next_eval_idx(self) -> int:
        """Get the next eval subset index (advances rotator)."""
        self._eval_call_count += 1
        return self._cache.next_eval_idx()

    def evaluate_batch(
        self,
        genomes: list[ClusterGenome],
        train_subset_idx: Optional[int] = None,  # If None, auto-advances rotation
        eval_subset_idx: Optional[int] = None,   # If None, uses 0 (full eval)
        logger: Optional[Callable[[str], None]] = None,
        generation: Optional[int] = None,
        total_generations: Optional[int] = None,
        min_accuracy: Optional[float] = None,
        streaming: bool = True,  # Log per-genome as they complete
        stream_batch_size: int = 1,  # How many genomes per Rust call in streaming mode
    ) -> list[EvalResult]:
        """
        Evaluate multiple genomes using specified train/eval subsets.

        This is the main evaluation function. Uses pre-cached data in Rust,
        no data copying occurs - just pointer arithmetic to select subsets.

        Args:
            genomes: List of genomes to evaluate
            train_subset_idx: Which train subset to use (0 to num_parts-1). If None, auto-advances.
            eval_subset_idx: Which eval subset to use (typically 0 for full eval). If None, uses 0.
            logger: Optional logging function
            generation: Current generation number for logging
            total_generations: Total number of generations for logging
            min_accuracy: If provided, genomes below this log at TRACE level
            streaming: If True, evaluate and log genomes incrementally (default True)
            stream_batch_size: Number of genomes per Rust call in streaming mode (default 1)

        Returns:
            List of EvalResult for each genome (bit_accuracy=None for tiered)
        """
        import time
        import sys

        # Auto-advance rotation if no explicit index provided
        if train_subset_idx is None:
            train_subset_idx = self.next_train_idx()
        if eval_subset_idx is None:
            eval_subset_idx = 0  # Eval typically uses full data

        # Use OptimizationLogger for leveled logging, or fallback
        if isinstance(logger, OptimizationLogger):
            log_debug = logger.debug
            log_trace = logger.trace
            log_info = logger.info  # For streaming (always visible)
        elif logger is not None:
            log_debug = logger
            log_trace = logger
            log_info = logger
        else:
            log_debug = lambda x: None
            log_trace = lambda x: None
            log_info = lambda x: None

        num_genomes = len(genomes)
        total_gens = total_generations if total_generations is not None else num_genomes

        # Generation prefix for logs using shared formatter
        if generation is not None:
            gen_prefix = format_gen_prefix(generation + 1, total_gens)
        else:
            gen_prefix = f"[Gen 01/{total_gens:0{len(str(total_gens))}d}]"

        # Subset info for TRACE level logging
        subset_info = f" [train:{train_subset_idx+1}/{self._num_parts}, eval:{eval_subset_idx+1}/1]"

        overall_start = time.time()

        if streaming:
            # Streaming mode: enable Rust per-genome progress logging
            # Rust will log each genome as it completes (~3.5s/genome)
            current_gen = (generation + 1) if generation is not None else 1

            # Enable Rust progress logging
            os.environ['WNN_PROGRESS_LOG'] = '1'
            os.environ['WNN_PROGRESS_GEN'] = str(current_gen)
            os.environ['WNN_PROGRESS_TOTAL_GENS'] = str(total_gens)
            os.environ['WNN_PROGRESS_TYPE'] = 'Init'
            os.environ['WNN_PROGRESS_TOTAL'] = str(num_genomes)
            os.environ['WNN_PROGRESS_OFFSET'] = '0'
            if self._log_path:
                os.environ['WNN_LOG_PATH'] = self._log_path

            # Flatten ALL genomes (single Rust call, Rust handles per-genome logging)
            genomes_bits_flat = []
            genomes_neurons_flat = []
            genomes_connections_flat = []
            for g in genomes:
                genomes_bits_flat.extend(g.bits_per_neuron)
                genomes_neurons_flat.extend(g.neurons_per_cluster)
                if g.connections is not None:
                    genomes_connections_flat.extend(g.connections)

            # Single Rust call - Rust logs each genome as it completes
            if self._use_hybrid:
                raw_results = self._cache.evaluate_genomes_hybrid(
                    genomes_bits_flat,
                    genomes_neurons_flat,
                    genomes_connections_flat,
                    num_genomes,
                    train_subset_idx,
                    eval_subset_idx,
                    self._empty_value,
                )
            else:
                raw_results = self._cache.evaluate_genomes(
                    genomes_bits_flat,
                    genomes_neurons_flat,
                    genomes_connections_flat,
                    num_genomes,
                    train_subset_idx,
                    eval_subset_idx,
                    self._empty_value,
                )

            # Disable progress logging after evaluation
            os.environ.pop('WNN_PROGRESS_LOG', None)
        else:
            # Non-streaming mode: evaluate all at once
            current_gen = (generation + 1) if generation is not None else 1

            # Check if Rust is handling progress logging
            rust_progress = os.environ.get('WNN_PROGRESS_LOG') == '1'

            # Set generation info for Rust progress logging
            os.environ['WNN_PROGRESS_GEN'] = str(current_gen)
            os.environ['WNN_PROGRESS_TOTAL_GENS'] = str(total_gens)
            os.environ['WNN_PROGRESS_TYPE'] = 'Init'
            os.environ['WNN_PROGRESS_OFFSET'] = '0'
            os.environ['WNN_PROGRESS_TOTAL'] = str(num_genomes)

            genomes_bits_flat = []
            genomes_neurons_flat = []
            genomes_connections_flat = []

            for g in genomes:
                genomes_bits_flat.extend(g.bits_per_neuron)
                genomes_neurons_flat.extend(g.neurons_per_cluster)
                if g.connections is not None:
                    genomes_connections_flat.extend(g.connections)

            # Use hybrid evaluation if enabled (4-8x faster)
            if self._use_hybrid:
                raw_results = self._cache.evaluate_genomes_hybrid(
                    genomes_bits_flat,
                    genomes_neurons_flat,
                    genomes_connections_flat,
                    num_genomes,
                    train_subset_idx,
                    eval_subset_idx,
                    self._empty_value,
                )
            else:
                raw_results = self._cache.evaluate_genomes(
                    genomes_bits_flat,
                    genomes_neurons_flat,
                    genomes_connections_flat,
                    num_genomes,
                    train_subset_idx,
                    eval_subset_idx,
                    self._empty_value,
                )

        elapsed = time.time() - overall_start

        # Count how many pass the threshold
        if min_accuracy is not None:
            shown_count = sum(1 for _, acc in raw_results if acc >= min_accuracy)
        else:
            shown_count = num_genomes

        # Check if Rust is handling progress logging (skip Python duplicate)
        rust_progress = os.environ.get('WNN_PROGRESS_LOG') == '1'

        # Log each result (only for non-streaming mode - streaming already logged per-genome)
        # Skip if Rust is handling progress logging
        streaming_was_used = streaming and stream_batch_size < num_genomes
        if not streaming_was_used and not rust_progress:
            current_gen = (generation + 1) if generation is not None else 1
            avg_duration = elapsed / num_genomes if num_genomes > 0 else 0.0
            for i, (ce, acc) in enumerate(raw_results):
                passes = min_accuracy is None or acc >= min_accuracy
                base_msg = format_genome_log(
                    current_gen, total_gens, GenomeLogType.INITIAL,
                    i + 1, num_genomes, ce, acc, duration=avg_duration
                )
                if passes:
                    log_debug(base_msg)
                else:
                    # Log at TRACE level with subset info for filtered genomes
                    log_trace(base_msg + subset_info)

        # Log generation/iteration duration summary (always show this)
        current_gen = (generation + 1) if generation is not None else 1
        log_debug(format_completion_log(current_gen, total_gens, elapsed, num_genomes, shown_count))

        # Force flush for real-time output
        sys.stdout.flush()
        sys.stderr.flush()

        return [EvalResult(ce=ce, accuracy=acc) for ce, acc in raw_results]

    def evaluate_batch_full(
        self,
        genomes: list[ClusterGenome],
        logger: Optional[Callable[[str], None]] = None,
    ) -> list[EvalResult]:
        """
        Evaluate genomes using full train/eval data (for final evaluation).

        Args:
            genomes: List of genomes to evaluate
            logger: Optional logging function

        Returns:
            List of EvalResult for each genome (bit_accuracy=None for tiered)
        """
        log = logger or (lambda x: None)

        num_genomes = len(genomes)
        genomes_bits_flat = []
        genomes_neurons_flat = []
        genomes_connections_flat = []

        for g in genomes:
            genomes_bits_flat.extend(g.bits_per_neuron)
            genomes_neurons_flat.extend(g.neurons_per_cluster)
            if g.connections is not None:
                genomes_connections_flat.extend(g.connections)

        # Call Rust evaluator with full cached data (use hybrid if enabled)
        if self._use_hybrid:
            raw_results = self._cache.evaluate_genomes_full_hybrid(
                genomes_bits_flat,
                genomes_neurons_flat,
                genomes_connections_flat,
                num_genomes,
                self._empty_value,
            )
        else:
            raw_results = self._cache.evaluate_genomes_full(
                genomes_bits_flat,
                genomes_neurons_flat,
                genomes_connections_flat,
                num_genomes,
                self._empty_value,
            )

        for i, (ce, acc) in enumerate(raw_results):
            log(format_genome_log(1, 1, GenomeLogType.FINAL, i + 1, num_genomes, ce, acc))

        return [EvalResult(ce=ce, accuracy=acc) for ce, acc in raw_results]

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset subset rotators with optional new seed."""
        self._cache.reset(seed)
        self._train_call_count = 0
        self._eval_call_count = 0

    @property
    def num_train_subsets(self) -> int:
        """Number of train subsets."""
        return self._cache.num_train_subsets()

    def search_neighbors(
        self,
        genome: ClusterGenome,
        target_count: int,
        max_attempts: int,
        accuracy_threshold: float,
        min_bits: int,
        max_bits: int,
        min_neurons: int,
        max_neurons: int,
        bits_mutation_rate: float = 0.1,
        neurons_mutation_rate: float = 0.05,
        train_subset_idx: Optional[int] = None,
        eval_subset_idx: Optional[int] = None,
        seed: Optional[int] = None,
        log_path: Optional[str] = None,
        generation: Optional[int] = None,
        total_generations: Optional[int] = None,
        return_best_n: bool = True,
        mutable_clusters: Optional[list[int]] = None,
    ) -> list[ClusterGenome]:
        """
        Search for neighbor genomes above accuracy threshold, entirely in Rust.

        This eliminates Python↔Rust round trips by doing mutation, evaluation,
        and filtering all in a single Rust call. Much faster than the traditional
        approach of: Python generate → Rust evaluate → Python filter → repeat.

        Args:
            genome: Base genome to mutate from
            target_count: How many good candidates we want
            max_attempts: Maximum candidates to evaluate (cap)
            accuracy_threshold: Minimum accuracy to pass (e.g., 0.0001 for 0.01%)
            min_bits, max_bits: Bits bounds for mutation
            min_neurons, max_neurons: Neurons bounds for mutation
            bits_mutation_rate: Probability of mutating bits per cluster (default 0.1)
            neurons_mutation_rate: Probability of mutating neurons per cluster (default 0.05)
            train_subset_idx: Which train subset to use (auto-advances if None)
            eval_subset_idx: Which eval subset to use (0 = full if None)
            seed: Random seed for mutations (time-based if None)
            log_path: Optional log file path for progress logging
            generation: Current generation for log prefix
            total_generations: Total generations for log prefix
            return_best_n: If True, return best N even if threshold not met

        Returns:
            List of ClusterGenome objects that passed the threshold
            (or best N if return_best_n=True and threshold not met)
        """
        import time

        # Auto-advance rotation if no explicit index
        if train_subset_idx is None:
            train_subset_idx = self.next_train_idx()
        if eval_subset_idx is None:
            eval_subset_idx = 0

        # Time-based seed if not provided
        if seed is None:
            seed = int(time.time() * 1000) % (2**32)

        # Use instance log_path if not explicitly provided
        effective_log_path = log_path if log_path is not None else self._log_path

        # Call Rust search
        results = self._cache.search_neighbors(
            base_bits=genome.bits_per_neuron,
            base_neurons=genome.neurons_per_cluster,
            base_connections=genome.connections if genome.connections else [],
            target_count=target_count,
            max_attempts=max_attempts,
            accuracy_threshold=accuracy_threshold,
            min_bits=min_bits,
            max_bits=max_bits,
            min_neurons=min_neurons,
            max_neurons=max_neurons,
            bits_mutation_rate=bits_mutation_rate,
            neurons_mutation_rate=neurons_mutation_rate,
            train_subset_idx=train_subset_idx,
            eval_subset_idx=eval_subset_idx,
            empty_value=self._empty_value,
            seed=seed,
            log_path=effective_log_path,
            generation=generation,
            total_generations=total_generations,
            return_best_n=return_best_n,
            mutable_clusters=mutable_clusters,
        )

        # Convert results to ClusterGenome objects
        genomes = []
        for bits, neurons, connections, ce, acc in results:
            g = ClusterGenome(
                bits_per_neuron=list(bits),
                neurons_per_cluster=list(neurons),
                connections=list(connections) if connections else None,
            )
            # Store fitness for later use
            g._cached_fitness = (ce, acc)
            genomes.append(g)

        return genomes

    def search_offspring(
        self,
        population: list[tuple[ClusterGenome, float]],  # (genome, fitness) tuples
        target_count: int,
        max_attempts: int,
        accuracy_threshold: float,
        min_bits: int,
        max_bits: int,
        min_neurons: int,
        max_neurons: int,
        bits_mutation_rate: float = 0.1,
        neurons_mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        tournament_size: int = 3,
        train_subset_idx: Optional[int] = None,
        eval_subset_idx: Optional[int] = None,
        seed: Optional[int] = None,
        log_path: Optional[str] = None,
        generation: Optional[int] = None,
        total_generations: Optional[int] = None,
        return_best_n: bool = True,
        mutable_clusters: Optional[list[int]] = None,  # List of cluster indices to mutate (None = all)
    ) -> OffspringSearchResult:
        """
        Search for GA offspring above accuracy threshold, entirely in Rust.

        Performs tournament selection, crossover, mutation, and evaluation
        all in a single Rust call. Much faster than Python-side GA loops.

        Args:
            population: List of (genome, fitness) tuples for parent selection
            target_count: How many viable offspring we want
            max_attempts: Maximum offspring to generate (cap)
            accuracy_threshold: Minimum accuracy to pass (e.g., 0.0001 for 0.01%)
            min_bits, max_bits: Bits bounds for mutation
            min_neurons, max_neurons: Neurons bounds for mutation
            bits_mutation_rate: Probability of mutating bits per cluster (0.0 to disable)
            neurons_mutation_rate: Probability of mutating neurons per cluster (0.0 to disable)
            crossover_rate: Probability of crossover vs clone (default 0.7)
            tournament_size: Tournament selection size (default 3)
            train_subset_idx: Which train subset to use (auto-advances if None)
            eval_subset_idx: Which eval subset to use (0 = full if None)
            seed: Random seed (time-based if None)
            log_path: Optional log file path
            generation: Current generation for log prefix
            total_generations: Total generations for log prefix
            return_best_n: If True, return best N by CE even if threshold not met

        Returns:
            OffspringSearchResult with:
            - genomes: List of ClusterGenome objects (viable + fallback if return_best_n=True)
            - evaluated: Total candidates evaluated
            - viable: Candidates that passed accuracy threshold (before fallback)
        """
        import time

        if not population:
            return []

        # Auto-advance rotation if no explicit index
        if train_subset_idx is None:
            train_subset_idx = self.next_train_idx()
        if eval_subset_idx is None:
            eval_subset_idx = 0

        # Time-based seed if not provided
        if seed is None:
            seed = int(time.time() * 1000) % (2**32)

        # Convert population to Rust format: (bits, neurons, connections, fitness)
        rust_population = []
        for genome, fitness in population:
            rust_population.append((
                genome.bits_per_neuron,
                genome.neurons_per_cluster,
                genome.connections if genome.connections else [],
                fitness,
            ))

        # Use instance log_path if not explicitly provided
        effective_log_path = log_path if log_path is not None else self._log_path

        # Call Rust search_offspring - returns (candidates, evaluated, viable)
        candidates, evaluated, viable = self._cache.search_offspring(
            population=rust_population,
            target_count=target_count,
            max_attempts=max_attempts,
            accuracy_threshold=accuracy_threshold,
            min_bits=min_bits,
            max_bits=max_bits,
            min_neurons=min_neurons,
            max_neurons=max_neurons,
            bits_mutation_rate=bits_mutation_rate,
            neurons_mutation_rate=neurons_mutation_rate,
            crossover_rate=crossover_rate,
            tournament_size=tournament_size,
            train_subset_idx=train_subset_idx,
            eval_subset_idx=eval_subset_idx,
            empty_value=self._empty_value,
            seed=seed,
            log_path=effective_log_path,
            generation=generation,
            total_generations=total_generations,
            return_best_n=return_best_n,
            mutable_clusters=mutable_clusters,  # Tier0-only: only mutate first N clusters
        )

        # Convert results to ClusterGenome objects
        genomes = []
        for bits, neurons, connections, ce, acc in candidates:
            g = ClusterGenome(
                bits_per_neuron=list(bits),
                neurons_per_cluster=list(neurons),
                connections=list(connections) if connections else None,
            )
            # Store fitness for later use
            g._cached_fitness = (ce, acc)
            genomes.append(g)

        return OffspringSearchResult(genomes=genomes, evaluated=evaluated, viable=viable)

    # ─────────────────────────────────────────────────────────────────────
    # Gating Support
    # ─────────────────────────────────────────────────────────────────────

    def create_gating_model(
        self,
        genome: ClusterGenome,
        neurons_per_gate: int = 8,
        bits_per_neuron: int = 12,
        threshold: float = 0.5,
        rng: Optional[int] = None,
        prefer_rust: bool = True,
    ) -> GatingModel:
        """
        Create a gating model for use with this evaluator's genomes.

        The gating model learns to filter cluster scores based on input context.
        It's designed for staged training:
        1. Base RAM is trained via Rust accelerator (GA/TS optimization)
        2. Gating is trained separately (Rust or Python)
        3. At inference, gating is applied to Rust-computed scores

        Uses Rust-accelerated gating by default (faster), falls back to Python
        if Rust accelerator is not available.

        Args:
            genome: ClusterGenome to create gating for (uses num_clusters)
            neurons_per_gate: Neurons per gate - majority voting (default 8)
            bits_per_neuron: Address space for gate neurons (default 12)
            threshold: Majority voting threshold (default 0.5)
            rng: Random seed for reproducibility
            prefer_rust: If True, use Rust gating if available (default True)

        Returns:
            GatingModel instance (RustRAMGating or RAMGating)
        """
        return create_gating(
            total_input_bits=self.total_input_bits,
            num_clusters=genome.num_clusters,
            neurons_per_gate=neurons_per_gate,
            bits_per_neuron=bits_per_neuron,
            threshold=threshold,
            rng=rng,
            prefer_rust=prefer_rust,
        )

    def create_gating_from_config(
        self,
        genome: ClusterGenome,
        config: 'CachedEvaluatorConfig',
        rng: Optional[int] = None,
    ) -> Optional[GatingModel]:
        """
        Create a gating model using config parameters.

        Uses Rust-accelerated gating by default for better performance.

        Args:
            genome: ClusterGenome to create gating for
            config: CachedEvaluatorConfig with gating parameters
            rng: Random seed

        Returns:
            GatingModel if config.gating_enabled, else None
        """
        if not config.gating_enabled:
            return None

        return self.create_gating_model(
            genome=genome,
            neurons_per_gate=config.gating_neurons_per_cluster,
            bits_per_neuron=config.gating_bits_per_neuron,
            threshold=config.gating_threshold,
            rng=rng,
            prefer_rust=True,
        )

    def evaluate_single_full_gated(
        self,
        genome: ClusterGenome,
        train_tokens: list[int],
        neurons_per_gate: int = 8,
        bits_per_neuron: int = 12,
        threshold: float = 0.5,
        batch_size: int = 256,
        logger: Optional[Callable[[str], None]] = None,
        gating_seed: int = 42,
    ) -> dict:
        """
        Evaluate a genome with and without gating for comparison.

        This method uses the Rust accelerator to:
        1. Train base RAM on full training data
        2. Train gating model on training data (target gate = true only for target cluster)
        3. Evaluate WITHOUT gating → (ce, acc)
        4. Evaluate WITH gating → (gated_ce, gated_acc)

        The gated evaluation properly applies gates to cluster scores during
        prediction, giving accurate gated metrics (not just coverage-based estimates).

        Used by the UI-driven gating analysis feature to compare
        gated vs non-gated performance on completed experiments.

        Args:
            genome: ClusterGenome to evaluate
            train_tokens: Training tokens (unused - cache already has train data)
            neurons_per_gate: Neurons per gate for majority voting (default 8)
            bits_per_neuron: Address space per gate neuron (default 12)
            threshold: Majority voting threshold (default 0.5)
            batch_size: Unused (kept for API compatibility)
            logger: Optional logging function
            gating_seed: Random seed for gating connectivity initialization

        Returns:
            Dict with:
                ce: Cross-entropy without gating
                acc: Accuracy without gating
                gated_ce: Cross-entropy with gating
                gated_acc: Accuracy with gating
                gating_config: Dict with neurons_per_gate, bits_per_neuron, threshold
        """
        import time

        log = logger or (lambda x: None)

        log("  Evaluating with Rust gated evaluation...")
        start = time.time()

        # Get genome configuration (direct attributes, not methods)
        bits_flat = genome.bits_per_neuron
        neurons_flat = genome.neurons_per_cluster
        connections_flat = genome.connections if genome.connections else []

        # Call Rust implementation via TokenCacheWrapper
        ce, acc, gated_ce, gated_acc = self._cache.evaluate_genome_with_gating(
            bits_flat=bits_flat,
            neurons_flat=neurons_flat,
            connections_flat=connections_flat,
            neurons_per_gate=neurons_per_gate,
            bits_per_gate_neuron=bits_per_neuron,
            vote_threshold_frac=threshold,
            empty_value=self._empty_value,
            gating_seed=gating_seed,
        )

        elapsed = time.time() - start
        log(f"    Non-gated: CE={ce:.4f}, Acc={acc*100:.4f}%")
        log(f"    Gated:     CE={gated_ce:.4f}, Acc={gated_acc*100:.4f}%")
        log(f"    Evaluation completed in {elapsed:.1f}s")

        return {
            'ce': ce,
            'acc': acc,
            'gated_ce': gated_ce,
            'gated_acc': gated_acc,
            'gating_config': {
                'neurons_per_gate': neurons_per_gate,
                'bits_per_neuron': bits_per_neuron,
                'threshold': threshold,
            },
        }

    def __repr__(self) -> str:
        return (
            f"CachedEvaluator(vocab={self._vocab_size}, "
            f"context={self._context_size}, "
            f"parts={self._num_parts}, "
            f"train_calls={self._train_call_count}, "
            f"eval_calls={self._eval_call_count})"
        )
