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

from dataclasses import dataclass
from typing import Optional, Callable

from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
from wnn.ram.architecture.genome_log import (
    GenomeLogType,
    format_genome_log,
    format_gen_prefix,
    format_completion_log,
)
from wnn.ram.strategies.connectivity.generic_strategies import OptimizationLogger


@dataclass
class CachedEvaluatorConfig:
    """Configuration for CachedEvaluator."""
    vocab_size: int = 50257
    context_size: int = 4
    num_parts: int = 3  # Default to thirds
    num_negatives: int = 5
    empty_value: float = 0.0
    seed: Optional[int] = None  # None = time-based


class CachedEvaluator:
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
        """
        try:
            import ram_accelerator
        except ImportError:
            raise ImportError(
                "ram_accelerator not available. Build with: "
                "cd src/wnn/ram/strategies/accelerator && maturin develop --release"
            )

        self._vocab_size = vocab_size
        self._context_size = context_size
        self._num_parts = num_parts
        self._empty_value = empty_value
        self._log_path = log_path

        # Time-based seed if not specified
        if seed is None:
            import time
            seed = int(time.time() * 1000) % (2**32)

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
            seed=seed,
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
    ) -> list[tuple[float, float]]:
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
            List of (cross-entropy, accuracy) tuples for each genome
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
        elif logger is not None:
            log_debug = logger
            log_trace = logger
        else:
            log_debug = lambda x: None
            log_trace = lambda x: None

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

        if streaming and stream_batch_size < num_genomes:
            # Streaming mode: evaluate in small batches
            all_results = []

            for batch_start in range(0, num_genomes, stream_batch_size):
                batch_end = min(batch_start + stream_batch_size, num_genomes)
                batch_genomes = genomes[batch_start:batch_end]

                # Flatten this batch
                batch_bits = []
                batch_neurons = []
                batch_conns = []
                for g in batch_genomes:
                    batch_bits.extend(g.bits_per_cluster)
                    batch_neurons.extend(g.neurons_per_cluster)
                    if g.connections is not None:
                        batch_conns.extend(g.connections)

                # Evaluate this batch
                batch_results = self._cache.evaluate_genomes(
                    batch_bits,
                    batch_neurons,
                    batch_conns,
                    len(batch_genomes),
                    train_subset_idx,
                    eval_subset_idx,
                    self._empty_value,
                )

                all_results.extend(batch_results)

            results = all_results
        else:
            # Non-streaming mode: evaluate all at once
            genomes_bits_flat = []
            genomes_neurons_flat = []
            genomes_connections_flat = []

            for g in genomes:
                genomes_bits_flat.extend(g.bits_per_cluster)
                genomes_neurons_flat.extend(g.neurons_per_cluster)
                if g.connections is not None:
                    genomes_connections_flat.extend(g.connections)

            results = self._cache.evaluate_genomes(
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
            shown_count = sum(1 for _, acc in results if acc >= min_accuracy)
        else:
            shown_count = num_genomes

        # Log each result with consistent numbering using shared formatter
        current_gen = (generation + 1) if generation is not None else 1
        for i, (ce, acc) in enumerate(results):
            passes = min_accuracy is None or acc >= min_accuracy
            base_msg = format_genome_log(
                current_gen, total_gens, GenomeLogType.INITIAL,
                i + 1, num_genomes, ce, acc
            )
            if passes:
                log_debug(base_msg)
            else:
                # Log at TRACE level with subset info for filtered genomes
                log_trace(base_msg + subset_info)

        # Log generation/iteration duration summary using shared formatter
        log_debug(format_completion_log(current_gen, total_gens, elapsed, num_genomes, shown_count))

        # Force flush for real-time output
        sys.stdout.flush()
        sys.stderr.flush()

        return results

    def evaluate_batch_full(
        self,
        genomes: list[ClusterGenome],
        logger: Optional[Callable[[str], None]] = None,
    ) -> list[tuple[float, float]]:
        """
        Evaluate genomes using full train/eval data (for final evaluation).

        Args:
            genomes: List of genomes to evaluate
            logger: Optional logging function

        Returns:
            List of (cross-entropy, accuracy) tuples for each genome
        """
        log = logger or (lambda x: None)

        num_genomes = len(genomes)
        genomes_bits_flat = []
        genomes_neurons_flat = []
        genomes_connections_flat = []

        for g in genomes:
            genomes_bits_flat.extend(g.bits_per_cluster)
            genomes_neurons_flat.extend(g.neurons_per_cluster)
            if g.connections is not None:
                genomes_connections_flat.extend(g.connections)

        # Call Rust evaluator with full cached data
        results = self._cache.evaluate_genomes_full(
            genomes_bits_flat,
            genomes_neurons_flat,
            genomes_connections_flat,
            num_genomes,
            self._empty_value,
        )

        for i, (ce, acc) in enumerate(results):
            log(format_genome_log(1, 1, GenomeLogType.FINAL, i + 1, num_genomes, ce, acc))

        return results

    def evaluate_single(
        self,
        genome: ClusterGenome,
        train_subset_idx: Optional[int] = None,
        eval_subset_idx: Optional[int] = None,
    ) -> float:
        """Evaluate a single genome, returning CE only. Auto-advances rotation if indices not provided."""
        ce, _ = self.evaluate_batch([genome], train_subset_idx, eval_subset_idx)[0]
        return ce

    def evaluate_single_with_accuracy(
        self,
        genome: ClusterGenome,
        train_subset_idx: Optional[int] = None,
        eval_subset_idx: Optional[int] = None,
    ) -> tuple[float, float]:
        """Evaluate a single genome, returning (CE, accuracy). Auto-advances rotation if indices not provided."""
        return self.evaluate_batch([genome], train_subset_idx, eval_subset_idx)[0]

    def evaluate_single_full(self, genome: ClusterGenome) -> tuple[float, float]:
        """Evaluate a single genome with full data, returning (CE, accuracy)."""
        return self.evaluate_batch_full([genome])[0]

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset subset rotators with optional new seed."""
        self._cache.reset(seed)
        self._train_call_count = 0
        self._eval_call_count = 0

    @property
    def num_train_subsets(self) -> int:
        """Number of train subsets."""
        return self._cache.num_train_subsets()

    @property
    def vocab_size(self) -> int:
        """Vocabulary size."""
        return self._cache.vocab_size()

    @property
    def total_input_bits(self) -> int:
        """Total input bits per example."""
        return self._cache.total_input_bits()

    @property
    def num_parts(self) -> int:
        """Number of parts tokens are divided into."""
        return self._num_parts

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

        # Call Rust search
        results = self._cache.search_neighbors(
            base_bits=genome.bits_per_cluster,
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
            log_path=log_path,
            generation=generation,
            total_generations=total_generations,
            return_best_n=return_best_n,
        )

        # Convert results to ClusterGenome objects
        genomes = []
        for bits, neurons, connections, ce, acc in results:
            g = ClusterGenome(
                bits_per_cluster=list(bits),
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
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        tournament_size: int = 3,
        train_subset_idx: Optional[int] = None,
        eval_subset_idx: Optional[int] = None,
        seed: Optional[int] = None,
        log_path: Optional[str] = None,
        generation: Optional[int] = None,
        total_generations: Optional[int] = None,
        return_best_n: bool = True,
    ) -> list[ClusterGenome]:
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
            mutation_rate: Probability of mutation per cluster (default 0.1)
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
            List of ClusterGenome objects (viable offspring, or best N if return_best_n=True)
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
                genome.bits_per_cluster,
                genome.neurons_per_cluster,
                genome.connections if genome.connections else [],
                fitness,
            ))

        # Use instance log_path if not explicitly provided
        effective_log_path = log_path if log_path is not None else self._log_path

        # Call Rust search_offspring
        results = self._cache.search_offspring(
            population=rust_population,
            target_count=target_count,
            max_attempts=max_attempts,
            accuracy_threshold=accuracy_threshold,
            min_bits=min_bits,
            max_bits=max_bits,
            min_neurons=min_neurons,
            max_neurons=max_neurons,
            mutation_rate=mutation_rate,
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
        )

        # Convert results to ClusterGenome objects
        genomes = []
        for bits, neurons, connections, ce, acc in results:
            g = ClusterGenome(
                bits_per_cluster=list(bits),
                neurons_per_cluster=list(neurons),
                connections=list(connections) if connections else None,
            )
            # Store fitness for later use
            g._cached_fitness = (ce, acc)
            genomes.append(g)

        return genomes

    def __repr__(self) -> str:
        return (
            f"CachedEvaluator(vocab={self._vocab_size}, "
            f"context={self._context_size}, "
            f"parts={self._num_parts}, "
            f"train_calls={self._train_call_count}, "
            f"eval_calls={self._eval_call_count})"
        )
