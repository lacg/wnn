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
from typing import List, Optional, Tuple, Callable

from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
from wnn.ram.strategies.connectivity.generic_strategies import OptimizationLogger


@dataclass
class CachedEvaluatorConfig:
    """Configuration for CachedEvaluator."""
    vocab_size: int = 50257
    context_size: int = 4
    num_parts: int = 3  # Default to thirds
    num_negatives: int = 5
    empty_value: float = 0.0
    seed: int = 42


class CachedEvaluator:
    """
    Rust-backed evaluator with persistent token storage.

    All tokens are stored in Rust memory once. Per-iteration evaluations
    use zero-copy subset selection via indices.
    """

    def __init__(
        self,
        train_tokens: List[int],
        eval_tokens: List[int],
        vocab_size: int,
        context_size: int,
        cluster_order: List[int],
        num_parts: int = 3,
        num_negatives: int = 5,
        empty_value: float = 0.0,
        seed: int = 42,
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
        genomes: List[ClusterGenome],
        train_subset_idx: Optional[int] = None,  # If None, auto-advances rotation
        eval_subset_idx: Optional[int] = None,   # If None, uses 0 (full eval)
        logger: Optional[Callable[[str], None]] = None,
        generation: Optional[int] = None,
        total_generations: Optional[int] = None,
        min_accuracy: Optional[float] = None,
        streaming: bool = True,  # Log per-genome as they complete
        stream_batch_size: int = 1,  # How many genomes per Rust call in streaming mode
    ) -> List[Tuple[float, float]]:
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
        genome_width = len(str(num_genomes))

        # Generation prefix for logs
        if generation is not None:
            if total_generations is not None:
                gen_width = len(str(total_generations))
                gen_prefix = f"[Gen {generation + 1:0{gen_width}d}/{total_generations}]"
            else:
                gen_prefix = f"[Gen {generation + 1}]"
        else:
            gen_prefix = "[Init]"

        # Subset info for TRACE level logging
        subset_info = f" [train:{train_subset_idx+1}/{self._num_parts}, eval:{eval_subset_idx+1}/1]"

        if streaming and stream_batch_size < num_genomes:
            # Streaming mode: evaluate in small batches and log immediately
            all_results = []
            overall_start = time.time()

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

                batch_start_time = time.time()

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

                batch_elapsed = time.time() - batch_start_time

                # Log each result immediately
                for i, (ce, acc) in enumerate(batch_results):
                    genome_idx = batch_start + i
                    base_msg = f"{gen_prefix} Genome {genome_idx+1:0{genome_width}d}/{num_genomes}: CE={ce:.4f}, Acc={acc:.2%} in {batch_elapsed:.1f}s"
                    trace_msg = base_msg + subset_info

                    if min_accuracy is not None and acc < min_accuracy:
                        log_trace(trace_msg)
                    else:
                        log_debug(base_msg)
                        if isinstance(logger, OptimizationLogger):
                            log_trace(f"    └─ subset: train={train_subset_idx+1}/{self._num_parts}, eval={eval_subset_idx+1}/1")

                    # Force flush for real-time output
                    sys.stdout.flush()
                    sys.stderr.flush()

                all_results.extend(batch_results)

            return all_results
        else:
            # Non-streaming mode: evaluate all at once (original behavior)
            genomes_bits_flat = []
            genomes_neurons_flat = []
            genomes_connections_flat = []

            for g in genomes:
                genomes_bits_flat.extend(g.bits_per_cluster)
                genomes_neurons_flat.extend(g.neurons_per_cluster)
                if g.connections is not None:
                    genomes_connections_flat.extend(g.connections)

            start_time = time.time()

            results = self._cache.evaluate_genomes(
                genomes_bits_flat,
                genomes_neurons_flat,
                genomes_connections_flat,
                num_genomes,
                train_subset_idx,
                eval_subset_idx,
                self._empty_value,
            )

            elapsed = time.time() - start_time

            # Log each result
            for i, (ce, acc) in enumerate(results):
                base_msg = f"{gen_prefix} Genome {i+1:0{genome_width}d}/{num_genomes}: CE={ce:.4f}, Acc={acc:.2%} in {elapsed:.1f}s"
                trace_msg = base_msg + subset_info

                if min_accuracy is not None and acc < min_accuracy:
                    log_trace(trace_msg)
                else:
                    log_debug(base_msg)
                    if isinstance(logger, OptimizationLogger):
                        log_trace(f"    └─ subset: train={train_subset_idx+1}/{self._num_parts}, eval={eval_subset_idx+1}/1")

            return results

    def evaluate_batch_full(
        self,
        genomes: List[ClusterGenome],
        logger: Optional[Callable[[str], None]] = None,
    ) -> List[Tuple[float, float]]:
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
            log(f"[Final] Genome {i+1}/{num_genomes}: CE={ce:.4f}, Acc={acc:.2%}")

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
    ) -> Tuple[float, float]:
        """Evaluate a single genome, returning (CE, accuracy). Auto-advances rotation if indices not provided."""
        return self.evaluate_batch([genome], train_subset_idx, eval_subset_idx)[0]

    def evaluate_single_full(self, genome: ClusterGenome) -> Tuple[float, float]:
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
    ) -> List[ClusterGenome]:
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

    def __repr__(self) -> str:
        return (
            f"CachedEvaluator(vocab={self._vocab_size}, "
            f"context={self._context_size}, "
            f"parts={self._num_parts}, "
            f"train_calls={self._train_call_count}, "
            f"eval_calls={self._eval_call_count})"
        )
