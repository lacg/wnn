#!/usr/bin/env python3
"""
Systematic Benchmarks for RAM Networks

Comprehensive test suite for:
1. Generalization Strategies (DIRECT, BIT_LEVEL, COMPOSITIONAL, HASH, RESIDUAL)
2. Training Modes (GREEDY, ITERATIVE)
3. Tasks (Successor, Copy, Reverse, Increment, Parity)

Run with: python tests/benchmarks.py
"""

import sys
sys.path.insert(0, 'src')

from torch import tensor, uint8, zeros
from dataclasses import dataclass
from time import perf_counter
from typing import Callable

from wnn.ram.factories import MapperFactory
from wnn.ram.enums import MapperStrategy, ContextMode

import random


# ============================================================
# Benchmark Infrastructure
# ============================================================

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    strategy: str
    task: str
    n_bits: int
    train_examples: int
    test_examples: int
    train_accuracy: float
    test_accuracy: float  # Generalization accuracy
    training_time_ms: float
    epochs: int


@dataclass
class BenchmarkTask:
    """Definition of a benchmark task."""
    name: str
    description: str
    generator: Callable[[int], tuple[list, list]]  # (train_data, test_data)
    n_bits: int = 8


def int_to_bits(n: int, n_bits: int) -> tensor:
    """Convert integer to bit tensor (MSB first)."""
    bits = []
    for i in range(n_bits - 1, -1, -1):
        bits.append((n >> i) & 1)
    return tensor(bits, dtype=uint8)


def bits_to_int(bits: tensor) -> int:
    """Convert bit tensor to integer."""
    bits = bits.squeeze()
    n_bits = len(bits)
    return sum(int(bits[i].item()) << (n_bits - 1 - i) for i in range(n_bits))


# ============================================================
# Task Generators
# ============================================================

def random_train_test_split(max_val: int, train_ratio: float, seed: int = 42) -> tuple[list[int], list[int]]:
    """
    Create random train/test split.

    IMPORTANT: Random sampling ensures all bit patterns are covered in training,
    unlike sequential sampling which may miss high-bit patterns.
    """
    random.seed(seed)
    all_indices = list(range(max_val))
    random.shuffle(all_indices)
    train_size = int(max_val * train_ratio)
    return all_indices[:train_size], all_indices[train_size:]


def generate_successor_task(n_bits: int, train_ratio: float = 0.4, random_split: bool = True):
    """
    Successor task: output = input + 1 (mod 2^n_bits)

    Train on subset, test on all to measure generalization.
    """
    max_val = 2 ** n_bits

    if random_split:
        train_indices, test_indices = random_train_test_split(max_val, train_ratio)
    else:
        train_size = int(max_val * train_ratio)
        train_indices = list(range(train_size))
        test_indices = list(range(train_size, max_val))

    train_data = []
    for i in train_indices:
        inp = int_to_bits(i, n_bits)
        out = int_to_bits((i + 1) % max_val, n_bits)
        train_data.append((inp, out))

    test_data = []
    for i in test_indices:
        inp = int_to_bits(i, n_bits)
        out = int_to_bits((i + 1) % max_val, n_bits)
        test_data.append((inp, out))

    return train_data, test_data


def generate_copy_task(n_bits: int, train_ratio: float = 0.4, random_split: bool = True):
    """
    Copy task: output = input (identity)

    Baseline task - should be easy for all strategies.
    """
    max_val = 2 ** n_bits

    if random_split:
        train_indices, test_indices = random_train_test_split(max_val, train_ratio)
    else:
        train_size = int(max_val * train_ratio)
        train_indices = list(range(train_size))
        test_indices = list(range(train_size, max_val))

    train_data = []
    for i in train_indices:
        bits = int_to_bits(i, n_bits)
        train_data.append((bits.clone(), bits.clone()))

    test_data = []
    for i in test_indices:
        bits = int_to_bits(i, n_bits)
        test_data.append((bits.clone(), bits.clone()))

    return train_data, test_data


def generate_complement_task(n_bits: int, train_ratio: float = 0.4, random_split: bool = True):
    """
    Complement task: output = bitwise NOT of input

    Tests bit-level operations. With LOCAL context (window=0), each bit
    only needs to learn NOT (2 patterns), achieving 100% generalization.
    """
    max_val = 2 ** n_bits

    if random_split:
        train_indices, test_indices = random_train_test_split(max_val, train_ratio)
    else:
        train_size = int(max_val * train_ratio)
        train_indices = list(range(train_size))
        test_indices = list(range(train_size, max_val))

    train_data = []
    for i in train_indices:
        inp = int_to_bits(i, n_bits)
        out = 1 - inp  # Bitwise complement
        train_data.append((inp, out.to(uint8)))

    test_data = []
    for i in test_indices:
        inp = int_to_bits(i, n_bits)
        out = 1 - inp
        test_data.append((inp, out.to(uint8)))

    return train_data, test_data


def generate_parity_task(n_bits: int, train_ratio: float = 0.4, random_split: bool = True):
    """
    Parity task: output bit 0 = XOR of all input bits

    Classic hard task for neural networks.
    Output is n_bits with only the last bit being the parity.

    NOTE: Feedforward mappers struggle with this (global dependency).
    For 100% generalization, use recurrent approach that learns XOR (4 patterns).
    """
    max_val = 2 ** n_bits

    def compute_parity(n: int) -> int:
        parity = 0
        while n:
            parity ^= (n & 1)
            n >>= 1
        return parity

    if random_split:
        train_indices, test_indices = random_train_test_split(max_val, train_ratio)
    else:
        train_size = int(max_val * train_ratio)
        train_indices = list(range(train_size))
        test_indices = list(range(train_size, max_val))

    train_data = []
    for i in train_indices:
        inp = int_to_bits(i, n_bits)
        parity = compute_parity(i)
        # Output: copy input but set last bit to parity
        out = inp.clone()
        out[-1] = parity
        train_data.append((inp, out))

    test_data = []
    for i in test_indices:
        inp = int_to_bits(i, n_bits)
        parity = compute_parity(i)
        out = inp.clone()
        out[-1] = parity
        test_data.append((inp, out))

    return train_data, test_data


def generate_shift_left_task(n_bits: int, train_ratio: float = 0.4, random_split: bool = True):
    """
    Shift left task: output = input << 1 (with wraparound)

    Tests position-based transformations.
    NOTE: All strategies struggle with this because it requires cross-position
    learning (output[i] depends on input[i+1]).
    """
    max_val = 2 ** n_bits

    if random_split:
        train_indices, test_indices = random_train_test_split(max_val, train_ratio)
    else:
        train_size = int(max_val * train_ratio)
        train_indices = list(range(train_size))
        test_indices = list(range(train_size, max_val))

    train_data = []
    for i in train_indices:
        inp = int_to_bits(i, n_bits)
        # Shift left with wraparound
        shifted = ((i << 1) | (i >> (n_bits - 1))) & (max_val - 1)
        out = int_to_bits(shifted, n_bits)
        train_data.append((inp, out))

    test_data = []
    for i in test_indices:
        inp = int_to_bits(i, n_bits)
        shifted = ((i << 1) | (i >> (n_bits - 1))) & (max_val - 1)
        out = int_to_bits(shifted, n_bits)
        test_data.append((inp, out))

    return train_data, test_data


# ============================================================
# Benchmark Runner
# ============================================================

def run_strategy_benchmark(
    strategy: MapperStrategy,
    train_data: list[tuple[tensor, tensor]],
    test_data: list[tuple[tensor, tensor]],
    n_bits: int,
    max_epochs: int = 10,
    **strategy_kwargs,
) -> BenchmarkResult:
    """
    Run a single strategy benchmark.

    Args:
        strategy: The mapper strategy to test
        train_data: List of (input, target) tensor pairs for training
        test_data: List of (input, target) tensor pairs for testing
        n_bits: Number of bits per example
        max_epochs: Maximum training epochs
        strategy_kwargs: Additional arguments for the strategy

    Returns:
        BenchmarkResult with accuracy and timing information
    """
    # Create mapper
    try:
        mapper = MapperFactory.create(
            strategy=strategy,
            n_bits=n_bits,
            rng=42,
            **strategy_kwargs,
        )
    except Exception as e:
        return BenchmarkResult(
            strategy=strategy.name,
            task="",
            n_bits=n_bits,
            train_examples=len(train_data),
            test_examples=len(test_data),
            train_accuracy=0.0,
            test_accuracy=0.0,
            training_time_ms=0.0,
            epochs=0,
        )

    # Training
    start_time = perf_counter()

    for epoch in range(max_epochs):
        updates = 0
        for inp, tgt in train_data:
            if hasattr(mapper, 'train_mapping'):
                updates += mapper.train_mapping(inp, tgt)
            elif hasattr(mapper, 'commit'):
                current = mapper(inp.unsqueeze(0)).squeeze()
                if not (current == tgt).all():
                    mapper.commit(inp.unsqueeze(0), tgt.unsqueeze(0))
                    updates += 1

        # Check if converged
        if updates == 0:
            break

    training_time = (perf_counter() - start_time) * 1000

    # Evaluate on training data
    train_correct = 0
    for inp, tgt in train_data:
        result = mapper(inp)
        if (result.squeeze() == tgt.squeeze()).all():
            train_correct += 1
    train_accuracy = 100 * train_correct / len(train_data) if train_data else 0

    # Evaluate on test data (generalization)
    test_correct = 0
    for inp, tgt in test_data:
        result = mapper(inp)
        if (result.squeeze() == tgt.squeeze()).all():
            test_correct += 1
    test_accuracy = 100 * test_correct / len(test_data) if test_data else 0

    return BenchmarkResult(
        strategy=strategy.name,
        task="",
        n_bits=n_bits,
        train_examples=len(train_data),
        test_examples=len(test_data),
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
        training_time_ms=training_time,
        epochs=epoch + 1,
    )


def run_full_benchmark(
    tasks: list[BenchmarkTask] | None = None,
    strategies: list[MapperStrategy] | None = None,
    verbose: bool = True,
) -> list[BenchmarkResult]:
    """
    Run full benchmark suite.

    Args:
        tasks: List of benchmark tasks (default: all tasks)
        strategies: List of strategies to test (default: all)
        verbose: Print progress

    Returns:
        List of all benchmark results
    """
    if tasks is None:
        tasks = [
            BenchmarkTask("successor", "x → x+1", lambda n: generate_successor_task(n), n_bits=6),
            BenchmarkTask("copy", "x → x", lambda n: generate_copy_task(n), n_bits=6),
            BenchmarkTask("complement", "x → ~x", lambda n: generate_complement_task(n), n_bits=6),
            BenchmarkTask("parity", "x → parity(x)", lambda n: generate_parity_task(n), n_bits=6),
            BenchmarkTask("shift_left", "x → x<<1", lambda n: generate_shift_left_task(n), n_bits=6),
        ]

    if strategies is None:
        strategies = [
            MapperStrategy.DIRECT,
            MapperStrategy.BIT_LEVEL,
            MapperStrategy.COMPOSITIONAL,
            MapperStrategy.HASH,
            MapperStrategy.RESIDUAL,
        ]

    all_results = []

    for task in tasks:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Task: {task.name} ({task.description})")
            print(f"{'='*60}")

        # Generate data
        train_data, test_data = task.generator(task.n_bits)

        if verbose:
            print(f"  Train: {len(train_data)} examples, Test: {len(test_data)} examples")
            print()

        for strategy in strategies:
            # Get strategy-specific kwargs
            kwargs = {}
            if strategy == MapperStrategy.COMPOSITIONAL:
                # Need divisible n_bits
                if task.n_bits % 2 == 0:
                    kwargs['n_groups'] = 2
                else:
                    kwargs['n_groups'] = task.n_bits
            elif strategy == MapperStrategy.HASH:
                kwargs['hash_bits'] = max(4, task.n_bits - 2)

            result = run_strategy_benchmark(
                strategy=strategy,
                train_data=train_data,
                test_data=test_data,
                n_bits=task.n_bits,
                **kwargs,
            )
            result.task = task.name
            all_results.append(result)

            if verbose:
                gen_indicator = "★" if result.test_accuracy > 50 else " "
                print(f"  {strategy.name:15s}: train={result.train_accuracy:5.1f}% "
                      f"test={result.test_accuracy:5.1f}% {gen_indicator} "
                      f"({result.training_time_ms:.1f}ms, {result.epochs} epochs)")

    return all_results


def print_summary_table(results: list[BenchmarkResult]) -> None:
    """Print a summary table of benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    # Group by task
    tasks = sorted(set(r.task for r in results))
    strategies = sorted(set(r.strategy for r in results))

    # Header
    header = f"{'Task':<15}"
    for s in strategies:
        header += f" {s[:8]:>8}"
    print(header)
    print("-" * 80)

    # Rows (test accuracy)
    for task in tasks:
        row = f"{task:<15}"
        for strategy in strategies:
            matching = [r for r in results if r.task == task and r.strategy == strategy]
            if matching:
                acc = matching[0].test_accuracy
                marker = "★" if acc > 50 else " "
                row += f" {acc:6.1f}%{marker}"
            else:
                row += "     N/A "
        print(row)

    print("-" * 80)
    print("★ = Generalizes (>50% on unseen examples)")

    # Best strategy per task
    print("\nBest strategies per task:")
    for task in tasks:
        task_results = [r for r in results if r.task == task]
        if task_results:
            best = max(task_results, key=lambda r: r.test_accuracy)
            print(f"  {task}: {best.strategy} ({best.test_accuracy:.1f}% test accuracy)")


def print_strategy_analysis(results: list[BenchmarkResult]) -> None:
    """Print analysis of each strategy's strengths."""
    print("\n" + "=" * 80)
    print("STRATEGY ANALYSIS")
    print("=" * 80)

    strategies = sorted(set(r.strategy for r in results))

    for strategy in strategies:
        strategy_results = [r for r in results if r.strategy == strategy]

        avg_train = sum(r.train_accuracy for r in strategy_results) / len(strategy_results)
        avg_test = sum(r.test_accuracy for r in strategy_results) / len(strategy_results)
        avg_time = sum(r.training_time_ms for r in strategy_results) / len(strategy_results)

        generalizes = [r.task for r in strategy_results if r.test_accuracy > 50]

        print(f"\n{strategy}:")
        print(f"  Average train accuracy: {avg_train:.1f}%")
        print(f"  Average test accuracy:  {avg_test:.1f}% (generalization)")
        print(f"  Average training time:  {avg_time:.1f}ms")
        print(f"  Generalizes on: {', '.join(generalizes) if generalizes else 'none'}")


# ============================================================
# Compositional Benchmarks
# ============================================================

@dataclass
class CompositionResult:
    """Results from a compositional benchmark."""
    primitive_a: str
    primitive_b: str
    composition: str
    a_train_acc: float
    a_test_acc: float
    b_train_acc: float
    b_test_acc: float
    composed_train_acc: float
    composed_test_acc: float  # Key metric: does composition generalize?


def run_compositional_benchmark(
    n_bits: int = 6,
    train_ratio: float = 0.4,
    strategy: MapperStrategy = MapperStrategy.BIT_LEVEL,
    verbose: bool = True,
) -> list[CompositionResult]:
    """
    Test compositional generalization: learn A, learn B, compose A∘B.

    Key insight: If primitives generalize independently, does their
    composition also generalize?

    Tests:
    1. COPY∘INCREMENT: Copy then add 1 (= INCREMENT)
    2. INCREMENT∘COPY: Add 1 then copy (= INCREMENT)
    3. COMPLEMENT∘COMPLEMENT: NOT then NOT (= IDENTITY)
    4. INCREMENT∘INCREMENT: Add 1 twice (= ADD 2)

    Args:
        n_bits: Number of bits per value
        train_ratio: Fraction of data for training
        strategy: Generalization strategy to use (BIT_LEVEL, COMPOSITIONAL, etc.)
        verbose: Print progress

    Returns:
        List of CompositionResult with accuracy metrics
    """
    from wnn.ram.factories import MapperFactory

    if verbose:
        print("\n" + "=" * 60)
        print("COMPOSITIONAL GENERALIZATION BENCHMARK")
        print("=" * 60)
        print(f"""
Strategy: Learn primitives A and B separately, then compose A∘B.
Tests whether RAM networks support modular, reusable computation.
Generalization strategy: {strategy.name}
""")

    max_val = 2 ** n_bits

    # Split data
    train_indices, test_indices = random_train_test_split(max_val, train_ratio)

    # Define primitives
    def copy_fn(x: int) -> int:
        return x

    def increment_fn(x: int) -> int:
        return (x + 1) % max_val

    def complement_fn(x: int) -> int:
        return (~x) & (max_val - 1)

    def double_fn(x: int) -> int:
        return (2 * x) % max_val

    primitives = {
        'COPY': copy_fn,
        'INCREMENT': increment_fn,
        'COMPLEMENT': complement_fn,
        'DOUBLE': double_fn,
    }

    # Compositions to test
    compositions = [
        ('COPY', 'INCREMENT', 'copy_then_inc'),
        ('INCREMENT', 'COPY', 'inc_then_copy'),
        ('COMPLEMENT', 'COMPLEMENT', 'not_not'),
        ('INCREMENT', 'INCREMENT', 'add_2'),
        ('DOUBLE', 'INCREMENT', 'double_then_inc'),
    ]

    results = []

    for prim_a_name, prim_b_name, comp_name in compositions:
        if verbose:
            print(f"\n--- {prim_a_name} ∘ {prim_b_name} ({comp_name}) ---")

        prim_a = primitives[prim_a_name]
        prim_b = primitives[prim_b_name]

        # Create layers for each primitive using the specified strategy
        kwargs = {}
        if strategy == MapperStrategy.COMPOSITIONAL:
            kwargs['n_groups'] = 2 if n_bits % 2 == 0 else n_bits

        layer_a = MapperFactory.create(
            strategy=strategy,
            n_bits=n_bits,
            rng=42,
            **kwargs,
        )
        layer_b = MapperFactory.create(
            strategy=strategy,
            n_bits=n_bits,
            rng=43,
            **kwargs,
        )

        # Train primitive A
        for i in train_indices:
            inp = int_to_bits(i, n_bits)
            out = int_to_bits(prim_a(i), n_bits)
            layer_a.train_mapping(inp, out)

        # Train primitive B
        for i in train_indices:
            inp = int_to_bits(i, n_bits)
            out = int_to_bits(prim_b(i), n_bits)
            layer_b.train_mapping(inp, out)

        # Evaluate primitives
        def eval_layer(layer, fn, indices):
            correct = 0
            for i in indices:
                inp = int_to_bits(i, n_bits)
                result = layer(inp).squeeze()
                expected = int_to_bits(fn(i), n_bits)
                if (result == expected).all():
                    correct += 1
            return 100 * correct / len(indices) if indices else 0

        a_train_acc = eval_layer(layer_a, prim_a, train_indices)
        a_test_acc = eval_layer(layer_a, prim_a, test_indices)
        b_train_acc = eval_layer(layer_b, prim_b, train_indices)
        b_test_acc = eval_layer(layer_b, prim_b, test_indices)

        if verbose:
            print(f"  {prim_a_name}: train={a_train_acc:.1f}%, test={a_test_acc:.1f}%")
            print(f"  {prim_b_name}: train={b_train_acc:.1f}%, test={b_test_acc:.1f}%")

        # Evaluate composition: A∘B (apply A, then B)
        def composed_fn(x: int) -> int:
            return prim_b(prim_a(x))

        def eval_composition(indices):
            correct = 0
            for i in indices:
                # Apply A
                inp = int_to_bits(i, n_bits)
                intermediate = layer_a(inp)

                # Apply B to A's output
                final = layer_b(intermediate).squeeze()

                # Check against ground truth
                expected = int_to_bits(composed_fn(i), n_bits)
                if (final == expected).all():
                    correct += 1
            return 100 * correct / len(indices) if indices else 0

        composed_train_acc = eval_composition(train_indices)
        composed_test_acc = eval_composition(test_indices)

        if verbose:
            gen_mark = "★" if composed_test_acc > 50 else " "
            print(f"  {comp_name}: train={composed_train_acc:.1f}%, test={composed_test_acc:.1f}% {gen_mark}")

        results.append(CompositionResult(
            primitive_a=prim_a_name,
            primitive_b=prim_b_name,
            composition=comp_name,
            a_train_acc=a_train_acc,
            a_test_acc=a_test_acc,
            b_train_acc=b_train_acc,
            b_test_acc=b_test_acc,
            composed_train_acc=composed_train_acc,
            composed_test_acc=composed_test_acc,
        ))

    if verbose:
        print("\n" + "-" * 60)
        print("COMPOSITIONAL GENERALIZATION SUMMARY")
        print("-" * 60)
        print(f"{'Composition':<25} {'A test':>8} {'B test':>8} {'A∘B test':>10}")
        print("-" * 60)
        for r in results:
            gen_mark = "★" if r.composed_test_acc > 50 else " "
            print(f"{r.composition:<25} {r.a_test_acc:>7.1f}% {r.b_test_acc:>7.1f}% "
                  f"{r.composed_test_acc:>9.1f}%{gen_mark}")

        # Insight
        print("\n★ Key Insight: Composition generalization depends on BOTH primitives")
        print("  generalizing. If A or B fails to generalize, A∘B also fails.")

    return results


# ============================================================
# Recurrent Benchmarks
# ============================================================

def run_recurrent_parity_benchmark(
    train_bits: int = 6,
    test_bits_list: list[int] | None = None,
    verbose: bool = True,
) -> dict:
    """
    Demonstrate 100% parity generalization using recurrent XOR learning.

    Key insight: Instead of memorizing 2^n patterns, we learn XOR (4 patterns)
    and apply it recurrently. This generalizes to ANY sequence length!

    Args:
        train_bits: Number of bits to train on
        test_bits_list: List of bit lengths to test generalization
        verbose: Print progress

    Returns:
        Dict with accuracy per test length
    """
    from wnn.ram.core import RAMLayer

    if test_bits_list is None:
        test_bits_list = [6, 8, 10, 12]

    if verbose:
        print("\n" + "=" * 60)
        print("RECURRENT PARITY BENCHMARK")
        print("=" * 60)
        print("""
Strategy: Learn XOR function (4 patterns), apply recurrently.
This achieves 100% generalization to ANY sequence length!
""")

    # Create a RAM that learns XOR: (prev_state, input) -> new_state
    xor_ram = RAMLayer(
        total_input_bits=2,    # [prev_state, input]
        num_neurons=1,         # 1 output
        n_bits_per_neuron=2,   # See all 2 bits
        rng=42,
    )

    # Train XOR truth table
    xor_patterns = [
        (tensor([[0, 0]], dtype=uint8), tensor([[0]], dtype=uint8)),  # 0 XOR 0 = 0
        (tensor([[0, 1]], dtype=uint8), tensor([[1]], dtype=uint8)),  # 0 XOR 1 = 1
        (tensor([[1, 0]], dtype=uint8), tensor([[1]], dtype=uint8)),  # 1 XOR 0 = 1
        (tensor([[1, 1]], dtype=uint8), tensor([[0]], dtype=uint8)),  # 1 XOR 1 = 0
    ]

    for inp, tgt in xor_patterns:
        xor_ram.commit(inp, tgt)

    if verbose:
        print("XOR function learned (4 patterns)")

    # Function to compute parity using the XOR RAM
    def compute_parity_with_xor(n: int, n_bits: int) -> int:
        state = 0
        for b in range(n_bits):
            bit = (n >> b) & 1
            inp = tensor([[state, bit]], dtype=uint8)
            state = xor_ram(inp)[0, 0].item()
        return state

    def compute_parity_truth(n: int, n_bits: int) -> int:
        parity = 0
        for b in range(n_bits):
            parity ^= (n >> b) & 1
        return parity

    # Test on various bit lengths
    results = {}
    for n_bits in test_bits_list:
        max_val = 2 ** n_bits
        correct = 0
        for i in range(max_val):
            expected = compute_parity_truth(i, n_bits)
            result = compute_parity_with_xor(i, n_bits)
            if result == expected:
                correct += 1
        accuracy = 100 * correct / max_val
        results[n_bits] = accuracy
        if verbose:
            print(f"  {n_bits}-bit parity: {accuracy:.1f}% ({correct}/{max_val})")

    if verbose:
        print("\n★ Recurrent XOR achieves 100% generalization to any length!")

    return results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 80)
    print("RAM Network Systematic Benchmarks")
    print("=" * 80)
    print("""
This benchmark suite tests generalization strategies on various tasks.

Key metrics:
- Train accuracy: How well the strategy learns training examples
- Test accuracy: How well it generalizes to unseen examples (generalization!)
- Training time: Computational efficiency

Sampling: RANDOM (ensures all bit patterns covered in training)

Tasks:
- successor: x → x+1 (requires learning increment pattern)
- copy: x → x (baseline identity)
- complement: x → ~x (bitwise NOT)
- parity: x → parity(x) (classic hard problem)
- shift_left: x → x<<1 (position transformation)
""")

    # Run feedforward benchmarks
    results = run_full_benchmark(verbose=True)

    # Print summary
    print_summary_table(results)
    print_strategy_analysis(results)

    # Run compositional benchmark with multiple strategies
    print("\n" + "=" * 80)
    print("COMPOSITIONAL BENCHMARKS")
    print("=" * 80)
    for strategy in [MapperStrategy.BIT_LEVEL, MapperStrategy.COMPOSITIONAL]:
        run_compositional_benchmark(strategy=strategy, verbose=True)

    # Run recurrent parity benchmark
    run_recurrent_parity_benchmark(verbose=True)

    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)
