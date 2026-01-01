"""
Training Features Test

Tests for enhanced training capabilities:
- CurriculumTrainer: Progressive difficulty scheduling
- MultiTaskTrainer: Train on multiple tasks simultaneously

Key insight: These techniques improve convergence and generalization
by structuring the learning process.
"""

from datetime import datetime

from torch import tensor, uint8, zeros

from wnn.ram.core import (
    CurriculumTrainer,
    CurriculumSchedule,
    MultiTaskTrainer,
    Task,
    MixingStrategy,
    RAMTrainer,
    combined_difficulty,
    length_difficulty,
)
from wnn.ram.core.models import RAMSeq2Seq


# =============================================================================
# UTILITIES
# =============================================================================

def int_to_bits(n: int, n_bits: int) -> tensor:
    """Convert integer to bit tensor (MSB first)."""
    return tensor([(n >> i) & 1 for i in range(n_bits - 1, -1, -1)], dtype=uint8)


def create_copy_dataset(n_bits: int = 4, seq_lengths: list[int] = None):
    """Create copy task dataset with varying sequence lengths."""
    if seq_lengths is None:
        seq_lengths = [2, 3, 4, 5, 6]

    dataset = []
    for length in seq_lengths:
        for i in range(min(8, 2 ** n_bits)):
            tokens = [int_to_bits(i, n_bits) for _ in range(length)]
            dataset.append((tokens, tokens))  # Copy: input == output
    return dataset


def create_increment_dataset(n_bits: int = 4, count: int = 16):
    """Create increment task dataset."""
    max_val = 2 ** n_bits - 1
    dataset = []
    for i in range(count):
        val = i % max_val
        inp = [int_to_bits(val, n_bits)]
        out = [int_to_bits((val + 1) % (max_val + 1), n_bits)]
        dataset.append((inp, out))
    return dataset


def create_parity_dataset(n_bits: int = 4, count: int = 32):
    """Create parity task dataset (output = XOR of all input bits)."""
    dataset = []
    for i in range(min(count, 2 ** n_bits)):
        inp = int_to_bits(i, n_bits)
        parity = sum(int(b) for b in inp) % 2
        out = int_to_bits(parity, 1)
        dataset.append(([inp], [out]))
    return dataset


# =============================================================================
# CURRICULUM LEARNING TEST
# =============================================================================

def test_curriculum_learning():
    """Test curriculum learning with progressive difficulty."""
    print(f"\n{'='*60}")
    print("Testing Curriculum Learning")
    print(f"{'='*60}")

    # Create dataset with varying difficulty (sequence lengths)
    dataset = create_copy_dataset(n_bits=4, seq_lengths=[2, 3, 4, 5, 6, 7, 8])
    print(f"Dataset: {len(dataset)} examples")

    # Create model
    model = RAMSeq2Seq(
        input_bits=4,
        hidden_bits=8,
        output_bits=4,
        num_layers=1,
        num_heads=2,
        use_residual=True,
        rng=42,
    )

    # Create trainer with curriculum
    trainer = RAMTrainer(model, verbose=False)
    curriculum = CurriculumTrainer(trainer, difficulty_metric=combined_difficulty)

    # Define schedule
    schedule = CurriculumSchedule(
        num_stages=3,
        epochs_per_stage=5,
        overlap=0.2,
        patience_per_stage=2,
    )

    print(f"\nSchedule: {schedule.num_stages} stages, "
          f"{schedule.epochs_per_stage} epochs each")

    # Train with curriculum
    history = curriculum.train(dataset, schedule=schedule, verbose=True)

    # Evaluate
    final_acc = history[-1].accuracy if history else 0
    print(f"\nFinal accuracy: {final_acc:.1f}%")

    return final_acc


# =============================================================================
# MULTI-TASK LEARNING TEST
# =============================================================================

def test_multi_task_learning():
    """Test multi-task learning with shared primitives."""
    print(f"\n{'='*60}")
    print("Testing Multi-Task Learning")
    print(f"{'='*60}")

    # Create tasks
    copy_data = create_copy_dataset(n_bits=4, seq_lengths=[2, 3, 4])
    increment_data = create_increment_dataset(n_bits=4, count=16)

    print(f"Copy task: {len(copy_data)} examples")
    print(f"Increment task: {len(increment_data)} examples")

    # Create model
    model = RAMSeq2Seq(
        input_bits=4,
        hidden_bits=8,
        output_bits=4,
        num_layers=1,
        num_heads=2,
        use_residual=True,
        rng=42,
    )

    # Create multi-task trainer
    trainer = RAMTrainer(model, verbose=False)
    mt = MultiTaskTrainer(trainer, mixing=MixingStrategy.INTERLEAVED)

    # Add tasks
    mt.add_task(Task(
        name="copy",
        dataset=copy_data,
        weight=1.0,
        shared_primitives=["position_encoding"],
    ))
    mt.add_task(Task(
        name="increment",
        dataset=increment_data,
        weight=1.5,  # Slightly higher weight for smaller task
        shared_primitives=["position_encoding", "bit_transform"],
    ))

    # Train
    history = mt.train(epochs=15, early_stop=True, verbose=True)

    # Evaluate
    print("\nFinal evaluation:")
    results = mt.evaluate(verbose=True)

    # Show shared primitive stats
    shared = mt.get_shared_primitive_stats()
    print(f"\nShared primitives: {shared}")

    return results


# =============================================================================
# MIXING STRATEGY COMPARISON
# =============================================================================

def test_mixing_strategies():
    """Compare different mixing strategies for multi-task learning."""
    print(f"\n{'='*60}")
    print("Comparing Mixing Strategies")
    print(f"{'='*60}")

    # Create small tasks for quick comparison
    copy_data = create_copy_dataset(n_bits=4, seq_lengths=[2, 3])
    increment_data = create_increment_dataset(n_bits=4, count=8)

    results = {}

    for strategy in [MixingStrategy.ROUND_ROBIN, MixingStrategy.INTERLEAVED]:
        print(f"\n--- {strategy.name} ---")

        # Fresh model for each strategy
        model = RAMSeq2Seq(
            input_bits=4,
            hidden_bits=8,
            output_bits=4,
            num_layers=1,
            num_heads=2,
            use_residual=True,
            rng=42,
        )

        trainer = RAMTrainer(model, verbose=False)
        mt = MultiTaskTrainer(trainer, mixing=strategy)

        mt.add_task(Task("copy", copy_data, weight=1.0))
        mt.add_task(Task("increment", increment_data, weight=1.0))

        history = mt.train(epochs=10, early_stop=True, verbose=False)

        # Get final accuracies
        eval_results = mt.evaluate(verbose=False)
        results[strategy.name] = eval_results

        print(f"  Copy: {eval_results['copy']:.1f}%")
        print(f"  Increment: {eval_results['increment']:.1f}%")

    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("Training Features Test")
    print(f"Started at: {datetime.now()}")
    print(f"{'='*60}")

    # Test curriculum learning
    curriculum_acc = test_curriculum_learning()

    # Test multi-task learning
    mt_results = test_multi_task_learning()

    # Compare mixing strategies
    mixing_results = test_mixing_strategies()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    print("\nCurriculum Learning:")
    print(f"  Final accuracy: {curriculum_acc:.1f}%")
    print("  Key: Progressive difficulty helps convergence")

    print("\nMulti-Task Learning:")
    for task, acc in mt_results.items():
        print(f"  {task}: {acc:.1f}%")
    print("  Key: Shared primitives transfer between tasks")

    print("\nMixing Strategies:")
    for strategy, results in mixing_results.items():
        avg_acc = sum(results.values()) / len(results)
        print(f"  {strategy}: avg {avg_acc:.1f}%")

    print(f"""
Key insights:
- Curriculum learning: Start easy, increase difficulty
- Multi-task learning: Share primitives across related tasks
- Mixing strategies: INTERLEAVED generally works best
""")
    print(f"Finished at: {datetime.now()}")
    print(f"{'='*60}")
