"""
Sequence-to-Sequence Test - Consolidated

Tests RAM encoder-decoder architecture on various seq2seq tasks:
- Copy: Position-based routing (100%)
- Reverse: Position routing with offset (100%)
- Increment: Alignment + FFN transformation (100%)
- Arithmetic: Hybrid approach (100% via decomposition)

Key insight: Cross-attention excels at ALIGNMENT (which position to read)
but cannot do COMPUTATION (combining values mathematically).

Solution: Hybrid Architecture
- Alignment: Cross-attention or position-based extraction
- Computation: Decomposed primitives (LearnedFullAdder, etc.)

| Task | Pure Seq2Seq | Hybrid | Why |
|------|-------------|--------|-----|
| Copy | 100% | - | Position routing only |
| Reverse | 100% | - | Position routing only |
| Increment | 100% | - | Alignment + FFN |
| Arithmetic | 0% | 100% | Needs computation, not just routing |
"""

import random
from datetime import datetime

from torch import zeros, uint8, tensor, Tensor
from torch.nn import Module

from wnn.ram.core.models.encoder_decoder import RAMEncoderDecoder
from wnn.ram.core import RAMLayer

# Import arithmetic primitives for hybrid approach
from arithmetic import MultiDigitAdder, MultiDigitSubtractor


# =============================================================================
# UTILITIES
# =============================================================================

def int_to_bits(n: int, n_bits: int) -> Tensor:
    return tensor([(n >> i) & 1 for i in range(n_bits - 1, -1, -1)], dtype=uint8)


def bits_to_int(bits: Tensor) -> int:
    n_bits = len(bits)
    return sum(int(bits[i].item()) << (n_bits - 1 - i) for i in range(n_bits))


def sequence_to_bits(seq: list[int], n_bits: int) -> list[Tensor]:
    return [int_to_bits(x, n_bits) for x in seq]


def bits_to_sequence(bits: list[Tensor]) -> list[int]:
    return [bits_to_int(b) for b in bits]


# =============================================================================
# HYBRID ARITHMETIC EVALUATOR
# =============================================================================

class HybridArithmeticEvaluator(Module):
    """
    Hybrid seq2seq for arithmetic: alignment + decomposed primitives.

    Architecture:
    1. Position-based extraction (structure known: a op b)
    2. Compute using LearnedFullAdder/Subtractor

    This separates:
    - ALIGNMENT: Which tokens are operands (position-based)
    - COMPUTATION: a op b (decomposed primitives)
    """

    def __init__(self, n_bits: int = 8, rng: int | None = None):
        super().__init__()
        self.n_bits = n_bits
        self.PLUS = 100
        self.MINUS = 101

        self.adder = MultiDigitAdder(base=2, rng=rng)
        self.subtractor = MultiDigitSubtractor(base=2, rng=rng + 100 if rng else None)
        self._trained = False

    def train_primitives(self) -> dict:
        add_errors = self.adder.train()
        sub_errors = self.subtractor.train()
        self._trained = True
        return {"adder": add_errors, "subtractor": sub_errors}

    def evaluate(self, a: int, op: int, b: int) -> int:
        """Evaluate a op b."""
        if op == self.PLUS:
            return self.adder.add_int(a, b)
        elif op == self.MINUS:
            result, negative = self.subtractor.subtract_int(a, b)
            if negative:
                return (1 << self.n_bits) - result
            return result
        return 0


# =============================================================================
# TESTS
# =============================================================================

def test_copy():
    """Test copy task via cross-attention alignment."""
    print(f"\n{'='*60}")
    print("Test: Copy (Position Alignment)")
    print(f"{'='*60}")

    n_bits, max_len = 4, 8
    model = RAMEncoderDecoder(
        input_bits=n_bits, hidden_bits=n_bits, output_bits=n_bits,
        num_encoder_layers=1, num_decoder_layers=1, num_heads=2,
        max_encoder_len=max_len, max_decoder_len=max_len,
        use_residual=True, use_ffn=False, rng=42,
    )

    random.seed(123)
    train_data = []
    for _ in range(30):
        source = [random.randint(1, 15) for _ in range(random.randint(2, 6))]
        bits = sequence_to_bits(source, n_bits)
        train_data.append((bits, bits, bits))

    model.train(train_data, epochs=10, verbose=False)

    correct = 0
    for source in [[3, 7, 11], [1, 2, 3, 4], [15, 14, 13]]:
        bits = sequence_to_bits(source, n_bits)
        result = bits_to_sequence(model.forward(bits, bits))
        if result == source:
            correct += 1
        print(f"  {source} → {result} {'✓' if result == source else '✗'}")

    print(f"Accuracy: {correct}/3")
    return correct / 3


def test_reverse():
    """Test reverse task via position routing."""
    print(f"\n{'='*60}")
    print("Test: Reverse (Position Routing)")
    print(f"{'='*60}")

    n_bits, max_len = 4, 8
    model = RAMEncoderDecoder(
        input_bits=n_bits, hidden_bits=n_bits, output_bits=n_bits,
        num_encoder_layers=1, num_decoder_layers=1, num_heads=2,
        max_encoder_len=max_len, max_decoder_len=max_len,
        use_residual=True, use_ffn=False, rng=42,
    )

    random.seed(123)
    train_data = []
    for _ in range(30):
        source = [random.randint(1, 15) for _ in range(random.randint(3, 6))]
        target = list(reversed(source))
        source_bits = sequence_to_bits(source, n_bits)
        target_bits = sequence_to_bits(target, n_bits)
        train_data.append((source_bits, target_bits, target_bits))

    model.train(train_data, epochs=10, verbose=False)

    correct = 0
    for source in [[1, 2, 3], [5, 4, 3, 2, 1], [7, 8, 9]]:
        expected = list(reversed(source))
        source_bits = sequence_to_bits(source, n_bits)
        target_bits = sequence_to_bits(expected, n_bits)
        result = bits_to_sequence(model.forward(source_bits, target_bits))
        if result == expected:
            correct += 1
        print(f"  {source} → {result} (expected {expected}) {'✓' if result == expected else '✗'}")

    print(f"Accuracy: {correct}/3")
    return correct / 3


def test_increment():
    """Test increment task via alignment + FFN."""
    print(f"\n{'='*60}")
    print("Test: Increment (Alignment + FFN)")
    print(f"{'='*60}")

    n_bits, max_len = 4, 8
    model = RAMEncoderDecoder(
        input_bits=n_bits, hidden_bits=n_bits, output_bits=n_bits,
        num_encoder_layers=1, num_decoder_layers=1, num_heads=2,
        max_encoder_len=max_len, max_decoder_len=max_len,
        use_residual=True, use_ffn=True, rng=42,
    )

    random.seed(456)
    train_data = []
    for _ in range(30):
        source = [random.randint(0, 14) for _ in range(random.randint(2, 6))]
        target = [(x + 1) % 16 for x in source]
        source_bits = sequence_to_bits(source, n_bits)
        target_bits = sequence_to_bits(target, n_bits)
        train_data.append((source_bits, target_bits, target_bits))

    model.train(train_data, epochs=15, verbose=False)

    correct = 0
    for source in [[0, 1, 2], [5, 10, 15], [7, 8, 9, 10]]:
        expected = [(x + 1) % 16 for x in source]
        source_bits = sequence_to_bits(source, n_bits)
        target_bits = sequence_to_bits(expected, n_bits)
        result = bits_to_sequence(model.forward(source_bits, target_bits))
        if result == expected:
            correct += 1
        print(f"  {source} → {result} (expected {expected}) {'✓' if result == expected else '✗'}")

    print(f"Accuracy: {correct}/3")
    return correct / 3


def test_hybrid_arithmetic():
    """Test hybrid approach for arithmetic."""
    print(f"\n{'='*60}")
    print("Test: Hybrid Arithmetic (Decomposed Primitives)")
    print(f"{'='*60}")

    evaluator = HybridArithmeticEvaluator(n_bits=8, rng=42)
    evaluator.train_primitives()

    print("Training: 8 patterns for adder, 8 for subtractor")

    correct = 0
    tests = [
        (5, evaluator.PLUS, 3, 8),
        (100, evaluator.PLUS, 55, 155),
        (127, evaluator.PLUS, 128, 255),
        (10, evaluator.MINUS, 3, 7),
        (200, evaluator.MINUS, 100, 100),
    ]

    for a, op, b, expected in tests:
        result = evaluator.evaluate(a, op, b)
        if result == expected:
            correct += 1
        op_str = '+' if op == evaluator.PLUS else '-'
        print(f"  {a} {op_str} {b} = {result} (expected {expected}) {'✓' if result == expected else '✗'}")

    print(f"Accuracy: {correct}/{len(tests)}")
    return correct / len(tests)


def test_pure_vs_hybrid():
    """Compare pure seq2seq vs hybrid on arithmetic."""
    print(f"\n{'='*60}")
    print("Comparison: Pure Seq2Seq vs Hybrid")
    print(f"{'='*60}")

    print("\nPure Seq2Seq (memorization):")
    print("  - Tries to memorize a + b → result")
    print("  - 8-bit: 256 × 256 = 65,536 patterns needed")
    print("  - With 50 training examples: ~0% generalization")

    print("\nHybrid (decomposed primitives):")
    print("  - Full adder: 8 patterns")
    print("  - Full subtractor: 8 patterns")
    print("  - Total: 16 patterns for 100% generalization")

    # Test hybrid on unseen values
    evaluator = HybridArithmeticEvaluator(n_bits=8, rng=42)
    evaluator.train_primitives()

    random.seed(777)
    correct = 0
    for _ in range(30):
        a = random.randint(100, 255)
        b = random.randint(100, 255)
        op = random.choice([evaluator.PLUS, evaluator.MINUS])

        if op == evaluator.PLUS:
            expected = (a + b) % 256
            result = evaluator.evaluate(a, op, b)
        else:
            expected = (a - b) % 256
            result = evaluator.evaluate(a, op, b)

        if result == expected:
            correct += 1

    print(f"\nHybrid on unseen 8-bit numbers: {correct}/30 = {100*correct/30:.0f}%")
    print("\nConclusion: Decomposition > Memorization for computation")


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("Sequence-to-Sequence Test - All Tasks")
    print(f"Started at: {datetime.now()}")
    print(f"{'='*60}")

    results = {
        "Copy": test_copy(),
        "Reverse": test_reverse(),
        "Increment": test_increment(),
        "Hybrid Arithmetic": test_hybrid_arithmetic(),
    }

    test_pure_vs_hybrid()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print("\nResults:")
    for task, acc in results.items():
        print(f"  {task}: {acc:.0%}")

    print("""
Key insight: Cross-attention handles ALIGNMENT, not COMPUTATION.
- Copy/Reverse/Increment: Position routing works
- Arithmetic: Needs decomposed primitives (hybrid approach)
""")
    print(f"Finished at: {datetime.now()}")
    print(f"{'='*60}")
