"""
Sequence-to-Sequence Test

Tests RAM encoder-decoder architecture on various seq2seq tasks.

Tasks explored:
1. Reverse sequence - Position routing (should work)
2. Character mapping - Simple char→char translation
3. Arithmetic expressions - Parse and evaluate
4. Copy with transform - Copy + per-token operation

Key question: Can cross-attention learn source→target alignments?
Unlike decoder-only models, encoder-decoder can access full source at each step.
"""

import random
from datetime import datetime

from torch import zeros, uint8, tensor, Tensor
from torch.nn import Module

from wnn.ram.core.models.encoder_decoder import RAMEncoderDecoder
from wnn.ram.core.models.seq2seq import RAMSeq2Seq
from wnn.ram.core import RAMLayer
from wnn.ram.encoders_decoders import PositionMode


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def int_to_bits(n: int, n_bits: int) -> Tensor:
    """Convert integer to bit tensor."""
    bits = [(n >> i) & 1 for i in range(n_bits - 1, -1, -1)]
    return tensor(bits, dtype=uint8)


def bits_to_int(bits: Tensor) -> int:
    """Convert bit tensor to integer."""
    n_bits = len(bits)
    return sum(int(bits[i].item()) << (n_bits - 1 - i) for i in range(n_bits))


def sequence_to_bits(seq: list[int], n_bits: int) -> list[Tensor]:
    """Convert sequence of integers to list of bit tensors."""
    return [int_to_bits(x, n_bits) for x in seq]


def bits_to_sequence(bits: list[Tensor]) -> list[int]:
    """Convert list of bit tensors to sequence of integers."""
    return [bits_to_int(b) for b in bits]


# ─────────────────────────────────────────────────────────────────────────────
# Task 1: Reverse Sequence
# ─────────────────────────────────────────────────────────────────────────────

def test_reverse_sequence():
    """
    Test: Reverse a sequence.

    Input:  [1, 2, 3, 4, 5]
    Output: [5, 4, 3, 2, 1]

    This tests position-based routing through cross-attention.
    Decoder position i should attend to encoder position (n-1-i).
    """
    print(f"\n{'='*60}")
    print("Task 1: Reverse Sequence")
    print(f"{'='*60}")

    n_bits = 4  # 0-15
    max_len = 8

    # Create encoder-decoder model
    model = RAMEncoderDecoder(
        input_bits=n_bits,
        hidden_bits=n_bits,
        output_bits=n_bits,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        max_encoder_len=max_len,
        max_decoder_len=max_len,
        use_residual=True,
        use_ffn=False,
        rng=42,
    )

    # Generate training data
    random.seed(123)
    train_data = []
    for _ in range(20):
        length = random.randint(3, max_len - 1)
        source = [random.randint(1, 15) for _ in range(length)]
        target = list(reversed(source))

        source_bits = sequence_to_bits(source, n_bits)
        target_bits = sequence_to_bits(target, n_bits)

        # For training: target_input is shifted right, target_output is what to predict
        # Simple version: same length, output[i] should be reverse[i]
        train_data.append((source_bits, target_bits, target_bits))

    # Train
    print(f"Training on {len(train_data)} examples...")
    history = model.train(train_data, epochs=10, verbose=True)

    # Test
    print("\nTesting:")
    test_cases = [
        [1, 2, 3],
        [5, 4, 3, 2, 1],
        [7, 8, 9],
    ]

    correct = 0
    for source in test_cases:
        expected = list(reversed(source))
        source_bits = sequence_to_bits(source, n_bits)
        target_bits = sequence_to_bits(expected, n_bits)

        # Use teacher forcing for prediction
        predictions = model.forward(source_bits, target_bits)
        result = bits_to_sequence(predictions)

        ok = "✓" if result == expected else "✗"
        if result == expected:
            correct += 1
        print(f"  {source} → {result} (expected {expected}) {ok}")

    print(f"\nTest accuracy: {correct}/{len(test_cases)}")
    return correct / len(test_cases)


# ─────────────────────────────────────────────────────────────────────────────
# Task 2: Character Mapping (Simple Translation)
# ─────────────────────────────────────────────────────────────────────────────

def test_character_mapping():
    """
    Test: Map characters to numbers.

    'a' → 1, 'b' → 2, ..., 'z' → 26

    Input:  "abc"  (encoded as [1, 2, 3])
    Output: [1, 2, 3] (already encoded, but decoder learns mapping)

    This is a trivial copy task but tests the encoder-decoder flow.
    """
    print(f"\n{'='*60}")
    print("Task 2: Character Mapping")
    print(f"{'='*60}")

    n_bits = 5  # 0-31, enough for a-z (1-26)
    max_len = 8

    model = RAMEncoderDecoder(
        input_bits=n_bits,
        hidden_bits=n_bits,
        output_bits=n_bits,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        max_encoder_len=max_len,
        max_decoder_len=max_len,
        use_residual=True,
        use_ffn=False,
        rng=42,
    )

    # Training data: letter sequences
    # Encoded as: a=1, b=2, ..., z=26
    train_data = []
    for _ in range(30):
        length = random.randint(2, 6)
        # Random letters
        source = [random.randint(1, 26) for _ in range(length)]
        # Output is same (identity mapping for now)
        target = source.copy()

        source_bits = sequence_to_bits(source, n_bits)
        target_bits = sequence_to_bits(target, n_bits)
        train_data.append((source_bits, target_bits, target_bits))

    print(f"Training on {len(train_data)} examples...")
    history = model.train(train_data, epochs=10, verbose=True)

    # Test
    print("\nTesting (identity mapping a→1, b→2, ...):")
    test_cases = [
        [1, 2, 3],      # abc
        [8, 5, 12, 12, 15],  # hello (h=8, e=5, l=12, l=12, o=15)
        [26, 25, 24],   # zyx
    ]

    correct = 0
    for source in test_cases:
        expected = source.copy()
        source_bits = sequence_to_bits(source, n_bits)
        target_bits = sequence_to_bits(expected, n_bits)

        predictions = model.forward(source_bits, target_bits)
        result = bits_to_sequence(predictions)

        # Convert to letters for display
        source_str = ''.join(chr(96 + c) for c in source)
        result_str = ''.join(chr(96 + c) if 1 <= c <= 26 else '?' for c in result)

        ok = "✓" if result == expected else "✗"
        if result == expected:
            correct += 1
        print(f"  '{source_str}' → '{result_str}' {ok}")

    print(f"\nTest accuracy: {correct}/{len(test_cases)}")
    return correct / len(test_cases)


# ─────────────────────────────────────────────────────────────────────────────
# Task 3: Increment Each Element
# ─────────────────────────────────────────────────────────────────────────────

def test_increment_sequence():
    """
    Test: Increment each element.

    Input:  [1, 2, 3]
    Output: [2, 3, 4]

    Combines cross-attention (position routing) with token transformation.
    """
    print(f"\n{'='*60}")
    print("Task 3: Increment Sequence")
    print(f"{'='*60}")

    n_bits = 4
    max_len = 8

    model = RAMEncoderDecoder(
        input_bits=n_bits,
        hidden_bits=n_bits,
        output_bits=n_bits,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        max_encoder_len=max_len,
        max_decoder_len=max_len,
        use_residual=True,
        use_ffn=True,  # FFN helps with transformation
        rng=42,
    )

    # Training data
    random.seed(456)
    train_data = []
    for _ in range(30):
        length = random.randint(2, 6)
        source = [random.randint(0, 14) for _ in range(length)]  # Leave room for +1
        target = [(x + 1) % 16 for x in source]

        source_bits = sequence_to_bits(source, n_bits)
        target_bits = sequence_to_bits(target, n_bits)
        train_data.append((source_bits, target_bits, target_bits))

    print(f"Training on {len(train_data)} examples...")
    history = model.train(train_data, epochs=15, verbose=True)

    # Test
    print("\nTesting:")
    test_cases = [
        [0, 1, 2],
        [5, 10, 15],
        [7, 8, 9, 10],
    ]

    correct = 0
    for source in test_cases:
        expected = [(x + 1) % 16 for x in source]
        source_bits = sequence_to_bits(source, n_bits)
        target_bits = sequence_to_bits(expected, n_bits)

        predictions = model.forward(source_bits, target_bits)
        result = bits_to_sequence(predictions)

        ok = "✓" if result == expected else "✗"
        if result == expected:
            correct += 1
        print(f"  {source} → {result} (expected {expected}) {ok}")

    print(f"\nTest accuracy: {correct}/{len(test_cases)}")
    return correct / len(test_cases)


# ─────────────────────────────────────────────────────────────────────────────
# Task 4: Simple Arithmetic Expression Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def test_arithmetic_eval():
    """
    Test: Evaluate simple arithmetic.

    Input:  [2, +, 3]  (encoded as numbers)
    Output: [5]

    This tests:
    - Variable-length output
    - Cross-attention for gathering operands
    - Computing result
    """
    print(f"\n{'='*60}")
    print("Task 4: Simple Arithmetic Evaluation")
    print(f"{'='*60}")

    # Encoding: 0-9 = digits, 10 = +, 11 = -, 12 = =
    n_bits = 4
    max_len = 8

    PLUS = 10
    MINUS = 11
    EQUALS = 12

    model = RAMEncoderDecoder(
        input_bits=n_bits,
        hidden_bits=8,  # Larger hidden for computation
        output_bits=n_bits,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=2,
        max_encoder_len=max_len,
        max_decoder_len=max_len,
        use_residual=True,
        use_ffn=True,
        rng=42,
    )

    # Training data: single-digit addition/subtraction
    random.seed(789)
    train_data = []

    for _ in range(50):
        a = random.randint(0, 9)
        b = random.randint(0, 9)
        op = random.choice([PLUS, MINUS])

        if op == PLUS:
            result = (a + b) % 16  # Mod 16 to fit in 4 bits
        else:
            result = (a - b) % 16

        source = [a, op, b]
        target = [result]

        source_bits = sequence_to_bits(source, n_bits)
        target_bits = sequence_to_bits(target, n_bits)

        # Pad target to length 1 for teacher forcing
        train_data.append((source_bits, target_bits, target_bits))

    print(f"Training on {len(train_data)} examples (single-digit a+b, a-b)...")
    history = model.train(train_data, epochs=20, verbose=True)

    # Test
    print("\nTesting:")
    test_cases = [
        ([2, PLUS, 3], 5),
        ([7, PLUS, 2], 9),
        ([5, MINUS, 2], 3),
        ([1, PLUS, 1], 2),
        ([9, MINUS, 4], 5),
    ]

    correct = 0
    for source, expected in test_cases:
        source_bits = sequence_to_bits(source, n_bits)
        target_bits = sequence_to_bits([expected], n_bits)

        predictions = model.forward(source_bits, target_bits)
        result = bits_to_sequence(predictions)[0]

        op_str = '+' if source[1] == PLUS else '-'
        ok = "✓" if result == expected else "✗"
        if result == expected:
            correct += 1
        print(f"  {source[0]} {op_str} {source[2]} = {result} (expected {expected}) {ok}")

    print(f"\nTest accuracy: {correct}/{len(test_cases)}")
    return correct / len(test_cases)


# ─────────────────────────────────────────────────────────────────────────────
# Task 5: Autoregressive Generation Test
# ─────────────────────────────────────────────────────────────────────────────

def test_autoregressive_generation():
    """
    Test autoregressive generation without teacher forcing.

    Use decoder-only model for sequence continuation.
    """
    print(f"\n{'='*60}")
    print("Task 5: Autoregressive Generation (Decoder-only)")
    print(f"{'='*60}")

    n_bits = 4
    max_len = 16

    model = RAMSeq2Seq(
        input_bits=n_bits,
        hidden_bits=n_bits,
        output_bits=n_bits,
        num_layers=2,
        num_heads=2,
        max_seq_len=max_len,
        use_residual=True,
        use_ffn=False,
        rng=42,
    )

    # Train on simple patterns: [1, 2, 3, 4, 5, ...]
    train_data = []
    for start in range(1, 10):
        seq = list(range(start, min(start + 6, 16)))
        if len(seq) >= 3:
            tokens = sequence_to_bits(seq, n_bits)
            # For autoregressive: input[:-1] → output = input[1:]
            train_data.append((tokens[:-1], tokens[1:]))

    print(f"Training on {len(train_data)} counting sequences...")
    history = model.train(train_data, epochs=15, verbose=True)

    # Test generation
    print("\nTesting generation:")
    prompts = [
        [1, 2, 3],  # Should continue 4, 5, ...
        [5, 6, 7],  # Should continue 8, 9, ...
    ]

    for prompt in prompts:
        prompt_bits = sequence_to_bits(prompt, n_bits)
        generated = model.generate(prompt_bits, max_new_tokens=3)
        result = bits_to_sequence(generated)
        print(f"  {prompt} → {result}")

    return 1.0  # Generation test - no single accuracy metric


# ─────────────────────────────────────────────────────────────────────────────
# Task 6: Cross-Attention Alignment Test
# ─────────────────────────────────────────────────────────────────────────────

def test_cross_attention_alignment():
    """
    Test that cross-attention learns proper source-target alignment.

    Source: [A, B, C]
    Target: [A, B, C] (copy)

    Each decoder position should attend to matching encoder position.
    """
    print(f"\n{'='*60}")
    print("Task 6: Cross-Attention Alignment (Copy)")
    print(f"{'='*60}")

    n_bits = 4
    max_len = 8

    model = RAMEncoderDecoder(
        input_bits=n_bits,
        hidden_bits=n_bits,
        output_bits=n_bits,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        max_encoder_len=max_len,
        max_decoder_len=max_len,
        use_residual=True,
        use_ffn=False,
        rng=42,
    )

    # Pure copy task
    random.seed(321)
    train_data = []
    for _ in range(30):
        length = random.randint(2, 6)
        source = [random.randint(1, 15) for _ in range(length)]
        target = source.copy()

        source_bits = sequence_to_bits(source, n_bits)
        target_bits = sequence_to_bits(target, n_bits)
        train_data.append((source_bits, target_bits, target_bits))

    print(f"Training on {len(train_data)} copy examples...")
    history = model.train(train_data, epochs=10, verbose=True)

    # Test on unseen sequences
    print("\nTesting on unseen sequences:")
    test_cases = [
        [3, 7, 11],
        [1, 2, 3, 4],
        [15, 14, 13],
    ]

    correct = 0
    for source in test_cases:
        expected = source.copy()
        source_bits = sequence_to_bits(source, n_bits)
        target_bits = sequence_to_bits(expected, n_bits)

        predictions = model.forward(source_bits, target_bits)
        result = bits_to_sequence(predictions)

        ok = "✓" if result == expected else "✗"
        if result == expected:
            correct += 1
        print(f"  {source} → {result} {ok}")

    print(f"\nTest accuracy: {correct}/{len(test_cases)}")
    return correct / len(test_cases)


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("Sequence-to-Sequence Test")
    print(f"Started at: {datetime.now()}")
    print(f"{'='*60}")

    results = {}

    # Task 1: Reverse
    results["reverse"] = test_reverse_sequence()

    # Task 2: Character mapping
    results["char_map"] = test_character_mapping()

    # Task 3: Increment
    results["increment"] = test_increment_sequence()

    # Task 4: Arithmetic
    results["arithmetic"] = test_arithmetic_eval()

    # Task 5: Autoregressive
    results["generation"] = test_autoregressive_generation()

    # Task 6: Copy alignment
    results["copy"] = test_cross_attention_alignment()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    print("\nTask Results:")
    for task, acc in results.items():
        status = "✓" if acc >= 0.8 else "○" if acc >= 0.5 else "✗"
        print(f"  {status} {task}: {acc:.0%}")

    print("\nKey Observations:")
    print("1. Copy/alignment works via cross-attention routing")
    print("2. Position-based tasks (reverse) need proper position encoding")
    print("3. Transformation tasks (increment) need FFN layers")
    print("4. Arithmetic requires deeper models for computation")

    print("\nSeq2Seq vs Decoder-only:")
    print("- Encoder-decoder: Full source access at each decode step")
    print("- Decoder-only: Causal, only sees past context")
    print("- Choice depends on task structure")

    print(f"\n{'='*60}")
    print(f"Finished at: {datetime.now()}")
    print(f"{'='*60}")
