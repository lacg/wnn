#!/usr/bin/env python3
"""
Error Recovery Test

If scheduled sampling doesn't help because BIT_LEVEL generalizes,
what if we test a scenario where the model WILL make errors and
needs to recover?

Idea: Train on noisy data where sometimes the "previous token" is wrong.
This simulates what happens during autoregressive generation when
the model makes a mistake.
"""

import sys
sys.path.insert(0, 'src')

from wnn.ram.RAMLayer import RAMLayer
from wnn.ram.RAMGeneralization import MapperStrategy, GeneralizingProjection
from wnn.ram.encoders_decoders import TransformerDecoderFactory, OutputMode
from torch import zeros, uint8, Tensor
from torch.nn import Module
import random

print("=" * 70)
print("Error Recovery for Autoregressive Generation")
print("=" * 70)

# Setup
decoder = TransformerDecoderFactory.create(OutputMode.TOKEN)
bits_per_token = decoder.bits_per_token

def encode_char(c: str) -> Tensor:
    return decoder.encode(c).squeeze()

def decode_bits(bits: Tensor) -> str:
    return decoder.decode(bits.unsqueeze(0))

def decode_sequence(tokens: list) -> str:
    return ''.join(decode_bits(t) for t in tokens)


# =============================================================
# Error-Tolerant Autoregressive Model
# =============================================================
class ErrorTolerantModel(Module):
    """
    Model that learns to predict the next token even when
    given incorrect previous tokens.

    Key insight: Train on (wrong_prev, correct_next) pairs
    to teach error recovery.
    """

    def __init__(
        self,
        bits: int,
        strategy: MapperStrategy = MapperStrategy.BIT_LEVEL,
        rng: int | None = None,
    ):
        super().__init__()
        self.bits = bits
        self.strategy = strategy

        self.transition = GeneralizingProjection(
            input_bits=bits,
            output_bits=bits,
            strategy=strategy,
            rng=rng,
        )

        print(f"[ErrorTolerantModel] bits={bits}, strategy={strategy.name}")

    def forward(self, token: Tensor) -> Tensor:
        out = self.transition(token.squeeze())
        if out.ndim > 1:
            out = out.squeeze()
        return out

    def generate(self, start: Tensor, length: int) -> list[Tensor]:
        outputs = [start.squeeze().clone()]
        current = start.squeeze()

        for _ in range(length - 1):
            next_token = self.forward(current)
            outputs.append(next_token.clone())
            current = next_token

        return outputs

    def train_standard(self, sequence: list[Tensor]) -> int:
        """Standard training: correct_prev → correct_next."""
        sequence = [s.squeeze() for s in sequence]
        errors = 0

        for i in range(len(sequence) - 1):
            errors += self.transition.train_mapping(sequence[i], sequence[i + 1])

        return errors

    def train_with_noise(
        self,
        sequence: list[Tensor],
        noise_prob: float = 0.3,
    ) -> int:
        """
        Training with noisy previous tokens.

        Sometimes replace the "previous" token with a random nearby token.
        This teaches the model to still predict correctly even after errors.
        """
        sequence = [s.squeeze() for s in sequence]
        errors = 0

        for i in range(len(sequence) - 1):
            current = sequence[i]
            target = sequence[i + 1]

            # Sometimes corrupt the input
            if random.random() < noise_prob and i > 0:
                # Use a nearby token (simulate one-off error)
                offset = random.choice([-2, -1, 1, 2])
                corrupted_idx = max(0, min(25, i + offset))
                # Create corrupted token by bit-flipping
                current = current.clone()
                flip_bit = random.randint(0, self.bits - 1)
                current[flip_bit] = 1 - current[flip_bit]

            errors += self.transition.train_mapping(current, target)

        return errors


# =============================================================
# Test 1: Standard vs Noise-Trained on Counting
# =============================================================
print("\n" + "=" * 60)
print("Test 1: Standard vs Noise-Trained Counting")
print("=" * 60)

train_sequences = ["ABCDEFGH", "IJKLMNOP", "QRSTUVWX"]

for noise_prob in [0.0, 0.2, 0.4]:
    print(f"\n--- Noise probability: {noise_prob:.0%} ---")

    model = ErrorTolerantModel(
        bits=bits_per_token,
        strategy=MapperStrategy.BIT_LEVEL,
        rng=42,
    )

    # Train
    for epoch in range(10):
        for seq_str in train_sequences:
            sequence = [encode_char(c) for c in seq_str]
            if noise_prob > 0:
                model.train_with_noise(sequence, noise_prob)
            else:
                model.train_standard(sequence)

    # Test autoregressive generation
    test_cases = [
        ('A', 'ABCDEFGH'),
        ('I', 'IJKLMNOP'),
        ('M', 'MNOPQRST'),  # Unseen start
    ]

    for start_char, expected in test_cases:
        generated = model.generate(encode_char(start_char), length=8)
        result = decode_sequence(generated)
        correct = sum(1 for r, e in zip(result, expected) if r == e)
        pct = 100 * correct / len(expected)
        status = "OK" if pct >= 80 else "X"
        print(f"  [{status}] '{start_char}' -> '{result}' ({pct:.0f}%)")


# =============================================================
# Test 2: Error Injection During Generation
# =============================================================
print("\n" + "=" * 60)
print("Test 2: Error Injection During Generation")
print("=" * 60)
print("Inject an error mid-sequence, see if model recovers")

def generate_with_error(model, start: Tensor, length: int, error_pos: int, error_token: Tensor):
    """Generate but inject an error at a specific position."""
    outputs = [start.squeeze().clone()]
    current = start.squeeze()

    for i in range(length - 1):
        if i == error_pos - 1:
            # Inject error: use wrong token as input
            current = error_token.squeeze()
            next_token = model.forward(current)
            outputs.append(next_token.clone())
        else:
            next_token = model.forward(current)
            outputs.append(next_token.clone())

        current = next_token

    return outputs

# Train model with noise
print("\n--- Training with 30% noise ---")
model = ErrorTolerantModel(
    bits=bits_per_token,
    strategy=MapperStrategy.BIT_LEVEL,
    rng=42,
)

for epoch in range(15):
    for seq_str in ["ABCDEFGH", "IJKLMNOP", "QRSTUVWX"]:
        sequence = [encode_char(c) for c in seq_str]
        model.train_with_noise(sequence, noise_prob=0.3)

# Test: inject error at position 3
print("\nInjecting error at position 3:")
for start_char in ['A', 'I']:
    # Normal generation
    normal = model.generate(encode_char(start_char), length=8)
    normal_result = decode_sequence(normal)

    # Generation with error
    error_token = encode_char('Z')  # Inject 'Z' at position 3
    with_error = generate_with_error(model, encode_char(start_char), 8, error_pos=3, error_token=error_token)
    error_result = decode_sequence(with_error)

    print(f"  Normal:     '{start_char}' -> '{normal_result}'")
    print(f"  With error: '{start_char}' -> '{error_result}' (Z injected at pos 3)")


# =============================================================
# Test 3: Compare DIRECT vs BIT_LEVEL for Error Recovery
# =============================================================
print("\n" + "=" * 60)
print("Test 3: DIRECT vs BIT_LEVEL for Error Recovery")
print("=" * 60)

for strategy in [MapperStrategy.DIRECT, MapperStrategy.BIT_LEVEL]:
    print(f"\n--- {strategy.name} with 30% noise training ---")

    model = ErrorTolerantModel(
        bits=bits_per_token,
        strategy=strategy,
        rng=42,
    )

    # Train with noise
    for epoch in range(15):
        for seq_str in ["ABCDEFGH", "IJKLMNOP"]:
            sequence = [encode_char(c) for c in seq_str]
            model.train_with_noise(sequence, noise_prob=0.3)

    # Test
    for start_char, expected in [('A', 'ABCDEFGH'), ('E', 'EFGHIJKL')]:
        generated = model.generate(encode_char(start_char), length=8)
        result = decode_sequence(generated)
        correct = sum(1 for r, e in zip(result, expected) if r == e)
        pct = 100 * correct / len(expected)
        print(f"  '{start_char}' -> '{result}' ({pct:.0f}%)")


# =============================================================
# Summary
# =============================================================
print("\n" + "=" * 60)
print("Summary: Error Recovery Training")
print("=" * 60)

print("""
Key Findings:

1. NOISE TRAINING:
   - Training with corrupted inputs teaches error tolerance
   - Model learns: "even if prev is wrong, next should be X"

2. RAM NETWORK BEHAVIOR:
   - With noise: More patterns stored (corrupted → correct)
   - May help recover from generation errors
   - Risk: Conflicts if too many different inputs → same output

3. BIT_LEVEL vs DIRECT:
   - BIT_LEVEL: Noise training may hurt pattern learning
     (bit patterns become inconsistent)
   - DIRECT: Noise training adds more explicit transitions
     (but explodes memory)

4. PRACTICAL INSIGHT:
   For RAM networks, the best error handling strategy is:
   - Use BIT_LEVEL for its generalization power
   - Accept that some patterns won't generalize
   - For critical applications, use DIRECT with all transitions

The fundamental limit is not training strategy but pattern coverage.
""")

print("=" * 70)
print("Tests completed!")
print("=" * 70)
