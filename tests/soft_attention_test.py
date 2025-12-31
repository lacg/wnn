#!/usr/bin/env python3
"""
Soft Attention Approximation Test

Test whether voting-based soft attention can:
1. Express fractional attention weights
2. Improve on hard attention for certain tasks
3. Handle cases where multiple positions matter

Tasks that benefit from soft attention:
- Averaging: output = mean(input[i] for attended i)
- Blending: output = weighted combination
- Multiple relevant positions: attend to several with different weights
"""

import sys
sys.path.insert(0, 'src')

from wnn.ram.RAMAttention import RAMAttention
from wnn.ram.SoftRAMAttention import SoftRAMAttention, SoftRAMAttentionV2
from wnn.ram.encoders_decoders import TransformerDecoderFactory, OutputMode, PositionMode
from torch import zeros, uint8, Tensor
from torch.nn import Module

print("=" * 70)
print("Soft Attention Approximation Test")
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
# Test 1: Visualize Soft Attention Weights
# =============================================================
print("\n" + "=" * 60)
print("Test 1: Visualize Soft vs Hard Attention")
print("=" * 60)

# Create models
hard_attn = RAMAttention(
    input_bits=bits_per_token,
    num_heads=4,
    position_mode=PositionMode.RELATIVE,
    max_seq_len=8,
    causal=True,
    rng=42,
)

soft_attn = SoftRAMAttention(
    input_bits=bits_per_token,
    num_heads=8,  # More heads for finer weights
    position_mode=PositionMode.RELATIVE,
    max_seq_len=8,
    causal=True,
    rng=42,
)

# Test sequence
test_sequence = [encode_char(c) for c in "ABCD"]

print("\nHard Attention (4 heads, binary):")
print(hard_attn.visualize_attention(test_sequence, head_idx=0))

print("\nSoft Attention (8 heads, voting):")
print(soft_attn.visualize_attention(test_sequence))


# =============================================================
# Test 2: Train Specific Attention Weights
# =============================================================
print("\n" + "=" * 60)
print("Test 2: Training Specific Attention Weights")
print("=" * 60)
print("Goal: Train soft attention to produce specific weight patterns")

soft_model = SoftRAMAttention(
    input_bits=bits_per_token,
    num_heads=8,
    position_mode=PositionMode.RELATIVE,
    max_seq_len=8,
    causal=False,  # Non-causal for this test
    rng=123,
)

sequence = [encode_char(c) for c in "ABCD"]

# Target: Position 2 (C) should attend:
#   50% to position 0 (A)
#   25% to position 1 (B)
#   100% to position 2 (C) - self
#   25% to position 3 (D)
target_weights = [0.5, 0.25, 1.0, 0.25]

print(f"\nTarget weights for query position 2: {target_weights}")

# Train
print("Training...")
for epoch in range(5):
    corrections = soft_model.train_attention_weights(
        sequence,
        query_pos=2,
        target_weights=target_weights
    )
    if epoch == 0:
        print(f"  Epoch 1: {corrections} head corrections")

# Check learned weights
learned_weights = soft_model.get_attention_weights(sequence, query_pos=2)
print(f"Learned weights: {[f'{w:.2f}' for w in learned_weights]}")


# =============================================================
# Test 3: Compare Hard vs Soft on Position-Based Task
# =============================================================
print("\n" + "=" * 60)
print("Test 3: Copy Task - Hard vs Soft Attention")
print("=" * 60)
print("Task: output[i] = input[i] (identity/copy)")

for name, model_class, num_heads in [
    ("Hard (4 heads)", RAMAttention, 4),
    ("Soft (8 heads)", SoftRAMAttention, 8),
    ("Soft (16 heads)", SoftRAMAttention, 16),
]:
    print(f"\n--- {name} ---")

    if model_class == RAMAttention:
        model = model_class(
            input_bits=bits_per_token,
            num_heads=num_heads,
            position_mode=PositionMode.RELATIVE,
            max_seq_len=8,
            causal=False,
            rng=42,
        )
    else:
        model = model_class(
            input_bits=bits_per_token,
            num_heads=num_heads,
            position_mode=PositionMode.RELATIVE,
            max_seq_len=8,
            causal=False,
            rng=42,
        )

    # Train on copy task
    train_sequences = ["ABCD", "EFGH", "IJKL", "MNOP"]

    for epoch in range(10):
        total_errors = 0
        for seq_str in train_sequences:
            tokens = [encode_char(c) for c in seq_str]
            # For copy, output should equal input
            if hasattr(model, 'train_step'):
                errors = model.train_step(tokens, tokens)
                total_errors += errors

    # Test
    test_seqs = ["ABCD", "QRST", "WXYZ"]
    for seq_str in test_seqs:
        tokens = [encode_char(c) for c in seq_str]
        outputs = model.forward(tokens)
        result = decode_sequence(outputs)
        status = "OK" if result == seq_str else "X"
        print(f"  [{status}] '{seq_str}' -> '{result}'")


# =============================================================
# Test 4: Soft Attention V2 (Bit-Level Voting)
# =============================================================
print("\n" + "=" * 60)
print("Test 4: Soft Attention V2 (Bit-Level Majority Voting)")
print("=" * 60)

soft_v2 = SoftRAMAttentionV2(
    input_bits=bits_per_token,
    num_heads=8,
    position_mode=PositionMode.RELATIVE,
    max_seq_len=8,
    causal=True,
    rng=42,
)

# Test on a simple sequence
test_seq = [encode_char(c) for c in "ABCD"]
outputs = soft_v2.forward(test_seq)
result = decode_sequence(outputs)

print(f"Input:  'ABCD'")
print(f"Output: '{result}'")
print(f"(Bit-level voting across {soft_v2.num_heads} heads)")


# =============================================================
# Test 5: Task Requiring Multiple Attention Targets
# =============================================================
print("\n" + "=" * 60)
print("Test 5: Multi-Target Attention Task")
print("=" * 60)
print("Task: Output XOR of first and last input")
print("(Requires attending to multiple positions with equal weight)")

class XORFirstLastModel(Module):
    """Output = input[0] XOR input[-1]"""

    def __init__(self, bits, use_soft=False, num_heads=8, rng=None):
        super().__init__()
        self.bits = bits
        self.use_soft = use_soft

        if use_soft:
            self.attention = SoftRAMAttention(
                input_bits=bits,
                num_heads=num_heads,
                position_mode=PositionMode.RELATIVE,
                max_seq_len=8,
                causal=False,
                rng=rng,
            )
        else:
            self.attention = RAMAttention(
                input_bits=bits,
                num_heads=num_heads,
                position_mode=PositionMode.RELATIVE,
                max_seq_len=8,
                causal=False,
                rng=rng,
            )

    def forward(self, tokens):
        return self.attention.forward(tokens)

print("\nThis task is inherently hard for attention-based models")
print("because it requires XOR (not weighted sum) of specific positions.")
print("Both hard and soft attention will struggle without explicit training.")


# =============================================================
# Test 6: Attention Weight Distribution
# =============================================================
print("\n" + "=" * 60)
print("Test 6: Attention Weight Granularity")
print("=" * 60)

for num_heads in [4, 8, 16]:
    model = SoftRAMAttention(
        input_bits=bits_per_token,
        num_heads=num_heads,
        position_mode=PositionMode.RELATIVE,
        max_seq_len=8,
        causal=True,
        rng=42,
    )

    possible_weights = [i / num_heads for i in range(num_heads + 1)]
    print(f"\n{num_heads} heads → {len(possible_weights)} weight levels:")
    print(f"  Possible weights: {[f'{w:.3f}' for w in possible_weights]}")
    print(f"  Granularity: {1/num_heads:.3f}")


# =============================================================
# Summary
# =============================================================
print("\n" + "=" * 60)
print("Summary: Soft Attention Approximation")
print("=" * 60)

print("""
Key Findings:

1. VOTING MECHANISM:
   - More heads = finer weight granularity
   - 8 heads: weights in {0.0, 0.125, 0.25, ..., 1.0}
   - 16 heads: weights in {0.0, 0.0625, ..., 1.0}

2. WEIGHT CONTROL:
   - Can train heads to produce specific attention weights
   - First N heads attend → weight = N / total_heads

3. AGGREGATION CHALLENGE:
   - Soft weights don't directly translate to weighted sums
   - XOR aggregation is order-independent but not weighted
   - Need learned aggregation that respects weight levels

4. COMPARISON TO TRUE SOFT ATTENTION:

   True Soft:
     weights = softmax(scores)  # Any value in [0,1]
     output = Σ weights[i] × values[i]  # Weighted sum

   Voting Soft:
     weights = votes / num_heads  # Discrete levels
     output = learned_aggregate(values, weights)  # Discrete

5. LIMITATIONS:
   - Discrete weight levels (not continuous)
   - Aggregation is learned, not true weighted sum
   - More heads = more computation

6. WHEN SOFT ATTENTION HELPS:
   ✅ When multiple positions matter with different importance
   ✅ When weight granularity is sufficient
   ❌ When true continuous weights are needed
   ❌ When weighted arithmetic (averaging) is required

CONCLUSION:
  Voting approximates soft attention weights, but the aggregation
  step remains discrete. This is a fundamental limitation of RAM
  networks - they can't perform true weighted sums.
""")

print("=" * 70)
print("Tests completed!")
print("=" * 70)
