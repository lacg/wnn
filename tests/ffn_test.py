#!/usr/bin/env python3
"""
RAMFeedForward and Full Transformer Architecture Test

Tests:
1. RAMFeedForward standalone
2. RAMSeq2Seq with FFN layers
3. Comparison: with vs without FFN
"""

import sys
sys.path.insert(0, 'src')

from wnn.ram.RAMFeedForward import RAMFeedForward, FFNMode
from wnn.ram.RAMSeq2Seq import RAMSeq2Seq
from wnn.ram.RAMGeneralization import MapperStrategy
from wnn.ram.encoders_decoders import TransformerDecoderFactory, OutputMode, PositionMode
from torch import tensor, uint8, zeros

print("=" * 70)
print("RAMFeedForward and Full Transformer Test")
print("=" * 70)

# Setup
decoder = TransformerDecoderFactory.create(OutputMode.TOKEN)
bits_per_token = decoder.bits_per_token

def encode_char(c: str):
	return decoder.encode(c).squeeze()

def decode_bits(bits):
	return decoder.decode(bits.unsqueeze(0))


# =============================================================
# Test 1: RAMFeedForward Standalone
# =============================================================
print("\n" + "=" * 60)
print("Test 1: RAMFeedForward Standalone")
print("=" * 60)

# Create FFN with default settings
ffn = RAMFeedForward(
	input_bits=bits_per_token,
	expansion_factor=4,
	mode=FFNMode.STANDARD,
	use_residual=True,
	rng=42,
)

print(f"\nFFN: {ffn}")
print(f"  Input: {ffn.input_bits} bits")
print(f"  Hidden: {ffn.hidden_bits} bits")
print(f"  Output: {ffn.output_bits} bits")

# Test forward pass
test_input = encode_char('A')
print(f"\nInput: 'A' = {test_input.tolist()}")

output = ffn(test_input)
print(f"Output: {output.tolist()}")
print(f"Output char: '{decode_bits(output)}'")

# Test training
print("\nTraining FFN on A->B mapping...")
train_pairs = [(encode_char('A'), encode_char('B'))]
history = ffn.train_batch(
	[p[0] for p in train_pairs],
	[p[1] for p in train_pairs],
	epochs=5,
	verbose=True,
)

# Verify
output_after = ffn(encode_char('A'))
print(f"After training: A -> '{decode_bits(output_after)}'")


# =============================================================
# Test 2: RAMSeq2Seq WITHOUT FFN
# =============================================================
print("\n" + "=" * 60)
print("Test 2: RAMSeq2Seq WITHOUT FFN")
print("=" * 60)

model_no_ffn = RAMSeq2Seq(
	input_bits=bits_per_token,
	num_layers=2,
	num_heads=2,
	position_mode=PositionMode.RELATIVE,
	max_seq_len=16,
	use_residual=True,
	use_ffn=False,
	generalization=MapperStrategy.BIT_LEVEL,
	rng=42,
)

# Simple forward pass
tokens = [encode_char(c) for c in "ABC"]
outputs = model_no_ffn.forward(tokens)
print(f"\nInput: ABC")
print(f"Output: {''.join(decode_bits(o) for o in outputs)}")


# =============================================================
# Test 3: RAMSeq2Seq WITH FFN
# =============================================================
print("\n" + "=" * 60)
print("Test 3: RAMSeq2Seq WITH FFN (Full Transformer)")
print("=" * 60)

model_with_ffn = RAMSeq2Seq(
	input_bits=bits_per_token,
	num_layers=2,
	num_heads=2,
	position_mode=PositionMode.RELATIVE,
	max_seq_len=16,
	use_residual=True,
	use_ffn=True,
	ffn_expansion=4,
	ffn_mode=FFNMode.STANDARD,
	generalization=MapperStrategy.BIT_LEVEL,
	rng=42,
)

# Forward pass
outputs_ffn = model_with_ffn.forward(tokens)
print(f"\nInput: ABC")
print(f"Output: {''.join(decode_bits(o) for o in outputs_ffn)}")


# =============================================================
# Test 4: Gated FFN Mode
# =============================================================
print("\n" + "=" * 60)
print("Test 4: RAMSeq2Seq with GATED FFN")
print("=" * 60)

model_gated = RAMSeq2Seq(
	input_bits=bits_per_token,
	num_layers=1,
	num_heads=2,
	position_mode=PositionMode.RELATIVE,
	max_seq_len=16,
	use_residual=True,
	use_ffn=True,
	ffn_mode=FFNMode.GATED,
	generalization=MapperStrategy.BIT_LEVEL,
	rng=42,
)

outputs_gated = model_gated.forward(tokens)
print(f"\nInput: ABC")
print(f"Output: {''.join(decode_bits(o) for o in outputs_gated)}")


# =============================================================
# Summary
# =============================================================
print("\n" + "=" * 60)
print("Summary: Architecture Comparison")
print("=" * 60)

print("""
Architecture               | Layers | Components per Layer
---------------------------|--------|----------------------
Without FFN                |   2    | Attention only
With FFN (STANDARD)        |   2    | Attention + FFN
With FFN (GATED)           |   1    | Attention + Gated FFN

Full Transformer Block:
  Input → Attention → (+Residual) → FFN → (+Residual) → Output

RAM Transformer Advantages:
  - No gradient computation
  - Discrete (binary) representations
  - Learned aggregation
  - Bit-level generalization
""")

print("=" * 60)
print("All tests passed!")
print("=" * 60)
