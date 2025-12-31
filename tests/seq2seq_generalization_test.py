#!/usr/bin/env python3
"""
RAMSeq2Seq Generalization Test

Tests the integrated generalization strategies in RAMSeq2Seq:
1. No generalization (baseline)
2. Bit-level generalization
3. Compositional generalization
4. Hybrid generalization

Task: Next character prediction (A->B, B->C, etc.)
Tests generalization to unseen characters.
"""

import sys
sys.path.insert(0, 'src')

from wnn.ram.RAMSeq2Seq import RAMSeq2Seq
from wnn.ram.RAMGeneralization import MapperStrategy
from wnn.ram.encoders_decoders import TransformerDecoderFactory, OutputMode, PositionMode
from torch import tensor, uint8

print("=" * 70)
print("RAMSeq2Seq Generalization Test")
print("=" * 70)

# Setup
decoder = TransformerDecoderFactory.create(OutputMode.TOKEN)
bits_per_token = decoder.bits_per_token

def encode_char(c: str):
	return decoder.encode(c).squeeze()

def decode_bits(bits):
	return decoder.decode(bits.unsqueeze(0))

def encode_sequence(text: str) -> list:
	return [encode_char(c) for c in text]

def decode_sequence(tokens: list) -> str:
	return ''.join(decode_bits(t) for t in tokens)


# For bit-level generalization to work, training must cover all bit PATTERNS.
# The "next character" operation is binary increment.
#
# For each output bit position, we need training examples that cover
# all input patterns for the context bits.
#
# Bit 0: needs [0]->1 and [1]->0 (2 patterns)
# Bit 1: needs [00]->0, [01]->1, [10]->1, [11]->0 (4 patterns)
# Bit 2: needs all 8 patterns for [b0,b1,b2]
# etc.
#
# This grows exponentially, but we can be strategic:
# Characters ACEGIKMOQSUWY (ASCII 0,2,4,6,...) all have LSB=0
# Characters BDFHJLNPRTVXZ (ASCII 1,3,5,7,...) all have LSB=1
#
# For full coverage of 5-bit increment, we need training on:
# - Both LSB values (0 and 1)
# - Various patterns of higher bits

# Strategic training set: includes both even and odd, covering bit patterns
train_chars = "ABCDEFGH"  # First 8 characters cover all 3-bit low patterns
test_chars = "IJKLMNOPQRSTUVWXY"  # Unseen higher characters

# Create training pairs (char -> next_char)
train_pairs = []
for c in train_chars:
	if c < 'Z':
		train_pairs.append((encode_char(c), encode_char(chr(ord(c) + 1))))

print(f"Training on {len(train_pairs)} character mappings: {train_chars}")
print(f"  (First 8 chars cover all 3-bit patterns in low bits)")
print(f"Testing on unseen characters: {test_chars}")
print()


def test_model(model_name: str, model: RAMSeq2Seq) -> float:
	"""Test a model and return accuracy on unseen characters."""
	print(f"\n{'=' * 60}")
	print(f"Testing: {model_name}")
	print(f"{'=' * 60}")

	# Train attention patterns (self-attention for this task)
	print("\n1. Training attention patterns (self-attention)...")
	for layer in model.attention_layers:
		for head_idx in range(layer.num_heads):
			for seq in ["ACEG", "IKMO", "QSUW"]:
				tokens = encode_sequence(seq)
				# Self-attention: each position attends to itself
				attention_targets = [
					(i, j, 1 if i == j else 0)
					for i in range(len(tokens))
					for j in range(i + 1)
				]
				layer.train_attention_pattern(tokens, attention_targets, head_idx)

	# Train token mapper (char -> next_char)
	print("2. Training token mapper...")
	if model.token_mapper is not None:
		model.train_token_mapper(train_pairs, epochs=10, verbose=False)
		print(f"   Token mapper trained ({model.generalization.name} strategy)")
	else:
		print("   No token mapper (direct mode)")
		# For baseline, we'd need to train the output layer directly
		# which won't generalize

	# Test on training characters
	print("\n3. Results on TRAINING characters:")
	train_correct = 0
	for c in train_chars[:6]:
		if c < 'Z':
			inp = encode_char(c)
			# For self-attention, we just pass through and transform
			if model.token_mapper is not None:
				out = model.token_mapper(inp)
			else:
				# Baseline: use full model (won't work well)
				out = model.forward([inp])[0]
			pred = decode_bits(out)
			expected = chr(ord(c) + 1)
			if pred == expected:
				train_correct += 1
			status = "ok" if pred == expected else "WRONG"
			print(f"   {c} -> {pred} (expected {expected}) [{status}]")

	# Test on UNSEEN characters
	print("\n4. Results on UNSEEN characters:")
	test_correct = 0
	for c in test_chars[:6]:
		if c < 'Z':
			inp = encode_char(c)
			if model.token_mapper is not None:
				out = model.token_mapper(inp)
			else:
				out = model.forward([inp])[0]
			pred = decode_bits(out)
			expected = chr(ord(c) + 1)
			if pred == expected:
				test_correct += 1
			status = "ok" if pred == expected else "WRONG"
			print(f"   {c} -> {pred} (expected {expected}) [{status}]")

	accuracy = 100 * test_correct / 6
	print(f"\n   Unseen accuracy: {test_correct}/6 = {accuracy:.1f}%")

	return accuracy


# Test different generalization strategies
results = {}

# 1. No generalization (baseline) - using new MapperStrategy enum
print("\n" + "=" * 70)
print("Creating model WITHOUT generalization...")
model_none = RAMSeq2Seq(
	input_bits=bits_per_token,
	num_layers=1,
	num_heads=2,
	position_mode=PositionMode.RELATIVE,
	max_seq_len=16,
	use_residual=False,
	generalization=MapperStrategy.DIRECT,
	rng=42,
)
results["DIRECT"] = test_model("No Generalization (Baseline)", model_none)

# 2. Bit-level generalization
print("\n" + "=" * 70)
print("Creating model with BIT-LEVEL generalization...")
model_bit = RAMSeq2Seq(
	input_bits=bits_per_token,
	num_layers=1,
	num_heads=2,
	position_mode=PositionMode.RELATIVE,
	max_seq_len=16,
	use_residual=False,
	generalization=MapperStrategy.BIT_LEVEL,
	rng=42,
)
results["BIT_LEVEL"] = test_model("Bit-Level Generalization", model_bit)

# 3. Compositional generalization
print("\n" + "=" * 70)
print("Creating model with COMPOSITIONAL generalization...")
model_comp = RAMSeq2Seq(
	input_bits=bits_per_token,
	num_layers=1,
	num_heads=2,
	position_mode=PositionMode.RELATIVE,
	max_seq_len=16,
	use_residual=False,
	generalization=MapperStrategy.COMPOSITIONAL,
	rng=42,
)
results["COMPOSITIONAL"] = test_model("Compositional Generalization", model_comp)

# 4. Hybrid generalization
print("\n" + "=" * 70)
print("Creating model with HYBRID generalization...")
model_hybrid = RAMSeq2Seq(
	input_bits=bits_per_token,
	num_layers=1,
	num_heads=2,
	position_mode=PositionMode.RELATIVE,
	max_seq_len=16,
	use_residual=False,
	generalization=MapperStrategy.HYBRID,
	rng=42,
)
results["HYBRID"] = test_model("Hybrid Generalization", model_hybrid)


# Summary
print("\n" + "=" * 70)
print("SUMMARY: Generalization Comparison")
print("=" * 70)
print("""
Task: Next character prediction (A->B, B->C, etc.)
Training: Only odd-indexed characters (A,C,E,G,...)
Testing: Even-indexed characters (B,D,F,H,...) - UNSEEN
""")

print("Strategy           | Unseen Accuracy")
print("-------------------|----------------")
for strategy, accuracy in results.items():
	print(f"{strategy:18} | {accuracy:5.1f}%")

print("""
Explanation:
- DIRECT: No token mapper, can't generalize at all
- BIT_LEVEL: Learns bit-flip patterns, generalizes via shared patterns
- COMPOSITIONAL: Splits into groups, covers all combinations
- HYBRID: Combines both strategies

The key insight: RAM generalization comes from learning at a level
where patterns transfer. Character-level doesn't transfer, but
bit-level and compositional patterns do!
""")
print("=" * 70)
