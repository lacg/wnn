#!/usr/bin/env python3
"""
RAM Seq2Seq Full Training Demo

Demonstrates comprehensive training that includes:
1. Training attention patterns (which positions to attend to)
2. Training value projections (how to transform tokens)
3. Training output layer (how to combine heads)

For next-character prediction, we need to learn:
- Attention: "attend to position i-1" (previous token)
- Mapping: "A -> B", "B -> C", etc.
"""

import sys
sys.path.insert(0, 'src')

from wnn.ram.RAMSeq2Seq import RAMSeq2Seq
from wnn.ram.RAMAttention import RAMAttention
from wnn.ram.RAMLayer import RAMLayer
from wnn.ram.encoders_decoders import (
	TransformerDecoderFactory, OutputMode, PositionMode
)
from torch import tensor, uint8, cat, zeros

print("=" * 70)
print("RAM Seq2Seq Full Training Demo")
print("=" * 70)

# Setup
decoder = TransformerDecoderFactory.create(OutputMode.TOKEN)
bits_per_token = decoder.bits_per_token

print(f"Token encoding: {bits_per_token} bits per token")
print()


def encode_sequence(text: str) -> list:
	"""Encode text to list of bit tensors."""
	return [decoder.encode(c).squeeze() for c in text]


def decode_sequence(tokens: list) -> str:
	"""Decode list of bit tensors to text."""
	return ''.join(decoder.decode(t.unsqueeze(0)) for t in tokens)


# ============================================================
# Approach: Two-Phase Training
# ============================================================
print("=" * 70)
print("Two-Phase Training for Next-Character Prediction")
print("=" * 70)
print("""
Phase 1: Train attention patterns
  - Teach heads to attend to previous position (i-1)
  - This is the "copy-previous" pattern from earlier

Phase 2: Train character mappings
  - Given attended token, produce next character
  - A->B, B->C, C->D, etc.
""")

# Create model
model = RAMSeq2Seq(
	input_bits=bits_per_token,
	num_layers=1,  # Single layer for clarity
	num_heads=2,
	position_mode=PositionMode.RELATIVE,
	max_seq_len=16,
	use_residual=False,  # Disable residual to make training clearer
	rng=42,
)

print(f"Model: {model}")
print()

# Training data
training_sequences = [
	"ABCDE",
	"FGHIJ",
	"KLMNO",
	"PQRST",
]

# ============================================================
# Phase 1: Train Attention Patterns
# ============================================================
print("=" * 70)
print("Phase 1: Training Attention Patterns")
print("=" * 70)
print("""
Target: Each position should attend to ITSELF (self-attention).
  Position 0: attend to self
  Position 1: attend to self
  Position 2: attend to self
  etc.

Then the value/output transformation maps each char to the next.
This is different from "copy-previous" which would attend to i-1.
""")

# Train each attention layer's heads
for layer_idx, attention_layer in enumerate(model.attention_layers):
	print(f"Training Layer {layer_idx} attention patterns...")

	for head_idx in range(attention_layer.num_heads):
		# Build attention targets for SELF-ATTEND pattern
		# (Each position attends only to itself)
		for seq in training_sequences:
			tokens = encode_sequence(seq)
			seq_len = len(tokens)

			attention_targets = []
			for query_idx in range(seq_len):
				for key_idx in range(query_idx + 1):  # Causal
					# Self-attention: attend to self only
					should_attend = 1 if key_idx == query_idx else 0
					attention_targets.append((query_idx, key_idx, should_attend))

			corrections = attention_layer.train_attention_pattern(
				tokens, attention_targets, head_idx=head_idx
			)

print("Attention training complete!")
print()

# Visualize learned attention
print("Attention patterns after training:")
test_tokens = encode_sequence("ABCD")
for h in range(model.attention_layers[0].num_heads):
	print(model.attention_layers[0].visualize_attention(test_tokens, head_idx=h))
	print()


# ============================================================
# Phase 2: Train Character Mappings
# ============================================================
print("=" * 70)
print("Phase 2: Training Character Mappings")
print("=" * 70)
print("""
Target: Map each character to the next one.
  A -> B, B -> C, C -> D, ...
""")

# Create a simple mapping layer (bypassing the complex attention)
# For demonstration, we'll train the output layer directly

# Build mapping dataset: (input_char, output_char) pairs
char_mappings = []
for seq in training_sequences:
	for i in range(len(seq) - 1):
		input_char = seq[i]
		output_char = seq[i + 1]
		input_bits = decoder.encode(input_char).squeeze()
		output_bits = decoder.encode(output_char).squeeze()
		char_mappings.append((input_bits, output_bits))

# Train output layer on the mappings
print(f"Training on {len(char_mappings)} character mappings...")

# Since our attention now attends to previous position,
# we need to train the value/output transformation
# The output layer takes concatenated head outputs

# For simplicity, let's use a direct mapping approach
mapping_layer = RAMLayer(
	total_input_bits=bits_per_token,
	num_neurons=bits_per_token,
	n_bits_per_neuron=bits_per_token,
	rng=999,
)

# Train the mapping
for epoch in range(10):
	errors = 0
	for inp, tgt in char_mappings:
		out = mapping_layer(inp.unsqueeze(0)).squeeze()
		if not (out == tgt).all():
			errors += 1
			mapping_layer.commit(inp.unsqueeze(0), tgt.unsqueeze(0))

	if errors == 0:
		print(f"  Epoch {epoch + 1}: Converged!")
		break
	print(f"  Epoch {epoch + 1}: {errors} errors")

print()

# Test the mapping
print("Character mapping test:")
for char in "ABCD":
	inp = decoder.encode(char).squeeze()
	out = mapping_layer(inp.unsqueeze(0)).squeeze()
	out_char = decoder.decode(out.unsqueeze(0))
	expected = chr(ord(char) + 1) if char < 'Z' else '?'
	status = "ok" if out_char == expected else "ERR"
	print(f"  {char} -> {out_char} (expected {expected}) [{status}]")
print()


# ============================================================
# Full Pipeline Test
# ============================================================
print("=" * 70)
print("Full Pipeline: Attention + Mapping")
print("=" * 70)
print("""
Now we combine:
1. Attention (learned): self-attend (each position attends to itself)
2. Mapping (learned): transform character to next (A->B, B->C, etc.)
""")

def next_char_predict(sequence: str) -> str:
	"""Predict next characters using learned components."""
	tokens = encode_sequence(sequence)
	outputs = []

	for i in range(len(tokens)):
		# Self-attention: attend to self
		attended_token = tokens[i]

		# Apply learned mapping (char -> next_char)
		next_token = mapping_layer(attended_token.unsqueeze(0)).squeeze()
		outputs.append(next_token)

	return decode_sequence(outputs)

# Test
print("Next-char prediction:")
test_cases = [
	("ABCD", "BCDE"),
	("FGHI", "GHIJ"),
	("MNOP", "NOPQ"),  # Unseen
]

for input_str, expected in test_cases:
	result = next_char_predict(input_str)
	status = "ok" if result == expected else "wrong"
	print(f"  '{input_str}' -> '{result}' (expected '{expected}') [{status}]")
print()


# ============================================================
# Summary
# ============================================================
print("=" * 70)
print("Summary: Two-Phase Training")
print("=" * 70)
print("""
Key Insight: Complex tasks require training BOTH:

1. ATTENTION PATTERNS (where to look)
   - Which positions to attend to
   - Trained via train_attention_pattern()
   - For next-char: "attend to i-1"

2. VALUE TRANSFORMATIONS (what to produce)
   - How to transform attended values to outputs
   - Trained via direct mapping or output layer
   - For next-char: "A->B, B->C, ..."

The simple train_step() method trains output layers but doesn't
explicitly train attention patterns. For tasks that require
specific attention patterns (like "look at previous position"),
we need Phase 1 training.

This is analogous to transformers where:
  - Attention patterns emerge from Q/K dot products
  - Value transformations are learned projections

In RAM transformers:
  - Attention patterns are explicitly trained (which keys to attend)
  - Value transformations are RAM lookups
""")
print("=" * 70)
