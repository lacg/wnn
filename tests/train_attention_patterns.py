#!/usr/bin/env python3
"""
Training RAM Attention Patterns

Demonstrates how RAM attention learns WHERE to attend
through training on specific tasks.

Tasks:
1. Copy Previous: position i should attend to i-1
2. Copy First: all positions attend to position 0
3. Self-Attend: each position attends to itself
"""

import sys
sys.path.insert(0, 'src')

from wnn.ram.RAMLayer import RAMLayer
from wnn.ram.encoders_decoders import TransformerDecoderFactory, OutputMode, PositionEncoderFactory, PositionMode
from torch import tensor, uint8, zeros, cat, stack

print("="*70)
print("Training RAM Attention Patterns")
print("="*70)

# Setup
decoder = TransformerDecoderFactory.create(OutputMode.TOKEN)
bits_per_token = decoder.bits_per_token
pos_encoder = PositionEncoderFactory.create(PositionMode.BINARY, max_seq_len=16)
pos_bits = pos_encoder.n_bits

print(f"Token bits: {bits_per_token}, Position bits: {pos_bits}")
print()

def encode_sequence(text: str) -> list:
	"""Encode text to list of bit tensors."""
	return [decoder.encode(c).squeeze() for c in text]


def visualize_attention(attn_layer, tokens, title=""):
	"""Visualize attention pattern."""
	seq_len = len(tokens)
	print(f"{title}")
	print("    " + " ".join(f"{j:2d}" for j in range(seq_len)))

	for i in range(seq_len):
		row = f"{i:2d}: "
		q_pos = pos_encoder.encode(i)
		q = tokens[i]

		for j in range(seq_len):
			if j > i:  # Causal mask
				row += " - "
				continue

			k_pos = pos_encoder.encode(j)
			k = tokens[j]

			# Build attention input: [query, key, query_pos, key_pos]
			attn_input = cat([q, k, q_pos, k_pos]).unsqueeze(0)
			attend = attn_layer(attn_input).squeeze().item()

			row += " # " if attend else " . "
		print(row)
	print()


# ============================================================
# Task 1: Copy Previous Token
# Position i should attend ONLY to position i-1
# ============================================================
print("="*70)
print("Task 1: COPY PREVIOUS (attend to i-1)")
print("="*70)
print("""
Target pattern:     Meaning:
    0  1  2  3      Position 0: attend to self (no previous)
 0: #  -  -  -      Position 1: attend to position 0
 1: #  .  -  -      Position 2: attend to position 1
 2: .  #  .  -      Position 3: attend to position 2
 3: .  .  #  .
""")

# Create attention layer for this task
# Input: [query_token, key_token, query_pos, key_pos]
attn_input_size = 2 * bits_per_token + 2 * pos_bits
copy_prev_layer = RAMLayer(
	total_input_bits=attn_input_size,
	num_neurons=1,  # Binary: attend or not
	n_bits_per_neuron=min(attn_input_size, 12),
	rng=42,
)

# Training data
sequences = ["ABCD", "WXYZ", "HELLO", "TEST"]

print("Before training:")
tokens = encode_sequence("ABCD")
visualize_attention(copy_prev_layer, tokens, "Attention for 'ABCD':")

# Train: for each (query_pos, key_pos) pair, learn attend=1 if key_pos == query_pos-1
print("Training...")
epochs = 50
for epoch in range(epochs):
	for seq in sequences:
		tokens = encode_sequence(seq)
		seq_len = len(tokens)

		for i in range(seq_len):  # Query position
			q = tokens[i]
			q_pos = pos_encoder.encode(i)

			for j in range(i + 1):  # Key position (causal)
				k = tokens[j]
				k_pos = pos_encoder.encode(j)

				# Target: attend if j == i-1, or j == 0 when i == 0
				if i == 0:
					target_attend = 1 if j == 0 else 0  # First pos attends to self
				else:
					target_attend = 1 if j == i - 1 else 0

				# Build input and train
				attn_input = cat([q, k, q_pos, k_pos]).unsqueeze(0)
				target = tensor([[target_attend]], dtype=uint8)

				# Forward
				output = copy_prev_layer(attn_input)

				# Train if wrong
				if output.item() != target_attend:
					# Use commit to force the correct mapping
					copy_prev_layer.commit(attn_input, target)

print(f"Trained for {epochs} epochs on {len(sequences)} sequences")
print()

print("After training:")
for test_seq in ["ABCD", "WXYZ", "NEW"]:
	tokens = encode_sequence(test_seq)
	visualize_attention(copy_prev_layer, tokens, f"Attention for '{test_seq}':")


# ============================================================
# Task 2: Copy First Token
# All positions should attend to position 0
# ============================================================
print("="*70)
print("Task 2: COPY FIRST (all attend to position 0)")
print("="*70)
print("""
Target pattern:     Meaning:
    0  1  2  3      Every position attends to position 0
 0: #  -  -  -      This is like a "global context" pattern
 1: #  .  -  -
 2: #  .  .  -
 3: #  .  .  .
""")

copy_first_layer = RAMLayer(
	total_input_bits=attn_input_size,
	num_neurons=1,
	n_bits_per_neuron=min(attn_input_size, 12),
	rng=123,
)

print("Before training:")
tokens = encode_sequence("ABCD")
visualize_attention(copy_first_layer, tokens, "Attention for 'ABCD':")

print("Training...")
for epoch in range(epochs):
	for seq in sequences:
		tokens = encode_sequence(seq)
		seq_len = len(tokens)

		for i in range(seq_len):
			q = tokens[i]
			q_pos = pos_encoder.encode(i)

			for j in range(i + 1):
				k = tokens[j]
				k_pos = pos_encoder.encode(j)

				# Target: attend ONLY to position 0
				target_attend = 1 if j == 0 else 0

				attn_input = cat([q, k, q_pos, k_pos]).unsqueeze(0)
				target = tensor([[target_attend]], dtype=uint8)

				output = copy_first_layer(attn_input)
				if output.item() != target_attend:
					copy_first_layer.commit(attn_input, target)

print(f"Trained for {epochs} epochs")
print()

print("After training:")
for test_seq in ["ABCD", "TEST"]:
	tokens = encode_sequence(test_seq)
	visualize_attention(copy_first_layer, tokens, f"Attention for '{test_seq}':")


# ============================================================
# Task 3: Content-Based Attention
# Attend to positions with SAME token
# ============================================================
print("="*70)
print("Task 3: CONTENT-BASED (attend to same token)")
print("="*70)
print("""
For 'ABBA':
    0  1  2  3      Position 0 (A): attend to self
 0: #  -  -  -      Position 1 (B): attend to self
 1: .  #  -  -      Position 2 (B): attend to positions 1,2 (both B)
 2: .  #  #  -      Position 3 (A): attend to positions 0,3 (both A)
 3: #  .  .  #
""")

content_attn_layer = RAMLayer(
	total_input_bits=attn_input_size,
	num_neurons=1,
	n_bits_per_neuron=min(attn_input_size, 14),  # More connectivity
	rng=456,
)

# Training sequences with repeated tokens
content_sequences = ["ABBA", "ABAB", "AABB", "XYXY", "MAMA", "NOON"]

print("Before training:")
tokens = encode_sequence("ABBA")
visualize_attention(content_attn_layer, tokens, "Attention for 'ABBA':")

print("Training...")
for epoch in range(100):  # More epochs for content-based
	for seq in content_sequences:
		tokens = encode_sequence(seq)
		seq_len = len(tokens)

		for i in range(seq_len):
			q = tokens[i]
			q_pos = pos_encoder.encode(i)
			q_char = seq[i]

			for j in range(i + 1):
				k = tokens[j]
				k_pos = pos_encoder.encode(j)
				k_char = seq[j]

				# Target: attend if same character
				target_attend = 1 if q_char == k_char else 0

				attn_input = cat([q, k, q_pos, k_pos]).unsqueeze(0)
				target = tensor([[target_attend]], dtype=uint8)

				output = content_attn_layer(attn_input)
				if output.item() != target_attend:
					content_attn_layer.commit(attn_input, target)

print(f"Trained for 100 epochs on {len(content_sequences)} sequences")
print()

print("After training:")
for test_seq in ["ABBA", "NOON", "ABCD"]:
	tokens = encode_sequence(test_seq)
	visualize_attention(content_attn_layer, tokens, f"Attention for '{test_seq}':")


# ============================================================
# Summary
# ============================================================
print("="*70)
print("Summary: RAM Attention Pattern Learning")
print("="*70)
print("""
RAM attention CAN learn different patterns:

1. POSITIONAL patterns (copy previous, copy first)
   - Learns WHERE to attend based on position
   - Generalizes to new sequences

2. CONTENT-BASED patterns (attend to same token)
   - Learns WHAT to attend to based on content
   - This is the key transformer capability!

Key insight: The RAM neuron learns a RULE:
   "Given (query, key, positions) -> should I attend?"

This rule can encode:
   - Position relationships (i attends to i-1)
   - Content matching (attend if query == key)
   - Combinations (attend to recent similar tokens)

LIMITATIONS:
   - Binary (attend/not) vs continuous weights
   - No fine-grained weighting
   - Needs explicit training examples

ADVANTAGES:
   - Interpretable patterns (visualizable)
   - Memory efficient (binary decisions)
   - Discrete = no floating point errors
""")
print("="*70)
