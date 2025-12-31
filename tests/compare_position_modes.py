#!/usr/bin/env python3
"""
Compare Position Encoding Modes for Attention Learning

Demonstrates how RELATIVE position encoding makes positional
patterns like "copy previous" much easier to learn than BINARY.

Key insight:
  BINARY mode: Must learn (query_pos=1, key_pos=0) -> attend
                         (query_pos=2, key_pos=1) -> attend
                         (query_pos=3, key_pos=2) -> attend
               (many separate rules)

  RELATIVE mode: Just learn (distance=-1) -> attend
                 (single rule, generalizes to all positions)
"""

import sys
sys.path.insert(0, 'src')

from wnn.ram.RAMLayer import RAMLayer
from wnn.ram.encoders_decoders import (
	TransformerDecoderFactory, OutputMode,
	PositionEncoderFactory, PositionMode
)
from torch import tensor, uint8, zeros, cat

print("=" * 70)
print("Comparing Position Encoding Modes")
print("=" * 70)

# Setup
decoder = TransformerDecoderFactory.create(OutputMode.TOKEN)
bits_per_token = decoder.bits_per_token

def encode_sequence(text: str) -> list:
	"""Encode text to list of bit tensors."""
	return [decoder.encode(c).squeeze() for c in text]


def train_and_evaluate(pos_mode: PositionMode, task_name: str, epochs: int = 50):
	"""Train a single attention layer and evaluate on the copy-previous task."""

	print(f"\n{'=' * 70}")
	print(f"Position Mode: {pos_mode.name}")
	print(f"Task: {task_name}")
	print("=" * 70)

	# Create position encoder
	if pos_mode == PositionMode.BINARY:
		pos_encoder = PositionEncoderFactory.create(PositionMode.BINARY, max_seq_len=16)
		# Input: [query, key, query_pos, key_pos]
		attn_input_size = 2 * bits_per_token + 2 * pos_encoder.n_bits
		print(f"Input: [query({bits_per_token}b), key({bits_per_token}b), "
			  f"query_pos({pos_encoder.n_bits}b), key_pos({pos_encoder.n_bits}b)]")
		print(f"Total: {attn_input_size} bits")
	else:  # RELATIVE
		pos_encoder = PositionEncoderFactory.create(PositionMode.RELATIVE, max_distance=15)
		# Input: [query, key, relative_distance]
		attn_input_size = 2 * bits_per_token + pos_encoder.n_bits
		print(f"Input: [query({bits_per_token}b), key({bits_per_token}b), "
			  f"distance({pos_encoder.n_bits}b)]")
		print(f"Total: {attn_input_size} bits")

	# Create attention layer
	attn_layer = RAMLayer(
		total_input_bits=attn_input_size,
		num_neurons=1,  # Binary: attend or not
		n_bits_per_neuron=min(attn_input_size, 12),
		rng=42,
	)

	# Training sequences
	sequences = ["ABCD", "WXYZ", "HELLO", "TEST", "QUICK", "BROWN"]

	# Build attention input based on mode
	def build_attn_input(q_idx: int, k_idx: int, tokens: list):
		q = tokens[q_idx]
		k = tokens[k_idx]

		if pos_mode == PositionMode.BINARY:
			q_pos = pos_encoder.encode(q_idx)
			k_pos = pos_encoder.encode(k_idx)
			return cat([q, k, q_pos, k_pos]).unsqueeze(0)
		else:  # RELATIVE
			# distance = key_pos - query_pos (negative for past positions)
			rel_dist = pos_encoder.encode_relative(q_idx, k_idx)
			return cat([q, k, rel_dist]).unsqueeze(0)

	# Target for "copy previous"
	def target_attend(q_idx: int, k_idx: int) -> int:
		if q_idx == 0:
			return 1 if k_idx == 0 else 0  # First pos attends to self
		return 1 if k_idx == q_idx - 1 else 0  # Otherwise attend to previous

	# Training
	print(f"\nTraining for {epochs} epochs on {len(sequences)} sequences...")

	errors_per_epoch = []
	for epoch in range(epochs):
		errors = 0
		total = 0

		for seq in sequences:
			tokens = encode_sequence(seq)
			seq_len = len(tokens)

			for i in range(seq_len):  # Query position
				for j in range(i + 1):  # Key position (causal)
					target = target_attend(i, j)
					attn_input = build_attn_input(i, j, tokens)

					output = attn_layer(attn_input)
					total += 1

					if output.item() != target:
						errors += 1
						target_tensor = tensor([[target]], dtype=uint8)
						attn_layer.commit(attn_input, target_tensor)

		errors_per_epoch.append(errors)

		if epoch == 0 or (epoch + 1) % 10 == 0:
			print(f"  Epoch {epoch + 1:3d}: {errors:3d} errors / {total} pairs "
				  f"({100 * (1 - errors/total):.1f}% correct)")

	# Evaluation on training sequences
	print(f"\nEvaluation on training sequences:")
	for seq in sequences[:3]:
		tokens = encode_sequence(seq)
		print(f"\n  '{seq}':")
		visualize_attention(attn_layer, tokens, pos_encoder, pos_mode, build_attn_input)

	# Generalization to NEW sequences
	print(f"\nGeneralization to NEW sequences:")
	for seq in ["NOVEL", "XYZ"]:
		tokens = encode_sequence(seq)
		print(f"\n  '{seq}' (never seen):")
		visualize_attention(attn_layer, tokens, pos_encoder, pos_mode, build_attn_input)

	# Count final accuracy
	correct = 0
	total = 0
	for seq in ["NOVEL", "XYZ", "FRESH", "BRAND"]:
		tokens = encode_sequence(seq)
		for i in range(len(tokens)):
			for j in range(i + 1):
				target = target_attend(i, j)
				attn_input = build_attn_input(i, j, tokens)
				output = attn_layer(attn_input)
				if output.item() == target:
					correct += 1
				total += 1

	accuracy = 100 * correct / total
	print(f"\n  Generalization accuracy: {correct}/{total} = {accuracy:.1f}%")

	return accuracy, errors_per_epoch


def visualize_attention(attn_layer, tokens, pos_encoder, pos_mode, build_attn_fn):
	"""Visualize attention pattern."""
	seq_len = len(tokens)
	print("      " + " ".join(f"{j:2d}" for j in range(seq_len)))

	for i in range(seq_len):
		row = f"   {i:2d}: "
		for j in range(seq_len):
			if j > i:  # Causal mask
				row += " - "
				continue

			attn_input = build_attn_fn(i, j, tokens)
			attend = attn_layer(attn_input).squeeze().item()
			row += " # " if attend else " . "
		print(row)


# Run comparison
print("\n" + "=" * 70)
print("COPY PREVIOUS TASK")
print("=" * 70)
print("""
Target pattern:
    0  1  2  3
 0: #  -  -  -    (position 0 attends to self)
 1: #  .  -  -    (position 1 attends to 0)
 2: .  #  .  -    (position 2 attends to 1)
 3: .  .  #  .    (position 3 attends to 2)

This is "attend to j when j == i-1" (or self when i==0).
""")

# Compare both modes
binary_acc, binary_errors = train_and_evaluate(PositionMode.BINARY, "Copy Previous", epochs=50)
relative_acc, relative_errors = train_and_evaluate(PositionMode.RELATIVE, "Copy Previous", epochs=50)

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
Position Mode   | Generalization Accuracy
----------------|------------------------
BINARY          | {binary_acc:.1f}%
RELATIVE        | {relative_acc:.1f}%

Why RELATIVE is better for positional patterns:

  BINARY must learn separate rules for each (query_pos, key_pos) pair:
    (1, 0) -> attend
    (2, 1) -> attend
    (3, 2) -> attend
    ...etc for all positions in training data

  RELATIVE learns ONE rule:
    (distance = -1) -> attend

  This single rule generalizes to ANY sequence length!

  However, BINARY is better for content-based patterns like
  "attend to position 0" because the absolute position matters.
""")
print("=" * 70)
