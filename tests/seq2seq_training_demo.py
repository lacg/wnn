#!/usr/bin/env python3
"""
RAM Seq2Seq Training Demo

Demonstrates training the transformer-like architecture on:
1. Next character prediction (language modeling)
2. Copy task (identity mapping)
3. Shift task (output = input shifted by 1)
"""

import sys
sys.path.insert(0, 'src')

from wnn.ram.RAMSeq2Seq import RAMSeq2Seq
from wnn.ram.encoders_decoders import (
	TransformerDecoderFactory, OutputMode, PositionMode
)
from torch import tensor, uint8

print("=" * 70)
print("RAM Seq2Seq Training Demo")
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
# Task 1: Next Character Prediction
# ============================================================
print("=" * 70)
print("Task 1: Next Character Prediction")
print("=" * 70)
print("""
Train to predict the next character in a sequence.
  Input:  "ABCD"
  Target: "BCDE" (shifted by 1)

This is the core language modeling task.
""")

# Create dataset
training_sequences = [
	"ABCDE",
	"FGHIJ",
	"KLMNO",
	"PQRST",
	"UVWXY",
]

def make_next_char_dataset(sequences):
	"""Create (input, target) pairs for next-char prediction."""
	dataset = []
	for seq in sequences:
		# Input is seq[:-1], target is seq[1:]
		input_str = seq[:-1]
		target_str = seq[1:]
		input_tokens = encode_sequence(input_str)
		target_tokens = encode_sequence(target_str)
		dataset.append((input_tokens, target_tokens))
	return dataset

dataset = make_next_char_dataset(training_sequences)
print(f"Training examples: {len(dataset)}")
for inp, tgt in dataset[:2]:
	print(f"  '{decode_sequence(inp)}' -> '{decode_sequence(tgt)}'")
print()

# Create model
model = RAMSeq2Seq(
	input_bits=bits_per_token,
	num_layers=2,
	num_heads=4,
	position_mode=PositionMode.RELATIVE,
	max_seq_len=16,
	use_residual=True,
	rng=42,
)

print(f"Model: {model}")
print()

# Test before training
print("Before training:")
for inp, tgt in dataset[:2]:
	outputs = model.forward(inp)
	print(f"  Input:  '{decode_sequence(inp)}'")
	print(f"  Target: '{decode_sequence(tgt)}'")
	print(f"  Output: '{decode_sequence(outputs)}'")
	print()

# Train
print("Training...")
history = model.train(dataset, epochs=20, verbose=True)
print()

# Test after training
print("After training:")
for inp, tgt in dataset[:2]:
	outputs = model.forward(inp)
	correct = decode_sequence(outputs) == decode_sequence(tgt)
	status = "correct" if correct else "wrong"
	print(f"  Input:  '{decode_sequence(inp)}'")
	print(f"  Target: '{decode_sequence(tgt)}'")
	print(f"  Output: '{decode_sequence(outputs)}' ({status})")
	print()

# Test generalization
print("Generalization (unseen sequences):")
test_sequences = ["MNOP", "RSTU"]
for seq in test_sequences:
	input_str = seq[:-1]
	target_str = seq[1:]
	inp = encode_sequence(input_str)
	tgt = encode_sequence(target_str)
	outputs = model.forward(inp)
	correct = decode_sequence(outputs) == target_str
	status = "correct" if correct else "wrong"
	print(f"  '{input_str}' -> '{decode_sequence(outputs)}' (expected '{target_str}') [{status}]")
print()


# ============================================================
# Task 2: Copy Task
# ============================================================
print("=" * 70)
print("Task 2: Copy Task (Identity)")
print("=" * 70)
print("""
Train to output the same sequence as input.
  Input:  "ABC"
  Target: "ABC"

Tests information preservation through attention layers.
""")

copy_sequences = ["ABC", "XYZ", "DOG", "CAT", "SUN"]

def make_copy_dataset(sequences):
	"""Create (input, target) pairs for copy task."""
	dataset = []
	for seq in sequences:
		tokens = encode_sequence(seq)
		dataset.append((tokens, tokens))  # Input = Target
	return dataset

copy_dataset = make_copy_dataset(copy_sequences)

# Create fresh model
copy_model = RAMSeq2Seq(
	input_bits=bits_per_token,
	num_layers=1,  # Shallow for copy
	num_heads=2,
	position_mode=PositionMode.RELATIVE,
	max_seq_len=16,
	use_residual=True,
	rng=123,
)

print(f"Model: {copy_model}")
print()

# Test before training
print("Before training:")
for inp, tgt in copy_dataset[:2]:
	outputs = copy_model.forward(inp)
	print(f"  '{decode_sequence(inp)}' -> '{decode_sequence(outputs)}'")
print()

# Train
print("Training...")
history = copy_model.train(copy_dataset, epochs=15, verbose=True)
print()

# Test after training
print("After training:")
for inp, tgt in copy_dataset:
	outputs = copy_model.forward(inp)
	correct = decode_sequence(outputs) == decode_sequence(tgt)
	status = "ok" if correct else "ERR"
	print(f"  '{decode_sequence(inp)}' -> '{decode_sequence(outputs)}' [{status}]")
print()


# ============================================================
# Task 3: Autoregressive Generation
# ============================================================
print("=" * 70)
print("Task 3: Autoregressive Generation")
print("=" * 70)
print("""
Use the trained next-char model for generation.
""")

# Use the model trained on next-char prediction
prompt = "AB"
print(f"Prompt: '{prompt}'")
prompt_tokens = encode_sequence(prompt)

print("Generating 3 tokens...")
full_sequence = model.generate(
	prompt_tokens,
	max_new_tokens=3,
	decoder=decoder
)

print(f"\nFull sequence: '{decode_sequence(full_sequence)}'")
print()


# ============================================================
# Summary
# ============================================================
print("=" * 70)
print("Summary")
print("=" * 70)
print("""
Training Results:

1. Next Character Prediction:
   - Model learns sequential patterns
   - Uses EDRA to train attention and output layers
   - Generalization depends on pattern coverage

2. Copy Task:
   - Tests information flow through layers
   - Residual connections (XOR) help preserve input
   - Shallow networks work well for identity

3. Autoregressive Generation:
   - Uses trained model to generate tokens
   - Each generated token becomes input for next

Training Method (EDRA through layers):
  1. Forward pass records intermediate states
  2. Compare output to target, find errors
  3. Train output layer to produce correct output
  4. Backpropagate through attention layers
  5. Train each layer's RAM components

This is different from backprop:
  - No gradients (discrete neurons)
  - Direct commitment of correct mappings
  - Layer-by-layer constraint solving
""")
print("=" * 70)
