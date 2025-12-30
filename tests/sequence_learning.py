#!/usr/bin/env python3
"""
Test RAM-based sequence learning with discrete tokens.
Tests pattern learning (not memorization) using alphabetic sequences.
"""

import sys
sys.path.insert(0, 'src')

from wnn.ram.RAMSequence import RAMSequence
from wnn.ram.encoders_decoders import OutputMode
from wnn.ram.encoders_decoders import TransformerDecoderFactory
from torch import cat

def test_next_character():
	"""Test if model can learn 'next character' pattern."""

	print("="*60)
	print("Testing: Next Character Prediction")
	print("="*60)

	# Setup input encoder (for encoding input sequences)
	input_decoder = TransformerDecoderFactory.create(OutputMode.TOKEN_LIST)
	bits_per_token = input_decoder.bits_per_token  # 5 bits for A-Z

	# Only need 5 state neurons (2^5 = 32 states, enough for 26 letters)
	# Use FULL connectivity: each neuron sees all input + all state bits
	from torch import tensor
	from torch import long as tlong
	n_state_neurons = 5
	n_bits_per_state_neuron = bits_per_token + n_state_neurons  # 5 + 5 = 10 (fully connected)

	# Create full connectivity: each neuron sees [all 5 token bits, all 5 state bits]
	total_input_bits = bits_per_token + n_state_neurons  # 10 bits total
	state_connections = []
	for i in range(n_state_neurons):
		# Full connectivity: [0,1,2,3,4,5,6,7,8,9]
		connections = list(range(total_input_bits))
		state_connections.append(connections)

	state_connections_tensor = tensor(state_connections, dtype=tlong)

	# Use RAMSequence with full connectivity
	model = RAMSequence(
		input_bits=bits_per_token,
		n_state_neurons=n_state_neurons,
		n_output_neurons=bits_per_token,
		n_bits_per_state_neuron=n_bits_per_state_neuron,
		n_bits_per_output_neuron=bits_per_token,  # Output neurons also fully connected
		output_mode=OutputMode.TOKEN,
		use_hashing=False,
		rng=42,
	)

	# Override state layer connections for full connectivity
	model.state_layer.memory.connections = state_connections_tensor

	# Override output layer connections for full connectivity (sees all 5 state bits)
	output_connections = []
	for i in range(bits_per_token):
		connections = list(range(n_state_neurons))
		output_connections.append(connections)
	model.output_layer.memory.connections = tensor(output_connections, dtype=tlong)

	# Training sequences: learn "next letter in alphabet" pattern
	training_data = [
		("ABCD", "BCDE"),
		("EFGH", "FGHI"),
		("MNOP", "NOPQ"),
		("STUV", "TUVW"),
		("WXYZ", "XYZA"),  # Wrap around
	]

	print(f"\nTraining on {len(training_data)} sequences:")
	for input_seq, target_seq in training_data:
		print(f"  {input_seq} → {target_seq}")

	# DEBUG: Check connections
	print(f"\nDEBUG: Checking state layer connections...")
	print(f"State layer total input bits: {model.state_layer.memory.total_input_bits}")
	print(f"  Token bits: 0-{bits_per_token-1}")
	print(f"  State bits: {bits_per_token}-{model.state_layer.memory.total_input_bits-1}")
	print(f"\nFirst 5 neurons connections:")
	for i in range(min(5, model.state_layer.memory.num_neurons)):
		connections = model.state_layer.memory.connections[i].tolist()
		token_bits = [c for c in connections if c < bits_per_token]
		state_bits = [c for c in connections if c >= bits_per_token]
		print(f"  Neuron {i}: {connections}")
		print(f"    -> Sees {len(token_bits)} token bits, {len(state_bits)} state bits")

	# Train for multiple epochs
	epochs = 50
	print(f"\nTraining for {epochs} epochs...")

	for epoch in range(epochs):
		for seq_idx, (input_str, target_str) in enumerate(training_data):
			# Encode input sequence to list of token tensors
			input_windows = input_decoder.encode(input_str)

			# Debug first sequence in first epoch
			if epoch == 0 and seq_idx == 0:
				print(f"\n[DEBUG] Before training {input_str} → {target_str}")
				input_bits = cat([w.squeeze(0) for w in input_windows])
				pred_before = model.forward(input_bits)
				print(f"  Prediction: {pred_before} (expected {target_str[-1]})")

			# Train on sequence-to-sequence
			model.train(input_windows, target_str)

			# Debug first sequence in first epoch
			if epoch == 0 and seq_idx == 0:
				print(f"[DEBUG] After training {input_str} → {target_str}")
				pred_after = model.forward(input_bits)
				print(f"  Prediction: {pred_after} (expected {target_str[-1]})")

		if (epoch + 1) % 10 == 0:
			print(f"  Epoch {epoch + 1}/{epochs}")

	# Test on training sequences
	print(f"\nTesting on training sequences:")
	train_correct = 0
	train_total = len(training_data)

	for input_str, target_str in training_data:
		# Encode input and get prediction for last character
		input_windows = input_decoder.encode(input_str)
		# forward() needs concatenated bits - create from windows
		input_bits = cat([w.squeeze(0) for w in input_windows])
		predicted_char = model.forward(input_bits)  # decoder returns single char

		expected_char = target_str[-1]
		if predicted_char == expected_char:
			train_correct += 1

		match = "✓" if predicted_char == expected_char else "✗"
		print(f"  {input_str} → {predicted_char} (expected {expected_char}) {match}")

	train_acc = train_correct / train_total if train_total > 0 else 0
	print(f"\nTraining accuracy: {train_correct}/{train_total} = {train_acc:.1%}")

	# Test on NEW sequences (generalization)
	test_data = [
		("IJKL", "JKLM"),
		("QRST", "RSTU"),
		("CDEF", "DEFG"),
	]

	print(f"\nTesting on NEW sequences (generalization):")
	test_correct = 0
	test_total = len(test_data)

	for input_str, target_str in test_data:
		# Encode input and get prediction for last character
		input_windows = input_decoder.encode(input_str)
		input_bits = cat([w.squeeze(0) for w in input_windows])
		predicted_char = model.forward(input_bits)

		expected_char = target_str[-1]
		if predicted_char == expected_char:
			test_correct += 1

		match = "✓" if predicted_char == expected_char else "✗"
		print(f"  {input_str} → {predicted_char} (expected {expected_char}) {match}")

	test_acc = test_correct / test_total if test_total > 0 else 0
	print(f"\nGeneralization accuracy: {test_correct}/{test_total} = {test_acc:.1%}")

	# Summary
	print(f"\n" + "="*60)
	if train_acc > 0.8 and test_acc > 0.5:
		print("✅ SUCCESS: Model learned the pattern and generalizes!")
	elif train_acc > 0.8:
		print("⚠️  PARTIAL: Model memorized training but doesn't generalize")
	else:
		print("❌ FAILURE: Model failed to learn the pattern")
	print("="*60)


if __name__ == "__main__":
	test_next_character()
