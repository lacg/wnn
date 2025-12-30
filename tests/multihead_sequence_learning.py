#!/usr/bin/env python3
"""
Test multi-head RAM-based sequence learning.
Compares single-head vs multi-head architectures on next-character prediction.
"""

import sys
sys.path.insert(0, 'src')

from wnn.ram.RAMSequence import RAMSequence
from wnn.ram.RAMMultiHeadSequence import RAMMultiHeadSequence
from wnn.ram.encoders_decoders import OutputMode
from wnn.ram.encoders_decoders import TransformerDecoderFactory
from wnn.ram.cost import CostCalculatorType
from torch import cat
from torch import tensor
from torch import long as tlong

def test_multihead():
	"""Compare single-head vs multi-head sequence learning."""

	print("="*60)
	print("Testing: Multi-Head vs Single-Head Sequence Learning")
	print("="*60)

	# Setup
	input_decoder = TransformerDecoderFactory.create(OutputMode.TOKEN_LIST)
	bits_per_token = input_decoder.bits_per_token

	# Training data - more comprehensive coverage
	training_data = [
		("ABCD", "BCDE"),
		("EFGH", "FGHI"),
		("IJKL", "JKLM"),  # Now includes I,J,K,L
		("MNOP", "NOPQ"),
		("QRST", "RSTU"),  # Now includes Q,R
		("UVWX", "VWXY"),
		("YZAB", "ZABC"),  # Wrap around
	]

	# Test data - completely new sequences
	test_data = [
		("CDEF", "DEFG"),
		("NOPQ", "OPQR"),
		("STUV", "TUVW"),
	]

	print(f"\nTraining on {len(training_data)} sequences")
	print(f"Testing on {len(test_data)} NEW sequences\n")

	# Common architecture params
	n_state_neurons = 5
	n_bits_per_state_neuron = bits_per_token + n_state_neurons
	total_input_bits = bits_per_token + n_state_neurons

	# Create full connectivity connections
	state_connections = []
	for i in range(n_state_neurons):
		connections = list(range(total_input_bits))
		state_connections.append(connections)
	state_connections_tensor = tensor(state_connections, dtype=tlong)

	output_connections = []
	for i in range(bits_per_token):
		connections = list(range(n_state_neurons))
		output_connections.append(connections)
	output_connections_tensor = tensor(output_connections, dtype=tlong)

	# Test configurations
	configs = [
		("Single-Head", 1),
		("3-Head", 3),
		("5-Head", 5),
	]

	results = []

	for config_name, num_heads in configs:
		print(f"\n{'='*60}")
		print(f"Testing {config_name} Configuration")
		print(f"{'='*60}")

		if num_heads == 1:
			# Single head (RAMSequence)
			model = RAMSequence(
				input_bits=bits_per_token,
				n_state_neurons=n_state_neurons,
				n_output_neurons=bits_per_token,
				n_bits_per_state_neuron=n_bits_per_state_neuron,
				n_bits_per_output_neuron=bits_per_token,
				output_mode=OutputMode.TOKEN,
				use_hashing=False,
				rng=42,
			)
			model.state_layer.memory.connections = state_connections_tensor
			model.output_layer.memory.connections = output_connections_tensor
		else:
			# Multi-head
			model = RAMMultiHeadSequence(
				num_heads=num_heads,
				input_bits=bits_per_token,
				n_state_neurons_per_head=n_state_neurons,
				n_output_neurons=bits_per_token,
				n_bits_per_state_neuron=n_bits_per_state_neuron,
				n_bits_per_output_neuron=bits_per_token,
				output_mode=OutputMode.TOKEN,
				routing_mode="vote",  # All heads vote
				cost_calculator_type=CostCalculatorType.VOTE,
				use_hashing=False,
				rng=42,
			)
			# Set connections for all heads
			for head in model.heads:
				head.state_layer.memory.connections = state_connections_tensor
				head.output_layer.memory.connections = output_connections_tensor

		# Train
		epochs = 30
		print(f"Training for {epochs} epochs...")
		for epoch in range(epochs):
			for input_str, target_str in training_data:
				input_windows = input_decoder.encode(input_str)
				model.train(input_windows, target_str)

			if (epoch + 1) % 10 == 0:
				print(f"  Epoch {epoch + 1}/{epochs}")

		# Test on training data
		train_correct = 0
		for input_str, target_str in training_data:
			input_windows = input_decoder.encode(input_str)
			input_bits = cat([w.squeeze(0) for w in input_windows])
			predicted = model.forward(input_bits)
			expected = target_str[-1]
			if predicted == expected:
				train_correct += 1

		train_acc = train_correct / len(training_data)

		# Test on new data
		test_correct = 0
		test_details = []
		for input_str, target_str in test_data:
			input_windows = input_decoder.encode(input_str)
			input_bits = cat([w.squeeze(0) for w in input_windows])
			predicted = model.forward(input_bits)
			expected = target_str[-1]
			match = predicted == expected
			if match:
				test_correct += 1
			test_details.append((input_str, predicted, expected, match))

		test_acc = test_correct / len(test_data)

		print(f"\nResults:")
		print(f"  Training accuracy: {train_correct}/{len(training_data)} = {train_acc:.1%}")
		print(f"  Test accuracy: {test_correct}/{len(test_data)} = {test_acc:.1%}")
		print(f"\nTest details:")
		for input_str, predicted, expected, match in test_details:
			symbol = "✓" if match else "✗"
			print(f"  {input_str} → {predicted} (expected {expected}) {symbol}")

		results.append((config_name, train_acc, test_acc))

	# Summary
	print(f"\n{'='*60}")
	print("SUMMARY")
	print(f"{'='*60}")
	print(f"{'Configuration':<15} {'Train Acc':<12} {'Test Acc':<12}")
	print("-" * 60)
	for config_name, train_acc, test_acc in results:
		print(f"{config_name:<15} {train_acc:>11.1%} {test_acc:>11.1%}")

	# Determine winner
	best_config = max(results, key=lambda x: (x[2], x[1]))
	print(f"\n🏆 Best: {best_config[0]} (Test: {best_config[2]:.1%}, Train: {best_config[1]:.1%})")
	print("="*60)

if __name__ == "__main__":
	test_multihead()
