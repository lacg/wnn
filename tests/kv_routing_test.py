#!/usr/bin/env python3
"""Test KV routing in RAMMultiHeadSequence."""

import sys
sys.path.insert(0, 'src')

from wnn.ram.RAMMultiHeadSequence import RAMMultiHeadSequence
from wnn.ram.encoders_decoders import OutputMode, TransformerDecoderFactory
from wnn.ram.cost import CostCalculatorType
from torch import cat

print("="*60)
print("KV Routing Test")
print("="*60)

# Setup
input_decoder = TransformerDecoderFactory.create(OutputMode.TOKEN_LIST)
bits_per_token = input_decoder.bits_per_token

print(f"\nBits per token: {bits_per_token}")

# Test configurations: compare all routing modes
# Format: (name, num_heads, k_bits, key_position, selective, learned_router)
configs = [
	("4-Head VOTE", 4, 0, "last", False, False),
	("4-Head KV ensemble", 4, 2, "last", False, False),
	("4-Head KV selective", 4, 2, "last", True, False),
	("4-Head Learned", 4, 0, "last", False, True),
	("4-Head Learned+Sel", 4, 0, "last", True, True),
]

# Training data
training_data = [
	("ABCD", "BCDE"),
	("EFGH", "FGHI"),
	("IJKL", "JKLM"),
	("MNOP", "NOPQ"),
	("QRST", "RSTU"),
	("UVWX", "VWXY"),
	("YZAB", "ZABC"),
]

# Test data
test_data = [
	("CDEF", "DEFG"),
	("NOPQ", "OPQR"),
	("STUV", "TUVW"),
]

results = []

for config_name, num_heads, k_bits, key_position, selective, learned_router in configs:
	print(f"\n{'='*60}")
	print(f"Testing: {config_name}")
	print(f"{'='*60}")

	# Create model with appropriate routing mode
	model = RAMMultiHeadSequence(
		num_heads=num_heads,
		input_bits=bits_per_token,
		vocab_size=26,
		output_mode=OutputMode.TOKEN,
		k_bits=k_bits,
		key_position=key_position,
		selective_training=selective,
		use_learned_router=learned_router,
		cost_calculator_type=CostCalculatorType.VOTE,
		use_hashing=False,
		rng=42,
	)

	print(f"  Model: {model}")
	print(f"  State neurons/head: {model.n_state_neurons_per_head}")
	print(f"  KV routing: {model.use_kv_routing}")
	if model.use_kv_routing:
		print(f"  Key bits: {model.k_bits}, position: {model.key_position}")

	# Train
	epochs = 30
	print(f"\nTraining for {epochs} epochs...")
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
print(f"{'Configuration':<25} {'Train Acc':<12} {'Test Acc':<12}")
print("-" * 60)
for config_name, train_acc, test_acc in results:
	print(f"{config_name:<25} {train_acc:>11.1%} {test_acc:>11.1%}")

# Verify KV routing is functioning
print(f"\n{'='*60}")
print("KV Routing Verification")
print(f"{'='*60}")

# Create a model and check that different inputs route to different heads
model = RAMMultiHeadSequence(
	num_heads=4,
	input_bits=bits_per_token,
	vocab_size=26,
	output_mode=OutputMode.TOKEN,
	k_bits=2,
	key_position="last",  # Use last bits where entropy is
	use_hashing=False,
	rng=42,
)

print("\nInput → Head routing (last 2 bits of token):")
print("  Token encodings use index: A=0, B=1, etc.")
print("  Last 2 bits capture the varying part of the encoding\n")
for char in "ABCDEFGH":
	input_windows = input_decoder.encode(char)
	input_bits = input_windows[0].squeeze(0)
	head_idx = model._extract_key(input_bits)
	all_bits = ''.join(str(int(b)) for b in input_bits)
	last_bits = ''.join(str(int(b)) for b in input_bits[-2:])
	print(f"  '{char}' → {all_bits} → last2={last_bits} → Head {head_idx}")

print("="*60)
