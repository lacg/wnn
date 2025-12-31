#!/usr/bin/env python3
"""
Compare Shared-State vs Ensemble multi-head architectures.

Shared-State (RAMMultiHeadShared):
- ONE shared state layer
- Multiple output heads interpret same state
- Unified temporal context

Ensemble (RAMMultiHeadSequence):
- Multiple independent RAMSequence heads
- Each has its own state evolution
- Fragmented memory across heads
"""

import sys
sys.path.insert(0, 'src')

from wnn.ram.RAMMultiHeadSequence import RAMMultiHeadSequence
from wnn.ram.RAMMultiHeadShared import RAMMultiHeadShared
from wnn.ram.encoders_decoders import OutputMode, TransformerDecoderFactory
from wnn.ram.cost import CostCalculatorType
from torch import cat

print("="*70)
print("Shared-State vs Ensemble Architecture Comparison")
print("="*70)

# Setup
input_decoder = TransformerDecoderFactory.create(OutputMode.TOKEN_LIST)
bits_per_token = input_decoder.bits_per_token

# Training data - longer sequences to test memory
training_data = [
	("ABCDEFGH", "BCDEFGHI"),  # 8-step sequences
	("IJKLMNOP", "JKLMNOPQ"),
	("QRSTUVWX", "RSTUVWXY"),
	("YZABCDEF", "ZABCDEFG"),
]

# Test data
test_data = [
	("CDEFGHIJ", "DEFGHIJK"),
	("MNOPQRST", "NOPQRSTU"),
	("WXYZABCD", "XYZABCDE"),
]

print(f"\nBits per token: {bits_per_token}")
print(f"Training sequences: {len(training_data)} (8 chars each)")
print(f"Test sequences: {len(test_data)}")

# Configurations to compare
configs = [
	# (name, model_class, num_heads, n_state, k_bits, learned_router)
	("Ensemble-4H-5state", RAMMultiHeadSequence, 4, 5, 2, False),
	("Ensemble-4H-10state", RAMMultiHeadSequence, 4, 10, 2, False),
	("Shared-4H-5state", RAMMultiHeadShared, 4, 5, 2, False),
	("Shared-4H-10state", RAMMultiHeadShared, 4, 10, 2, False),
	("Shared-4H-10state-learned", RAMMultiHeadShared, 4, 10, 0, True),
]

results = []

for config_name, model_class, num_heads, n_state, k_bits, learned_router in configs:
	print(f"\n{'='*70}")
	print(f"Testing: {config_name}")
	print(f"{'='*70}")

	# Create model
	if model_class == RAMMultiHeadSequence:
		model = RAMMultiHeadSequence(
			num_heads=num_heads,
			input_bits=bits_per_token,
			vocab_size=26,
			n_state_neurons_per_head=n_state,
			n_output_neurons=bits_per_token,
			output_mode=OutputMode.TOKEN,
			k_bits=k_bits,
			key_position="last",
			selective_training=True,
			use_learned_router=learned_router,
			use_hashing=False,
			rng=42,
		)
	else:  # RAMMultiHeadShared
		model = RAMMultiHeadShared(
			num_heads=num_heads,
			input_bits=bits_per_token,
			n_state_neurons=n_state,
			n_output_neurons=bits_per_token,
			output_mode=OutputMode.TOKEN,
			k_bits=k_bits,
			key_position="last",
			use_learned_router=learned_router,
			use_hashing=False,
			rng=42,
		)

	print(f"  Model: {model}")

	# Train
	epochs = 50
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

	results.append((config_name, train_acc, test_acc, model_class.__name__))

# Summary
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"{'Configuration':<30} {'Type':<12} {'Train':<10} {'Test':<10}")
print("-" * 70)
for config_name, train_acc, test_acc, model_type in results:
	arch = "Shared" if "Shared" in model_type else "Ensemble"
	print(f"{config_name:<30} {arch:<12} {train_acc:>9.1%} {test_acc:>9.1%}")

# Analysis
print(f"\n{'='*70}")
print("ANALYSIS")
print(f"{'='*70}")

shared_results = [r for r in results if "Shared" in r[3]]
ensemble_results = [r for r in results if "Sequence" in r[3]]

if shared_results and ensemble_results:
	best_shared = max(shared_results, key=lambda x: (x[2], x[1]))
	best_ensemble = max(ensemble_results, key=lambda x: (x[2], x[1]))

	print(f"Best Shared:   {best_shared[0]} (Test: {best_shared[2]:.1%})")
	print(f"Best Ensemble: {best_ensemble[0]} (Test: {best_ensemble[2]:.1%})")

	if best_shared[2] > best_ensemble[2]:
		print("\n🏆 Shared-state architecture performs better!")
	elif best_shared[2] < best_ensemble[2]:
		print("\n🏆 Ensemble architecture performs better!")
	else:
		print("\n🏆 Both architectures perform equally!")

print("="*70)
