#!/usr/bin/env python3

from wnn.ram.factories import ModelsFactory
from wnn.ram.enums import ModelType
from wnn.ram.encoders_decoders import OutputMode

from random import Random
from wnn.ram.architecture import KVSpec

from datetime import datetime

from torch import cat

def run() -> float:
	# Simpler spec: 2 keys (k_bits=1), 2 values (v_bits=1, where 0=query, 1=store)
	# This creates 2 heads with simpler patterns to learn
	spec = KVSpec(k_bits=2, v_bits=2, query_value=0)
	rng = Random(123)

	# More bits per neuron to see more of the input pattern
	# With window_bits=4 and state=32 (4 heads * 8 neurons), total=36 input bits
	# State neurons should see ALL window bits + some state bits
	model = ModelsFactory.create(
		ModelType.KV_MEMORY,
		spec=spec,
		neurons_per_head=8,
		n_bits_per_state_neuron=10,  # See all 4 window bits + 6 state bits
		n_bits_per_output_neuron=6,  # See all 2 key bits + 4 state bits
		use_hashing=True,
		hash_size=4096,
		rng=42,
		max_iters=100,
		output_mode=OutputMode.RAW,  # Multi-bit values require RAW, not HAMMING
	)

	# ----------------------------
	# Generate exhaustive training set
	# ----------------------------
	# RAM networks are lookup-based, so we need to train on all possible
	# key-value patterns for generalization. With k_bits=3 (8 keys) and
	# v_bits=2 (values 1,2,3 - 0 is query), we have 8*3=24 write patterns.
	#
	# Strategy: Create episodes that cover all key-value combinations
	# by systematically writing each key with each possible value.

	print(f"\n=== Generating exhaustive training episodes ===\n")

	training_episodes = []
	num_keys = 1 << spec.k_bits      # 8
	num_values = (1 << spec.v_bits) - 1  # 3 (exclude query_value=0)

	# Create episodes: for each key, write a value and query it
	for key in range(num_keys):
		for value in range(1, num_values + 1):  # values 1, 2, 3
			# Simple episode: one write, one query
			windows = [
				spec.encode_window(key, value),      # Write
				spec.encode_window(key, spec.query_value),  # Query
			]
			target = spec.oracle_last_write_value([(key, value), (key, spec.query_value)])
			from torch import tensor, uint8
			target_bits = tensor(KVSpec.int_to_bits(target, spec.v_bits), dtype=uint8).unsqueeze(0)
			training_episodes.append((windows, target_bits))

	print(f"Created {len(training_episodes)} exhaustive training episodes")

	# ----------------------------
	# Train
	# ----------------------------
	print(f"\n=== Training model ===\n")

	epochs = 20  # Multiple passes over the exhaustive set
	width = len(str(epochs))
	for epoch in range(epochs):
		# Shuffle training order each epoch
		shuffled = training_episodes.copy()
		rng.shuffle(shuffled)
		for windows, target_bits in shuffled:
			model.train(windows, target_bits)
		print(f"\rEpoch {epoch+1:0{width}d} done.", end="", flush=True)

	# ----------------------------
	# Test on trained patterns
	# ----------------------------
	print(f"\n\n=== Testing on trained patterns ===\n")

	correct = 0
	for windows, target_bits in training_episodes:
		# Process windows sequentially (write then query)
		model._reset_state(batch_size=1, device=windows[0][0].device)
		for window in windows[:-1]:  # All but last (writes)
			model._get_outputs(window[0].unsqueeze(0), update_state=True)
		# Last window is query - get output
		*_, out_bits = model._get_outputs(windows[-1][0].unsqueeze(0), update_state=False)
		out_bits = model.decoder.decode(out_bits)

		pred = spec.decode_value_bits(out_bits)
		expected = int("".join(str(int(b)) for b in target_bits[0].tolist()), 2)
		if pred == expected:
			correct += 1

	train_accuracy = correct / len(training_episodes)
	print(f"Training set accuracy: {train_accuracy:.1%} ({correct}/{len(training_episodes)})")

	# ----------------------------
	# Test on novel episodes (multi-write sequences)
	# ----------------------------
	print(f"\n=== Testing on novel multi-write episodes ===\n")

	test_episodes = 100
	correct = 0
	for i in range(test_episodes):
		windows, target_bits, _ = spec.generate_episode(n_writes=4, rng=rng)
		# Process windows sequentially (writes then query)
		model._reset_state(batch_size=1, device=windows[0][0].device)
		for window in windows[:-1]:  # All but last (writes)
			model._get_outputs(window[0].unsqueeze(0), update_state=True)
		# Last window is query - get output
		*_, out_bits = model._get_outputs(windows[-1][0].unsqueeze(0), update_state=False)
		out_bits = model.decoder.decode(out_bits)
		print(f"\rTest {i+1:03d}/{test_episodes} done.", end="", flush=True)

		pred = spec.decode_value_bits(out_bits)
		expected = int("".join(str(int(b)) for b in target_bits[0].tolist()), 2)

		if pred == expected:
			correct += 1

	novel_accuracy = correct / test_episodes
	print(f"\nNovel episode accuracy: {novel_accuracy:.1%} ({correct}/{test_episodes})")

	print(model)
	return train_accuracy  # Return training accuracy as the primary metric


if __name__ == "__main__":
	start = datetime.now()
	test_name = "KV recall"

	print(f"\n=== Starting {test_name} test at {start} ===\n")

	accuracy = run()

	print(f"{test_name} Test Summary")
	print("====================")
	print(f"KV recall accuracy: {accuracy:.3f}")
	print(f"✅ {test_name} succeed!" if accuracy > 0.80 else f"❌ {test_name} failed!")

	end = datetime.now()
	print(f"\n=== End {test_name} at {end} ===")
	print(f"\n=== Duration: {end - start} ===\n")
