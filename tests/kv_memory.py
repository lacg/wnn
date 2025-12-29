#!/usr/bin/env python3

from wnn.ram.decoders import OutputMode

from random import Random
from wnn.ram import RAMTransformer
from wnn.ram.architecture import KVSpec

from datetime import datetime

from torch import cat

def run() -> float:
	spec = KVSpec(k_bits=3, v_bits=2, query_value=0)
	rng = Random(123)

	model = RAMTransformer(
		spec=spec,
		neurons_per_head=8,
		n_bits_per_state_neuron=8,
		n_bits_per_output_neuron=4,
		use_hashing=True,
		hash_size=4096,
		rng=42,
		max_iters=100,
	)

	# ----------------------------
	# Train
	# ----------------------------
	print(f"\n=== Training model ===\n")

	epochs = 300
	width = len(str(epochs))
	for epoch in range(epochs):
		windows, target_bits, _ = spec.generate_episode(n_writes=6, rng=rng)
		model.train(windows, target_bits)
		print(f"\rEpoch {epoch+1:0{width}d} done.", end="", flush=True)


	# ----------------------------
	# Test
	# ----------------------------
	print(f"\n=== Testing model ===\n")

	epochs = test_episodes = 200
	correct = 0
	width = len(str(epochs))
	for _ in range(epochs):
		windows, target_bits, _ = spec.generate_episode(n_writes=6, rng=rng)
		out_bits = model.forward(cat([w[0] for w in windows], dim=0).reshape(1, -1))  # depends on your forward() API
		print(f"\rEpoch {epoch+1:0{width}d} done.", end="", flush=True)
		# If your forward expects the full episode bits as [1, n_total_bits], the above is correct.
		# Otherwise: run through windows and read the last output_layer_output.

		pred = spec.decode_value_bits(out_bits)
		expected = int("".join(str(int(b)) for b in target_bits[0].tolist()), 2)

		if pred == expected:
			correct += 1

	accuracy = correct / test_episodes

	print(model)		# calls __str__()
	return accuracy


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
