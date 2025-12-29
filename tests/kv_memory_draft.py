#!/usr/bin/env python3

from wnn.ram.decoders import OutputMode
from wnn.ram import RAMRecurrentNetwork

from torch import cat
from torch import uint8
from torch import manual_seed
from torch import no_grad
from torch import tensor
from torch import Tensor
from torch import zeros

from datetime import datetime

start = datetime.now()
test_name = "KV recall"

print(f"\n=== Starting {test_name} test at {start} ===\n")

# ------------------------------------------------------------
# Helpers: bit encoders
# ------------------------------------------------------------

def bits_to_int(bits):
  return int("".join("1" if b else "0" for b in bits), 2)

def int_to_bits(x: int, n_bits: int) -> Tensor:
	"""MSB-first bit encoding."""
	return tensor([(x >> i) & 1 for i in reversed(range(n_bits))], dtype=uint8)

def make_kv_sequence(pairs, query_key, k_bits, v_bits):
	"""
	Build a KV episode:
		- Each (k, v) is a WRITE
		- Final window is QUERY: (k, 0)
	"""
	windows = []

	for k, v in pairs:
		window = cat([int_to_bits(k, k_bits), int_to_bits(v, v_bits)])
		windows.append(window.unsqueeze(0))

	# QUERY window: v == 0
	query_window = cat([int_to_bits(query_key, k_bits), zeros(v_bits, dtype=uint8)])
	windows.append(query_window.unsqueeze(0))

	return windows

# ------------------------------------------------------------
# Test
# ------------------------------------------------------------

def test_kv_recall_single_query():
	manual_seed(0)

	# ----------------------------
	# Token sizes
	# ----------------------------
	k_bits = 3		# 8 keys
	v_bits = 2		# 3 values + 00=query

	# ----------------------------
	# Model
	# ----------------------------
	model = RAMRecurrentNetwork(
			input_bits=k_bits + v_bits,
			n_state_neurons=4,
			n_output_neurons=v_bits,
			n_bits_per_state_neuron=8,
			n_bits_per_output_neuron=4,
			max_iters=50,
			output_mode=OutputMode.RAW
	)

	# ----------------------------
	# Training data
	# ----------------------------
	# Write sequence:
	#		k1 -> v2
	#		k3 -> v1
	#		k1 -> v3   (overwrite)
	#
	# Query:
	#		k1 => expect v3
	#
	pairs = [
		(1, 2),
		(3, 1),
		(1, 3),
	]

	query_key = 1
	target_value = 3

	windows = make_kv_sequence(pairs, query_key, k_bits, v_bits)
	target_bits = int_to_bits(target_value, v_bits).unsqueeze(0)

	# ----------------------------
	# Train
	# ----------------------------
	print(f"\n=== Training model ===\n")

	epochs = 200
	width = len(str(epochs))

	for epoch in range(epochs):
		model.train(windows, target_bits)
		print(f"\rEpoch {epoch+1:0{width}d} done.", end="", flush=True)

	# ----------------------------
	# Test
	# ----------------------------
	print(f"\n=== Testing model ===\n")

	with no_grad():
		test_result = model.forward(windows[-1][0])

	test_result = bits_to_int(test_result[0])
	expected_result = target_value

	print(model)		# calls __str__()

	print(f"{test_name} Test Summary")
	print("====================")
	print(f"Query key: {query_key}")
	print(f"Expected value: {expected_result}")
	print(f"Predicted value: {test_result}")

	end = datetime.now()
	print(f"✅ {test_name} succeed!" if test_result == expected_result else f"❌ {test_name} failed!")
	print(f"\n=== End {test_name} at {end} ===")
	print(f"\n=== Duration: {end - start} ===\n")

if __name__ == "__main__":
	test_kv_recall_single_query()