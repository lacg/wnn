#!/usr/bin/env python3

from wnn.ram import RAMRecurrentNetwork

from datetime import datetime

import os
import sys
import torch

start = datetime.now()

print(f"\n=== Starting Parity Check Run at {start} ===\n")

# Add project root to PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

n = 12
epochs = 250
width = len(str(epochs))
# ----------------------------
# Build a RAMRecurrentNetwork
# ----------------------------
model = RAMRecurrentNetwork(
	input_bits=12,
	n_state_neurons=1,
	n_output_neurons=1,
	n_bits_per_state_neuron=13,
	n_bits_per_output_neuron=1,
	use_hashing=False,
	rng=None,
	max_iters=100,
)

# ----------------------------
# Build parity dataset (all 2 ** n examples)
# ----------------------------
xs = []
ys = []
for i in range(2 ** n):
	vec = [(i >> (n - 1 - b)) & 1 for b in range(n)]
	xs.append(vec)
	ys.append(sum(vec) % 2)  # odd parity

xs = torch.tensor(xs, dtype=torch.bool)		# [2 ** n, n]
ys = torch.tensor(ys, dtype=torch.bool).unsqueeze(1)	# [2 ** n, 1]

# ----------------------------
# Training using EDRA (per-sample)
# ----------------------------
for epoch in range(epochs):   # a few passes is usually enough
	for i in range(2 ** n):
		input_bits = xs[i:i+1]
		target_bits = ys[i:i+1]
		windows = model.make_windows(input_bits)
		model.train(windows, target_bits)

	print(f"\rEpoch {epoch+1:0{width}d} done.", end="", flush=True)

# ----------------------------
# Test after training
# ----------------------------
print("\nTesting after EDRA training:\n")

with torch.no_grad():
	count = 0
	total = 2 ** n
	for i in range(total):
		x = xs[i:i+1]
		y_true = ys[i].item()
		y_pred = model.forward(x).item()
		check = "✅" if y_true == y_pred else "❌"
		if y_true != y_pred or total < 256:
			print(f"{i:0{n}b}: predicted={y_pred}\t\texpected={y_true}\t\t{check}")
		count += y_pred == y_true
	acceptance_rate = count / (2 ** n)

end = datetime.now()
print(model)   # calls __str__()
print(f"Network acceptance rate: {acceptance_rate:.0%}")
print(f"\n=== End Parity Check Run at {end} ===\n")
print(f"\n=== Duration: {end - start} ===\n")