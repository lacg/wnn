#!/usr/bin/env python3

from datetime import datetime
import sys, os

start = datetime.now()

print(f"\n=== Starting Parity Check Run at {start} ===\n")

# Add project root to PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from wnn.ram import RAMTransformer

n = 7
epochs = 5000
width = len(str(epochs))
# ----------------------------
# Build a tiny RAMTransformer
# ----------------------------
model = RAMTransformer(
	input_bits=n,
	n_input_neurons=14,
	n_state_neurons=0,         # no state for this toy
	n_output_neurons=1,
	n_bits_per_input_neuron=6,
	n_bits_per_state_neuron=2, # unused
	n_bits_per_output_neuron=14,
	use_hashing=False,
	rng=None,
	max_iters=4,
)

# ----------------------------
# Build parity dataset (all 2 ** n examples)
# ----------------------------
xs = []
ys = []
for i in range(2 ** n):
	vec = [(i >> b) & 1 for b in range(n)]
	xs.append(vec)
	ys.append(sum(vec) % 2)  # odd parity

xs = torch.tensor(xs, dtype=torch.bool)		# [2 ** n, n]
ys = torch.tensor(ys, dtype=torch.bool).unsqueeze(1)	# [2 ** n, 1]

# ----------------------------
# Training using EDRA (per-sample)
# ----------------------------
for epoch in range(epochs):   # a few passes is usually enough
	for i in range(2 ** n):
		x = xs[i:i+1]
		y = ys[i:i+1]
		model.train_one(x, y)

	print(f"\rEpoch {epoch+1:0{width}d} done.", end="", flush=True)

# ----------------------------
# Test after training
# ----------------------------
print("\nTesting after EDRA training:\n")

with torch.no_grad():
	count = 0
	for i in range(2 ** n):
		x = xs[i:i+1]
		y_true = ys[i].item()
		y_pred = model.forward(x).item()
		print(f"{i:0{n}b}: predicted={y_pred}   expected={y_true}")
		count += y_pred == y_true
	acceptance_rate = count / (2 ** n)

end = datetime.now()
print(model)   # calls __str__()
print(f"Network acceptance rate: {acceptance_rate:.0%}")
print(f"\n=== End Parity Check Run at {end} ===\n")
print(f"\n=== Duration: {end - start} ===\n")