#!/usr/bin/env python3
#!/usr/bin/env python3
import sys, os

# Add project root to PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from wnn.ram.RAMTransformer import RAMTransformer

n = 16
epochs = 50
# ----------------------------
# Build a tiny RAMTransformer
# ----------------------------
model = RAMTransformer(
	input_bits=4,
	n_input_neurons=5,
	n_state_neurons=0,         # no state for this toy
	n_output_neurons=1,
	n_bits_per_input_neuron=2,
	n_bits_per_state_neuron=0, # unused
	n_bits_per_output_neuron=4,
	use_hashing=False,
	rng=None,
)

# ----------------------------
# Build parity dataset (all 16 examples)
# ----------------------------
xs = []
ys = []
for i in range(n):
	vec = [(i >> b) & 1 for b in range(4)]
	xs.append(vec)
	ys.append(sum(vec) % 2)  # odd parity

xs = torch.tensor(xs, dtype=torch.bool)		# [16, 4]
ys = torch.tensor(ys, dtype=torch.bool).unsqueeze(1)	# [16, 1]

# ----------------------------
# Training using EDRA (per-sample)
# ----------------------------
for epoch in range(epochs):   # a few passes is usually enough
	for i in range(n):
		x = xs[i:i+1]
		y = ys[i:i+1]
		model.train_one(x, y)

	print(f"Epoch {epoch+1} done.")

# ----------------------------
# Test after training
# ----------------------------
print("\nTesting after EDRA training:\n")

with torch.no_grad():
	count = 0
	for i in range(n):
		x = xs[i:i+1]
		y_true = ys[i].item()
		y_pred = model.forward(x).item()
		print(f"{i:04b}: predicted={y_pred}   expected={y_true}")
		count += y_pred == y_true
	acceptance_rate = count / n
	print(f"Network acceptance rate: {acceptance_rate:.0%}")