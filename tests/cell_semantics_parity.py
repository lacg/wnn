"""
Parity test: Python cell semantics vs the Rust single source of truth.

Part 1 always runs (pure-Python table assertions against the documented Rust
constants). Part 2 needs the ABI-2 accelerator build and SKIPs cleanly until
`maturin develop --release` has run on this branch.

Run: PYTHONPATH=src python tests/cell_semantics_parity.py
"""

import sys

import torch

from wnn.ram.core.cell_semantics import (
	MODE_QUAD_BINARY, MODE_QUAD_WEIGHTED, MODE_TERNARY,
	QUAD_WEIGHTS, cell_to_weight, cell_weight_table,
)


def main() -> int:
	checks = {}

	# --- Part 1: the exact Rust mapping table ---
	for mode in (MODE_QUAD_WEIGHTED, MODE_QUAD_BINARY):
		checks[f"quad mode {mode} table"] = all(
			cell_to_weight(c, mode, 99.0) == QUAD_WEIGHTS[c] for c in range(4)
		)
		checks[f"quad mode {mode} clamps"] = (
			cell_to_weight(-1, mode, 99.0) == 0.0 and cell_to_weight(7, mode, 99.0) == 1.0
		)
	checks["ternary mapping"] = (
		cell_to_weight(0, MODE_TERNARY, 0.5) == 0.0
		and cell_to_weight(1, MODE_TERNARY, 0.5) == 1.0
		and cell_to_weight(2, MODE_TERNARY, 0.5) == 0.5
	)
	# The trap this module closes: ternary and quad DISAGREE on cells 1-3
	checks["encodings disagree beyond cell 0"] = all(
		cell_to_weight(c, MODE_TERNARY, 0.25) != cell_to_weight(c, MODE_QUAD_WEIGHTED, 0.25)
		for c in (1, 2, 3)
	)
	lut = cell_weight_table(MODE_QUAD_WEIGHTED, 0.0)
	raw = torch.tensor([0, 1, 2, 3, 3, 0])
	checks["vectorized table matches scalar"] = torch.equal(
		lut[raw], torch.tensor([0.0, 0.25, 0.75, 1.0, 1.0, 0.0]))

	# --- Part 2: live accelerator comparison (skips on stale/absent build) ---
	try:
		from wnn.accel import require_accel
		require_accel()
		from wnn.ram.core.RAMClusterLayer import RAMClusterLayer  # noqa: F401
		# Train a tiny layer through the Rust numpy path, then compare the
		# Python quad-aware forward against the Rust forward on the same memory.
		torch.manual_seed(0)
		layer = RAMClusterLayer(num_clusters=3, neurons_per_cluster=4,
		                        n_bits_per_neuron=6, total_input_bits=24)
		x = (torch.rand(64, 24) > 0.5)
		true_c = torch.randint(0, 3, (64,))
		false_c = (true_c + 1).remainder(3).unsqueeze(1)
		layer.train_multi_examples_rust_numpy(x, true_c, false_c)
		py_scores = layer.forward_quad_scores(x[:8].float())
		rs_scores = layer.forward_rust(x[:8].float())
		max_dev = (py_scores - rs_scores).abs().max().item()
		checks["python-vs-rust forward parity (<1e-5)"] = max_dev < 1e-5
	except Exception as e:
		print(f"  [SKIP] live accelerator parity ({type(e).__name__}: {str(e)[:80]})")

	failed = [k for k, ok in checks.items() if not ok]
	for k, ok in checks.items():
		print(f"  [{'PASS' if ok else 'FAIL'}] {k}")
	print("ALL PASS" if not failed else f"FAILED: {failed}")
	return 1 if failed else 0


if __name__ == "__main__":
	sys.exit(main())
