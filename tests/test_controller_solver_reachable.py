"""Exactness guard: reachable-address solver == exhaustive solver.

The reachable solvers (solve_partial_*_reachable_py) skip the O(2^n_bits) scan
by enumerating {trained cells} ∪ {k_top lowest-Hamming untrained}. This test
proves they return the IDENTICAL solution to the exhaustive solvers
(solve_partial_*_py) on the same logical memory, across random sparse configs.

For each config: build sparse entries (few written cells, rest = default EMPTY=2),
materialize the equivalent DENSE cell array, run exhaustive on the dense array
and reachable on the sparse entries, assert equal. n_bits kept <= 14 so the
exhaustive 2^n_bits scan is affordable as the reference.

Run:  python tests/test_controller_solver_reachable.py
"""

from __future__ import annotations

import sys
import numpy as np

import ram_accelerator as ra

EMPTY = 2  # default unwritten cell value (neuron_memory::EMPTY_U8)


def _rand_config(rng, n_bits, num_neurons, total_input_bits, n_entries_per_neuron, qsr):
	memory_size = 1 << n_bits
	connections = rng.integers(0, total_input_bits, size=num_neurons * n_bits).astype(np.int64).tolist()
	input_bits = (rng.integers(0, 2, size=total_input_bits) == 1).tolist()
	target_bits = (rng.integers(0, 2, size=num_neurons) == 1).tolist()

	# Sparse entries: a few random addresses per neuron with random values.
	e_neu, e_addr, e_val = [], [], []
	dense = [EMPTY] * (num_neurons * memory_size)
	val_hi = 4 if qsr else 3  # QSR: 0..3 ; TRINARY: 0..2
	for n in range(num_neurons):
		k = int(rng.integers(0, n_entries_per_neuron + 1))
		addrs = rng.choice(memory_size, size=min(k, memory_size), replace=False)
		for a in addrs:
			v = int(rng.integers(0, val_hi))
			e_neu.append(n); e_addr.append(int(a)); e_val.append(v)
			dense[n * memory_size + int(a)] = v
	return connections, input_bits, target_bits, e_neu, e_addr, e_val, dense


def main():
	rng = np.random.default_rng(0)
	configs = [
		# (n_bits, num_neurons, total_input_bits, entries/neuron, topk)
		(8, 4, 40, 6, 4),
		(10, 6, 60, 10, 4),
		(12, 5, 50, 8, 6),
		(14, 4, 48, 12, 4),
		(10, 8, 64, 20, 8),
		(13, 3, 40, 3, 4),   # very sparse → mostly untrained
		(11, 6, 55, 0, 4),   # ALL untrained (exercises pure low-Hamming gen)
	]
	n_fail = 0
	n_qsr = 0
	n_tri = 0
	for ci, (n_bits, nn, tot, epn, topk) in enumerate(configs):
		for seed in range(6):
			r = np.random.default_rng(1000 * ci + seed)

			# --- QSR ---
			conn, ib, tb, en, ea, ev, dense = _rand_config(r, n_bits, nn, tot, epn, qsr=True)
			exh = ra.solve_partial_qsr_py(dense, conn, nn, n_bits, tot, ib, tb, 0, topk)
			rch = ra.solve_partial_qsr_reachable_py(en, ea, ev, EMPTY, conn, nn, n_bits, tot, ib, tb, 0, topk)
			n_qsr += 1
			if exh != rch:
				n_fail += 1
				print(f"  [QSR FAIL] cfg={ci} seed={seed} n_bits={n_bits} nn={nn}: exhaustive != reachable")

			# --- TRINARY (both override modes) ---
			conn, ib, tb, en, ea, ev, dense = _rand_config(r, n_bits, nn, tot, epn, qsr=False)
			for ov in (False, True):
				exh = ra.solve_partial_trinary_py(dense, conn, nn, n_bits, tot, ib, tb, ov, 0, topk)
				rch = ra.solve_partial_trinary_reachable_py(en, ea, ev, EMPTY, conn, nn, n_bits, tot, ib, tb, ov, 0, topk)
				n_tri += 1
				if exh != rch:
					n_fail += 1
					print(f"  [TRI FAIL] cfg={ci} seed={seed} ov={ov} n_bits={n_bits}: exhaustive != reachable")

	print(f"\nQSR comparisons: {n_qsr}, TRINARY comparisons: {n_tri}, failures: {n_fail}")
	if n_fail == 0:
		print("PASS — reachable solver is EXACT (identical to exhaustive) on all configs.")
		return 0
	print("FAIL — reachable diverges from exhaustive.")
	return 1


if __name__ == "__main__":
	sys.exit(main())
