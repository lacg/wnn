"""Support-tiered per-class sizing (MCST arm — docs/MCST_TIERED_ARM_SPEC.md).

Pure functions: per-class neuron centres from train-label supports under a
joint cap, and per-class bits centres from the constant-fill-density formula
b_c = clamp(round(B_MAX * log2(s_c) / log2(S_model)), B_MIN, B_MAX).

Evidence base (flow 5985 S1 grid, banked in ids_results.md §9): neurons are
nearly inert (+0.98pp acc for 100x), bits is ~55x the lever (+54pp for 8.5x).
Neuron tiering is therefore an EFFICIENCY device (footprint + grid-lottery
kill); bits tiering and classnorm decode carry the accuracy claim.
"""

from __future__ import annotations

import math

NEURON_FLOOR = 10
BITS_MIN = 10
BITS_MAX = 34


def allocate_neurons(counts: list[int], cap: int, floor: int = NEURON_FLOOR) -> list[int]:
	"""Per-class neuron centres: cap * s_c / S_total, floored, renormalized.

	Classes whose proportional share falls under `floor` are pinned there and
	the remaining budget is rescaled over the rest — iterating, because the
	rescale can push another class under the floor. Total lands at <= cap
	(rounding may leave a unit or two unspent; never overspent above
	cap + one rounding unit per class).
	"""
	k = len(counts)
	if k == 0:
		return []
	if cap < floor * k:
		# Cap cannot pay the floors — floors win, caller sees the overshoot.
		return [floor] * k
	total = sum(counts)
	if total <= 0:
		base = max(floor, cap // k)
		return [base] * k

	pinned = [False] * k
	out = [0] * k
	while True:
		budget = cap - floor * sum(pinned)
		free_total = sum(c for c, p in zip(counts, pinned) if not p)
		changed = False
		for i, (c, p) in enumerate(zip(counts, pinned)):
			if p:
				out[i] = floor
				continue
			share = budget * c / free_total if free_total > 0 else 0.0
			if share < floor:
				pinned[i] = True
				changed = True
			else:
				out[i] = int(share + 0.5)
		if not changed:
			return out


def bits_centre(count: int, model_total: int, bmin: int = BITS_MIN, bmax: int = BITS_MAX) -> int:
	"""Constant-fill-density bits: bmax for a class fed every training row,
	scaling down with log2 of its support. Floors at bmin so even a tiny
	class keeps a usable address space."""
	if model_total <= 1 or count <= 1:
		return bmin
	b = bmax * math.log2(count) / math.log2(model_total)
	return max(bmin, min(bmax, int(b + 0.5)))


def bits_centres(counts: list[int], model_total: int,
                 bmin: int = BITS_MIN, bmax: int = BITS_MAX) -> list[int]:
	return [bits_centre(c, model_total, bmin, bmax) for c in counts]


def scaled_shape(neuron_centres: list[int], bits_centres_: list[int],
                 nm: float, bm: float,
                 bmin: int = BITS_MIN, bmax: int = BITS_MAX) -> tuple[list[int], list[int]]:
	"""One grid point: global multipliers applied to every class centre.
	Returns (neurons_per_cluster, bits_per_cluster). Neurons floor at 1;
	bits clamp to [bmin, bmax]."""
	neurons = [max(1, int(c * nm + 0.5)) for c in neuron_centres]
	bits = [max(bmin, min(bmax, int(b * bm + 0.5))) for b in bits_centres_]
	return neurons, bits
