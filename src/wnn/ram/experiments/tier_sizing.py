"""Support-tiered per-class sizing (MCST arm — docs/MCST_TIERED_ARM_SPEC.md).

Pure functions: per-class neuron centres from train-label supports under a
joint cap, and per-class bits centres sized so every class carries comparable
EVIDENCE DENSITY in its address space.

**The bits rule was corrected on 28/08/2026.** The old rule
`b = B_MAX * log2(s) / log2(S_model)` was documented as "constant-fill-density"
and is not: it makes `2^b` proportional to `s^(B_MAX/log2 S)` — at B_MAX=50 that
is `s^3.04`, so fill goes as `s^-2` and BIGGER classes end up SPARSER. Measured
on UNSW S1 it spread rows-per-address over 100x (8.7e-8 Analysis to 8.5e-10
Generic) while claiming to hold it constant. It is kept as
`bits_centre_logratio` ONLY to reproduce the banked tiered2/tiered3 cohorts.

True constant fill depends on a class's OWN support and nothing else:
    b_c = clamp(round(log2(s_c * sample_rate / FILL_TARGET)), B_MIN, B_MAX)
No model total appears — that was the error.

Evidence base (flow 5985 S1 grid, banked in ids_results.md §9): neurons are
nearly inert (+0.98pp acc for 100x), bits is ~55x the lever (+54pp for 8.5x).
Neuron tiering is therefore an EFFICIENCY device (footprint + grid-lottery
kill); bits tiering and classnorm decode carry the accuracy claim.
"""

from __future__ import annotations

import math

NEURON_FLOOR = 10
# Lowered 10 -> 4 on 28/08/2026. Under true constant fill a rare class genuinely
# WANTS a small address space: UNSW's Worms (97 train rows x 0.25 sample_rate =
# ~24 rows per neuron) wants b=4. A floor of 10 clamped it back up to a space it
# can never populate, which is what made it a 508x error sink — every lookup
# missed and scored the WEAK_FALSE pedestal instead of real evidence.
BITS_MIN = 4
BITS_MAX = 34

# Rows per address to aim for. The QUAD commit lattice (oi_bin_to_cell) needs
# obs >= 2 on a cell before it can leave the WEAK states: obs==1 pins to
# WEAK_TRUE/WEAK_FALSE no matter how large the weighted net is, so class
# weighting CANNOT rescue a class that never collides with itself. Two rows per
# address is therefore the minimum for a cell to reach TRUE/FALSE at all.
FILL_TARGET = 2.0


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


def bits_centre(count: int, fill_target: float = FILL_TARGET,
                sample_rate: float = 1.0,
                bmin: int = BITS_MIN, bmax: int = BITS_MAX) -> int:
	"""TRUE constant fill: size the address space to the class's OWN evidence.

	`b = log2(count * sample_rate / fill_target)` — so every class lands at the
	same rows-per-address regardless of how big it is. The model total does not
	appear; a class's right address width is a property of its own support.

	`sample_rate` is the neuron sample rate: each neuron only sees that fraction
	of the training rows (marker_train: effective_train = num_train * rate), so
	at the production 0.25 a 97-row class gives each neuron ~24 rows, not 97.

	Floors at bmin. A class at or below the fill target gets bmin — it has too
	little evidence to spread over any address space at all."""
	eff = count * max(0.0, min(1.0, sample_rate))
	if eff <= fill_target or fill_target <= 0:
		return bmin
	b = math.log2(eff / fill_target)
	return max(bmin, min(bmax, int(b + 0.5)))


def bits_centres(counts: list[int], fill_target: float = FILL_TARGET,
                 sample_rate: float = 1.0,
                 bmin: int = BITS_MIN, bmax: int = BITS_MAX) -> list[int]:
	return [bits_centre(c, fill_target, sample_rate, bmin, bmax) for c in counts]


def bits_centre_logratio(count: int, model_total: int,
                         bmin: int = 10, bmax: int = BITS_MAX) -> int:
	"""LEGACY rule — kept ONLY to reproduce the banked tiered2/tiered3 cohorts.

	`b = bmax * log2(count) / log2(model_total)`. Documented at the time as
	"constant-fill-density"; it is not (see the module docstring). DO NOT use
	for new arms. Its own default floor was 10, preserved here so a reproduction
	is exact."""
	if model_total <= 1 or count <= 1:
		return bmin
	b = bmax * math.log2(count) / math.log2(model_total)
	return max(bmin, min(bmax, int(b + 0.5)))


def bits_centres_logratio(counts: list[int], model_total: int,
                          bmin: int = 10, bmax: int = BITS_MAX) -> list[int]:
	return [bits_centre_logratio(c, model_total, bmin, bmax) for c in counts]


def scaled_shape(neuron_centres: list[int], bits_centres_: list[int],
                 nm: float, bm: float,
                 bmin: int = BITS_MIN, bmax: int = BITS_MAX) -> tuple[list[int], list[int]]:
	"""One grid point: global multipliers applied to every class centre.
	Returns (neurons_per_cluster, bits_per_cluster). Neurons floor at 1;
	bits clamp to [bmin, bmax]."""
	neurons = [max(1, int(c * nm + 0.5)) for c in neuron_centres]
	bits = [max(bmin, min(bmax, int(b * bm + 0.5))) for b in bits_centres_]
	return neurons, bits
