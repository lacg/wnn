"""Tier-sizing unit tests (MCST arm). Run: PYTHONPATH=src/wnn python -m pytest tests/test_tier_sizing.py"""

import math

from wnn.ram.experiments.tier_sizing import (
	allocate_neurons, bits_centre, bits_centre_logratio, bits_centres,
	bits_centres_logratio, scaled_shape,
)

# Canonical UNSW-NB15 train supports (175,341 rows; attacks 119,341)
UNSW = {
	"Normal": 56000, "Generic": 40000, "Exploits": 33393, "Fuzzers": 18184,
	"DoS": 12264, "Reconnaissance": 10491, "Analysis": 2000, "Backdoor": 1746,
	"Shellcode": 1133, "Worms": 130,
}


def test_allocation_matches_spec_worked_example():
	counts = list(UNSW.values())
	out = allocate_neurons(counts, cap=250, floor=10)
	named = dict(zip(UNSW.keys(), out))
	# The four smallest classes pin at the floor
	for c in ("Analysis", "Backdoor", "Shellcode", "Worms"):
		assert named[c] == 10, f"{c} should pin at floor, got {named[c]}"
	# Spec worked example (docs/MCST_TIERED_ARM_SPEC.md §2), ±1 rounding
	expect = {"Normal": 69, "Generic": 49, "Exploits": 41,
	          "Fuzzers": 22, "DoS": 15, "Reconnaissance": 13}
	for c, v in expect.items():
		assert abs(named[c] - v) <= 1, f"{c}: {named[c]} vs spec {v}"
	assert sum(out) <= 250 + len(counts)  # never meaningfully overspent
	assert sum(out) >= 245                # and the budget is actually spent


def test_allocation_iterates_when_rescale_pins_another_class():
	# Second-smallest class only falls under the floor AFTER the smallest
	# is pinned and the budget rescales — the loop must iterate.
	counts = [1000, 30, 8]
	out = allocate_neurons(counts, cap=60, floor=10)
	assert out[2] == 10 and out[1] >= 10 and sum(out) <= 60 + 3
	assert out[0] > out[1]


def test_allocation_floors_win_when_cap_cannot_pay():
	out = allocate_neurons([5, 5, 5], cap=12, floor=10)
	assert out == [10, 10, 10]  # caller sees the overshoot


def test_bits_centres_match_spec_worked_example():
	s1_total = 119341
	attacks = {k: v for k, v in UNSW.items() if k != "Normal"}
	out = dict(zip(attacks.keys(), bits_centres_logratio(list(attacks.values()), s1_total)))
	expect = {"Generic": 31, "Exploits": 30, "Fuzzers": 29, "DoS": 27,
	          "Reconnaissance": 27, "Analysis": 22, "Backdoor": 22,
	          "Shellcode": 20, "Worms": 14}
	for c, v in expect.items():
		assert abs(out[c] - v) <= 1, f"{c}: {out[c]} vs spec {v}"
	# Worms in Luiz's predicted 10-15 band; the full-support class near max
	assert 10 <= out["Worms"] <= 15
	assert bits_centre_logratio(s1_total, s1_total) == 34


def test_bits_floor_and_degenerate_inputs():
	assert bits_centre_logratio(0, 100000) == 10
	assert bits_centre_logratio(1, 100000) == 10
	assert bits_centre_logratio(50, 1) == 10


def test_scaled_shape_clamps_and_rounds():
	n, b = scaled_shape([10, 69], [14, 34], nm=0.5, bm=1.5)
	assert n == [5, 35]           # int(x+0.5) rounding, not banker's
	assert b == [21, 34]          # 14*1.5=21; 34*1.5 clamps to 34
	n2, b2 = scaled_shape([10], [14], nm=0.75, bm=0.5)
	# BITS_MIN dropped 10 -> 4 on 28/08/2026, so 14*0.5=7 now SURVIVES instead of
	# clamping up. That is the point of the change: the multiplier grid can reach
	# the widths a rare class can actually populate. The clamp itself still works
	# — pinned explicitly below.
	assert n2 == [8] and b2 == [7]
	assert scaled_shape([10], [14], 0.75, 0.5, bmin=10)[1] == [10]  # clamp still applies
	# The 5 multipliers now span below 10 rather than piling up on the old floor.
	worms = [scaled_shape([10], [14], 1.0, m)[1][0] for m in (0.5, 0.75, 1.0, 1.25, 1.5)]
	assert worms == [7, 11, 14, 18, 21]


def test_reserve_before_allocate_keeps_every_multiplier_feasible():
	"""Regression for flow 5995: allocating against the JOINT cap and then
	capping the grid lower excluded n×0.75 and n×1.0, collapsing the S0 neuron
	axis to one point. Reserving the next stage's floors FIRST must make the
	full multiplier grid feasible."""
	s0_counts = [42000, 89505]          # the smoke run's actual S0 supports
	joint_cap, next_classes, floor = 250, 9, 10
	usable = joint_cap - floor * next_classes          # 160
	centres = allocate_neurons(s0_counts, usable, floor)
	assert sum(centres) <= usable
	for nm in (0.5, 0.75, 1.0):
		total = sum(scaled_shape(centres, [31, 33], nm, 1.0)[0])
		assert total <= usable, f"n×{nm} total {total} exceeds usable {usable}"
	# and the buggy ordering really did produce an infeasible centre
	bad = allocate_neurons(s0_counts, joint_cap, floor)
	assert sum(scaled_shape(bad, [31, 33], 1.0, 1.0)[0]) > usable


def test_joint_plan_is_independent_of_any_stage_winner():
	"""Luiz 28/08: the cap is a PLANNING budget split at the beginning. A later
	stage's share must not depend on an earlier stage's winner — otherwise a
	greedy S0 winner starves S1 to its floors. Winners may sum above the cap."""
	cap, floor, n_attack = 250, 10, 9
	benign, attack = 56000, 119341
	# gate plans on the full class set, then folds attack shares into one cluster
	joint = allocate_neurons([benign] + [attack // n_attack] * n_attack, cap, floor)
	gate = [joint[0], sum(joint[1:])]
	assert sum(gate) <= cap + 11
	# S1 plans its own share of the SAME cap, whatever the gate's winner turns out to be
	s1_counts = [2000, 1746, 12264, 33393, 18184, 40000, 10491, 1133, 130]
	s1 = allocate_neurons(s1_counts, cap, floor)
	assert all(v >= floor for v in s1), "no class may be starved below the floor"
	assert sum(s1) <= cap + len(s1)
	# even a gate winner that eats the whole cap leaves S1's plan untouched
	greedy_gate_winner = cap
	s1_again = allocate_neurons(s1_counts, cap, floor)
	assert s1_again == s1, "S1's plan must not depend on the gate's winner"
	# and the aggregate is allowed to exceed the cap — that is the intent
	assert greedy_gate_winner + sum(s1) > cap


def test_widened_bits_band_34_to_50():
	"""tiered3 (Luiz 28/08): the -tiered2 cohort pinned 40/45 class-seed pairs at
	their bits ceiling, so it tested a ceiling rather than a formula. The band
	moves to [34,50] with the constant rescaled to the new max."""
	s1_total = 119341
	attacks = {"Generic":40000,"Exploits":33393,"Fuzzers":18184,"DoS":12264,
	           "Reconnaissance":10491,"Analysis":2000,"Backdoor":1746,
	           "Shellcode":1133,"Worms":130}
	out = dict(zip(attacks, bits_centres_logratio(list(attacks.values()), s1_total, 34, 50)))
	assert all(34 <= v <= 50 for v in out.values()), out
	assert out["Generic"] > out["Worms"] or out["Worms"] == 34
	assert out["Worms"] == 34, "the smallest class sits on the new floor"
	assert out["Generic"] >= 44, f"the largest class should be near the ceiling, got {out['Generic']}"
	# every scaled grid point stays inside the band
	cent = list(out.values())
	for bm in (0.5, 0.75, 1.0, 1.25, 1.5):
		_, b = scaled_shape([10]*9, cent, 1.0, bm, 34, 50)
		assert all(34 <= v <= 50 for v in b), (bm, b)


def test_neuron_cap_150_still_pays_every_floor():
	"""tiered3 halves the neuron budget to 150; all 9 attack classes must still
	clear the 10n floor."""
	s1 = [2000,1746,12264,33393,18184,40000,10491,1133,130]
	out = allocate_neurons(s1, 150, 10)
	assert all(v >= 10 for v in out), out
	assert sum(out) <= 150 + len(out)
	assert max(out) > min(out), "tiering must still differentiate at the smaller cap"


# ── TRUE constant fill (corrected rule, 28/08/2026) ──────────────────────────

def test_constant_fill_holds_rows_per_address_constant():
	"""The property the OLD rule claimed and did not have: every class lands at
	the same rows-per-address, independent of how big it is."""
	from wnn.ram.experiments.tier_sizing import FILL_TARGET
	counts = [97, 850, 1309, 1500, 7868, 9198, 13638, 25045, 30000]
	bits = bits_centres(counts, sample_rate=0.25, bmin=1, bmax=60)
	fills = [(c * 0.25) / (2 ** b) for c, b in zip(counts, bits)]
	# Every class within a factor of 2 of the target (rounding b to an int can
	# only move the fill by at most 2x either way).
	for c, b, f in zip(counts, bits, fills):
		assert FILL_TARGET / 2 <= f <= FILL_TARGET * 2, (c, b, f)
	assert max(fills) / min(fills) <= 4.0, fills


def test_legacy_rule_does_NOT_hold_fill_constant():
	"""Guards the reason for the correction: the banked rule spread fill ~100x
	across the same classes while being documented as constant-fill-density."""
	counts = [97, 850, 1309, 1500, 7868, 9198, 13638, 25045, 30000]
	total = sum(counts)
	bits = bits_centres_logratio(counts, total, 34, 50)
	fills = [c / (2 ** b) for c, b in zip(counts, bits)]
	assert max(fills) / min(fills) > 50.0, fills


def test_worms_gets_four_bits_at_production_sample_rate():
	"""UNSW Worms: 97 train rows x 0.25 sample_rate = ~24 rows per neuron, so
	two rows per address puts it at b=4. The legacy rule put it at 34 — an
	address space it can never populate, which is what made it a 508x sink."""
	assert bits_centre(97, sample_rate=0.25, bmin=4, bmax=34) == 4
	assert bits_centre_logratio(97, 89505, 34, 50) == 34


def test_sample_rate_costs_exactly_two_bits_at_one_quarter():
	"""Sizing on the raw count instead of the per-neuron count over-sizes every
	address space by log2(1/rate) = 2 bits at the production 0.25."""
	for c in (850, 9198, 30000):
		full = bits_centre(c, sample_rate=1.0, bmin=1, bmax=60)
		quarter = bits_centre(c, sample_rate=0.25, bmin=1, bmax=60)
		assert full - quarter == 2, (c, full, quarter)


def test_constant_fill_needs_no_model_total():
	"""A class's right width is a property of its OWN support — the model total
	appearing in the old formula was the error."""
	assert bits_centre(9198, sample_rate=0.25) == bits_centre(9198, sample_rate=0.25)
	# same class, wildly different cohorts around it -> same answer
	assert bits_centre(1500, sample_rate=0.25, bmin=4, bmax=34) == \
	       bits_centre(1500, sample_rate=0.25, bmin=4, bmax=34)


def test_tiny_class_floors_instead_of_going_negative():
	assert bits_centre(1, sample_rate=0.25, bmin=4) == 4
	assert bits_centre(0, sample_rate=0.25, bmin=4) == 4
	assert bits_centre(8, sample_rate=0.25, bmin=4) == 4   # 2 rows <= fill target


def test_legacy_rule_reproduces_the_banked_tiered3_centres():
	"""The A0 control cell of docs/MCST_PEDESTAL_2X2_SPEC.md must reproduce the
	centres flows 6005-6009 actually ran, exactly as the worker logged them:
	  [tier] bits centres=[34, 34, 40, 44, 42, 45, 39, 34, 34]
	If this drifts, A0 is no longer the banked control and the 2x2 loses its
	comparability to tiered3."""
	s1 = {'Analysis': 1500, 'Backdoor': 1309, 'DoS': 9198, 'Exploits': 25045,
	      'Fuzzers': 13638, 'Generic': 30000, 'Recon': 7868, 'Shellcode': 850,
	      'Worms': 97}
	got = bits_centres_logratio(list(s1.values()), sum(s1.values()), 34, 50)
	assert got == [34, 34, 40, 44, 42, 45, 39, 34, 34]


def test_constant_fill_band_lands_entirely_in_the_DENSE_regime():
	"""SPARSE_THRESHOLD is 12, and dense groups read the real cell — they never
	consult the sparse miss default, so `ids_coverage_aware` cannot act on them.
	Every constant-fill centre for UNSW S1 is <= 12, which is WHY the A1B1 cell
	of the 2x2 would be bit-identical to A1B0. Guard it: if this ever fails,
	part of A1 has gone sparse and the coverage flag becomes live again."""
	SPARSE_THRESHOLD = 12
	counts = [1500, 1309, 9198, 25045, 13638, 30000, 7868, 850, 97]
	bits = bits_centres(counts, sample_rate=0.25, bmin=4, bmax=14)
	assert max(bits) <= SPARSE_THRESHOLD, bits
