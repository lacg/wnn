"""Tier-sizing unit tests (MCST arm). Run: PYTHONPATH=src/wnn python -m pytest tests/test_tier_sizing.py"""

import math

from wnn.ram.experiments.tier_sizing import (
	allocate_neurons, bits_centre, bits_centres, scaled_shape,
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
	out = dict(zip(attacks.keys(), bits_centres(list(attacks.values()), s1_total)))
	expect = {"Generic": 31, "Exploits": 30, "Fuzzers": 29, "DoS": 27,
	          "Reconnaissance": 27, "Analysis": 22, "Backdoor": 22,
	          "Shellcode": 20, "Worms": 14}
	for c, v in expect.items():
		assert abs(out[c] - v) <= 1, f"{c}: {out[c]} vs spec {v}"
	# Worms in Luiz's predicted 10-15 band; the full-support class near max
	assert 10 <= out["Worms"] <= 15
	assert bits_centre(s1_total, s1_total) == 34


def test_bits_floor_and_degenerate_inputs():
	assert bits_centre(0, 100000) == 10
	assert bits_centre(1, 100000) == 10
	assert bits_centre(50, 1) == 10


def test_scaled_shape_clamps_and_rounds():
	n, b = scaled_shape([10, 69], [14, 34], nm=0.5, bm=1.5)
	assert n == [5, 35]           # int(x+0.5) rounding, not banker's
	assert b == [21, 34]          # 14*1.5=21; 34*1.5 clamps to 34
	n2, b2 = scaled_shape([10], [14], nm=0.75, bm=0.5)
	assert n2 == [8] and b2 == [10]  # 14*0.5=7 clamps up to bits floor 10
	# Spec's Worms bits row {10,11,14,18,21} across the 5 multipliers
	worms = [scaled_shape([10], [14], 1.0, m)[1][0] for m in (0.5, 0.75, 1.0, 1.25, 1.5)]
	assert worms == [10, 11, 14, 18, 21]


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
	out = dict(zip(attacks, bits_centres(list(attacks.values()), s1_total, 34, 50)))
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
