"""FFD sub-batch packing parity — REAL train + score + write-back (27/09/2026).

Sorted (first-fit-decreasing) packing groups genomes NON-contiguously. It is only
admissible if every genome still trains on its GLOBAL-index seed and comes back in
the original order. This runs the real Rust DAgger trainer + scorer on a small
population of HETEROGENEOUS genomes (different state/output neurons and suffixes)
three ways and requires bit-identical per-genome metrics, adaptation stats AND
written-back cells:

  (a) unbatched (one sub-batch),
  (b) the 25/09 contiguous packing,
  (c) the new FFD packing.

The budget is forced small and the per-genome cell estimates are forced
heterogeneous (instance hooks), so (b) and (c) really produce different, multi-
batch groupings — asserted, so the test cannot pass vacuously.
"""

import math

import numpy as np
import pytest

from wnn.control.evaluator import (
	ControllerEvaluator, ControllerSpec, arch_shape_from_spec, fit_thresholds_from_pid_rollouts,
)
from wnn.control.recurrent_genome import RecurrentArchGenome
from wnn.control.reward_gated import RewardGatedConfig
from wnn.control.training import EpisodeConfig

SEED = 7
# (state_neurons, output_neurons, state_suffix, output_suffix) per genome.
SHAPES = [(2, 32, 8, 8), (4, 48, 12, 12), (3, 32, 10, 6), (4, 64, 12, 14),
          (2, 48, 6, 10), (3, 64, 8, 12)]
# Forced estimates (cap 8): contiguous [[0],[1,2],[3,4],[5]], FFD [[1,2],[0,3],[4,5]].
EST = [3, 6, 2, 5, 1, 4]
CAP = 8


def _spec() -> ControllerSpec:
	return ControllerSpec(num_motors=4, levels_per_motor=12, bits_per_feature=8,
	                      input_window_k=4, state_neurons=4, state_bits_per_neuron=20,
	                      output_bits_per_neuron=20)


def _evaluator(packing: str, crn: bool, thr) -> ControllerEvaluator:
	ec = EpisodeConfig(dt=0.001, steps_per_episode=40, max_initial_tilt_rad=math.radians(5.0))
	rg = RewardGatedConfig(num_rounds=2, episodes_per_round=4, steps_per_episode=40,
	                       eval_episodes=4, seed=SEED, episode_config=ec)
	ev = ControllerEvaluator(_spec(), num_eval_episodes=6, seed=SEED, episode_config=ec,
	                         thresholds=thr, rg_config=rg, num_eval_folds=2, score_crn=crn)
	if packing == "unbatched":
		ev.EVAL_BUDGET_BYTES = 10 ** 15
	else:
		ev.EVAL_PACKING = packing
		ev.EVAL_BUDGET_BYTES = CAP * ev.EVAL_BYTES_PER_CELL
		ev._genome_cell_estimates = lambda gs: list(EST)
	return ev


def _population() -> list:
	shape = arch_shape_from_spec(_spec())
	return [RecurrentArchGenome.random(shape, state_neurons=sn, output_neurons=on,
	                                   state_suffix=ss, output_suffix=os_,
	                                   rng=np.random.default_rng(SEED + i))
	        for i, (sn, on, ss, os_) in enumerate(SHAPES)]


def _run(packing: str, crn: bool, thr):
	ev = _evaluator(packing, crn, thr)
	gs = _population()
	groups = ev._eval_batch_groups(gs)
	res = ev.evaluate_for_adaptation(gs, write_back=True)
	metrics = [(m.reward, m.stable_rate, m.mean_attitude_error_deg, m.motor_jerk_mean,
	            m.mono_violations_total, m.mean_steady_error_deg) for m, _ in res]
	stats = [(list(a.state_cell_counts), list(a.output_cell_counts)) for _, a in res]
	cells = [g.cells.to_triples() for g in gs]
	return groups, metrics, stats, cells


@pytest.fixture(autouse=True)
def _no_override(monkeypatch):
	monkeypatch.delenv("WNN_CTRL_EVAL_BATCH", raising=False)
	monkeypatch.delenv("WNN_STATE_SPLIT", raising=False)


@pytest.mark.parametrize("crn", [False, True])
def test_unbatched_contiguous_ffd_are_bit_identical(crn):
	thr = fit_thresholds_from_pid_rollouts(_spec(), num_episodes=4, seed=SEED)
	g_a, m_a, s_a, c_a = _run("unbatched", crn, thr)
	g_b, m_b, s_b, c_b = _run("contiguous", crn, thr)
	g_c, m_c, s_c, c_c = _run("ffd", crn, thr)
	assert g_a == [list(range(len(SHAPES)))]
	assert g_b == [[0], [1, 2], [3, 4], [5]]
	assert g_c == [[1, 2], [0, 3], [4, 5]], "FFD must group non-contiguously here"
	assert m_a == m_b == m_c, "per-genome metrics moved with the packing"
	assert s_a == s_b == s_c, "adaptation stats moved with the packing"
	assert c_a == c_b == c_c, "written-back cells moved with the packing"
	assert any(len(o) for _, o in c_a), "fixture must actually write cells back"
