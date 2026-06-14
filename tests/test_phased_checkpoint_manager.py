"""Tests for PhasedCheckpointManager — the unified checkpoint orchestrator
(13/06/2026). Covers both strands' codecs + the legacy IDS-*.json resume path.

Run: PYTHONPATH=src python tests/test_phased_checkpoint_manager.py
"""
import json
import tempfile
from pathlib import Path

from wnn.ram.strategies.phased import (
	PhasedCheckpointManager, SaveCadence, PhaseCheckpoint,
	ClusterGenomeCodec, ControllerGenomeCodec,
)

PASS = "[PASS]"


class _Clock:
	def __init__(self): self.t = 1000.0
	def __call__(self): return self.t
	def advance(self, dt): self.t += dt


def _cluster_pop(n):
	from wnn.ram.genome import ClusterGenome
	return [ClusterGenome(bits_per_neuron=[8] * 4, neurons_per_cluster=[4],
	                      connections=list(range(i, i + 32))) for i in range(n)]


def test_cluster_save_load_roundtrip():
	pop = _cluster_pop(6)
	ck = PhaseCheckpoint(phase_key="1", phase_name="GA Neurons", strategy_type="GA",
	                     best_genome=pop[0], final_population=pop, iterations_run=7, patience=3)
	d = Path(tempfile.mkdtemp())
	mgr = PhasedCheckpointManager(d / "ids_ck", ClusterGenomeCodec(),
	                              SaveCadence(None, 10))
	assert not mgr.has_checkpoint()
	mgr.save(ck)
	assert mgr.has_checkpoint()
	out = mgr.load()
	assert out.iterations_run == 7 and out.patience == 3 and len(out.final_population) == 6
	assert out.final_population[0].connections == pop[0].connections
	print(f"  {PASS} ClusterGenome save/load round-trip (pop=6, gen=7, pat=3)")


def test_maybe_save_cadence_gating():
	from wnn.control.recurrent_genome import RecurrentArchGenome, RecurrentArchShape
	g = RecurrentArchGenome(shape=RecurrentArchShape(1, 24, 24, 16), state_neurons=1,
	                        output_neurons=1, state_sampled=[[0]], output_sampled=[[0]], cells=None)
	ck = PhaseCheckpoint(phase_key="1", phase_name="NEURONS", strategy_type="GA",
	                     best_genome=g, final_population=[g], iterations_run=0)
	d = Path(tempfile.mkdtemp())
	mgr = PhasedCheckpointManager(d / "ctl_ck", ControllerGenomeCodec(),
	                              SaveCadence(target_loss_seconds=0.0, max_interval=10))
	assert mgr.maybe_save(0, ck) is False, "gen 0 is the cadence baseline → no save"
	assert mgr.maybe_save(1, ck) is True, "budget 0 ⇒ save after baseline"
	assert mgr.has_checkpoint()
	print(f"  {PASS} maybe_save gates on the shared SaveCadence (baseline gen0, save gen1)")


def test_async_single_writer():
	pop = _cluster_pop(4)
	ck = PhaseCheckpoint(phase_key="1", phase_name="GA", strategy_type="GA",
	                     best_genome=pop[0], final_population=pop, iterations_run=2)
	d = Path(tempfile.mkdtemp())
	mgr = PhasedCheckpointManager(d / "a_ck", ClusterGenomeCodec(),
	                              SaveCadence(None, 10), async_save=True)
	mgr.save(ck); mgr.save(ck)  # second joins the first → no temp collision
	mgr.join()
	assert mgr.has_checkpoint() and mgr.load().iterations_run == 2
	print(f"  {PASS} async save keeps a single in-flight writer (no temp race)")


def test_legacy_ids_json_resume():
	"""A checkpoint written by the OLD IDS CheckpointManager (*.json) must load
	through the unified manager so a post-migration worker restart resumes it."""
	d = Path(tempfile.mkdtemp())
	legacy = {
		"phase_name": "Phase 1a: GA Neurons", "optimizer_type": "GA",
		"current_iteration": 12, "total_iterations": 100,
		"population": [
			{"bits_per_neuron": [8, 8], "neurons_per_cluster": [2],
			 "connections": [0, 5, 11, 2, 7, 9, 1, 3, 6, 8, 4, 10, 12, 13, 14, 15],
			 "fitness": {"ce": 0.31, "acc": 0.94, "f1": 0.93, "fpr": 0.05}},
			{"bits_per_neuron": [8, 8], "neurons_per_cluster": [2],
			 "connections": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0],
			 "fitness": [0.40, 0.90]},
		],
		"best_genome": {"bits_per_neuron": [8, 8], "neurons_per_cluster": [2],
		                "connections": [0, 5, 11, 2, 7, 9, 1, 3, 6, 8, 4, 10, 12, 13, 14, 15],
		                "fitness": {"ce": 0.31, "acc": 0.94}},
		"best_fitness": [0.31, 0.94], "current_threshold": 0.5,
		"config": {"foo": "bar"}, "extra_state": {"patience": 4},
		"saved_at": "2026-06-10T00:00:00",
	}
	# OLD path scheme: {prefix}_{optimizer}.json
	p = d / "ga_checkpoint_ga.json"
	p.write_text(json.dumps(legacy, indent=2))
	mgr = PhasedCheckpointManager(p, ClusterGenomeCodec(), SaveCadence(None, 10))
	assert mgr.has_checkpoint()
	out = mgr.load()
	assert out.iterations_run == 12 and out.patience == 4
	assert out.final_fitness == 0.31 and out.final_accuracy == 0.94
	assert len(out.final_population) == 2
	assert out.final_population[0].connections == legacy["population"][0]["connections"]
	# the dict-fitness genome restored its cached metrics
	m = out.final_population[0].metrics
	assert m is not None and abs(m.ce - 0.31) < 1e-9 and abs(m.acc - 0.94) < 1e-9
	assert out.best_genome.connections == legacy["best_genome"]["connections"]
	assert out.extra.get("legacy_ids_json") is True
	print(f"  {PASS} legacy IDS *.json resumes through the unified manager")


if __name__ == "__main__":
	tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
	print(f"Running {len(tests)} PhasedCheckpointManager tests...")
	for t in tests:
		t()
	print("ALL PASS")
