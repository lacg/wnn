"""End-to-end: a controller flow runs through the REAL Experiment.run() path.

Proves the proper peer integration — architecture_type='controller' threads through
the same Experiment machinery as IDS (strategy build, initial population, optimize,
result/checkpoint), producing a RecurrentArchGenome result without crashing on the
ClusterGenome-shaped recording code. Uses ga_memory (paradigm B, no training) so it
runs in seconds.

Run: PYTHONPATH=src/wnn python tests/test_controller_worker_integration.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from wnn.ram.experiments.experiment import Experiment, ExperimentConfig, ExperimentType
from wnn.control.flow_adapter import build_controller_evaluator
from wnn.control.recurrent_genome import RecurrentArchGenome


def _cfg(phase_type, exp_type, pop=4, gens=2):
	return ExperimentConfig(
		name=f"Controller {phase_type}",
		experiment_type=exp_type,
		architecture_type="controller",
		phase_type=phase_type,
		generations=gens,
		population_size=pop,
		iterations=gens,
	)


def _params():
	# tiny: 3 state neurons, 4 levels, 2 eval episodes x 40 steps
	return {"controller_state_neurons": 3, "controller_levels_per_motor": 4,
	        "controller_bits_per_feature": 8, "controller_input_window_k": 4,
	        "controller_eval_episodes": 2, "controller_steps": 40, "seed": 0}


def _run(phase_type, exp_type):
	ev = build_controller_evaluator(_params())
	with tempfile.TemporaryDirectory() as td:
		exp = Experiment(_cfg(phase_type, exp_type), ev, lambda _m: None, checkpoint_dir=Path(td))
		result = exp.run()
		# checkpoint dir should hold a .json.gz (proves stats()/serialize() ran clean)
		ckpts = list(Path(td).glob("*.json.gz"))
	return result, ckpts


def test_controller_flow_runs_through_experiment():
	# ga_memory = paradigm B (no training) → fast full-path exercise.
	result, ckpts = _run("ga_memory", ExperimentType.GA)
	assert result is not None, "Experiment.run returned no result"
	assert isinstance(result.best_genome, RecurrentArchGenome), \
		f"best_genome is {type(result.best_genome).__name__}, expected RecurrentArchGenome"
	result.best_genome.assert_valid()
	# recording interface worked end-to-end
	s = result.best_genome.stats()
	assert s["total_neurons"] > 0 and s["total_connections"] > 0
	assert isinstance(result.best_genome.serialize(), dict)
	assert ckpts, "no checkpoint written (stats/serialize path may have failed)"
	print(f"✓ controller_flow_runs_through_experiment "
	      f"(best fitness={result.final_fitness:.3f}, neurons={s['total_neurons']}, "
	      f"checkpoint={ckpts[0].name})")


def test_ids_path_unaffected():
	# A non-controller ExperimentConfig keeps architecture_type default 'tiered' →
	# is_controller is False → none of the controller branches execute.
	cfg = ExperimentConfig(name="x", experiment_type=ExperimentType.GA)
	assert cfg.architecture_type == "tiered" and cfg.phase_type == ""
	print("✓ ids_path_unaffected (controller branches gated off by default)")


if __name__ == "__main__":
	test_ids_path_unaffected()
	test_controller_flow_runs_through_experiment()
	print("\nController worker-integration tests passed.")
