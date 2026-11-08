"""IDS-51 / IDS-54 (08/11/2026): one precedence rule for experiment settings,
and registered-but-inert flow params handled loudly.

IDS-51: worker.py read `exp_data.get(k) or params...` — the experiment row's
creation-time snapshot BEAT the live flow param (PUB50-ciciot ran ce/acc
0.2/0.1 under params 0.1/0.2) and `or` treated 0.0 as unset. Live params now
win for every key, `is not None`, with a loud warning on disagreement.
IDS-54: decode_mode / ids_arch_type are noted at their default and refused
otherwise; mutation_rate is wired into GAConfig / TSConfig; max_bit_delta is
wired into the Rust MutationConfig (worker ABI 26).
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wnn.ram.experiments.experiment import ExperimentType
from wnn.ram.experiments.experiment_setting_resolver import ExperimentSettingResolver
from wnn.ram.experiments.inert_params import enforce_inert_params, enforce_phase_order, warn_inapplicable_params
from wnn.ram.experiments.params import KNOWN_PARAMS, validate_flow_params
from wnn.ram.experiments.worker import FlowWorker
from wnn.ram.fitness import FitnessCalculatorType


def _worker(logs: list) -> FlowWorker:
	w = FlowWorker.__new__(FlowWorker)
	w._log = logs.append
	return w


def _ga_row(**cols) -> dict:
	row = {"name": "GA Neurons", "phase_type": "ga_neurons", "fitness_calculator": "harmonic_rank",
		"fitness_weight_ce": 1.0, "fitness_weight_acc": 1.0, "population_size": 50, "max_iterations": 250}
	row.update(cols)
	return row


IDS_PARAMS = {"architecture_type": "ids", "min_bits": 8, "max_bits": 16, "min_neurons": 5, "max_neurons": 50}


def _build(params: dict, rows: list, logs: list):
	return _worker(logs)._build_experiment_configs(rows, dict(IDS_PARAMS, **params), 2)


def test_live_params_beat_the_row_snapshot_and_warn():
	logs = []
	cfg = _build({"fitness_weight_ce": 0.1, "fitness_weight_acc": 0.2}, [_ga_row(fitness_weight_ce=0.2, fitness_weight_acc=0.1)], logs)[0]
	assert (cfg.fitness_weight_ce, cfg.fitness_weight_acc) == (0.1, 0.2)
	assert any("fitness_weight_ce=0.2 DISAGREES" in m for m in logs)
	assert any("fitness_weight_acc=0.1 DISAGREES" in m for m in logs)


def test_zero_is_a_value_not_unset():
	cfg = _build({"fitness_weight_ce": 0.0, "fitness_weight_f1": 0.0}, [_ga_row(fitness_weight_ce=0.0)], [])[0]
	assert cfg.fitness_weight_ce == 0.0
	row_zero = _build({}, [_ga_row(fitness_weight_ce=0.0)], [])[0]
	assert row_zero.fitness_weight_ce == 0.0, "row 0.0 with no live param must not become the 1.0 default"


def test_calculator_population_and_generations_follow_live_params():
	logs = []
	params = {"fitness_calculator": "ce", "population_size": 40, "ga_generations": 120}
	cfg = _build(params, [_ga_row()], logs)[0]
	assert cfg.fitness_calculator_type == FitnessCalculatorType.CE
	assert (cfg.population_size, cfg.generations) == (40, 120)
	assert sum("DISAGREES" in m for m in logs) == 3


def test_explicit_per_experiment_generations_still_win():
	row = _ga_row(max_iterations=30, params={"generations": 30})
	cfg = _build({"ga_generations": 250}, [row], [])[0]
	assert cfg.generations == 30


def test_matching_rows_are_silent_and_unchanged():
	logs = []
	params = {"fitness_weight_ce": 0.2, "fitness_weight_acc": 0.1, "population_size": 50, "ga_generations": 250}
	cfg = _build(params, [_ga_row(fitness_weight_ce=0.2, fitness_weight_acc=0.1)], logs)[0]
	assert (cfg.fitness_weight_ce, cfg.fitness_weight_acc, cfg.generations) == (0.2, 0.1, 250)
	assert not any("DISAGREES" in m for m in logs)


def test_ts_gets_ts_iterations_and_mutation_rate_reaches_the_config():
	ts = _ga_row(name="TS Neurons", phase_type="ts_neurons", max_iterations=77)
	cfg = _build({"ts_iterations": 77, "mutation_rate": 0.3}, [ts], [])[0]
	assert cfg.experiment_type == ExperimentType.TS
	assert (cfg.iterations, cfg.mutation_rate) == (77, 0.3)
	assert _build({}, [_ga_row()], [])[0].mutation_rate == 0.1, "absent = the OptimizationConfig default"


def test_random_search_on_ts_is_refused():
	ts = _ga_row(name="TS", phase_type="ts_neurons")
	with pytest.raises(ValueError, match="random_search"):
		_build({"random_search": True}, [ts], [])
	_build({"random_search": True}, [{"name": "Grid", "phase_type": "grid_search"}, _ga_row()], [])


def test_resolver_falls_back_to_row_then_default():
	r = ExperimentSettingResolver({}, {"fitness_weight_ce": 0.7}, print)
	assert r.resolve("fitness_weight_ce", default=1.0) == 0.7
	assert r.resolve("fitness_weight_acc", default=1.0) == 1.0


# ---- IDS-54 ----------------------------------------------------------------

def test_inert_keys_accept_default_with_a_note():
	logs = []
	enforce_inert_params({"ids_arch_type": "tiered", "decode_mode": "argmax",
		"fitness_ce_anchor_normalized": 0.2128}, logs.append)
	assert sum("note:" in m for m in logs) == 3


@pytest.mark.parametrize("key,value", [("ids_arch_type", "bitwise"), ("decode_mode", "benign_margin")])
def test_inert_keys_refuse_non_default(key, value):
	with pytest.raises(ValueError, match=key):
		validate_flow_params({"architecture_type": "ids", key: value}, log=lambda m: None)


def test_inert_keys_are_not_registered_as_working_params():
	for key in ("decode_mode", "ids_arch_type"):
		assert key not in KNOWN_PARAMS
	for key in ("mutation_rate", "max_bit_delta", "fitness_ce_anchor_normalized", "ids_feature_ranking"):
		assert key in KNOWN_PARAMS
	validate_flow_params({"architecture_type": "ids", "max_bit_delta": 3}, log=lambda m: None)


def test_max_bit_delta_reaches_the_rust_search():
	"""IDSEvaluator forwards Flow._propagate_max_bit_delta's value to both Rust searches."""
	import inspect
	from wnn.ram.architecture.ids_evaluator import IDSEvaluator
	for name in ("search_offspring", "search_neighbors"):
		assert 'max_bit_delta=getattr(self, "_max_bit_delta", 0)' in inspect.getsource(getattr(IDSEvaluator, name))


def test_phase_order_default_ok_non_default_must_match_rows():
	enforce_phase_order({"phase_order": "neurons_first"}, ["grid_search", "ga_neurons"])
	enforce_phase_order({"phase_order": "bits_first"}, ["grid_search", "ga_bits", "ga_neurons"])
	with pytest.raises(ValueError, match="phase_order"):
		enforce_phase_order({"phase_order": "bits_first"}, ["grid_search", "ga_neurons"])


def test_keys_with_no_effect_for_the_architecture_warn():
	logs = []
	noop = warn_inapplicable_params({"architecture_type": "controller", "ids_k_folds": 5, "tier_config": "x",
		"controller_steps": 100, "seed": 1}, logs.append)
	assert noop == ["ids_k_folds", "tier_config"]
	assert warn_inapplicable_params({"architecture_type": "ids", "ids_k_folds": 5, "seed": 1}, logs.append) == []
