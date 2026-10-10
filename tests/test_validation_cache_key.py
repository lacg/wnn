"""Cross-flow validation-cache key (PAPER-CRITICAL fix, 10/10/2026).

The old key ({ds}_{nb}b_{sp}{_raw}{_inv}{_oi}) omitted memory_mode, feature
selection, classification and the trainer ABI, so same-seed grid searches let a
QUAD flow inherit a QSR flow's final TEST row (B34-CTRL r = cached copy of void
flow 5929). These tests pin the new key's contents; the shared fixture
tests/fixtures/validation_cache_keys.json is ALSO replayed by the dashboard's
Rust tests (dashboard/src/db/validations.rs), so both sides agree byte-for-byte.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wnn.ram.experiments.dashboard_client import DashboardClient
from wnn.ram.experiments.experiment import Experiment, ExperimentConfig, ExperimentType
from wnn.ram.experiments.ids_param_resolution import oi_env_enabled, resolve_memory_mode
from wnn.ram.experiments.params import KNOWN_PARAMS
from wnn.ram.experiments.validation_cache_key import build_validation_cache_key
from wnn.ram.experiments.validation_cache_scope import ValidationCacheScope

FIXTURE = json.loads((ROOT / "tests" / "fixtures" / "validation_cache_keys.json").read_text())
CASES = {c["name"]: c for c in FIXTURE["cases"]}
BASE = dict(CASES["quad_default"]["params"])


def _key(delta: dict, oi: bool = True) -> str:
	return build_validation_cache_key(dict(BASE, **delta), oi)


class _RecordingParams(dict):
	"""dict that records every key read via .get()."""

	def __init__(self, data: dict):
		super().__init__(data)
		self.read: set = set()

	def get(self, key, default=None):
		self.read.add(key)
		return super().get(key, default)


# ---- fixture parity (the Rust tests replay these exact strings) -------------

def test_builder_reproduces_every_fixture_key():
	for case in FIXTURE["cases"]:
		assert build_validation_cache_key(case["params"], case["oi_enabled"]) == case["key"], case["name"]


def test_fixture_cases_are_distinct_except_documented_aliases():
	keys = {name: c["key"] for name, c in CASES.items()}
	assert keys["quad_default"] == keys["quad_explicit"]
	assert keys["neto_default_inv"] == keys["neto_explicit_inv"]
	aliases = {"quad_explicit", "neto_explicit_inv"}
	distinct = [k for n, k in keys.items() if n not in aliases]
	assert len(set(distinct)) == len(distinct)


# ---- memory mode ------------------------------------------------------------

def test_absent_explicit_and_unrecognised_mode_all_resolve_to_quad():
	assert resolve_memory_mode({}) == "QUAD_WEIGHTED"
	assert resolve_memory_mode({"memory_mode": None}) == "QUAD_WEIGHTED"
	assert resolve_memory_mode({"memory_mode": "quad_weighted"}) == "QUAD_WEIGHTED"
	assert _key({}) == _key({"memory_mode": "QUAD_WEIGHTED"}) == _key({"memory_mode": "bogus"})


def test_every_distinct_trainer_mode_gets_its_own_key():
	modes = ["TERNARY", "QUAD_BINARY", "QUAD_WEIGHTED", "BINARY", "QSR", "PLN"]
	assert len({_key({"memory_mode": m}) for m in modes}) == len(modes)


# ---- params that change a fixed genome's validation must split the key ------

def test_validation_relevant_params_split_the_key():
	deltas = [
		{"ids_feature_selection": "top20"}, {"ids_classification": "multi"},
		{"ids_single_cluster": False}, {"seed": 7}, {"neuron_sample_rate": 0.5},
		{"ids_coverage_aware": True}, {"ids_suppress_negatives": True},
		{"balance_classes": True}, {"undersample_majority": True}, {"flip_labels": True},
		{"class_weight_multiplier": 2.0}, {"ids_rest_bits": 4}, {"ids_auto_max_bits": 16},
		{"ids_streaming": True}, {"ids_raw": True}, {"ids_invalid_encoding": "single_bit"},
		{"ids_n_bits": 16}, {"ids_split": "temporal_3way"}, {"ids_dataset": "cicids2017"},
		{"fitness_weight_f1": 1.0},  # single-cluster test-evaluator threshold weights
	]
	base = _key({})
	for delta in deltas:
		assert _key(delta) != base, delta
	assert _key({}, oi=False) != base


def test_search_only_params_do_not_split_the_key():
	deltas = [
		{"patience": 9}, {"ga_generations": 500}, {"population_size": 10},
		{"fitness_aggregation": "zscore"}, {"fitness_calculator": "harmonic_rank"},
		{"ids_k_folds": 3}, {"ids_num_parts": 2}, {"ids_kfold_per_gen": 1},
		{"ids_encoded_storage": "memmap"}, {"wnn_hybrid_speed_ratio": 2},
		{"ids_val_fraction": 0.1},  # flat classification: the test evaluator ignores it
	]
	base = _key({})
	for delta in deltas:
		assert _key(delta) == base, delta


def test_threshold_weights_only_count_when_the_worker_applies_them():
	# multiclass (not single-cluster): worker never calls set_fitness_weights
	multi = {"ids_classification": "multi", "ids_single_cluster": False}
	assert _key(dict(multi, fitness_weight_f1=1.0)) == _key(multi)


def test_hierarchical_val_fraction_and_seed_matter():
	h = {"ids_classification": "hierarchical", "ids_single_cluster": False}
	assert _key(dict(h, ids_val_fraction=0.1)) != _key(h)
	assert _key(dict(h, seed=1)) != _key(h)


def test_every_param_the_builder_reads_is_registered():
	params = _RecordingParams(BASE)
	build_validation_cache_key(params, True)
	assert params.read - KNOWN_PARAMS == set()


# ---- OI flag mirrors the Rust env parse --------------------------------------

def test_oi_env_parse_mirrors_rust():
	assert oi_env_enabled("1") and oi_env_enabled("true") and oi_env_enabled("TRUE")
	assert not oi_env_enabled(None) and not oi_env_enabled("0") and not oi_env_enabled("yes")


# ---- experiment-level scope ---------------------------------------------------

def _config(w_ce: float, w_f1: float) -> ExperimentConfig:
	return ExperimentConfig(
		name="t", experiment_type=ExperimentType.GA,
		fitness_weight_ce=w_ce, fitness_weight_f1=w_f1,
	)


def test_experiment_fitness_weights_split_the_scope():
	scope = ValidationCacheScope(_key({}), 14)
	a = Experiment._experiment_validation_scope(scope, _config(1.0, 0.0))
	b = Experiment._experiment_validation_scope(scope, _config(2.0, 0.0))
	c = Experiment._experiment_validation_scope(scope, _config(1, 0))
	assert a.key != b.key and a.key == c.key and a.worker_abi == 14
	assert a.key.startswith(scope.key + "|ew=")
	assert Experiment._experiment_validation_scope(None, _config(1.0, 0.0)) is None


# ---- client wiring ------------------------------------------------------------

def test_no_scope_means_no_lookup():
	client = DashboardClient.__new__(DashboardClient)
	client._request = lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not query"))
	assert client.check_cached_validation("abc", None) is None


def test_check_params_carry_key_abi_and_legacy_guard():
	scope = ValidationCacheScope(_key({}), 14)
	p = DashboardClient._cache_check_params("abc", scope)
	assert p == {"genome_hash": "abc", "cache_key": scope.key, "worker_abi": 14, "dataset_key": scope.key}


# ---- worker wiring ------------------------------------------------------------

def _worker():
	from wnn.ram.experiments.worker import FlowWorker
	w = FlowWorker.__new__(FlowWorker)
	w._log = lambda msg: None
	return w


def test_worker_scope_uses_installed_abi_and_effective_oi(monkeypatch):
	import wnn.accel
	monkeypatch.setattr(wnn.accel, "installed_abi", lambda: 14)
	monkeypatch.setenv("WNN_ORDER_INDEPENDENT_TRAIN", "1")
	scope = _worker()._build_validation_scope(dict(BASE))
	assert scope == ValidationCacheScope(CASES["quad_default"]["key"], 14)
	monkeypatch.delenv("WNN_ORDER_INDEPENDENT_TRAIN")
	assert _worker()._build_validation_scope(dict(BASE)).key == CASES["oi_off"]["key"]


def test_worker_without_accelerator_disables_the_cache(monkeypatch):
	import wnn.accel
	monkeypatch.setattr(wnn.accel, "installed_abi", lambda: None)
	assert _worker()._build_validation_scope(dict(BASE)) is None
