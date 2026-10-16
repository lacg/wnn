"""IdsEvaluatorFactory builds exactly the evaluators worker._create_ids_evaluators built.

The worker now delegates to the factory (shared with the FPGA exporter), so the
factory must hand IDSEvaluator the same arguments the inline worker code did:
captured here with a fake dataset loader / evaluator against the SR1 flow params.
"""
import os

import pytest

from wnn.ram.experiments.IdsEvaluatorFactory import IdsEvaluatorFactory
from wnn.ram.experiments.IdsEvaluatorSettings import OI_ENV, IdsEvaluatorSettings

SR1 = {
	"ids_dataset": "unsw-nb15", "ids_split": "temporal_3way", "ids_n_bits": 16, "ids_feature_selection": "top20",
	"ids_classification": "binary", "ids_single_cluster": True, "ids_k_folds": 5, "ids_kfold_per_gen": 5,
	"ids_num_parts": 5, "neuron_sample_rate": 1.0, "balance_classes": True, "seed": 20419,
	"wnn_order_independent_train": True, "fitness_weight_ce": 0.525, "fitness_weight_f1": 0.15,
	"fitness_weight_fpr": 0.15, "fitness_weight_acc": 0.175,
}


class _Dataset:
	y_train_binary = [0, 1] * 4
	y_test_binary = [0, 1] * 2

	def __init__(self):
		self.X_train = type("M", (), {"shape": (8, 288), "__len__": lambda s: 8})()
		self.X_test = type("M", (), {"shape": (4, 288), "__len__": lambda s: 4})()


class _Evaluator:
	built: list = []

	def __init__(self, **args):
		self.args = args
		self._cache = object()
		self.weights = None
		_Evaluator.built.append(self)

	def set_fitness_weights(self, *w):
		self.weights = w


@pytest.fixture
def captured(monkeypatch):
	import wnn.ids
	import wnn.ram.architecture.ids_evaluator as ids_evaluator
	loads = []
	monkeypatch.setattr(wnn.ids, "load_ids_dataset", lambda name, **a: loads.append((name, a)) or _Dataset())
	monkeypatch.setattr(ids_evaluator, "IDSEvaluator", _Evaluator)
	_Evaluator.built = []
	return loads


def test_flat_binary_matches_the_worker(captured):
	optimizer, test = IdsEvaluatorFactory(lambda m: None).create(dict(SR1))
	name, load = captured[0]
	assert name == "unsw-nb15" and load["split"] == "temporal_3way" and load["n_bits"] == 16
	assert load["feature_selection"] == "top20" and load["encoded_storage"] == "memory"
	common = dict(classification="binary", empty_value=0.0, memory_mode=2, seed=20419, run_seed=20419,
		coverage_aware=False, write_rule="lattice", num_negatives=None, neuron_sample_rate=1.0, balance_classes=True,
		single_cluster=True, undersample_majority=False, flip_labels=False, class_weight_multiplier=1.0)
	assert optimizer.args == {**common, "dataset": optimizer.args["dataset"], "num_parts": 5, "k_folds": 5,
		"kfold_per_gen": 5, "reuse_cache": None}
	assert test.args == {**common, "dataset": optimizer.args["dataset"], "num_parts": 1, "k_folds": 1,
		"kfold_per_gen": 1, "reuse_cache": optimizer._cache}
	assert optimizer.weights == test.weights == (0.525, 0.15, 0.15, 0.175)


def test_settings_defaults_are_the_workers():
	s = IdsEvaluatorSettings.from_params({})
	assert (s.neuron_sample_rate, s.memory_mode_name, s.memory_mode, s.k_folds, s.num_parts) == (0.25, "QUAD_WEIGHTED", 2, 5, 5)
	assert (s.split, s.n_bits, s.feature_selection, s.seed, s.single_cluster) == ("standard", 8, "all", 42, False)
	assert IdsEvaluatorSettings.from_params({"ids_suppress_negatives": True}).num_negatives == 0
	assert IdsEvaluatorSettings.from_params({"memory_mode": "PLN"}).empty_value == 0.5


@pytest.mark.parametrize("value, enabled", [(True, True), (1, True), ("true", True), ("1", True), (False, False), (None, False)])
def test_oi_env(monkeypatch, value, enabled):
	monkeypatch.delenv(OI_ENV, raising=False)
	previous, on = IdsEvaluatorSettings.apply_oi_env({"wnn_order_independent_train": value})
	assert (previous, on) == (None, enabled)
	assert (os.environ.get(OI_ENV) == "1") == enabled
