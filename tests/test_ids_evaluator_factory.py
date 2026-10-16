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
		coverage_aware=False, write_rule="lattice", two_cluster_fitness_f1="macro", offspring_folds="all", num_negatives=None, neuron_sample_rate=1.0, balance_classes=True,
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


@pytest.mark.parametrize("folds, per_gen", [("all", 5), ("phase", 1)])
def test_hierarchical_stages_follow_the_offspring_fold_rule(monkeypatch, captured, folds, per_gen):
	"""Hierarchical stage OPTIMIZERS get every fold per generation under the
	strict ids_offspring_folds='all' (they used to run IDSEvaluator's default
	single fold) and keep that legacy single fold under 'phase'; stage TEST
	evaluators stay flat (k_folds=1). Both carry the fold rule."""
	import wnn.ids
	monkeypatch.setattr(wnn.ids, "split_train_validation", lambda ds, **kw: (ds, ds), raising=False)
	params = {**SR1, "ids_classification": "hierarchical", "ids_offspring_folds": folds}
	s = IdsEvaluatorSettings.from_params(params)
	factory = IdsEvaluatorFactory(lambda m: None)
	_Evaluator.built = []
	opt, test = factory._stage_pair(_Dataset(), s, "binary", "S0")
	assert (opt.args["k_folds"], opt.args["kfold_per_gen"], opt.args["offspring_folds"]) == (5, per_gen, folds)
	assert (test.args["k_folds"], test.args["kfold_per_gen"], test.args["offspring_folds"]) == (1, 1, folds)


def test_offspring_folds_resolution():
	from wnn.ram.experiments.ids_param_resolution import resolve_kfold_per_gen, resolve_offspring_folds
	from wnn.ram.experiments.params import KNOWN_PARAMS
	assert "ids_offspring_folds" in KNOWN_PARAMS
	assert resolve_offspring_folds({}) == "all"
	assert resolve_offspring_folds({"ids_offspring_folds": "phase", "ids_kfold_per_gen": 1}) == "phase"
	with pytest.raises(ValueError, match="expected one of"):
		resolve_offspring_folds({"ids_offspring_folds": "some"})
	with pytest.raises(ValueError, match="ids_kfold_per_gen=1"):
		resolve_offspring_folds({"ids_k_folds": 5, "ids_kfold_per_gen": 1})
	# Absent kfold_per_gen: every fold under strict, the historical 1 under phase.
	assert resolve_kfold_per_gen({"ids_k_folds": 5}) == 5
	assert resolve_kfold_per_gen({"ids_k_folds": 5, "ids_offspring_folds": "phase"}) == 1
	assert resolve_kfold_per_gen({"ids_k_folds": 5, "ids_kfold_per_gen": 5}) == 5
