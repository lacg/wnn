"""IDS-53 (08/11/2026): ids_feature_ranking = legacy | train_only.

The legacy top-20 lists were ranked on partitions that hold 76% (cicids) / 84%
(ciciot) of today's random_3way test/val rows. train_only reads the ranking
re-derived on the flow's own train split (versioned package data); legacy must
stay the exact constant the worker always used.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wnn.ids.feature_ranking import TRAIN_ONLY_VERSION, ranking_file, train_only_features
from wnn.ids.loaders import LoadSpec, get_loader


def _file_top20(key: str, split: str) -> list:
	return json.loads(ranking_file(key, split).read_text())["top20"]


@pytest.mark.parametrize("dataset,key", [
	("cicids2017", "cicids2017"),
	("ciciot2023_neto_subsample", "ciciot2023_neto_subsample"),
	("unsw-nb15", "unsw-nb15"),
])
def test_train_only_reads_the_versioned_file_and_legacy_the_constant(dataset, key):
	loader = get_loader(dataset)
	legacy = loader._ranked_features(LoadSpec(split="random_3way", feature_selection="top20"))
	assert legacy == loader._feature_list(LoadSpec(split="random_3way", feature_selection="top20"))
	train_only = loader._ranked_features(LoadSpec(split="random_3way", feature_selection="top20", feature_ranking="train_only"))
	assert train_only == _file_top20(key, "random_3way")
	assert train_only != list(legacy)[:20]


def test_legacy_lists_are_the_original_constants():
	from wnn.ids import cicids2017, ciciot2023
	assert get_loader("cicids2017")._ranked_features(LoadSpec(feature_selection="top20")) == cicids2017.TOP20_RF_FEATURES
	assert get_loader("ciciot2023_neto_subsample")._ranked_features(LoadSpec(feature_selection="top20")) == ciciot2023.TOP20_RF_FEATURES


def test_train_only_refuses_what_it_cannot_serve():
	with pytest.raises(ValueError, match="top-N"):
		train_only_features("cicids2017", "random_3way", "all")
	with pytest.raises(FileNotFoundError, match="derive it"):
		train_only_features("ciciot2023_neto_full", "random_3way", "top20")
	assert len(train_only_features("cicids2017", "random_3way", "top15")) == 15


def test_param_resolution_and_cache_key():
	from wnn.ram.experiments.ids_param_resolution import resolve_feature_ranking
	from wnn.ram.experiments.test_cache_key import build_test_cache_key
	assert resolve_feature_ranking({}) == "legacy"
	with pytest.raises(ValueError):
		resolve_feature_ranking({"ids_feature_ranking": "train"})
	base = {"ids_dataset": "cicids2017", "ids_feature_selection": "top20"}
	assert build_test_cache_key(base, True) == build_test_cache_key(dict(base, ids_feature_ranking="legacy"), True)
	assert f"|fr={TRAIN_ONLY_VERSION}|" in build_test_cache_key(dict(base, ids_feature_ranking="train_only"), True)


def test_settings_carry_the_ranking_to_the_loader():
	import inspect
	from wnn.ram.experiments.IdsEvaluatorFactory import IdsEvaluatorFactory
	from wnn.ram.experiments.IdsEvaluatorSettings import IdsEvaluatorSettings
	assert IdsEvaluatorSettings.from_params({"ids_feature_ranking": "train_only"}).feature_ranking == "train_only"
	assert "feature_ranking=s.feature_ranking" in inspect.getsource(IdsEvaluatorFactory.load)
