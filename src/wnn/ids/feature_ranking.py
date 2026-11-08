"""
Which top-N feature ranking a flow encodes with (IDS-53, flow param
`ids_feature_ranking`, 08/11/2026).

  legacy      (default) the hard-coded TOP20_RF_FEATURES lists — ranked on OTHER
              partitions (UNSW "standard" train, CICIDS "temporal" train, CIC-IoT
              neto-full train) that hold 76% (cicids) / 84% (ciciot) of today's
              random_3way test/val rows (scripts/ranking_partition_overlap.py).
  train_only  the ranking re-derived on the flow's OWN train split only
              (scripts/derive_top20_train_only.py), read from the versioned
              package data `feature_rankings/<TRAIN_ONLY_VERSION>/`.

Why package data next to the loaders (not docs/ or the git-ignored data/): the
worker imports the package from the tree, so the file ships with the code that
reads it; and the directory IS the version — a re-derivation goes to a new
directory, bumps TRAIN_ONLY_VERSION, and so re-keys the validation cache
(validation_cache_key `fr=`) instead of silently changing a list in place.
"""

import json
from pathlib import Path

FEATURE_RANKINGS = ("legacy", "train_only")
TRAIN_ONLY_VERSION = "train_only_v1"
_DIR = Path(__file__).resolve().parent / "feature_rankings" / TRAIN_ONLY_VERSION
# feature_selection -> how many of the ranking it takes (the MI lists and "all"
# do not use the RF ranking, so train_only is refused there).
_TOP_N = {"top15": 15, "top20": 20, "top25": 25, "top20_split": 20}


def ranking_file(ranking_key: str, split: str) -> Path:
	return _DIR / f"{ranking_key}_{split}.json"


def train_only_features(ranking_key: str, split: str, feature_selection: str) -> list:
	"""The train-only top-N for (dataset, split); refuses what it cannot serve."""
	if feature_selection not in _TOP_N:
		raise ValueError(
			f"ids_feature_ranking='train_only' needs a top-N RF feature_selection "
			f"{sorted(_TOP_N)}; got {feature_selection!r} (it would have no effect)"
		)
	path = ranking_file(ranking_key, split)
	if not path.exists():
		raise FileNotFoundError(
			f"no train-only ranking for {ranking_key} / {split} ({path}); derive it with "
			f"scripts/derive_top20_train_only.py {ranking_key} {split}"
		)
	ranking = [r["feature"] for r in json.loads(path.read_text())["ranking"]]
	return ranking[:_TOP_N[feature_selection]]


def provenance(ranking: str, ranking_key: str, split: str, features: list) -> str:
	"""One log line naming the list a flow encodes with."""
	source = ranking_file(ranking_key, split).name + f" ({TRAIN_ONLY_VERSION})" if ranking == "train_only" else "TOP20_RF_FEATURES (legacy)"
	return f"  [FEATURES] ranking={ranking} source={source} top={len(features)}: {features[:20]}"
