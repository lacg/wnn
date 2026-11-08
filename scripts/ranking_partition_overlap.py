"""IDS-53 (a): how many rows of a top-20 RANKING partition sit in today's test/val.

The top-20 feature lists were ranked by Random Forest on OTHER partitions than
the random_3way splits the S&P cohorts use:

  UNSW-NB15   "standard" (= HF `temporal`) train      dataset.py TOP20_RF_FEATURES
  CICIDS2017  `temporal` train (proxied by `temporal_3way` train, see --ranking-config)
  CIC-IoT     neto-full `random_3way` train          ciciot2023.py TOP20_RF_FEATURES

If those ranking rows reappear in the current test/val, the FEATURE CHOICE saw
the held-out rows — a ranking-level leak (no model weights leak, but the
selection is not train-only). This script counts it, read-only, by row CONTENT
(every shared feature column + the binary label, NaN-aware):

  ranking_in_test / _val  rows of the current test / val whose exact content
                          appears in the ranking partition
  baseline_train_in_test  the same for the CURRENT train partition — the
                          datasets carry natural duplicate flows, so this is the
                          rate a train-only ranking would ALSO "see"; the leak
                          that matters is the excess over it

The ranking partition is streamed in Arrow batches (neto-full train is 37M
rows), so memory stays at the test/val hash sets.

Usage (HF cache; add HF_DATASETS_OFFLINE=1 to stay offline):
  python scripts/ranking_partition_overlap.py unsw-nb15
  python scripts/ranking_partition_overlap.py cicids2017
  python scripts/ranking_partition_overlap.py ciciot2023
"""

import argparse
import json

import numpy as np
import pandas as pd

# dataset -> (current repo, current config, ranking repo, ranking config)
SOURCES = {
	"unsw-nb15": ("lacg030175/UNSW-NB15", "random_3way", "lacg030175/UNSW-NB15", "temporal"),
	"unsw-nb15-temporal": ("lacg030175/UNSW-NB15", "temporal_3way", "lacg030175/UNSW-NB15", "temporal"),
	"cicids2017": ("lacg030175/CICIDS2017", "random_3way", "lacg030175/CICIDS2017", "temporal_3way"),
	"ciciot2023": (
		"lacg030175/CIC-IoT-2023-neto-subsample", "random_3way", "lacg030175/CIC-IoT-2023-neto-full", "random_3way",
	),
}
NON_FEATURES = {"id", "Label", "attack_cat", "Attack_cat", "attack_class", "Label_orig", "is_benign",
	"srcip", "dstip", "sport", "dsport"}


def row_hashes(df: pd.DataFrame, columns: list) -> np.ndarray:
	"""uint64 content hash per row (numeric cast to float64, -0.0 folded)."""
	norm = pd.DataFrame({c: _normalised(df[c]) for c in columns})
	return pd.util.hash_pandas_object(norm, index=False).to_numpy()


def _normalised(series: pd.Series) -> pd.Series:
	if pd.api.types.is_numeric_dtype(series):
		return series.astype(np.float64) + 0.0
	return series.astype(str)


def shared_columns(current, ranking) -> list:
	cols = set(current["train"].column_names) & set(ranking.column_names)
	return sorted(c for c in cols if c not in NON_FEATURES)


def stream_hit_set(ranking, columns: list, targets: set, batch: int) -> set:
	"""Target hashes that occur in the streamed ranking partition."""
	hits = set()
	for chunk in ranking.iter(batch_size=batch):
		hits.update(targets.intersection(row_hashes(pd.DataFrame(chunk)[columns], columns).tolist()))
	return hits


def covered(hashes: np.ndarray, hits: set) -> int:
	return int(np.isin(hashes, np.fromiter(hits, dtype=np.uint64, count=len(hits))).sum()) if hits else 0


def measure(name: str, batch: int) -> dict:
	from datasets import load_dataset
	cur_repo, cur_cfg, rk_repo, rk_cfg = SOURCES[name]
	current = load_dataset(cur_repo, cur_cfg)
	ranking = load_dataset(rk_repo, rk_cfg, split="train")
	cols = shared_columns(current, ranking)
	held = {s: row_hashes(current[s].to_pandas(), cols) for s in ("test", "validation")}
	rk_hits = stream_hit_set(ranking, cols, set(np.concatenate(list(held.values())).tolist()), batch)
	tr_hits = stream_hit_set(current["train"], cols, set(held["test"].tolist()), batch)
	return {
		"dataset": name, "current": f"{cur_repo}:{cur_cfg}", "ranking": f"{rk_repo}:{rk_cfg} train",
		"columns": len(cols), "ranking_rows": ranking.num_rows,
		**{f"{s}_rows": int(len(h)) for s, h in held.items()},
		**{f"ranking_in_{s}": covered(h, rk_hits) for s, h in held.items()},
		"baseline_train_in_test": covered(held["test"], tr_hits),
	}


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("dataset", choices=sorted(SOURCES))
	ap.add_argument("--batch", type=int, default=500_000)
	print(json.dumps(measure(ap.parse_args().dataset, ap.parse_args().batch), indent=1))


if __name__ == "__main__":
	main()
