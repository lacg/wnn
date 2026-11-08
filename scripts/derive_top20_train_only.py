"""IDS-53 (b): re-derive the top-20 feature ranking on the CURRENT TRAIN partition only.

The worker's top-20 lists (UNSW dataset.py / CICIDS cicids2017.py / CIC-IoT
ciciot2023.py TOP20_RF_FEATURES) were ranked on OTHER partitions — UNSW
"standard" train, CICIDS "temporal" train, CIC-IoT neto-full train — that can
contain the current random_3way test/val rows (scripts/ranking_partition_overlap.py
measures it). This script ranks on the train split of the flow's own config
and NEVER loads test or validation (Protocol v2: held-out rows must not shape
any choice, the feature selection included).

Method (the one the existing lists used — scripts/derive_top20_neto_full.py):
  RandomForestClassifier(n_estimators=100, random_state=42) on the binary
  label; NaN/Inf median-imputed (train medians); string columns (UNSW proto /
  service / state) ordinal-coded over their sorted train categories;
  features ranked by impurity importance, ties broken by name.
Deterministic: fixed seed, fixed column order, fixed n_jobs-independent forest.

Output: src/wnn/ids/feature_rankings/train_only_v1/<dataset>_<config>.json — the full ranking with
importances, the top-20, and its overlap with the list the worker uses. The
lists are read by the worker ONLY under flow param ids_feature_ranking="train_only"
(wnn/ids/feature_ranking.py); a re-derivation with a changed method must go to a
NEW version directory (train_only_v2) so cached validations never mix lists.

Usage (HF cache; HF_DATASETS_OFFLINE=1 to stay offline; --jobs bounds CPU):
  python scripts/derive_top20_train_only.py unsw-nb15 random_3way
  python scripts/derive_top20_train_only.py cicids2017 random_3way
  python scripts/derive_top20_train_only.py ciciot2023_neto_subsample random_3way
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REPOS = {
	"unsw-nb15": "lacg030175/UNSW-NB15",
	"cicids2017": "lacg030175/CICIDS2017",
	"ciciot2023_neto_subsample": "lacg030175/CIC-IoT-2023-neto-subsample",
	"ciciot2023_neto_full": "lacg030175/CIC-IoT-2023-neto-full",
}
NON_FEATURES = {"id", "label", "Label", "attack_cat", "Attack_cat", "attack_class", "Label_orig", "is_benign",
	"srcip", "dstip", "sport", "dsport"}
# Capture timestamps (UNSW random configs): when the flow was recorded, not what
# it did — attack campaigns ran in fixed windows, so on a RANDOM split they rank
# near the top by identifying the campaign, a leak a deployed IDS cannot use.
# The worker's lists never contained them. --keep-timestamps to rank them anyway.
TIMESTAMPS = {"Stime", "Ltime", "stime", "ltime"}


def worker_top20(dataset: str) -> list:
	"""The list the worker uses today (feature_selection=top20)."""
	from wnn.ids.loaders import LoadSpec, get_loader
	return list(get_loader(dataset)._feature_list(LoadSpec(feature_selection="top20")))[:20]


def load_train(dataset: str, config: str) -> pd.DataFrame:
	"""The TRAIN split only — test/validation are never materialised."""
	from datasets import load_dataset
	return load_dataset(REPOS[dataset], config, split="train").to_pandas()


def design_matrix(df: pd.DataFrame, excluded: set) -> tuple:
	"""(X float32, feature names) — sorted columns, strings ordinal-coded."""
	names = sorted(c for c in df.columns if c not in excluded)
	cols = [_numeric(df[c]) for c in names]
	X = np.column_stack(cols).astype(np.float64)
	X[~np.isfinite(X)] = np.nan
	medians = np.nanmedian(X, axis=0)
	rows, cols_nan = np.where(np.isnan(X))
	X[rows, cols_nan] = np.take(np.nan_to_num(medians), cols_nan)
	return X.astype(np.float32), names


def _numeric(series: pd.Series) -> np.ndarray:
	if pd.api.types.is_numeric_dtype(series):
		return series.to_numpy(dtype=np.float64, na_value=np.nan)
	cats = sorted(series.dropna().astype(str).unique())
	return series.astype(str).map({c: i for i, c in enumerate(cats)}).to_numpy(dtype=np.float64)


def rank(X: np.ndarray, y: np.ndarray, names: list, jobs: int) -> list:
	from sklearn.ensemble import RandomForestClassifier
	rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=jobs).fit(X, y)
	return sorted(zip(names, rf.feature_importances_.tolist()), key=lambda t: (-t[1], t[0]))


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("dataset", choices=sorted(REPOS))
	ap.add_argument("config", help="HF config whose TRAIN split is ranked (e.g. random_3way)")
	ap.add_argument("--jobs", type=int, default=4, help="RF threads (keep low while the worker runs)")
	ap.add_argument("--out-dir", type=Path, default=ROOT / "src" / "wnn" / "ids" / "feature_rankings" / "train_only_v1")
	ap.add_argument("--keep-timestamps", action="store_true", help="rank capture timestamps too (see TIMESTAMPS)")
	a = ap.parse_args()
	df = load_train(a.dataset, a.config)
	excluded = NON_FEATURES if a.keep_timestamps else NON_FEATURES | TIMESTAMPS
	X, names = design_matrix(df, excluded)
	ranking = rank(X, df["label"].to_numpy(dtype=np.int64), names, a.jobs)
	top20, current = [n for n, _ in ranking[:20]], worker_top20(a.dataset)
	out = {
		"dataset": a.dataset, "config": a.config, "partition": "train", "rows": len(df),
		"method": "RandomForestClassifier(n_estimators=100, random_state=42), median-imputed, sorted columns",
		"excluded": sorted(excluded & set(df.columns)),
		"top20": top20, "worker_top20": current, "overlap_with_worker": len(set(top20) & set(current)),
		"added_vs_worker": [f for f in top20 if f not in current],
		"dropped_vs_worker": [f for f in current if f not in top20],
		"ranking": [{"feature": n, "importance": i} for n, i in ranking],
	}
	a.out_dir.mkdir(parents=True, exist_ok=True)
	path = a.out_dir / f"{a.dataset}_{a.config}.json"
	path.write_text(json.dumps(out, indent=1) + "\n")
	print(f"{path}: overlap {out['overlap_with_worker']}/20 with the worker list; +{out['added_vs_worker']} -{out['dropped_vs_worker']}")


if __name__ == "__main__":
	main()
