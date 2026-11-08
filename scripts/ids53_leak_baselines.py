"""IDS-53 leak size: RF/XGB on the LEGACY vs TRAIN_ONLY top-20 lists (08/11/2026).

Reuses scripts/verify_ids_baselines.py end to end — its loaders (_load_raw /
the thermometer LoadSpec path), its model hyper-parameters (MODELS), its
val_cal rule (F1-optimal threshold on the 10% VAL) and its TEST scoring
(_evaluate). The only changes: the feature list (legacy TOP20_RF_FEATURES vs
feature_ranking.train_only_features), the model seed (random_state looped),
and n_jobs capped so the live IDS worker keeps its cores.

Usage (niced, n_jobs<=3):
  PYTHONPATH=src/wnn nice -n 15 python scripts/ids53_leak_baselines.py \
      --dataset cicids --encoding raw --seeds 42,43,44,45,46 --out results.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_ids_baselines as vib   # the banked comparator protocol — one implementation

from wnn.ids.feature_ranking import train_only_features
from wnn.ids.loaders import CICIDSLoader, CICIoTLoader, LoadSpec

SPLIT = "random_3way"
RANKING_KEY = {"cicids": "cicids2017", "ciciot": "ciciot2023_neto_subsample", "unsw": "unsw-nb15"}
LOADER = {
	"cicids": CICIDSLoader,
	"ciciot": lambda: CICIoTLoader("neto_subsample"),
}


def feature_list(dataset: str, ranking: str) -> list:
	if ranking == "legacy":
		return list(vib.TOP20[dataset])
	return train_only_features(RANKING_KEY[dataset], SPLIT, "top20")


def load(dataset: str, ranking: str, encoding: str, n_bits: int):
	if encoding == "raw":
		vib.TOP20[dataset] = feature_list(dataset, ranking)
		return vib._load_raw(dataset, SPLIT)
	if dataset == "unsw":
		from wnn.ids.loaders import UNSWLoader
		loader = UNSWLoader()
	else:
		loader = LOADER[dataset]()
	ds = loader.load(LoadSpec(split=SPLIT, n_bits=n_bits, feature_selection="top20", feature_ranking=ranking))
	X_val = ds.X_val.to_numpy_bool() if ds.X_val is not None else None
	return (ds.X_train.to_numpy_bool(), ds.y_train_binary, ds.X_test.to_numpy_bool(), ds.y_test_binary,
	        X_val, ds.y_val_binary)


def model(key: str, seed: int, n_jobs: int):
	name, clf, needs_imputed = vib.MODELS[key]()
	clf.set_params(random_state=seed, n_jobs=n_jobs)
	return name, clf, needs_imputed


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--dataset", required=True, choices=["cicids", "ciciot", "unsw"])
	ap.add_argument("--encoding", default="raw", help="raw | thermo")
	ap.add_argument("--n-bits", type=int, default=8)
	ap.add_argument("--rankings", default="legacy,train_only")
	ap.add_argument("--models", default="rf,xgb")
	ap.add_argument("--seeds", default="42,43,44,45,46")
	ap.add_argument("--n-jobs", type=int, default=3)
	ap.add_argument("--out", required=True)
	args = ap.parse_args()
	seeds = [int(s) for s in args.seeds.split(",")]
	for ranking in args.rankings.split(","):
		feats = feature_list(args.dataset, ranking)
		print(f"\n##### {args.dataset} {SPLIT} {args.encoding} ranking={ranking}: {feats}", flush=True)
		t0 = time.time()
		X_train, y_train, X_test, y_test, X_val, y_val = load(args.dataset, ranking, args.encoding, args.n_bits)
		load_s = time.time() - t0
		sk = vib._impute(X_train, X_test, X_val) if args.encoding == "raw" else (X_train, X_test, X_val)
		for key in args.models.split(","):
			for seed in seeds:
				name, clf, needs_imputed = model(key, seed, args.n_jobs)
				Xtr, Xte, Xva = sk if needs_imputed else (X_train, X_test, X_val)
				t1 = time.time()
				rows = vib._evaluate(f"{name} seed={seed}", clf, Xtr, y_train, Xte, y_test, Xva, y_val)
				wall = time.time() - t1
				with open(args.out, "a") as fh:
					for r in rows:
						fh.write(json.dumps({
							"dataset": args.dataset, "split": SPLIT, "encoding": args.encoding,
							"n_bits": args.n_bits if args.encoding != "raw" else None, "ranking": ranking,
							"model": key, "seed": seed, "mode": r.mode, "thr": r.threshold,
							"f1": r.f1, "fpr": r.fpr, "acc": r.acc, "size_bytes": r.size_bytes,
							"fit_eval_s": wall, "load_s": load_s, "features": feats,
							"n_train": int(len(y_train)), "n_test": int(len(y_test)), "n_val": int(len(y_val)),
						}) + "\n")
				del clf


if __name__ == "__main__":
	main()
