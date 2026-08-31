"""
Classical ML baselines for all three IDS datasets (RF + XGBoost).

Runs on the same random_3way 80/20 splits and top-20 features used by
the WNN experiments. Each dataset uses raw numeric features (the natural
format for RF/XGBoost) and optionally 8-bit thermometer-encoded features.

Usage:
    source wnn/bin/activate
    python scripts/run_all_baselines.py
    python scripts/run_all_baselines.py --dataset ciciot   # single dataset
    python scripts/run_all_baselines.py --dataset unsw
    python scripts/run_all_baselines.py --dataset cicids
    python scripts/run_all_baselines.py --skip-thermo      # raw only (faster)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "wnn"))


def compute_metrics(y_true, y_pred):
	f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
	acc = accuracy_score(y_true, y_pred)
	cm = confusion_matrix(y_true, y_pred)
	tn, fp, fn, tp = cm.ravel()
	fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
	return {"f1": f1, "fpr": fpr, "acc": acc, "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def run_classifier(name, clf, X_train, y_train, X_eval, y_eval):
	print(f"\n  Training {name}...")
	t0 = time.time()
	clf.fit(X_train, y_train)
	train_s = time.time() - t0

	t0 = time.time()
	y_pred = clf.predict(X_eval)
	infer_s = time.time() - t0

	m = compute_metrics(y_eval, y_pred)
	m["train_s"] = train_s
	m["infer_s"] = infer_s

	print(f"    F1:  {m['f1']*100:.2f}%  |  FPR: {m['fpr']*100:.2f}%  |  Acc: {m['acc']*100:.2f}%")
	print(f"    Confusion: TN={m['tn']:,} FP={m['fp']:,} FN={m['fn']:,} TP={m['tp']:,}")
	print(f"    Train: {train_s:.1f}s  |  Infer: {infer_s:.2f}s ({1e6*infer_s/len(X_eval):.1f} us/sample)")
	return m


# ── Dataset loaders (raw numeric features) ────────────────────────────

def load_raw_dataset(dataset_name, split="random_3way", eval_split="merged"):
	"""Load raw numeric features from HuggingFace, top-20 only."""
	from datasets import load_dataset

	HF_REPOS = {
		"ciciot": "lacg030175/CIC-IoT-2023",
		"unsw": "lacg030175/UNSW-NB15",
		"cicids": "lacg030175/CICIDS2017",
	}

	# Top-20 features per dataset
	TOP20 = {
		"ciciot": [
			"HTTPS", "Number", "Time_To_Live", "Max", "ack_flag_number",
			"Rate", "IAT", "ack_count", "Header_Length", "Min",
			"Variance", "psh_flag_number", "Tot sum", "Std", "Tot size",
			"syn_count", "AVG", "rst_flag_number", "DNS", "rst_count",
		],
		"unsw": [
			"ct_dst_sport_ltm", "ct_src_dport_ltm", "ct_srv_dst", "ct_state_ttl",
			"dinpkt", "dmean", "dpkts", "dttl", "dur", "rate",
			"sbytes", "sinpkt", "sjit", "sloss", "smean",
			"sttl", "stcpb", "tcprtt", "trans_depth", "ct_srv_src",
		],
		"cicids": [
			"Bwd Packet Length Std", "Destination Port", "Packet Length Std",
			"Bwd Packet Length Max", "Avg Bwd Segment Size", "Average Packet Size",
			"Packet Length Mean", "Max Packet Length", "Fwd Packet Length Max",
			"Fwd Packet Length Mean", "Bwd Packet Length Mean", "Subflow Fwd Bytes",
			"Total Length of Fwd Packets", "Fwd Packet Length Std", "Init_Win_bytes_forward",
			"Packet Length Variance", "Total Length of Bwd Packets", "Subflow Bwd Bytes",
			"Init_Win_bytes_backward", "min_seg_size_forward",
		],
	}

	repo = HF_REPOS[dataset_name]
	features_list = TOP20[dataset_name]

	print(f"\nLoading {dataset_name} ({split}) from {repo}...")
	ds = load_dataset(repo, split)

	df_train = ds["train"].to_pandas()

	# Filter to features that actually exist in this dataset
	features = [f for f in features_list if f in df_train.columns]
	print(f"  Features: {len(features)} of {len(features_list)} top-20 found")

	# Label column
	label_col = "label" if "label" in df_train.columns else "Label"

	X_train = df_train[features].values.astype(np.float32)
	y_train = df_train[label_col].values.astype(np.int32)

	# WHICH PARTITION IS THE EVAL SET (added 31/08/2026).
	# "merged" (the historical default) concatenates test+validation into one 20% eval
	# set — a 2-way evaluation carried out on a 3-way dataset. Protocol v2 does NOT
	# merge them: the WNN calibrates its threshold modes on VAL and reports on the 10%
	# TEST partition alone, so a baseline meant to sit beside a Protocol-v2 WNN number
	# must be scored on TEST ONLY or the two are measured on different sets.
	# Default stays "merged" so previously banked baselines reproduce bit-for-bit.
	if eval_split != "merged" and eval_split in ds:
		df_eval = ds[eval_split].to_pandas()
		print(f"  Eval = {eval_split} only: {len(df_eval):,} samples")
	elif "validation" in ds:
		df_test = ds["test"].to_pandas()
		df_val = ds["validation"].to_pandas()
		df_eval = pd.concat([df_test, df_val], ignore_index=True)
		print(f"  Merged test ({len(df_test):,}) + val ({len(df_val):,}) = {len(df_eval):,} eval")
	else:
		df_eval = ds["test"].to_pandas()
		print(f"  Test: {len(df_eval):,} samples")

	X_eval = df_eval[features].values.astype(np.float32)
	y_eval = df_eval[label_col].values.astype(np.int32)

	n_attack_tr = y_train.sum()
	n_normal_tr = (y_train == 0).sum()
	print(f"  Train: {X_train.shape} ({n_attack_tr:,} attack / {n_normal_tr:,} normal)")
	print(f"  Eval:  {X_eval.shape} ({y_eval.sum():,} attack / {(y_eval==0).sum():,} normal)")

	return X_train, y_train, X_eval, y_eval


def load_thermo_dataset(dataset_name, n_bits=8):
	"""Load thermometer-encoded features (same as WNN)."""
	if dataset_name == "ciciot":
		from wnn.ids.ciciot2023 import load_ciciot2023
		ds = load_ciciot2023(n_bits=n_bits, split="random_3way", feature_selection="top20")
	elif dataset_name == "unsw":
		from wnn.ids.dataset import load_unsw_nb15
		ds = load_unsw_nb15(n_bits=n_bits, split="random", feature_selection="top20")
	elif dataset_name == "cicids":
		from wnn.ids.cicids2017 import load_cicids2017
		ds = load_cicids2017(n_bits=n_bits, split="random", feature_selection="top20")
	else:
		raise ValueError(f"Unknown dataset: {dataset_name}")

	if ds.X_val is not None:
		X_eval = np.concatenate([ds.X_test, ds.X_val])
		y_eval = np.concatenate([ds.y_test_binary, ds.y_val_binary])
	else:
		X_eval = ds.X_test
		y_eval = ds.y_test_binary

	print(f"  Thermo {n_bits}b: train {ds.X_train.shape}, eval {X_eval.shape}")
	return ds.X_train, ds.y_train_binary, X_eval, y_eval


def run_dataset(dataset_name, skip_thermo=False, split="random_3way", eval_split="merged"):
	"""Run all baselines for one dataset."""
	DISPLAY = {"ciciot": "CIC-IoT-2023", "unsw": "UNSW-NB15", "cicids": "CICIDS2017"}
	display = DISPLAY[dataset_name]

	print(f"\n{'='*70}")
	print(f"  {display} — {split}, eval={eval_split} (Top-20 Features)")
	print(f"{'='*70}")

	results = {}

	# Raw features
	X_tr, y_tr, X_ev, y_ev = load_raw_dataset(dataset_name, split=split, eval_split=eval_split)

	results["rf_raw"] = run_classifier(
		"RF (raw)", RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
		X_tr, y_tr, X_ev, y_ev)
	results["xgb_raw"] = run_classifier(
		"XGBoost (raw)", XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
			n_jobs=-1, random_state=42, eval_metric="logloss", verbosity=0),
		X_tr, y_tr, X_ev, y_ev)

	# Thermometer-encoded
	if not skip_thermo:
		X_tr_t, y_tr_t, X_ev_t, y_ev_t = load_thermo_dataset(dataset_name)
		results["rf_thermo"] = run_classifier(
			"RF (8b thermo)", RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
			X_tr_t, y_tr_t, X_ev_t, y_ev_t)
		results["xgb_thermo"] = run_classifier(
			"XGBoost (8b thermo)", XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
				n_jobs=-1, random_state=42, eval_metric="logloss", verbosity=0),
			X_tr_t, y_tr_t, X_ev_t, y_ev_t)

	# Summary
	print(f"\n  {'Model':<25s} {'F1':>8s} {'FPR':>8s} {'Acc':>8s} {'Train':>7s}")
	print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*7}")
	for key, label in [("rf_raw", "RF (raw)"), ("xgb_raw", "XGBoost (raw)"),
		("rf_thermo", "RF (8b thermo)"), ("xgb_thermo", "XGBoost (8b thermo)")]:
		if key not in results:
			continue
		r = results[key]
		print(f"  {label:<25s} {r['f1']*100:>7.2f}% {r['fpr']*100:>7.2f}% {r['acc']*100:>7.2f}% {r['train_s']:>6.1f}s")

	return {dataset_name: results}


def main():
	parser = argparse.ArgumentParser(description="IDS ML baselines (RF + XGBoost)")
	parser.add_argument("--dataset", choices=["ciciot", "unsw", "cicids", "all"], default="all")
	parser.add_argument("--skip-thermo", action="store_true", help="Skip thermometer-encoded runs")
	parser.add_argument("--split", default="random_3way",
	                    help="HF config: random_3way | temporal_3way | random | temporal")
	parser.add_argument("--eval-split", default="merged", choices=["test", "validation", "merged"],
	                    help="Which partition to score on. test = Protocol-v2 comparable.")
	args = parser.parse_args()

	print("=" * 70)
	print("IDS Classical ML Baselines — RAID 2026 Paper")
	print("=" * 70)

	datasets = ["unsw", "cicids", "ciciot"] if args.dataset == "all" else [args.dataset]
	all_results = {}

	for ds in datasets:
		all_results.update(run_dataset(ds, skip_thermo=args.skip_thermo,
			split=args.split, eval_split=args.eval_split))

	# Final cross-dataset summary
	if len(datasets) > 1:
		DISPLAY = {"ciciot": "CIC-IoT-2023", "unsw": "UNSW-NB15", "cicids": "CICIDS2017"}
		print(f"\n{'='*70}")
		print("CROSS-DATASET SUMMARY (raw features, RF and XGBoost)")
		print(f"{'='*70}")
		print(f"  {'Dataset':<15s} {'Model':<12s} {'F1':>8s} {'FPR':>8s} {'Acc':>8s}")
		print(f"  {'-'*15} {'-'*12} {'-'*8} {'-'*8} {'-'*8}")
		for ds in datasets:
			for key, label in [("rf_raw", "RF"), ("xgb_raw", "XGBoost")]:
				r = all_results[ds][key]
				print(f"  {DISPLAY[ds]:<15s} {label:<12s} {r['f1']*100:>7.2f}% {r['fpr']*100:>7.2f}% {r['acc']*100:>7.2f}%")


if __name__ == "__main__":
	main()
