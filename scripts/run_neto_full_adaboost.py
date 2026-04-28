"""AdaBoost baseline on lacg030175/CIC-IoT-2023-neto-full (46.7M, 46 feat).

Apples-to-apples with Neto et al.'s published AdaBoost baseline (95.63% F1).

Pass `--all-features` to use all 46 features (likely matches Neto's setup more
closely). Default uses top-20 (matches our WNN input subset for tight comparison
with WNN numbers).

CPU-only — safe to run alongside the worker.
"""

import argparse
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "wnn"))

TOP20_CICIOT = [
	"HTTPS", "Number", "Time_To_Live", "Max", "ack_flag_number",
	"Rate", "IAT", "ack_count", "Header_Length", "Min",
	"Variance", "psh_flag_number", "Tot sum", "Std", "Tot size",
	"syn_count", "AVG", "rst_flag_number", "DNS", "rst_count",
]


def metrics(y_true, y_pred):
	tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
	return {
		"f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
		"fpr": fp / (fp + tn) if (fp + tn) > 0 else 0,
		"acc": accuracy_score(y_true, y_pred),
		"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
	}


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--all-features", action="store_true",
						help="Use all 46 features (default: top-20 matching WNN input)")
	args = parser.parse_args()

	from datasets import load_dataset
	repo = "lacg030175/CIC-IoT-2023-neto-full"
	print(f"Loading {repo} (random_3way)...", flush=True)
	ds = load_dataset(repo, "random_3way")
	df_train = ds["train"].to_pandas()
	df_eval = pd.concat([ds["test"].to_pandas(), ds["validation"].to_pandas()], ignore_index=True)
	print(f"  Train: {len(df_train):,}  Eval: {len(df_eval):,}")

	non_features = {"Label", "Label_orig", "label", "attack_class"}
	if args.all_features:
		features = [c for c in df_train.columns if c not in non_features]
		print(f"  Features: ALL {len(features)} features")
	else:
		features = [f for f in TOP20_CICIOT if f in df_train.columns]
		print(f"  Features: {len(features)}/20 (top-20 mode)")

	X_train = df_train[features].values.astype(np.float32)
	X_eval = df_eval[features].values.astype(np.float32)
	y_train = df_train["label"].values.astype(np.int32)
	y_eval = df_eval["label"].values.astype(np.int32)
	attack_classes = df_eval["attack_class"].values

	# AdaBoost can't handle NaN — impute
	X_train = np.where(np.isinf(X_train), np.nan, X_train)
	X_eval = np.where(np.isinf(X_eval), np.nan, X_eval)
	imp = SimpleImputer(strategy="median")
	X_train = imp.fit_transform(X_train)
	X_eval = imp.transform(X_eval)

	print("\nTraining AdaBoost (50 estimators, depth-1 stumps — classical config)...", flush=True)
	clf = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42)
	t0 = time.time()
	clf.fit(X_train, y_train)
	train_s = time.time() - t0
	t0 = time.time()
	y_pred = clf.predict(X_eval)
	infer_s = time.time() - t0

	m = metrics(y_eval, y_pred)
	print(f"\n=== AdaBoost on neto-full (46.7M, 46 feat, top-20) ===")
	print(f"  F1:  {m['f1']*100:.2f}%   FPR: {m['fpr']*100:.2f}%   Acc: {m['acc']*100:.2f}%")
	print(f"  Train: {train_s:.1f}s   Infer: {infer_s:.1f}s")
	print(f"  Confusion: TN={m['tn']:,} FP={m['fp']:,} FN={m['fn']:,} TP={m['tp']:,}")

	print(f"\nPer-class breakdown:")
	ac = np.asarray(attack_classes)
	for cls in sorted(set(ac.tolist())):
		mask = (ac == cls)
		n = int(mask.sum())
		if n == 0: continue
		n_pred = int(((y_pred == 1) & mask).sum())
		typ = "FPR" if cls == "Benign" else "recall"
		print(f"  {cls:<15s}: {typ} = {n_pred/n*100:>6.2f}%  ({n_pred:,}/{n:,})")

	print(f"\nFor paper comparison:")
	print(f"  Neto AdaBoost (published): F1=??.??%  Acc=99.55% (paper says ~)")
	print(f"  Our AdaBoost (46M, top-20): F1={m['f1']*100:.2f}%  Acc={m['acc']*100:.2f}%")


if __name__ == "__main__":
	main()
