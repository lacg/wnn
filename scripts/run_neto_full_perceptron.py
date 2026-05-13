"""sklearn Perceptron baseline on lacg030175/CIC-IoT-2023-neto-full (46.7M, 46 feat).

Apples-to-apples with Neto et al.'s published Perceptron baseline (81.05% F1, 98.18% Acc).
Top-20 features, random_3way (test+val merged). CPU-only.
"""

import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "wnn"))

# Canonical TOP20 RF importance on neto-full (re-derived 13/05/2026).
from wnn.ids.ciciot2023 import TOP20_RF_FEATURES as TOP20_CICIOT


def main():
	from datasets import load_dataset
	repo = "lacg030175/CIC-IoT-2023-neto-full"
	print(f"Loading {repo}...", flush=True)
	ds = load_dataset(repo, "random_3way")
	df_train = ds["train"].to_pandas()
	df_eval = pd.concat([ds["test"].to_pandas(), ds["validation"].to_pandas()], ignore_index=True)
	print(f"  Train: {len(df_train):,}, Eval: {len(df_eval):,}")
	features = [f for f in TOP20_CICIOT if f in df_train.columns]
	print(f"  Features: {len(features)}/20")

	X_train = df_train[features].values.astype(np.float32)
	X_eval = df_eval[features].values.astype(np.float32)
	y_train = df_train["label"].values.astype(np.int32)
	y_eval = df_eval["label"].values.astype(np.int32)
	attack_classes = df_eval["attack_class"].values

	X_train = np.where(np.isinf(X_train), np.nan, X_train)
	X_eval = np.where(np.isinf(X_eval), np.nan, X_eval)

	# Perceptron requires no NaN AND benefits from feature scaling (raw scales vary 100×+).
	# Pipeline: median impute → StandardScaler → Perceptron with sklearn defaults.
	imp = SimpleImputer(strategy="median")
	X_train = imp.fit_transform(X_train)
	X_eval = imp.transform(X_eval)
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_eval = scaler.transform(X_eval)

	print("\nTraining Perceptron (sklearn defaults)...", flush=True)
	clf = Perceptron(random_state=42, n_jobs=-1)
	t0 = time.time()
	clf.fit(X_train, y_train)
	train_s = time.time() - t0
	t0 = time.time()
	y_pred = clf.predict(X_eval)
	infer_s = time.time() - t0

	tn, fp, fn, tp = confusion_matrix(y_eval, y_pred).ravel()
	f1 = f1_score(y_eval, y_pred, average="macro", zero_division=0)
	acc = accuracy_score(y_eval, y_pred)
	fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

	print(f"\n=== Perceptron on neto-full (46.7M, top-20) ===")
	print(f"  F1: {f1*100:.2f}%   FPR: {fpr*100:.2f}%   Acc: {acc*100:.2f}%")
	print(f"  Train: {train_s:.1f}s   Infer: {infer_s:.2f}s")
	print(f"  Confusion: TN={int(tn):,} FP={int(fp):,} FN={int(fn):,} TP={int(tp):,}")
	print(f"\nNeto Perceptron (published): F1=81.05% / Acc=98.18%")
	print(f"Our Perceptron:               F1={f1*100:.2f}% / FPR={fpr*100:.2f}% / Acc={acc*100:.2f}%")

	print("\nPer-class breakdown:")
	ac = np.asarray(attack_classes)
	for cls in sorted(set(ac.tolist())):
		mask = (ac == cls)
		n = int(mask.sum())
		if n == 0: continue
		n_pred = int(((y_pred == 1) & mask).sum())
		typ = "FPR" if cls == "Benign" else "recall"
		print(f"  {cls:<15s}: {typ}={n_pred/n*100:>6.2f}%  ({n_pred:,}/{n:,})")


if __name__ == "__main__":
	main()
