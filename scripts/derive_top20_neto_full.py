"""Re-derive TOP20 RF-importance feature ranking on canonical neto-full.

This is the C1 path: the original TOP20 list (hardcoded in
src/wnn/ids/ciciot2023.py:TOP20_RF_FEATURES) was derived on bencorn's mirror,
which included a fabricated `Time_To_Live` column not present in the canonical
Neto distribution. To produce defensible camera-ready numbers, we re-derive
TOP20 from a Random Forest trained on canonical neto-full (Kaggle
akashdogra/ciciot23csv-derived, 46 features, no TTL).

Methodology:
- Train RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
  on the full neto-full train set (37,349,263 rows × 46 features).
- Read feature_importances_, sort descending, take top 20.
- Compare against the current (bencorn-derived) TOP20.

Output: ranked feature list + side-by-side comparison.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "wnn"))

CURRENT_TOP20 = [
	"HTTPS", "Number", "Time_To_Live", "Max", "ack_flag_number",
	"Rate", "IAT", "ack_count", "Header_Length", "Min",
	"Variance", "psh_flag_number", "Tot sum", "Std", "Tot size",
	"syn_count", "AVG", "rst_flag_number", "DNS", "rst_count",
]


def main():
	from datasets import load_dataset

	print("=" * 78)
	print("Re-derive TOP20 RF-importance on canonical neto-full")
	print("=" * 78)

	repo = "lacg030175/CIC-IoT-2023-neto-full"
	print(f"\nLoading {repo} (random_3way)...", flush=True)
	t0 = time.time()
	ds = load_dataset(repo, "random_3way")
	df_train = ds["train"].to_pandas()
	print(f"  Loaded {len(df_train):,} train rows in {time.time() - t0:.1f}s")

	# Binary label — already int (0=Benign, 1=Attack) in lacg030175 HF dataset
	y_train = df_train["label"].astype(np.int32).values

	# All numeric feature columns (drop label cols)
	all_features = [
		c for c in df_train.columns
		if c not in ("label", "Label", "Label_orig", "attack_class", "is_benign")
		and pd.api.types.is_numeric_dtype(df_train[c])
	]
	print(f"  Available canonical features: {len(all_features)}")

	X_train = df_train[all_features].values.astype(np.float32)
	print(f"\nMedian-imputing NaN/Inf...")
	X_train = SimpleImputer(strategy="median").fit_transform(X_train)

	print(f"\nTraining RandomForestClassifier(n_estimators=100, random_state=42) on 37.3M rows × {len(all_features)} features...")
	t0 = time.time()
	rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
	rf.fit(X_train, y_train)
	print(f"  Trained in {time.time() - t0:.1f}s")

	# Rank
	importances = rf.feature_importances_
	order = np.argsort(-importances)
	ranked = [(all_features[i], importances[i]) for i in order]

	# Output
	new_top20 = [f for f, _ in ranked[:20]]
	print("\n" + "=" * 78)
	print("CANONICAL TOP20 (neto-full RF feature_importances_, descending)")
	print("=" * 78)
	print(f"{'Rank':<5} {'Feature':<22} {'Importance':>12}   {'In bencorn TOP20?':<18}")
	for i, (feat, imp) in enumerate(ranked[:25], 1):
		in_current = "✓" if feat in CURRENT_TOP20 else "—"
		marker = "★" if i <= 20 else " "
		print(f"{marker} {i:<3} {feat:<22} {imp:>10.4f}     {in_current}")

	# Diffs
	current_set = set(CURRENT_TOP20)
	new_set = set(new_top20)
	added = [f for f in new_top20 if f not in current_set and f != "Time_To_Live"]
	removed = [f for f in CURRENT_TOP20 if f not in new_set and f != "Time_To_Live"]

	print("\n" + "=" * 78)
	print("DIFF: bencorn TOP20 → canonical neto-full TOP20")
	print("=" * 78)
	print(f"Bencorn TOP20 had:           {CURRENT_TOP20}")
	print(f"\nCanonical neto-full TOP20:   {new_top20}")
	print(f"\nDropped from list (besides TTL): {removed if removed else 'none'}")
	print(f"New entries (replacing TTL + above): {added if added else 'none'}")

	# 19-of-20 overlap analysis
	overlap = current_set & new_set
	print(f"\nOverlap: {len(overlap)} of 20 features in common.")
	print(f"  In both: {sorted(overlap)}")

	# Save TOP20 to a JSON for later programmatic use
	import json
	out_path = Path("/Users/lacg/wnn/data") / "top20_canonical_neto_full.json"
	out_path.parent.mkdir(parents=True, exist_ok=True)
	with open(out_path, "w") as f:
		json.dump({
			"top20": new_top20,
			"all_ranked": [(f, float(imp)) for f, imp in ranked],
			"source_dataset": repo,
			"methodology": "RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42) on train set, feature_importances_, descending",
			"train_rows": int(len(df_train)),
			"num_features": len(all_features),
		}, f, indent=2)
	print(f"\nSaved canonical ranking to: {out_path}")


if __name__ == "__main__":
	main()
