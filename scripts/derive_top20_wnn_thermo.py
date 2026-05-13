"""Re-derive TOP20 features by post-quantization mutual information (WNN-friendly).

The RF-importance TOP20 is biased toward continuous features that tree models
handle natively. WNN with N-bit thermometer encoding loses information when
continuous features have wide dynamic range. This script ranks features by
mutual information between their *quantized* representation and the binary
label — directly measuring "signal retained after thermometer encoding".

Methodology:
- Load canonical neto-full train set (37.3M rows × 46 features).
- For each feature, discretize via KBinsDiscretizer with strategy='quantile',
  n_bins = thermometer_bits + 1 (equivalent to thermometer encoding's
  information content: N thresholds → N+1 distinct values).
- Compute mutual_info_classif between the quantized feature and the binary
  label.
- Rank features by MI at the target bit width.

We compute MI at several bit widths so we can compare 8b (Micro/Small) vs 96b
(250n×100b cohort) feature rankings.

Outputs:
- data/top20_wnn_thermo_{nbits}b.json — per-bit-width canonical TOP20.
- Side-by-side comparison vs RF-importance TOP20.

CPU-only — safe to run alongside the worker.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "wnn"))

from wnn.ids.ciciot2023 import TOP20_RF_FEATURES as CANONICAL_TOP20

# Bit widths to evaluate (matches the WNN cohort thermometer choices)
BIT_WIDTHS = [4, 8, 16, 32, 96]

# Sub-sample for MI estimation — 5M rows is plenty for stable rankings and
# completes in ~10 min vs ~3h on full 37M. MI estimates converge with √n,
# so 5M gives an MI standard error of <0.5% of the value — well below
# inter-feature ranking gaps.
MI_SUBSAMPLE = 5_000_000

# Smaller still for the per-bit-width sweep (we re-run for each bit width)
MI_SUBSAMPLE_PER_WIDTH = 2_000_000


def main():
	from datasets import load_dataset

	print("=" * 78)
	print("Re-derive TOP20 by post-quantization MI (WNN-friendly ranking)")
	print(f"  Source: lacg030175/CIC-IoT-2023-neto-full (canonical 46.7M)")
	print(f"  Bit widths: {BIT_WIDTHS}")
	print(f"  MI subsample: {MI_SUBSAMPLE_PER_WIDTH:,} rows per bit width")
	print("=" * 78)

	repo = "lacg030175/CIC-IoT-2023-neto-full"
	print(f"\nLoading {repo} (random_3way)...", flush=True)
	t0 = time.time()
	ds = load_dataset(repo, "random_3way")
	df_train = ds["train"].to_pandas()
	print(f"  Loaded {len(df_train):,} train rows in {time.time() - t0:.1f}s")

	# Binary label
	y = df_train["label"].astype(np.int32).values

	# Numeric features only (drop label cols)
	all_features = [
		c for c in df_train.columns
		if c not in ("label", "Label", "Label_orig", "attack_class", "is_benign")
		and pd.api.types.is_numeric_dtype(df_train[c])
	]
	print(f"  Numeric features available: {len(all_features)}")

	X = df_train[all_features].values.astype(np.float64)

	# Median-impute NaN/Inf (one-shot, shared across bit widths)
	print("\nMedian-imputing NaN/Inf...")
	X = SimpleImputer(strategy="median").fit_transform(X)

	# Subsample once for reproducibility across bit widths
	rng = np.random.default_rng(42)
	idx = rng.choice(len(X), size=min(MI_SUBSAMPLE_PER_WIDTH, len(X)), replace=False)
	X_sub = X[idx]
	y_sub = y[idx]
	print(f"  Using {len(X_sub):,}-row subsample for MI (seed=42, idx fixed across bit widths)")

	rankings = {}
	for n_bits in BIT_WIDTHS:
		n_bins = n_bits + 1  # thermometer encoding produces n_bits+1 distinct values
		# Cap n_bins to a reasonable value — KBinsDiscretizer caps at unique values anyway,
		# but for very wide bit widths we want to actually use them.
		print(f"\n--- Bit width = {n_bits} (n_bins={n_bins}) ---", flush=True)
		t0 = time.time()
		disc = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None)
		# Suppress warnings about fewer unique values than bins
		import warnings
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			X_q = disc.fit_transform(X_sub)
		print(f"  Quantized in {time.time() - t0:.1f}s")

		t0 = time.time()
		# discrete_features=True since we just quantized
		mi = mutual_info_classif(X_q, y_sub, discrete_features=True, random_state=42)
		print(f"  Computed MI in {time.time() - t0:.1f}s")

		# Rank features by MI
		order = np.argsort(-mi)
		ranked = [(all_features[i], float(mi[i])) for i in order]
		rankings[n_bits] = ranked

	# Output
	print("\n" + "=" * 78)
	print("TOP20 RANKINGS BY POST-QUANTIZATION MUTUAL INFORMATION")
	print("=" * 78)

	# Side-by-side table
	canonical_set = set(CANONICAL_TOP20)
	# Build a lookup: feature → rank in each ranking
	col_headers = ["RF"] + [f"MI-{b}b" for b in BIT_WIDTHS]

	# Build a union of top-25 across all rankings to know which features to display
	displayed = set()
	for b in BIT_WIDTHS:
		for feat, _ in rankings[b][:25]:
			displayed.add(feat)
	displayed.update(CANONICAL_TOP20)

	# For each displayed feature, compute its rank in each ranking
	def rank_in(ranking_list, feat):
		for i, (f, _) in enumerate(ranking_list, 1):
			if f == feat:
				return i
		return None

	# Print: rank 1..20 side by side per bit width
	max_rows = 22
	print(f"\n{'#':<4} | {'RF (canonical)':<22} | " + " | ".join(f"MI-{b}b{'':<{16}}" for b in BIT_WIDTHS))
	print("-" * (4 + 22 + 3 + len(BIT_WIDTHS) * 23))
	for i in range(max_rows):
		row = [f"{i+1:<3}"]
		# RF column
		if i < len(CANONICAL_TOP20):
			f = CANONICAL_TOP20[i]
			row.append(f"{f:<22}")
		else:
			row.append(" " * 22)
		# MI columns
		for b in BIT_WIDTHS:
			if i < len(rankings[b]):
				f, mi = rankings[b][i]
				mark = " " if f in canonical_set else "*"
				cell = f"{mark}{f:<18}{mi:.4f}"[:20]
				row.append(cell)
			else:
				row.append(" " * 20)
		print(" | ".join(row))

	# Diffs at the target bit widths
	for b in [8, 96]:
		ranking_b_top20 = [f for f, _ in rankings[b][:20]]
		new_in_b = [f for f in ranking_b_top20 if f not in canonical_set]
		dropped_from_b = [f for f in CANONICAL_TOP20 if f not in set(ranking_b_top20)]
		overlap = set(ranking_b_top20) & canonical_set
		print(f"\n=== Diff: RF TOP20 → MI-{b}b TOP20 ===")
		print(f"  Overlap: {len(overlap)} of 20 features")
		print(f"  MI-{b}b adds: {new_in_b}")
		print(f"  RF had (MI-{b}b dropped): {dropped_from_b}")

	# Save JSON
	out_dir = Path("/Users/lacg/wnn/data")
	out_dir.mkdir(parents=True, exist_ok=True)
	for b, ranked in rankings.items():
		path = out_dir / f"top20_wnn_thermo_{b}b.json"
		with open(path, "w") as f:
			json.dump({
				"top20": [feat for feat, _ in ranked[:20]],
				"all_ranked": [[feat, mi] for feat, mi in ranked],
				"thermometer_bits": b,
				"methodology": "mutual_info_classif on KBinsDiscretizer(n_bins=bits+1, strategy=quantile)",
				"subsample_rows": MI_SUBSAMPLE_PER_WIDTH,
				"seed": 42,
			}, f, indent=2)
		print(f"\nSaved: {path}")


if __name__ == "__main__":
	main()
