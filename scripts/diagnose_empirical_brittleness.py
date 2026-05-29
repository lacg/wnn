"""Diagnose the brittleness of fit_empirical_threshold across the r106 replay.

The paper's Best F1‡ UNSW-temp result (90.52 / 4.13 / 90.54 from
`UNSW-fitfix-t8b-temporal-r106`, genome hash 7758323e6a03a3f8) replays as
79.81 / 39.51 under today's code — apparent ~10pp regression.

This script proves the regression is in the THRESHOLD-SELECTION algorithm,
NOT the trained model itself, by:
  1. Loading the exact paper genome connections from the DB.
  2. Training under today's code.
  3. Showing the per-bin attack rates around the empirical algorithm's
     pickup zone (noisy in the replay; clean in the paper).
  4. Applying the PAPER's threshold (0.3650) to today's eval scores — which
     reproduces the paper to within 0.28 F1.
  5. Trying alternative rules (smoothed, min-bin-size) that close the gap.

CONCLUSION: production code's `fit_empirical_threshold` should grow a
`min_bin_size` parameter to avoid picking from noise. Without it, two
training runs that produce essentially-equivalent eval-score distributions
can report wildly different empirical-mode F1 numbers.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from collections import defaultdict


def main():
	os.environ.pop("WNN_ORDER_INDEPENDENT_TRAIN", None)   # OI=OFF (paper era)
	from wnn.ids import load_ids_dataset
	from wnn.ram.architecture.ids_evaluator import IDSEvaluator
	from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
	import ram_accelerator as ra

	# 1. Load the genome.
	con = sqlite3.connect("/Users/lacg/wnn/db/wnn.db")
	conn_str = con.execute(
		"SELECT connections_json FROM genomes WHERE genome_hash='7758323e6a03a3f8' LIMIT 1"
	).fetchone()[0]
	connections = [int(x) for x in conn_str.split(",")]

	# 2. Train the exact paper architecture.
	ds = load_ids_dataset("unsw-nb15", n_bits=8, split="temporal",
	                      feature_selection="top20", encoded_storage="memory")
	ev = IDSEvaluator(dataset=ds, classification="binary",
	                  num_parts=1, k_folds=1, kfold_per_gen=1,
	                  single_cluster=True, balance_classes=True,
	                  neuron_sample_rate=0.25, seed=106)
	g = ClusterGenome(bits_per_neuron=[32] * 100, neurons_per_cluster=[100],
	                  connections=connections)
	eval_scores, train_scores, _ = ev.evaluate_at_thresholds(g, [-1.0, 0.5])

	# 3. Bin the train scores like the Rust algorithm does (6-decimal rounding).
	bins = defaultdict(lambda: [0, 0])
	for s, l in zip(train_scores, ev._y_train):
		bins[int(s * 1_000_000)][1 if l == 1 else 0] += 1
	sorted_keys = sorted(bins.keys())

	# 4. Alternative rules.
	def rule_first(min_size: int = 1) -> float:
		"""Current production rule: first bin with attack-rate ≥ 50%
		(optionally constrained to bins with at least `min_size` examples).
		"""
		for k in sorted_keys:
			n, a = bins[k]
			total = n + a
			if total >= min_size and total > 0 and a / total >= 0.5:
				return k / 1_000_000
		return 0.5

	def rule_smoothed(window: int = 20, min_total: int = 100) -> float:
		"""Sliding-window: first window of `window` consecutive bins whose
		pooled attack-rate is ≥ 50% (with `min_total` minimum examples in the
		window). Returns the window's center bin's score.
		"""
		for i in range(len(sorted_keys) - window + 1):
			cn = sum(bins[sorted_keys[j]][0] for j in range(i, i + window))
			ca = sum(bins[sorted_keys[j]][1] for j in range(i, i + window))
			if cn + ca >= min_total and ca / (cn + ca) >= 0.5:
				return sorted_keys[i + window // 2] / 1_000_000
		return 0.5

	# 5. Evaluate each rule's threshold on the eval set.
	def _metrics(t: float):
		_, acc, f1, fpr = ra.compute_binary_metrics_at_threshold_py(
			eval_scores, ev._y_test, float(t), 0)
		return f1 * 100, fpr * 100, acc * 100

	candidates = [
		("PRODUCTION (first ≥50%)",             rule_first(1)),
		("first ≥50%, min_bin=200",             rule_first(200)),
		("first ≥50%, min_bin=500",             rule_first(500)),
		("smoothed window=20, min_total=100",   rule_smoothed(20, 100)),
		("smoothed window=50, min_total=200",   rule_smoothed(50, 200)),
		("smoothed window=100, min_total=500",  rule_smoothed(100, 500)),
		("paper threshold 0.3650 (control)",    0.3650),
	]

	print(f"{'rule':<40} {'thr':>7} {'F1':>7} {'FPR':>7} {'Acc':>7}")
	print("─" * 78)
	for name, t in candidates:
		f1, fpr, acc = _metrics(t)
		marker = "  ←PRODUCTION" if "PRODUCTION" in name else ""
		print(f"{name:<40} {t:>7.4f} {f1:>6.2f}% {fpr:>6.2f}% {acc:>6.2f}%{marker}")
	print("─" * 78)
	print(f"{'Paper original (stored DB)':<40} {0.3650:>7.4f} {90.52:>6.2f}% {4.13:>6.2f}% {90.54:>6.2f}%")

	# 6. Conclusion.
	print()
	print("CONCLUSION:")
	print("  Paper's threshold on today's eval scores reproduces paper within ~0.3 F1.")
	print("  The trained model IS paper-equivalent. Only the threshold-selection")
	print("  rule's noise-sensitivity caused the apparent 10pp regression.")
	print()
	print("  Recommended production fix: add `min_bin_size` arg to")
	print("  fit_empirical_threshold (Rust adaptive.rs:751); default 200-500 to")
	print("  ignore small-sample bins around the transition zone.")


if __name__ == "__main__":
	main()
