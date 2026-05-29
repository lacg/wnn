"""Replay paper Table 1's Best F1‡ UNSW-temp genome under TODAY's code.

Takes the EXACT genome that produced F1=90.52 / FPR=4.13 / Acc=90.54 in the
pre-fix `UNSW-fitfix-t8b-temporal-r106` cohort (flow 1567, genome hash
`7758323e6a03a3f8` — 100 neurons × 32 bits each, 3200 connections) and
re-trains + re-evaluates it under the post-fix OI/QUAD code.

Hypothesis (29/05/2026): pre-fix tournament_select used CE-only ranking,
so the GA evolved an architecture biased toward CE-pressure outcomes. With
OI/QUAD-corrected training applied to the SAME architecture, do we still
get close to 90.52/4.13, or did the fixes shift the trained cells enough
to substantially change the per-mode metrics?

This is a TRAIN+EVAL replay (cells aren't stored; the IDS pipeline re-derives
them from connections + training data + seed). So we're testing whether the
ARCHITECTURE generalizes across the training fix, not the cells.

Usage:
  python scripts/replay_paper_unsw_genome.py            # default seed 106
  python scripts/replay_paper_unsw_genome.py --seed 42  # different training seed
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time


DB = "/Users/lacg/wnn/db/wnn.db"
GENOME_HASH = "7758323e6a03a3f8"
PAPER_F1 = 90.52
PAPER_FPR = 4.13
PAPER_ACC = 90.54


def load_genome_connections() -> list[int]:
	"""Pull the exact connections that produced 90.52/4.13."""
	con = sqlite3.connect(DB)
	cur = con.cursor()
	cur.execute(
		"SELECT connections_json FROM genomes WHERE genome_hash = ? LIMIT 1",
		(GENOME_HASH,),
	)
	row = cur.fetchone()
	con.close()
	if not row or not row[0]:
		raise SystemExit(f"Genome hash {GENOME_HASH} has no connections_json")
	# Stored as comma-separated ints.
	return [int(x) for x in row[0].split(",")]


def main():
	ap = argparse.ArgumentParser(description=__doc__)
	ap.add_argument("--seed", type=int, default=106,
	                help="Training seed (default 106 = the original flow 1567 seed).")
	ap.add_argument("--no-oi", action="store_true",
	                help="Run with OI training DISABLED — for a pre-fix-like baseline.")
	ap.add_argument("--k-folds", type=int, default=1,
	                help="K-fold count. Default 1 (the 'test' evaluator path used "
	                     "for paper VALIDATION metrics: train on full 80%, eval on "
	                     "the 20%% held-out test set — matches paper Table 1). Use "
	                     "5 to use the 'optimizer' K-fold evaluator instead (the "
	                     "intra-train fold split used DURING GA search, not at "
	                     "final reporting). The empirical-mode mismatch in the "
	                     "earlier replay was caused by using k_folds=5 here.")
	args = ap.parse_args()

	# OI flag — set BEFORE importing ram_accelerator so the AtomicU32 gets the
	# right initial value at module load.
	if args.no_oi:
		os.environ.pop("WNN_ORDER_INDEPENDENT_TRAIN", None)
	else:
		os.environ["WNN_ORDER_INDEPENDENT_TRAIN"] = "1"

	from wnn.ids import load_ids_dataset
	from wnn.ram.architecture.ids_evaluator import IDSEvaluator
	from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
	import ram_accelerator

	print(f"=" * 70)
	print(f"Paper r106 genome REPLAY under today's code")
	print(f"  Genome hash:  {GENOME_HASH}")
	print(f"  Architecture: 100 neurons × 32 bits each (3,200 connections)")
	print(f"  OI training:  {'ON' if not args.no_oi else 'OFF (pre-fix-like)'}")
	print(f"  K-fold:       {args.k_folds}")
	print(f"  Seed:         {args.seed}")
	print(f"  Paper target: F1={PAPER_F1}  FPR={PAPER_FPR}  Acc={PAPER_ACC}")
	print(f"=" * 70)

	print("\nLoading UNSW-NB15 temporal split at 8-bit thermometer...")
	t0 = time.time()
	dataset = load_ids_dataset(
		"unsw-nb15", n_bits=8, split="temporal",
		feature_selection="top20",
		encoded_storage="memory",
	)
	print(f"  Loaded in {time.time()-t0:.1f}s. "
	      f"Train: {dataset.X_train.shape[0]:,}  Test: {dataset.X_test.shape[0]:,}  "
	      f"Features: {dataset.X_train.shape[1]}")

	print("\nBuilding IDSEvaluator (single-cluster, K-fold, balanced classes)...")
	# num_parts vs k_folds: the IDSEvaluator requires them equal. num_parts=1
	# is the worker's "test evaluator" path (train full → 20% held-out test),
	# which is what produces the paper's reported metrics. num_parts=5 (with
	# k_folds=5) is the worker's "optimizer evaluator" used DURING GA search.
	# The empirical-mode mismatch in the original replay was caused by the
	# latter (K-fold split → ~35K eval rows vs paper's 82K held-out test).
	evaluator = IDSEvaluator(
		dataset=dataset,
		classification="binary",
		num_parts=args.k_folds,
		k_folds=args.k_folds,
		kfold_per_gen=args.k_folds,
		single_cluster=True,
		balance_classes=True,
		neuron_sample_rate=0.25,
		seed=args.seed,
	)

	print("Pulling genome connections from DB...")
	connections = load_genome_connections()
	expected = 100 * 32
	if len(connections) != expected:
		raise SystemExit(f"Expected {expected} connections, got {len(connections)}")

	genome = ClusterGenome(
		bits_per_neuron=[32] * 100,
		neurons_per_cluster=[100],
		connections=connections,
	)
	print(f"  Genome: {sum(genome.neurons_per_cluster)} neurons × "
	      f"{genome.bits_per_neuron[0]} bits ({len(connections)} connections)")

	print("\nTraining + evaluating (single training pass, seven-mode breakdown)...")
	t0 = time.time()
	eval_scores, train_scores, anchor_metrics = evaluator.evaluate_at_thresholds(
		genome, [-1.0, 0.5],
	)
	train_labels = evaluator._y_train
	eval_labels = evaluator._y_test
	normal_class = getattr(evaluator, "_normal_class", 0)

	def _metrics_at(t: float):
		ce, acc, f1, fpr = ram_accelerator.compute_binary_metrics_at_threshold_py(
			eval_scores, eval_labels, float(t), normal_class,
		)
		return f1 * 100.0, fpr * 100.0, acc * 100.0, ce

	# train_cal: optimal F1 threshold on train scores, applied to eval.
	train_thresh, _, _ = ram_accelerator.find_optimal_threshold_f1_py(
		train_scores, train_labels,
	)
	tc_f1, tc_fpr, tc_acc, _ = _metrics_at(train_thresh)

	# fixed_05.
	f5_f1, f5_fpr, f5_acc, _ = _metrics_at(0.5)

	# Platt / Beta calibration on train. Each helper returns the CALIBRATED
	# threshold directly (the inverse of the σ(a + b·t) = 0.5 / beta-cdf
	# inverse is baked into the Rust side). Signatures:
	#   fit_platt_scaling_py  → (threshold, a, b)
	#   fit_beta_calibration_py → (threshold, a, b, c)
	#   fit_empirical_threshold_py → (threshold, n_bins)
	# The previous version of this script mis-unpacked Platt as (a, b, _) and
	# computed t=-a/b, which produced F1=35 / FPR=100 (saturation).
	platt_thresh, _platt_a, _platt_b = ram_accelerator.fit_platt_scaling_py(
		train_scores, train_labels,
	)
	p_f1, p_fpr, p_acc, _ = _metrics_at(platt_thresh)

	beta_thresh, _beta_a, _beta_b, _beta_c = ram_accelerator.fit_beta_calibration_py(
		train_scores, train_labels,
	)
	b_f1, b_fpr, b_acc, _ = _metrics_at(beta_thresh)

	# Empirical: train-FPR-aligned threshold.
	emp_thresh, _n_bins = ram_accelerator.fit_empirical_threshold_py(
		train_scores, train_labels,
	)
	e_f1, e_fpr, e_acc, _ = _metrics_at(emp_thresh)

	# Empirical cumulative: was using a cascade-optimization helper (wrong API);
	# the paper's actual call needs to be traced from experiment.py:1221. Task 5
	# follow-up — for now report NaN and don't pretend a value.
	ec_f1 = ec_fpr = ec_acc = float("nan")

	# val_cal: F1-optimal threshold on EVAL scores (oracle — looks at test set).
	vc_thresh, _, _ = ram_accelerator.find_optimal_threshold_f1_py(
		eval_scores, eval_labels,
	)
	vc_f1, vc_fpr, vc_acc, _ = _metrics_at(vc_thresh)

	dt = time.time() - t0
	print(f"  Train + eval + 7-mode metrics in {dt:.1f}s")

	# Report.
	print(f"\n{'='*70}")
	print(f"  REPLAY RESULTS — r106 genome under today's code (OI={'ON' if not args.no_oi else 'OFF'})")
	print(f"{'='*70}")
	print(f"  Paper Best F1‡: F1={PAPER_F1:5.2f}  FPR={PAPER_FPR:5.2f}  Acc={PAPER_ACC:5.2f}")
	print(f"  {'─'*68}")
	header = f"  {'mode':<24}  {'F1':>7}  {'FPR':>7}  {'Acc':>7}    ΔF1     ΔFPR"
	print(header)
	print(f"  {'─'*68}")
	for mode, f1, fpr, acc in [
		("train_cal",            tc_f1,  tc_fpr,  tc_acc),
		("fixed_05",             f5_f1,  f5_fpr,  f5_acc),
		("platt",                p_f1,   p_fpr,   p_acc),
		("beta",                 b_f1,   b_fpr,   b_acc),
		("empirical",            e_f1,   e_fpr,   e_acc),
		("empirical_cumulative", ec_f1,  ec_fpr,  ec_acc),
		("val_cal",              vc_f1,  vc_fpr,  vc_acc),
	]:
		df1 = f1 - PAPER_F1
		dfpr = fpr - PAPER_FPR
		mark = " ★" if f1 >= 90.0 and fpr <= 5.0 else "  "
		print(f"  {mode:<24}  {f1:7.2f}  {fpr:7.2f}  {acc:7.2f}  {df1:+6.2f}  {dfpr:+6.2f}{mark}")
	print(f"{'='*70}")


if __name__ == "__main__":
	main()
