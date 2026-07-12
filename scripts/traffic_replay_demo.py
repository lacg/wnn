"""Traffic-replay demo (RAID'26 Review A): stream the temporally-later UNSW
partition through a trained WNN in arrival order, like an IDS on a wire.

The UNSW temporal split IS a replay: train on the earlier traffic, then the
test partition arrives in captured order. This script trains one genome once
(Rust accelerator, same path as the experiment worker), scores the test
stream, and replays it in time buckets — rolling F1/FPR/alert-rate over a
sliding window, then end-of-run totals + pipeline throughput.

What this demonstrates (reviewer-facing):
  1. end-to-end train->deploy->stream latency budget (WNN trains in seconds),
  2. detection quality is stable across the replayed timeline (no drift cliff),
  3. amortized per-flow inference cost at bulk-scoring throughput.

Usage:
  python3 scripts/traffic_replay_demo.py                     # 200n x 16b default
  python3 scripts/traffic_replay_demo.py --neurons 100 --bits 24 --window 20000
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from wnn.ids.dataset import load_unsw_nb15
from wnn.ram.architecture.ids_evaluator import IDSEvaluator
from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome, generate_connections


def build_genome(neurons: int, bits: int, total_input_bits: int, seed: int) -> ClusterGenome:
	"""Binary IDS genome, single-cluster discriminator mode (the SP-cohort
	convention, ids_single_cluster=True): one cluster whose raw score is the
	attack response, thresholded downstream."""
	bits_per_neuron = [bits] * neurons
	connections = generate_connections(bits_per_neuron, total_input_bits, seed)
	return ClusterGenome(bits_per_neuron=bits_per_neuron,
	                     neurons_per_cluster=[neurons],
	                     connections=connections)


def f1_optimal_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
	"""Train-calibrated threshold: sweep unique score quantiles for max F1."""
	candidates = np.quantile(scores, np.linspace(0.01, 0.99, 197))
	best_tau, best_f1 = 0.5, -1.0
	for tau in np.unique(candidates):
		pred = scores >= tau
		tp = int((pred & (labels == 1)).sum())
		fp = int((pred & (labels == 0)).sum())
		fn = int((~pred & (labels == 1)).sum())
		f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
		if f1 > best_f1:
			best_f1, best_tau = f1, float(tau)
	return best_tau


def window_metrics(pred: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
	"""(f1, fpr, alert_rate) over one window."""
	tp = int((pred & (labels == 1)).sum())
	fp = int((pred & (labels == 0)).sum())
	fn = int((~pred & (labels == 1)).sum())
	tn = int((~pred & (labels == 0)).sum())
	f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
	fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
	return f1, fpr, float(pred.mean())


def replay(pred: np.ndarray, labels: np.ndarray, window: int) -> None:
	"""Walk the stream in window-sized buckets, printing the rolling picture."""
	n = len(pred)
	print(f"\n  {'flows':>12} {'window F1':>10} {'window FPR':>11} {'alerts':>8} {'cum F1':>8} {'cum FPR':>9}")
	for end in range(window, n + window, window):
		end = min(end, n)
		start = end - window if end - window >= 0 else 0
		wf1, wfpr, walert = window_metrics(pred[start:end], labels[start:end])
		cf1, cfpr, _ = window_metrics(pred[:end], labels[:end])
		print(f"  {end:>12,} {100 * wf1:>9.2f}% {100 * wfpr:>10.2f}% {100 * walert:>7.1f}% "
		      f"{100 * cf1:>7.2f}% {100 * cfpr:>8.2f}%")
		if end == n:
			break


def main():
	ap = argparse.ArgumentParser(description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--neurons", type=int, default=200, help="Neurons per cluster (2 clusters).")
	ap.add_argument("--bits", type=int, default=16, help="Address bits per neuron.")
	ap.add_argument("--n-bits", type=int, default=16, help="Thermometer bits per feature.")
	ap.add_argument("--window", type=int, default=10_000, help="Rolling window size (flows).")
	ap.add_argument("--seed", type=int, default=42)
	args = ap.parse_args()

	print("=" * 78)
	print(f"  UNSW-NB15 temporal replay — WNN {args.neurons}n x {args.bits}b "
	      f"(top-20 features, {args.n_bits}-bit thermo)")
	print("=" * 78)

	t_start = time.time()
	t0 = time.time()
	ds = load_unsw_nb15(split="temporal", feature_selection="top20", n_bits=args.n_bits)
	t_load = time.time() - t0
	n_train, n_test = len(ds.y_train_binary), len(ds.y_test_binary)
	total_bits = ds.X_train.total_bits
	print(f"  Load: {t_load:.1f}s — train {n_train:,} flows (earlier traffic), "
	      f"replay {n_test:,} flows (later traffic), {total_bits} input bits")

	t0 = time.time()
	evaluator = IDSEvaluator(ds, classification="binary", num_parts=5, single_cluster=True)
	t_cache = time.time() - t0
	print(f"  Cache build (Rust upload): {t_cache:.1f}s")

	genome = build_genome(args.neurons, args.bits, total_bits, args.seed)

	# One training pass each — the accelerator trains-then-scores per call.
	t0 = time.time()
	train_scores = np.asarray(evaluator.score_train_examples(genome))
	t_train_score = time.time() - t0
	t0 = time.time()
	test_scores = np.asarray(evaluator.score_examples(genome))
	t_test_score = time.time() - t0

	tau = f1_optimal_threshold(train_scores, ds.y_train_binary)
	pred = test_scores >= tau
	labels = ds.y_test_binary.astype(np.int64)

	print(f"\n  Train+score(train): {t_train_score:.1f}s   Train+score(replay): {t_test_score:.1f}s")
	print(f"  Amortized replay scoring: {1e6 * t_test_score / n_test:.1f} µs/flow "
	      f"({n_test / t_test_score:,.0f} flows/s incl. the training pass)")
	print(f"  Train-calibrated threshold tau = {tau:.4f} (no test-set peeking)")

	replay(pred, labels, args.window)

	f1, fpr, alert = window_metrics(pred, labels)
	acc = float((pred == (labels == 1)).mean())
	print(f"\n  End of replay — F1 {100 * f1:.2f}% | FPR {100 * fpr:.2f}% | "
	      f"Acc {100 * acc:.2f}% | alert rate {100 * alert:.1f}%")
	print(f"  Total pipeline (load->train->replay): {time.time() - t_start:.0f}s")


if __name__ == "__main__":
	main()
