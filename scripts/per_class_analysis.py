"""Per-attack-class recall analysis for an IDS flow's best genomes.

For each of the flow's best genomes (best_f1, best_fpr, best_acc, best_ce,
best_fitness), reproduce its training on the flow's dataset, predict on
test+val, and compute:
  - Per-attack-class recall (detection rate)
  - Overall FPR (false alarm rate on benign)
  - Class counts

Output is markdown — drop-in-able into the camera-ready paper.

⚠️ GPU CONCURRENCY: this script uses the Rust accelerator (Metal GPU). Running
it while the worker is mid-flow will fight for the same GPU context and may
crash one of them. ONLY run this when the worker is idle (no flow running).

Usage:
    # Analyze r98 (flow_id=1156) on its original dataset
    python scripts/per_class_analysis.py --flow-id 1156

    # Analyze r125 (flow_id=1687) when it finishes — uses canonical-neto
    python scripts/per_class_analysis.py --flow-id 1687

    # Dry-run (no GPU): verify data loading and analysis logic without predicting
    python scripts/per_class_analysis.py --flow-id 1156 --dry-run
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from collections import Counter

import numpy as np

DB_PATH = Path(__file__).resolve().parents[1] / "db" / "wnn.db"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "wnn"))


GENOME_TYPES = ["best_f1", "best_fpr", "best_acc", "best_ce", "best_fitness"]
THRESHOLD_MODE_CANONICAL = "train_cal"  # The headline threshold for the paper


def get_flow_config(flow_id: int) -> dict:
	con = sqlite3.connect(str(DB_PATH))
	row = con.execute("SELECT name, config_json FROM flows WHERE id = ?", (flow_id,)).fetchone()
	con.close()
	if not row:
		raise ValueError(f"Flow {flow_id} not found in DB")
	name, cfg_json = row
	cfg = json.loads(cfg_json)
	return {"name": name, **cfg["params"]}


def get_best_genomes(flow_id: int, threshold_mode: str = THRESHOLD_MODE_CANONICAL) -> dict:
	"""Fetch best genome metadata + reconstruction info from DB + checkpoint.

	The DB's tiers_json column stores a Python repr (not real JSON), so we can't
	reconstruct ClusterGenome from DB alone. Instead, we look up the flow's
	checkpoint file (which has full bits_per_neuron/connections) and match
	checkpoint genomes to DB best_genomes records by neuron count.

	Returns dict keyed by metric ('f1_macro', 'fpr', 'accuracy', 'ce', 'fitness'),
	each value has: reported_*, total_neurons, total_clusters, ckpt_genome (full
	dict with bits_per_neuron + neurons_per_cluster + connections + threshold)
	or None if not reconstructable.
	"""
	import gzip
	con = sqlite3.connect(str(DB_PATH))

	# Find the flow's checkpoint dir (best_genomes db rows reference flow_id)
	flow_row = con.execute("SELECT name FROM flows WHERE id = ?", (flow_id,)).fetchone()
	flow_name = flow_row[0] if flow_row else None
	ckpt_root = Path(__file__).resolve().parents[1] / "checkpoints"
	ckpt_dir = ckpt_root / (flow_name.lower() if flow_name else "")

	# Try the GA Neurons checkpoint first (exp_01); fall back to grid (exp_00)
	ckpt_paths = [ckpt_dir / "exp_01" / "ga_neurons.json.gz",
				  ckpt_dir / "exp_00" / "grid_search.json.gz",
				  ckpt_dir / "exp_00" / "ga_neurons.json.gz"]
	ckpt_path = next((p for p in ckpt_paths if p.exists()), None)
	ckpt_genomes_by_size = {}  # n_neurons → list of genome dicts
	if ckpt_path:
		with gzip.open(ckpt_path, "rt") as f:
			ckpt = json.load(f)
		# Single named genomes
		for key in ["best_ce_genome", "best_acc_genome", "best_f1_genome",
					"best_fpr_genome", "best_fitness_genome"]:
			g = ckpt.get(key)
			if g and "bits_per_neuron" in g:
				n = sum(g.get("neurons_per_cluster", []))
				ckpt_genomes_by_size.setdefault(n, []).append(g)
		# final_population
		for g in ckpt.get("phase_result", {}).get("final_population", []):
			if g and "bits_per_neuron" in g:
				n = sum(g.get("neurons_per_cluster", []))
				ckpt_genomes_by_size.setdefault(n, []).append(g)
		print(f"  Loaded {sum(len(v) for v in ckpt_genomes_by_size.values())} candidate genomes from {ckpt_path.name}")
	else:
		print(f"  WARNING: no checkpoint found at {ckpt_dir}")

	results = {}
	for metric in ["f1_macro", "fpr", "accuracy", "ce", "fitness"]:
		row = con.execute(
			"""
			SELECT bg.genome_id, bg.f1_macro, bg.fpr, bg.accuracy, bg.ce,
			       g.total_neurons, g.total_clusters
			FROM best_genomes bg
			JOIN genomes g ON g.id = bg.genome_id
			WHERE bg.flow_id = ?
			  AND bg.metric = ?
			  AND bg.threshold_mode = ?
			ORDER BY bg.rank ASC
			LIMIT 1
			""",
			(flow_id, metric, threshold_mode),
		).fetchone()
		if not row:
			print(f"  WARNING: no genome found for metric={metric}, threshold={threshold_mode}")
			continue
		(gid, f1, fpr, acc, ce, n_neurons, n_clusters) = row
		# Try to find a matching checkpoint genome by total_neurons
		ckpt_match = None
		candidates = ckpt_genomes_by_size.get(n_neurons, [])
		if len(candidates) >= 1:
			# Among candidates, pick the one whose cached_metrics most closely match
			# the DB's reported values
			def closeness(g):
				cm = g.get("cached_metrics", {}) or {}
				return abs(cm.get("ce", 1e9) - ce) + abs(cm.get("accuracy", 1e9) - acc)
			ckpt_match = min(candidates, key=closeness)
		results[metric] = {
			"genome_id": gid,
			"reported_f1": f1, "reported_fpr": fpr, "reported_acc": acc, "reported_ce": ce,
			"total_neurons": n_neurons,
			"total_clusters": n_clusters,
			"ckpt_genome": ckpt_match,  # None if not reconstructable
		}
	con.close()
	return results


def load_dataset_for_flow(flow_cfg: dict):
	"""Load the dataset specified by the flow's config."""
	from wnn.ids.ciciot2023 import load_ciciot2023
	from wnn.ids.cicids2017 import load_cicids2017
	from wnn.ids.dataset import load_unsw_nb15

	dataset_name = flow_cfg.get("ids_dataset", "unsw-nb15")
	n_bits = flow_cfg.get("ids_n_bits", 8)
	split = flow_cfg.get("ids_split", "random_3way")
	feature_selection = flow_cfg.get("ids_feature_selection", "all")
	rest_bits = flow_cfg.get("ids_rest_bits")
	auto_max_bits = flow_cfg.get("ids_auto_max_bits", 32)
	ids_raw = flow_cfg.get("ids_raw", False)
	ids_invalid_encoding = flow_cfg.get("ids_invalid_encoding")  # None = smart default

	print(f"Loading {dataset_name} (split={split}, n_bits={n_bits}, feature_selection={feature_selection})...")

	if dataset_name == "ciciot2023":
		return load_ciciot2023(n_bits=n_bits, split=split, feature_selection=feature_selection,
							   raw=ids_raw, invalid_encoding=ids_invalid_encoding)
	elif dataset_name == "ciciot2023_full":
		return load_ciciot2023(n_bits=n_bits, split=split, feature_selection=feature_selection,
							   dataset_size="full", raw=ids_raw,
							   invalid_encoding=ids_invalid_encoding)
	elif dataset_name == "ciciot2023_canonical":
		return load_ciciot2023(n_bits=n_bits, split=split, feature_selection=feature_selection,
							   dataset_size="canonical",
							   invalid_encoding=ids_invalid_encoding)
	elif dataset_name == "cicids2017":
		return load_cicids2017(n_bits=n_bits, split=split, feature_selection=feature_selection,
							   raw=ids_raw, invalid_encoding=ids_invalid_encoding)
	else:
		return load_unsw_nb15(n_bits=n_bits, split=split, feature_selection=feature_selection,
							  rest_bits=rest_bits, auto_max_bits=auto_max_bits,
							  invalid_encoding=ids_invalid_encoding)


def compute_per_class_metrics(y_true_binary, y_pred_binary, attack_class_strs) -> dict:
	"""Compute per-attack-class recall + overall FPR.

	Args:
		y_true_binary: 0/1 ground truth (1 = attack)
		y_pred_binary: 0/1 predicted
		attack_class_strs: per-row attack_class string (e.g. "DDoS", "Benign")
	Returns:
		dict with per-class recall, overall FPR, etc.
	"""
	y_true = np.asarray(y_true_binary)
	y_pred = np.asarray(y_pred_binary)
	classes = np.asarray(attack_class_strs)

	results = {}
	# Overall FPR (computed on benign)
	benign_mask = (y_true == 0)
	fp = ((y_pred == 1) & benign_mask).sum()
	tn = ((y_pred == 0) & benign_mask).sum()
	overall_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
	results["overall"] = {
		"benign_count": int(benign_mask.sum()),
		"attack_count": int((y_true == 1).sum()),
		"overall_fpr": float(overall_fpr),
		"fp": int(fp), "tn": int(tn),
	}

	# Per-class recall: for each class, what fraction of rows in that class were predicted as attack (1)?
	# For "Benign": this is the FALSE positive rate (we want this LOW)
	# For attack classes: this is the detection rate (we want this HIGH)
	per_class = {}
	for cls in sorted(set(classes)):
		mask = (classes == cls)
		n_total = mask.sum()
		if n_total == 0: continue
		n_predicted_attack = ((y_pred == 1) & mask).sum()
		recall_or_fpr = n_predicted_attack / n_total
		per_class[cls] = {
			"count": int(n_total),
			"predicted_attack": int(n_predicted_attack),
			"rate": float(recall_or_fpr),  # recall for attack classes; FPR for Benign
		}
	results["per_class"] = per_class
	return results


def render_markdown(flow_id: int, flow_name: str, dataset_name: str,
					results_by_metric: dict) -> str:
	"""Render the per-class analysis as paper-ready markdown."""
	lines = []
	lines.append(f"# Per-class analysis: flow {flow_id} — {flow_name}")
	lines.append(f"Dataset: `{dataset_name}`. Threshold mode: `{THRESHOLD_MODE_CANONICAL}`.")
	lines.append("")

	# Get the union of all classes across results
	all_classes = set()
	for m, r in results_by_metric.items():
		all_classes.update(r["per_class"].keys())
	# Order: Benign first, then attack classes alphabetically
	ordered_classes = (["Benign"] if "Benign" in all_classes else []) + sorted(c for c in all_classes if c != "Benign")

	# Per-class table
	header = "| Class | Count | " + " | ".join([f"{m} (recall%)" for m in results_by_metric]) + " |"
	sep = "|---|---|" + "|".join(["---" for _ in results_by_metric]) + "|"
	lines.append(header)
	lines.append(sep)
	for cls in ordered_classes:
		row = [cls]
		# count from any result that has it
		any_r = next(iter(results_by_metric.values()))
		count = any_r["per_class"].get(cls, {}).get("count", 0)
		row.append(f"{count:,}")
		for m, r in results_by_metric.items():
			rate = r["per_class"].get(cls, {}).get("rate")
			# For Benign, rate IS FPR (we want low); for attacks, rate IS recall (we want high)
			rate_str = f"{rate*100:.2f}" if rate is not None else "—"
			row.append(rate_str)
		lines.append("| " + " | ".join(row) + " |")
	lines.append("")
	lines.append(f"_Note: Benign row's % is FPR (false alarms); attack rows' % is recall (detection)._")

	return "\n".join(lines)


def main():
	parser = argparse.ArgumentParser(description="Per-attack-class analysis for an IDS flow")
	parser.add_argument("--flow-id", type=int, required=True, help="Flow ID to analyze")
	parser.add_argument("--threshold-mode", default=THRESHOLD_MODE_CANONICAL,
						help=f"Threshold mode (default: {THRESHOLD_MODE_CANONICAL})")
	parser.add_argument("--dry-run", action="store_true",
						help="Skip GPU prediction; only verify data loading + parsing")
	parser.add_argument("--metrics", nargs="+", default=["f1_macro", "fpr", "accuracy"],
						help=f"Which metric-best genomes to analyze (subset of {GENOME_TYPES})")
	parser.add_argument("--out", help="Write markdown to file (also prints to stdout)")
	args = parser.parse_args()

	flow_cfg = get_flow_config(args.flow_id)
	dataset_name = flow_cfg.get("ids_dataset", "?")
	flow_name = flow_cfg.get("name", "?")
	print(f"=== Per-class analysis for flow {args.flow_id}: {flow_name} ===")
	print(f"Dataset: {dataset_name}")

	genomes = get_best_genomes(args.flow_id, threshold_mode=args.threshold_mode)
	available = list(genomes.keys())
	print(f"Best genomes available ({args.threshold_mode}): {available}")
	to_analyze = [m for m in args.metrics if m in available]
	if not to_analyze:
		print(f"ERROR: none of requested metrics {args.metrics} found in DB")
		sys.exit(1)
	for m in to_analyze:
		g = genomes[m]
		print(f"  {m:<12s}: genome_id={g['genome_id']:<7d} neurons={g['total_neurons']:<4d} "
			  f"clusters={g['total_clusters']:<2d}  reported F1={g['reported_f1']:.4f} "
			  f"FPR={g['reported_fpr']:.4f} Acc={g['reported_acc']:.4f}")

	# Reconstructability summary
	print(f"\nReconstructability check:")
	for m in to_analyze:
		ok = genomes[m].get("ckpt_genome") is not None
		mark = "✓" if ok else "⊘"
		print(f"  {mark} {m:<12s} (genome_id={genomes[m]['genome_id']}, neurons={genomes[m]['total_neurons']}): "
			  f"{'reconstructable from checkpoint' if ok else 'NOT in checkpoint — will be skipped'}")

	if args.dry_run:
		print("\n[DRY RUN] Would now load dataset and predict — skipping (use without --dry-run for real run)")
		return

	# === Real run ===
	print(f"\nLoading dataset for flow {args.flow_id}...")
	ds = load_dataset_for_flow(flow_cfg)
	print(f"  X_train: {ds.X_train.shape}, X_test: {ds.X_test.shape}, "
		  f"X_val: {ds.X_val.shape if ds.X_val is not None else 'None'}")

	# Build evaluator
	from wnn.ram.architecture.ids_evaluator import IDSEvaluator
	from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome

	# Merge test+val (matches the worker's eval set protocol)
	if ds.X_val is not None:
		X_eval = np.concatenate([ds.X_test, ds.X_val])
		y_eval_binary = np.concatenate([ds.y_test_binary, ds.y_val_binary])
		# Get attack_class for test+val by re-loading the dataframes (lossy via .values)
		# We rebuild attack_class from y_eval_multi (an int per row) + ds.category_names
		y_eval_multi = np.concatenate([ds.y_test_multi, ds.y_val_multi])
	else:
		X_eval = ds.X_test
		y_eval_binary = ds.y_test_binary
		y_eval_multi = ds.y_test_multi
	attack_class_strs = np.array([ds.category_names[i] for i in y_eval_multi])

	# Build evaluator (single-cluster, binary)
	evaluator = IDSEvaluator(
		X_train=ds.X_train, y_train=ds.y_train_binary,
		X_test=X_eval, y_test=y_eval_binary,
		num_classes=2, class_names=["Benign", "Attack"],
	)

	results_by_metric = {}
	for m in to_analyze:
		g = genomes[m]
		ckpt_g = g.get("ckpt_genome")
		if not ckpt_g:
			print(f"\n⊘ {m} genome (id={g['genome_id']}, neurons={g['total_neurons']}): "
				  f"NOT reconstructable (not in available checkpoint). Skipping.")
			continue
		print(f"\n→ Predicting with {m} genome (id={g['genome_id']}, neurons={g['total_neurons']})...")
		genome = ClusterGenome(
			bits_per_neuron=ckpt_g["bits_per_neuron"],
			neurons_per_cluster=ckpt_g["neurons_per_cluster"],
			connections=ckpt_g["connections"],
			threshold=ckpt_g.get("threshold", 0.5),
		)
		import time
		t0 = time.time()
		y_pred = evaluator.predict(genome)
		dt = time.time() - t0
		print(f"  predict() took {dt:.1f}s")

		metrics = compute_per_class_metrics(y_eval_binary, y_pred, attack_class_strs)
		print(f"  Overall FPR: {metrics['overall']['overall_fpr']*100:.4f}%")
		for cls, info in metrics["per_class"].items():
			print(f"    {cls:<15s}: rate={info['rate']*100:.4f}% ({info['predicted_attack']}/{info['count']:,})")

		results_by_metric[m] = metrics

	md = render_markdown(args.flow_id, flow_name, dataset_name, results_by_metric)
	print("\n" + "=" * 70)
	print(md)
	if args.out:
		Path(args.out).write_text(md)
		print(f"\nWrote markdown to {args.out}")


if __name__ == "__main__":
	main()
