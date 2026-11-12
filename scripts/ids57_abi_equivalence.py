#!/usr/bin/env python
"""IDS-57 A1 wheel-equivalence harness (docs/ids57_a1_prereg.md §8, W1-W4).

Runs on WHATEVER ram_accelerator wheel is installed and writes a JSON of digests,
so the same command before and after a worker-wheel swap gives two files that
must be bit-identical. Everything trains and scores in Rust through the worker's
own construction path (IdsEvaluatorSettings -> IdsEvaluatorFactory -> IDSEvaluator
-> IDSCacheWrapper via wnn.accel). Nothing here re-implements the accelerator.

Modes
  digest (default)   seeded [250, 250] genomes at 34 / 48 / 64 bits (+ two mixed-width)
                     on the reference flow's configuration. Per genome:
                       cv    5-fold CV metrics exactly as the GA scores a pool
                             (IDSEvaluator.evaluate_batch, ids_k_folds=5, ids_kfold_per_gen=5)
                       val   raw per-row VAL scores (sha256 of the float64 bytes) and the
                             val-calibrated tau + VAL confusion per decode mode
                       cells sha256 of the exported trained cells (export_trained)
                     The TEST partition never enters the Rust cache in this mode: the
                     evaluators are built on a dataset whose report set IS the VAL partition.
  --retrain-finals   W3: rebuild each flow's GA best_f1 final genome from the DB, retrain it
                     on the worker's (optimizer, test) evaluator pair and compare the TEST
                     decode modes against test_summaries.threshold_metadata. Those rows are
                     already reported, so this TEST read is allowed.
  --compare A B      bit-for-bit comparison of two output files (exit 1 on any difference).

DB is opened read-only. Run with `nice -n 10`, one process.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import os
import resource
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DB_URI = "file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro"
REFERENCE_FLOW = 7509
CONTROL_FLOWS = (7509, 7513, 7517)
NEURONS = 250
GENOME_SEED_BASE = 57_000
PRODUCTION_ENV = {"WNN_EVAL_IN_PLACE": "1"}
RECORDED_ENV = ("WNN_EVAL_IN_PLACE", "WNN_HYBRID_SPEED_RATIO", "WNN_ORDER_INDEPENDENT_TRAIN", "WNN_NO_METAL",
	"WNN_COALESCE_GROUPS", "WNN_ALLOW_PY_FALLBACK", "RAYON_NUM_THREADS")
FLOAT_FIELDS = ("ce", "acc", "macro_f1", "weighted_f1", "benign_fpr", "f1", "fpr", "attack_f1", "tau")


# ------------------------------------------------------------------ small helpers

def _sha(data: bytes) -> str:
	return hashlib.sha256(data).hexdigest()


def _hex(value: float) -> str:
	"""Exact, NaN-safe float text (JSON decimal round-trips too, hex is unambiguous)."""
	return float(value).hex()


def _file_sha(path: Path) -> str:
	h = hashlib.sha256()
	with open(path, "rb") as f:
		for block in iter(lambda: f.read(1 << 20), b""):
			h.update(block)
	return h.hexdigest()


def _git(*args: str) -> str:
	return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()


def _log(message: str) -> None:
	print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _peak_rss_gb() -> float:
	return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 30)


# ------------------------------------------------------------------ environment + provenance

def apply_production_env(params: dict) -> None:
	"""The env the live flow_runner trains under: eval-in-place, the flow's OI
	switch and the worker's predicted CPU/GPU split."""
	from wnn.ram.experiments.IdsEvaluatorSettings import IdsEvaluatorSettings
	from wnn.ram.experiments.hsr_function import predict_hsr_from_params
	for key, value in PRODUCTION_ENV.items():
		os.environ[key] = value
	IdsEvaluatorSettings.apply_oi_env(params)
	os.environ["WNN_HYBRID_SPEED_RATIO"] = str(predict_hsr_from_params(params))


def wheel_record() -> dict:
	"""W4: the installed wheel (ABI, binary hash) and the Python worker tree."""
	from wnn.accel import require_accel
	accel = require_accel()
	binary = next(Path(accel.__file__).parent.glob("ram_accelerator*.so"))
	return {
		"abi": int(accel.ABI_VERSION),
		"wheel_binary": str(binary),
		"wheel_binary_sha256": _file_sha(binary),
		"wheel_binary_mtime_utc": datetime.fromtimestamp(binary.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
		"git_head": _git("rev-parse", "HEAD"),
		"git_tree_src_wnn": _git("rev-parse", "HEAD:src/wnn"),
		"git_dirty_src_wnn": _git("status", "--porcelain", "--", "src/wnn"),
		"harness_sha256": _file_sha(Path(__file__).resolve()),
	}


def run_meta(started: float) -> dict:
	return {
		**wheel_record(),
		"env": {key: os.environ.get(key) for key in RECORDED_ENV},
		"captured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
		"runtime_s": round(time.time() - started, 1),
		"peak_rss_gb": round(_peak_rss_gb(), 2),
	}


# ------------------------------------------------------------------ construction (the worker's path)

def flow_params(db: sqlite3.Connection, flow_id: int) -> dict:
	row = db.execute("select config_json from flows where id=?", (flow_id,)).fetchone()
	if row is None:
		raise SystemExit(f"flow {flow_id} not found")
	return json.loads(row[0])["params"]


def settings_for(params: dict):
	from wnn.ram.experiments.IdsEvaluatorSettings import IdsEvaluatorSettings
	return IdsEvaluatorSettings.from_params(params)


def check_reference_settings(s) -> None:
	"""The arm's path per the pre-registration; refuse to digest anything else."""
	expected = {"dataset_name": "unsw-nb15", "split": "random_3way", "classification": "binary",
		"memory_mode_name": "BINARY", "write_rule": "lattice", "single_cluster": False, "neuron_sample_rate": 0.25,
		"balance_classes": True, "order_independent_train": True, "k_folds": 5, "kfold_per_gen": 5,
		"encoding_dedupe": False, "feature_ranking": "legacy", "receptive_fields": "off"}
	wrong = {key: getattr(s, key) for key, want in expected.items() if getattr(s, key) != want}
	if wrong:
		raise SystemExit(f"reference flow settings differ from the pre-registered path: {wrong}")


def val_as_report(dataset):
	"""The dataset with the VAL partition in the report slot (TEST is left out)."""
	return dataclasses.replace(dataset, X_test=dataset.X_val, y_test_binary=dataset.y_val_binary,
		y_test_multi=dataset.y_val_multi)


def build_pair(factory, dataset, s) -> tuple:
	"""(optimizer, report) exactly as IdsEvaluatorFactory.flat pairs them: the
	K-fold optimizer and a full-train evaluator sharing its Rust cache."""
	optimizer = factory._optimizer(dataset, s)
	return optimizer, factory._test(dataset, s, optimizer._cache)


# ------------------------------------------------------------------ digest cases

def case_specs() -> list[tuple[str, int, tuple[int, ...], tuple[int, int]]]:
	"""(name, numpy seed, widths drawn per neuron, neurons per cluster)."""
	specs = []
	for bits in (34, 48, 64):
		for k in (0, 1):
			specs.append((f"b{bits}_s{k}", GENOME_SEED_BASE + bits * 10 + k, (bits,), (NEURONS, NEURONS)))
	specs.append(("mixed_34_48_64", GENOME_SEED_BASE + 1, (34, 48, 64), (NEURONS, NEURONS)))
	specs.append(("mixed_4_to_64", GENOME_SEED_BASE + 2, tuple(range(4, 65)), (NEURONS, 137)))
	return specs


def make_genome(seed: int, widths: tuple[int, ...], neurons_per_cluster: tuple[int, int], total_bits: int):
	import numpy as np
	from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
	rng = np.random.default_rng(seed)
	bits = [int(b) for b in rng.choice(widths, size=sum(neurons_per_cluster))]
	connections = [int(c) for c in rng.integers(0, total_bits, sum(bits))]
	return ClusterGenome(bits_per_neuron=bits, neurons_per_cluster=list(neurons_per_cluster), connections=connections)


def genome_record(genome) -> dict:
	import numpy as np
	bits = np.asarray(genome.bits_per_neuron, dtype=np.int64)
	return {
		"neurons_per_cluster": [int(n) for n in genome.neurons_per_cluster],
		"bits_min": int(bits.min()), "bits_max": int(bits.max()), "bits_sum": int(bits.sum()),
		"neurons_above_34_bits": int((bits > 34).sum()),
		"bits_sha256": _sha(bits.tobytes()),
		"connections_sha256": _sha(np.asarray(genome.connections, dtype=np.int64).tobytes()),
	}


def metrics_record(metrics) -> dict:
	return {key: _hex(getattr(metrics, key)) for key in ("ce", "acc", "f1", "fpr", "threshold")}


def cv_record(optimizer, genome) -> dict:
	"""The GA's pool fitness inputs: K-fold CV mean over all 5 folds."""
	return metrics_record(optimizer.evaluate_batch([genome])[0])


def mode_record(entry: dict) -> dict:
	record = {key: _hex(entry[key]) for key in FLOAT_FIELDS if key in entry}
	record["confusion"] = entry["confusion"]
	return record


def val_modes_record(val_evaluator, genome) -> dict:
	"""Decode modes scored on VAL (tau of margin_val_cal is fit on VAL)."""
	result = val_evaluator.evaluate_multiclass_at_thresholds(genome)
	return {mode: mode_record(entry) for mode, entry in sorted(result["modes"].items())}


def scores_sha(scores) -> str:
	import numpy as np
	return _sha(np.asarray(scores, dtype=np.float64).tobytes())


def key_reach(export) -> dict:
	"""How far the trained addresses reach: proof the case writes past 34 bits
	and (at 64 bits) sets the top bit of the u64 key."""
	import numpy as np
	keys = [unit.keys for unit in export.units if len(unit.keys)]
	top = max(int(k.max()) for k in keys)
	return {"max_key_bit_length": top.bit_length(),
		"keys_at_or_above_2_pow_34": int(sum(int((k >= np.uint64(1 << 34)).sum()) for k in keys)),
		"keys_with_bit_63_set": int(sum(int((k >= np.uint64(1 << 63)).sum()) for k in keys))}


def cells_sha(export) -> tuple[str, int]:
	"""sha256 over every physical unit (cluster, bits, connections, keys, cells)
	plus the grouping and the miss cell; second value = stored cells."""
	import numpy as np
	h = hashlib.sha256()
	stored = 0
	for unit in export.units:
		h.update(np.asarray([unit.cluster, unit.bits, len(unit.keys)], dtype=np.int64).tobytes())
		h.update(np.asarray(unit.connections, dtype=np.int64).tobytes())
		h.update(np.ascontiguousarray(unit.keys, dtype=np.uint64).tobytes())
		h.update(np.ascontiguousarray(unit.cells, dtype=np.uint8).tobytes())
		stored += len(unit.keys)
	h.update(np.asarray(export.counts, dtype=np.int64).tobytes())
	h.update(np.asarray([export.miss_cell], dtype=np.int64).tobytes())
	return h.hexdigest(), stored


def export_record(val_evaluator, genome) -> dict:
	"""One more training pass (the FPGA-export path): raw VAL scores + cells."""
	export = val_evaluator.export_trained_layer(genome)
	cells, stored = cells_sha(export)
	footprint = dataclasses.asdict(export.footprint)
	record = {
		"val_scores_sha256": scores_sha(export.eval_scores), "val_scores_n": len(export.eval_scores),
		"val_scores_sha256_val_slot": None if export.val_scores is None else scores_sha(export.val_scores),
		"cells_sha256": cells, "physical_units": len(export.units), "stored_cells": stored, "footprint": footprint,
		"key_reach": key_reach(export),
	}
	del export
	gc.collect()
	return record


def score_examples_record(val_evaluator, genome) -> dict:
	"""Raw VAL scores through the scoring-only entry point (a third pass)."""
	scores = val_evaluator.score_examples(genome)
	return {"sha256": scores_sha(scores), "n": len(scores)}


def digest_case(optimizer, val_evaluator, spec: tuple, total_bits: int) -> dict:
	name, seed, widths, neurons_per_cluster = spec
	genome = make_genome(seed, widths, neurons_per_cluster, total_bits)
	started = time.time()
	record = {"genome": genome_record(genome), "cv": cv_record(optimizer, genome),
		"val_modes": val_modes_record(val_evaluator, genome), "export": export_record(val_evaluator, genome),
		"score_examples": score_examples_record(val_evaluator, genome)}
	_log(f"  {name}: {time.time() - started:.0f}s, cells {record['export']['cells_sha256'][:12]}, "
		f"val scores {record['export']['val_scores_sha256'][:12]}, peak RSS {_peak_rss_gb():.1f} GB")
	return record


def batch_cv_record(optimizer, specs: list[tuple], total_bits: int) -> dict:
	"""The same genomes scored as ONE pool (how a GA generation calls it)."""
	genomes = [make_genome(seed, widths, npc, total_bits) for _, seed, widths, npc in specs]
	metrics = optimizer.evaluate_batch(genomes)
	return {spec[0]: metrics_record(m) for spec, m in zip(specs, metrics)}


def dataset_record(dataset, optimizer) -> dict:
	import numpy as np
	return {
		"n_train": int(len(dataset.y_train_binary)), "n_val": int(len(dataset.y_val_binary)),
		"total_input_bits": int(dataset.X_train.total_bits),
		"y_train_sha256": _sha(np.asarray(dataset.y_train_binary, dtype=np.int64).tobytes()),
		"y_val_sha256": _sha(np.asarray(dataset.y_val_binary, dtype=np.int64).tobytes()),
		"desirability_ce_anchor": _hex(optimizer.desirability_ce_anchor()),
	}


def run_digest(db: sqlite3.Connection, only: list[str] | None) -> dict:
	from wnn.ram.experiments.IdsEvaluatorFactory import IdsEvaluatorFactory
	params = flow_params(db, REFERENCE_FLOW)
	apply_production_env(params)
	s = settings_for(params)
	check_reference_settings(s)
	factory = IdsEvaluatorFactory(log=_log)
	dataset = val_as_report(factory.load(s))
	optimizer, val_evaluator = build_pair(factory, dataset, s)
	total_bits = int(dataset.X_train.total_bits)
	specs = [spec for spec in case_specs() if only is None or spec[0] in only]
	cases = {spec[0]: digest_case(optimizer, val_evaluator, spec, total_bits) for spec in specs}
	_log("  pooled CV pass (all genomes in one evaluate_batch)")
	pooled = batch_cv_record(optimizer, specs, total_bits)
	return {"mode": "digest", "reference_flow": REFERENCE_FLOW, "dataset": dataset_record(dataset, optimizer),
		"cases": cases, "cv_pooled": pooled}


# ------------------------------------------------------------------ W3: retrain the controls' finals

def stored_final(db: sqlite3.Connection, flow_id: int) -> tuple[str, dict, int]:
	"""(genome_hash, stored decode modes, worker_abi) of the GA best_f1 final row."""
	row = db.execute("""select ts.genome_hash, ts.threshold_metadata, ts.worker_abi from test_summaries ts
		join experiments e on e.id = ts.experiment_id
		where ts.flow_id=? and e.phase_type='ga_neurons' and ts.test_point='final' and ts.genome_type='best_f1'""",
		(flow_id,)).fetchone()
	if row is None:
		raise SystemExit(f"flow {flow_id}: no GA best_f1 final row in test_summaries")
	return row[0], json.loads(row[1]), int(row[2])


def stored_genome(db: sqlite3.Connection, genome_hash: str):
	from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
	tiers_json, connections_json = db.execute("""select tiers_json, connections_json from genomes
		where genome_hash=? and connections_json is not null limit 1""", (genome_hash,)).fetchone()
	tiers = json.loads(tiers_json)
	return ClusterGenome(bits_per_neuron=tiers["bits_per_neuron"], neurons_per_cluster=tiers["neurons_per_cluster"],
		connections=[int(c) for c in connections_json.split(",")])


def mode_differences(stored: dict, fresh: dict) -> list[str]:
	"""Field-level differences of one decode mode (floats compared exactly)."""
	diffs = [] if stored["confusion"] == fresh["confusion"] else [f"confusion {stored['confusion']} -> {fresh['confusion']}"]
	for key in FLOAT_FIELDS:
		if key in stored and _hex(stored[key]) != _hex(fresh.get(key, float("nan"))):
			diffs.append(f"{key} {stored[key]!r} -> {fresh.get(key)!r}")
	return diffs


def compare_modes(stored: dict, fresh: dict) -> dict:
	missing = sorted(set(stored) - set(fresh))
	diffs = {mode: mode_differences(stored[mode], fresh[mode]) for mode in sorted(stored) if mode in fresh}
	diffs = {mode: d for mode, d in diffs.items() if d}
	confusion_ok = not missing and all(stored[mode]["confusion"] == fresh[mode]["confusion"] for mode in stored)
	return {"modes_compared": sorted(set(stored) & set(fresh)), "modes_missing": missing,
		"all_confusions_match": confusion_ok, "exact_match": confusion_ok and not diffs, "differences": diffs}


def retrain_final(db: sqlite3.Connection, factory, dataset, flow_id: int) -> dict:
	params = flow_params(db, flow_id)
	apply_production_env(params)
	s = settings_for(params)
	genome_hash, stored, stored_abi = stored_final(db, flow_id)
	genome = stored_genome(db, genome_hash)
	_optimizer, test_evaluator = factory.flat(dataset, s)
	fresh = test_evaluator.evaluate_multiclass_at_thresholds(genome)["modes"]
	verdict = compare_modes(stored, fresh)
	record = {"seed": params["seed"], "genome_hash": genome_hash, "stored_worker_abi": stored_abi,
		"genome": genome_record(genome), "stored_margin_val_cal": mode_record(stored["margin_val_cal"]),
		"retrained_margin_val_cal": mode_record(fresh["margin_val_cal"]), **verdict}
	_log(f"  flow {flow_id}: margin_val_cal {fresh['margin_val_cal']['confusion']} vs stored "
		f"{stored['margin_val_cal']['confusion']} -> exact_match={verdict['exact_match']}")
	del _optimizer, test_evaluator
	gc.collect()
	return record


def run_retrain(db: sqlite3.Connection, flow_ids: list[int]) -> dict:
	"""The three controls share one dataset load (same dataset params; only the seed differs)."""
	from wnn.ram.experiments.IdsEvaluatorFactory import IdsEvaluatorFactory
	factory = IdsEvaluatorFactory(log=_log)
	dataset = factory.load(settings_for(flow_params(db, flow_ids[0])))
	flows = {str(flow_id): retrain_final(db, factory, dataset, flow_id) for flow_id in flow_ids}
	return {"mode": "retrain_finals", "flows": flows,
		"all_exact": all(record["exact_match"] for record in flows.values())}


# ------------------------------------------------------------------ compare two outputs

def payload_of(document: dict) -> dict:
	"""Everything that must be bit-identical across wheels (meta is provenance)."""
	return {key: value for key, value in document.items() if key != "meta"}


def flatten(node, prefix: str = "") -> dict:
	if isinstance(node, dict):
		out = {}
		for key, value in node.items():
			out.update(flatten(value, f"{prefix}/{key}"))
		return out
	return {prefix: node}


def run_compare(path_a: str, path_b: str) -> int:
	a, b = (json.loads(Path(p).read_text()) for p in (path_a, path_b))
	flat_a, flat_b = flatten(payload_of(a)), flatten(payload_of(b))
	differing = sorted(key for key in set(flat_a) | set(flat_b) if flat_a.get(key) != flat_b.get(key))
	print(f"A: {path_a}  ABI {a['meta']['abi']}  git {a['meta']['git_head'][:10]}  payload {a['payload_sha256']}")
	print(f"B: {path_b}  ABI {b['meta']['abi']}  git {b['meta']['git_head'][:10]}  payload {b['payload_sha256']}")
	print(f"fields compared: {len(set(flat_a) | set(flat_b))}, differing: {len(differing)}")
	for key in differing:
		print(f"  DIFF {key}: {flat_a.get(key)!r} != {flat_b.get(key)!r}")
	print("VERDICT: BIT-IDENTICAL" if not differing else "VERDICT: DIFFERENT")
	return 0 if not differing else 1


# ------------------------------------------------------------------ entry point

def write_output(payload: dict, out: str, started: float) -> None:
	document = {"meta": run_meta(started), **payload}
	document["payload_sha256"] = _sha(json.dumps(payload, sort_keys=True).encode())
	path = Path(out)
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
	_log(f"wrote {path} (ABI {document['meta']['abi']}, payload {document['payload_sha256']}, "
		f"{document['meta']['runtime_s']}s, peak RSS {document['meta']['peak_rss_gb']} GB)")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--out", help="output JSON (digest / retrain modes)")
	parser.add_argument("--cases", nargs="+", help="digest mode: only these case names (timing probes)")
	parser.add_argument("--retrain-finals", nargs="+", type=int, metavar="FLOW", help=f"W3 mode, e.g. {CONTROL_FLOWS}")
	parser.add_argument("--compare", nargs=2, metavar=("A", "B"), help="compare two output files bit for bit")
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	if args.compare:
		return run_compare(args.compare[0], args.compare[1])
	if not args.out:
		raise SystemExit("--out is required")
	os.environ.setdefault("RAYON_NUM_THREADS", "4")
	started = time.time()
	db = sqlite3.connect(DB_URI, uri=True)
	payload = run_retrain(db, args.retrain_finals) if args.retrain_finals else run_digest(db, args.cases)
	write_output(payload, args.out, started)
	return 0 if payload.get("all_exact", True) else 1


if __name__ == "__main__":
	sys.exit(main())
