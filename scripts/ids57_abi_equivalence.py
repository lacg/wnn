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
                     Harness v2 (14/11/2026, W3v): the write rule may be "lattice" or "vote".
                     The rule, ids_vote_share and balance_classes reach Rust only through the
                     worker's own path (IdsEvaluatorSettings.from_params -> IdsEvaluatorFactory.flat
                     -> IDSEvaluator -> IDSCacheWrapper.set_write_rule); the harness only CHECKS
                     that the resolved settings carry the flow's stored values and records them
                     in meta. Options (v2, both off by default = the v1 lattice payload):
                       --genome-types  also retrain the other GA final genome types
                       --negative-control  retrain the first flow's best_f1 genome once more
                                       with the lattice rule; it must NOT match (proves the
                                       comparison can see a write-rule change)
  --breeding         W1b: one GA generation's offspring search, through the real
                     ArchitectureGAStrategy (built as experiment.py builds it for the controls'
                     "GA Neurons" phase) -> IDSEvaluator.search_offspring -> Rust. Seeded parent
                     pools at bit caps 34 (the controls' regime) and 64 (the arm's), each with and
                     without duplicate neurons. Digest: the exact call arguments, the parents' CV
                     metrics and fitness, every returned child (bits / neurons / connections
                     hashes + metrics) and the percentile-selected offspring. TEST is never loaded.
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
HARNESS_VERSION = 2  # v1 = lattice-only (sha256 04c668cc...); v2 adds the vote rule to --retrain-finals
RETRAIN_RULES = ("lattice", "vote")
DEFAULT_GENOME_TYPE = "best_f1"
# Settings that decide the loaded dataset: flows retrained together must agree on all of them.
DATASET_FIELDS = ("dataset_name", "split", "classification", "n_bits", "feature_selection", "rest_bits", "ids_raw",
	"invalid_encoding", "auto_max_bits", "encoded_storage", "streaming", "encoding_dedupe", "feature_ranking")


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
		"harness_version": HARNESS_VERSION,
	}


def run_meta(started: float, extra: dict) -> dict:
	return {
		**wheel_record(),
		**extra,
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


# ------------------------------------------------------------------ W1b: GA breeding (search_offspring)

BREEDING_PARENTS = 10
BREEDING_OFFSPRING = 9
BREEDING_GENERATION = 3
BREEDING_TRAIN_IDX = 0
BREEDING_DUP_FRACTION = 0.2
GA_PHASE_GENERATIONS = 250
# (name, bit cap = the flow's max_bits, per-genome widths the parents draw from, duplicate neurons?)
BREEDING_CASES = (
	("cap34", 34, (14, 24, 34, 34), False),
	("cap34_dup", 34, (14, 24, 34, 34), True),
	("cap64", 64, (24, 34, 48, 64, 64), False),
	("cap64_dup", 64, (24, 34, 48, 64, 64), True),
)


def cluster_neurons(rng, count: int, bits: int, total_bits: int, duplicates: bool) -> list[list[int]]:
	"""`count` neurons of one cluster; with `duplicates`, the last fifth are exact
	copies of earlier neurons of the same cluster (the multiplicity layer's input)."""
	distinct = count - int(count * BREEDING_DUP_FRACTION) if duplicates else count
	neurons = [[int(c) for c in rng.integers(0, total_bits, bits)] for _ in range(distinct)]
	copies = [neurons[int(i)] for i in rng.integers(0, distinct, count - distinct)]
	return neurons + copies


def make_parent(rng, widths: tuple[int, ...], duplicates: bool, total_bits: int, lo: int, hi: int):
	"""A grid-like parent: one width for the whole genome, neuron counts in [lo, hi]."""
	from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
	bits = int(rng.choice(widths))
	counts = [int(n) for n in rng.integers(lo, hi + 1, 2)]
	neurons = [n for count in counts for n in cluster_neurons(rng, count, bits, total_bits, duplicates)]
	return ClusterGenome(bits_per_neuron=[bits] * len(neurons), neurons_per_cluster=counts,
		connections=[c for neuron in neurons for c in neuron])


def make_parents(case: tuple, total_bits: int, params: dict) -> list:
	import numpy as np
	name, _cap, widths, duplicates = case
	rng = np.random.default_rng(GENOME_SEED_BASE + 500 + [c[0] for c in BREEDING_CASES].index(name))
	lo, hi = max(int(params["min_neurons"]), 60), int(params["max_neurons"])
	return [make_parent(rng, widths, duplicates, total_bits, lo, hi) for _ in range(BREEDING_PARENTS)]


def duplicate_neurons(genome) -> int:
	"""Neurons that repeat an earlier neuron of the same cluster (description only)."""
	seen, repeats, offset, neuron = set(), 0, 0, 0
	for cluster, count in enumerate(genome.neurons_per_cluster):
		for _ in range(count):
			bits = genome.bits_per_neuron[neuron]
			key = (cluster, tuple(genome.connections[offset:offset + bits]))
			repeats += key in seen
			seen.add(key)
			offset, neuron = offset + bits, neuron + 1
	return repeats


def bred_record(genome) -> dict:
	record = genome_record(genome)
	record["duplicate_neurons"] = duplicate_neurons(genome)
	if getattr(genome, "metrics", None) is not None:
		record["metrics"] = {key: _hex(getattr(genome.metrics, key)) for key in ("ce", "acc", "f1", "fpr")}
	return record


def ga_configs(params: dict, optimizer, parents: list, max_bits: int, total_bits: int) -> tuple:
	"""(ArchitectureConfig, GAConfig) for the controls' "GA Neurons" phase, field by
	field as Experiment builds them from the worker's ExperimentConfig (worker.py
	_run_flow_experiments + experiment.py). Only max_bits is overridable (the arm)."""
	from wnn.ram.experiments.experiment import _phase_defaults
	from wnn.ram.fitness import FitnessCalculatorType
	from wnn.ram.strategies.connectivity.architecture_strategies import ArchitectureConfig
	from wnn.ram.strategies.connectivity.framework import GAConfig
	min_bits, min_neurons, max_neurons = int(params["min_bits"]), int(params["min_neurons"]), int(params["max_neurons"])
	default_bits, default_neurons = _phase_defaults(parents, min_bits, max_bits, min_neurons, max_neurons)
	arch = ArchitectureConfig(
		num_clusters=2, min_bits=min_bits, max_bits=max_bits, min_neurons=min_neurons, max_neurons=max_neurons,
		optimize_bits=False, optimize_neurons=True, optimize_connections=False,
		default_bits=default_bits, default_neurons=default_neurons, bits_grid=None, total_input_bits=total_bits,
		mutable_clusters=None, cluster_crossover_ratio=params["cluster_crossover_ratio"],
		pool_shuffle_ratio=params["pool_shuffle_ratio"], assortative_mating_ratio=params["assortative_mating_ratio"],
		dedupe_neurons=bool(params.get("ga_dedupe_neurons", False)), dedupe_fill=bool(params.get("ga_dedupe_fill", True)),
		crossover_bits=str(params.get("ga_crossover_bits", "inherit")), max_neurons_total=params.get("max_neurons_total"),
		joint_mutation=bool(params.get("ga_joint_mutation", False)), receptive_fields=None,
	)
	ga = GAConfig(
		generations=GA_PHASE_GENERATIONS, population_size=BREEDING_OFFSPRING, patience=int(params["patience"]),
		mutation_rate=float(params.get("mutation_rate", 0.1)), immigrant_fraction=float(params.get("immigrant_fraction", 0.0)),
		initial_threshold=None, threshold_delta=params.get("threshold_step", 1) / 100.0,
		threshold_reference=GA_PHASE_GENERATIONS, fitness_percentile=params.get("fitness_percentile"),
		fitness_calculator_type=FitnessCalculatorType[str(params["fitness_calculator"]).upper()],
		fitness_weight_ce=params["fitness_weight_ce"], fitness_weight_acc=params["fitness_weight_acc"],
		fitness_weight_f1=params["fitness_weight_f1"], fitness_weight_fpr=params["fitness_weight_fpr"],
		fitness_aggregation=params.get("fitness_aggregation", "harmonic"),
		zrank_clamp=params.get("fitness_zrank_clamp", 3.0), ce_anchor=optimizer.desirability_ce_anchor(),
		min_accuracy_floor=0.0,
	)
	return arch, ga


def ga_strategy(params: dict, optimizer, parents: list, max_bits: int, total_bits: int):
	"""The real ArchitectureGAStrategy, positioned at one generation of its loop."""
	from wnn.ram.strategies.factory import OptimizerStrategyFactory, OptimizerStrategyType
	arch, ga = ga_configs(params, optimizer, parents, max_bits, total_bits)
	strategy = OptimizerStrategyFactory.create(OptimizerStrategyType.ARCHITECTURE_GA, ga, arch_config=arch,
		seed=int(params["seed"]), logger=_log, batch_evaluator=optimizer)
	strategy._setup_fitness_calculator()
	strategy._phase_train_idx = BREEDING_TRAIN_IDX
	strategy._reseed_generation(BREEDING_GENERATION)
	return strategy


class SearchRecorder:
	"""Wraps IDSEvaluator.search_offspring: forwards the GA's call untouched and
	keeps what was passed (the call arguments) and what Rust returned."""

	def __init__(self, evaluator):
		self._search = evaluator.search_offspring
		self.arguments: dict = {}
		self.result = None

	def __call__(self, population, logger=None, **arguments):
		self.arguments = dict(arguments, population_ce=[_hex(ce) for _, ce in population])
		self.result = self._search(population=population, logger=logger, **arguments)
		return self.result


def arguments_record(arguments: dict) -> dict:
	"""JSON form of the recorded call (floats exact, nothing dropped)."""
	record = {}
	for key, value in sorted(arguments.items()):
		if isinstance(value, float):
			record[key] = _hex(value)
		elif key == "fitness_scores" and value is not None:
			record[key] = [_hex(v) for v in value]
		else:
			record[key] = value
	return record


def breed_case(optimizer, params: dict, case: tuple, total_bits: int) -> dict:
	name, max_bits, _widths, _duplicates = case
	started = time.time()
	parents = make_parents(case, total_bits, params)
	population = list(zip(parents, optimizer.evaluate_batch(parents)))
	strategy = ga_strategy(params, optimizer, parents, max_bits, total_bits)
	recorder = SearchRecorder(optimizer)
	optimizer.search_offspring = recorder
	try:
		threshold = strategy._compute_threshold(BREEDING_GENERATION / strategy.config.threshold_reference)
		selected = strategy._generate_offspring(population, BREEDING_OFFSPRING, threshold, BREEDING_GENERATION)
	finally:
		del optimizer.search_offspring
	record = {
		"phase_type": int(strategy._phase_type), "offspring_seed": int(strategy._offspring_seed),
		"call_arguments": arguments_record(recorder.arguments),
		"parents": [bred_record(g) for g in parents],
		"returned": [bred_record(g) for g in recorder.result.genomes],
		"evaluated": int(recorder.result.evaluated), "viable": int(recorder.result.viable),
		"selected": [bred_record(g) for g, _metrics in selected],
	}
	_log(f"  {name}: {time.time() - started:.0f}s, {len(record['returned'])} returned of {record['evaluated']} evaluated, "
		f"{len(record['selected'])} selected, child bits up to {max(r['bits_max'] for r in record['returned'])}, "
		f"child duplicate neurons {sum(r['duplicate_neurons'] for r in record['returned'])}")
	return record


def run_breeding(db: sqlite3.Connection, only: list[str] | None) -> dict:
	from wnn.ram.experiments.IdsEvaluatorFactory import IdsEvaluatorFactory
	params = flow_params(db, REFERENCE_FLOW)
	apply_production_env(params)
	s = settings_for(params)
	check_reference_settings(s)
	factory = IdsEvaluatorFactory(log=_log)
	dataset = val_as_report(factory.load(s))
	optimizer, _report = build_pair(factory, dataset, s)
	total_bits = int(dataset.X_train.total_bits)
	cases = [case for case in BREEDING_CASES if only is None or case[0] in only]
	return {"mode": "breeding", "reference_flow": REFERENCE_FLOW, "dataset": dataset_record(dataset, optimizer),
		"generation": BREEDING_GENERATION, "cases": {case[0]: breed_case(optimizer, params, case, total_bits) for case in cases}}


# ------------------------------------------------------------------ W3: retrain the controls' finals

def stored_final(db: sqlite3.Connection, flow_id: int, genome_type: str = DEFAULT_GENOME_TYPE) -> tuple[str, dict, int]:
	"""(genome_hash, stored decode modes, worker_abi) of the GA `genome_type` final row."""
	rows = db.execute("""select ts.genome_hash, ts.threshold_metadata, ts.worker_abi from test_summaries ts
		join experiments e on e.id = ts.experiment_id
		where ts.flow_id=? and e.phase_type='ga_neurons' and ts.test_point='final' and ts.genome_type=?""",
		(flow_id, genome_type)).fetchall()
	if len(rows) != 1:
		raise SystemExit(f"flow {flow_id}: expected one GA {genome_type} final row in test_summaries, found {len(rows)}")
	row = rows[0]
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


def check_retrain_settings(flow_id: int, params: dict, s) -> None:
	"""v2 guard: the resolved settings must carry the flow's STORED rule, theta and
	class balance (no silent default), and the flow must be on the flat path this
	mode rebuilds (factory.flat; receptive fields are attached only by create())."""
	stored_rule = params.get("ids_write_rule") or "lattice"
	checks = {
		"write_rule": (s.write_rule, stored_rule),
		"balance_classes": (bool(s.balance_classes), bool(params.get("balance_classes", False))),
		"classification": (s.classification, "binary"),
		"receptive_fields": (s.receptive_fields, "off"),
		"streaming": (bool(s.streaming), False),
	}
	if stored_rule == "vote":
		checks["vote_share"] = (s.vote_share, params.get("ids_vote_share"))
	wrong = {key: pair for key, pair in checks.items() if pair[0] != pair[1]}
	if s.write_rule not in RETRAIN_RULES:
		wrong["write_rule_supported"] = (s.write_rule, RETRAIN_RULES)
	if wrong:
		raise SystemExit(f"flow {flow_id}: retrain path refused, (resolved, expected) = {wrong}")


def path_settings(flow_id: int, params: dict, s) -> dict:
	"""Provenance (meta): what the worker's construction path resolved for this flow."""
	return {"flow": flow_id, "write_rule": s.write_rule, "vote_share": s.vote_share,
		"balance_classes": bool(s.balance_classes), "memory_mode": s.memory_mode_name,
		"order_independent_train": bool(s.order_independent_train), "neuron_sample_rate": s.neuron_sample_rate,
		"max_bits": s.max_bits, "seed": s.seed, "dataset": {key: getattr(s, key) for key in DATASET_FIELDS}}


def retrain_genome(db: sqlite3.Connection, test_evaluator, flow_id: int, params: dict, genome_type: str) -> dict:
	genome_hash, stored, stored_abi = stored_final(db, flow_id, genome_type)
	genome = stored_genome(db, genome_hash)
	fresh = test_evaluator.evaluate_multiclass_at_thresholds(genome)["modes"]
	verdict = compare_modes(stored, fresh)
	record = {"seed": params["seed"], "genome_hash": genome_hash, "stored_worker_abi": stored_abi,
		"genome": genome_record(genome), "stored_margin_val_cal": mode_record(stored["margin_val_cal"]),
		"retrained_margin_val_cal": mode_record(fresh["margin_val_cal"]), **verdict}
	_log(f"  flow {flow_id} {genome_type}: margin_val_cal {fresh['margin_val_cal']['confusion']} vs stored "
		f"{stored['margin_val_cal']['confusion']} -> exact_match={verdict['exact_match']}")
	return record


def retrain_flow(db: sqlite3.Connection, factory, dataset, flow_id: int, genome_types: list[str],
		params_override: dict | None = None) -> tuple[dict, dict]:
	"""({genome_type: record}, path settings) for one flow, on ONE evaluator pair
	built by the worker's factory from the flow's params (+ an override for the
	negative control only)."""
	params = {**flow_params(db, flow_id), **(params_override or {})}
	apply_production_env(params)
	s = settings_for(params)
	check_retrain_settings(flow_id, params, s)
	_optimizer, test_evaluator = factory.flat(dataset, s)
	records = {gt: retrain_genome(db, test_evaluator, flow_id, params, gt) for gt in genome_types}
	del _optimizer, test_evaluator
	gc.collect()
	return records, path_settings(flow_id, params, s)


def check_shared_dataset(db: sqlite3.Connection, flow_ids: list[int]) -> None:
	"""One dataset load serves every flow only if they agree on what decides it."""
	first = {key: getattr(settings_for(flow_params(db, flow_ids[0])), key) for key in DATASET_FIELDS}
	for flow_id in flow_ids[1:]:
		other = {key: getattr(settings_for(flow_params(db, flow_id)), key) for key in DATASET_FIELDS}
		if other != first:
			raise SystemExit(f"flow {flow_id} loads a different dataset than {flow_ids[0]}: "
				f"{ {k: (first[k], other[k]) for k in first if first[k] != other[k]} }")


def negative_control(db: sqlite3.Connection, factory, dataset, flow_id: int) -> dict:
	"""The first vote flow's best_f1 genome retrained under the LATTICE rule (theta
	dropped): if this also matched, the comparison could not see the write rule."""
	override = {"ids_write_rule": "lattice", "ids_vote_share": None}
	records, settings = retrain_flow(db, factory, dataset, flow_id, [DEFAULT_GENOME_TYPE], override)
	record = records[DEFAULT_GENOME_TYPE]
	return {"flow": flow_id, "override": override, "write_rule": settings["write_rule"],
		"exact_match": record["exact_match"], "all_confusions_match": record["all_confusions_match"],
		"retrained_margin_val_cal": record["retrained_margin_val_cal"], "differs": not record["exact_match"]}


def run_retrain(db: sqlite3.Connection, flow_ids: list[int], genome_types: list[str] | None,
		with_negative_control: bool) -> dict:
	"""The flows share one dataset load (same dataset params; checked). With the
	defaults (best_f1 only, no control) the payload is exactly the v1 payload."""
	from wnn.ram.experiments.IdsEvaluatorFactory import IdsEvaluatorFactory
	check_shared_dataset(db, flow_ids)
	factory = IdsEvaluatorFactory(log=_log)
	dataset = factory.load(settings_for(flow_params(db, flow_ids[0])))
	types = [DEFAULT_GENOME_TYPE] + [gt for gt in (genome_types or []) if gt != DEFAULT_GENOME_TYPE]
	results = {flow_id: retrain_flow(db, factory, dataset, flow_id, types) for flow_id in flow_ids}
	flows = {str(f): records[DEFAULT_GENOME_TYPE] for f, (records, _s) in results.items()}
	exact = [record["exact_match"] for records, _s in results.values() for record in records.values()]
	payload = {"mode": "retrain_finals", "flows": flows, "all_exact": all(exact)}
	if len(types) > 1:
		payload["other_genome_types"] = {str(f): {gt: r for gt, r in records.items() if gt != DEFAULT_GENOME_TYPE}
			for f, (records, _s) in results.items()}
	if with_negative_control:
		payload["negative_control"] = negative_control(db, factory, dataset, flow_ids[0])
	payload["meta_extra"] = {"retrain_paths": [settings for _r, settings in results.values()]}
	return payload


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
	if isinstance(node, list) and any(isinstance(item, (dict, list)) for item in node):
		out = {f"{prefix}/#len": len(node)}
		for index, item in enumerate(node):
			out.update(flatten(item, f"{prefix}/{index}"))
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
	extra = payload.pop("meta_extra", {})
	document = {"meta": run_meta(started, extra), **payload}
	document["payload_sha256"] = _sha(json.dumps(payload, sort_keys=True).encode())
	path = Path(out)
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
	_log(f"wrote {path} (ABI {document['meta']['abi']}, payload {document['payload_sha256']}, "
		f"{document['meta']['runtime_s']}s, peak RSS {document['meta']['peak_rss_gb']} GB)")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--out", help="output JSON (digest / retrain modes)")
	parser.add_argument("--cases", nargs="+", help="digest / breeding mode: only these case names (timing probes)")
	parser.add_argument("--breeding", action="store_true", help="W1b mode: one GA generation's offspring search")
	parser.add_argument("--retrain-finals", nargs="+", type=int, metavar="FLOW", help=f"W3 mode, e.g. {CONTROL_FLOWS}")
	parser.add_argument("--genome-types", nargs="+", metavar="TYPE",
		help="retrain mode (v2): also these GA final genome types, e.g. best_ce best_acc best_fpr best_fitness")
	parser.add_argument("--negative-control", action="store_true",
		help="retrain mode (v2): first flow's best_f1 genome under the lattice rule; must NOT match")
	parser.add_argument("--compare", nargs=2, metavar=("A", "B"), help="compare two output files bit for bit")
	return parser.parse_args()


def run_mode(db: sqlite3.Connection, args: argparse.Namespace) -> dict:
	if args.retrain_finals:
		return run_retrain(db, args.retrain_finals, args.genome_types, args.negative_control)
	if args.breeding:
		return run_breeding(db, args.cases)
	return run_digest(db, args.cases)


def main() -> int:
	args = parse_args()
	if args.compare:
		return run_compare(args.compare[0], args.compare[1])
	if not args.out:
		raise SystemExit("--out is required")
	os.environ.setdefault("RAYON_NUM_THREADS", "4")
	started = time.time()
	db = sqlite3.connect(DB_URI, uri=True)
	payload = run_mode(db, args)
	write_output(payload, args.out, started)
	control_ok = payload.get("negative_control", {}).get("differs", True)
	return 0 if payload.get("all_exact", True) and control_ok else 1


if __name__ == "__main__":
	sys.exit(main())
