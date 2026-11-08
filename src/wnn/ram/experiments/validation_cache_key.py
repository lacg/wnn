"""
Cross-flow validation-cache key (PAPER-CRITICAL fix, 10/10/2026).

The cache matches a genome by genome_hash (bits + neurons + connections), and
same-seed grid searches regenerate identical genomes in every flow — so the key
MUST pin every flow param that changes how a FIXED genome is trained or scored
on the final validation. The old key ({ds}_{nb}b_{sp}{_raw}{_inv}{_oi}) omitted
memory_mode, feature selection, classification, the QSR/PLN coin seed, … — a
QUAD flow inherited a QSR flow's TEST row, an ABI-13 QSR flow a void pre-ABI-13
row. The trainer version is pinned separately (ValidationCacheScope.worker_abi).

INCLUDED (changes the trained memory or the reported score of a fixed genome):
  dataset/encoding  ids_dataset, ids_n_bits, ids_split, ids_feature_selection,
                    ids_rest_bits, ids_auto_max_bits, ids_raw, ids_invalid_encoding
                    (resolved incl. its dataset-dependent default), ids_streaming
                    (+ chunk size: the streaming encoder fits by t-digest),
                    ids_encoding_dedupe (encoder version, only when "on"),
                    ids_feature_ranking (list version, only when "train_only"),
                    ids_receptive_fields (family-map version + ids_rf_cross, only
                    when "family" — strictly it steers the search, which
                    genome_hash already pins; kept CONSERVATIVELY so a row is never
                    shared across family-map versions)
  model/training    ids_classification, ids_single_cluster, memory_mode (resolved),
                    seed (QSR/PLN coin run_seed; hierarchical split seed),
                    ids_write_rule (only when "vote"; lattice keys unchanged),
                    neuron_sample_rate, ids_coverage_aware, ids_suppress_negatives,
                    balance_classes, undersample_majority, flip_labels,
                    class_weight_multiplier, ids_val_fraction (hierarchical only),
                    effective WNN_ORDER_INDEPENDENT_TRAIN
  scoring           single-cluster threshold-fitness weights set on the test
                    evaluator (fitness_weight_*, default 0); the experiment's own
                    fitness weights are appended by the Experiment (empirical_cumulative)
EXCLUDED (steer the search only): fitness aggregation/calculator/anchors,
  ids_two_cluster_fitness_f1 (the search F1; validation decodes are unaffected),
  ids_offspring_folds (which folds a search genome's fitness averages),
  ids_calib_sample (the search threshold's calibration rows), GA/TS
  budgets, patience, grids, ids_k_folds / ids_num_parts / ids_kfold_per_gen
  (the validation evaluator is num_parts=1, no K-fold), storage/prefetch and
  performance knobs (memmap, HSR, threads, no_metal, batch size), ids_arch_type
  and tier sizing (they shape the genome, which genome_hash already pins).

Bump VALIDATION_CACHE_VERSION whenever Python-side validation/calibration
semantics change without an accelerator ABI bump — it invalidates every row.
"""

import json

from wnn.ram.experiments.ids_param_resolution import (
	DEFAULT_WRITE_RULE,
	resolve_encoding_dedupe,
	resolve_feature_ranking,
	resolve_invalid_encoding,
	resolve_memory_mode,
	resolve_receptive_fields,
	resolve_rf_cross,
	resolve_write_rule,
)

VALIDATION_CACHE_VERSION = 2
# Bump when the deduped encoder's semantics change (thermometer_dedupe.py /
# duplicate_columns.py) — it re-keys only ids_encoding_dedupe="on" flows.
ENCODER_VERSION_DEDUPE = "dedupe1"


def build_validation_cache_key(params: dict, oi_enabled: bool) -> str:
	"""Flow-level cache key from flows.config_json.params + the effective OI flag."""
	parts = _dataset_parts(params) + _model_parts(params) + _scoring_parts(params)
	parts.append(("oi", bool(oi_enabled)))
	body = "|".join(f"{name}={canon_value(value)}" for name, value in parts)
	return f"v{VALIDATION_CACHE_VERSION}|{body}"


def canon_value(value) -> str:
	"""Canonical text: None '-', bools 1/0, floats repr, else str / compact JSON."""
	if value is None:
		return "-"
	if isinstance(value, bool):
		return "1" if value else "0"
	if isinstance(value, (int, float, str)):
		return repr(value) if isinstance(value, float) else str(value)
	return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _dataset_parts(params: dict) -> list:
	"""Which rows exist and how they are encoded into bits."""
	streaming = bool(params.get("ids_streaming", False))
	return [
		("ds", params.get("ids_dataset", "unsw-nb15")),
		("nb", params.get("ids_n_bits", 8)),
		("sp", params.get("ids_split", "standard")),
		("fs", params.get("ids_feature_selection", "all")),
		("rest", params.get("ids_rest_bits")),
		("amb", params.get("ids_auto_max_bits", 32)),
		("raw", bool(params.get("ids_raw", False))),
		("inv", resolve_invalid_encoding(params)),
		("strm", params.get("ids_streaming_chunk_size", 1_000_000) if streaming else None),
	] + _encoder_version_parts(params) + _feature_ranking_parts(params) + _receptive_field_parts(params)


def _receptive_field_parts(params: dict) -> list:
	"""ids_receptive_fields (ABI 27): emitted ONLY for "family" (map version +
	cross budget), so every default key stays byte-identical."""
	if resolve_receptive_fields(params) == "off":
		return []
	from wnn.ids.feature_families import FAMILY_MAP_VERSION
	return [("rf", f"{FAMILY_MAP_VERSION}:{resolve_rf_cross(params)}")]


def _feature_ranking_parts(params: dict) -> list:
	"""IDS-53 ranking version: emitted ONLY for ids_feature_ranking="train_only"
	(the directory version of the list file), so legacy keys stay byte-identical."""
	from wnn.ids.feature_ranking import TRAIN_ONLY_VERSION
	return [("fr", TRAIN_ONLY_VERSION)] if resolve_feature_ranking(params) == "train_only" else []


def _encoder_version_parts(params: dict) -> list:
	"""IDS-52 encoder version: emitted ONLY for ids_encoding_dedupe="on" (a
	different bit layout), so every default-encoding key stays byte-identical."""
	return [("enc", ENCODER_VERSION_DEDUPE)] if resolve_encoding_dedupe(params) else []


def _model_parts(params: dict) -> list:
	"""How a fixed genome's memory is trained (and decoded) on the full train set."""
	classification = params.get("ids_classification", "binary")
	hierarchical = classification == "hierarchical"
	return [
		("cls", classification),
		("sc", bool(params.get("ids_single_cluster", False))),
		("mm", resolve_memory_mode(params)),
		("seed", int(params.get("seed", 42))),
		("nsr", float(params.get("neuron_sample_rate", 0.25))),
		("cov", bool(params.get("ids_coverage_aware", False))),
		("negs", bool(params.get("ids_suppress_negatives", False))),
	] + _write_rule_parts(params) + _sampling_parts(params) + [
		("hvf", float(params.get("ids_val_fraction", 0.25)) if hierarchical else None),
	]


def _write_rule_parts(params: dict) -> list:
	"""ids_write_rule changes the trained cells. Emitted ONLY when non-default,
	so every pre-ABI-19 (lattice) key stays byte-identical."""
	rule = resolve_write_rule(params)
	return [] if rule == DEFAULT_WRITE_RULE else [("wr", rule)]


def _sampling_parts(params: dict) -> list:
	"""Training-set reweighting / relabelling knobs."""
	return [
		("bal", bool(params.get("balance_classes", False))),
		("us", bool(params.get("undersample_majority", False))),
		("flip", bool(params.get("flip_labels", False))),
		("cwm", float(params.get("class_weight_multiplier", 1.0))),
	]


def _scoring_parts(params: dict) -> list:
	"""Threshold-fitness weights the worker sets on the TEST evaluator
	(worker._create_ids_evaluators: single-cluster, non-hierarchical, any of
	f1/fpr/acc > 0) — they move the train_cal threshold."""
	weights = [float(params.get(f"fitness_weight_{m}", 0)) for m in ("ce", "f1", "fpr", "acc")]
	applied = (
		params.get("ids_classification", "binary") != "hierarchical"
		and bool(params.get("ids_single_cluster", False))
		and any(w > 0 for w in weights[1:])
	)
	return [("thw", weights if applied else None)]
