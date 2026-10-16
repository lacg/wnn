"""
Resolve IDS flow params to the values the worker ACTUALLY trains/scores with.

Single source for the resolutions that were inlined (and duplicated) in
worker._create_ids_evaluators / _create_hierarchical_ids_evaluators. The
validation-cache key (validation_cache_key.py) uses the SAME resolvers, so the
key always describes the configuration that really ran — absent, explicit and
unrecognised memory_mode strings that all train as QUAD_WEIGHTED share a key;
anything that trains differently never does.
"""

from types import MappingProxyType
from typing import Optional

# memory_mode string -> EvalSettings.memory_mode code (IDSCacheWrapper.set_memory_mode).
IDS_MEMORY_MODE_CODES = MappingProxyType({
	"TERNARY": 0, "QUAD_BINARY": 1, "QUAD_WEIGHTED": 2, "BINARY": 3, "QSR": 4, "PLN": 5,
})
DEFAULT_MEMORY_MODE = "QUAD_WEIGHTED"

# Datasets that are raw by construction: their invalid-encoding default is single_bit.
RAW_BY_CONSTRUCTION_DATASETS = frozenset({
	"ciciot2023_canonical", "ciciot2023_neto_full", "ciciot2023_neto_subsample",
})


def resolve_memory_mode(params: dict) -> str:
	"""Canonical memory-mode name. Absent, None or unrecognised -> QUAD_WEIGHTED
	(exactly the worker's historical `map.get(mode_str, 2)` fallback)."""
	mode = params.get("memory_mode", DEFAULT_MEMORY_MODE)
	return mode if mode in IDS_MEMORY_MODE_CODES else DEFAULT_MEMORY_MODE


# ids_write_rule values (IDSCacheWrapper.set_write_rule). "lattice" = each
# mode's own writer (BINARY Set, TERNARY/PLN TRUE-wins lattice); "vote" = the
# majority-vote writer, which exists only for those three modes.
WRITE_RULES = frozenset({"lattice", "vote"})
DEFAULT_WRITE_RULE = "lattice"
VOTE_MODES = frozenset({"BINARY", "TERNARY", "PLN"})


def resolve_write_rule(params: dict) -> str:
	"""Canonical ids_write_rule. Absent/None -> "lattice" (bit-identical to every
	flow queued before ABI 19). Unknown values and "vote" on a mode without a
	vote writer (the QUAD family tallies through its own OI counters) raise —
	a typo or a silently ignored selector would mislabel an A/B arm."""
	rule = params.get("ids_write_rule") or DEFAULT_WRITE_RULE
	if rule not in WRITE_RULES:
		raise ValueError(f"ids_write_rule={rule!r}: expected one of {sorted(WRITE_RULES)}")
	mode = resolve_memory_mode(params)
	if rule == "vote" and mode not in VOTE_MODES:
		raise ValueError(
			f"ids_write_rule='vote' needs memory_mode in {sorted(VOTE_MODES)}; got {mode} "
			"(the 4-state modes already train through vote-tallying OI counters)"
		)
	return rule


# ids_two_cluster_fitness_f1 values (IDSCacheWrapper.set_two_cluster_fitness_f1):
# which F1 a binary TWO-cluster search reports to the GA fitness calculators and
# the patience tracker. "macro" = the 2-class macro F1 (today, bit-identical);
# "attack" = the attack class's F1.
TWO_CLUSTER_FITNESS_F1 = frozenset({"macro", "attack"})
DEFAULT_TWO_CLUSTER_FITNESS_F1 = "macro"


def resolve_two_cluster_fitness_f1(params: dict) -> str:
	"""Canonical ids_two_cluster_fitness_f1. Absent/None -> "macro". Unknown
	values raise; so does any non-default value on a flow that is not binary
	two-cluster (ids_classification='binary', ids_single_cluster=false) — there
	the selector would silently do nothing and mislabel the arm."""
	value = params.get("ids_two_cluster_fitness_f1") or DEFAULT_TWO_CLUSTER_FITNESS_F1
	if value not in TWO_CLUSTER_FITNESS_F1:
		raise ValueError(
			f"ids_two_cluster_fitness_f1={value!r}: expected one of {sorted(TWO_CLUSTER_FITNESS_F1)}"
		)
	two_cluster_binary = (
		params.get("ids_classification", "binary") == "binary"
		and not bool(params.get("ids_single_cluster", False))
	)
	if value != DEFAULT_TWO_CLUSTER_FITNESS_F1 and not two_cluster_binary:
		raise ValueError(
			f"ids_two_cluster_fitness_f1={value!r} applies to binary two-cluster flows only "
			"(ids_classification='binary', ids_single_cluster=false); got "
			f"ids_classification={params.get('ids_classification', 'binary')!r}, "
			f"ids_single_cluster={bool(params.get('ids_single_cluster', False))}"
		)
	return value


def memory_mode_code(mode: str) -> int:
	"""EvalSettings code for a canonical mode name (from resolve_memory_mode)."""
	return IDS_MEMORY_MODE_CODES[mode]


def empty_value_for(mode: str) -> float:
	"""TERNARY/PLN decode the EMPTY u-state as 0.5; every other mode uses 0.0."""
	return 0.5 if mode in ("TERNARY", "PLN") else 0.0


def resolve_invalid_encoding(params: dict) -> Optional[str]:
	"""ids_invalid_encoding with the dataset-dependent default: single_bit for
	raw data (ids_raw or a raw-by-construction dataset), else none."""
	dataset_name = params.get("ids_dataset", "unsw-nb15")
	is_raw = params.get("ids_raw", False) or dataset_name in RAW_BY_CONSTRUCTION_DATASETS
	return params.get("ids_invalid_encoding", "single_bit" if is_raw else "none")


def oi_env_enabled(env_value: Optional[str]) -> bool:
	"""Mirror of ram_core neuron_memory::order_independent_training_enabled():
	WNN_ORDER_INDEPENDENT_TRAIN is on iff "1" or "true" (any case)."""
	return env_value is not None and (env_value == "1" or env_value.lower() == "true")
