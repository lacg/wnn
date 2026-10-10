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
