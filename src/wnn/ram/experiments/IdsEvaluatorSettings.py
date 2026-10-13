"""IdsEvaluatorSettings: the IDS flow params resolved ONCE to the values the worker trains/scores with.

Every default here is the worker's historical default (moved verbatim from
worker._create_ids_evaluators, IDS-37 stage 7), so a flow's evaluators are built
identically by the worker and by the FPGA exporter (IdsEvaluatorFactory).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from wnn.ram.experiments.ids_param_resolution import (
	RAW_BY_CONSTRUCTION_DATASETS,
	empty_value_for,
	memory_mode_code,
	resolve_invalid_encoding,
	resolve_memory_mode,
)

OI_ENV = "WNN_ORDER_INDEPENDENT_TRAIN"


@dataclass(frozen=True)
class IdsEvaluatorSettings:
	classification: str
	n_bits: int
	val_fraction: float
	num_parts: int
	k_folds: int
	kfold_per_gen: int
	split: str
	seed: int
	neuron_sample_rate: float
	balance_classes: bool
	single_cluster: bool
	undersample_majority: bool
	flip_labels: bool
	class_weight_multiplier: float
	memory_mode_name: str
	memory_mode: int
	empty_value: float
	coverage_aware: bool
	num_negatives: Optional[int]
	feature_selection: str
	rest_bits: Optional[int]
	auto_max_bits: int
	ids_raw: bool
	dataset_name: str
	encoded_storage: str
	encoded_storage_dir: Optional[str]
	streaming: bool
	streaming_chunk_size: int
	memmap_prefetch: str
	invalid_encoding: str
	fitness_weights: tuple[float, float, float, float]
	order_independent_train: bool

	@classmethod
	def from_params(cls, params: dict) -> IdsEvaluatorSettings:
		mode_str = resolve_memory_mode(params)
		return cls(
			classification=params.get("ids_classification", "binary"),
			n_bits=params.get("ids_n_bits", 8),
			val_fraction=params.get("ids_val_fraction", 0.25),
			num_parts=params.get("ids_num_parts", 5),
			k_folds=params.get("ids_k_folds", 5),
			kfold_per_gen=params.get("ids_kfold_per_gen", 1),
			split=params.get("ids_split", "standard"),
			seed=int(params.get("seed", 42)),
			neuron_sample_rate=params.get("neuron_sample_rate", 0.25),
			balance_classes=params.get("balance_classes", False),
			single_cluster=params.get("ids_single_cluster", False),
			undersample_majority=params.get("undersample_majority", False),
			flip_labels=params.get("flip_labels", False),
			class_weight_multiplier=params.get("class_weight_multiplier", 1.0),
			memory_mode_name=mode_str,
			memory_mode=memory_mode_code(mode_str),
			empty_value=empty_value_for(mode_str),
			coverage_aware=bool(params.get("ids_coverage_aware", False)),
			num_negatives=0 if bool(params.get("ids_suppress_negatives", False)) else None,
			feature_selection=params.get("ids_feature_selection", "all"),
			rest_bits=params.get("ids_rest_bits", None),
			auto_max_bits=params.get("ids_auto_max_bits", 32),
			ids_raw=params.get("ids_raw", False),
			dataset_name=params.get("ids_dataset", "unsw-nb15"),
			encoded_storage=params.get("ids_encoded_storage", "memory"),
			encoded_storage_dir=params.get("ids_encoded_storage_dir", None),
			streaming=params.get("ids_streaming", False),
			streaming_chunk_size=params.get("ids_streaming_chunk_size", 1_000_000),
			memmap_prefetch=params.get("ids_memmap_prefetch", "touch"),
			invalid_encoding=resolve_invalid_encoding(params),
			fitness_weights=(
				params.get("fitness_weight_ce", 0), params.get("fitness_weight_f1", 0),
				params.get("fitness_weight_fpr", 0), params.get("fitness_weight_acc", 0),
			),
			order_independent_train=cls.oi_requested(params),
		)

	@staticmethod
	def oi_requested(params: dict) -> bool:
		"""`wnn_order_independent_train` truthiness (True / 1 / "1" / "true")."""
		oi = params.get("wnn_order_independent_train")
		return oi is True or oi == 1 or (isinstance(oi, str) and oi.lower() in ("1", "true"))

	@staticmethod
	def apply_oi_env(params: dict) -> tuple[Optional[str], bool]:
		"""Set the env var the Rust trainer reads when the flow asks for OI.
		Returns (previous value, whether it was enabled) so callers restore it."""
		previous = os.environ.get(OI_ENV)
		enabled = IdsEvaluatorSettings.oi_requested(params)
		if enabled:
			os.environ[OI_ENV] = "1"
		return previous, enabled

	@property
	def is_raw_data(self) -> bool:
		return self.ids_raw or self.dataset_name in RAW_BY_CONSTRUCTION_DATASETS

	@property
	def run_seed(self) -> int:
		"""QSR/PLN coin seed: the flow seed (each run reproducible, a cohort varies)."""
		return self.seed
