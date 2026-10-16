"""IdsEvaluatorFactory: build a flow's IDS evaluators — the ONE construction path.

Extracted from worker._create_ids_evaluators / _create_hierarchical_ids_evaluators
(IDS-37 stage 7) so the worker and the FPGA exporter build bit-identical
evaluators from a flow's real params (dataset, split, encoding, feature
selection, neuron_sample_rate, memory_mode, OI, class weights, run seed).

Flat classification -> (optimizer_eval, test_eval): the optimizer K-folds the 80%
train; the test evaluator trains on the full 80% and reports on the held-out
partition (it shares the optimizer's Rust cache — same rows). Hierarchical ->
(s0_opt, s0_test, s1_opt, s1_test, s1_route).
"""

from __future__ import annotations

from typing import Callable

from wnn.ram.experiments.IdsEvaluatorSettings import IdsEvaluatorSettings


class IdsEvaluatorFactory:

	def __init__(self, log: Callable[[str], None]):
		self._log = log

	def create(self, params: dict) -> tuple:
		s = IdsEvaluatorSettings.from_params(params)
		dataset = self.load(s)
		if s.classification == "hierarchical":
			return self._hierarchical(dataset, s)
		return self.flat(dataset, s)

	# ------------------------------------------------------------------ dataset

	def load(self, s: IdsEvaluatorSettings):
		"""The flow's dataset (split, encoding, feature selection, storage)."""
		self._check_streaming(s)
		raw = " RAW" if s.is_raw_data else ""
		streaming = " (streaming)" if s.streaming else ""
		self._log(f"Loading {s.dataset_name}{raw}{streaming} dataset (classification={s.classification}, split={s.split}, "
			f"feature_selection={s.feature_selection}, invalid_encoding={s.invalid_encoding}, encoded_storage={s.encoded_storage})...")
		from wnn.ids import load_ids_dataset
		dataset = load_ids_dataset(
			s.dataset_name, n_bits=s.n_bits, split=s.split, feature_selection=s.feature_selection,
			rest_bits=s.rest_bits, raw=s.ids_raw, invalid_encoding=s.invalid_encoding,
			auto_max_bits=s.auto_max_bits, encoded_storage=s.encoded_storage,
			storage_dir=s.encoded_storage_dir, streaming=s.streaming,
			streaming_chunk_size=s.streaming_chunk_size,
		)
		self._attach_prefetch(dataset, s)
		return dataset

	@staticmethod
	def _check_streaming(s: IdsEvaluatorSettings) -> None:
		"""Streaming v1 constraints (K-fold is auto-handled by IDSEvaluator since F7)."""
		if not s.streaming:
			return
		if not s.dataset_name.startswith("ciciot2023"):
			raise NotImplementedError(
				f"ids_streaming=True only supports ciciot2023* datasets in v1 (got dataset_name={s.dataset_name!r}). "
				f"cicids2017 and unsw-nb15 streaming variants are tracked as follow-up work.")
		if s.encoded_storage != "memory":
			raise NotImplementedError(
				f"ids_streaming=True is incompatible with encoded_storage={s.encoded_storage!r}; "
				f"streaming and memmap are alternative ways to bound memory.")

	@staticmethod
	def _attach_prefetch(dataset, s: IdsEvaluatorSettings) -> None:
		"""Phase F10: memmap page-cache warm mode, read by IDSEvaluator."""
		try:
			dataset.memmap_prefetch_mode = s.memmap_prefetch
		except AttributeError:
			pass  # frozen dataclass — IDSEvaluator falls back to its default

	# ------------------------------------------------------------------ flat

	def flat(self, dataset, s: IdsEvaluatorSettings) -> tuple:
		"""(optimizer_eval, test_eval) for flat (binary / multi) classification."""
		binary = s.classification == "binary"
		y_train = dataset.y_train_binary if binary else dataset.y_train_multi
		y_test = dataset.y_test_binary if binary else dataset.y_test_multi
		classes = 2 if binary else len(dataset.category_names)
		self._log(f"  Train: {len(y_train):,} examples, Test: {len(y_test):,} examples, "
			f"Classes: {classes}, Features: {dataset.X_train.shape[1]}")
		self._log("Using 80/20 split: K-fold on 80% train, 20% val for reporting only")
		optimizer = self._optimizer(dataset, s)
		test = self._test(dataset, s, getattr(optimizer, "_cache", None))
		self._set_fitness_weights(s, optimizer, test)
		return optimizer, test

	def _optimizer(self, dataset, s: IdsEvaluatorSettings):
		kfold = f", k_folds={s.k_folds}" if s.k_folds > 1 else ""
		kfpg = f", kfold_per_gen={s.kfold_per_gen}" if s.kfold_per_gen > 1 else ""
		self._log(f"Creating optimizer IDSEvaluator ({len(dataset.X_train):,} train / "
			f"{len(dataset.X_test):,} eval, {s.num_parts} parts{kfold}{kfpg})...")
		sc = ", single-cluster" if s.single_cluster else ""
		self._log(f"  neuron_sample_rate={s.neuron_sample_rate}, balance_classes={s.balance_classes}{sc}")
		us = ", undersample" if s.undersample_majority else ""
		self._log(f"  undersample_majority={s.undersample_majority}{us}")
		return self._evaluator(dataset, s, s.num_parts, s.k_folds, s.kfold_per_gen, None)

	def _test(self, dataset, s: IdsEvaluatorSettings, shared_cache):
		"""Full 80% train -> held-out report. Reuses the optimizer's Rust cache (M2:
		same dataset, ~19 GB saved); None in streaming mode -> builds its own."""
		balance = ", balanced" if s.balance_classes else ""
		share = ", shared cache" if shared_cache is not None else ""
		self._log(f"Creating test IDSEvaluator ({len(dataset.X_train):,} train / "
			f"{len(dataset.X_test):,} eval, 1 part{balance}{share})...")
		return self._evaluator(dataset, s, 1, 1, 1, shared_cache)

	@staticmethod
	def _evaluator(dataset, s: IdsEvaluatorSettings, num_parts: int, k_folds: int, kfold_per_gen: int, reuse_cache):
		"""One flat IDSEvaluator with the flow's training settings (k_folds=1,
		kfold_per_gen=1, reuse_cache=None are IDSEvaluator's own defaults)."""
		from wnn.ram.architecture.ids_evaluator import IDSEvaluator
		return IDSEvaluator(
			dataset=dataset, classification=s.classification, num_parts=num_parts,
			empty_value=s.empty_value, memory_mode=s.memory_mode, seed=s.seed, run_seed=s.run_seed,
			coverage_aware=s.coverage_aware, write_rule=s.write_rule, num_negatives=s.num_negatives,
			two_cluster_fitness_f1=s.two_cluster_fitness_f1,
			neuron_sample_rate=s.neuron_sample_rate, k_folds=k_folds, kfold_per_gen=kfold_per_gen,
			balance_classes=s.balance_classes, single_cluster=s.single_cluster,
			undersample_majority=s.undersample_majority, flip_labels=s.flip_labels,
			class_weight_multiplier=s.class_weight_multiplier, reuse_cache=reuse_cache,
		)

	def _set_fitness_weights(self, s: IdsEvaluatorSettings, *evaluators) -> None:
		"""Single-cluster threshold sweep maximizes the flow's fitness, not F1."""
		w_ce, w_f1, w_fpr, w_acc = s.fitness_weights
		if not (s.single_cluster and (w_f1 > 0 or w_fpr > 0 or w_acc > 0)):
			return
		self._log(f"  Fitness threshold: CE={w_ce}, F1={w_f1}, FPR={w_fpr}, Acc={w_acc}")
		for evaluator in evaluators:
			evaluator.set_fitness_weights(w_ce, w_f1, w_fpr, w_acc)

	# ------------------------------------------------------------------ hierarchical

	def _hierarchical(self, dataset, s: IdsEvaluatorSettings) -> tuple:
		"""S0 binary gate (all flows) + S1 attack-type classifier (attack flows),
		plus S1-route: same S1 training, the WHOLE test partition (S0-routed rows)."""
		from wnn.ids import create_attack_only_dataset
		self._log("── S0: Binary (Normal vs Attack) ──")
		self._log(f"  Train: {len(dataset.y_train_binary):,}, Test: {len(dataset.y_test_binary):,}, Classes: 2")
		s0_opt, s0_test = self._stage_pair(dataset, s, "binary", "S0")
		self._log("── S1: Attack Classification (9 types) ──")
		s1_opt, s1_test = self._stage_pair(create_attack_only_dataset(dataset), s, "multi", "S1")
		s1_route = self._s1_route(dataset, s)
		return s0_opt, s0_test, s1_opt, s1_test, s1_route

	def _stage_pair(self, dataset, s: IdsEvaluatorSettings, classification: str, label: str) -> tuple:
		from wnn.ids import split_train_validation
		train_val, test_ds = split_train_validation(dataset, val_fraction=s.val_fraction, seed=s.seed)
		kfold = f", k_folds={s.k_folds}" if s.k_folds > 1 else ""
		self._log(f"  {label} optimizer: {len(train_val.X_train):,} train / {len(train_val.X_test):,} eval, {s.num_parts} parts{kfold}")
		opt = self._stage_evaluator(train_val, s, classification, s.num_parts, s.k_folds)
		classes = f", Classes: {len(dataset.category_names)}" if classification == "multi" else ""
		self._log(f"  {label} test: {len(test_ds.X_train):,} train / {len(test_ds.X_test):,} eval{classes}")
		return opt, self._stage_evaluator(test_ds, s, classification, 1, 1)

	def _s1_route(self, dataset, s: IdsEvaluatorSettings):
		"""Same attack-only TRAIN, full test partition: its own eval metrics are
		meaningless (benign rows included) and must never be reported."""
		from wnn.ids import create_attack_only_dataset, split_train_validation
		route = create_attack_only_dataset(dataset, keep_full_test=True)
		_, route_ds = split_train_validation(route, val_fraction=s.val_fraction, seed=s.seed)
		self._log(f"  S1 route: {len(route_ds.X_train):,} train / {len(route_ds.X_test):,} eval "
			f"(full test partition, for S0-routed scoring)")
		return self._stage_evaluator(route_ds, s, "multi", 1, 1)

	@staticmethod
	def _stage_evaluator(dataset, s: IdsEvaluatorSettings, classification: str, num_parts: int, k_folds: int):
		"""Hierarchical stages take only the shared training settings (no
		single-cluster / undersample / class-weight knobs — as before)."""
		from wnn.ram.architecture.ids_evaluator import IDSEvaluator
		return IDSEvaluator(
			dataset=dataset, classification=classification, num_parts=num_parts,
			empty_value=s.empty_value, memory_mode=s.memory_mode, seed=s.seed, run_seed=s.run_seed,
			coverage_aware=s.coverage_aware, write_rule=s.write_rule, num_negatives=s.num_negatives,
			neuron_sample_rate=s.neuron_sample_rate, k_folds=k_folds, balance_classes=s.balance_classes,
		)
