"""
IDS Evaluator — Genome evaluator for intrusion detection classification.

Parallel to TieredEvaluator but operates on pre-encoded binary features
instead of token sequences. Uses the Rust IDSCacheWrapper for zero-copy
stratified subset rotation and hybrid CPU+GPU evaluation.

Usage:
	from wnn.ids import load_unsw_nb15
	dataset = load_unsw_nb15(n_bits=8)

	evaluator = IDSEvaluator(dataset, classification="binary", num_parts=3)

	# Per iteration: evaluate genomes with subset rotation
	results = evaluator.evaluate_batch(genomes)

	# Final evaluation with full data
	final_results = evaluator.evaluate_batch_full(genomes)
"""

import time
from typing import Optional, Callable, Any

import numpy as np

from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
from wnn.ram.metrics import Metrics
from wnn.ram.architecture.base_evaluator import BaseEvaluator, EvalResult, OffspringSearchResult


def _compute_class_weights(labels: np.ndarray, num_classes: int, multiplier: float = 1.0) -> "list[int]":
	"""Mirror of Rust adaptive::compute_class_weights_with_multiplier.

	For each class c with count `n_c`, weight = max(1, max_count / n_c * multiplier).
	Used to oversample minority classes via per-row repetition during training.
	Returns a list of uint32 weights of length `num_classes`.

	Streaming mode (Phase F9): the streaming IDSEvaluator computes weights from
	the already-materialized y_train labels (small — 8 bytes/row, ~370 MB
	for 46M). This is the "balance_classes pre-pass" that avoids re-streaming
	just to count labels.
	"""
	counts = np.bincount(labels.astype(np.int64), minlength=num_classes)
	if counts.size > num_classes:
		counts = counts[:num_classes]
	max_count = int(counts.max()) if counts.max() > 0 else 1
	weights = []
	for c in counts:
		if c == 0:
			weights.append(1)
		else:
			base = max(max_count // int(c), 1)
			weights.append(max(int(base * multiplier), 1))
	return weights


class _ReplacedXDataset:
	"""Lightweight proxy that overrides X_train/X_test on an existing dataset.

	Used by IDSEvaluator's F7 auto-detect path: when a streaming dataset is
	small enough to materialize via memmap, we wrap the original dataset and
	swap in MemmapEncoded X_train/X_test while preserving all the labels and
	metadata (y_train_binary, y_train_multi, encoder, category_names, etc.).

	Attribute access falls through to the underlying dataset for anything
	we don't override — this keeps the proxy invisible to downstream code
	that reads dataset.* attributes.
	"""

	__slots__ = ("_inner", "X_train", "X_test", "X_val")

	def __init__(self, inner, X_train, X_test, X_val=None):
		object.__setattr__(self, "_inner", inner)
		object.__setattr__(self, "X_train", X_train)
		object.__setattr__(self, "X_test", X_test)
		object.__setattr__(self, "X_val", X_val if X_val is not None else getattr(inner, "X_val", None))

	def __getattr__(self, name):
		# __getattr__ only called for attrs not found by normal lookup, so
		# X_train/X_test/X_val (set via __init__) don't reach here.
		return getattr(self._inner, name)


class IDSEvaluator(BaseEvaluator):
	"""
	Rust-backed evaluator for IDS classification tasks.

	Holds all pre-encoded binary features in Rust memory. Per-iteration
	evaluations use zero-copy stratified subset selection via indices.
	"""

	def __init__(
		self,
		dataset,  # IDSDataset
		classification: str = "binary",  # "binary" or "multi"
		num_parts: int = 3,
		num_negatives: Optional[int] = None,  # None = num_classes - 1 (exhaustive)
		empty_value: float = 0.0,
		memory_mode: int = 2,  # TERNARY=0/QUAD_BINARY=1/QUAD_WEIGHTED=2/BINARY=3/QSR=4/PLN=5
		run_seed: int = 0,     # QSR/PLN stochastic-coin seed (ignored by deterministic modes)
		seed: Optional[int] = None,
		log_path: Optional[str] = None,
		neuron_sample_rate: float = 0.25,
		k_folds: int = 1,  # 1 = no k-fold (original behavior), >1 = K-fold CV
		kfold_per_gen: int = 1,  # How many folds to eval per generation (1..k_folds)
		balance_classes: bool = False,
		single_cluster: bool = False,  # Single-cluster discriminator mode
		undersample_majority: bool = False,  # Undersample majority class to match minority
		flip_labels: bool = False,  # Swap normal↔attack labels (detect normals in attack-heavy data)
		class_weight_multiplier: float = 1.0,  # Scale class balancing weights (>1 = stronger minority boost)
		reuse_cache: Optional[Any] = None,  # M2: share an existing IDSCacheWrapper instead of building one
	):
		if classification == "binary":
			y_train = dataset.y_train_binary
			y_test = dataset.y_test_binary
			y_val = getattr(dataset, "y_val_binary", None)
			if flip_labels:
				y_train = 1 - y_train  # 0→1, 1→0
				y_test = 1 - y_test
				if y_val is not None:
					y_val = 1 - y_val
			num_classes = 2
		elif classification == "multi":
			y_train = dataset.y_train_multi
			y_test = dataset.y_test_multi
			y_val = getattr(dataset, "y_val_multi", None)
			num_classes = len(dataset.category_names)
		else:
			raise ValueError(f"classification must be 'binary' or 'multi', got '{classification}'")

		total_features = dataset.X_train.shape[1]

		if single_cluster:
			num_negatives = 0  # No negative clusters in single-cluster mode

		if num_negatives is None:
			num_negatives = num_classes - 1

		# Single-cluster mode: 1 genome cluster (threshold discriminator)
		# Multi-cluster mode: num_classes genome clusters (softmax argmax)
		genome_clusters = 1 if single_cluster else num_classes

		# BaseEvaluator expects LM-style args; pass minimal values
		# vocab_size controls genome cluster count via experiment.py
		super().__init__(
			train_tokens=[],
			eval_tokens=[],
			vocab_size=genome_clusters,
			context_size=1,
			num_parts=num_parts,
			seed=seed,
			neuron_sample_rate=neuron_sample_rate,
			log_path=log_path,
		)

		# Override the total_input_bits computed by BaseEvaluator
		self._total_input_bits = total_features
		self._empty_value = empty_value
		self._num_classes = num_classes
		self._classification = classification
		self._y_test = [int(y) for y in y_test]
		self._y_train = [int(y) for y in y_train]
		# Protocol v2: val-partition labels (3-way splits) for threshold
		# calibration; None for legacy 2-way datasets.
		self._y_val = [int(y) for y in y_val] if y_val is not None else None
		self._class_names = list(dataset.category_names) if hasattr(dataset, 'category_names') else None

		# Per-class breakdown support: stash per-row attack-class indices so the
		# validation phase can group binary predictions by subclass.
		self._y_test_multi = None
		if classification == "binary" and hasattr(dataset, "y_test_multi") and dataset.y_test_multi is not None:
			self._y_test_multi = [int(y) for y in dataset.y_test_multi]

		# Import and create Rust cache
		try:
			import ram_accelerator
		except ImportError:
			raise ImportError(
				"ram_accelerator not available. Build with: "
				"cd src/wnn/ram/strategies/accelerator && maturin develop --release"
			)

		# Detect streaming mode: if X_train is a StreamingEncoded, we route
		# through IDSGenomeStreamer (per-genome streaming eval) instead of
		# building a full in-RAM IDSCacheWrapper. The two paths share config
		# but differ fundamentally in compute pattern — streaming makes one
		# pass per train + one pass per score, per genome.
		#
		# Phase F7 auto-detect: if the streaming dataset is small enough to
		# fit in a memmap (under streaming_materialize_threshold_gb), drain
		# the stream once to disk and switch to the in-memory path. This
		# gives the best of both worlds for moderately-sized streaming flows:
		# bounded memory during load, fast random-access K-fold afterwards.
		from wnn.ids.encoded_array import StreamingEncoded, MemmapEncoded, write_stream_to_memmap
		# Phase F10: optional memmap page-cache warmup. Controlled by an
		# attribute on the dataset object (set by loaders/worker) or by a
		# kwarg-equivalent attribute. Modes: "touch" (default, eager full
		# read), "willneed" (async hint), "none". For Phase 6 V2-class
		# K-fold flows the touch pass amortizes cold-cache faults across
		# all subsequent K-fold reads.
		memmap_prefetch_mode = str(getattr(dataset, "memmap_prefetch_mode", "touch"))
		_streaming_input = isinstance(dataset.X_train, StreamingEncoded)
		# Streaming + multiclass (Gap #4, closed 11/07/2026): the chunk factory
		# still bundles BINARY labels (row carriers for the binary path), but
		# multiclass streaming ignores them — the materialized y_*_multi arrays
		# are in stream order BY CONSTRUCTION (_materialize_labels iterates the
		# same factory), so the streaming helpers slice them by global chunk
		# offset. The F7 auto-materialize path is label-safe too: memmap rows
		# keep stream order and _ReplacedXDataset preserves the original
		# y_*_multi.
		if _streaming_input and k_folds > 1:
			est_train_gb = (dataset.X_train.n_rows * dataset.X_train.bytes_per_row) / (1024 ** 3)
			est_test_gb = (dataset.X_test.n_rows * dataset.X_test.bytes_per_row) / (1024 ** 3)
			total_gb = est_train_gb + est_test_gb
			# default: < 8 GB packed → materialize; ≥ 8 GB → streaming K-fold.
			# Explicit is-None check so threshold_gb=0.0 means "force streaming".
			_override = getattr(dataset, "streaming_materialize_threshold_gb", None)
			threshold_gb = float(_override) if _override is not None else 8.0
			if total_gb < threshold_gb:
				# Materialize to memmap, replace X_train/X_test on the
				# dataset, fall through to the in-memory path.
				print(f"[IDSEvaluator] streaming dataset ~{total_gb:.2f} GB packed < "
				      f"{threshold_gb:.1f} GB threshold; materializing to memmap "
				      f"for K-fold-friendly random access...")
				# Pass prefetch through — the just-written pages are already
				# in the OS write cache, so "none" is fine. K-fold reads
				# will keep them hot. Caller can override via the attribute.
				X_train_mm, _ = write_stream_to_memmap(dataset.X_train, prefetch=memmap_prefetch_mode)
				X_test_mm, _ = write_stream_to_memmap(dataset.X_test, prefetch=memmap_prefetch_mode)
				# Protocol v2: materialize X_val too (val is ~10% — small) so
				# the in-memory cache path below can upload it for
				# val-calibrated thresholds.
				X_val_mm = None
				if getattr(dataset, "X_val", None) is not None and isinstance(dataset.X_val, StreamingEncoded):
					X_val_mm, _ = write_stream_to_memmap(dataset.X_val, prefetch=memmap_prefetch_mode)
				# Rebind via a shallow dataset proxy; original streaming
				# instance retains its own factories for inspection.
				dataset = _ReplacedXDataset(dataset, X_train=X_train_mm, X_test=X_test_mm, X_val=X_val_mm)
				_streaming_input = False
			else:
				print(f"[IDSEvaluator] streaming dataset ~{total_gb:.2f} GB packed ≥ "
				      f"{threshold_gb:.1f} GB threshold; using streaming K-fold "
				      f"(re-stream per fold).")
		self._streaming_mode = _streaming_input

		# Phase F10: warm the OS page cache when X_train is a MemmapEncoded
		# coming from the Phase 4 path (encoded_storage="memmap"). This
		# amortizes cold-cache faults — the first K-fold partition transition
		# on V2 took 1h26m on 46M, suspected to be mostly page faults.
		# Touching pages once up front turns that into a few seconds.
		if reuse_cache is None and not self._streaming_mode and isinstance(dataset.X_train, MemmapEncoded) and memmap_prefetch_mode != "none":
			import time as _time
			for name, X in (("X_train", dataset.X_train), ("X_test", dataset.X_test)):
				if isinstance(X, MemmapEncoded):
					t0 = _time.time()
					X.prefetch(memmap_prefetch_mode)
					elapsed = _time.time() - t0
					gb = (X.n_rows * X.bytes_per_row) / (1024 ** 3)
					print(f"[IDSEvaluator] prefetch({memmap_prefetch_mode}) {name} "
					      f"~{gb:.2f} GB in {elapsed:.2f}s")
		self._single_cluster = single_cluster
		# Dataset-size hint for concurrency scaling (grid search drops its
		# eval parallelism above 10M rows — see BaseEvaluator.train_rows_hint).
		self.train_rows_hint = int(dataset.X_train.n_rows)
		self._num_classes = num_classes
		self._total_features = total_features
		self._num_parts = num_parts
		self._num_negatives = num_negatives
		self._neuron_sample_rate = neuron_sample_rate
		self._empty_value = empty_value
		self._memory_mode = memory_mode
		self._run_seed = run_seed
		self._normal_class = 1 if flip_labels else 0

		if reuse_cache is not None:
			# M2: share an existing Rust IDSCacheWrapper (built by the optimizer
			# evaluator) instead of building a second ~19 GB duplicate. Safe here
			# because the two evaluators wrap the IDENTICAL full_dataset (same 80%
			# train, same 20% val) and the worker sets the SAME mutable cache state
			# on both (fitness_weights, normal_class). This evaluator only ever uses
			# the full-train / full-eval methods (evaluate_genomes_full_hybrid,
			# score_train_examples, predict_examples, evaluate_at_thresholds), which
			# read full_train/full_eval and ignore the subset partitioning — so the
			# donor's num_parts (5 vs this evaluator's 1) is irrelevant.
			self._streaming_mode = False
			self._cache = reuse_cache
		elif self._streaming_mode:
			# Streaming mode: stash the streams, build IDSCacheWrapper lazily
			# (or not at all — evaluate_batch_full goes direct to IDSGenomeStreamer).
			# IDSEvaluator methods that depend on cached in-RAM data (subset
			# rotation, K-fold-via-Rust, neighbor search) raise NotImplementedError
			# in streaming mode; only evaluate_batch_full is supported in v1.
			if undersample_majority:
				raise NotImplementedError(
					"streaming mode does not support undersample_majority — "
					"would require per-class row-rejection mask computed from "
					"the materialized labels + filtered streaming. Tracked as "
					"future work."
				)
			# Phase F9: balance_classes IS supported in streaming mode via
			# materialized-labels pre-pass. Class weights are computed from
			# the already-in-RAM y_train labels (no extra stream pass) and
			# passed to IDSGenomeStreamer, which forwards them to
			# train_genome_in_slot's class_weights argument.
			if balance_classes:
				self._streaming_class_weights = _compute_class_weights(
					np.asarray(self._y_train, dtype=np.int64),
					num_classes,
					class_weight_multiplier,
				)
			else:
				self._streaming_class_weights = None
			self._train_stream = dataset.X_train
			self._eval_stream = dataset.X_test
			self._cache = None  # never built — direct genome-streamer path
			# Protocol v2: stash the val source for the streaming threshold
			# sweep. A StreamingEncoded X_val is materialized LAZILY on the
			# first evaluate_at_thresholds call (val is ~10% of the dataset —
			# small enough to memmap; only TRAIN needs true streaming), so
			# optimizer evaluators that never sweep thresholds pay nothing.
			self._val_source = getattr(dataset, "X_val", None) if self._y_val is not None else None
			self._val_materialized = None
		else:
			# Phase 2: hand the Rust accelerator bit-packed bytes directly.
			# `as_packed_uint8()` is a zero-copy view when the InMemoryEncoded
			# wraps a uint8 (packed) buffer — which is now the encoder's native
			# output. For a 46M × 96-bit dataset, this transfers ~552 MB of
			# packed bytes instead of the ~4.4 GB Vec<bool> form (and 8x
			# smaller than the historical Python-list path).
			train_features_np = np.ascontiguousarray(
				dataset.X_train.as_packed_uint8().ravel()
			)
			eval_features_np = np.ascontiguousarray(
				dataset.X_test.as_packed_uint8().ravel()
			)
			train_labels = [int(y) for y in y_train]
			eval_labels = [int(y) for y in y_test]

			# Protocol v2: upload the val partition (3-way splits) so threshold
			# calibrations can fit on val scores. Same materialization path as
			# X_test. StreamingEncoded X_val is not plumbed in this pass (46M
			# follow-up) — skip it so `_3way` streaming flows stay legacy.
			val_features_np = None
			val_labels = None
			X_val = getattr(dataset, "X_val", None)
			if X_val is not None and self._y_val is not None and not isinstance(X_val, StreamingEncoded):
				val_features_np = np.ascontiguousarray(
					X_val.as_packed_uint8().ravel()
				)
				val_labels = list(self._y_val)

			self._cache = ram_accelerator.IDSCacheWrapper.new_from_numpy(
				train_features=train_features_np,
				train_labels=train_labels,
				eval_features=eval_features_np,
				eval_labels=eval_labels,
				num_classes=num_classes,
				total_features=total_features,
				num_parts=num_parts,
				num_negatives=num_negatives,
				seed=self._seed,
				balance_classes=balance_classes,
				single_cluster=single_cluster,
				undersample_majority=undersample_majority,
				class_weight_multiplier=class_weight_multiplier,
				val_features=val_features_np,
				val_labels=val_labels,
			)

			# When flip_labels is active, original benign is class 1 after flipping.
			# Tell Rust to compute FPR on class 1 so it always measures false alarms
			# on the original benign traffic.
			if flip_labels:
				self._cache.set_normal_class(1)

		# Thread memory_mode + run_seed to Rust. Before 14/07/2026 the IDS cache
		# hardcoded EvalSettings::default() → QUAD_WEIGHTED, so the flow's
		# memory_mode param was silently dropped (TERNARY/BINARY/QSR/PLN all
		# trained+scored as QUAD). Idempotent on a reused cache; the streaming
		# lazy-build path sets it when it constructs the cache. run_seed feeds
		# the QSR/PLN stochastic coin (ignored by deterministic modes); it is
		# separate from `seed` so it never perturbs the K-fold permutation.
		if self._cache is not None:
			self._cache.set_memory_mode(memory_mode, run_seed)

		# Store fitness weights for potential threshold optimization
		self._fitness_weights = None

		self._train_call_count = 0
		self._k_folds = k_folds
		self._kfold_per_gen = min(kfold_per_gen, k_folds) if k_folds > 1 else 1
		self._kfold_rotation_offset = 0  # tracks which folds to use next
		if k_folds > 1 and k_folds != num_parts:
			raise ValueError(
				f"k_folds={k_folds} must equal num_parts={num_parts} "
				f"(each part becomes one fold)"
			)

		# Phase 5 F-prep: pre-shuffled K-fold permutation. Stored alongside
		# the existing Rust-side stratified partitioning — not yet used to
		# drive eval (the Rust cache still does its own stratified split).
		# Option F (streaming, post-paper) will switch the eval path to
		# consume folds as contiguous slabs of `_perm` order, so that
		# memmap or stream-source readers see sequential access instead of
		# scattered random indices.
		n_train = len(y_train)
		fold_seed = (seed if seed is not None else 42) + 7777
		self._kfold_perm = np.random.RandomState(fold_seed).permutation(n_train)
		self._kfold_n_train = n_train
		self._kfold_fold_size = n_train // max(k_folds, 1)

	def get_fold_indices(self, fold_idx: int) -> "tuple[np.ndarray, np.ndarray]":
		"""Return (train_idx, val_idx) for fold `fold_idx` as np.ndarrays.

		Uses a deterministic global permutation (seeded by the evaluator's
		`seed + 7777`) so that fold `i` is always the contiguous slab
		`_perm[i*fold_size:(i+1)*fold_size]`. The rest of `_perm` is the
		training split. Statistically equivalent to a fresh random K-fold
		(every example lands in exactly one validation fold across the K
		runs), but with contiguous-in-permutation-order access patterns —
		the load-bearing property for Option F's streaming/memmap path.

		Args:
		    fold_idx: 0 ≤ fold_idx < k_folds.

		Returns:
		    (train_idx, val_idx) — int arrays indexing into the original
		    X_train rows.
		"""
		if fold_idx < 0 or fold_idx >= self._k_folds:
			raise ValueError(f"fold_idx must be in [0, {self._k_folds}), got {fold_idx}")
		start = fold_idx * self._kfold_fold_size
		# Last fold absorbs any remainder (n_train % k_folds rows)
		stop = self._kfold_n_train if fold_idx == self._k_folds - 1 else start + self._kfold_fold_size
		val_idx = self._kfold_perm[start:stop]
		train_idx = np.concatenate([self._kfold_perm[:start], self._kfold_perm[stop:]])
		return train_idx, val_idx

	def next_train_idx(self) -> int:
		self._train_call_count += 1
		return self._cache.next_train_idx()

	def next_eval_idx(self) -> int:
		return 0  # IDS always uses full eval (no eval rotation)

	def _flatten_genomes(self, genomes: list[ClusterGenome]):
		"""Flatten genome arrays for Rust (canonical marshaller in wnn.accel)."""
		from wnn.accel import flatten_genomes
		return flatten_genomes(genomes)

	def evaluate_batch(
		self,
		genomes: list[ClusterGenome],
		train_subset_idx: Optional[int] = None,
		eval_subset_idx: Optional[int] = None,
		logger: Optional[Callable[[str], None]] = None,
		generation: Optional[int] = None,
		total_generations: Optional[int] = None,
		**kwargs,
	) -> list[Metrics]:
		if self._streaming_mode:
			# Phase F7 streaming evaluation. K-fold goes through the
			# rotated-fold streaming path; otherwise standard train/test.
			if self._k_folds > 1:
				if self._kfold_per_gen > 1:
					fold_indices = []
					for i in range(self._kfold_per_gen):
						fold_indices.append((self._kfold_rotation_offset + i) % self._k_folds)
					self._kfold_rotation_offset = (
						self._kfold_rotation_offset + self._kfold_per_gen
					) % self._k_folds
				else:
					fold_indices = [train_subset_idx if train_subset_idx is not None else 0]
				return self._evaluate_batch_streaming_kfold(genomes, fold_indices)
			return self._evaluate_batch_streaming(genomes)

		if train_subset_idx is None:
			train_subset_idx = self.next_train_idx()

		# If _adapt_config is set (by AdaptationStrategy for dedicated *genesis phases),
		# delegate to adaptive path — train, adapt architecture, retrain, eval.
		# Genomes are updated in-place with adapted architecture.
		if self._adapt_config is not None:
			config = self._adapt_config
			adaptive_results = self.evaluate_batch_adaptive(
				genomes,
				train_subset_idx=train_subset_idx,
				generation=self._generation,
				total_generations=config.total_generations,
				synaptogenesis=config.synaptogenesis_enabled,
				neurogenesis=config.neurogenesis_enabled,
				axonogenesis=config.axonogenesis_enabled,
				min_bits=config.min_bits,
				max_bits=config.max_bits,
				warmup_generations=config.warmup_generations,
				stats_sample_size=config.stats_sample_size,
				passes_per_eval=config.passes_per_eval,
			)
			# Update genomes in-place (like BitwiseEvaluator does)
			results = []
			for genome, (eval_result, adapted_genome) in zip(genomes, adaptive_results):
				genome.bits_per_neuron = adapted_genome.bits_per_neuron
				genome.neurons_per_cluster = adapted_genome.neurons_per_cluster
				genome.connections = adapted_genome.connections
				genome.metrics = adapted_genome.metrics
				results.append(eval_result)
			return results

		bits_flat, neurons_flat, connections_flat = self._flatten_genomes(genomes)

		if self._k_folds > 1 and self._kfold_per_gen > 1:
			# Multi-fold per generation: evaluate on N folds, average results
			# Pick which folds to use this generation (rotating window)
			fold_indices = []
			for i in range(self._kfold_per_gen):
				fold_indices.append((self._kfold_rotation_offset + i) % self._k_folds)
			self._kfold_rotation_offset = (self._kfold_rotation_offset + self._kfold_per_gen) % self._k_folds

			# Evaluate on each fold
			n_genomes = len(genomes)
			accum_ce = [0.0] * n_genomes
			accum_acc = [0.0] * n_genomes
			accum_f1 = [0.0] * n_genomes
			accum_fpr = [0.0] * n_genomes
			accum_threshold = [0.0] * n_genomes

			# Accumulate eval_time_ms across folds (sum, then represents
			# total train+eval ms for this genome over all evaluated folds).
			accum_ms = [0] * n_genomes
			for fold_idx in fold_indices:
				fold_results = self._cache.evaluate_genomes_kfold_hybrid(
					bits_flat, neurons_flat, connections_flat,
					n_genomes, fold_idx,
					self._empty_value, self._neuron_sample_rate, 0,
				)
				for g_idx, (ce, acc, f1, fpr, threshold, ms) in enumerate(fold_results):
					accum_ce[g_idx] += ce
					accum_acc[g_idx] += acc
					accum_f1[g_idx] += f1
					accum_fpr[g_idx] += fpr
					accum_threshold[g_idx] += threshold
					accum_ms[g_idx] += ms

			# Average metrics across folds; eval_time_ms is summed (total
			# time spent on this genome across all folds).
			n_folds = len(fold_indices)
			raw_results = [
				(accum_ce[i] / n_folds, accum_acc[i] / n_folds,
				 accum_f1[i] / n_folds, accum_fpr[i] / n_folds,
				 accum_threshold[i] / n_folds,
				 accum_ms[i])
				for i in range(n_genomes)
			]

		elif self._k_folds > 1:
			# Single fold per generation (original k-fold behavior)
			raw_results = self._cache.evaluate_genomes_kfold_hybrid(
				bits_flat, neurons_flat, connections_flat,
				len(genomes), train_subset_idx,
				self._empty_value, self._neuron_sample_rate, 0,
			)
		else:
			# No k-fold: train on one subset, eval on separate eval set
			raw_results = self._cache.evaluate_genomes_hybrid(
				bits_flat, neurons_flat, connections_flat,
				len(genomes), train_subset_idx,
				self._empty_value, self._neuron_sample_rate, 0,
			)

		results = []
		for genome, (ce, acc, f1, fpr, threshold, ms) in zip(genomes, raw_results):
			genome.threshold = threshold
			genome.metrics = Metrics(ce=ce, acc=acc, f1=f1, fpr=fpr, threshold=threshold, eval_time_ms=int(ms))
			results.append(genome.metrics)
		return results

	def set_fitness_weights(self, w_ce: float, w_f1: float, w_fpr: float, w_acc: float):
		"""Set fitness weights for threshold optimization.
		When set, threshold sweep maximizes fitness instead of F1."""
		self._fitness_weights = (w_ce, w_f1, w_fpr, w_acc)
		self._cache.set_fitness_weights(w_ce, w_f1, w_fpr, w_acc)

	def score_examples(self, genome) -> list[float]:
		"""Get raw per-example scores for a single genome (no thresholding).

		Trains on full training data, returns raw score for each eval example.
		Used for Platt scaling calibration.
		"""
		bits_flat, neurons_flat, connections_flat = self._flatten_genomes([genome])
		return self._cache.score_examples(
			bits_flat, neurons_flat, connections_flat,
			self._empty_value, self._neuron_sample_rate, 0,
		)

	def score_train_examples(self, genome) -> list[float]:
		"""Get raw per-example scores for training examples (no thresholding).

		Trains on full training data, returns raw score for each TRAINING example.
		Used for fitting calibration (platt/beta/empirical) on training data
		so the validation set remains untouched.
		"""
		bits_flat, neurons_flat, connections_flat = self._flatten_genomes([genome])
		return self._cache.score_train_examples(
			bits_flat, neurons_flat, connections_flat,
			self._empty_value, self._neuron_sample_rate, 0,
		)

	def evaluate_at_thresholds(
		self,
		genome,
		thresholds: list[float],
	) -> tuple[list[float], list[float], Optional[list[float]], list[Metrics]]:
		"""Train ONCE, score eval+train (+val), evaluate metrics at multiple thresholds.

		Returns:
			eval_scores: raw per-example scores on the report (test) set
			train_scores: raw per-example scores on training set
			val_scores: raw per-example scores on the validation partition
				(Protocol v2, 3-way splits); None for legacy 2-way datasets
			metrics: list of Metrics (ce, acc, f1, fpr, threshold) — one per
				threshold in `thresholds`. A threshold of -1.0 is the oracle
				sentinel: sweeps eval scores for the optimal F1 threshold.

		Replaces 7×evaluate_batch_full + score_examples + score_train_examples
		(9 trainings) with a single training pass.

		Streaming mode (Protocol v2): supported via IDSGenomeStreamer — train
		once, then multi-set scoring passes over eval/train/val streams.

		Multiclass (classification="multi", K clusters): routes to
		`evaluate_multiclass_at_thresholds` and returns ITS structure (a dict
		of decode modes — argmax / benign-margin — not the binary 4-tuple);
		`thresholds` is ignored (taus are calibrated Rust-side).
		"""
		if self._classification == "multi" and not self._single_cluster:
			return self.evaluate_multiclass_at_thresholds(genome)
		if self._streaming_mode:
			return self._evaluate_at_thresholds_streaming(genome, thresholds)

		bits_flat, neurons_flat, connections_flat = self._flatten_genomes([genome])
		eval_scores, train_scores, val_scores, raw_metrics = self._cache.evaluate_at_thresholds(
			bits_flat, neurons_flat, connections_flat,
			thresholds,
			self._empty_value, self._neuron_sample_rate, 0,
		)
		metrics = [
			Metrics(ce=ce, acc=acc, f1=f1, fpr=fpr, threshold=t)
			for (ce, acc, f1, fpr, t) in raw_metrics
		]
		return eval_scores, train_scores, val_scores, metrics

	def evaluate_multiclass_at_thresholds(self, genome) -> dict:
		"""Train ONCE, score eval/train(/val) as K-vectors, return per-mode
		multiclass metrics computed on the EVAL set.

		Decode modes (docs/MULTICLASS_DESIGN.md §3): argmax (no tau) and
		benign-margin with tau in {0.0 fixed, train-calibrated (macro-F1
		sweep on train margins), val-calibrated (sweep on val margins —
		Protocol v2, only when a val partition exists)}.

		Returns:
			{'num_classes': K, 'modes': {mode_name: {
				'ce', 'acc', 'macro_f1', 'weighted_f1', 'benign_fpr',
				'f1', 'fpr',    # aliases (= macro_f1 / benign_fpr) so shared
				                # consumers (leaderboard submit, 5-table
				                # reports) keep working unchanged
				'tau',          # margin modes only (absent for argmax)
				'confusion',    # K×K nested list, row = true class
				'per_class',    # {class_name: {precision, recall, f1, support}}
			}}}
		"""
		import math
		if self._streaming_mode:
			num_classes, mode_results = self._evaluate_multiclass_streaming(genome)
		else:
			bits_flat, neurons_flat, connections_flat = self._flatten_genomes([genome])
			num_classes, mode_results = self._cache.evaluate_multiclass_at_thresholds(
				bits_flat, neurons_flat, connections_flat,
				self._empty_value, self._neuron_sample_rate, 0,
			)
		if self._class_names and len(self._class_names) == num_classes:
			names = list(self._class_names)
		else:
			names = [str(c) for c in range(num_classes)]
		modes = {}
		for (mode, tau, ce, acc, macro_f1, weighted_f1, benign_fpr,
				confusion, precision, recall, f1s, support) in mode_results:
			entry = {
				'ce': ce, 'acc': acc,
				'macro_f1': macro_f1, 'weighted_f1': weighted_f1,
				'benign_fpr': benign_fpr,
				'f1': macro_f1, 'fpr': benign_fpr,
				'confusion': [
					[int(v) for v in confusion[r * num_classes:(r + 1) * num_classes]]
					for r in range(num_classes)
				],
				'per_class': {
					names[c]: {
						'precision': precision[c], 'recall': recall[c],
						'f1': f1s[c], 'support': int(support[c]),
					}
					for c in range(num_classes)
				},
			}
			if not math.isnan(tau):
				entry['tau'] = tau
			modes[mode] = entry
		return {'num_classes': num_classes, 'modes': modes}

	def evaluate_batch_full(
		self,
		genomes: list[ClusterGenome],
		logger: Optional[Callable[[str], None]] = None,
		override_threshold: Optional[float] = None,
	) -> list[Metrics]:
		if self._streaming_mode:
			return self._evaluate_batch_streaming(genomes, override_threshold=override_threshold)

		bits_flat, neurons_flat, connections_flat = self._flatten_genomes(genomes)

		raw_results = self._cache.evaluate_genomes_full_hybrid(
			bits_flat,
			neurons_flat,
			connections_flat,
			len(genomes),
			self._empty_value,
			self._neuron_sample_rate,
			0,  # rng_seed
			override_threshold=override_threshold,
		)

		results = []
		for genome, (ce, acc, f1, fpr, threshold, ms) in zip(genomes, raw_results):
			genome.threshold = threshold
			genome.metrics = Metrics(ce=ce, acc=acc, f1=f1, fpr=fpr, threshold=threshold, eval_time_ms=int(ms))
			results.append(genome.metrics)
		return results

	def _evaluate_batch_streaming(
		self,
		genomes: list[ClusterGenome],
		override_threshold: Optional[float] = None,
	) -> list[Metrics]:
		"""Per-genome streaming evaluation on the standard train/test split (no K-fold).

		Each genome gets its own IDSGenomeStreamer instance:
		  1. Stream train data → train_chunk per chunk
		  2. seal_for_scoring
		  3. Stream eval data → score_chunk per chunk
		  4. finalize_metrics → (ce, acc, f1, fpr, threshold)
		"""
		if override_threshold is not None:
			raise NotImplementedError(
				"override_threshold not supported in streaming mode yet"
			)
		return [
			self._streaming_train_score_genome(
				g, self._train_stream, self._eval_stream,
				train_labels=self._stream_label_override(self._y_train),
				eval_labels=self._stream_label_override(self._y_test),
			)
			for g in genomes
		]

	def _stream_label_override(self, y_list):
		"""Materialized label array to use in place of factory-bundled chunk
		labels, or None to keep the chunk labels.

		Multiclass streaming: chunks carry BINARY labels only, so the K-class
		targets come from the materialized `y_*_multi`-derived lists — valid
		because `_materialize_labels` fills them in stream order (same factory,
		same iteration). Binary keeps the chunk labels (path untouched).
		"""
		return y_list if self._classification == "multi" else None

	@staticmethod
	def _chunk_labels(labels_chunk, label_override, offset: int, n: int):
		"""Labels for one streamed chunk: the global-offset slice of the
		override array when set, else the factory-bundled chunk labels."""
		if label_override is not None:
			return np.asarray(label_override[offset:offset + n], dtype=np.int64)
		return labels_chunk

	def _streaming_train_pass(self, streamer, train_stream, train_filter=None, train_labels=None):
		"""Stream the train source through `streamer.train_chunk` (+ optional
		row filter + optional label override), then seal for scoring."""
		global_offset = 0
		for packed_chunk, labels_chunk in train_stream.iter_chunks():
			n = packed_chunk.shape[0]
			chunk_y = self._chunk_labels(labels_chunk, train_labels, global_offset, n)
			if train_filter is not None:
				row_indices = np.arange(global_offset, global_offset + n)
				mask = train_filter(row_indices)
				if mask.sum() > 0:
					filtered_packed = np.ascontiguousarray(packed_chunk[mask].ravel())
					filtered_labels = chunk_y[mask].tolist()
					streamer.train_chunk(filtered_packed, filtered_labels, self._total_features)
			else:
				streamer.train_chunk(
					np.ascontiguousarray(packed_chunk.ravel()),
					chunk_y.tolist() if hasattr(chunk_y, "tolist") else list(chunk_y),
					self._total_features,
				)
			global_offset += n
		streamer.seal_for_scoring()

	def _streaming_train_score_genome(
		self,
		genome: ClusterGenome,
		train_stream,
		eval_stream,
		train_filter=None,
		eval_filter=None,
		train_labels=None,
		eval_labels=None,
	) -> Metrics:
		"""Single-genome streaming evaluation primitive (Phase F + F7).

		`train_filter` / `eval_filter`, when provided, are callables
		`(row_indices_in_chunk: np.ndarray) → bool mask`. The mask selects
		which rows of each streamed chunk go through train_chunk/score_chunk.
		Used by streaming K-fold to gate rows by `_kfold_perm`-derived
		fold-membership: rows assigned to the current val fold are filtered
		OUT of training and INTO scoring, and vice versa.

		When filters are None, every row passes through (the no-K-fold path).

		`train_labels` / `eval_labels`, when provided, override the
		factory-bundled chunk labels via global-offset slicing (see
		`_stream_label_override` — the multiclass path). They must pair with
		their stream: K-fold passes the TRAIN label array for both, since
		both passes read the train stream.
		"""
		streamer = self._build_streamer(genome)
		self._streaming_train_pass(streamer, train_stream, train_filter, train_labels)

		# Score pass — stream + optionally filter
		global_offset = 0
		for packed_chunk, labels_chunk in eval_stream.iter_chunks():
			n = packed_chunk.shape[0]
			chunk_y = self._chunk_labels(labels_chunk, eval_labels, global_offset, n)
			if eval_filter is not None:
				row_indices = np.arange(global_offset, global_offset + n)
				mask = eval_filter(row_indices)
				if mask.sum() > 0:
					filtered_packed = np.ascontiguousarray(packed_chunk[mask].ravel())
					filtered_labels = chunk_y[mask].tolist()
					streamer.score_chunk(filtered_packed, filtered_labels, self._total_features)
			else:
				streamer.score_chunk(
					np.ascontiguousarray(packed_chunk.ravel()),
					chunk_y.tolist() if hasattr(chunk_y, "tolist") else list(chunk_y),
					self._total_features,
				)
			global_offset += n

		ce, acc, f1, fpr, threshold = streamer.finalize_metrics()
		genome.threshold = threshold
		genome.metrics = Metrics(ce=ce, acc=acc, f1=f1, fpr=fpr, threshold=threshold)
		return genome.metrics

	def _build_streamer(self, genome: ClusterGenome):
		"""Construct a fresh IDSGenomeStreamer wrapper for this genome."""
		import ram_accelerator

		return ram_accelerator.IDSGenomeStreamerWrapper(
			bits_flat=list(genome.bits_per_neuron),
			neurons_flat=list(genome.neurons_per_cluster),
			connections=list(genome.connections) if genome.connections is not None else [],
			num_classes=self._num_classes,
			num_negatives=self._num_negatives,
			single_cluster=self._single_cluster,
			total_features=self._total_features,
			empty_value=self._empty_value,
			neuron_sample_rate=self._neuron_sample_rate,
			rng_seed=self._seed,
			normal_class=self._normal_class,
			class_weights=getattr(self, "_streaming_class_weights", None),
		)

	def _streaming_score_stream(self, streamer, stream, labels=None) -> list[float]:
		"""Score every row of a (packed, labels) chunk stream; drain raw scores.

		`labels`, when provided, overrides the chunk labels by global-offset
		slicing (multiclass — see `_stream_label_override`). Multi-cluster
		streamers return the FLAT K-vector buffer (len = rows × K).
		"""
		global_offset = 0
		for packed_chunk, labels_chunk in stream.iter_chunks():
			n = packed_chunk.shape[0]
			chunk_y = self._chunk_labels(labels_chunk, labels, global_offset, n)
			streamer.score_chunk(
				np.ascontiguousarray(packed_chunk.ravel()),
				chunk_y.tolist() if hasattr(chunk_y, "tolist") else list(chunk_y),
				self._total_features,
			)
			global_offset += n
		return streamer.take_scores()

	def _streaming_score_encoded(
		self,
		streamer,
		encoded,
		labels: list[int],
		chunk_rows: int = 1_000_000,
	) -> list[float]:
		"""Score a materialized LazyEncodedArray (packed rows); drain raw scores."""
		packed = encoded.as_packed_uint8()
		n = packed.shape[0]
		for start in range(0, n, chunk_rows):
			stop = min(start + chunk_rows, n)
			streamer.score_chunk(
				np.ascontiguousarray(packed[start:stop].ravel()),
				labels[start:stop],
				self._total_features,
			)
		return streamer.take_scores()

	def _val_encoded_for_streaming(self):
		"""Materialized val features for the streaming threshold sweep (or None).

		A StreamingEncoded X_val is drained to memmap ONCE and cached — the
		val partition is ~10% of the dataset, small enough to materialize
		(Protocol v2 design: only TRAIN needs true streaming).
		"""
		if self._y_val is None or self._val_source is None:
			return None
		from wnn.ids.encoded_array import StreamingEncoded, write_stream_to_memmap
		if isinstance(self._val_source, StreamingEncoded):
			if self._val_materialized is None:
				self._val_materialized, _ = write_stream_to_memmap(self._val_source)
			return self._val_materialized
		return self._val_source

	def _evaluate_at_thresholds_streaming(
		self,
		genome,
		thresholds: list[float],
	) -> tuple[list[float], list[float], Optional[list[float]], list[Metrics]]:
		"""Streaming evaluate_at_thresholds (Protocol v2): train ONCE via
		IDSGenomeStreamer, then score eval + train (+ materialized val) sets
		from the sealed memory via multi-set take_scores passes.

		Same return contract as the cache path. The extra train-scoring
		stream pass is the Protocol v2 cost on streaming flows (the
		threshold sweep runs once per genome at validation, not per
		generation). The -1.0 oracle sentinel resolves on EVAL scores,
		matching evaluate_at_thresholds_ids_cached.
		"""
		import ram_accelerator

		streamer = self._build_streamer(genome)
		self._streaming_train_pass(streamer, self._train_stream)

		eval_scores = self._streaming_score_stream(streamer, self._eval_stream)
		train_scores = self._streaming_score_stream(streamer, self._train_stream)
		val_encoded = self._val_encoded_for_streaming()
		val_scores = None
		if val_encoded is not None:
			val_scores = self._streaming_score_encoded(streamer, val_encoded, self._y_val)

		metrics = []
		for t in thresholds:
			if t < 0.0:
				resolved, _f1, _fpr = ram_accelerator.find_optimal_threshold_f1_py(
					eval_scores, self._y_test,
				)
			else:
				resolved = float(t)
			ce, acc, f1, fpr = ram_accelerator.compute_binary_metrics_at_threshold_py(
				eval_scores, self._y_test, resolved, self._normal_class,
			)
			metrics.append(Metrics(ce=ce, acc=acc, f1=f1, fpr=fpr, threshold=resolved))
		return eval_scores, train_scores, val_scores, metrics

	def _evaluate_multiclass_streaming(self, genome) -> "tuple[int, list]":
		"""Streaming multiclass Protocol v2: train ONCE via IDSGenomeStreamer,
		score eval + train (+ materialized val) as flat K-vectors
		(`take_scores` drains rows × K), then compute the decode-mode
		metrics Rust-side. `multiclass_modes_from_scores` returns the exact
		`(num_classes, mode tuples)` contract of the in-memory
		`IDSCacheWrapper.evaluate_multiclass_at_thresholds`, so the caller's
		dict building is shared.

		Memory note: the flat train-score buffer is rows × K floats as a
		Python list (~11M for the 1.4M×K=8 subsample — fine; a 46M
		multiclass run would want a numpy passthrough here first).
		"""
		import ram_accelerator

		streamer = self._build_streamer(genome)
		self._streaming_train_pass(
			streamer, self._train_stream,
			train_labels=self._stream_label_override(self._y_train),
		)
		eval_flat = self._streaming_score_stream(
			streamer, self._eval_stream, labels=self._stream_label_override(self._y_test),
		)
		train_flat = self._streaming_score_stream(
			streamer, self._train_stream, labels=self._stream_label_override(self._y_train),
		)
		val_encoded = self._val_encoded_for_streaming()
		val_flat = None
		if val_encoded is not None:
			val_flat = self._streaming_score_encoded(streamer, val_encoded, self._y_val)
		return ram_accelerator.multiclass_modes_from_scores(
			eval_flat, self._y_test,
			train_flat, self._y_train,
			self._num_classes, self._normal_class,
			val_scores=val_flat,
			val_labels=self._y_val if val_flat is not None else None,
		)

	def _evaluate_batch_streaming_kfold(
		self,
		genomes: list[ClusterGenome],
		fold_indices: list[int],
	) -> list[Metrics]:
		"""Streaming K-fold (F7): per genome × per fold, run train+score with
		row-level filtering against `_kfold_perm`.

		For each fold k in `fold_indices`:
		  val_set_k = set(_kfold_perm[k * fold_size : (k+1) * fold_size])
		  Per genome: stream train_set (rows where perm[i] not in val_set_k),
		             stream val_set (rows where perm[i] in val_set_k).
		Results are averaged across folds per genome.

		Stream passes per generation: 2 × len(fold_indices) × len(genomes).
		Stratification is NOT enforced — the global permutation is uniform
		random (seed + 7777). For class-balanced K-fold, future work could
		stream-pre-pass to collect class counts; in v1 we rely on the
		dataset already being roughly balanced (or accept approximate
		stratification).
		"""
		n_genomes = len(genomes)
		accum_ce = [0.0] * n_genomes
		accum_acc = [0.0] * n_genomes
		accum_f1 = [0.0] * n_genomes
		accum_fpr = [0.0] * n_genomes
		accum_threshold = [0.0] * n_genomes
		accum_ms = [0] * n_genomes  # summed across folds (total eval cost)

		for fold_idx in fold_indices:
			train_idx_set, val_idx_set = self.get_fold_indices(fold_idx)
			# Build O(N) bool array for fast in-mask lookup (val rows). For
			# 1B rows this is 1 GB — acceptable since labels-materialization
			# already requires similar.
			val_mask_global = np.zeros(self._kfold_n_train, dtype=bool)
			val_mask_global[val_idx_set] = True

			def train_filter(row_indices, _mask=val_mask_global):
				return ~_mask[row_indices]

			def eval_filter(row_indices, _mask=val_mask_global):
				return _mask[row_indices]

			# For streaming K-fold, both train and val come from the SAME
			# train_stream (eval_stream is irrelevant — held-out test is
			# separate from K-fold val).
			for g_idx, genome in enumerate(genomes):
				m = self._streaming_train_score_genome(
					genome,
					self._train_stream,
					self._train_stream,  # val rows come from train stream filtered
					train_filter=train_filter,
					eval_filter=eval_filter,
					# Both passes read the TRAIN stream → train labels for both.
					train_labels=self._stream_label_override(self._y_train),
					eval_labels=self._stream_label_override(self._y_train),
				)
				accum_ce[g_idx] += m.ce
				accum_acc[g_idx] += m.acc
				accum_f1[g_idx] += m.f1
				accum_fpr[g_idx] += m.fpr
				accum_threshold[g_idx] += m.threshold
				accum_ms[g_idx] += int(m.eval_time_ms or 0)

		n_folds = len(fold_indices)
		results = []
		for g_idx, genome in enumerate(genomes):
			m = Metrics(
				ce=accum_ce[g_idx] / n_folds,
				acc=accum_acc[g_idx] / n_folds,
				f1=accum_f1[g_idx] / n_folds,
				fpr=accum_fpr[g_idx] / n_folds,
				threshold=accum_threshold[g_idx] / n_folds,
				eval_time_ms=accum_ms[g_idx],
			)
			genome.metrics = m
			genome.threshold = m.threshold
			results.append(m)
		return results

	def evaluate_batch_adaptive(
		self,
		genomes: list[ClusterGenome],
		train_subset_idx: Optional[int] = None,
		generation: int = 0,
		total_generations: int = 250,
		synaptogenesis: bool = True,
		neurogenesis: bool = False,
		axonogenesis: bool = False,
		min_bits: int = 4,
		max_bits: int = 24,
		warmup_generations: int = 10,
		stats_sample_size: int = 10000,
		passes_per_eval: int = 1,
		**kwargs,
	) -> list[tuple[EvalResult, ClusterGenome]]:
		"""Evaluate with training-time adaptation, returning adapted genomes.

		After initial training, computes per-neuron/cluster statistics and applies
		adaptation passes:
		  - synaptogenesis: prune/grow connections per neuron
		  - neurogenesis: add/remove neurons per cluster
		  - axonogenesis: rewire low-value connections to high-MI inputs

		If the genome was modified, it is retrained and the adapted architecture
		is returned alongside the scores.

		Returns list of (EvalResult, adapted_ClusterGenome) tuples.
		"""
		if train_subset_idx is None:
			train_subset_idx = self.next_train_idx()

		bits_flat, neurons_flat, connections_flat = self._flatten_genomes(genomes)

		raw_results = self._cache.evaluate_genomes_hybrid_adaptive(
			bits_flat, neurons_flat, connections_flat,
			len(genomes), train_subset_idx,
			self._empty_value, self._neuron_sample_rate, 0,
			synaptogenesis_enabled=synaptogenesis,
			neurogenesis_enabled=neurogenesis,
			axonogenesis_enabled=axonogenesis,
			min_bits=min_bits,
			max_bits=max_bits,
			warmup_generations=warmup_generations,
			total_generations=total_generations,
			generation=generation,
			total_input_bits=self._total_input_bits,
			stats_sample_size=stats_sample_size,
			passes_per_eval=passes_per_eval,
			**kwargs,
		)

		results = []
		for ce, acc, f1, fpr, adapted_bits, adapted_neurons, adapted_conns, pruned, grown, added, removed, rewired in raw_results:
			eval_result = Metrics(ce=ce, acc=acc, f1=f1, fpr=fpr)
			adapted_genome = ClusterGenome(
				bits_per_neuron=list(adapted_bits),
				neurons_per_cluster=list(adapted_neurons),
				connections=list(adapted_conns) if adapted_conns else None,
			)
			adapted_genome.metrics = Metrics(ce=ce, acc=acc, f1=f1, fpr=fpr)
			results.append((eval_result, adapted_genome))
		return results

	def search_neighbors(
		self,
		genome: ClusterGenome,
		target_count: int,
		max_attempts: int,
		accuracy_threshold: float,
		min_bits: int,
		max_bits: int,
		min_neurons: int,
		max_neurons: int,
		bits_mutation_rate: float = 0.1,
		neurons_mutation_rate: float = 0.05,
		train_subset_idx: Optional[int] = None,
		eval_subset_idx: Optional[int] = None,
		seed: Optional[int] = None,
		log_path: Optional[str] = None,
		logger: Optional[Callable[[str], None]] = None,
		generation: Optional[int] = None,
		total_generations: Optional[int] = None,
		return_best_n: bool = True,
		mutable_clusters: Optional[list[int]] = None,
		phase_type: int = 0,
	) -> list[ClusterGenome]:
		if train_subset_idx is None:
			train_subset_idx = self.next_train_idx()
		if seed is None:
			seed = int(time.time() * 1000) % (2**32)

		effective_log_path = log_path if log_path is not None else self._log_path

		results = self._cache.search_neighbors(
			base_bits=genome.bits_per_neuron,
			base_neurons=genome.neurons_per_cluster,
			base_connections=genome.connections if genome.connections else [],
			target_count=target_count,
			max_attempts=max_attempts,
			accuracy_threshold=accuracy_threshold,
			min_bits=min_bits,
			max_bits=max_bits,
			min_neurons=min_neurons,
			max_neurons=max_neurons,
			bits_mutation_rate=bits_mutation_rate,
			neurons_mutation_rate=neurons_mutation_rate,
			train_subset_idx=train_subset_idx,
			empty_value=self._empty_value,
			seed=seed,
			log_path=effective_log_path,
			generation=generation,
			total_generations=total_generations,
			return_best_n=return_best_n,
			mutable_clusters=mutable_clusters,
			phase_type=phase_type,
		)

		genomes = []
		for bits, neurons, connections, ce, acc, f1, fpr in results:
			g = ClusterGenome(
				bits_per_neuron=list(bits),
				neurons_per_cluster=list(neurons),
				connections=list(connections) if connections else None,
			)
			g.metrics = Metrics(ce=ce, acc=acc, f1=f1, fpr=fpr)
			genomes.append(g)
		return genomes

	def search_offspring(
		self,
		population: list[tuple[ClusterGenome, float]],
		target_count: int,
		max_attempts: int,
		accuracy_threshold: float,
		min_bits: int,
		max_bits: int,
		min_neurons: int,
		max_neurons: int,
		bits_mutation_rate: float = 0.1,
		neurons_mutation_rate: float = 0.1,
		crossover_rate: float = 0.7,
		tournament_size: int = 3,
		train_subset_idx: Optional[int] = None,
		eval_subset_idx: Optional[int] = None,
		seed: Optional[int] = None,
		log_path: Optional[str] = None,
		logger: Optional[Callable[[str], None]] = None,
		generation: Optional[int] = None,
		total_generations: Optional[int] = None,
		return_best_n: bool = True,
		mutable_clusters: Optional[list[int]] = None,
		phase_type: int = 0,
		fitness_scores: Optional[list[float]] = None,
		cluster_crossover_ratio: float = 0.0,
		pool_shuffle_ratio: float = 0.0,
		assortative_mating_ratio: float = 0.0,
	) -> OffspringSearchResult:
		if not population:
			return OffspringSearchResult(genomes=[], evaluated=0, viable=0)

		if train_subset_idx is None:
			train_subset_idx = self.next_train_idx()
		if seed is None:
			seed = int(time.time() * 1000) % (2**32)

		rust_population = []
		for genome, fitness in population:
			rust_population.append((
				genome.bits_per_neuron,
				genome.neurons_per_cluster,
				genome.connections if genome.connections else [],
				fitness,
			))

		effective_log_path = log_path if log_path is not None else self._log_path

		candidates, evaluated, viable = self._cache.search_offspring(
			population=rust_population,
			target_count=target_count,
			max_attempts=max_attempts,
			accuracy_threshold=accuracy_threshold,
			min_bits=min_bits,
			max_bits=max_bits,
			min_neurons=min_neurons,
			max_neurons=max_neurons,
			bits_mutation_rate=bits_mutation_rate,
			neurons_mutation_rate=neurons_mutation_rate,
			crossover_rate=crossover_rate,
			tournament_size=tournament_size,
			train_subset_idx=train_subset_idx,
			empty_value=self._empty_value,
			# Match the elite eval path's neuron_sample_rate (production: 0.25);
			# the prior hardcoded 1.0 in lib.rs scored offspring on a different
			# sampling regime, making offspring metrics non-comparable to elite
			# metrics. Fixed alongside the kfold_hybrid eval-path correction.
			neuron_sample_rate=self._neuron_sample_rate,
			seed=seed,
			# Pass the harmonic-rank fitness Python already computed so Rust's
			# tournament_select uses it for parent selection instead of CE.
			# Before this fix (lib.rs commit 18662b5f shipping date 2026-03-09)
			# every IDS GA run silently fell back to CE-based parent selection,
			# defeating the weight-set design (weights only affected elite
			# preservation + reporting, not GA exploration). The parameter is
			# Optional[list[float]] — if None, the prior CE-based behavior is
			# preserved (for backward compat with non-fitness-aware callers).
			fitness_scores=fitness_scores,
			log_path=effective_log_path,
			generation=generation,
			total_generations=total_generations,
			return_best_n=return_best_n,
			mutable_clusters=mutable_clusters,
			phase_type=phase_type,
			cluster_crossover_ratio=cluster_crossover_ratio,
			pool_shuffle_ratio=pool_shuffle_ratio,
			assortative_mating_ratio=assortative_mating_ratio,
		)

		genomes = []
		for bits, neurons, connections, ce, acc, f1, fpr in candidates:
			g = ClusterGenome(
				bits_per_neuron=list(bits),
				neurons_per_cluster=list(neurons),
				connections=list(connections) if connections else None,
			)
			g.metrics = Metrics(ce=ce, acc=acc, f1=f1, fpr=fpr)
			genomes.append(g)

		return OffspringSearchResult(genomes=genomes, evaluated=evaluated, viable=viable)

	def predict(
		self,
		genome: ClusterGenome,
		rng_seed: int = 0,
	) -> list[int]:
		"""Train genome on full training data and return per-example test predictions.

		Returns list of predicted class indices for each eval example.
		Uses the Rust accelerator with GPU for both training and inference.
		"""
		bits_flat, neurons_flat, connections_flat = self._flatten_genomes([genome])
		return self._cache.predict_examples(
			bits_flat,
			neurons_flat,
			connections_flat,
			self._empty_value,
			self._neuron_sample_rate,
			rng_seed,
		)

	def reset(self, seed: Optional[int] = None) -> None:
		self._cache.reset(seed)
		self._train_call_count = 0

	@property
	def num_train_subsets(self) -> int:
		return self._cache.num_train_subsets()

	@property
	def num_classes(self) -> int:
		return self._num_classes

	@property
	def y_test(self) -> list[int]:
		return self._y_test

	@property
	def class_names(self) -> list[str] | None:
		return self._class_names

	def predict_classes(self, genome: ClusterGenome, rng_seed: int = 0) -> list[int]:
		"""Alias for predict() — returns per-example class predictions on eval set."""
		return self.predict(genome, rng_seed)

	def __repr__(self) -> str:
		kfold_str = f", k_folds={self._k_folds}" if self._k_folds > 1 else ""
		sc_str = ", single_cluster" if self._single_cluster else ""
		return (
			f"IDSEvaluator(classes={self._num_classes}, "
			f"features={self._total_input_bits}, "
			f"classification='{self._classification}', "
			f"parts={self._num_parts}{kfold_str}{sc_str})"
		)
