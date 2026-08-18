//! Multi-stage token cache — stage-agnostic RAM evaluation.
//!
//! Split out of lib.rs (D3, 11/06/2026).

use crate::*;

// =============================================================================
// Multi-Stage Token Cache — PyO3 wrapper for stage-agnostic RAM evaluation
// =============================================================================

#[pyclass]
pub(crate) struct MultiStageCacheWrapper
{
	inner: multistage::MultiStageTokenCache,
	sparse_threshold_override: Option<usize>,
	reweight_rounds: usize,
	reweight_max_boost: usize,
	live_progress: Arc<RwLock<Option<neighbor_search::LiveProgress>>>,
	experiment_id: i64,
	progress_generation: i32,
	progress_total_generations: i32,
	progress_phase: String,
}

#[pymethods]
impl MultiStageCacheWrapper
{
	#[new]
	#[pyo3(signature = (train_tokens, eval_tokens, vocab_size, context_size, k, num_parts, num_eval_parts, pad_token_id, sparse_threshold=None, stage_cluster_types=None, custom_cluster_of=None, stage_context_sizes=None, reweight_rounds=None, reweight_max_boost=None))]
	fn new(
		train_tokens: Vec<u32>,
		eval_tokens: Vec<u32>,
		vocab_size: usize,
		context_size: usize,
		k: usize,
		num_parts: usize,
		num_eval_parts: usize,
		pad_token_id: u32,
		sparse_threshold: Option<usize>,
		stage_cluster_types: Option<Vec<String>>,
		custom_cluster_of: Option<Vec<u16>>,
		stage_context_sizes: Option<Vec<usize>>,
		reweight_rounds: Option<usize>,
		reweight_max_boost: Option<usize>,
	) -> Self
	{
		let rw_rounds = reweight_rounds.unwrap_or(0);
		let rw_max_boost = reweight_max_boost.unwrap_or(4);
		let mut cache = multistage::MultiStageTokenCache::new(
			train_tokens,
			eval_tokens,
			vocab_size,
			context_size,
			k,
			num_parts,
			num_eval_parts,
			pad_token_id,
			stage_cluster_types,
			custom_cluster_of,
			stage_context_sizes,
		);
		cache.reweight_rounds = rw_rounds;
		cache.reweight_max_boost = rw_max_boost;
		Self {
			inner: cache,
			sparse_threshold_override: sparse_threshold,
			reweight_rounds: rw_rounds,
			reweight_max_boost: rw_max_boost,
			live_progress: Arc::new(RwLock::new(None)),
			experiment_id: 0,
			progress_generation: 0,
			progress_total_generations: 0,
			progress_phase: "evaluate_batch".into(),
		}
	}

	fn set_experiment_context(&mut self, experiment_id: i64)
	{
		self.experiment_id = experiment_id;
	}

	/// Set progress context (generation, total, phase) so Rust sub-batch
	/// updates report correct values to the observer thread.
	/// Pre-seeds the LiveProgress Arc so Rust only needs to update
	/// evaluated/target_count/elapsed_secs in place.
	fn set_progress_context(&mut self, generation: i32, total_generations: i32, phase: String)
	{
		self.progress_generation = generation;
		self.progress_total_generations = total_generations;
		self.progress_phase = phase.clone();
		// Pre-seed the Arc so Rust sub-batch updates preserve these fields
		if let Ok(mut guard) = self.live_progress.write()
		{
			*guard = Some(neighbor_search::LiveProgress {
				experiment_id: self.experiment_id,
				generation,
				total_generations,
				phase,
				evaluated: 0,
				target_count: 0,
				viable: None,
				best_ce: 0.0,
				best_acc: 0.0,
				elapsed_secs: 0.0,
				updated_at: neighbor_search::LiveProgress::now_unix(),
			});
		}
	}

	fn get_live_progress(&self, py: Python<'_>) -> PyResult<Option<pyo3::PyObject>>
	{
		let guard = self
			.live_progress
			.read()
			.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("lock: {e}")))?;
		match &*guard
		{
			None => Ok(None),
			Some(lp) =>
			{
				let dict = pyo3::types::PyDict::new(py);
				dict.set_item("experiment_id", lp.experiment_id)?;
				dict.set_item("generation", lp.generation)?;
				dict.set_item("total_generations", lp.total_generations)?;
				dict.set_item("phase", &lp.phase)?;
				dict.set_item("evaluated", lp.evaluated)?;
				dict.set_item("target_count", lp.target_count)?;
				match lp.viable
				{
					Some(v) => dict.set_item("viable", v)?,
					None => dict.set_item("viable", py.None())?,
				}
				dict.set_item("best_ce", lp.best_ce)?;
				dict.set_item("best_acc", lp.best_acc)?;
				dict.set_item("elapsed_secs", lp.elapsed_secs)?;
				Ok(Some(dict.into()))
			}
		}
	}

	// ── Bitwise evaluation (stage-agnostic) ─────────────────────────

	/// Evaluate bitwise genomes for any stage with subset rotation.
	#[allow(clippy::too_many_arguments)]
	fn evaluate_bitwise_genomes(
		&self,
		py: Python<'_>,
		stage: usize,
		bits_per_neuron_flat: Vec<usize>,
		neurons_per_cluster_flat: Vec<usize>,
		connections_flat: Vec<i64>,
		num_genomes: usize,
		train_subset_idx: usize,
		eval_subset_idx: usize,
		memory_mode: u8,
		neuron_sample_rate: f32,
		rng_seed: u64,
	) -> PyResult<Vec<(f64, f64, f64, f64)>>
	{
		let override_val = self.sparse_threshold_override;
		let lp_arc = self.live_progress.clone();
		let exp_id = self.experiment_id;
		py.allow_threads(|| {
			Ok(multistage::evaluate_bitwise_genomes(
				&self.inner,
				stage,
				&bits_per_neuron_flat,
				&neurons_per_cluster_flat,
				&connections_flat,
				num_genomes,
				train_subset_idx,
				eval_subset_idx,
				memory_mode,
				neuron_sample_rate,
				rng_seed,
				override_val,
				Some(&lp_arc),
				exp_id,
			))
		})
	}

	/// Evaluate bitwise genomes with full (non-rotated) data.
	#[allow(clippy::too_many_arguments)]
	fn evaluate_bitwise_genomes_full(
		&self,
		py: Python<'_>,
		stage: usize,
		bits_per_neuron_flat: Vec<usize>,
		neurons_per_cluster_flat: Vec<usize>,
		connections_flat: Vec<i64>,
		num_genomes: usize,
		memory_mode: u8,
		neuron_sample_rate: f32,
		rng_seed: u64,
	) -> PyResult<Vec<(f64, f64, f64, f64)>>
	{
		let override_val = self.sparse_threshold_override;
		py.allow_threads(|| {
			Ok(multistage::evaluate_bitwise_genomes_full(
				&self.inner,
				stage,
				&bits_per_neuron_flat,
				&neurons_per_cluster_flat,
				&connections_flat,
				num_genomes,
				memory_mode,
				neuron_sample_rate,
				rng_seed,
				override_val,
			))
		})
	}

	/// Evaluate bitwise genomes for one group (selector mode).
	#[allow(clippy::too_many_arguments)]
	fn evaluate_bitwise_selector_genomes(
		&self,
		py: Python<'_>,
		stage: usize,
		bits_per_neuron_flat: Vec<usize>,
		neurons_per_cluster_flat: Vec<usize>,
		connections_flat: Vec<i64>,
		num_genomes: usize,
		group_id: usize,
		memory_mode: u8,
		neuron_sample_rate: f32,
		rng_seed: u64,
	) -> PyResult<Vec<(f64, f64, f64, f64)>>
	{
		let override_val = self.sparse_threshold_override;
		py.allow_threads(|| {
			Ok(multistage::evaluate_bitwise_selector_genomes(
				&self.inner,
				stage,
				&bits_per_neuron_flat,
				&neurons_per_cluster_flat,
				&connections_flat,
				num_genomes,
				group_id,
				memory_mode,
				neuron_sample_rate,
				rng_seed,
				override_val,
			))
		})
	}

	/// Get the number of eval examples per selector group for a given stage.
	fn selector_eval_counts(&self, stage: usize) -> Vec<usize>
	{
		if stage < self.inner.bitwise_selector_eval.len()
		{
			self.inner.bitwise_selector_eval[stage]
				.iter()
				.map(|s| s.num_examples)
				.collect()
		}
		else
		{
			Vec::new()
		}
	}

	// ── Rotation ────────────────────────────────────────────────────

	fn next_train_idx(&self) -> usize
	{
		self.inner.next_train_idx()
	}

	fn next_eval_idx(&self) -> usize
	{
		self.inner.next_eval_idx()
	}

	fn reset(&self)
	{
		self.inner.reset();
	}

	// ── Clustering info (stage-agnostic) ────────────────────────────

	fn k(&self) -> usize
	{
		self.inner.k
	}
	fn vocab_size(&self) -> usize
	{
		self.inner.vocab_size
	}
	fn context_size(&self) -> usize
	{
		self.inner.max_context_size
	}
	fn stage_context_sizes(&self) -> Vec<usize>
	{
		self.inner.stage_context_sizes.clone()
	}
	fn max_cluster_size(&self) -> usize
	{
		self.inner.max_cluster_size
	}
	fn num_parts(&self) -> usize
	{
		self.inner.num_parts
	}
	fn num_eval_parts(&self) -> usize
	{
		self.inner.num_eval_parts
	}

	fn cluster_sizes(&self) -> Vec<usize>
	{
		self.inner.cluster_sizes.clone()
	}

	/// Get total input bits for a given stage.
	fn stage_input_bits(&self, stage: usize) -> usize
	{
		self.inner.stage_input_bits.get(stage).copied().unwrap_or(0)
	}

	/// Get output bits (target bit count) for a given bitwise stage.
	fn bitwise_output_bits(&self, stage: usize) -> usize
	{
		self
			.inner
			.bitwise_output_bits
			.get(stage)
			.copied()
			.unwrap_or(0)
	}

	// ── Combined CE computation ─────────────────────────────────────

	/// Compute combined multi-stage CE from per-stage genome params.
	///
	/// Takes flat concatenated arrays for all stages, plus stage_num_clusters
	/// to partition them.
	///
	/// Returns: (combined_ce, combined_accuracy, stage0_ce, stage1_ce)
	#[allow(clippy::too_many_arguments)]
	fn evaluate_combined_ce(
		&self,
		py: Python<'_>,
		all_bits_per_neuron: Vec<usize>,
		all_neurons_per_cluster: Vec<usize>,
		all_connections: Vec<i64>,
		stage_num_clusters: Vec<usize>,
		memory_mode: u8,
		neuron_sample_rate: f32,
		rng_seed: u64,
		sparse_threshold: usize,
		label_smoothing: f64,
		unigram_lambda: f64,
		bigram_lambda: f64,
	) -> PyResult<(f64, f64, f64, f64, f64, f64)>
	{
		py.allow_threads(|| {
			// Partition flat arrays by stage
			let num_stages = stage_num_clusters.len();
			let mut stage_bits: Vec<&[usize]> = Vec::with_capacity(num_stages);
			let mut stage_neurons: Vec<&[usize]> = Vec::with_capacity(num_stages);
			let mut stage_conns: Vec<&[i64]> = Vec::with_capacity(num_stages);

			let mut neuron_offset = 0usize;
			let mut cluster_offset = 0usize;
			let mut conn_offset = 0usize;

			for s in 0..num_stages
			{
				let n_clusters = stage_num_clusters[s];
				let neurons_slice = &all_neurons_per_cluster[cluster_offset..cluster_offset + n_clusters];
				let total_neurons: usize = neurons_slice.iter().sum();
				let bits_slice = &all_bits_per_neuron[neuron_offset..neuron_offset + total_neurons];
				let total_conns: usize = bits_slice.iter().sum();
				let conns_slice = &all_connections[conn_offset..conn_offset + total_conns];

				stage_bits.push(bits_slice);
				stage_neurons.push(neurons_slice);
				stage_conns.push(conns_slice);

				neuron_offset += total_neurons;
				cluster_offset += n_clusters;
				conn_offset += total_conns;
			}

			Ok(multistage::compute_combined_ce(
				&self.inner,
				&stage_bits,
				&stage_neurons,
				&stage_conns,
				memory_mode,
				neuron_sample_rate,
				rng_seed,
				sparse_threshold,
				label_smoothing,
				unigram_lambda,
				bigram_lambda,
			))
		})
	}

	/// Compute combined CE for SELECTOR mode.
	///
	/// S0 is evaluated normally (bitwise or tiered).
	/// S1 is evaluated per-group using selector data (K sub-models).
	///
	/// When `invalid_mode` is true (Phase C), S1 groups train on ALL examples
	/// with an "invalid" target for out-of-group data, enabling self-correction
	/// of S0 mistakes. `top_m` limits the number of groups each example trains
	/// on (0 = all groups).
	#[allow(clippy::too_many_arguments)]
	fn evaluate_combined_ce_selector(
		&self,
		py: Python<'_>,
		all_bits_per_neuron: Vec<usize>,
		all_neurons_per_cluster: Vec<usize>,
		all_connections: Vec<i64>,
		stage_num_clusters: Vec<usize>,
		memory_mode: u8,
		neuron_sample_rate: f32,
		rng_seed: u64,
		sparse_threshold: usize,
		label_smoothing: f64,
		invalid_mode: bool,
		top_m: usize,
		unigram_lambda: f64,
		bigram_lambda: f64,
	) -> PyResult<(f64, f64, f64, f64, f64, f64)>
	{
		py.allow_threads(|| {
			// Partition flat arrays by stage (same as evaluate_combined_ce)
			let num_stages = stage_num_clusters.len();
			let mut stage_bits: Vec<&[usize]> = Vec::with_capacity(num_stages);
			let mut stage_neurons: Vec<&[usize]> = Vec::with_capacity(num_stages);
			let mut stage_conns: Vec<&[i64]> = Vec::with_capacity(num_stages);

			let mut neuron_offset = 0usize;
			let mut cluster_offset = 0usize;
			let mut conn_offset = 0usize;

			for s in 0..num_stages
			{
				let n_clusters = stage_num_clusters[s];
				let neurons_slice = &all_neurons_per_cluster[cluster_offset..cluster_offset + n_clusters];
				let total_neurons: usize = neurons_slice.iter().sum();
				let bits_slice = &all_bits_per_neuron[neuron_offset..neuron_offset + total_neurons];
				let total_conns: usize = bits_slice.iter().sum();
				let conns_slice = &all_connections[conn_offset..conn_offset + total_conns];

				stage_bits.push(bits_slice);
				stage_neurons.push(neurons_slice);
				stage_conns.push(conns_slice);

				neuron_offset += total_neurons;
				cluster_offset += n_clusters;
				conn_offset += total_conns;
			}

			Ok(multistage::compute_combined_ce_selector(
				&self.inner,
				&stage_bits,
				&stage_neurons,
				&stage_conns,
				memory_mode,
				neuron_sample_rate,
				rng_seed,
				sparse_threshold,
				label_smoothing,
				invalid_mode,
				top_m,
				unigram_lambda,
				bigram_lambda,
			))
		})
	}

	// ── Tiered stage methods ────────────────────────────────────────

	fn is_stage_tiered(&self, stage: usize) -> bool
	{
		self
			.inner
			.stage_is_tiered
			.get(stage)
			.copied()
			.unwrap_or(false)
	}

	fn stage_num_output_clusters(&self, stage: usize) -> usize
	{
		self
			.inner
			.stage_num_output_clusters
			.get(stage)
			.copied()
			.unwrap_or(0)
	}

	#[allow(clippy::too_many_arguments)]
	fn evaluate_tiered_genomes(
		&self,
		py: Python<'_>,
		stage: usize,
		bits_per_neuron_flat: Vec<usize>,
		neurons_per_cluster_flat: Vec<usize>,
		connections_flat: Vec<i64>,
		num_genomes: usize,
		train_subset_idx: usize,
		eval_subset_idx: usize,
		memory_mode: u8,
		neuron_sample_rate: f32,
		rng_seed: u64,
	) -> PyResult<Vec<(f64, f64, f64, f64)>>
	{
		py.allow_threads(|| {
			Ok(multistage::evaluate_tiered_genomes(
				&self.inner,
				stage,
				&bits_per_neuron_flat,
				&neurons_per_cluster_flat,
				&connections_flat,
				num_genomes,
				train_subset_idx,
				eval_subset_idx,
				memory_mode,
				neuron_sample_rate,
				rng_seed,
			))
		})
	}

	#[allow(clippy::too_many_arguments)]
	fn evaluate_tiered_genomes_full(
		&self,
		py: Python<'_>,
		stage: usize,
		bits_per_neuron_flat: Vec<usize>,
		neurons_per_cluster_flat: Vec<usize>,
		connections_flat: Vec<i64>,
		num_genomes: usize,
		memory_mode: u8,
		neuron_sample_rate: f32,
		rng_seed: u64,
	) -> PyResult<Vec<(f64, f64, f64, f64)>>
	{
		py.allow_threads(|| {
			Ok(multistage::evaluate_tiered_genomes_full(
				&self.inner,
				stage,
				&bits_per_neuron_flat,
				&neurons_per_cluster_flat,
				&connections_flat,
				num_genomes,
				memory_mode,
				neuron_sample_rate,
				rng_seed,
			))
		})
	}

	/// Re-encode data for `target_stage` using frozen previous stage's actual predictions.
	///
	/// Trains the frozen stage on full train data, gets predictions for both train and
	/// eval, then re-encodes all `target_stage` data with those predictions instead of
	/// teacher forcing.
	///
	/// Returns: (train_accuracy, eval_accuracy) — prediction accuracy for the frozen stage.
	#[allow(clippy::too_many_arguments)]
	fn recompute_stage_with_predictions(
		&mut self,
		py: Python<'_>,
		frozen_stage: usize,
		target_stage: usize,
		bits_per_neuron: Vec<usize>,
		neurons_per_cluster: Vec<usize>,
		connections: Vec<i64>,
		memory_mode: u8,
		neuron_sample_rate: f32,
		rng_seed: u64,
		sparse_threshold: usize,
	) -> PyResult<(f64, f64)>
	{
		let rw_rounds = self.reweight_rounds;
		let rw_max_boost = self.reweight_max_boost;
		py.allow_threads(|| {
			let (train_preds, eval_preds, train_correct, eval_correct) =
				multistage::predict_stage_clusters(
					&self.inner,
					frozen_stage,
					&bits_per_neuron,
					&neurons_per_cluster,
					&connections,
					memory_mode,
					neuron_sample_rate,
					rng_seed,
					sparse_threshold,
					rw_rounds,
					rw_max_boost,
				);

			let num_train = self.inner.bitwise_full_train[frozen_stage].num_examples;
			let num_eval = self.inner.bitwise_full_eval[frozen_stage].num_examples;

			self
				.inner
				.recompute_stage_data(target_stage, &train_preds, &eval_preds);

			let train_acc = train_correct as f64 / num_train.max(1) as f64;
			let eval_acc = eval_correct as f64 / num_eval.max(1) as f64;

			Ok((train_acc, eval_acc))
		})
	}
}
