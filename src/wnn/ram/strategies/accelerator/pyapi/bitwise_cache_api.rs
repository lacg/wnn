//! Bitwise RAMLM cache wrapper.
//!
//! Split out of lib.rs (D3, 11/06/2026).

use crate::*;

// =============================================================================
// Bitwise RAMLM Cache Wrapper (PyO3)
// =============================================================================

/// Persistent cache for bitwise genome evaluation.
///
/// Holds pre-encoded tokens in Rust memory. Evaluates genomes
/// entirely in Rust+Metal (no Python overhead per genome).
#[pyclass]
pub(crate) struct BitwiseCacheWrapper
{
	inner: bitwise_ramlm::BitwiseTokenCache,
	/// Optional override: None = auto-compute per genome based on budget.
	sparse_threshold_override: Option<usize>,
	experiment_id: Option<i64>,
}

#[pymethods]
impl BitwiseCacheWrapper
{
	#[new]
	#[pyo3(signature = (train_tokens, eval_tokens, vocab_size, context_size, num_parts, num_eval_parts, pad_token_id, sparse_threshold=None))]
	fn new(
		train_tokens: Vec<u32>,
		eval_tokens: Vec<u32>,
		vocab_size: usize,
		context_size: usize,
		num_parts: usize,
		num_eval_parts: usize,
		pad_token_id: u32,
		sparse_threshold: Option<usize>,
	) -> Self
	{
		Self {
			inner: bitwise_ramlm::BitwiseTokenCache::new(
				train_tokens,
				eval_tokens,
				vocab_size,
				context_size,
				num_parts,
				num_eval_parts,
				pad_token_id,
			),
			sparse_threshold_override: sparse_threshold,
			experiment_id: None,
		}
	}

	/// Evaluate genomes with per-neuron heterogeneous configs (subset training + eval).
	///
	/// bits_per_neuron_flat: variable total (sum of total_neurons per genome)
	/// neurons_per_cluster_flat: [num_genomes * num_clusters]
	/// connections_flat: variable total (sum of all genomes' connections)
	#[allow(clippy::too_many_arguments)]
	fn evaluate_genomes(
		&self,
		py: Python<'_>,
		bits_per_neuron_flat: Vec<usize>,
		neurons_per_cluster_flat: Vec<usize>,
		connections_flat: Vec<i64>,
		num_genomes: usize,
		train_subset_idx: usize,
		eval_subset_idx: usize,
		memory_mode: u8,
		empty_value: f32,
		neuron_sample_rate: f32,
		rng_seed: u64,
	) -> PyResult<Vec<(f64, f64, f64, f64)>>
	{
		let override_val = self.sparse_threshold_override;
		py.allow_threads(|| {
			Ok(bitwise_ramlm::evaluate_genomes(
				&self.inner,
				&bits_per_neuron_flat,
				&neurons_per_cluster_flat,
				&connections_flat,
				num_genomes,
				train_subset_idx,
				eval_subset_idx,
				memory_mode,
				empty_value,
				neuron_sample_rate,
				rng_seed,
				override_val,
			))
		})
	}

	/// Evaluate genomes with per-neuron heterogeneous configs (full training + full eval).
	#[allow(clippy::too_many_arguments)]
	fn evaluate_genomes_full(
		&self,
		py: Python<'_>,
		bits_per_neuron_flat: Vec<usize>,
		neurons_per_cluster_flat: Vec<usize>,
		connections_flat: Vec<i64>,
		num_genomes: usize,
		memory_mode: u8,
		empty_value: f32,
		neuron_sample_rate: f32,
		rng_seed: u64,
	) -> PyResult<Vec<(f64, f64, f64, f64)>>
	{
		let override_val = self.sparse_threshold_override;
		py.allow_threads(|| {
			Ok(bitwise_ramlm::evaluate_genomes_full(
				&self.inner,
				&bits_per_neuron_flat,
				&neurons_per_cluster_flat,
				&connections_flat,
				num_genomes,
				memory_mode,
				empty_value,
				neuron_sample_rate,
				rng_seed,
				override_val,
			))
		})
	}

	/// Get next train subset index (advances rotator).
	fn next_train_idx(&self) -> usize
	{
		self.inner.next_train_idx()
	}

	/// Get next eval subset index (advances rotator).
	fn next_eval_idx(&self) -> usize
	{
		self.inner.next_eval_idx()
	}

	/// Reset subset rotation (both train and eval).
	fn reset(&self)
	{
		self.inner.reset();
	}

	fn vocab_size(&self) -> usize
	{
		self.inner.vocab_size
	}
	fn total_input_bits(&self) -> usize
	{
		self.inner.total_input_bits
	}
	fn num_parts(&self) -> usize
	{
		self.inner.num_parts
	}
	fn num_eval_parts(&self) -> usize
	{
		self.inner.num_eval_parts
	}
	fn num_bits(&self) -> usize
	{
		self.inner.num_bits
	}

	/// Set experiment context for live progress reporting.
	fn set_experiment_context(&mut self, experiment_id: i64)
	{
		self.experiment_id = Some(experiment_id);
	}

	/// Get current live progress from active search (if any).
	fn get_live_progress(&self, py: Python<'_>) -> PyResult<Option<pyo3::PyObject>>
	{
		let guard = self
			.inner
			.live_progress
			.read()
			.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Lock poisoned: {}", e)))?;
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
				dict.set_item("updated_at", lp.updated_at)?;
				Ok(Some(dict.into()))
			}
		}
	}

	/// Search for neighbors above accuracy threshold (bitwise eval backend).
	///
	/// Same interface as TokenCacheWrapper::search_neighbors but uses the
	/// bitwise evaluation path (heterogeneous per-neuron configs).
	#[allow(clippy::too_many_arguments)]
	#[pyo3(signature = (
        base_bits,
        base_neurons,
        base_connections,
        target_count,
        max_attempts,
        accuracy_threshold,
        min_bits,
        max_bits,
        min_neurons,
        max_neurons,
        bits_mutation_rate,
        neurons_mutation_rate,
        train_subset_idx,
        eval_subset_idx,
        memory_mode,
        empty_value,
        neuron_sample_rate,
        rng_seed,
        seed,
        log_path = None,
        generation = None,
        total_generations = None,
        return_best_n = true,
        mutable_clusters = None,
        phase_type = 0
    ))]
	fn search_neighbors(
		&self,
		py: Python<'_>,
		base_bits: Vec<usize>,
		base_neurons: Vec<usize>,
		base_connections: Vec<i64>,
		target_count: usize,
		max_attempts: usize,
		accuracy_threshold: f64,
		min_bits: usize,
		max_bits: usize,
		min_neurons: usize,
		max_neurons: usize,
		bits_mutation_rate: f64,
		neurons_mutation_rate: f64,
		train_subset_idx: usize,
		eval_subset_idx: usize,
		memory_mode: u8,
		empty_value: f32,
		neuron_sample_rate: f32,
		rng_seed: u64,
		seed: u64,
		log_path: Option<String>,
		generation: Option<usize>,
		total_generations: Option<usize>,
		return_best_n: bool,
		mutable_clusters: Option<Vec<usize>>,
		phase_type: u8,
	) -> PyResult<Vec<(Vec<usize>, Vec<usize>, Vec<i64>, f64, f64, f64, f64)>>
	{
		let num_clusters = base_neurons.len();
		let total_input_bits = self.inner.total_input_bits;
		let phase = match phase_type
		{
			1 => neighbor_search::PhaseType::Bits,
			2 => neighbor_search::PhaseType::Connections,
			3 => neighbor_search::PhaseType::Cluster,
			_ => neighbor_search::PhaseType::Neurons,
		};

		let config = neighbor_search::MutationConfig {
			num_clusters,
			mutable_clusters,
			min_bits,
			max_bits,
			min_neurons,
			max_neurons,
			bits_mutation_rate,
			neurons_mutation_rate,
			total_input_bits,
			phase_type: phase,
		};

		let override_val = self.sparse_threshold_override;

		// Set up live progress for observer thread
		let lp_arc = self.inner.live_progress.clone();
		let exp_id = self.experiment_id.unwrap_or(0);
		if let Ok(mut guard) = lp_arc.write()
		{
			*guard = Some(neighbor_search::LiveProgress {
				experiment_id: exp_id,
				generation: generation.map(|g| g as i32 + 1).unwrap_or(1),
				total_generations: total_generations.map(|g| g as i32).unwrap_or(100),
				phase: "ts_neighbors".into(),
				evaluated: 0,
				target_count,
				viable: Some(0),
				best_ce: f64::MAX,
				best_acc: 0.0,
				elapsed_secs: 0.0,
				updated_at: neighbor_search::LiveProgress::now_unix(),
			});
		}

		let result = py.allow_threads(|| {
			let log_path_ref = log_path.as_deref();
			let cache = &self.inner;

			let eval_fn = |bits: &[usize],
			               neurons: &[usize],
			               conns: &[i64],
			               count: usize|
			 -> Vec<(f64, f64, f64, f64)> {
				bitwise_ramlm::evaluate_genomes(
					cache,
					bits,
					neurons,
					conns,
					count,
					train_subset_idx,
					eval_subset_idx,
					memory_mode,
					empty_value,
					neuron_sample_rate,
					rng_seed,
					override_val,
				)
			};

			let lp_ref = Some(&lp_arc);

			let candidates = if return_best_n
			{
				neighbor_search::search_neighbors_best_n(
					&base_bits,
					&base_neurons,
					&base_connections,
					target_count,
					max_attempts,
					accuracy_threshold,
					&config,
					&eval_fn,
					seed,
					log_path_ref,
					generation,
					total_generations,
					lp_ref,
				)
			}
			else
			{
				let (passed, _) = neighbor_search::search_neighbors_with_threshold(
					&base_bits,
					&base_neurons,
					&base_connections,
					target_count,
					max_attempts,
					accuracy_threshold,
					&config,
					&eval_fn,
					seed,
					log_path_ref,
					generation,
					total_generations,
					lp_ref,
				);
				passed
			};

			Ok(
				candidates
					.into_iter()
					.map(|c| {
						(
							c.bits_per_neuron,
							c.neurons_per_cluster,
							c.connections,
							c.cross_entropy,
							c.accuracy,
							c.f1_macro,
							c.fpr,
						)
					})
					.collect(),
			)
		});

		// Clear live progress after search completes
		if let Ok(mut guard) = lp_arc.write()
		{
			*guard = None;
		}
		result
	}

	/// Search for GA offspring above accuracy threshold (bitwise eval backend).
	#[allow(clippy::too_many_arguments)]
	#[pyo3(signature = (
        population,
        target_count,
        max_attempts,
        accuracy_threshold,
        min_bits,
        max_bits,
        min_neurons,
        max_neurons,
        bits_mutation_rate,
        neurons_mutation_rate,
        crossover_rate,
        tournament_size,
        train_subset_idx,
        eval_subset_idx,
        memory_mode,
        empty_value,
        neuron_sample_rate,
        rng_seed,
        seed,
        log_path = None,
        generation = None,
        total_generations = None,
        return_best_n = true,
        mutable_clusters = None,
        phase_type = 0,
        cluster_crossover_ratio = 0.0,
        pool_shuffle_ratio = 0.0,
        assortative_mating_ratio = 0.0
    ))]
	fn search_offspring(
		&self,
		py: Python<'_>,
		population: Vec<(Vec<usize>, Vec<usize>, Vec<i64>, f64)>,
		target_count: usize,
		max_attempts: usize,
		accuracy_threshold: f64,
		min_bits: usize,
		max_bits: usize,
		min_neurons: usize,
		max_neurons: usize,
		bits_mutation_rate: f64,
		neurons_mutation_rate: f64,
		crossover_rate: f64,
		tournament_size: usize,
		train_subset_idx: usize,
		eval_subset_idx: usize,
		memory_mode: u8,
		empty_value: f32,
		neuron_sample_rate: f32,
		rng_seed: u64,
		seed: u64,
		log_path: Option<String>,
		generation: Option<usize>,
		total_generations: Option<usize>,
		return_best_n: bool,
		mutable_clusters: Option<Vec<usize>>,
		phase_type: u8,
		cluster_crossover_ratio: f64,
		pool_shuffle_ratio: f64,
		assortative_mating_ratio: f64,
	) -> PyResult<(
		Vec<(Vec<usize>, Vec<usize>, Vec<i64>, f64, f64, f64, f64)>,
		usize,
		usize,
	)>
	{
		let num_clusters = if !population.is_empty()
		{
			population[0].1.len()
		}
		else
		{
			return Ok((Vec::new(), 0, 0));
		};
		let total_input_bits = self.inner.total_input_bits;
		let phase = match phase_type
		{
			1 => neighbor_search::PhaseType::Bits,
			2 => neighbor_search::PhaseType::Connections,
			3 => neighbor_search::PhaseType::Cluster,
			_ => neighbor_search::PhaseType::Neurons,
		};

		let ga_config = neighbor_search::GAConfig {
			num_clusters,
			mutable_clusters,
			min_bits,
			max_bits,
			min_neurons,
			max_neurons,
			bits_mutation_rate,
			neurons_mutation_rate,
			crossover_rate,
			tournament_size,
			total_input_bits,
			phase_type: phase,
			cluster_crossover_ratio,
			pool_shuffle_ratio,
			assortative_mating_ratio,
		};

		let override_val = self.sparse_threshold_override;

		// Set up live progress for observer thread
		let lp_arc = self.inner.live_progress.clone();
		let exp_id = self.experiment_id.unwrap_or(0);
		if let Ok(mut guard) = lp_arc.write()
		{
			*guard = Some(neighbor_search::LiveProgress {
				experiment_id: exp_id,
				generation: generation.map(|g| g as i32 + 1).unwrap_or(1),
				total_generations: total_generations.map(|g| g as i32).unwrap_or(100),
				phase: "ga_offspring".into(),
				evaluated: 0,
				target_count,
				viable: Some(0),
				best_ce: f64::MAX,
				best_acc: 0.0,
				elapsed_secs: 0.0,
				updated_at: neighbor_search::LiveProgress::now_unix(),
			});
		}

		let result = py.allow_threads(|| {
			let log_path_ref = log_path.as_deref();
			let cache = &self.inner;

			let eval_fn = |bits: &[usize],
			               neurons: &[usize],
			               conns: &[i64],
			               count: usize|
			 -> Vec<(f64, f64, f64, f64)> {
				bitwise_ramlm::evaluate_genomes(
					cache,
					bits,
					neurons,
					conns,
					count,
					train_subset_idx,
					eval_subset_idx,
					memory_mode,
					empty_value,
					neuron_sample_rate,
					rng_seed,
					override_val,
				)
			};

			let lp_ref = Some(&lp_arc);

			let result = neighbor_search::search_offspring(
				&population,
				target_count,
				max_attempts,
				accuracy_threshold,
				&ga_config,
				&eval_fn,
				seed,
				log_path_ref,
				generation,
				total_generations,
				return_best_n,
				lp_ref,
			);

			let candidates: Vec<_> = result
				.candidates
				.into_iter()
				.map(|c| {
					(
						c.bits_per_neuron,
						c.neurons_per_cluster,
						c.connections,
						c.cross_entropy,
						c.accuracy,
						c.f1_macro,
						c.fpr,
					)
				})
				.collect();
			Ok((candidates, result.evaluated, result.viable))
		});

		// Clear live progress after search completes
		if let Ok(mut guard) = lp_arc.write()
		{
			*guard = None;
		}
		result
	}

	/// Evaluate multiple genomes with per-genome adaptation (Baldwin effect).
	///
	/// Each genome is adapted (synaptogenesis/neurogenesis) during evaluation,
	/// so GA/TS sees adapted fitness. Returns adapted architecture per genome.
	fn evaluate_genomes_adaptive(
		&self,
		py: Python<'_>,
		bits_per_neuron_flat: Vec<usize>,
		neurons_per_cluster_flat: Vec<usize>,
		connections_flat: Vec<i64>,
		num_genomes: usize,
		train_subset_idx: usize,
		eval_subset_idx: usize,
		memory_mode: u8,
		empty_value: f32,
		neuron_sample_rate: f32,
		rng_seed: u64,
		generation: usize,
		// Adaptation config fields (relative thresholds)
		synaptogenesis_enabled: bool,
		neurogenesis_enabled: bool,
		axonogenesis_enabled: bool,
		prune_entropy_ratio: f32,
		grow_fill_utilization: f32,
		grow_error_baseline: f32,
		min_bits: usize,
		max_bits: usize,
		cluster_error_factor: f32,
		cluster_fill_utilization: f32,
		neuron_prune_percentile: f32,
		neuron_removal_factor: f32,
		max_growth_ratio: f32,
		min_neurons: usize,
		max_neurons_per_pass: usize,
		axon_entropy_threshold: f32,
		axon_improvement_factor: f32,
		axon_rewire_count: usize,
		warmup_generations: usize,
		cooldown_iterations: usize,
		stabilize_fraction: f32,
		total_generations: usize,
		passes_per_eval: usize,
		stats_sample_size: usize,
	) -> PyResult<
		Vec<(
			f64,
			f64,
			f64, // ce, acc, bit_acc
			Vec<usize>,
			Vec<usize>,
			Vec<i64>, // adapted bits, neurons, connections
			usize,
			usize,
			usize,
			usize,
			usize, // pruned, grown, added, removed, rewired
		)>,
	>
	{
		let override_val = self.sparse_threshold_override;
		let total_input_bits = self.inner.total_input_bits;

		let config = adaptation::AdaptationConfig {
			synaptogenesis_enabled,
			neurogenesis_enabled,
			axonogenesis_enabled,
			axon_entropy_threshold,
			axon_improvement_factor,
			axon_rewire_count,
			prune_entropy_ratio,
			grow_fill_utilization,
			grow_error_baseline,
			min_bits,
			max_bits,
			cluster_error_factor,
			cluster_fill_utilization,
			neuron_prune_percentile,
			neuron_removal_factor,
			max_growth_ratio,
			min_neurons,
			max_neurons_per_pass,
			warmup_generations,
			cooldown_iterations,
			stabilize_fraction,
			total_generations,
			passes_per_eval,
			total_input_bits,
			stats_sample_size,
			neuron_sample_rate,
		};

		py.allow_threads(|| {
			let cache = &self.inner;
			let train_subset = &cache.train_subsets[train_subset_idx % cache.num_parts];
			let eval_subset = &cache.eval_subsets[eval_subset_idx % cache.num_eval_parts];

			let results = bitwise_ramlm::evaluate_genomes_adaptive(
				cache,
				&bits_per_neuron_flat,
				&neurons_per_cluster_flat,
				&connections_flat,
				num_genomes,
				train_subset,
				eval_subset,
				memory_mode,
				empty_value,
				neuron_sample_rate,
				rng_seed,
				override_val,
				&config,
				generation,
			);

			Ok(
				results
					.into_iter()
					.map(|r| {
						(
							r.ce,
							r.acc,
							r.bit_acc,
							r.adapted_bits,
							r.adapted_neurons,
							r.adapted_connections,
							r.pruned,
							r.grown,
							r.added,
							r.removed,
							r.rewired,
						)
					})
					.collect(),
			)
		})
	}
}
