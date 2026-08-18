//! TokenCache — persistent token storage with subset rotation.
//!
//! Split out of lib.rs (D3, 11/06/2026).

use crate::*;

// =============================================================================
// TOKEN CACHE - Persistent token storage with subset rotation
// =============================================================================

/// Python-accessible TokenCache for persistent token storage.
///
/// Create once at session start, then use for all evaluations without
/// any data transfer overhead.
#[pyclass]
pub(crate) struct TokenCacheWrapper
{
	inner: token_cache::TokenCache,
	experiment_id: Option<i64>,
}

#[pymethods]
impl TokenCacheWrapper
{
	/// Create a new token cache with all data pre-encoded and partitioned.
	///
	/// # Arguments
	/// * `encoding_table` - Optional semantic encoding table (token_id → semantic_bits).
	///   If provided, tokens are encoded using learned semantic bits instead of raw binary.
	///   Similar tokens will have similar bit patterns, enabling better generalization.
	///   Pre-computed in Python using MutualInfoEncoder or similar.
	/// * `encoding_bits` - Number of bits in semantic encoding (required if encoding_table provided).
	#[new]
	#[allow(clippy::too_many_arguments)]
	#[pyo3(signature = (train_tokens, eval_tokens, test_tokens, vocab_size, context_size, cluster_order, num_parts, num_negatives, seed, encoding_table=None, encoding_bits=None, num_eval_parts=None))]
	fn new(
		train_tokens: Vec<u32>,
		eval_tokens: Vec<u32>,
		test_tokens: Vec<u32>,
		vocab_size: usize,
		context_size: usize,
		cluster_order: Vec<usize>,
		num_parts: usize,
		num_negatives: usize,
		seed: u64,
		encoding_table: Option<Vec<u64>>,
		encoding_bits: Option<usize>,
		num_eval_parts: Option<usize>,
	) -> Self
	{
		Self {
			inner: token_cache::TokenCache::new(
				train_tokens,
				eval_tokens,
				test_tokens,
				vocab_size,
				context_size,
				cluster_order,
				num_parts,
				num_negatives,
				seed,
				encoding_table,
				encoding_bits,
				num_eval_parts.unwrap_or(1),
			),
			experiment_id: None,
		}
	}

	/// Get the next train subset index (advances rotator).
	fn next_train_idx(&mut self) -> usize
	{
		self.inner.next_train_idx()
	}

	/// Get the next eval subset index (advances rotator).
	fn next_eval_idx(&mut self) -> usize
	{
		self.inner.next_eval_idx()
	}

	/// Reset rotators with optional new seed.
	#[pyo3(signature = (seed=None))]
	fn reset(&mut self, seed: Option<u64>)
	{
		self.inner.reset(seed);
	}

	/// Get number of train subsets.
	fn num_train_subsets(&self) -> usize
	{
		self.inner.num_train_subsets()
	}

	/// Get vocab size.
	fn vocab_size(&self) -> usize
	{
		self.inner.vocab_size()
	}

	/// Get total input bits.
	fn total_input_bits(&self) -> usize
	{
		self.inner.total_input_bits()
	}

	/// Evaluate genomes using a specific train/eval subset combination.
	///
	/// This is the main evaluation function - zero data copy, just uses
	/// pre-cached data selected by indices.
	#[allow(clippy::too_many_arguments)]
	fn evaluate_genomes(
		&self,
		py: Python<'_>,
		genomes_bits_flat: Vec<usize>,
		genomes_neurons_flat: Vec<usize>,
		genomes_connections_flat: Vec<i64>,
		num_genomes: usize,
		train_subset_idx: usize,
		eval_subset_idx: usize,
		empty_value: f32,
		neuron_sample_rate: f32,
		rng_seed: u64,
	) -> PyResult<Vec<(f64, f64, f64, f64)>>
	{
		validate_flat_genomes_py(
			&genomes_bits_flat,
			&genomes_neurons_flat,
			&genomes_connections_flat,
			num_genomes,
			self.inner.vocab_size(),
		)?;
		py.allow_threads(|| {
			Ok(token_cache::evaluate_genomes_cached(
				&self.inner,
				&genomes_bits_flat,
				&genomes_neurons_flat,
				&genomes_connections_flat,
				num_genomes,
				train_subset_idx,
				eval_subset_idx,
				empty_value,
				neuron_sample_rate,
				rng_seed,
			))
		})
	}

	/// Evaluate genomes using full train/eval data (for final evaluation).
	fn evaluate_genomes_full(
		&self,
		py: Python<'_>,
		genomes_bits_flat: Vec<usize>,
		genomes_neurons_flat: Vec<usize>,
		genomes_connections_flat: Vec<i64>,
		num_genomes: usize,
		empty_value: f32,
		neuron_sample_rate: f32,
		rng_seed: u64,
	) -> PyResult<Vec<(f64, f64, f64, f64)>>
	{
		validate_flat_genomes_py(
			&genomes_bits_flat,
			&genomes_neurons_flat,
			&genomes_connections_flat,
			num_genomes,
			self.inner.vocab_size(),
		)?;
		py.allow_threads(|| {
			Ok(token_cache::evaluate_genomes_cached_full(
				&self.inner,
				&genomes_bits_flat,
				&genomes_neurons_flat,
				&genomes_connections_flat,
				num_genomes,
				empty_value,
				neuron_sample_rate,
				rng_seed,
			))
		})
	}

	/// Evaluate genomes using hybrid CPU+GPU parallel evaluation (4-8x speedup).
	///
	/// Uses memory pool for parallel training, GPU batch evaluation, and pipelining.
	#[allow(clippy::too_many_arguments)]
	fn evaluate_genomes_hybrid(
		&self,
		py: Python<'_>,
		genomes_bits_flat: Vec<usize>,
		genomes_neurons_flat: Vec<usize>,
		genomes_connections_flat: Vec<i64>,
		num_genomes: usize,
		train_subset_idx: usize,
		eval_subset_idx: usize,
		empty_value: f32,
		neuron_sample_rate: f32,
		rng_seed: u64,
	) -> PyResult<Vec<(f64, f64, f64, f64)>>
	{
		validate_flat_genomes_py(
			&genomes_bits_flat,
			&genomes_neurons_flat,
			&genomes_connections_flat,
			num_genomes,
			self.inner.vocab_size(),
		)?;
		py.allow_threads(|| {
			Ok(token_cache::evaluate_genomes_cached_hybrid(
				&self.inner,
				&genomes_bits_flat,
				&genomes_neurons_flat,
				&genomes_connections_flat,
				num_genomes,
				train_subset_idx,
				eval_subset_idx,
				empty_value,
				neuron_sample_rate,
				rng_seed,
			))
		})
	}

	/// Evaluate genomes using full data with hybrid CPU+GPU (4-8x speedup).
	fn evaluate_genomes_full_hybrid(
		&self,
		py: Python<'_>,
		genomes_bits_flat: Vec<usize>,
		genomes_neurons_flat: Vec<usize>,
		genomes_connections_flat: Vec<i64>,
		num_genomes: usize,
		empty_value: f32,
		neuron_sample_rate: f32,
		rng_seed: u64,
	) -> PyResult<Vec<(f64, f64, f64, f64)>>
	{
		validate_flat_genomes_py(
			&genomes_bits_flat,
			&genomes_neurons_flat,
			&genomes_connections_flat,
			num_genomes,
			self.inner.vocab_size(),
		)?;
		py.allow_threads(|| {
			Ok(token_cache::evaluate_genomes_cached_full_hybrid(
				&self.inner,
				&genomes_bits_flat,
				&genomes_neurons_flat,
				&genomes_connections_flat,
				num_genomes,
				empty_value,
				neuron_sample_rate,
				rng_seed,
			))
		})
	}

	/// Evaluate a single genome WITH gating, returning both gated and non-gated metrics.
	///
	/// This function:
	/// 1. Trains base RAM on full training data
	/// 2. Trains gating model on training data (target gate = true only for target cluster)
	/// 3. Evaluates WITHOUT gating → (ce, acc)
	/// 4. Evaluates WITH gating → (gated_ce, gated_acc)
	///
	/// # Arguments
	/// * `bits_flat` - Bits per cluster [num_clusters]
	/// * `neurons_flat` - Neurons per cluster [num_clusters]
	/// * `connections_flat` - Connections [total_connections]
	/// * `neurons_per_gate` - Number of RAM neurons per gate (default 8)
	/// * `bits_per_gate_neuron` - Address bits per gate neuron (default 12)
	/// * `vote_threshold_frac` - Fraction of neurons that must fire for gate=1 (default 0.5)
	/// * `empty_value` - Value for EMPTY cells (default 0.5)
	/// * `gating_seed` - Random seed for gating connectivity
	///
	/// # Returns
	/// (ce, accuracy, gated_ce, gated_accuracy)
	#[allow(clippy::too_many_arguments)]
	#[pyo3(signature = (
        bits_flat,
        neurons_flat,
        connections_flat,
        neurons_per_gate = 8,
        bits_per_gate_neuron = 12,
        vote_threshold_frac = 0.5,
        empty_value = 0.5,
        gating_seed = 42
    ))]
	fn evaluate_genome_with_gating(
		&self,
		py: Python<'_>,
		bits_flat: Vec<usize>,
		neurons_flat: Vec<usize>,
		connections_flat: Vec<i64>,
		neurons_per_gate: usize,
		bits_per_gate_neuron: usize,
		vote_threshold_frac: f32,
		empty_value: f32,
		gating_seed: u64,
	) -> PyResult<(f64, f64, f64, f64)>
	{
		py.allow_threads(|| {
			Ok(token_cache::evaluate_genome_with_gating(
				&self.inner,
				&bits_flat,
				&neurons_flat,
				&connections_flat,
				neurons_per_gate,
				bits_per_gate_neuron,
				vote_threshold_frac,
				empty_value,
				gating_seed,
			))
		})
	}

	/// Set experiment context for live progress reporting.
	fn set_experiment_context(&mut self, experiment_id: i64)
	{
		self.experiment_id = Some(experiment_id);
	}

	/// Get current live progress from active search (if any).
	///
	/// Returns None if no search is in progress, otherwise returns a dict
	/// with progress fields. Called by the Python observer thread.
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

	/// Search for neighbors above accuracy threshold, all in Rust.
	///
	/// This eliminates Python↔Rust round trips by doing mutation, evaluation,
	/// and filtering entirely in Rust. Logs progress to file with flush.
	///
	/// Returns: List of (bits_flat, neurons_flat, connections_flat, CE, accuracy)
	/// for candidates that passed the threshold.
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
        empty_value,
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
		empty_value: f32,
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
		let total_input_bits = self.inner.total_input_bits();
		let phase = match phase_type
		{
			1 => neighbor_search::PhaseType::Bits,
			2 => neighbor_search::PhaseType::Connections,
			3 => neighbor_search::PhaseType::Cluster,
			_ => neighbor_search::PhaseType::Neurons,
		};

		let config = neighbor_search::MutationConfig {
			num_clusters,
			mutable_clusters, // None = all clusters, Some(indices) = only those
			min_bits,
			max_bits,
			min_neurons,
			max_neurons,
			bits_mutation_rate,
			neurons_mutation_rate,
			total_input_bits,
			phase_type: phase,
		};

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

			// Closure captures the token cache and eval params
			let eval_fn = |bits: &[usize],
			               neurons: &[usize],
			               conns: &[i64],
			               count: usize|
			 -> Vec<(f64, f64, f64, f64)> {
				crate::token_cache::evaluate_genomes_cached_hybrid(
					cache,
					bits,
					neurons,
					conns,
					count,
					train_subset_idx,
					eval_subset_idx,
					empty_value,
					1.0,
					0, // no neuron sampling for neighbor search
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

	/// Search for GA offspring above accuracy threshold, all in Rust.
	///
	/// Performs tournament selection, crossover, mutation, and evaluation
	/// entirely in Rust. Returns viable offspring (accuracy >= threshold).
	///
	/// Args:
	///   - population: List of (bits, neurons, connections, fitness) tuples
	///   - target_count: Number of viable offspring needed
	///   - max_attempts: Maximum offspring to generate
	///   - accuracy_threshold: Minimum accuracy for viable offspring
	///
	/// Returns: List of (bits, neurons, connections, CE, accuracy) tuples
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
        empty_value,
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
		empty_value: f32,
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
		// Returns: (candidates, evaluated, viable)
		let num_clusters = if !population.is_empty()
		{
			population[0].1.len() // neurons_per_cluster length = num_clusters
		}
		else
		{
			return Ok((Vec::new(), 0, 0));
		};
		let total_input_bits = self.inner.total_input_bits();
		let phase = match phase_type
		{
			1 => neighbor_search::PhaseType::Bits,
			2 => neighbor_search::PhaseType::Connections,
			3 => neighbor_search::PhaseType::Cluster,
			_ => neighbor_search::PhaseType::Neurons,
		};

		let ga_config = neighbor_search::GAConfig {
			num_clusters,
			mutable_clusters, // None = all clusters, Some(indices) = only those
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
				crate::token_cache::evaluate_genomes_cached_hybrid(
					cache,
					bits,
					neurons,
					conns,
					count,
					train_subset_idx,
					eval_subset_idx,
					empty_value,
					1.0,
					0, // no neuron sampling for GA offspring search
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
}
