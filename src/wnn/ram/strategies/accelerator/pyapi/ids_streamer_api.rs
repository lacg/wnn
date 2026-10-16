//! IDSGenomeStreamer — true streaming genome evaluator (Option F).
//!
//! Split out of lib.rs (D3, 11/06/2026).

use crate::*;

// =============================================================================
// IDSGenomeStreamer — true streaming genome evaluator (Option F)
// =============================================================================

/// Python wrapper for `ids_streaming::IDSGenomeStreamer`.
///
/// Per-genome streaming evaluator: holds only memory cells + a small score
/// buffer across chunks. Peak memory bounded by one chunk regardless of N.
/// See `ids_streaming.rs` for the full design.
///
/// Lifecycle (Python):
///     s = IDSGenomeStreamer(bits_flat, neurons_flat, connections, ...)
///     for packed_chunk, labels in train_stream:
///         s.train_chunk(packed_chunk.ravel(), labels)
///     s.seal_for_scoring()
///     for packed_chunk, labels in train_stream:   # single-cluster only
///         s.score_chunk(packed_chunk.ravel(), labels)
///     s.calibrate_threshold()
///     for packed_chunk, labels in eval_stream:
///         s.score_chunk(packed_chunk.ravel(), labels)
///     ce, acc, f1, fpr, threshold = s.finalize_metrics()
#[pyclass]
pub(crate) struct IDSGenomeStreamerWrapper
{
	inner: Option<ids_streaming::IDSGenomeStreamer>,
}

#[pymethods]
impl IDSGenomeStreamerWrapper
{
	/// `memory_mode` / `run_seed` / `coverage_aware` / `write_rule` /
	/// `fitness_weights` are the settings the in-memory `IDSCacheWrapper`
	/// takes through `set_memory_mode` / `set_coverage_aware` /
	/// `set_write_rule` / `set_fitness_weights` (worker ABI 20). They are
	/// constructor arguments here because the streamer allocates its cells
	/// and counters for the mode up front. Raises ValueError on an unknown
	/// mode or `write_rule="vote"` on a 4-state mode.
	#[new]
	#[allow(clippy::too_many_arguments)]
	#[pyo3(signature = (bits_flat, neurons_flat, connections, num_classes, num_negatives, single_cluster, total_features, empty_value=0.5, neuron_sample_rate=1.0, rng_seed=42, normal_class=0, class_weights=None, memory_mode=ram_core::neuron_memory::QUAD_WEIGHTED, run_seed=0, coverage_aware=false, write_rule="lattice", fitness_weights=None))]
	fn new(
		bits_flat: Vec<usize>,
		neurons_flat: Vec<usize>,
		connections: Vec<i64>,
		num_classes: usize,
		num_negatives: usize,
		single_cluster: bool,
		total_features: usize,
		empty_value: f32,
		neuron_sample_rate: f32,
		rng_seed: u64,
		normal_class: usize,
		class_weights: Option<Vec<u32>>,
		memory_mode: u8,
		run_seed: u64,
		coverage_aware: bool,
		write_rule: &str,
		fitness_weights: Option<(f32, f32, f32, f32)>,
	) -> PyResult<Self>
	{
		let _ = total_features; // used by Python-side validation; not needed in Rust ctor
		let settings = ram_core::neuron_memory::EvalSettings {
			empty_value,
			memory_mode,
			normal_class,
			fitness_weights,
			run_seed,
			coverage_aware,
			write_vote: parse_write_rule(write_rule)?,
		};
		let inner = ids_streaming::IDSGenomeStreamer::new(
			bits_flat,
			neurons_flat,
			connections,
			num_classes,
			num_negatives,
			single_cluster,
			settings,
			neuron_sample_rate,
			rng_seed,
			class_weights,
		)
		.map_err(pyo3::exceptions::PyValueError::new_err)?;
		Ok(Self { inner: Some(inner) })
	}

	/// Train on a single chunk. Writes to memory cells in-place.
	fn train_chunk<'py>(
		&mut self,
		chunk_packed: PyReadonlyArray1<'py, u8>,
		labels: Vec<i64>,
		total_features: usize,
	) -> PyResult<()>
	{
		let inner = self.inner.as_mut().ok_or_else(|| {
			pyo3::exceptions::PyRuntimeError::new_err("IDSGenomeStreamer already finalized")
		})?;
		let slice = chunk_packed.as_slice().map_err(|e| {
			pyo3::exceptions::PyValueError::new_err(format!("chunk not contiguous: {}", e))
		})?;
		let pb = ram_core::packed_bits::PackedBits::from_packed_slice(slice, total_features);
		inner.train_chunk(&pb, &labels);
		Ok(())
	}

	/// Finalize training; build the GPU-ready genome export. After this call,
	/// `train_chunk` raises and `score_chunk` becomes valid.
	fn seal_for_scoring(&mut self) -> PyResult<()>
	{
		let inner = self.inner.as_mut().ok_or_else(|| {
			pyo3::exceptions::PyRuntimeError::new_err("IDSGenomeStreamer already finalized")
		})?;
		inner.seal_for_scoring();
		Ok(())
	}

	/// Score a single eval chunk. Accumulates per-row scores.
	fn score_chunk<'py>(
		&mut self,
		chunk_packed: PyReadonlyArray1<'py, u8>,
		labels: Vec<i64>,
		total_features: usize,
	) -> PyResult<()>
	{
		let inner = self.inner.as_mut().ok_or_else(|| {
			pyo3::exceptions::PyRuntimeError::new_err("IDSGenomeStreamer already finalized")
		})?;
		let slice = chunk_packed.as_slice().map_err(|e| {
			pyo3::exceptions::PyValueError::new_err(format!("chunk not contiguous: {}", e))
		})?;
		let pb = ram_core::packed_bits::PackedBits::from_packed_slice(slice, total_features);
		inner.score_chunk(&pb, &labels);
		Ok(())
	}

	/// Drain the accumulated per-row scores, resetting the buffer (Protocol v2
	/// multi-set scoring: score eval, take, score train, take, score val, take).
	/// Subsequent score_chunk calls start a fresh accumulation.
	fn take_scores(&mut self) -> PyResult<Vec<f64>>
	{
		self.inner.as_mut().map(|i| i.take_scores()).ok_or_else(|| {
			pyo3::exceptions::PyRuntimeError::new_err("IDSGenomeStreamer already finalized")
		})
	}

	/// Single-cluster: the accumulated scores (score the TRAIN rows first)
	/// become the decision threshold; the buffer is cleared. Returns it.
	/// ValueError on a K-cluster streamer or with nothing scored (ABI 20).
	fn calibrate_threshold(&mut self) -> PyResult<f64>
	{
		let inner = self.inner.as_mut().ok_or_else(|| {
			pyo3::exceptions::PyRuntimeError::new_err("IDSGenomeStreamer already finalized")
		})?;
		inner.calibrate_threshold().map_err(pyo3::exceptions::PyValueError::new_err)
	}

	/// Compute final metrics. Consumes the inner state (one-shot).
	/// Returns (ce, acc, f1, fpr, threshold). Single-cluster threshold:
	/// `override_threshold` >= 0 fixed, < 0 eval-swept (oracle sentinel), None
	/// = the `calibrate_threshold` result (ValueError if never calibrated).
	#[pyo3(signature = (override_threshold=None))]
	fn finalize_metrics(&mut self, override_threshold: Option<f64>) -> PyResult<(f64, f64, f64, f64, f64)>
	{
		let inner = self.inner.take().ok_or_else(|| {
			pyo3::exceptions::PyRuntimeError::new_err(
				"IDSGenomeStreamer already finalized — cannot finalize twice",
			)
		})?;
		inner.finalize_metrics(override_threshold).map_err(pyo3::exceptions::PyValueError::new_err)
	}

	/// (logical_neurons, physical_neurons, logical_cells, physical_cells) of the
	/// sealed memory. Raises before `seal_for_scoring` (ABI 17).
	fn footprint(&self) -> PyResult<FootprintTuple>
	{
		let inner = self.inner.as_ref().ok_or_else(|| {
			pyo3::exceptions::PyRuntimeError::new_err("IDSGenomeStreamer already finalized")
		})?;
		inner.footprint().map(footprint_to_py).ok_or_else(|| {
			pyo3::exceptions::PyRuntimeError::new_err("IDSGenomeStreamer.footprint() before seal_for_scoring()")
		})
	}

	fn train_seen(&self) -> PyResult<usize>
	{
		self.inner.as_ref().map(|i| i.train_seen()).ok_or_else(|| {
			pyo3::exceptions::PyRuntimeError::new_err("IDSGenomeStreamer already finalized")
		})
	}

	fn eval_scored(&self) -> PyResult<usize>
	{
		self.inner.as_ref().map(|i| i.eval_scored()).ok_or_else(|| {
			pyo3::exceptions::PyRuntimeError::new_err("IDSGenomeStreamer already finalized")
		})
	}
}

/// Multiclass decode-mode metrics from pre-computed flat K-vector scores
/// (`scores_flat[ex * K + c]` — the buffers `take_scores` drains after
/// multi-cluster streaming score passes). The streaming counterpart of
/// `IDSCacheWrapper.evaluate_multiclass_at_thresholds`: same
/// `(num_classes, mode tuples)` return contract, same decode modes
/// (argmax / margin_fixed0 / margin_train_cal / margin_val_cal-when-val /
/// argmax_platt / argmax_beta — per-class one-vs-rest calibrated argmax,
/// fit on val when present else train). Metrics are ALWAYS computed on the
/// EVAL scores; train/val partitions only calibrate τ / the per-class maps.
#[pyfunction]
#[pyo3(signature = (eval_scores, eval_labels, train_scores, train_labels, num_classes, normal_class=0, val_scores=None, val_labels=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn multiclass_modes_from_scores(
	py: Python<'_>,
	eval_scores: Vec<f64>,
	eval_labels: Vec<i64>,
	train_scores: Vec<f64>,
	train_labels: Vec<i64>,
	num_classes: usize,
	normal_class: usize,
	val_scores: Option<Vec<f64>>,
	val_labels: Option<Vec<i64>>,
) -> PyResult<(usize, Vec<MulticlassModeTuple>)>
{
	if num_classes < 2
	{
		return Err(pyo3::exceptions::PyValueError::new_err(format!(
			"multiclass_modes_from_scores requires num_classes >= 2, got {num_classes}",
		)));
	}
	let check_set = |name: &str, scores: &[f64], labels: &[i64]| -> PyResult<()> {
		if scores.len() != labels.len() * num_classes
		{
			return Err(pyo3::exceptions::PyValueError::new_err(format!(
				"{name}: {} scores != {} labels × {num_classes} classes",
				scores.len(),
				labels.len(),
			)));
		}
		Ok(())
	};
	check_set("eval", &eval_scores, &eval_labels)?;
	check_set("train", &train_scores, &train_labels)?;
	if val_scores.is_some() != val_labels.is_some()
	{
		return Err(pyo3::exceptions::PyValueError::new_err(
			"val_scores and val_labels must be provided together",
		));
	}
	if let (Some(vs), Some(vl)) = (val_scores.as_ref(), val_labels.as_ref())
	{
		check_set("val", vs, vl)?;
	}
	py.allow_threads(|| {
		let modes = multiclass_metrics::modes_from_scores(
			&eval_scores,
			&eval_labels,
			&train_scores,
			&train_labels,
			val_scores.as_deref(),
			val_labels.as_deref(),
			num_classes,
			normal_class,
		);
		Ok((num_classes, multiclass_modes_to_py(modes)))
	})
}
