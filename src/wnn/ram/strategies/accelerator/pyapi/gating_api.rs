//! RAMGating wrapper.
//!
//! Split out of lib.rs (D3, 11/06/2026).

use crate::*;

// =============================================================================
// RAMGating Python Wrapper
// =============================================================================

/// Python wrapper for RAM-based gating
///
/// Uses dedicated RAM neurons to learn which clusters should be active
/// for each input context. Gate output is binary (0 or 1) via majority voting.
#[pyclass]
pub(crate) struct RAMGatingWrapper {
    inner: gating::RAMGating,
}

#[pymethods]
impl RAMGatingWrapper {
    /// Create a new RAMGating model
    ///
    /// # Arguments
    /// * `num_clusters` - Number of clusters to gate (vocabulary size)
    /// * `neurons_per_gate` - Number of RAM neurons per cluster (default 8)
    /// * `bits_per_neuron` - Address bits per neuron (default 12)
    /// * `total_input_bits` - Total input bits (context_size * bits_per_token)
    /// * `threshold` - Fraction of neurons that must fire (default 0.5)
    /// * `seed` - Random seed for connectivity initialization
    #[new]
    #[pyo3(signature = (num_clusters, neurons_per_gate, bits_per_neuron, total_input_bits, threshold=0.5, seed=None))]
    fn new(
        num_clusters: usize,
        neurons_per_gate: usize,
        bits_per_neuron: usize,
        total_input_bits: usize,
        threshold: f32,
        seed: Option<u64>,
    ) -> Self {
        Self {
            inner: gating::RAMGating::new(
                num_clusters,
                neurons_per_gate,
                bits_per_neuron,
                total_input_bits,
                threshold,
                seed,
            ),
        }
    }

    /// Compute binary gates for a batch of inputs
    ///
    /// # Arguments
    /// * `input_bits_flat` - Flattened input bits [batch_size * total_input_bits]
    /// * `batch_size` - Number of examples in batch
    ///
    /// # Returns
    /// Flattened gate values [batch_size * num_clusters] (0.0 or 1.0)
    fn forward_batch<'py>(&self, py: Python<'py>, input_bits_flat: PyReadonlyArray1<'py, u8>, batch_size: usize) -> Vec<f32> {
        let total_bits = self.inner.config().total_input_bits;
        let input_slice = input_bits_flat.as_slice()
            .expect("input_bits_flat must be contiguous");
        let packed = packed_bits::PackedBits::from_bool_bytes(input_slice, total_bits);
        py.allow_threads(|| self.inner.forward_batch(&packed, batch_size))
    }

    /// Train gate neurons for a batch of examples
    ///
    /// # Arguments
    /// * `input_bits_flat` - Flattened input bits [batch_size * total_input_bits]
    /// * `target_gates_flat` - Flattened target gates [batch_size * num_clusters]
    /// * `batch_size` - Number of examples
    /// * `allow_override` - Whether to override non-EMPTY cells
    ///
    /// # Returns
    /// Total cells modified across batch
    fn train_batch<'py>(
        &self,
        py: Python<'py>,
        input_bits_flat: PyReadonlyArray1<'py, u8>,
        target_gates_flat: PyReadonlyArray1<'py, u8>,
        batch_size: usize,
        allow_override: bool,
    ) -> usize {
        let total_bits = self.inner.config().total_input_bits;
        let input_slice = input_bits_flat.as_slice()
            .expect("input_bits_flat must be contiguous");
        let packed = packed_bits::PackedBits::from_bool_bytes(input_slice, total_bits);
        let target_bools: Vec<bool> = target_gates_flat.as_slice()
            .expect("target_gates_flat must be contiguous")
            .iter().map(|&b| b != 0).collect();
        py.allow_threads(|| {
            self.inner.train_batch(&packed, &target_bools, batch_size, allow_override)
        })
    }

    /// Reset all memory cells to EMPTY
    fn reset(&self) {
        self.inner.reset();
    }

    /// Get statistics about memory usage
    ///
    /// # Returns
    /// (empty_count, false_count, true_count)
    fn memory_stats(&self) -> (usize, usize, usize) {
        self.inner.memory_stats()
    }

    /// Export memory state as bytes (for serialization)
    fn export_memory(&self) -> Vec<u8> {
        self.inner.export_memory()
    }

    /// Import memory state from bytes
    fn import_memory(&self, data: Vec<u8>) -> PyResult<()> {
        self.inner.import_memory(&data).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(e.to_string())
        })
    }

    /// Get total number of gate neurons
    fn total_neurons(&self) -> usize {
        self.inner.total_neurons()
    }

    /// Get gating configuration
    fn config(&self) -> (usize, usize, usize, usize, usize) {
        let cfg = self.inner.config();
        (cfg.num_clusters, cfg.neurons_per_gate, cfg.bits_per_neuron, cfg.total_input_bits, cfg.vote_threshold)
    }

    /// Compute binary gates on Metal GPU
    ///
    /// # Arguments
    /// * `input_bits_flat` - Flattened input bits [batch_size * total_input_bits]
    /// * `batch_size` - Number of examples in batch
    ///
    /// # Returns
    /// Flattened gate values [batch_size * num_clusters] (0.0 or 1.0)
    #[cfg(target_os = "macos")]
    fn forward_batch_metal<'py>(&self, py: Python<'py>, input_bits_flat: PyReadonlyArray1<'py, u8>, batch_size: usize) -> PyResult<Vec<f32>> {
        let config = self.inner.config();
        let input_slice = input_bits_flat.as_slice()
            .expect("input_bits_flat must be contiguous");
        let pb = packed_bits::PackedBits::from_bool_bytes(input_slice, config.total_input_bits);
        py.allow_threads(|| {
            let (packed, wpe) = neuron_memory::pack_packed_to_u64(&pb);
            let evaluator_lock = get_cached_metal_gating_evaluator()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            let guard = evaluator_lock.lock().unwrap();
            let evaluator = guard.as_ref()
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Metal gating evaluator not initialized"))?;
            evaluator.forward_batch(&self.inner, &packed, batch_size, wpe)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
        })
    }

    /// Compute binary gates with hybrid CPU+GPU
    ///
    /// Splits the batch between CPU (rayon) and GPU (Metal) for maximum throughput.
    ///
    /// # Arguments
    /// * `input_bits_flat` - Flattened input bits [batch_size * total_input_bits]
    /// * `batch_size` - Number of examples
    /// * `cpu_fraction` - Fraction of batch to process on CPU (0.0-1.0, default 0.3)
    ///
    /// # Returns
    /// Flattened gate values [batch_size * num_clusters] (0.0 or 1.0)
    #[cfg(target_os = "macos")]
    #[pyo3(signature = (input_bits_flat, batch_size, cpu_fraction=0.3))]
    fn forward_batch_hybrid<'py>(&self, py: Python<'py>, input_bits_flat: PyReadonlyArray1<'py, u8>, batch_size: usize, cpu_fraction: f32) -> PyResult<Vec<f32>> {
        let total_bits = self.inner.config().total_input_bits;
        let input_slice = input_bits_flat.as_slice()
            .expect("input_bits_flat must be contiguous");
        let pb = packed_bits::PackedBits::from_bool_bytes(input_slice, total_bits);
        py.allow_threads(|| {
            let evaluator_lock = get_cached_metal_gating_evaluator()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            let guard = evaluator_lock.lock().unwrap();
            let evaluator = guard.as_ref()
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Metal gating evaluator not initialized"))?;
            metal_gating::forward_batch_hybrid(&self.inner, evaluator, &pb, batch_size, cpu_fraction)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
        })
    }

    /// Train gate neurons on Metal GPU
    ///
    /// Uses Metal compute shaders with atomic memory writes for parallel training.
    /// After training, memory is synced back from GPU.
    ///
    /// # Arguments
    /// * `input_bits_flat` - Flattened input bits [batch_size * total_input_bits]
    /// * `target_gates_flat` - Flattened target gates [batch_size * num_clusters]
    /// * `batch_size` - Number of training examples
    ///
    /// # Returns
    /// Number of training examples processed (batch_size)
    #[cfg(target_os = "macos")]
    fn train_batch_metal<'py>(
        &self,
        py: Python<'py>,
        input_bits_flat: PyReadonlyArray1<'py, u8>,
        target_gates_flat: PyReadonlyArray1<'py, u8>,
        batch_size: usize,
    ) -> PyResult<usize> {
        let input_bools: Vec<bool> = input_bits_flat.as_slice()
            .expect("input_bits_flat must be contiguous")
            .iter().map(|&b| b != 0).collect();
        let target_bools: Vec<bool> = target_gates_flat.as_slice()
            .expect("target_gates_flat must be contiguous")
            .iter().map(|&b| b != 0).collect();
        py.allow_threads(|| {
            let config = self.inner.config();
            let (packed, wpe) = neuron_memory::pack_bools_to_u64(&input_bools, batch_size, config.total_input_bits);
            let evaluator_lock = get_cached_metal_gating_evaluator()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            let guard = evaluator_lock.lock().unwrap();
            let evaluator = guard.as_ref()
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Metal gating evaluator not initialized"))?;

            // Train on GPU and get updated memory
            let updated_memory = evaluator.train_batch(&self.inner, &packed, &target_bools, batch_size, wpe)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

            // Import the updated memory back into the gating model
            self.inner.import_memory(&updated_memory)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

            // Invalidate forward cache since memory changed
            metal_gating::invalidate_gating_cache();

            Ok(batch_size)
        })
    }
}

/// Check if Metal gating is available
#[pyfunction]
pub(crate) fn gating_metal_available() -> bool {
    metal_gating::MetalGatingEvaluator::is_available()
}

/// Compute target gates from target cluster indices
///
/// Creates a boolean vector where only the target cluster is true for each example.
/// Used for training gating from supervised targets.
///
/// # Arguments
/// * `targets` - [batch_size] target cluster indices
/// * `num_clusters` - Number of clusters (vocabulary size)
///
/// # Returns
/// Flattened target gates [batch_size * num_clusters]
#[pyfunction]
pub(crate) fn compute_target_gates<'py>(py: Python<'py>, targets: Vec<i64>, num_clusters: usize) -> pyo3::Bound<'py, numpy::PyArray1<u8>> {
    let gates = gating::compute_target_gates(&targets, num_clusters);
    let bytes: Vec<u8> = gates.iter().map(|&b| b as u8).collect();
    numpy::PyArray1::from_vec(py, bytes)
}
