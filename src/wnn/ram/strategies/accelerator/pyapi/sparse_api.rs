//! Sparse memory backend (>10 bits per neuron).
//!
//! Split out of lib.rs (D3, 11/06/2026).

use crate::*;

// =============================================================================
// SPARSE MEMORY BACKEND (for >10 bits per neuron)
// =============================================================================

/// Python wrapper for SparseLayerMemory
/// Provides HashMap-based sparse storage for neurons with >10 bits
#[pyclass]
pub(crate) struct SparseMemory {
    inner: Arc<ram_core::sparse_memory::SparseLayerMemory>,
    num_neurons: usize,
    bits_per_neuron: usize,
}

#[pymethods]
impl SparseMemory {
    /// Create a new sparse memory layer
    #[new]
    fn new(num_neurons: usize, bits_per_neuron: usize) -> Self {
        Self {
            inner: Arc::new(ram_core::sparse_memory::SparseLayerMemory::new(num_neurons, bits_per_neuron)),
            num_neurons,
            bits_per_neuron,
        }
    }

    /// Read a single cell value (returns 0=FALSE, 1=TRUE, 2=EMPTY)
    fn read_cell(&self, neuron_idx: usize, address: u64) -> u8 {
        self.inner.read_cell(neuron_idx, address)
    }

    /// Write a single cell value
    /// Returns True if the cell was modified
    fn write_cell(&self, neuron_idx: usize, address: u64, value: u8, allow_override: bool) -> bool {
        self.inner.write_cell(neuron_idx, address, value, allow_override)
    }

    /// Get total number of written cells across all neurons
    fn total_cells(&self) -> usize {
        self.inner.total_cells()
    }

    /// Get per-neuron cell counts
    fn cell_counts(&self) -> Vec<usize> {
        self.inner.cell_counts()
    }

    /// Export to list of (neuron_idx, address, value) tuples
    fn export(&self) -> Vec<(usize, u64, u8)> {
        self.inner.export()
    }

    /// Import from list of (neuron_idx, address, value) tuples
    fn import_cells(&self, cells: Vec<(usize, u64, u8)>) {
        self.inner.import(&cells);
    }

    /// Reset all memory to empty
    fn reset(&self) {
        self.inner.reset();
    }

    /// Get number of neurons
    #[getter]
    fn num_neurons(&self) -> usize {
        self.num_neurons
    }

    /// Get bits per neuron
    #[getter]
    fn bits_per_neuron(&self) -> usize {
        self.bits_per_neuron
    }

    /// Memory size if this were dense (for comparison)
    #[getter]
    fn dense_memory_size(&self) -> u64 {
        1u64 << self.bits_per_neuron
    }
}

/// Batch training for sparse memory backend (parallel)
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn sparse_train_batch(
    py: Python<'_>,
    memory: &SparseMemory,
    input_bits_flat: Vec<bool>,
    true_clusters: Vec<i64>,
    false_clusters_flat: Vec<i64>,
    connections_flat: Vec<i64>,
    num_examples: usize,
    total_input_bits: usize,
    neurons_per_cluster: usize,
    num_negatives: usize,
    allow_override: bool,
) -> PyResult<usize> {
    py.allow_threads(|| {
        let modified = ram_core::sparse_memory::train_batch_sparse(
            &memory.inner,
            &input_bits_flat,
            &true_clusters,
            &false_clusters_flat,
            &connections_flat,
            num_examples,
            total_input_bits,
            memory.bits_per_neuron,
            neurons_per_cluster,
            num_negatives,
            allow_override,
        );
        Ok(modified)
    })
}

/// Bitwise batch training for sparse memory backend
///
/// Multi-label training: each example trains ALL clusters (one per output bit).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn sparse_bitwise_train_batch<'py>(
    py: Python<'py>,
    memory: &SparseMemory,
    input_bits_flat: PyReadonlyArray1<'py, u8>,
    target_bits_flat: PyReadonlyArray1<'py, u8>,
    connections_flat: PyReadonlyArray1<'py, i64>,
    num_examples: usize,
    total_input_bits: usize,
    neurons_per_cluster: usize,
    num_clusters: usize,
    allow_override: bool,
) -> PyResult<usize> {
    let target_bits_flat: Vec<u8> = target_bits_flat.as_slice()
        .expect("target_bits_flat must be contiguous").to_vec();
    let connections_flat: Vec<i64> = connections_flat.as_slice()
        .expect("connections_flat must be contiguous").to_vec();
    let input_bools: Vec<bool> = input_bits_flat.as_slice()
        .expect("input_bits_flat must be contiguous")
        .iter().map(|&b| b != 0).collect();
    py.allow_threads(|| {
        let modified = ram_core::sparse_memory::bitwise_train_batch_sparse(
            &memory.inner,
            &input_bools,
            &target_bits_flat,
            &connections_flat,
            num_examples,
            total_input_bits,
            memory.bits_per_neuron,
            neurons_per_cluster,
            num_clusters,
            allow_override,
        );
        Ok(modified)
    })
}

/// Batch forward pass for sparse memory backend (parallel)
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn sparse_forward_batch(
    py: Python<'_>,
    memory: &SparseMemory,
    input_bits_flat: Vec<bool>,
    connections_flat: Vec<i64>,
    num_examples: usize,
    total_input_bits: usize,
    neurons_per_cluster: usize,
    num_clusters: usize,
    empty_value: f32,
) -> PyResult<Vec<f32>> {
    py.allow_threads(|| {
        let probs = ram_core::sparse_memory::forward_batch_sparse(
            &memory.inner,
            &input_bits_flat,
            &connections_flat,
            num_examples,
            total_input_bits,
            memory.bits_per_neuron,
            neurons_per_cluster,
            num_clusters,
            empty_value,
        );
        Ok(probs)
    })
}
