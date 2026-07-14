//! Tiered sparse memory (variable bits-per-tier architectures).
//!
//! Split out of lib.rs (D3, 11/06/2026).

use crate::*;

// =============================================================================
// TIERED SPARSE MEMORY (for variable bits-per-tier architectures)
// =============================================================================

/// Python wrapper for TieredSparseMemory
/// Provides tiered sparse storage for architectures with different bits per tier
#[pyclass]
pub(crate) struct TieredSparseMemory {
    inner: Arc<ram_core::sparse_memory::TieredSparseMemory>,
    num_clusters: usize,
    tier_configs: Vec<(usize, usize, usize)>,  // (end_cluster, neurons_per_cluster, bits_per_neuron)
}

#[pymethods]
impl TieredSparseMemory {
    /// Create a new tiered sparse memory
    /// tier_configs: List of (end_cluster, neurons_per_cluster, bits_per_neuron) tuples
    ///               Tiers must be consecutive starting from 0
    #[new]
    fn new(tier_configs: Vec<(usize, usize, usize)>, num_clusters: usize) -> Self {
        Self {
            inner: Arc::new(ram_core::sparse_memory::TieredSparseMemory::new(&tier_configs, num_clusters)),
            num_clusters,
            tier_configs,
        }
    }

    /// Get total number of written cells across all tiers
    fn total_cells(&self) -> usize {
        self.inner.total_cells()
    }

    /// Reset all memory to empty
    fn reset(&self) {
        self.inner.reset();
    }

    /// Get number of clusters
    #[getter]
    fn num_clusters(&self) -> usize {
        self.num_clusters
    }

    /// Get tier configurations
    #[getter]
    fn tier_configs(&self) -> Vec<(usize, usize, usize)> {
        self.tier_configs.clone()
    }
}

/// Batch training for tiered sparse memory backend (parallel)
/// Memory stays in Rust - only returns count of modified cells
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn sparse_train_batch_tiered(
    py: Python<'_>,
    memory: &TieredSparseMemory,
    input_bits_flat: Vec<bool>,
    true_clusters: Vec<i64>,
    false_clusters_flat: Vec<i64>,
    connections_flat: Vec<i64>,
    num_examples: usize,
    total_input_bits: usize,
    num_negatives: usize,
) -> PyResult<usize> {
    py.allow_threads(|| {
        let modified = ram_core::sparse_memory::train_batch_tiered(
            &memory.inner,
            &input_bits_flat,
            &true_clusters,
            &false_clusters_flat,
            &connections_flat,
            num_examples,
            total_input_bits,
            num_negatives,
        );
        Ok(modified)
    })
}

/// Batch training for tiered sparse memory using NumPy arrays (FAST)
///
/// Uses numpy arrays to avoid Python list conversion overhead.
/// Typically 3-5x faster than sparse_train_batch_tiered for large batches.
///
/// Args:
///   input_bits: [num_examples * total_input_bits] u8 numpy array (0/1 values)
///   true_clusters: [num_examples] i64 numpy array of logical cluster indices
///   false_clusters_flat: [num_examples * num_negatives] i64 numpy array
///   connections_flat: [total_neurons * max_bits_per_neuron] i64 numpy array
///
/// Returns: number of memory cells modified
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn sparse_train_batch_tiered_numpy<'py>(
    py: Python<'py>,
    memory: &TieredSparseMemory,
    input_bits: PyReadonlyArray1<'py, u8>,
    true_clusters: PyReadonlyArray1<'py, i64>,
    false_clusters_flat: PyReadonlyArray1<'py, i64>,
    connections_flat: PyReadonlyArray1<'py, i64>,
    num_examples: usize,
    total_input_bits: usize,
    num_negatives: usize,
) -> PyResult<usize> {
    // Extract slices before allow_threads
    let input_slice = input_bits.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input array not contiguous: {}", e))
    })?;
    let true_slice = true_clusters.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("True clusters array not contiguous: {}", e))
    })?;
    let false_slice = false_clusters_flat.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("False clusters array not contiguous: {}", e))
    })?;
    let conn_slice = connections_flat.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Connections array not contiguous: {}", e))
    })?;

    // Convert to owned vecs (needed for Send across allow_threads)
    let input_bools: Vec<bool> = input_slice.iter().map(|&b| b != 0).collect();
    let true_vec: Vec<i64> = true_slice.to_vec();
    let false_vec: Vec<i64> = false_slice.to_vec();
    let conn_vec: Vec<i64> = conn_slice.to_vec();

    // Run training in parallel (releases GIL)
    py.allow_threads(|| {
        let modified = ram_core::sparse_memory::train_batch_tiered(
            &memory.inner,
            &input_bools,
            &true_vec,
            &false_vec,
            &conn_vec,
            num_examples,
            total_input_bits,
            num_negatives,
        );
        Ok(modified)
    })
}

/// Batch forward pass for tiered sparse memory backend (parallel)
/// Legacy version using Vec - prefer sparse_forward_batch_tiered_numpy for speed
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn sparse_forward_batch_tiered(
    py: Python<'_>,
    memory: &TieredSparseMemory,
    input_bits_flat: Vec<bool>,
    connections_flat: Vec<i64>,
    num_examples: usize,
    total_input_bits: usize,
    empty_value: f32,
) -> PyResult<Vec<f32>> {
    py.allow_threads(|| {
        let probs = ram_core::sparse_memory::forward_batch_tiered(
            &memory.inner,
            &input_bits_flat,
            &connections_flat,
            num_examples,
            total_input_bits,
            empty_value,
        );
        Ok(probs)
    })
}

/// Batch forward pass for tiered sparse memory using NumPy arrays (FAST)
///
/// Uses numpy arrays to avoid Python list conversion overhead.
/// Returns probabilities as a flat numpy array [num_examples * num_clusters].
///
/// Args:
///   input_bits: [num_examples * total_input_bits] u8 numpy array (0/1 values)
///   connections_flat: [total_neurons * max_bits_per_neuron] i64 numpy array
///
/// Returns: [num_examples * num_clusters] f32 numpy array of probabilities
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn sparse_forward_batch_tiered_numpy<'py>(
    py: Python<'py>,
    memory: &TieredSparseMemory,
    input_bits: PyReadonlyArray1<'py, u8>,
    connections_flat: PyReadonlyArray1<'py, i64>,
    num_examples: usize,
    total_input_bits: usize,
    empty_value: f32,
) -> PyResult<Py<numpy::PyArray1<f32>>> {
    // Extract slices before allow_threads
    let input_slice = input_bits.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input array not contiguous: {}", e))
    })?;
    let conn_slice = connections_flat.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Connections array not contiguous: {}", e))
    })?;

    // Convert u8 to bool
    let input_bools: Vec<bool> = input_slice.iter().map(|&b| b != 0).collect();
    let conn_vec: Vec<i64> = conn_slice.to_vec();

    // Run forward pass in parallel (releases GIL)
    let probs = py.allow_threads(|| {
        ram_core::sparse_memory::forward_batch_tiered(
            &memory.inner,
            &input_bools,
            &conn_vec,
            num_examples,
            total_input_bits,
            empty_value,
        )
    });

    // Convert to numpy array
    Ok(numpy::PyArray1::from_vec(py, probs).into())
}

/// Cached GPU export data for sparse forward pass via Metal.
///
/// Holds the exported sparse memory (keys, values, offsets, counts, cluster_infos)
/// and a Metal evaluator. This avoids re-exporting on every forward call.
///
/// Usage:
///   cache = sparse_export_for_gpu(memory)           # For TieredRAMClusterLayer
///   cache = sparse_export_groups_for_gpu(...)        # For AdaptiveClusteredRAM
///   probs = sparse_forward_metal_numpy(cache, ...)   # Shared forward
#[pyclass]
pub(crate) struct SparseGpuCache {
    keys: Vec<u64>,
    values: Vec<u8>,
    offsets: Vec<u32>,
    counts: Vec<u32>,
    cluster_infos: Vec<(u32, u32, u32, u32)>,
    num_clusters: usize,
    evaluator: ram_core::metal_sparse::MetalSparseEvaluator,
}

#[pymethods]
impl SparseGpuCache {
    /// Get number of clusters
    #[getter]
    fn num_clusters(&self) -> usize {
        self.num_clusters
    }

    /// Get total entries in sparse memory
    #[getter]
    fn total_entries(&self) -> usize {
        self.keys.len()
    }

    /// Get memory size in bytes (approximate)
    #[getter]
    fn memory_bytes(&self) -> usize {
        self.keys.len() * 8 + self.values.len() + self.offsets.len() * 4
            + self.counts.len() * 4 + self.cluster_infos.len() * 16
    }
}

/// Export a TieredSparseMemory to GPU cache for Metal forward pass.
///
/// For TieredRAMClusterLayer: exports single memory with all tiers.
#[pyfunction]
pub(crate) fn sparse_export_for_gpu(
    py: Python<'_>,
    memory: &TieredSparseMemory,
) -> PyResult<SparseGpuCache> {
    py.allow_threads(|| {
        let export = memory.inner.export_for_gpu_general();
        let evaluator = ram_core::metal_sparse::MetalSparseEvaluator::new()
            .map_err(|e| format!("Failed to create Metal evaluator: {}", e))?;

        Ok(SparseGpuCache {
            keys: export.keys,
            values: export.values,
            offsets: export.offsets,
            counts: export.counts,
            cluster_infos: export.cluster_infos,
            num_clusters: export.num_clusters,
            evaluator,
        })
    }).map_err(|e: String| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
}

/// Export multiple TieredSparseMemory groups to a single GPU cache.
///
/// For AdaptiveClusteredRAM: combines multiple group exports, mapping each
/// group's clusters to their correct global cluster positions.
///
/// Args:
///   memories: List of TieredSparseMemory objects (one per config group)
///   cluster_ids_per_group: List of cluster ID lists (mapping local → global)
///   num_clusters: Total number of clusters across all groups
#[pyfunction]
pub(crate) fn sparse_export_groups_for_gpu(
    py: Python<'_>,
    memories: Vec<pyo3::PyRef<'_, TieredSparseMemory>>,
    cluster_ids_per_group: Vec<Vec<usize>>,
    num_clusters: usize,
) -> PyResult<SparseGpuCache> {
    // Extract Arc references before releasing GIL (PyRef can't cross thread boundary)
    let inner_refs: Vec<Arc<ram_core::sparse_memory::TieredSparseMemory>> = memories.iter()
        .map(|m| Arc::clone(&m.inner))
        .collect();

    py.allow_threads(|| {
        let mut all_keys: Vec<u64> = Vec::new();
        let mut all_values: Vec<u8> = Vec::new();
        let mut all_offsets: Vec<u32> = Vec::new();
        let mut all_counts: Vec<u32> = Vec::new();
        let mut all_cluster_infos: Vec<(u32, u32, u32, u32)> = vec![(0, 0, 0, 0); num_clusters];

        let mut global_neuron_base: u32 = 0;
        let mut global_conn_offset: u32 = 0;

        for (group_idx, inner) in inner_refs.iter().enumerate() {
            let export = inner.export_for_gpu_general();
            let cluster_ids = &cluster_ids_per_group[group_idx];

            // Adjust offsets for global arrays
            let key_base = all_keys.len() as u32;
            for &off in &export.offsets {
                all_offsets.push(key_base + off);
            }
            all_counts.extend(&export.counts);
            all_keys.extend(&export.keys);
            all_values.extend(&export.values);

            // Map cluster_infos from local to global positions
            for (local_idx, &(neurons, bits, start_neuron, conn_offset)) in export.cluster_infos.iter().enumerate() {
                if local_idx < cluster_ids.len() {
                    let global_cluster = cluster_ids[local_idx];
                    all_cluster_infos[global_cluster] = (
                        neurons,
                        bits,
                        global_neuron_base + start_neuron,
                        global_conn_offset + conn_offset,
                    );
                }
            }

            // Update bases for next group
            let total_neurons: u32 = export.offsets.len() as u32;
            global_neuron_base += total_neurons;

            // Connection offset: sum of all neurons * bits across this group's tiers
            let group_conn_size: u32 = export.cluster_infos.iter()
                .map(|&(n, b, _, _)| n * b)
                .sum();
            global_conn_offset += group_conn_size;
        }

        let evaluator = ram_core::metal_sparse::MetalSparseEvaluator::new()
            .map_err(|e| format!("Failed to create Metal evaluator: {}", e))?;

        Ok(SparseGpuCache {
            keys: all_keys,
            values: all_values,
            offsets: all_offsets,
            counts: all_counts,
            cluster_infos: all_cluster_infos,
            num_clusters,
            evaluator,
        })
    }).map_err(|e: String| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
}

/// Run Metal GPU forward pass using cached export data.
///
/// Args:
///   cache: SparseGpuCache from sparse_export_for_gpu or sparse_export_groups_for_gpu
///   input_bits: [num_examples * total_input_bits] u8 numpy array
///   connections_flat: [total_connections] i64 numpy array
///   num_examples: batch size
///   total_input_bits: bits per example
///
/// Returns: [num_examples * num_clusters] f32 numpy array
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn sparse_forward_metal_numpy<'py>(
    py: Python<'py>,
    cache: &SparseGpuCache,
    input_bits: PyReadonlyArray1<'py, u8>,
    connections_flat: PyReadonlyArray1<'py, i64>,
    num_examples: usize,
    total_input_bits: usize,
    empty_value: f32,
) -> PyResult<Py<numpy::PyArray1<f32>>> {
    let input_slice = input_bits.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input array not contiguous: {}", e))
    })?;
    let conn_slice = connections_flat.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Connections array not contiguous: {}", e))
    })?;

    let input_bools: Vec<bool> = input_slice.iter().map(|&b| b != 0).collect();
    let conn_vec: Vec<i64> = conn_slice.to_vec();

    let probs = py.allow_threads(|| {
        let (packed_input, wpe) = ram_core::neuron_memory::pack_bools_to_u64(
            &input_bools, num_examples, total_input_bits
        );
        cache.evaluator.forward_batch_general(
            &packed_input,
            &conn_vec,
            &cache.keys,
            &cache.values,
            &cache.offsets,
            &cache.counts,
            &cache.cluster_infos,
            num_examples,
            wpe,
            cache.num_clusters,
            ram_core::neuron_memory::TERNARY,
            empty_value,
            0, // run_seed: TERNARY path, not QSR
        ).map_err(|e| format!("Metal forward failed: {}", e))
    }).map_err(|e: String| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

    Ok(numpy::PyArray1::from_vec(py, probs).into())
}

/// Option B B2 — parity test for the Metal marker-FSM training kernel.
///
/// Builds a small synthetic training scenario, trains via the CPU
/// MarkerHashTable path AND via the Metal GPU kernel, and asserts the
/// resulting (key, value) snapshots are identical.
#[cfg(target_os = "macos")]
#[pyfunction]
#[pyo3(signature = (num_neurons=128, num_examples=200, bits_per_neuron=16, total_input_bits=96, seed=42))]
pub(crate) fn run_marker_train_parity_test(
    num_neurons: usize,
    num_examples: usize,
    bits_per_neuron: usize,
    total_input_bits: usize,
    seed: u64,
) -> PyResult<Vec<(String, bool, String, f64, f64)>> {
    use atomic_hashtable::MarkerHashTable;
    use marker_train::{MarkerTrainer, NeuronTrainMeta, TrainParams};
    use metal::MTLResourceOptions;
    use rand::{Rng, SeedableRng};
    use rand::rngs::SmallRng;

    let mut rng = SmallRng::seed_from_u64(seed);

    let words_per_example = (total_input_bits + 63) / 64;

    // ---- Synthetic data ----
    // packed_input: each example's bits packed into u64 words
    let mut packed_input: Vec<u64> = vec![0; num_examples * words_per_example];
    for i in 0..(num_examples * words_per_example) {
        packed_input[i] = rng.gen();
        // Mask to total_input_bits
        if (i + 1) % words_per_example == 0 {
            let extra_bits = words_per_example * 64 - total_input_bits;
            if extra_bits > 0 {
                let mask = (1u64 << (64 - extra_bits)) - 1;
                packed_input[i] &= mask;
            }
        }
    }

    // connections: bits_per_neuron per neuron, indices into total_input_bits
    let mut connections: Vec<i32> = Vec::with_capacity(num_neurons * bits_per_neuron);
    for _ in 0..(num_neurons * bits_per_neuron) {
        connections.push(rng.gen_range(0..total_input_bits) as i32);
    }

    // train_targets: 80/20 binary labels
    let train_targets: Vec<i64> = (0..num_examples).map(|i| if i % 5 == 0 { 1 } else { 0 }).collect();

    // No negatives for single-cluster binary IDS
    let train_negatives: Vec<i64> = vec![0; 1];

    // class_weights — all 1s for parity test
    let class_weights: Vec<u32> = vec![1, 1];

    // Slot capacity per neuron — pre-sized for the test's worst case
    // (each example is a unique address; we want ≤ 75% load to keep probing
    // efficient). Next power of two above num_examples × 4/3, min 256.
    let target = (num_examples * 4 / 3).max(256);
    let slot_capacity: usize = target.next_power_of_two();
    let total_slots = num_neurons * slot_capacity;

    // Neuron metadata (slot offsets contiguous)
    let mut neuron_meta: Vec<NeuronTrainMeta> = Vec::with_capacity(num_neurons);
    for n in 0..num_neurons {
        neuron_meta.push(NeuronTrainMeta {
            bits: bits_per_neuron as u32,
            conn_offset: (n * bits_per_neuron) as u32,
            slot_offset: (n * slot_capacity) as u32,
            slot_capacity: slot_capacity as u32,
            cluster_idx: 0,
            _pad: 0,
        });
    }

    // ---- Allocate Metal buffers for GPU side ----
    let trainer = MarkerTrainer::new()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
    let device = trainer.device();

    let packed_bytes = (packed_input.len() * 8) as u64;
    let packed_buf = device.new_buffer_with_data(
        packed_input.as_ptr() as *const _,
        packed_bytes,
        MTLResourceOptions::StorageModeShared,
    );
    let conn_bytes = (connections.len() * 4) as u64;
    let conn_buf = device.new_buffer_with_data(
        connections.as_ptr() as *const _,
        conn_bytes,
        MTLResourceOptions::StorageModeShared,
    );
    let targets_bytes = (train_targets.len() * 8) as u64;
    let targets_buf = device.new_buffer_with_data(
        train_targets.as_ptr() as *const _,
        targets_bytes,
        MTLResourceOptions::StorageModeShared,
    );
    let negs_bytes = (train_negatives.len() * 8) as u64;
    let negs_buf = device.new_buffer_with_data(
        train_negatives.as_ptr() as *const _,
        negs_bytes,
        MTLResourceOptions::StorageModeShared,
    );

    // GPU side: one Metal-backed MarkerHashTable with `total_slots` capacity
    // (single flat buffer; per-neuron slot regions inside via offsets)
    let gpu_table = MarkerHashTable::new_metal(device, total_slots, 1);
    let (markers_buf, keys_buf, values_buf) = gpu_table.metal_buffers().unwrap();

    let params = TrainParams {
        num_examples: num_examples as u32,
        num_negatives: 0,
        num_neurons: num_neurons as u32,
        num_genomes: 1,  // single-genome dispatch in this test
        words_per_example: words_per_example as u32,
        num_classes: 2,
        memory_mode: 2,  // QUAD_WEIGHTED
        single_cluster: 1,
        normal_class: 0,
        conn_stride: (num_neurons * bits_per_neuron) as u32,
        neuron_sample_rate: 1.0,  // no sampling in this single-genome test
        rng_seed: 0,
        num_example_chunks: 1,  // parity test: no B10
        oi_mode: 0,
        example_offset: 0,
        examples_in_dispatch: 0,
        neuron_index_offset: 0,
    };

    let t_gpu_start = std::time::Instant::now();
    let gpu_kernel_ms = trainer.train(
        &packed_buf, &conn_buf, &neuron_meta, &targets_buf, &negs_buf,
        &class_weights, params,
        &markers_buf, &keys_buf, &values_buf,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
    let gpu_total_ms = t_gpu_start.elapsed().as_secs_f64() * 1000.0;
    let _ = gpu_kernel_ms;

    // ---- CPU reference: train an identical workload via MarkerHashTable
    // per neuron (Heap-backed) and merge into a single sorted snapshot ----
    let t_cpu_start = std::time::Instant::now();
    let cpu_tables: Vec<MarkerHashTable> = (0..num_neurons)
        .map(|_| MarkerHashTable::new(slot_capacity, 1))
        .collect();
    for ex_idx in 0..num_examples {
        let target = train_targets[ex_idx];
        let nudge_dir = target == 1;
        let ex_words = &packed_input[ex_idx * words_per_example..(ex_idx + 1) * words_per_example];
        for n in 0..num_neurons {
            let conn_start = n * bits_per_neuron;
            let mut addr: u64 = 0;
            for i in 0..bits_per_neuron {
                let c = connections[conn_start + i];
                if c < 0 { continue; }
                let cu = c as usize;
                let word_idx = cu / 64;
                let bit_idx = cu % 64;
                let bit = (ex_words[word_idx] >> bit_idx) & 1;
                addr |= bit << (bits_per_neuron - 1 - i);
            }
            cpu_tables[n].nudge(addr, nudge_dir);
        }
    }
    let cpu_ms = t_cpu_start.elapsed().as_secs_f64() * 1000.0;

    // ---- Build per-neuron sorted snapshots from both paths ----
    // GPU: walk the flat buffers' per-neuron sub-regions
    let markers_slice = unsafe {
        std::slice::from_raw_parts(markers_buf.contents() as *const u32, total_slots)
    };
    let keys_slice = unsafe {
        std::slice::from_raw_parts(keys_buf.contents() as *const u64, total_slots)
    };
    let values_slice = unsafe {
        std::slice::from_raw_parts(values_buf.contents() as *const u32, total_slots)
    };

    let mut results: Vec<(String, bool, String, f64, f64)> = Vec::new();
    let mut mismatches = 0usize;
    let mut total_keys_gpu = 0usize;
    let mut total_keys_cpu = 0usize;
    for n in 0..num_neurons {
        let off = n * slot_capacity;
        let mut gpu_entries: Vec<(u64, u8)> = (0..slot_capacity)
            .filter_map(|i| {
                if markers_slice[off + i] == 0xFFFFFFFF {
                    Some((keys_slice[off + i], (values_slice[off + i] & 0xFF) as u8))
                } else { None }
            })
            .collect();
        gpu_entries.sort_by_key(|(k, _)| *k);
        let cpu_entries: Vec<(u64, u8)> = cpu_tables[n].snapshot_sorted();
        total_keys_gpu += gpu_entries.len();
        total_keys_cpu += cpu_entries.len();
        if gpu_entries != cpu_entries {
            mismatches += 1;
        }
    }

    let ok = mismatches == 0 && total_keys_gpu == total_keys_cpu;
    let detail = format!(
        "neurons={}, mismatches={}, gpu_keys={}, cpu_keys={}",
        num_neurons, mismatches, total_keys_gpu, total_keys_cpu,
    );
    results.push((
        "marker_train_gpu_cpu_parity".into(),
        ok,
        detail,
        gpu_total_ms,
        cpu_ms,
    ));

    // B3a: Test export_per_neuron against direct per-neuron snapshot
    // (also confirms the export is consumable by the eval path).
    {
        let slot_offsets: Vec<u32> = (0..num_neurons as u32)
            .map(|n| n * slot_capacity as u32).collect();
        let slot_capacities: Vec<u32> = vec![slot_capacity as u32; num_neurons];
        let (keys, values, offsets, counts) =
            gpu_table.export_per_neuron(&slot_offsets, &slot_capacities, false);
        // Verify offsets+counts are consistent
        let mut consistent = true;
        for n in 0..num_neurons {
            let start = offsets[n] as usize;
            let cnt = counts[n] as usize;
            if start + cnt > keys.len() {
                consistent = false;
                break;
            }
            // Per-neuron must be sorted by key
            for i in 1..cnt {
                if keys[start + i - 1] > keys[start + i] {
                    consistent = false;
                    break;
                }
            }
            // Compare against direct per-neuron walk
            let mut direct: Vec<(u64, u8)> = (0..slot_capacity)
                .filter_map(|i| {
                    let slot = n * slot_capacity + i;
                    if markers_slice[slot] == 0xFFFFFFFF {
                        Some((keys_slice[slot], (values_slice[slot] & 0xFF) as u8))
                    } else { None }
                })
                .collect();
            direct.sort_by_key(|(k, _)| *k);
            let from_export: Vec<(u64, u8)> = (0..cnt)
                .map(|i| (keys[start + i], values[start + i])).collect();
            if direct != from_export {
                consistent = false;
                break;
            }
        }
        let detail = format!(
            "exported {} keys across {} neurons; per-neuron sorted={}",
            keys.len(), num_neurons, consistent
        );
        results.push(("export_per_neuron".into(), consistent, detail, 0.0, 0.0));
    }

    Ok(results)
}

/// Option B B4-batched — multi-genome Metal kernel parity test.
///
/// Builds `num_genomes` synthetic genomes (different random connections,
/// same shape), trains them all via a SINGLE batched Metal dispatch AND
/// individually via the existing CPU MarkerHashTable path. Validates
/// that the per-genome output is identical and reports GPU vs CPU
/// wall-time.
#[cfg(target_os = "macos")]
#[pyfunction]
#[pyo3(signature = (num_genomes=16, num_neurons=100, num_examples=5000, bits_per_neuron=48, total_input_bits=96, seed=42, neuron_sample_rate=1.0, rng_seed=0))]
pub(crate) fn run_marker_train_batched_parity_test(
    num_genomes: usize,
    num_neurons: usize,
    num_examples: usize,
    bits_per_neuron: usize,
    total_input_bits: usize,
    seed: u64,
    neuron_sample_rate: f32,
    rng_seed: u32,
) -> PyResult<Vec<(String, bool, String, f64, f64)>> {
    use atomic_hashtable::MarkerHashTable;
    use marker_train::{MarkerTrainer, NeuronTrainMeta, TrainParams};
    use metal::MTLResourceOptions;
    use rand::{Rng, SeedableRng};
    use rand::rngs::SmallRng;

    let mut rng = SmallRng::seed_from_u64(seed);
    let words_per_example = (total_input_bits + 63) / 64;
    let conn_per_genome = num_neurons * bits_per_neuron;

    // Shared input + targets
    let mut packed_input: Vec<u64> = vec![0; num_examples * words_per_example];
    for i in 0..(num_examples * words_per_example) { packed_input[i] = rng.gen(); }
    let train_targets: Vec<i64> = (0..num_examples).map(|i| if i % 5 == 0 { 1 } else { 0 }).collect();

    // Per-genome connections (different random seed pattern)
    let total_conns = num_genomes * conn_per_genome;
    let mut connections: Vec<i32> = Vec::with_capacity(total_conns);
    for _ in 0..total_conns {
        connections.push(rng.gen_range(0..total_input_bits) as i32);
    }

    // Slot capacity per neuron — pre-sized for worst case
    let target = (num_examples * 4 / 3).max(256);
    let slot_capacity_per_neuron: usize = target.next_power_of_two();
    let slots_per_genome = num_neurons * slot_capacity_per_neuron;
    let total_slots = num_genomes * slots_per_genome;

    // Per-(genome, neuron) metadata — slot_offset is global into the flat
    // buffer; conn_offset is genome-relative (kernel adds genome's base).
    let mut neuron_meta: Vec<NeuronTrainMeta> = Vec::with_capacity(num_genomes * num_neurons);
    for g in 0..num_genomes {
        for n in 0..num_neurons {
            neuron_meta.push(NeuronTrainMeta {
                bits: bits_per_neuron as u32,
                conn_offset: (n * bits_per_neuron) as u32,
                slot_offset: ((g * num_neurons + n) * slot_capacity_per_neuron) as u32,
                slot_capacity: slot_capacity_per_neuron as u32,
                cluster_idx: 0,
                _pad: 0,
            });
        }
    }

    let class_weights: Vec<u32> = vec![1, 1];
    let train_negatives: Vec<i64> = vec![0; 1];

    let trainer = MarkerTrainer::new()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
    let device = trainer.device();

    let packed_buf = device.new_buffer_with_data(
        packed_input.as_ptr() as *const _,
        (packed_input.len() * 8) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let conn_buf = device.new_buffer_with_data(
        connections.as_ptr() as *const _,
        (connections.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let targets_buf = device.new_buffer_with_data(
        train_targets.as_ptr() as *const _,
        (train_targets.len() * 8) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let negs_buf = device.new_buffer_with_data(
        train_negatives.as_ptr() as *const _,
        (train_negatives.len() * 8) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let gpu_table = MarkerHashTable::new_metal(device, total_slots, 1);
    let (markers_buf, keys_buf, values_buf) = gpu_table.metal_buffers().unwrap();

    let params = TrainParams {
        num_examples: num_examples as u32,
        num_negatives: 0,
        num_neurons: num_neurons as u32,
        num_genomes: num_genomes as u32,
        words_per_example: words_per_example as u32,
        num_classes: 2,
        memory_mode: 2,
        single_cluster: 1,
        normal_class: 0,
        conn_stride: conn_per_genome as u32,
        neuron_sample_rate,
        rng_seed,
        num_example_chunks: 1,  // parity test: no B10
        oi_mode: 0,
        example_offset: 0,
        examples_in_dispatch: 0,
        neuron_index_offset: 0,
    };

    let t_gpu_start = std::time::Instant::now();
    let gpu_kernel_ms = trainer.train(
        &packed_buf, &conn_buf, &neuron_meta, &targets_buf, &negs_buf,
        &class_weights, params,
        &markers_buf, &keys_buf, &values_buf,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
    let gpu_total_ms = t_gpu_start.elapsed().as_secs_f64() * 1000.0;
    let _ = gpu_kernel_ms;

    // Sampling skip — same xorshift as kernel + production CPU path.
    let should_skip = |neuron_idx: usize, ex_idx: usize| -> bool {
        if neuron_sample_rate >= 1.0 { return false; }
        let mut rng = rng_seed
            .wrapping_add((neuron_idx as u32).wrapping_mul(1000003))
            .wrapping_add((ex_idx as u32).wrapping_mul(2654435761));
        if rng == 0 { rng = 1; }
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        ((rng >> 8) as f32) / 16777216.0 >= neuron_sample_rate
    };

    // CPU reference: for each genome, build per-neuron MarkerHashTables,
    // train sequentially (with matching sampling decisions).
    let t_cpu_start = std::time::Instant::now();
    let mut cpu_per_genome: Vec<Vec<Vec<(u64, u8)>>> = Vec::with_capacity(num_genomes);
    for g in 0..num_genomes {
        let tables: Vec<MarkerHashTable> = (0..num_neurons)
            .map(|_| MarkerHashTable::new(slot_capacity_per_neuron, 1))
            .collect();
        let conn_base = g * conn_per_genome;
        for ex_idx in 0..num_examples {
            let target = train_targets[ex_idx];
            let nudge_dir = target == 1;
            let ex_words = &packed_input[ex_idx * words_per_example..(ex_idx + 1) * words_per_example];
            for n in 0..num_neurons {
                if should_skip(n, ex_idx) { continue; }
                let conn_start = conn_base + n * bits_per_neuron;
                let mut addr: u64 = 0;
                for i in 0..bits_per_neuron {
                    let c = connections[conn_start + i];
                    if c < 0 { continue; }
                    let cu = c as usize;
                    let bit = (ex_words[cu / 64] >> (cu % 64)) & 1;
                    addr |= bit << (bits_per_neuron - 1 - i);
                }
                tables[n].nudge(addr, nudge_dir);
            }
        }
        cpu_per_genome.push(tables.iter().map(|t| t.snapshot_sorted()).collect());
    }
    let cpu_ms = t_cpu_start.elapsed().as_secs_f64() * 1000.0;

    // GPU per-genome snapshots from the flat buffer
    let markers_slice = unsafe {
        std::slice::from_raw_parts(markers_buf.contents() as *const u32, total_slots)
    };
    let keys_slice = unsafe {
        std::slice::from_raw_parts(keys_buf.contents() as *const u64, total_slots)
    };
    let values_slice = unsafe {
        std::slice::from_raw_parts(values_buf.contents() as *const u32, total_slots)
    };

    let mut results: Vec<(String, bool, String, f64, f64)> = Vec::new();
    let mut total_mismatches = 0usize;
    let mut total_keys_gpu = 0usize;
    let mut total_keys_cpu = 0usize;
    for g in 0..num_genomes {
        for n in 0..num_neurons {
            let off = (g * num_neurons + n) * slot_capacity_per_neuron;
            let mut gpu_entries: Vec<(u64, u8)> = (0..slot_capacity_per_neuron)
                .filter_map(|i| {
                    if markers_slice[off + i] == 0xFFFFFFFF {
                        Some((keys_slice[off + i], (values_slice[off + i] & 0xFF) as u8))
                    } else { None }
                })
                .collect();
            gpu_entries.sort_by_key(|(k, _)| *k);
            let cpu_entries = &cpu_per_genome[g][n];
            total_keys_gpu += gpu_entries.len();
            total_keys_cpu += cpu_entries.len();
            if gpu_entries != *cpu_entries { total_mismatches += 1; }
        }
    }

    let ok = total_mismatches == 0 && total_keys_gpu == total_keys_cpu;
    let speedup = cpu_ms / gpu_total_ms;
    let detail = format!(
        "{} genomes × {} neurons; mismatches={}; gpu_keys={}, cpu_keys={}; speedup={:.2}x",
        num_genomes, num_neurons, total_mismatches, total_keys_gpu, total_keys_cpu, speedup,
    );
    results.push((
        "marker_train_batched_parity".into(),
        ok,
        detail,
        gpu_total_ms,
        cpu_ms,
    ));
    Ok(results)
}

/// Option B B5 — multi-cluster Metal kernel parity test.
///
/// Builds `num_genomes` synthetic genomes, each with K clusters of `neurons_per_cluster`
/// neurons. Targets are random class labels in [0, K). For each example, K-1
/// `train_negatives` are sampled (all other clusters). Trains via the batched
/// Metal kernel AND a CPU MarkerHashTable reference; validates exact per-neuron
/// snapshot match.
#[cfg(target_os = "macos")]
#[pyfunction]
#[pyo3(signature = (num_genomes=4, num_clusters=8, neurons_per_cluster=12, num_examples=2000, bits_per_neuron=24, total_input_bits=512, seed=42, neuron_sample_rate=1.0, rng_seed=0))]
pub(crate) fn run_marker_train_multicluster_parity_test(
    num_genomes: usize,
    num_clusters: usize,
    neurons_per_cluster: usize,
    num_examples: usize,
    bits_per_neuron: usize,
    total_input_bits: usize,
    seed: u64,
    neuron_sample_rate: f32,
    rng_seed: u32,
) -> PyResult<Vec<(String, bool, String, f64, f64)>> {
    use atomic_hashtable::MarkerHashTable;
    use marker_train::{MarkerTrainer, NeuronTrainMeta, TrainParams};
    use metal::MTLResourceOptions;
    use rand::{Rng, SeedableRng};
    use rand::rngs::SmallRng;

    let mut rng = SmallRng::seed_from_u64(seed);
    let num_neurons_per_genome = num_clusters * neurons_per_cluster;
    let words_per_example = (total_input_bits + 63) / 64;
    let conn_per_genome = num_neurons_per_genome * bits_per_neuron;
    let num_negatives = num_clusters.saturating_sub(1).min(8);  // up to 8 negatives

    // Shared inputs
    let mut packed_input: Vec<u64> = vec![0; num_examples * words_per_example];
    for i in 0..(num_examples * words_per_example) { packed_input[i] = rng.gen(); }
    let train_targets: Vec<i64> = (0..num_examples)
        .map(|_| rng.gen_range(0..num_clusters) as i64)
        .collect();
    // For each example, generate `num_negatives` distinct negative cluster ids
    // (all different from target). Layout: flat [num_examples * num_negatives].
    let mut train_negatives: Vec<i64> = Vec::with_capacity(num_examples * num_negatives);
    for ex in 0..num_examples {
        let target = train_targets[ex] as usize;
        let mut neg_pool: Vec<usize> = (0..num_clusters).filter(|&c| c != target).collect();
        // shuffle and take first num_negatives
        for i in (1..neg_pool.len()).rev() {
            let j = rng.gen_range(0..=i);
            neg_pool.swap(i, j);
        }
        for k in 0..num_negatives {
            train_negatives.push(neg_pool[k % neg_pool.len()] as i64);
        }
    }

    let total_conns = num_genomes * conn_per_genome;
    let mut connections: Vec<i32> = Vec::with_capacity(total_conns);
    for _ in 0..total_conns {
        connections.push(rng.gen_range(0..total_input_bits) as i32);
    }

    let target_cap = (num_examples * 4 / 3).max(256);
    let slot_capacity_per_neuron: usize = target_cap.next_power_of_two();
    let slots_per_genome = num_neurons_per_genome * slot_capacity_per_neuron;
    let total_slots = num_genomes * slots_per_genome;

    // Per-(genome, neuron) metadata. Neurons within a genome are grouped by
    // cluster: cluster c's neurons are at [c * neurons_per_cluster, (c+1) * neurons_per_cluster).
    let mut neuron_meta: Vec<NeuronTrainMeta> = Vec::with_capacity(num_genomes * num_neurons_per_genome);
    for g in 0..num_genomes {
        for c in 0..num_clusters {
            for n_local in 0..neurons_per_cluster {
                let n_global = c * neurons_per_cluster + n_local;
                neuron_meta.push(NeuronTrainMeta {
                    bits: bits_per_neuron as u32,
                    conn_offset: (n_global * bits_per_neuron) as u32,
                    slot_offset: ((g * num_neurons_per_genome + n_global) * slot_capacity_per_neuron) as u32,
                    slot_capacity: slot_capacity_per_neuron as u32,
                    cluster_idx: c as u32,
                    _pad: 0,
                });
            }
        }
    }

    let class_weights: Vec<u32> = vec![1; num_clusters];

    let trainer = MarkerTrainer::new()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
    let device = trainer.device();

    let packed_buf = device.new_buffer_with_data(
        packed_input.as_ptr() as *const _,
        (packed_input.len() * 8) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let conn_buf = device.new_buffer_with_data(
        connections.as_ptr() as *const _,
        (connections.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let targets_buf = device.new_buffer_with_data(
        train_targets.as_ptr() as *const _,
        (train_targets.len() * 8) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let negs_buf = device.new_buffer_with_data(
        train_negatives.as_ptr() as *const _,
        (train_negatives.len() * 8) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let gpu_table = MarkerHashTable::new_metal(device, total_slots, 1);
    let (markers_buf, keys_buf, values_buf) = gpu_table.metal_buffers().unwrap();

    let params = TrainParams {
        num_examples: num_examples as u32,
        num_negatives: num_negatives as u32,
        num_neurons: num_neurons_per_genome as u32,
        num_genomes: num_genomes as u32,
        words_per_example: words_per_example as u32,
        num_classes: num_clusters as u32,
        memory_mode: 2,
        single_cluster: 0,   // <-- multi-cluster path
        normal_class: 0,
        conn_stride: conn_per_genome as u32,
        neuron_sample_rate,
        rng_seed,
        num_example_chunks: 1,  // multi-cluster parity test: no B10
        oi_mode: 0,
        example_offset: 0,
        examples_in_dispatch: 0,
        neuron_index_offset: 0,
    };

    let t_gpu_start = std::time::Instant::now();
    let _ = trainer.train(
        &packed_buf, &conn_buf, &neuron_meta, &targets_buf, &negs_buf,
        &class_weights, params,
        &markers_buf, &keys_buf, &values_buf,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
    let gpu_total_ms = t_gpu_start.elapsed().as_secs_f64() * 1000.0;

    // Sampling skip — must match the Metal kernel's should_skip_sample()
    // and the production CPU path at adaptive.rs:2867-2880.
    let should_skip = |neuron_idx: usize, ex_idx: usize| -> bool {
        if neuron_sample_rate >= 1.0 { return false; }
        let mut rng = rng_seed
            .wrapping_add((neuron_idx as u32).wrapping_mul(1000003))
            .wrapping_add((ex_idx as u32).wrapping_mul(2654435761));
        if rng == 0 { rng = 1; }
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        ((rng >> 8) as f32) / 16777216.0 >= neuron_sample_rate
    };

    // CPU reference — mirror the GPU semantics exactly (incl. sampling).
    let t_cpu_start = std::time::Instant::now();
    let mut cpu_per_genome: Vec<Vec<Vec<(u64, u8)>>> = Vec::with_capacity(num_genomes);
    for g in 0..num_genomes {
        let tables: Vec<MarkerHashTable> = (0..num_neurons_per_genome)
            .map(|_| MarkerHashTable::new(slot_capacity_per_neuron, 1))
            .collect();
        let conn_base = g * conn_per_genome;
        for ex_idx in 0..num_examples {
            let target = train_targets[ex_idx] as usize;
            let ex_words = &packed_input[ex_idx * words_per_example..(ex_idx + 1) * words_per_example];

            // Positive: nudge TRUE for neurons in cluster `target`
            for n_local in 0..neurons_per_cluster {
                let n_global = target * neurons_per_cluster + n_local;
                if should_skip(n_global, ex_idx) { continue; }
                let conn_start = conn_base + n_global * bits_per_neuron;
                let mut addr: u64 = 0;
                for i in 0..bits_per_neuron {
                    let c = connections[conn_start + i];
                    if c < 0 { continue; }
                    let cu = c as usize;
                    let bit = (ex_words[cu / 64] >> (cu % 64)) & 1;
                    addr |= bit << (bits_per_neuron - 1 - i);
                }
                tables[n_global].nudge(addr, true);
            }

            // Negatives: nudge FALSE for neurons in each negative cluster
            // (mirror kernel's skip-if-equal-target rule)
            let neg_base = ex_idx * num_negatives;
            for k in 0..num_negatives {
                let false_cluster = train_negatives[neg_base + k] as usize;
                if false_cluster == target { continue; }
                for n_local in 0..neurons_per_cluster {
                    let n_global = false_cluster * neurons_per_cluster + n_local;
                    if should_skip(n_global, ex_idx) { continue; }
                    let conn_start = conn_base + n_global * bits_per_neuron;
                    let mut addr: u64 = 0;
                    for i in 0..bits_per_neuron {
                        let c = connections[conn_start + i];
                        if c < 0 { continue; }
                        let cu = c as usize;
                        let bit = (ex_words[cu / 64] >> (cu % 64)) & 1;
                        addr |= bit << (bits_per_neuron - 1 - i);
                    }
                    tables[n_global].nudge(addr, false);
                }
            }
        }
        cpu_per_genome.push(tables.iter().map(|t| t.snapshot_sorted()).collect());
    }
    let cpu_ms = t_cpu_start.elapsed().as_secs_f64() * 1000.0;

    // Compare per-neuron snapshots
    let markers_slice = unsafe { std::slice::from_raw_parts(markers_buf.contents() as *const u32, total_slots) };
    let keys_slice = unsafe { std::slice::from_raw_parts(keys_buf.contents() as *const u64, total_slots) };
    let values_slice = unsafe { std::slice::from_raw_parts(values_buf.contents() as *const u32, total_slots) };

    let mut results: Vec<(String, bool, String, f64, f64)> = Vec::new();
    let mut total_mismatches = 0usize;
    let mut total_keys_gpu = 0usize;
    let mut total_keys_cpu = 0usize;
    let mut first_mismatch: Option<String> = None;
    for g in 0..num_genomes {
        for n_global in 0..num_neurons_per_genome {
            let off = (g * num_neurons_per_genome + n_global) * slot_capacity_per_neuron;
            let mut gpu_entries: Vec<(u64, u8)> = (0..slot_capacity_per_neuron)
                .filter_map(|i| {
                    if markers_slice[off + i] == 0xFFFFFFFF {
                        Some((keys_slice[off + i], (values_slice[off + i] & 0xFF) as u8))
                    } else { None }
                })
                .collect();
            gpu_entries.sort_by_key(|(k, _)| *k);
            let cpu_entries = &cpu_per_genome[g][n_global];
            total_keys_gpu += gpu_entries.len();
            total_keys_cpu += cpu_entries.len();
            if gpu_entries != *cpu_entries {
                total_mismatches += 1;
                if first_mismatch.is_none() {
                    let cluster = n_global / neurons_per_cluster;
                    first_mismatch = Some(format!(
                        "genome={} neuron={} cluster={} gpu_n={} cpu_n={}",
                        g, n_global, cluster, gpu_entries.len(), cpu_entries.len()
                    ));
                }
            }
        }
    }

    let ok = total_mismatches == 0 && total_keys_gpu == total_keys_cpu;
    let speedup = cpu_ms / gpu_total_ms.max(0.001);
    let detail = format!(
        "{} genomes × {} clusters × {} neurons/cluster; mismatches={}; gpu_keys={}, cpu_keys={}; first_mismatch={:?}; speedup={:.2}x",
        num_genomes, num_clusters, neurons_per_cluster,
        total_mismatches, total_keys_gpu, total_keys_cpu, first_mismatch, speedup,
    );
    results.push(("marker_train_multicluster_parity".into(), ok, detail, gpu_total_ms, cpu_ms));
    Ok(results)
}

/// Run MarkerHashTable unit tests and return (name, passed, details) tuples.
/// Used to validate B0 from Python since cargo test --lib can't link with the
/// PyO3 extension-module feature.
#[pyfunction]
pub(crate) fn run_marker_hashtable_tests() -> PyResult<Vec<(String, bool, String)>> {
    use atomic_hashtable::{AtomicHashTable, MarkerHashTable};
    use rayon::prelude::*;
    let mut results: Vec<(String, bool, String)> = Vec::new();

    // basic_write_read
    {
        let t = MarkerHashTable::new(64, 2);
        let mut ok = true;
        let mut why = String::new();
        if !t.write(42, 1, false) { ok = false; why.push_str("write returned false; "); }
        if t.read(42) != 1 { ok = false; why.push_str(&format!("read(42)={} expected 1; ", t.read(42))); }
        if t.read(99) != 2 { ok = false; why.push_str(&format!("read(99)={} expected default 2; ", t.read(99))); }
        if t.len() != 1 { ok = false; why.push_str(&format!("len={} expected 1; ", t.len())); }
        results.push(("basic_write_read".into(), ok, why));
    }
    // true_wins_over_false
    {
        let t = MarkerHashTable::new(64, 2);
        let mut ok = true;
        let mut why = String::new();
        if !t.write(7, 1, false) { ok = false; why.push_str("initial write TRUE failed; "); }
        if t.write(7, 0, false) { ok = false; why.push_str("FALSE-over-TRUE was accepted; "); }
        if t.read(7) != 1 { ok = false; why.push_str(&format!("read(7)={} expected 1; ", t.read(7))); }
        results.push(("true_wins_over_false".into(), ok, why));
    }
    // nudge_clamping
    {
        let t = MarkerHashTable::new(64, 1);
        let mut ok = true;
        let mut why = String::new();
        t.nudge(11, true);
        if t.read(11) != 2 { ok = false; why.push_str(&format!("after 1 nudge_true: {} expected 2; ", t.read(11))); }
        t.nudge(11, true);
        if t.read(11) != 3 { ok = false; why.push_str(&format!("after 2 nudge_true: {} expected 3; ", t.read(11))); }
        if t.nudge(11, true) { ok = false; why.push_str("saturated nudge_true reported true; "); }
        if t.read(11) != 3 { ok = false; why.push_str(&format!("saturated value: {} expected 3; ", t.read(11))); }
        t.nudge(11, false);
        if t.read(11) != 2 { ok = false; why.push_str(&format!("after nudge_false: {} expected 2; ", t.read(11))); }
        results.push(("nudge_clamping".into(), ok, why));
    }
    // resize_under_load
    {
        let t = MarkerHashTable::new(16, 2);
        let mut ok = true;
        let mut why = String::new();
        for k in 0u64..1000 {
            if !t.write(k, 1, false) { ok = false; why.push_str(&format!("write {} failed; ", k)); break; }
        }
        if t.len() != 1000 { ok = false; why.push_str(&format!("len={} expected 1000; ", t.len())); }
        for k in 0u64..1000 {
            if t.read(k) != 1 { ok = false; why.push_str(&format!("read {}={} expected 1; ", k, t.read(k))); break; }
        }
        results.push(("resize_under_load".into(), ok, why));
    }
    // parallel_writes_distinct_keys
    {
        let t = MarkerHashTable::new(64, 2);
        (0u64..10_000).into_par_iter().for_each(|k| { t.write(k, 1, false); });
        let mut ok = t.len() == 10_000;
        let mut why = if ok { String::new() } else { format!("len={} expected 10000; ", t.len()) };
        for k in 0u64..10_000 {
            if t.read(k) != 1 { ok = false; why.push_str(&format!("missing {}; ", k)); break; }
        }
        results.push(("parallel_writes_distinct_keys".into(), ok, why));
    }
    // parallel_nudges_same_keys (the contention stress test)
    {
        let t = MarkerHashTable::new(64, 1);
        let work: Vec<u64> = (0u64..10).cycle().take(10_000).collect();
        work.par_iter().for_each(|&k| { t.nudge(k, true); });
        let mut ok = true;
        let mut why = String::new();
        for k in 0u64..10 {
            if t.read(k) != 3 { ok = false; why.push_str(&format!("key {} read={} expected 3; ", k, t.read(k))); }
        }
        results.push(("parallel_nudges_same_keys".into(), ok, why));
    }
    // snapshot_sorted
    {
        let t = MarkerHashTable::new(64, 2);
        for (k, v) in &[(100u64, 1u8), (50, 0), (75, 1), (25, 0)] {
            t.write(*k, *v, true);
        }
        let snap = t.snapshot_sorted();
        let expected = vec![(25u64, 0u8), (50, 0), (75, 1), (100, 1)];
        let ok = snap == expected;
        let why = if ok { String::new() } else { format!("snap={:?} expected {:?}", snap, expected) };
        results.push(("snapshot_sorted".into(), ok, why));
    }
    // parity_with_atomic_random_workload
    {
        let a = AtomicHashTable::new(64, 1);
        let m = MarkerHashTable::new(64, 1);
        let mut state: u64 = 0xC0FFEE;
        for _ in 0..5000 {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            let key = state & 0xFFFFFFFFFFFF;
            let nudge_true = (state >> 32) & 1 == 1;
            a.nudge(key, nudge_true);
            m.nudge(key, nudge_true);
        }
        let snap_a = a.snapshot_sorted();
        let snap_m = m.snapshot_sorted();
        let ok = snap_a == snap_m;
        let why = if ok {
            format!("both produced {} entries", snap_a.len())
        } else {
            format!("DIVERGED: atomic {} vs marker {} entries", snap_a.len(), snap_m.len())
        };
        results.push(("parity_with_atomic_random_workload".into(), ok, why));
    }

    // B1: Metal-backed MarkerHashTable parity vs Heap-backed
    #[cfg(target_os = "macos")]
    {
        let device = metal::Device::system_default();
        if device.is_none() {
            results.push(("metal_backed_parity".into(), false,
                "no Metal device available; skipping".into()));
        } else {
            let device = device.unwrap();
            // Capacity sized generously so no resize is needed (Metal-backed
            // doesn't support live resize). 5000 distinct keys at 25% load:
            let metal_table = MarkerHashTable::new_metal(&device, 32768, 1);
            let heap_table = MarkerHashTable::new(32768, 1);
            let mut state: u64 = 0xC0FFEE;
            for _ in 0..5000 {
                state ^= state << 13; state ^= state >> 7; state ^= state << 17;
                let key = state & 0xFFFFFFFFFFFF;
                let nudge_true = (state >> 32) & 1 == 1;
                metal_table.nudge(key, nudge_true);
                heap_table.nudge(key, nudge_true);
            }
            let snap_metal = metal_table.snapshot_sorted();
            let snap_heap = heap_table.snapshot_sorted();
            let ok = snap_metal == snap_heap;
            let buffers = metal_table.metal_buffers();
            let why = if ok {
                format!(
                    "metal & heap MarkerHashTable identical ({} entries); metal_buffers={}",
                    snap_metal.len(),
                    if buffers.is_some() { "Some(markers,keys,values)" } else { "None" }
                )
            } else {
                format!("DIVERGED: metal {} vs heap {} entries", snap_metal.len(), snap_heap.len())
            };
            results.push(("metal_backed_parity".into(), ok, why));
        }
    }

    // B1: Metal-backed parallel writes — stresses the CAS-coherent-within-CPU
    // case where the underlying memory is a Metal shared buffer (no GPU
    // involvement yet).
    #[cfg(target_os = "macos")]
    {
        let device = metal::Device::system_default();
        if let Some(device) = device {
            let t = MarkerHashTable::new_metal(&device, 32768, 1);
            (0u64..10_000).into_par_iter().for_each(|k| { t.write(k, 1, false); });
            let mut ok = t.len() == 10_000;
            let mut why = if ok { String::new() } else { format!("len={} expected 10000; ", t.len()) };
            for k in 0u64..10_000 {
                if t.read(k) != 1 { ok = false; why.push_str(&format!("missing {}; ", k)); break; }
            }
            results.push(("metal_backed_parallel_writes".into(), ok, why));
        }
    }

    Ok(results)
}

/// Atomic CAS microbenchmark for Option C foundation validation.
///
/// Returns a list of (test_name, passed, expected, observed, details, elapsed_ms)
/// tuples. The three tests validate:
///   1. GPU-only atomic CAS produces correct count (sanity)
///   2. Concurrent CPU+GPU CAS on a shared buffer is atomic (the key test)
///   3. CAS-claim-from-EMPTY with contention yields exactly one winner per slot
///
/// On non-macOS platforms returns an error (Metal not available).
#[pyfunction]
#[pyo3(signature = (num_slots=256, gpu_threads=4096, cpu_threads=64, iterations=1000))]
pub(crate) fn run_atomic_cas_microbench(
    num_slots: usize,
    gpu_threads: usize,
    cpu_threads: usize,
    iterations: usize,
) -> PyResult<Vec<(String, bool, u64, u64, String, f64)>> {
    #[cfg(target_os = "macos")]
    {
        let results = metal_atomic_test::run_microbench(
            num_slots, gpu_threads, cpu_threads, iterations,
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        Ok(results.into_iter().map(|r| (
            r.test_name, r.passed, r.expected, r.observed, r.details, r.elapsed_ms,
        )).collect())
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (num_slots, gpu_threads, cpu_threads, iterations);
        Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "Atomic CAS microbench requires Metal (macOS).",
        ))
    }
}
