//! Numpy-based RAMLM functions (zero-copy).
//!
//! Split out of lib.rs (D3, 11/06/2026).

use crate::*;

// =============================================================================
// NUMPY-BASED RAMLM FUNCTIONS (Zero-copy for maximum performance)
// =============================================================================

/// Batch forward pass using numpy arrays (FAST - zero-copy)
///
/// This is the optimized version that avoids Python list conversion overhead.
/// Accepts numpy arrays directly for near-zero-copy access.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ramlm_forward_batch_numpy<'py>(
    py: Python<'py>,
    input_bits: PyReadonlyArray1<'py, u8>,       // numpy bool/uint8 array
    connections: PyReadonlyArray1<'py, i64>,      // numpy int64 array
    memory_words: PyReadonlyArray1<'py, i64>,     // numpy int64 array
    num_examples: usize,
    total_input_bits: usize,
    num_neurons: usize,
    bits_per_neuron: usize,
    neurons_per_cluster: usize,
    num_clusters: usize,
    words_per_neuron: usize,
    empty_value: f32,
) -> PyResult<Vec<f32>> {
    // Extract data BEFORE allow_threads (numpy arrays aren't thread-safe)
    let input_slice = input_bits.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input array not contiguous: {}", e))
    })?;
    let conn_slice = connections.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Connections array not contiguous: {}", e))
    })?;
    let mem_slice = memory_words.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Memory array not contiguous: {}", e))
    })?;

    // Convert u8 to bool for input bits
    let input_bools: Vec<bool> = input_slice.iter().map(|&b| b != 0).collect();
    // Copy connections and memory for thread safety
    let conn_vec: Vec<i64> = conn_slice.to_vec();
    let mem_vec: Vec<i64> = mem_slice.to_vec();

    py.allow_threads(|| {
        let probs = ramlm::forward_batch(
            &input_bools,
            &conn_vec,
            &mem_vec,
            num_examples,
            total_input_bits,
            num_neurons,
            bits_per_neuron,
            neurons_per_cluster,
            num_clusters,
            words_per_neuron,
            empty_value,
        );
        Ok(probs)
    })
}

/// Batch forward pass using numpy arrays on Metal GPU (FAST - zero-copy)
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ramlm_forward_batch_metal_numpy<'py>(
    py: Python<'py>,
    input_bits: PyReadonlyArray1<'py, u8>,
    connections: PyReadonlyArray1<'py, i64>,
    memory_words: PyReadonlyArray1<'py, i64>,
    num_examples: usize,
    total_input_bits: usize,
    num_neurons: usize,
    bits_per_neuron: usize,
    neurons_per_cluster: usize,
    num_clusters: usize,
    words_per_neuron: usize,
    empty_value: f32,
) -> PyResult<Vec<f32>> {
    // Extract data BEFORE allow_threads
    let input_slice = input_bits.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input array not contiguous: {}", e))
    })?;
    let conn_slice = connections.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Connections array not contiguous: {}", e))
    })?;
    let mem_slice = memory_words.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Memory array not contiguous: {}", e))
    })?;

    // Convert u8 to bool for input bits
    let input_bools: Vec<bool> = input_slice.iter().map(|&b| b != 0).collect();
    let conn_vec: Vec<i64> = conn_slice.to_vec();
    let mem_vec: Vec<i64> = mem_slice.to_vec();

    py.allow_threads(|| {
        let evaluator = metal_ramlm::MetalRAMLMEvaluator::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

        let (packed_input, wpe) = ram_core::neuron_memory::pack_bools_to_u64(
            &input_bools, num_examples, total_input_bits
        );

        evaluator
            .forward_batch(
                &packed_input,
                &conn_vec,
                &mem_vec,
                num_examples,
                wpe,
                num_neurons,
                bits_per_neuron,
                neurons_per_cluster,
                num_clusters,
                words_per_neuron,
                ram_core::neuron_memory::MODE_TERNARY,
                empty_value,
            )
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    })
}

/// Batch forward pass using CACHED Metal evaluator (no recompilation)
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ramlm_forward_batch_metal_cached<'py>(
    py: Python<'py>,
    input_bits: PyReadonlyArray1<'py, u8>,
    connections: PyReadonlyArray1<'py, i64>,
    memory_words: PyReadonlyArray1<'py, i64>,
    num_examples: usize,
    total_input_bits: usize,
    num_neurons: usize,
    bits_per_neuron: usize,
    neurons_per_cluster: usize,
    num_clusters: usize,
    words_per_neuron: usize,
    empty_value: f32,
) -> PyResult<Vec<f32>> {
    // Extract data BEFORE allow_threads
    let input_slice = input_bits.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input array not contiguous: {}", e))
    })?;
    let conn_slice = connections.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Connections array not contiguous: {}", e))
    })?;
    let mem_slice = memory_words.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Memory array not contiguous: {}", e))
    })?;

    // Convert u8 to bool for input bits
    let input_bools: Vec<bool> = input_slice.iter().map(|&b| b != 0).collect();
    let conn_vec: Vec<i64> = conn_slice.to_vec();
    let mem_vec: Vec<i64> = mem_slice.to_vec();

    py.allow_threads(|| {
        // Get cached evaluator
        let guard = get_cached_metal_evaluator()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?
            .lock()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let evaluator = guard.as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Metal not available"))?;

        let (packed_input, wpe) = ram_core::neuron_memory::pack_bools_to_u64(
            &input_bools, num_examples, total_input_bits
        );

        evaluator
            .forward_batch(
                &packed_input,
                &conn_vec,
                &mem_vec,
                num_examples,
                wpe,
                num_neurons,
                bits_per_neuron,
                neurons_per_cluster,
                num_clusters,
                words_per_neuron,
                ram_core::neuron_memory::MODE_TERNARY,
                empty_value,
            )
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    })
}

/// Hybrid CPU+GPU forward pass (uses all 56 cores on M4 Max)
///
/// Splits work between CPU (16 cores via rayon) and GPU (40 cores via Metal).
/// Optimal for large batches where both can work in parallel.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ramlm_forward_batch_hybrid_numpy<'py>(
    py: Python<'py>,
    input_bits: PyReadonlyArray1<'py, u8>,
    connections: PyReadonlyArray1<'py, i64>,
    memory_words: PyReadonlyArray1<'py, i64>,
    num_examples: usize,
    total_input_bits: usize,
    num_neurons: usize,
    bits_per_neuron: usize,
    neurons_per_cluster: usize,
    num_clusters: usize,
    words_per_neuron: usize,
    empty_value: f32,
) -> PyResult<Vec<f32>> {
    // Extract data BEFORE allow_threads
    let input_slice = input_bits.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input array not contiguous: {}", e))
    })?;
    let conn_slice = connections.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Connections array not contiguous: {}", e))
    })?;
    let mem_slice = memory_words.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Memory array not contiguous: {}", e))
    })?;

    // Convert u8 to bool for input bits
    let input_bools: Vec<bool> = input_slice.iter().map(|&b| b != 0).collect();
    let conn_vec: Vec<i64> = conn_slice.to_vec();
    let mem_vec: Vec<i64> = mem_slice.to_vec();

    py.allow_threads(|| {
        // Split work: GPU gets 70%, CPU gets 30% (GPU is faster for this workload)
        let gpu_examples = (num_examples * 7) / 10;
        let cpu_examples = num_examples - gpu_examples;

        if gpu_examples == 0 || cpu_examples == 0 {
            // Fall back to single-backend for small batches
            if gpu_examples == 0 {
                return Ok(ramlm::forward_batch(
                    &input_bools,
                    &conn_vec,
                    &mem_vec,
                    num_examples,
                    total_input_bits,
                    num_neurons,
                    bits_per_neuron,
                    neurons_per_cluster,
                    num_clusters,
                    words_per_neuron,
                    empty_value,
                ));
            } else {
                let evaluator = metal_ramlm::MetalRAMLMEvaluator::new()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
                let (packed_input, wpe) = ram_core::neuron_memory::pack_bools_to_u64(
                    &input_bools, num_examples, total_input_bits
                );
                return evaluator
                    .forward_batch(
                        &packed_input,
                        &conn_vec,
                        &mem_vec,
                        num_examples,
                        wpe,
                        num_neurons,
                        bits_per_neuron,
                        neurons_per_cluster,
                        num_clusters,
                        words_per_neuron,
                        ram_core::neuron_memory::MODE_TERNARY,
                        empty_value,
                    )
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e));
            }
        }

        // Run GPU and CPU in parallel using std::thread
        let gpu_input: Vec<bool> = input_bools[..gpu_examples * total_input_bits].to_vec();
        let cpu_input: Vec<bool> = input_bools[gpu_examples * total_input_bits..].to_vec();
        let conn_vec_gpu = conn_vec.clone();
        let mem_vec_gpu = mem_vec.clone();

        let gpu_handle = std::thread::spawn(move || {
            let evaluator = metal_ramlm::MetalRAMLMEvaluator::new()?;
            let (packed_gpu_input, gpu_wpe) = ram_core::neuron_memory::pack_bools_to_u64(
                &gpu_input, gpu_examples, total_input_bits
            );
            evaluator.forward_batch(
                &packed_gpu_input,
                &conn_vec_gpu,
                &mem_vec_gpu,
                gpu_examples,
                gpu_wpe,
                num_neurons,
                bits_per_neuron,
                neurons_per_cluster,
                num_clusters,
                words_per_neuron,
                ram_core::neuron_memory::MODE_TERNARY,
                empty_value,
            )
        });

        // CPU processes remaining examples
        let cpu_probs = ramlm::forward_batch(
            &cpu_input,
            &conn_vec,
            &mem_vec,
            cpu_examples,
            total_input_bits,
            num_neurons,
            bits_per_neuron,
            neurons_per_cluster,
            num_clusters,
            words_per_neuron,
            empty_value,
        );

        // Wait for GPU and combine results
        let gpu_probs = gpu_handle
            .join()
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("GPU thread panicked"))?
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

        // Combine: GPU results first, then CPU results
        let mut all_probs = gpu_probs;
        all_probs.extend(cpu_probs);
        Ok(all_probs)
    })
}
