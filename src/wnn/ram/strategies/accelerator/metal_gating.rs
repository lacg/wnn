//! Metal GPU Accelerator for RAM-based Gating
//!
//! Uses Metal compute shaders to evaluate gating on Apple Silicon GPUs.
//! The M4 Max has 40 GPU cores which can process many (example, cluster)
//! pairs in parallel.
//!
//! This complements the rayon CPU parallelism for hybrid CPU+GPU evaluation.

use metal::*;
use std::mem;
use std::sync::atomic::Ordering;

use super::gating::RAMGating;

/// Thread-local buffer cache for gating evaluation
/// Avoids repeated Metal buffer allocations
struct GatingBufferCache {
    conn_buffer: Option<CachedBuffer>,
    memory_buffer: Option<CachedBuffer>,
    input_buffer: Option<CachedBuffer>,
    params_buffer: Option<CachedBuffer>,
    output_buffer: Option<CachedBuffer>,
}

struct CachedBuffer {
    buffer: Buffer,
    capacity_bytes: u64,
}

impl GatingBufferCache {
    fn new() -> Self {
        Self {
            conn_buffer: None,
            memory_buffer: None,
            input_buffer: None,
            params_buffer: None,
            output_buffer: None,
        }
    }
}

thread_local! {
    static GATING_BUFFER_CACHE: std::cell::RefCell<GatingBufferCache> =
        std::cell::RefCell::new(GatingBufferCache::new());
}

/// Parameters struct matching the Metal shader
#[repr(C)]
struct GatingParams {
    num_clusters: u32,
    neurons_per_gate: u32,
    bits_per_neuron: u32,
    total_input_bits: u32,
    vote_threshold: u32,
    address_space_size: u32,
    batch_size: u32,
    _padding: u32,
}

/// Metal-based Gating evaluator
pub struct MetalGatingEvaluator {
    device: Device,
    command_queue: CommandQueue,
    forward_pipeline: ComputePipelineState,
    forward_per_example_pipeline: ComputePipelineState,
    apply_gates_pipeline: ComputePipelineState,
}

impl MetalGatingEvaluator {
    /// Check if Metal is available
    pub fn is_available() -> bool {
        Device::system_default().is_some()
    }

    /// Create new Metal gating evaluator
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;
        let command_queue = device.new_command_queue();

        // Compile gating shader
        let shader_source = include_str!("shaders/gating.metal");
        let library = device
            .new_library_with_source(shader_source, &CompileOptions::new())
            .map_err(|e| format!("Failed to compile gating shader: {}", e))?;

        // Get kernel functions
        let forward_kernel = library
            .get_function("gating_forward", None)
            .map_err(|e| format!("Failed to get gating_forward kernel: {}", e))?;

        let forward_per_example_kernel = library
            .get_function("gating_forward_per_example", None)
            .map_err(|e| format!("Failed to get gating_forward_per_example kernel: {}", e))?;

        let apply_gates_kernel = library
            .get_function("apply_gates_gpu", None)
            .map_err(|e| format!("Failed to get apply_gates_gpu kernel: {}", e))?;

        // Create pipelines
        let forward_pipeline = device
            .new_compute_pipeline_state_with_function(&forward_kernel)
            .map_err(|e| format!("Failed to create forward pipeline: {}", e))?;

        let forward_per_example_pipeline = device
            .new_compute_pipeline_state_with_function(&forward_per_example_kernel)
            .map_err(|e| format!("Failed to create forward_per_example pipeline: {}", e))?;

        let apply_gates_pipeline = device
            .new_compute_pipeline_state_with_function(&apply_gates_kernel)
            .map_err(|e| format!("Failed to create apply_gates pipeline: {}", e))?;

        Ok(Self {
            device,
            command_queue,
            forward_pipeline,
            forward_per_example_pipeline,
            apply_gates_pipeline,
        })
    }

    /// Forward pass on GPU
    ///
    /// # Arguments
    /// * `gating` - The RAMGating model (provides memory and connections)
    /// * `input_bits_flat` - Flattened input bits [batch_size * total_input_bits]
    /// * `batch_size` - Number of examples in batch
    ///
    /// # Returns
    /// Flattened gate values [batch_size * num_clusters]
    pub fn forward_batch(
        &self,
        gating: &RAMGating,
        input_bits_flat: &[bool],
        batch_size: usize,
    ) -> Result<Vec<f32>, String> {
        if batch_size == 0 {
            return Ok(vec![]);
        }

        let config = gating.config();

        // Convert bools to u8 for GPU
        let input_bits_u8: Vec<u8> = input_bits_flat.iter().map(|&b| b as u8).collect();

        // Get connections as i32
        let connections: Vec<i32> = gating.get_connections().to_vec();

        // Export memory as u8
        let memory = gating.export_memory();

        // Prepare params
        let params = GatingParams {
            num_clusters: config.num_clusters as u32,
            neurons_per_gate: config.neurons_per_gate as u32,
            bits_per_neuron: config.bits_per_neuron as u32,
            total_input_bits: config.total_input_bits as u32,
            vote_threshold: config.vote_threshold as u32,
            address_space_size: (1 << config.bits_per_neuron) as u32,
            batch_size: batch_size as u32,
            _padding: 0,
        };

        // Create buffers
        let conn_buffer = self.device.new_buffer_with_data(
            connections.as_ptr() as *const _,
            (connections.len() * mem::size_of::<i32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let memory_buffer = self.device.new_buffer_with_data(
            memory.as_ptr() as *const _,
            (memory.len() * mem::size_of::<u8>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let input_buffer = self.device.new_buffer_with_data(
            input_bits_u8.as_ptr() as *const _,
            (input_bits_u8.len() * mem::size_of::<u8>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let params_buffer = self.device.new_buffer_with_data(
            &params as *const _ as *const _,
            mem::size_of::<GatingParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Output buffer
        let output_size = batch_size * config.num_clusters;
        let output_buffer = self.device.new_buffer(
            (output_size * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Choose kernel based on problem size
        // For large num_clusters (50K vocab), per-example is more memory-efficient
        let use_per_example = config.num_clusters > 1000;

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        if use_per_example {
            // One thread per example
            encoder.set_compute_pipeline_state(&self.forward_per_example_pipeline);
            encoder.set_buffer(0, Some(&conn_buffer), 0);
            encoder.set_buffer(1, Some(&memory_buffer), 0);
            encoder.set_buffer(2, Some(&input_buffer), 0);
            encoder.set_buffer(3, Some(&params_buffer), 0);
            encoder.set_buffer(4, Some(&output_buffer), 0);

            let grid_size = MTLSize::new(batch_size as u64, 1, 1);
            let max_threads = self.forward_per_example_pipeline.max_total_threads_per_threadgroup();
            let thread_group_size = MTLSize::new(max_threads.min(batch_size as u64), 1, 1);
            encoder.dispatch_threads(grid_size, thread_group_size);
        } else {
            // One thread per (cluster, example) pair
            encoder.set_compute_pipeline_state(&self.forward_pipeline);
            encoder.set_buffer(0, Some(&conn_buffer), 0);
            encoder.set_buffer(1, Some(&memory_buffer), 0);
            encoder.set_buffer(2, Some(&input_buffer), 0);
            encoder.set_buffer(3, Some(&params_buffer), 0);
            encoder.set_buffer(4, Some(&output_buffer), 0);

            let grid_size = MTLSize::new(config.num_clusters as u64, batch_size as u64, 1);
            let max_threads = self.forward_pipeline.max_total_threads_per_threadgroup();
            let threads_x = (max_threads as f64).sqrt() as u64;
            let threads_y = max_threads / threads_x;
            let thread_group_size = MTLSize::new(
                threads_x.min(config.num_clusters as u64),
                threads_y.min(batch_size as u64),
                1,
            );
            encoder.dispatch_threads(grid_size, thread_group_size);
        }

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read results
        let ptr = output_buffer.contents() as *const f32;
        let results = unsafe { std::slice::from_raw_parts(ptr, output_size) };
        Ok(results.to_vec())
    }

    /// Apply gates to scores on GPU
    pub fn apply_gates(
        &self,
        scores: &[f32],
        gates: &[f32],
    ) -> Result<Vec<f32>, String> {
        if scores.len() != gates.len() {
            return Err("scores and gates must have same length".to_string());
        }

        if scores.is_empty() {
            return Ok(vec![]);
        }

        let total_elements = scores.len() as u32;

        // Create buffers
        let scores_buffer = self.device.new_buffer_with_data(
            scores.as_ptr() as *const _,
            (scores.len() * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let gates_buffer = self.device.new_buffer_with_data(
            gates.as_ptr() as *const _,
            (gates.len() * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let output_buffer = self.device.new_buffer(
            (scores.len() * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let elements_buffer = self.device.new_buffer_with_data(
            &total_elements as *const _ as *const _,
            mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.apply_gates_pipeline);
        encoder.set_buffer(0, Some(&scores_buffer), 0);
        encoder.set_buffer(1, Some(&gates_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_buffer(3, Some(&elements_buffer), 0);

        let grid_size = MTLSize::new(scores.len() as u64, 1, 1);
        let max_threads = self.apply_gates_pipeline.max_total_threads_per_threadgroup();
        let thread_group_size = MTLSize::new(max_threads.min(scores.len() as u64), 1, 1);
        encoder.dispatch_threads(grid_size, thread_group_size);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read results
        let ptr = output_buffer.contents() as *const f32;
        let results = unsafe { std::slice::from_raw_parts(ptr, scores.len()) };
        Ok(results.to_vec())
    }
}

/// Hybrid CPU+GPU forward pass
///
/// Splits the batch between CPU (rayon) and GPU (Metal) for maximum throughput.
/// The split ratio is determined by the number of CPU cores vs GPU cores.
///
/// # Arguments
/// * `gating` - The RAMGating model
/// * `input_bits_flat` - Flattened input bits [batch_size * total_input_bits]
/// * `batch_size` - Number of examples
/// * `cpu_fraction` - Fraction of batch to process on CPU (0.0-1.0, default 0.3)
///
/// # Returns
/// Flattened gate values [batch_size * num_clusters]
pub fn forward_batch_hybrid(
    gating: &RAMGating,
    metal_eval: &MetalGatingEvaluator,
    input_bits_flat: &[bool],
    batch_size: usize,
    cpu_fraction: f32,
) -> Result<Vec<f32>, String> {
    use rayon::prelude::*;
    use std::sync::Arc;

    if batch_size == 0 {
        return Ok(vec![]);
    }

    let config = gating.config();
    let total_input_bits = config.total_input_bits;
    let num_clusters = config.num_clusters;

    // Determine split point
    let cpu_batch_size = ((batch_size as f32 * cpu_fraction) as usize).max(1).min(batch_size);
    let gpu_batch_size = batch_size - cpu_batch_size;

    // If GPU batch is empty, just use CPU
    if gpu_batch_size == 0 {
        return Ok(gating.forward_batch(input_bits_flat, batch_size));
    }

    // Split input
    let cpu_input_end = cpu_batch_size * total_input_bits;
    let cpu_input = &input_bits_flat[..cpu_input_end];
    let gpu_input = &input_bits_flat[cpu_input_end..];

    // Allocate output
    let mut results = vec![0.0f32; batch_size * num_clusters];

    // Run CPU and GPU in parallel using rayon's join
    let (cpu_results, gpu_results) = rayon::join(
        || gating.forward_batch(cpu_input, cpu_batch_size),
        || metal_eval.forward_batch(gating, gpu_input, gpu_batch_size),
    );

    // Check GPU result
    let gpu_results = gpu_results?;

    // Copy results
    let cpu_output_end = cpu_batch_size * num_clusters;
    results[..cpu_output_end].copy_from_slice(&cpu_results);
    results[cpu_output_end..].copy_from_slice(&gpu_results);

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_gating_available() {
        // Should work on M4 Max
        let available = MetalGatingEvaluator::is_available();
        println!("Metal gating available: {}", available);
    }

    #[test]
    fn test_metal_gating_forward() {
        if !MetalGatingEvaluator::is_available() {
            println!("Metal not available, skipping test");
            return;
        }

        let gating = RAMGating::new(10, 4, 6, 32, 0.5, Some(42));
        let input = vec![true; 32];

        // Train some gates
        let mut targets = vec![false; 10];
        targets[3] = true;
        targets[7] = true;
        gating.train_single(&input, &targets, false);

        // Test Metal forward
        let metal_eval = MetalGatingEvaluator::new().unwrap();
        let gates = metal_eval.forward_batch(&gating, &input, 1).unwrap();

        assert_eq!(gates.len(), 10);
        // Trained gates should be open
        assert_eq!(gates[3], 1.0);
        assert_eq!(gates[7], 1.0);
    }

    #[test]
    fn test_metal_cpu_equivalence() {
        if !MetalGatingEvaluator::is_available() {
            println!("Metal not available, skipping test");
            return;
        }

        let gating = RAMGating::new(100, 8, 10, 64, 0.5, Some(42));

        // Train with random patterns
        for i in 0..50 {
            let input: Vec<bool> = (0..64).map(|j| ((i * 7 + j) % 3) == 0).collect();
            let mut targets = vec![false; 100];
            targets[(i * 3) % 100] = true;
            targets[(i * 7) % 100] = true;
            gating.train_single(&input, &targets, false);
        }

        // Test batch
        let batch_size = 20;
        let input_flat: Vec<bool> = (0..batch_size * 64)
            .map(|i| ((i * 11) % 5) == 0)
            .collect();

        // CPU forward
        let cpu_gates = gating.forward_batch(&input_flat, batch_size);

        // GPU forward
        let metal_eval = MetalGatingEvaluator::new().unwrap();
        let gpu_gates = metal_eval.forward_batch(&gating, &input_flat, batch_size).unwrap();

        // Should be identical (both binary)
        assert_eq!(cpu_gates.len(), gpu_gates.len());
        for (i, (&cpu, &gpu)) in cpu_gates.iter().zip(gpu_gates.iter()).enumerate() {
            assert_eq!(
                cpu, gpu,
                "Mismatch at index {}: CPU={}, GPU={}",
                i, cpu, gpu
            );
        }
    }

    #[test]
    fn test_hybrid_forward() {
        if !MetalGatingEvaluator::is_available() {
            println!("Metal not available, skipping test");
            return;
        }

        let gating = RAMGating::new(50, 8, 10, 64, 0.5, Some(42));
        let metal_eval = MetalGatingEvaluator::new().unwrap();

        // Train some patterns
        for i in 0..30 {
            let input: Vec<bool> = (0..64).map(|j| ((i + j) % 4) == 0).collect();
            let mut targets = vec![false; 50];
            targets[(i * 2) % 50] = true;
            gating.train_single(&input, &targets, false);
        }

        // Test hybrid forward
        let batch_size = 100;
        let input_flat: Vec<bool> = (0..batch_size * 64)
            .map(|i| (i % 3) == 0)
            .collect();

        let hybrid_gates = forward_batch_hybrid(&gating, &metal_eval, &input_flat, batch_size, 0.3).unwrap();

        // Compare with pure CPU
        let cpu_gates = gating.forward_batch(&input_flat, batch_size);

        assert_eq!(hybrid_gates.len(), cpu_gates.len());
        for (i, (&hybrid, &cpu)) in hybrid_gates.iter().zip(cpu_gates.iter()).enumerate() {
            assert_eq!(
                hybrid, cpu,
                "Hybrid mismatch at index {}: hybrid={}, cpu={}",
                i, hybrid, cpu
            );
        }
    }
}
