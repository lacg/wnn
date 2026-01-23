//! Metal GPU Accelerator for RAMLM Forward Pass
//!
//! Uses Metal compute shaders to evaluate RAMLM on Apple Silicon GPUs.
//! The M4 Max has 40 GPU cores which can process thousands of (example, cluster)
//! pairs in parallel.
//!
//! This is particularly effective for evaluation where we need to compute
//! probabilities over the full 50K vocabulary for each example.

use metal::*;
use std::mem;

/// Metal-based RAMLM evaluator
pub struct MetalRAMLMEvaluator {
    device: Device,
    command_queue: CommandQueue,
    forward_pipeline: ComputePipelineState,
    forward_per_example_pipeline: ComputePipelineState,
}

impl MetalRAMLMEvaluator {
    /// Check if Metal is available
    pub fn is_available() -> bool {
        Device::system_default().is_some()
    }

    /// Get device info
    pub fn device_info() -> Result<String, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;
        Ok(format!(
            "Device: {}\nUnified Memory: {}\nMax Threads/Group: {}",
            device.name(),
            device.has_unified_memory(),
            device.max_threads_per_threadgroup().width
        ))
    }

    /// Create new Metal RAMLM evaluator
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;
        let command_queue = device.new_command_queue();

        // Compile Metal shader
        let shader_source = include_str!("shaders/ramlm.metal");
        let library = device
            .new_library_with_source(shader_source, &CompileOptions::new())
            .map_err(|e| format!("Failed to compile RAMLM shader: {}", e))?;

        // Get kernel functions
        let forward_kernel = library
            .get_function("ramlm_forward_pass", None)
            .map_err(|e| format!("Failed to get forward kernel: {}", e))?;

        let forward_per_example_kernel = library
            .get_function("ramlm_forward_pass_per_example", None)
            .map_err(|e| format!("Failed to get forward_per_example kernel: {}", e))?;

        // Create pipelines
        let forward_pipeline = device
            .new_compute_pipeline_state_with_function(&forward_kernel)
            .map_err(|e| format!("Failed to create forward pipeline: {}", e))?;

        let forward_per_example_pipeline = device
            .new_compute_pipeline_state_with_function(&forward_per_example_kernel)
            .map_err(|e| format!("Failed to create forward_per_example pipeline: {}", e))?;

        Ok(Self {
            device,
            command_queue,
            forward_pipeline,
            forward_per_example_pipeline,
        })
    }

    /// Batch forward pass on GPU
    ///
    /// Args:
    ///   input_bits_flat: [num_examples * total_input_bits] bool array
    ///   connections_flat: [num_neurons * bits_per_neuron] connection indices
    ///   memory_words: [num_neurons * words_per_neuron] packed memory
    ///   ... dimension parameters ...
    ///
    /// Returns: [num_examples * num_clusters] probabilities
    pub fn forward_batch(
        &self,
        input_bits_flat: &[bool],
        connections_flat: &[i64],
        memory_words: &[i64],
        num_examples: usize,
        total_input_bits: usize,
        num_neurons: usize,
        bits_per_neuron: usize,
        neurons_per_cluster: usize,
        num_clusters: usize,
        words_per_neuron: usize,
    ) -> Result<Vec<f32>, String> {
        if num_examples == 0 {
            return Ok(vec![]);
        }

        // Convert bools to u8 for GPU
        let input_bits_u8: Vec<u8> = input_bits_flat.iter().map(|&b| b as u8).collect();

        // Convert i64 connections to i32 for GPU
        let connections_i32: Vec<i32> = connections_flat.iter().map(|&c| c as i32).collect();

        // Parameters struct (must match shader)
        #[repr(C)]
        struct RAMLMParams {
            num_examples: u32,
            total_input_bits: u32,
            num_neurons: u32,
            bits_per_neuron: u32,
            neurons_per_cluster: u32,
            num_clusters: u32,
            words_per_neuron: u32,
            empty_value: f32,
        }

        let params = RAMLMParams {
            num_examples: num_examples as u32,
            total_input_bits: total_input_bits as u32,
            num_neurons: num_neurons as u32,
            bits_per_neuron: bits_per_neuron as u32,
            neurons_per_cluster: neurons_per_cluster as u32,
            num_clusters: num_clusters as u32,
            words_per_neuron: words_per_neuron as u32,
            empty_value: crate::ramlm::get_empty_value(),
        };

        // Create buffers
        let input_buffer = self.device.new_buffer_with_data(
            input_bits_u8.as_ptr() as *const _,
            (input_bits_u8.len() * mem::size_of::<u8>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let conn_buffer = self.device.new_buffer_with_data(
            connections_i32.as_ptr() as *const _,
            (connections_i32.len() * mem::size_of::<i32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let memory_buffer = self.device.new_buffer_with_data(
            memory_words.as_ptr() as *const _,
            (memory_words.len() * mem::size_of::<i64>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let params_buffer = self.device.new_buffer_with_data(
            &params as *const _ as *const _,
            mem::size_of::<RAMLMParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Output buffer
        let output_size = num_examples * num_clusters;
        let output_buffer = self.device.new_buffer(
            (output_size * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Choose kernel based on problem size
        // For large num_clusters (50K vocab), per-example is more memory-efficient
        let use_per_example = num_clusters > 1000;

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        if use_per_example {
            // One thread per example
            encoder.set_compute_pipeline_state(&self.forward_per_example_pipeline);
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&conn_buffer), 0);
            encoder.set_buffer(2, Some(&memory_buffer), 0);
            encoder.set_buffer(3, Some(&params_buffer), 0);
            encoder.set_buffer(4, Some(&output_buffer), 0);

            let grid_size = MTLSize::new(num_examples as u64, 1, 1);
            let max_threads = self.forward_per_example_pipeline.max_total_threads_per_threadgroup();
            let thread_group_size = MTLSize::new(max_threads.min(num_examples as u64), 1, 1);
            encoder.dispatch_threads(grid_size, thread_group_size);
        } else {
            // One thread per (cluster, example) pair
            encoder.set_compute_pipeline_state(&self.forward_pipeline);
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&conn_buffer), 0);
            encoder.set_buffer(2, Some(&memory_buffer), 0);
            encoder.set_buffer(3, Some(&params_buffer), 0);
            encoder.set_buffer(4, Some(&output_buffer), 0);

            let grid_size = MTLSize::new(num_clusters as u64, num_examples as u64, 1);
            let thread_group_size = MTLSize::new(
                32.min(num_clusters as u64),
                8.min(num_examples as u64),
                1,
            );
            encoder.dispatch_threads(grid_size, thread_group_size);
        }

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read results
        let result_ptr = output_buffer.contents() as *const f32;
        let results: Vec<f32> = unsafe {
            std::slice::from_raw_parts(result_ptr, output_size).to_vec()
        };

        Ok(results)
    }
}

// =============================================================================
// SPARSE METAL EVALUATOR
// =============================================================================

/// Metal-based sparse RAMLM evaluator using binary search
/// Works with high-bit architectures (11-30+ bits) that can't use dense storage
pub struct MetalSparseEvaluator {
    device: Device,
    command_queue: CommandQueue,
    sparse_forward_pipeline: ComputePipelineState,
    sparse_forward_per_example_pipeline: ComputePipelineState,
    tiered_forward_pipeline: ComputePipelineState,
}

impl MetalSparseEvaluator {
    /// Create new sparse Metal evaluator
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;
        let command_queue = device.new_command_queue();

        // Compile sparse forward shader
        let shader_source = include_str!("shaders/sparse_forward.metal");
        let library = device
            .new_library_with_source(shader_source, &CompileOptions::new())
            .map_err(|e| format!("Failed to compile sparse forward shader: {}", e))?;

        let sparse_forward_kernel = library
            .get_function("sparse_forward_pass", None)
            .map_err(|e| format!("Failed to get sparse_forward_pass: {}", e))?;

        let sparse_forward_per_example_kernel = library
            .get_function("sparse_forward_pass_per_example", None)
            .map_err(|e| format!("Failed to get sparse_forward_pass_per_example: {}", e))?;

        let tiered_forward_kernel = library
            .get_function("tiered_sparse_forward_pass", None)
            .map_err(|e| format!("Failed to get tiered_sparse_forward_pass: {}", e))?;

        let sparse_forward_pipeline = device
            .new_compute_pipeline_state_with_function(&sparse_forward_kernel)
            .map_err(|e| format!("Failed to create sparse forward pipeline: {}", e))?;

        let sparse_forward_per_example_pipeline = device
            .new_compute_pipeline_state_with_function(&sparse_forward_per_example_kernel)
            .map_err(|e| format!("Failed to create sparse forward per-example pipeline: {}", e))?;

        let tiered_forward_pipeline = device
            .new_compute_pipeline_state_with_function(&tiered_forward_kernel)
            .map_err(|e| format!("Failed to create tiered forward pipeline: {}", e))?;

        Ok(Self {
            device,
            command_queue,
            sparse_forward_pipeline,
            sparse_forward_per_example_pipeline,
            tiered_forward_pipeline,
        })
    }

    /// Forward pass using sparse memory with binary search on GPU
    ///
    /// Args:
    ///   input_bits_flat: [num_examples * total_input_bits]
    ///   connections_flat: [num_neurons * bits_per_neuron]
    ///   keys_flat: Sorted keys for all neurons, concatenated
    ///   values_flat: Values corresponding to keys
    ///   offsets: [num_neurons] start offset per neuron
    ///   counts: [num_neurons] entry count per neuron
    ///
    /// Returns: [num_examples * num_clusters] probabilities
    pub fn forward_batch_sparse(
        &self,
        input_bits_flat: &[bool],
        connections_flat: &[i64],
        keys_flat: &[u64],
        values_flat: &[u8],
        offsets: &[u32],
        counts: &[u32],
        num_examples: usize,
        total_input_bits: usize,
        num_neurons: usize,
        bits_per_neuron: usize,
        neurons_per_cluster: usize,
        num_clusters: usize,
    ) -> Result<Vec<f32>, String> {
        if num_examples == 0 {
            return Ok(vec![]);
        }

        // Convert bools to u8
        let input_bits_u8: Vec<u8> = input_bits_flat.iter().map(|&b| b as u8).collect();
        let connections_i32: Vec<i32> = connections_flat.iter().map(|&c| c as i32).collect();

        #[repr(C)]
        struct SparseParams {
            num_examples: u32,
            total_input_bits: u32,
            num_neurons: u32,
            bits_per_neuron: u32,
            neurons_per_cluster: u32,
            num_clusters: u32,
            empty_value: f32,
        }

        let params = SparseParams {
            num_examples: num_examples as u32,
            total_input_bits: total_input_bits as u32,
            num_neurons: num_neurons as u32,
            bits_per_neuron: bits_per_neuron as u32,
            neurons_per_cluster: neurons_per_cluster as u32,
            num_clusters: num_clusters as u32,
            empty_value: crate::ramlm::get_empty_value(),
        };

        // Create buffers
        let input_buffer = self.device.new_buffer_with_data(
            input_bits_u8.as_ptr() as *const _,
            (input_bits_u8.len() * mem::size_of::<u8>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let conn_buffer = self.device.new_buffer_with_data(
            connections_i32.as_ptr() as *const _,
            (connections_i32.len() * mem::size_of::<i32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let keys_buffer = self.device.new_buffer_with_data(
            keys_flat.as_ptr() as *const _,
            (keys_flat.len() * mem::size_of::<u64>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let values_buffer = self.device.new_buffer_with_data(
            values_flat.as_ptr() as *const _,
            (values_flat.len() * mem::size_of::<u8>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let offsets_buffer = self.device.new_buffer_with_data(
            offsets.as_ptr() as *const _,
            (offsets.len() * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let counts_buffer = self.device.new_buffer_with_data(
            counts.as_ptr() as *const _,
            (counts.len() * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let params_buffer = self.device.new_buffer_with_data(
            &params as *const _ as *const _,
            mem::size_of::<SparseParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let output_size = num_examples * num_clusters;
        let output_buffer = self.device.new_buffer(
            (output_size * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let use_per_example = num_clusters > 1000;

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        if use_per_example {
            encoder.set_compute_pipeline_state(&self.sparse_forward_per_example_pipeline);
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&conn_buffer), 0);
            encoder.set_buffer(2, Some(&keys_buffer), 0);
            encoder.set_buffer(3, Some(&values_buffer), 0);
            encoder.set_buffer(4, Some(&offsets_buffer), 0);
            encoder.set_buffer(5, Some(&counts_buffer), 0);
            encoder.set_buffer(6, Some(&params_buffer), 0);
            encoder.set_buffer(7, Some(&output_buffer), 0);

            let grid_size = MTLSize::new(num_examples as u64, 1, 1);
            let max_threads = self.sparse_forward_per_example_pipeline.max_total_threads_per_threadgroup();
            let thread_group_size = MTLSize::new(max_threads.min(num_examples as u64), 1, 1);
            encoder.dispatch_threads(grid_size, thread_group_size);
        } else {
            encoder.set_compute_pipeline_state(&self.sparse_forward_pipeline);
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&conn_buffer), 0);
            encoder.set_buffer(2, Some(&keys_buffer), 0);
            encoder.set_buffer(3, Some(&values_buffer), 0);
            encoder.set_buffer(4, Some(&offsets_buffer), 0);
            encoder.set_buffer(5, Some(&counts_buffer), 0);
            encoder.set_buffer(6, Some(&params_buffer), 0);
            encoder.set_buffer(7, Some(&output_buffer), 0);

            let grid_size = MTLSize::new(num_clusters as u64, num_examples as u64, 1);
            let thread_group_size = MTLSize::new(32.min(num_clusters as u64), 8.min(num_examples as u64), 1);
            encoder.dispatch_threads(grid_size, thread_group_size);
        }

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let result_ptr = output_buffer.contents() as *const f32;
        let results: Vec<f32> = unsafe {
            std::slice::from_raw_parts(result_ptr, output_size).to_vec()
        };

        Ok(results)
    }

    /// Tiered forward pass using sparse memory with binary search on GPU
    ///
    /// Supports architectures with different bits/neurons per tier.
    pub fn forward_batch_tiered(
        &self,
        input_bits_flat: &[bool],
        connections_flat: &[i64],
        keys_flat: &[u64],
        values_flat: &[u8],
        offsets: &[u32],
        counts: &[u32],
        tier_configs: &[(usize, usize, usize, usize)], // (end_cluster, neurons_per_cluster, bits_per_neuron, start_neuron)
        num_examples: usize,
        total_input_bits: usize,
        num_clusters: usize,
    ) -> Result<Vec<f32>, String> {
        if num_examples == 0 {
            return Ok(vec![]);
        }

        let input_bits_u8: Vec<u8> = input_bits_flat.iter().map(|&b| b as u8).collect();
        let connections_i32: Vec<i32> = connections_flat.iter().map(|&c| c as i32).collect();

        #[repr(C)]
        struct TieredParams {
            num_examples: u32,
            total_input_bits: u32,
            num_clusters: u32,
            num_tiers: u32,
            empty_value: f32,
        }

        #[repr(C)]
        struct TierInfo {
            end_cluster: u32,
            neurons_per_cluster: u32,
            bits_per_neuron: u32,
            start_neuron: u32,
        }

        let params = TieredParams {
            num_examples: num_examples as u32,
            total_input_bits: total_input_bits as u32,
            num_clusters: num_clusters as u32,
            num_tiers: tier_configs.len() as u32,
            empty_value: crate::ramlm::get_empty_value(),
        };

        let tiers: Vec<TierInfo> = tier_configs.iter().map(|&(ec, npc, bpn, sn)| TierInfo {
            end_cluster: ec as u32,
            neurons_per_cluster: npc as u32,
            bits_per_neuron: bpn as u32,
            start_neuron: sn as u32,
        }).collect();

        // Create buffers
        let input_buffer = self.device.new_buffer_with_data(
            input_bits_u8.as_ptr() as *const _,
            (input_bits_u8.len() * mem::size_of::<u8>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let conn_buffer = self.device.new_buffer_with_data(
            connections_i32.as_ptr() as *const _,
            (connections_i32.len() * mem::size_of::<i32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let keys_buffer = self.device.new_buffer_with_data(
            keys_flat.as_ptr() as *const _,
            (keys_flat.len() * mem::size_of::<u64>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let values_buffer = self.device.new_buffer_with_data(
            values_flat.as_ptr() as *const _,
            (values_flat.len() * mem::size_of::<u8>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let offsets_buffer = self.device.new_buffer_with_data(
            offsets.as_ptr() as *const _,
            (offsets.len() * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let counts_buffer = self.device.new_buffer_with_data(
            counts.as_ptr() as *const _,
            (counts.len() * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let params_buffer = self.device.new_buffer_with_data(
            &params as *const _ as *const _,
            mem::size_of::<TieredParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let tiers_buffer = self.device.new_buffer_with_data(
            tiers.as_ptr() as *const _,
            (tiers.len() * mem::size_of::<TierInfo>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let output_size = num_examples * num_clusters;
        let output_buffer = self.device.new_buffer(
            (output_size * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.tiered_forward_pipeline);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&conn_buffer), 0);
        encoder.set_buffer(2, Some(&keys_buffer), 0);
        encoder.set_buffer(3, Some(&values_buffer), 0);
        encoder.set_buffer(4, Some(&offsets_buffer), 0);
        encoder.set_buffer(5, Some(&counts_buffer), 0);
        encoder.set_buffer(6, Some(&params_buffer), 0);
        encoder.set_buffer(7, Some(&tiers_buffer), 0);
        encoder.set_buffer(8, Some(&output_buffer), 0);

        let grid_size = MTLSize::new(num_clusters as u64, num_examples as u64, 1);
        let thread_group_size = MTLSize::new(32.min(num_clusters as u64), 8.min(num_examples as u64), 1);
        encoder.dispatch_threads(grid_size, thread_group_size);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let result_ptr = output_buffer.contents() as *const f32;
        let results: Vec<f32> = unsafe {
            std::slice::from_raw_parts(result_ptr, output_size).to_vec()
        };

        Ok(results)
    }
}

// ============================================================================
// Batched Sparse Evaluator - Multiple Genomes in One Dispatch
// ============================================================================

/// Parameters for batched sparse forward pass (must match Metal struct)
#[repr(C)]
struct BatchedSparseParams {
    num_examples: u32,
    total_input_bits: u32,
    num_neurons: u32,
    bits_per_neuron: u32,
    neurons_per_cluster: u32,
    num_clusters: u32,
    num_genomes: u32,
    empty_value: f32,
}

/// Metal evaluator for batched genome evaluation
///
/// Evaluates MULTIPLE genomes in a single GPU dispatch, which is more efficient
/// than launching separate kernels for each genome.
pub struct MetalBatchedEvaluator {
    device: Device,
    command_queue: CommandQueue,
    batched_forward_pipeline: ComputePipelineState,
    batched_per_example_pipeline: ComputePipelineState,
}

impl MetalBatchedEvaluator {
    /// Create new batched Metal evaluator
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;
        let command_queue = device.new_command_queue();

        // Compile batched sparse forward shader
        let shader_source = include_str!("shaders/batched_sparse_forward.metal");
        let library = device
            .new_library_with_source(shader_source, &CompileOptions::new())
            .map_err(|e| format!("Failed to compile batched sparse shader: {}", e))?;

        let batched_forward_kernel = library
            .get_function("batched_sparse_forward_pass", None)
            .map_err(|e| format!("Failed to get batched_sparse_forward_pass: {}", e))?;

        let batched_per_example_kernel = library
            .get_function("batched_sparse_forward_per_example", None)
            .map_err(|e| format!("Failed to get batched_sparse_forward_per_example: {}", e))?;

        let batched_forward_pipeline = device
            .new_compute_pipeline_state_with_function(&batched_forward_kernel)
            .map_err(|e| format!("Failed to create batched forward pipeline: {}", e))?;

        let batched_per_example_pipeline = device
            .new_compute_pipeline_state_with_function(&batched_per_example_kernel)
            .map_err(|e| format!("Failed to create batched per-example pipeline: {}", e))?;

        Ok(Self {
            device,
            command_queue,
            batched_forward_pipeline,
            batched_per_example_pipeline,
        })
    }

    /// Batched forward pass for multiple genomes
    ///
    /// Evaluates N genomes in a single GPU dispatch.
    /// Input bits are shared across genomes; each genome has its own connections and memory.
    ///
    /// Args:
    ///   input_bits: [num_examples * total_input_bits] - SHARED across genomes
    ///   connections_flat: [num_genomes * num_neurons * bits_per_neuron]
    ///   keys_flat: All genomes' sparse keys concatenated
    ///   values_flat: All genomes' sparse values concatenated
    ///   offsets_flat: [num_genomes * num_neurons]
    ///   counts_flat: [num_genomes * num_neurons]
    ///   genome_key_offsets: [num_genomes] offset into keys/values for each genome
    ///   params: Evaluation parameters
    ///
    /// Returns: [num_genomes * num_examples * num_clusters] probabilities
    pub fn forward_batch_genomes(
        &self,
        input_bits: &[bool],
        connections_flat: &[i64],
        keys_flat: &[u64],
        values_flat: &[u8],
        offsets_flat: &[u32],
        counts_flat: &[u32],
        genome_key_offsets: &[u32],
        num_examples: usize,
        total_input_bits: usize,
        num_neurons: usize,
        bits_per_neuron: usize,
        neurons_per_cluster: usize,
        num_clusters: usize,
        num_genomes: usize,
        empty_value: f32,
    ) -> Result<Vec<f32>, String> {
        if num_genomes == 0 || num_examples == 0 {
            return Ok(vec![]);
        }

        // Convert input bits to u8
        let input_u8: Vec<u8> = input_bits.iter().map(|&b| b as u8).collect();

        // Convert connections to i32 for Metal
        let connections_i32: Vec<i32> = connections_flat.iter().map(|&c| c as i32).collect();

        // Create Metal buffers
        let input_buffer = self.device.new_buffer_with_data(
            input_u8.as_ptr() as *const _,
            (input_u8.len() * mem::size_of::<u8>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let connections_buffer = self.device.new_buffer_with_data(
            connections_i32.as_ptr() as *const _,
            (connections_i32.len() * mem::size_of::<i32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let keys_buffer = self.device.new_buffer_with_data(
            keys_flat.as_ptr() as *const _,
            (keys_flat.len() * mem::size_of::<u64>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let values_buffer = self.device.new_buffer_with_data(
            values_flat.as_ptr() as *const _,
            (values_flat.len() * mem::size_of::<u8>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let offsets_buffer = self.device.new_buffer_with_data(
            offsets_flat.as_ptr() as *const _,
            (offsets_flat.len() * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let counts_buffer = self.device.new_buffer_with_data(
            counts_flat.as_ptr() as *const _,
            (counts_flat.len() * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let genome_offsets_buffer = self.device.new_buffer_with_data(
            genome_key_offsets.as_ptr() as *const _,
            (genome_key_offsets.len() * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let params = BatchedSparseParams {
            num_examples: num_examples as u32,
            total_input_bits: total_input_bits as u32,
            num_neurons: num_neurons as u32,
            bits_per_neuron: bits_per_neuron as u32,
            neurons_per_cluster: neurons_per_cluster as u32,
            num_clusters: num_clusters as u32,
            num_genomes: num_genomes as u32,
            empty_value,
        };

        let params_buffer = self.device.new_buffer_with_data(
            &params as *const _ as *const _,
            mem::size_of::<BatchedSparseParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let output_size = num_genomes * num_examples * num_clusters;
        let output_buffer = self.device.new_buffer(
            (output_size * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create command buffer and encoder
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Decide which kernel to use based on problem size
        let use_per_example = num_clusters <= 256;

        if use_per_example {
            encoder.set_compute_pipeline_state(&self.batched_per_example_pipeline);
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&connections_buffer), 0);
            encoder.set_buffer(2, Some(&keys_buffer), 0);
            encoder.set_buffer(3, Some(&values_buffer), 0);
            encoder.set_buffer(4, Some(&offsets_buffer), 0);
            encoder.set_buffer(5, Some(&counts_buffer), 0);
            encoder.set_buffer(6, Some(&genome_offsets_buffer), 0);
            encoder.set_buffer(7, Some(&params_buffer), 0);
            encoder.set_buffer(8, Some(&output_buffer), 0);

            // Grid: (num_examples, num_genomes)
            let grid_size = MTLSize::new(num_examples as u64, num_genomes as u64, 1);
            let max_threads = self.batched_per_example_pipeline.max_total_threads_per_threadgroup();
            let thread_group_size = MTLSize::new(
                max_threads.min(num_examples as u64),
                1,
                1,
            );
            encoder.dispatch_threads(grid_size, thread_group_size);
        } else {
            encoder.set_compute_pipeline_state(&self.batched_forward_pipeline);
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&connections_buffer), 0);
            encoder.set_buffer(2, Some(&keys_buffer), 0);
            encoder.set_buffer(3, Some(&values_buffer), 0);
            encoder.set_buffer(4, Some(&offsets_buffer), 0);
            encoder.set_buffer(5, Some(&counts_buffer), 0);
            encoder.set_buffer(6, Some(&genome_offsets_buffer), 0);
            encoder.set_buffer(7, Some(&params_buffer), 0);
            encoder.set_buffer(8, Some(&output_buffer), 0);

            // Grid: (num_clusters, num_examples, num_genomes)
            let grid_size = MTLSize::new(
                num_clusters as u64,
                num_examples as u64,
                num_genomes as u64,
            );
            let thread_group_size = MTLSize::new(
                32.min(num_clusters as u64),
                8.min(num_examples as u64),
                1.min(num_genomes as u64),
            );
            encoder.dispatch_threads(grid_size, thread_group_size);
        }

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read results
        let result_ptr = output_buffer.contents() as *const f32;
        let results: Vec<f32> = unsafe {
            std::slice::from_raw_parts(result_ptr, output_size).to_vec()
        };

        Ok(results)
    }
}

// ============================================================================
// Sparse CE Evaluator - Computes CE/Accuracy Directly on GPU
// ============================================================================

/// Parameters for sparse CE computation (must match Metal struct)
#[repr(C)]
struct SparseCEParams {
    num_examples: u32,
    total_input_bits: u32,
    num_neurons: u32,
    bits_per_neuron: u32,
    neurons_per_cluster: u32,
    num_clusters: u32,
    empty_value: f32,
}

/// Metal evaluator that computes CE and accuracy directly on GPU
///
/// Instead of returning all probabilities (10GB for 50K×50K), this evaluator
/// computes cross-entropy and accuracy ON THE GPU and returns just the results.
/// This eliminates the massive GPU→CPU data transfer.
pub struct MetalSparseCEEvaluator {
    device: Device,
    command_queue: CommandQueue,
    ce_online_pipeline: ComputePipelineState,
}

impl MetalSparseCEEvaluator {
    /// Create new sparse CE evaluator
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;
        let command_queue = device.new_command_queue();

        // Compile sparse CE shader
        let shader_source = include_str!("shaders/sparse_ce.metal");
        let library = device
            .new_library_with_source(shader_source, &CompileOptions::new())
            .map_err(|e| format!("Failed to compile sparse CE shader: {}", e))?;

        let ce_online_kernel = library
            .get_function("sparse_forward_with_ce_online", None)
            .map_err(|e| format!("Failed to get sparse_forward_with_ce_online: {}", e))?;

        let ce_online_pipeline = device
            .new_compute_pipeline_state_with_function(&ce_online_kernel)
            .map_err(|e| format!("Failed to create CE online pipeline: {}", e))?;

        Ok(Self {
            device,
            command_queue,
            ce_online_pipeline,
        })
    }

    /// Compute CE and accuracy directly on GPU
    ///
    /// Returns (average_ce, accuracy) instead of all probabilities.
    /// This eliminates the 10GB data transfer for 50K×50K.
    ///
    /// Args:
    ///   input_bits: [num_examples * total_input_bits]
    ///   connections: [num_neurons * bits_per_neuron]
    ///   keys: Sorted addresses for all neurons
    ///   values: Corresponding cell values
    ///   offsets: [num_neurons] start index per neuron
    ///   counts: [num_neurons] count of entries per neuron
    ///   targets: [num_examples] target cluster for each example
    ///   params: Evaluation parameters
    ///
    /// Returns: (average_ce, accuracy)
    pub fn compute_ce(
        &self,
        input_bits: &[bool],
        connections: &[i64],
        keys: &[u64],
        values: &[u8],
        offsets: &[u32],
        counts: &[u32],
        targets: &[i64],
        num_examples: usize,
        total_input_bits: usize,
        num_neurons: usize,
        bits_per_neuron: usize,
        neurons_per_cluster: usize,
        num_clusters: usize,
        empty_value: f32,
    ) -> Result<(f64, f64), String> {
        if num_examples == 0 {
            return Ok((0.0, 0.0));
        }

        // Convert input bits to u8
        let input_u8: Vec<u8> = input_bits.iter().map(|&b| b as u8).collect();

        // Convert connections to i32 for Metal
        let connections_i32: Vec<i32> = connections.iter().map(|&c| c as i32).collect();

        // Convert targets to i32
        let targets_i32: Vec<i32> = targets.iter().map(|&t| t as i32).collect();

        // Create Metal buffers
        let input_buffer = self.device.new_buffer_with_data(
            input_u8.as_ptr() as *const _,
            (input_u8.len() * mem::size_of::<u8>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let connections_buffer = self.device.new_buffer_with_data(
            connections_i32.as_ptr() as *const _,
            (connections_i32.len() * mem::size_of::<i32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let keys_buffer = self.device.new_buffer_with_data(
            keys.as_ptr() as *const _,
            (keys.len() * mem::size_of::<u64>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let values_buffer = self.device.new_buffer_with_data(
            values.as_ptr() as *const _,
            (values.len() * mem::size_of::<u8>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let offsets_buffer = self.device.new_buffer_with_data(
            offsets.as_ptr() as *const _,
            (offsets.len() * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let counts_buffer = self.device.new_buffer_with_data(
            counts.as_ptr() as *const _,
            (counts.len() * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let targets_buffer = self.device.new_buffer_with_data(
            targets_i32.as_ptr() as *const _,
            (targets_i32.len() * mem::size_of::<i32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let params = SparseCEParams {
            num_examples: num_examples as u32,
            total_input_bits: total_input_bits as u32,
            num_neurons: num_neurons as u32,
            bits_per_neuron: bits_per_neuron as u32,
            neurons_per_cluster: neurons_per_cluster as u32,
            num_clusters: num_clusters as u32,
            empty_value,
        };

        let params_buffer = self.device.new_buffer_with_data(
            &params as *const _ as *const _,
            mem::size_of::<SparseCEParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Output buffers: one float CE and one uint correct per example
        let ce_buffer = self.device.new_buffer(
            (num_examples * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let correct_buffer = self.device.new_buffer(
            (num_examples * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create command buffer and encoder
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.ce_online_pipeline);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&connections_buffer), 0);
        encoder.set_buffer(2, Some(&keys_buffer), 0);
        encoder.set_buffer(3, Some(&values_buffer), 0);
        encoder.set_buffer(4, Some(&offsets_buffer), 0);
        encoder.set_buffer(5, Some(&counts_buffer), 0);
        encoder.set_buffer(6, Some(&targets_buffer), 0);
        encoder.set_buffer(7, Some(&params_buffer), 0);
        encoder.set_buffer(8, Some(&ce_buffer), 0);
        encoder.set_buffer(9, Some(&correct_buffer), 0);

        // Grid: one thread per example
        let grid_size = MTLSize::new(num_examples as u64, 1, 1);
        let max_threads = self.ce_online_pipeline.max_total_threads_per_threadgroup();
        let thread_group_size = MTLSize::new(max_threads.min(num_examples as u64), 1, 1);
        encoder.dispatch_threads(grid_size, thread_group_size);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read results and reduce on CPU (GPU reduction would be faster but more complex)
        let ce_ptr = ce_buffer.contents() as *const f32;
        let correct_ptr = correct_buffer.contents() as *const u32;

        let (total_ce, total_correct): (f64, u64) = unsafe {
            let ce_slice = std::slice::from_raw_parts(ce_ptr, num_examples);
            let correct_slice = std::slice::from_raw_parts(correct_ptr, num_examples);

            let total_ce: f64 = ce_slice.iter().map(|&c| c as f64).sum();
            let total_correct: u64 = correct_slice.iter().map(|&c| c as u64).sum();

            (total_ce, total_correct)
        };

        let avg_ce = total_ce / num_examples as f64;
        let accuracy = total_correct as f64 / num_examples as f64;

        Ok((avg_ce, accuracy))
    }
}

// ============================================================================
// CE Reduction Evaluator - For Tiered Configs
// ============================================================================

/// Parameters for CE reduction (must match Metal struct)
#[repr(C)]
struct CEReduceParams {
    num_examples: u32,
    num_clusters: u32,
}

/// Metal evaluator for CE reduction from pre-computed scores
///
/// For tiered configs where groups are evaluated separately, this takes
/// all the scores (accumulated on GPU) and computes CE/accuracy.
pub struct MetalCEReduceEvaluator {
    device: Device,
    command_queue: CommandQueue,
    ce_reduce_pipeline: ComputePipelineState,
    scatter_pipeline: ComputePipelineState,
}

impl MetalCEReduceEvaluator {
    /// Create new CE reduction evaluator
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;
        let command_queue = device.new_command_queue();

        let shader_source = include_str!("shaders/ce_reduce.metal");
        let library = device
            .new_library_with_source(shader_source, &CompileOptions::new())
            .map_err(|e| format!("Failed to compile CE reduce shader: {}", e))?;

        let ce_reduce_kernel = library
            .get_function("reduce_scores_to_ce", None)
            .map_err(|e| format!("Failed to get reduce_scores_to_ce: {}", e))?;

        let scatter_kernel = library
            .get_function("scatter_group_scores", None)
            .map_err(|e| format!("Failed to get scatter_group_scores: {}", e))?;

        let ce_reduce_pipeline = device
            .new_compute_pipeline_state_with_function(&ce_reduce_kernel)
            .map_err(|e| format!("Failed to create CE reduce pipeline: {}", e))?;

        let scatter_pipeline = device
            .new_compute_pipeline_state_with_function(&scatter_kernel)
            .map_err(|e| format!("Failed to create scatter pipeline: {}", e))?;

        Ok(Self {
            device,
            command_queue,
            ce_reduce_pipeline,
            scatter_pipeline,
        })
    }

    /// Create a GPU buffer for accumulating all scores, initialized to 0
    pub fn create_scores_buffer(&self, num_examples: usize, num_clusters: usize) -> Buffer {
        let size = num_examples * num_clusters;
        // Initialize to 0.0 (important: unwritten clusters must have score 0, not garbage)
        let zeros: Vec<f32> = vec![0.0; size];
        self.device.new_buffer_with_data(
            zeros.as_ptr() as *const _,
            (size * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Scatter group scores to the full scores buffer on GPU
    ///
    /// This copies scores from a group's output buffer to the correct positions
    /// in the full scores buffer, avoiding CPU round-trip.
    pub fn scatter_to_buffer(
        &self,
        group_scores: &[f32],
        scores_buffer: &Buffer,
        cluster_ids: &[usize],
        num_examples: usize,
        num_clusters: usize,
    ) {
        let num_group_clusters = cluster_ids.len();

        // Create temporary buffers
        let group_buffer = self.device.new_buffer_with_data(
            group_scores.as_ptr() as *const _,
            (group_scores.len() * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let cluster_ids_u32: Vec<u32> = cluster_ids.iter().map(|&c| c as u32).collect();
        let cluster_ids_buffer = self.device.new_buffer_with_data(
            cluster_ids_u32.as_ptr() as *const _,
            (cluster_ids_u32.len() * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let num_examples_u32 = num_examples as u32;
        let num_group_clusters_u32 = num_group_clusters as u32;
        let num_clusters_u32 = num_clusters as u32;

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.scatter_pipeline);
        encoder.set_buffer(0, Some(&group_buffer), 0);
        encoder.set_buffer(1, Some(scores_buffer), 0);
        encoder.set_buffer(2, Some(&cluster_ids_buffer), 0);
        encoder.set_bytes(3, mem::size_of::<u32>() as u64, &num_examples_u32 as *const _ as *const _);
        encoder.set_bytes(4, mem::size_of::<u32>() as u64, &num_group_clusters_u32 as *const _ as *const _);
        encoder.set_bytes(5, mem::size_of::<u32>() as u64, &num_clusters_u32 as *const _ as *const _);

        // Grid: (num_group_clusters, num_examples)
        let grid_size = MTLSize::new(num_group_clusters as u64, num_examples as u64, 1);
        let thread_group_size = MTLSize::new(
            32.min(num_group_clusters as u64),
            8.min(num_examples as u64),
            1,
        );
        encoder.dispatch_threads(grid_size, thread_group_size);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    /// Compute CE and accuracy from accumulated scores buffer
    pub fn compute_ce_from_buffer(
        &self,
        scores_buffer: &Buffer,
        targets: &[i64],
        num_examples: usize,
        num_clusters: usize,
    ) -> Result<(f64, f64), String> {
        let targets_i32: Vec<i32> = targets.iter().map(|&t| t as i32).collect();
        let targets_buffer = self.device.new_buffer_with_data(
            targets_i32.as_ptr() as *const _,
            (targets_i32.len() * mem::size_of::<i32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let params = CEReduceParams {
            num_examples: num_examples as u32,
            num_clusters: num_clusters as u32,
        };
        let params_buffer = self.device.new_buffer_with_data(
            &params as *const _ as *const _,
            mem::size_of::<CEReduceParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let ce_buffer = self.device.new_buffer(
            (num_examples * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let correct_buffer = self.device.new_buffer(
            (num_examples * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.ce_reduce_pipeline);
        encoder.set_buffer(0, Some(scores_buffer), 0);
        encoder.set_buffer(1, Some(&targets_buffer), 0);
        encoder.set_buffer(2, Some(&params_buffer), 0);
        encoder.set_buffer(3, Some(&ce_buffer), 0);
        encoder.set_buffer(4, Some(&correct_buffer), 0);

        let grid_size = MTLSize::new(num_examples as u64, 1, 1);
        let max_threads = self.ce_reduce_pipeline.max_total_threads_per_threadgroup();
        let thread_group_size = MTLSize::new(max_threads.min(num_examples as u64), 1, 1);
        encoder.dispatch_threads(grid_size, thread_group_size);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Sum CE and correct on CPU
        let ce_ptr = ce_buffer.contents() as *const f32;
        let correct_ptr = correct_buffer.contents() as *const u32;

        let (total_ce, total_correct): (f64, u64) = unsafe {
            let ce_slice = std::slice::from_raw_parts(ce_ptr, num_examples);
            let correct_slice = std::slice::from_raw_parts(correct_ptr, num_examples);

            let total_ce: f64 = ce_slice.iter().map(|&c| c as f64).sum();
            let total_correct: u64 = correct_slice.iter().map(|&c| c as u64).sum();

            (total_ce, total_correct)
        };

        let avg_ce = total_ce / num_examples as f64;
        let accuracy = total_correct as f64 / num_examples as f64;

        Ok((avg_ce, accuracy))
    }
}

// ============================================================================
// Unified Group Evaluator - Writes Directly to Shared GPU Buffer
// ============================================================================

/// Parameters for sparse forward to buffer (must match Metal struct)
#[repr(C)]
struct SparseToBufferParams {
    num_examples: u32,
    total_input_bits: u32,
    num_neurons: u32,
    bits_per_neuron: u32,
    neurons_per_cluster: u32,
    num_group_clusters: u32,
    total_clusters: u32,
    empty_value: f32,
}

/// Parameters for dense forward to buffer (must match Metal struct)
#[repr(C)]
struct DenseToBufferParams {
    num_examples: u32,
    total_input_bits: u32,
    num_neurons: u32,
    bits_per_neuron: u32,
    neurons_per_cluster: u32,
    num_group_clusters: u32,
    total_clusters: u32,
    words_per_neuron: u32,
    empty_value: f32,
}

/// Unified Metal evaluator that writes group results directly to shared GPU buffer
///
/// This avoids the GPU→CPU→GPU round-trip that was slowing down tiered evaluation.
/// All groups write to the same shared buffer on GPU, then CE is computed once.
pub struct MetalGroupEvaluator {
    device: Device,
    command_queue: CommandQueue,
    sparse_to_buffer_pipeline: ComputePipelineState,
    dense_to_buffer_pipeline: ComputePipelineState,
    ce_reduce_pipeline: ComputePipelineState,
}

impl MetalGroupEvaluator {
    /// Create new unified group evaluator
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;
        let command_queue = device.new_command_queue();

        // Compile sparse forward shader
        let sparse_shader = include_str!("shaders/sparse_forward.metal");
        let sparse_library = device
            .new_library_with_source(sparse_shader, &CompileOptions::new())
            .map_err(|e| format!("Failed to compile sparse shader: {}", e))?;

        let sparse_to_buffer_kernel = sparse_library
            .get_function("sparse_forward_to_buffer", None)
            .map_err(|e| format!("Failed to get sparse_forward_to_buffer: {}", e))?;

        let sparse_to_buffer_pipeline = device
            .new_compute_pipeline_state_with_function(&sparse_to_buffer_kernel)
            .map_err(|e| format!("Failed to create sparse to buffer pipeline: {}", e))?;

        // Compile dense forward shader
        let dense_shader = include_str!("shaders/ramlm.metal");
        let dense_library = device
            .new_library_with_source(dense_shader, &CompileOptions::new())
            .map_err(|e| format!("Failed to compile dense shader: {}", e))?;

        let dense_to_buffer_kernel = dense_library
            .get_function("ramlm_forward_to_buffer", None)
            .map_err(|e| format!("Failed to get ramlm_forward_to_buffer: {}", e))?;

        let dense_to_buffer_pipeline = device
            .new_compute_pipeline_state_with_function(&dense_to_buffer_kernel)
            .map_err(|e| format!("Failed to create dense to buffer pipeline: {}", e))?;

        // Compile CE reduce shader
        let ce_shader = include_str!("shaders/ce_reduce.metal");
        let ce_library = device
            .new_library_with_source(ce_shader, &CompileOptions::new())
            .map_err(|e| format!("Failed to compile CE reduce shader: {}", e))?;

        let ce_reduce_kernel = ce_library
            .get_function("reduce_scores_to_ce", None)
            .map_err(|e| format!("Failed to get reduce_scores_to_ce: {}", e))?;

        let ce_reduce_pipeline = device
            .new_compute_pipeline_state_with_function(&ce_reduce_kernel)
            .map_err(|e| format!("Failed to create CE reduce pipeline: {}", e))?;

        Ok(Self {
            device,
            command_queue,
            sparse_to_buffer_pipeline,
            dense_to_buffer_pipeline,
            ce_reduce_pipeline,
        })
    }

    /// Create shared scores buffer initialized to 0
    pub fn create_scores_buffer(&self, num_examples: usize, num_clusters: usize) -> Buffer {
        let size = num_examples * num_clusters;
        let zeros: Vec<f32> = vec![0.0; size];
        self.device.new_buffer_with_data(
            zeros.as_ptr() as *const _,
            (size * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Create input bits buffer (shared across all group evaluations)
    pub fn create_input_buffer(&self, input_bits: &[bool]) -> Buffer {
        let input_u8: Vec<u8> = input_bits.iter().map(|&b| b as u8).collect();
        self.device.new_buffer_with_data(
            input_u8.as_ptr() as *const _,
            (input_u8.len() * mem::size_of::<u8>()) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Evaluate sparse group and write directly to shared buffer on GPU
    pub fn eval_sparse_to_buffer(
        &self,
        input_buffer: &Buffer,
        scores_buffer: &Buffer,
        connections: &[i64],
        keys: &[u64],
        values: &[u8],
        offsets: &[u32],
        counts: &[u32],
        cluster_ids: &[usize],
        num_examples: usize,
        total_input_bits: usize,
        bits_per_neuron: usize,
        neurons_per_cluster: usize,
        num_clusters: usize,
        empty_value: f32,
    ) {
        let num_group_clusters = cluster_ids.len();
        let num_neurons = num_group_clusters * neurons_per_cluster;

        // Convert to GPU-friendly formats
        let connections_i32: Vec<i32> = connections.iter().map(|&c| c as i32).collect();
        let cluster_ids_u32: Vec<u32> = cluster_ids.iter().map(|&c| c as u32).collect();

        // Create buffers
        let conn_buffer = self.device.new_buffer_with_data(
            connections_i32.as_ptr() as *const _,
            (connections_i32.len() * mem::size_of::<i32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let keys_buffer = self.device.new_buffer_with_data(
            keys.as_ptr() as *const _,
            (keys.len() * mem::size_of::<u64>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let values_buffer = self.device.new_buffer_with_data(
            values.as_ptr() as *const _,
            (values.len() * mem::size_of::<u8>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let offsets_buffer = self.device.new_buffer_with_data(
            offsets.as_ptr() as *const _,
            (offsets.len() * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let counts_buffer = self.device.new_buffer_with_data(
            counts.as_ptr() as *const _,
            (counts.len() * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let cluster_ids_buffer = self.device.new_buffer_with_data(
            cluster_ids_u32.as_ptr() as *const _,
            (cluster_ids_u32.len() * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let params = SparseToBufferParams {
            num_examples: num_examples as u32,
            total_input_bits: total_input_bits as u32,
            num_neurons: num_neurons as u32,
            bits_per_neuron: bits_per_neuron as u32,
            neurons_per_cluster: neurons_per_cluster as u32,
            num_group_clusters: num_group_clusters as u32,
            total_clusters: num_clusters as u32,
            empty_value,
        };

        let params_buffer = self.device.new_buffer_with_data(
            &params as *const _ as *const _,
            mem::size_of::<SparseToBufferParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Dispatch kernel
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.sparse_to_buffer_pipeline);
        encoder.set_buffer(0, Some(input_buffer), 0);
        encoder.set_buffer(1, Some(&conn_buffer), 0);
        encoder.set_buffer(2, Some(&keys_buffer), 0);
        encoder.set_buffer(3, Some(&values_buffer), 0);
        encoder.set_buffer(4, Some(&offsets_buffer), 0);
        encoder.set_buffer(5, Some(&counts_buffer), 0);
        encoder.set_buffer(6, Some(&cluster_ids_buffer), 0);
        encoder.set_buffer(7, Some(&params_buffer), 0);
        encoder.set_buffer(8, Some(scores_buffer), 0);

        let grid_size = MTLSize::new(num_group_clusters as u64, num_examples as u64, 1);
        let thread_group_size = MTLSize::new(
            32.min(num_group_clusters as u64),
            8.min(num_examples as u64),
            1,
        );
        encoder.dispatch_threads(grid_size, thread_group_size);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    /// Evaluate dense group and write directly to shared buffer on GPU
    pub fn eval_dense_to_buffer(
        &self,
        input_buffer: &Buffer,
        scores_buffer: &Buffer,
        connections: &[i64],
        memory_words: &[i64],
        cluster_ids: &[usize],
        num_examples: usize,
        total_input_bits: usize,
        bits_per_neuron: usize,
        neurons_per_cluster: usize,
        num_clusters: usize,
        words_per_neuron: usize,
        empty_value: f32,
    ) {
        let num_group_clusters = cluster_ids.len();
        let num_neurons = num_group_clusters * neurons_per_cluster;

        // Convert to GPU-friendly formats
        let connections_i32: Vec<i32> = connections.iter().map(|&c| c as i32).collect();
        let cluster_ids_u32: Vec<u32> = cluster_ids.iter().map(|&c| c as u32).collect();

        // Create buffers
        let conn_buffer = self.device.new_buffer_with_data(
            connections_i32.as_ptr() as *const _,
            (connections_i32.len() * mem::size_of::<i32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let memory_buffer = self.device.new_buffer_with_data(
            memory_words.as_ptr() as *const _,
            (memory_words.len() * mem::size_of::<i64>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let cluster_ids_buffer = self.device.new_buffer_with_data(
            cluster_ids_u32.as_ptr() as *const _,
            (cluster_ids_u32.len() * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let params = DenseToBufferParams {
            num_examples: num_examples as u32,
            total_input_bits: total_input_bits as u32,
            num_neurons: num_neurons as u32,
            bits_per_neuron: bits_per_neuron as u32,
            neurons_per_cluster: neurons_per_cluster as u32,
            num_group_clusters: num_group_clusters as u32,
            total_clusters: num_clusters as u32,
            words_per_neuron: words_per_neuron as u32,
            empty_value,
        };

        let params_buffer = self.device.new_buffer_with_data(
            &params as *const _ as *const _,
            mem::size_of::<DenseToBufferParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Dispatch kernel
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.dense_to_buffer_pipeline);
        encoder.set_buffer(0, Some(input_buffer), 0);
        encoder.set_buffer(1, Some(&conn_buffer), 0);
        encoder.set_buffer(2, Some(&memory_buffer), 0);
        encoder.set_buffer(3, Some(&cluster_ids_buffer), 0);
        encoder.set_buffer(4, Some(&params_buffer), 0);
        encoder.set_buffer(5, Some(scores_buffer), 0);

        let grid_size = MTLSize::new(num_group_clusters as u64, num_examples as u64, 1);
        let thread_group_size = MTLSize::new(
            32.min(num_group_clusters as u64),
            8.min(num_examples as u64),
            1,
        );
        encoder.dispatch_threads(grid_size, thread_group_size);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    /// Compute CE and accuracy from accumulated scores buffer
    pub fn compute_ce_from_buffer(
        &self,
        scores_buffer: &Buffer,
        targets: &[i64],
        num_examples: usize,
        num_clusters: usize,
    ) -> Result<(f64, f64), String> {
        let targets_i32: Vec<i32> = targets.iter().map(|&t| t as i32).collect();
        let targets_buffer = self.device.new_buffer_with_data(
            targets_i32.as_ptr() as *const _,
            (targets_i32.len() * mem::size_of::<i32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let params = CEReduceParams {
            num_examples: num_examples as u32,
            num_clusters: num_clusters as u32,
        };
        let params_buffer = self.device.new_buffer_with_data(
            &params as *const _ as *const _,
            mem::size_of::<CEReduceParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let ce_buffer = self.device.new_buffer(
            (num_examples * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let correct_buffer = self.device.new_buffer(
            (num_examples * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.ce_reduce_pipeline);
        encoder.set_buffer(0, Some(scores_buffer), 0);
        encoder.set_buffer(1, Some(&targets_buffer), 0);
        encoder.set_buffer(2, Some(&params_buffer), 0);
        encoder.set_buffer(3, Some(&ce_buffer), 0);
        encoder.set_buffer(4, Some(&correct_buffer), 0);

        let grid_size = MTLSize::new(num_examples as u64, 1, 1);
        let max_threads = self.ce_reduce_pipeline.max_total_threads_per_threadgroup();
        let thread_group_size = MTLSize::new(max_threads.min(num_examples as u64), 1, 1);
        encoder.dispatch_threads(grid_size, thread_group_size);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Sum CE and correct on CPU
        let ce_ptr = ce_buffer.contents() as *const f32;
        let correct_ptr = correct_buffer.contents() as *const u32;

        let (total_ce, total_correct): (f64, u64) = unsafe {
            let ce_slice = std::slice::from_raw_parts(ce_ptr, num_examples);
            let correct_slice = std::slice::from_raw_parts(correct_ptr, num_examples);

            let total_ce: f64 = ce_slice.iter().map(|&c| c as f64).sum();
            let total_correct: u64 = correct_slice.iter().map(|&c| c as u64).sum();

            (total_ce, total_correct)
        };

        let avg_ce = total_ce / num_examples as f64;
        let accuracy = total_correct as f64 / num_examples as f64;

        Ok((avg_ce, accuracy))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_available() {
        // Should be available on any Mac with Metal support
        if MetalRAMLMEvaluator::is_available() {
            println!("Metal available!");
            println!("{}", MetalRAMLMEvaluator::device_info().unwrap());
        } else {
            println!("Metal not available (running in VM or non-Mac?)");
        }
    }

    #[test]
    fn test_sparse_evaluator_creation() {
        if MetalRAMLMEvaluator::is_available() {
            let evaluator = MetalSparseEvaluator::new();
            assert!(evaluator.is_ok(), "Failed to create sparse evaluator: {:?}", evaluator.err());
        }
    }

    #[test]
    fn test_sparse_ce_evaluator_creation() {
        if MetalRAMLMEvaluator::is_available() {
            let evaluator = MetalSparseCEEvaluator::new();
            assert!(evaluator.is_ok(), "Failed to create sparse CE evaluator: {:?}", evaluator.err());
        }
    }

    #[test]
    fn test_sparse_ce_small() {
        if !MetalRAMLMEvaluator::is_available() {
            return;
        }

        let evaluator = MetalSparseCEEvaluator::new().unwrap();

        // Small test: 3 examples, 4 clusters, 2 neurons per cluster, 3 bits
        let num_examples = 3;
        let num_clusters = 4;
        let neurons_per_cluster = 2;
        let bits = 3;
        let total_neurons = num_clusters * neurons_per_cluster;
        let total_input_bits = 10;

        // Input bits: all true for simplicity
        let input_bits: Vec<bool> = vec![true; num_examples * total_input_bits];

        // Connections: point to bits 0, 1, 2 for all neurons
        let connections: Vec<i64> = (0..total_neurons)
            .flat_map(|_| vec![0i64, 1, 2])
            .collect();

        // Sparse memory: address 7 (all bits true) → TRUE for all neurons
        let keys: Vec<u64> = vec![7; total_neurons]; // address 7 = 111 in binary
        let values: Vec<u8> = vec![1; total_neurons]; // 1 = TRUE
        let offsets: Vec<u32> = (0..total_neurons).map(|i| i as u32).collect();
        let counts: Vec<u32> = vec![1; total_neurons];

        // Targets: [0, 1, 2] (each example targets a different cluster)
        let targets: Vec<i64> = vec![0, 1, 2];

        let result = evaluator.compute_ce(
            &input_bits,
            &connections,
            &keys,
            &values,
            &offsets,
            &counts,
            &targets,
            num_examples,
            total_input_bits,
            total_neurons,
            bits,
            neurons_per_cluster,
            num_clusters,
            0.5, // empty_value
        );

        assert!(result.is_ok(), "compute_ce failed: {:?}", result.err());
        let (avg_ce, accuracy) = result.unwrap();

        // With all neurons returning TRUE, all clusters should have score 1.0
        // Softmax over [1.0, 1.0, 1.0, 1.0] = [0.25, 0.25, 0.25, 0.25]
        // CE = -log(0.25) ≈ 1.386 for each example
        // Accuracy: prediction is arbitrary when all scores equal, so 0.25 expected

        println!("Small test: avg_ce={:.4}, accuracy={:.4}", avg_ce, accuracy);
        assert!(avg_ce > 1.0 && avg_ce < 2.0, "CE should be around 1.386");
    }
}
