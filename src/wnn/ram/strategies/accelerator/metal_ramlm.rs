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
}
