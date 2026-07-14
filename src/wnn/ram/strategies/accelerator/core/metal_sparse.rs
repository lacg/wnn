//! Sparse-RAM GPU forward evaluation (binary search over sorted addresses).
//!
//! Shared substrate primitive — the GPU half of the sparse `Memory` backend
//! (`sparse_memory.rs`). Used by IDS evaluation AND the drone controller (via
//! sparse_memory), so it lives in `ram_core`, not in the LM-specific
//! (`metal_ramlm.rs`) or IDS-eval (`metal_genome_eval.rs`) files. Extracted
//! from the former monolithic metal_ramlm.rs in the 2026-06-19 crate split.

use metal::*;
use std::mem;

/// Compute default cell value for a memory mode
pub fn default_cell_for_mode(memory_mode: u8) -> u32 {
    match memory_mode {
        0 => 2,  // TERNARY: CELL_EMPTY
        3 => 0,  // BINARY: FALSE (classical 1-bit — unwritten = no vote)
        _ => 1,  // QUAD_*: QUAD_WEAK_FALSE
    }
}

/// Metal-based sparse RAMLM evaluator using binary search
/// Works with high-bit architectures (11-30+ bits) that can't use dense storage
pub struct MetalSparseEvaluator {
    device: Device,
    command_queue: CommandQueue,
    sparse_forward_pipeline: ComputePipelineState,
    sparse_forward_per_example_pipeline: ComputePipelineState,
    general_forward_pipeline: ComputePipelineState,
}

impl MetalSparseEvaluator {
    /// Create new sparse Metal evaluator
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;
        let command_queue = device.new_command_queue();

        // Compile sparse forward shader
        let shader_source = concat!(include_str!("shaders/common.metal"), "\n", include_str!("shaders/sparse_forward.metal"));
        let library = device
            .new_library_with_source(shader_source, &CompileOptions::new())
            .map_err(|e| format!("Failed to compile sparse forward shader: {}", e))?;

        let sparse_forward_kernel = library
            .get_function("sparse_forward_pass", None)
            .map_err(|e| format!("Failed to get sparse_forward_pass: {}", e))?;

        let sparse_forward_per_example_kernel = library
            .get_function("sparse_forward_pass_per_example", None)
            .map_err(|e| format!("Failed to get sparse_forward_pass_per_example: {}", e))?;


        let general_forward_kernel = library
            .get_function("general_sparse_forward_pass", None)
            .map_err(|e| format!("Failed to get general_sparse_forward_pass: {}", e))?;

        let sparse_forward_pipeline = device
            .new_compute_pipeline_state_with_function(&sparse_forward_kernel)
            .map_err(|e| format!("Failed to create sparse forward pipeline: {}", e))?;

        let sparse_forward_per_example_pipeline = device
            .new_compute_pipeline_state_with_function(&sparse_forward_per_example_kernel)
            .map_err(|e| format!("Failed to create sparse forward per-example pipeline: {}", e))?;


        let general_forward_pipeline = device
            .new_compute_pipeline_state_with_function(&general_forward_kernel)
            .map_err(|e| format!("Failed to create general forward pipeline: {}", e))?;

        Ok(Self {
            device,
            command_queue,
            sparse_forward_pipeline,
            sparse_forward_per_example_pipeline,
            general_forward_pipeline,
        })
    }

    /// Forward pass using sparse memory with binary search on GPU
    ///
    /// Args:
    ///   packed_input: [num_examples * words_per_example] packed u64
    ///   connections_flat: [num_neurons * bits_per_neuron]
    ///   keys_flat: Sorted keys for all neurons, concatenated
    ///   values_flat: Values corresponding to keys
    ///   offsets: [num_neurons] start offset per neuron
    ///   counts: [num_neurons] entry count per neuron
    ///
    /// Returns: [num_examples * num_clusters] probabilities
    pub fn forward_batch_sparse(
        &self,
        packed_input: &[u64],
        connections_flat: &[i64],
        keys_flat: &[u64],
        values_flat: &[u8],
        offsets: &[u32],
        counts: &[u32],
        num_examples: usize,
        words_per_example: usize,
        num_neurons: usize,
        bits_per_neuron: usize,
        neurons_per_cluster: usize,
        num_clusters: usize,
        memory_mode: u8,
        empty_value: f32,
        run_seed: u64,
    ) -> Result<Vec<f32>, String> {
        if num_examples == 0 {
            return Ok(vec![]);
        }

        let connections_i32: Vec<i32> = connections_flat.iter().map(|&c| c as i32).collect();

        #[repr(C)]
        struct SparseParams {
            num_examples: u32,
            words_per_example: u32,
            num_neurons: u32,
            bits_per_neuron: u32,
            neurons_per_cluster: u32,
            num_clusters: u32,
            empty_value: f32,
            memory_mode: u32,
            default_cell_value: u32,
            run_seed: u64,
        }

        let params = SparseParams {
            num_examples: num_examples as u32,
            words_per_example: words_per_example as u32,
            num_neurons: num_neurons as u32,
            bits_per_neuron: bits_per_neuron as u32,
            neurons_per_cluster: neurons_per_cluster as u32,
            num_clusters: num_clusters as u32,
            empty_value,
            memory_mode: memory_mode as u32,
            default_cell_value: default_cell_for_mode(memory_mode),
            run_seed,
        };

        // Create buffers
        let input_buffer = self.device.new_buffer_with_data(
            packed_input.as_ptr() as *const _,
            (packed_input.len() * mem::size_of::<u64>()) as u64,
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

    /// General forward pass using per-cluster metadata
    ///
    /// Unified kernel for both tiered and adaptive architectures.
    /// Each cluster has its own ClusterInfo (neurons, bits, start_neuron, connection_offset).
    ///
    /// Args:
    ///   cluster_infos: [(neurons_per_cluster, bits_per_neuron, start_neuron, connection_offset)]
    pub fn forward_batch_general(
        &self,
        packed_input: &[u64],
        connections_flat: &[i64],
        keys_flat: &[u64],
        values_flat: &[u8],
        offsets: &[u32],
        counts: &[u32],
        cluster_infos: &[(u32, u32, u32, u32)],
        num_examples: usize,
        words_per_example: usize,
        num_clusters: usize,
        memory_mode: u8,
        empty_value: f32,
        run_seed: u64,
    ) -> Result<Vec<f32>, String> {
        if num_examples == 0 || num_clusters == 0 {
            return Ok(vec![]);
        }

        let connections_i32: Vec<i32> = connections_flat.iter().map(|&c| c as i32).collect();

        #[repr(C)]
        struct GeneralParams {
            num_examples: u32,
            words_per_example: u32,
            num_clusters: u32,
            empty_value: f32,
            memory_mode: u32,
            default_cell_value: u32,
            run_seed: u64,
        }

        #[repr(C)]
        struct ClusterInfo {
            neurons_per_cluster: u32,
            bits_per_neuron: u32,
            start_neuron: u32,
            connection_offset: u32,
        }

        let params = GeneralParams {
            num_examples: num_examples as u32,
            words_per_example: words_per_example as u32,
            num_clusters: num_clusters as u32,
            empty_value,
            memory_mode: memory_mode as u32,
            default_cell_value: default_cell_for_mode(memory_mode),
            run_seed,
        };

        let cluster_info_structs: Vec<ClusterInfo> = cluster_infos.iter().map(|&(n, b, s, c)| ClusterInfo {
            neurons_per_cluster: n,
            bits_per_neuron: b,
            start_neuron: s,
            connection_offset: c,
        }).collect();

        // Create buffers
        let input_buffer = self.device.new_buffer_with_data(
            packed_input.as_ptr() as *const _,
            (packed_input.len() * mem::size_of::<u64>()) as u64,
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

        let cluster_info_buffer = self.device.new_buffer_with_data(
            cluster_info_structs.as_ptr() as *const _,
            (cluster_info_structs.len() * mem::size_of::<ClusterInfo>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let params_buffer = self.device.new_buffer_with_data(
            &params as *const _ as *const _,
            mem::size_of::<GeneralParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let output_size = num_examples * num_clusters;
        let output_buffer = self.device.new_buffer(
            (output_size * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.general_forward_pipeline);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&conn_buffer), 0);
        encoder.set_buffer(2, Some(&keys_buffer), 0);
        encoder.set_buffer(3, Some(&values_buffer), 0);
        encoder.set_buffer(4, Some(&offsets_buffer), 0);
        encoder.set_buffer(5, Some(&counts_buffer), 0);
        encoder.set_buffer(6, Some(&cluster_info_buffer), 0);
        encoder.set_buffer(7, Some(&params_buffer), 0);
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
    fn test_sparse_evaluator_creation() {
        if metal::Device::system_default().is_some() {
            let evaluator = MetalSparseEvaluator::new();
            assert!(evaluator.is_ok(), "Failed to create sparse evaluator: {:?}", evaluator.err());
        }
    }
}
