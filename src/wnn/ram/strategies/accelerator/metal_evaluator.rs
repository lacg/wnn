//! Metal GPU Evaluator for M-series Macs (v2 - Three-Pass)
//!
//! Uses Metal compute shaders to evaluate many connectivity patterns in parallel.
//! Three-pass approach:
//! 1. Training pass: Build hash tables using atomics
//! 2. Evaluation pass: Use tables to make predictions
//! 3. Finalize pass: Compute error rates

use metal::*;
use std::collections::HashMap;
use std::mem;

const HASH_TABLE_SIZE: usize = 256;

/// Metal-based batch evaluator for RAM connectivity patterns
pub struct MetalEvaluator {
    device: Device,
    command_queue: CommandQueue,
    train_pipeline: ComputePipelineState,
    eval_pipeline: ComputePipelineState,
    finalize_pipeline: ComputePipelineState,
}

impl MetalEvaluator {
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

    /// Create new Metal evaluator
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;
        let command_queue = device.new_command_queue();

        // Compile Metal shader
        let shader_source = include_str!("shaders/evaluate.metal");
        let library = device
            .new_library_with_source(shader_source, &CompileOptions::new())
            .map_err(|e| format!("Failed to compile shader: {}", e))?;

        // Get all three kernel functions
        let train_kernel = library
            .get_function("train_rams_pass1", None)
            .map_err(|e| format!("Failed to get train kernel: {}", e))?;

        let eval_kernel = library
            .get_function("evaluate_rams_pass2", None)
            .map_err(|e| format!("Failed to get eval kernel: {}", e))?;

        let finalize_kernel = library
            .get_function("finalize_results", None)
            .map_err(|e| format!("Failed to get finalize kernel: {}", e))?;

        // Create pipelines
        let train_pipeline = device
            .new_compute_pipeline_state_with_function(&train_kernel)
            .map_err(|e| format!("Failed to create train pipeline: {}", e))?;

        let eval_pipeline = device
            .new_compute_pipeline_state_with_function(&eval_kernel)
            .map_err(|e| format!("Failed to create eval pipeline: {}", e))?;

        let finalize_pipeline = device
            .new_compute_pipeline_state_with_function(&finalize_kernel)
            .map_err(|e| format!("Failed to create finalize pipeline: {}", e))?;

        Ok(Self {
            device,
            command_queue,
            train_pipeline,
            eval_pipeline,
            finalize_pipeline,
        })
    }

    /// Evaluate batch of connectivity patterns on GPU
    pub fn evaluate_batch(
        &self,
        connectivities: &[Vec<Vec<i64>>],
        word_to_cluster: &HashMap<String, u64>,
        train_tokens: &[String],
        test_tokens: &[String],
        _bits_per_neuron: usize,  // Derived from connectivity
        eval_subset: usize,
    ) -> Result<Vec<f64>, String> {
        let n_patterns = connectivities.len();
        if n_patterns == 0 {
            return Ok(vec![]);
        }

        let n_neurons = connectivities[0].len();
        let bits_per_neuron = connectivities[0].get(0).map(|v| v.len()).unwrap_or(0);

        if bits_per_neuron == 0 || n_neurons == 0 {
            return Err("Empty connectivity pattern".to_string());
        }

        // Flatten connectivity to [n_patterns, n_neurons, bits_per_neuron]
        let mut conn_flat: Vec<i32> = Vec::with_capacity(n_patterns * n_neurons * bits_per_neuron);
        for conn in connectivities {
            for neuron in conn {
                for i in 0..bits_per_neuron {
                    conn_flat.push(neuron.get(i).copied().unwrap_or(-1) as i32);
                }
            }
        }

        // Encode training tokens as cluster IDs
        let train_clusters: Vec<u32> = train_tokens
            .iter()
            .map(|t| word_to_cluster.get(t).copied().unwrap_or(0) as u32)
            .collect();

        // Encode test tokens
        let test_clusters: Vec<u32> = test_tokens
            .iter()
            .take(eval_subset + 5)  // +5 for context window
            .map(|t| word_to_cluster.get(t).copied().unwrap_or(0) as u32)
            .collect();

        let actual_eval_subset = eval_subset.min(test_clusters.len().saturating_sub(4));

        // Parameters struct (must match shader)
        #[repr(C)]
        struct Params {
            n_patterns: u32,
            n_neurons: u32,
            bits_per_neuron: u32,
            train_len: u32,
            test_len: u32,
            eval_subset: u32,
        }
        let params = Params {
            n_patterns: n_patterns as u32,
            n_neurons: n_neurons as u32,
            bits_per_neuron: bits_per_neuron as u32,
            train_len: train_clusters.len() as u32,
            test_len: test_clusters.len() as u32,
            eval_subset: actual_eval_subset as u32,
        };

        // Create buffers
        let conn_buffer = self.device.new_buffer_with_data(
            conn_flat.as_ptr() as *const _,
            (conn_flat.len() * mem::size_of::<i32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let train_buffer = self.device.new_buffer_with_data(
            train_clusters.as_ptr() as *const _,
            (train_clusters.len() * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let test_buffer = self.device.new_buffer_with_data(
            test_clusters.as_ptr() as *const _,
            (test_clusters.len() * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let params_buffer = self.device.new_buffer_with_data(
            &params as *const _ as *const _,
            mem::size_of::<Params>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Hash tables buffer: [n_patterns, n_neurons, HASH_TABLE_SIZE] of u32
        let hash_table_size = n_patterns * n_neurons * HASH_TABLE_SIZE * mem::size_of::<u32>();
        let hash_tables_buffer = self.device.new_buffer(
            hash_table_size as u64,
            MTLResourceOptions::StorageModeShared,
        );
        // Zero-initialize hash tables
        unsafe {
            std::ptr::write_bytes(hash_tables_buffer.contents() as *mut u8, 0, hash_table_size);
        }

        // Correct/covered counts buffers: [n_patterns] of u32
        let counts_size = n_patterns * mem::size_of::<u32>();
        let correct_buffer = self.device.new_buffer(counts_size as u64, MTLResourceOptions::StorageModeShared);
        let covered_buffer = self.device.new_buffer(counts_size as u64, MTLResourceOptions::StorageModeShared);
        unsafe {
            std::ptr::write_bytes(correct_buffer.contents() as *mut u8, 0, counts_size);
            std::ptr::write_bytes(covered_buffer.contents() as *mut u8, 0, counts_size);
        }

        // Results buffer
        let results_buffer = self.device.new_buffer(
            (n_patterns * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // === PASS 1: Training ===
        let train_tokens_count = train_clusters.len().saturating_sub(4);
        if train_tokens_count > 0 {
            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(&self.train_pipeline);
            encoder.set_buffer(0, Some(&conn_buffer), 0);
            encoder.set_buffer(1, Some(&train_buffer), 0);
            encoder.set_buffer(2, Some(&params_buffer), 0);
            encoder.set_buffer(3, Some(&hash_tables_buffer), 0);

            // Grid: (n_patterns, train_tokens_count)
            let grid_size = MTLSize::new(n_patterns as u64, train_tokens_count as u64, 1);
            let thread_group_size = MTLSize::new(
                8.min(n_patterns as u64),
                32.min(train_tokens_count as u64),
                1
            );
            encoder.dispatch_threads(grid_size, thread_group_size);
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();
        }

        // === PASS 2: Evaluation ===
        if actual_eval_subset > 0 {
            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(&self.eval_pipeline);
            encoder.set_buffer(0, Some(&conn_buffer), 0);
            encoder.set_buffer(1, Some(&test_buffer), 0);
            encoder.set_buffer(2, Some(&params_buffer), 0);
            encoder.set_buffer(3, Some(&hash_tables_buffer), 0);
            encoder.set_buffer(4, Some(&correct_buffer), 0);
            encoder.set_buffer(5, Some(&covered_buffer), 0);

            // Grid: (n_patterns, eval_subset)
            let grid_size = MTLSize::new(n_patterns as u64, actual_eval_subset as u64, 1);
            let thread_group_size = MTLSize::new(
                8.min(n_patterns as u64),
                32.min(actual_eval_subset as u64),
                1
            );
            encoder.dispatch_threads(grid_size, thread_group_size);
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();
        }

        // === PASS 3: Finalize ===
        {
            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(&self.finalize_pipeline);
            encoder.set_buffer(0, Some(&correct_buffer), 0);
            encoder.set_buffer(1, Some(&covered_buffer), 0);
            encoder.set_buffer(2, Some(&params_buffer), 0);
            encoder.set_buffer(3, Some(&results_buffer), 0);

            let grid_size = MTLSize::new(n_patterns as u64, 1, 1);
            let thread_group_size = MTLSize::new(64.min(n_patterns as u64), 1, 1);
            encoder.dispatch_threads(grid_size, thread_group_size);
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();
        }

        // Read results
        let results_ptr = results_buffer.contents() as *const f32;
        let results: Vec<f64> = unsafe {
            std::slice::from_raw_parts(results_ptr, n_patterns)
                .iter()
                .map(|&x| x as f64)
                .collect()
        };

        Ok(results)
    }
}
