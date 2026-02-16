//! Metal GPU Accelerator for Adaptation Stats
//!
//! Replaces CPU-bound `compute_neuron_stats` + `compute_cluster_stats` with
//! two GPU dispatches:
//!   Pass 1: Per-neuron forward + error/fill stats (3200 threads)
//!   Pass 2: Per-cluster majority vote + uniqueness/accuracy (clusters × samples threads)
//!
//! The votes buffer connects both passes, avoiding redundant forward computation.

use metal::*;
use std::mem;

use crate::neuron_memory::{
	ClusterStorage,
	MODE_QUAD_BINARY, MODE_QUAD_WEIGHTED, EMPTY_U8,
};

// =============================================================================
// GPU Struct Layout (must match Metal shader exactly)
// =============================================================================

#[repr(C)]
struct NeuronMeta {
	cluster_id: u32,
	bits: u32,
	conn_offset: u32,
	is_sparse: u32,
	dense_word_offset: u32,
	words_per_neuron: u32,
	sparse_neuron_idx: u32,
}

#[repr(C)]
struct StatsParams {
	num_neurons: u32,
	sample_size: u32,
	words_per_example: u32,
	num_clusters: u32,
	empty_cell_value: u32,
}

#[repr(C)]
struct ClusterMetaGpu {
	first_neuron: u32,
	num_neurons: u32,
}

// =============================================================================
// MetalStatsComputer
// =============================================================================

pub struct MetalStatsComputer {
	device: Device,
	command_queue: CommandQueue,
	neuron_stats_pipeline: ComputePipelineState,
	cluster_stats_pipeline: ComputePipelineState,
}

impl MetalStatsComputer {
	pub fn new() -> Result<Self, String> {
		let device = Device::system_default().ok_or("No Metal device")?;
		let queue = device.new_command_queue();

		let src = include_str!("shaders/neuron_stats.metal");
		let lib = device
			.new_library_with_source(src, &CompileOptions::new())
			.map_err(|e| format!("Shader compile: {e}"))?;

		let neuron_fn = lib
			.get_function("compute_neuron_stats_kernel", None)
			.map_err(|e| format!("Kernel not found: {e}"))?;
		let cluster_fn = lib
			.get_function("compute_cluster_stats_kernel", None)
			.map_err(|e| format!("Kernel not found: {e}"))?;

		let neuron_pipeline = device
			.new_compute_pipeline_state_with_function(&neuron_fn)
			.map_err(|e| format!("Pipeline: {e}"))?;
		let cluster_pipeline = device
			.new_compute_pipeline_state_with_function(&cluster_fn)
			.map_err(|e| format!("Pipeline: {e}"))?;

		Ok(Self {
			device,
			command_queue: queue,
			neuron_stats_pipeline: neuron_pipeline,
			cluster_stats_pipeline: cluster_pipeline,
		})
	}

	/// Compute all adaptation stats on GPU (neuron + cluster).
	///
	/// Returns:
	///   - `neuron_errors`: `[total_neurons]` error count per neuron
	///   - `neuron_filled`: `[total_neurons]` filled cell count per neuron
	///   - `cluster_errors`: `[num_clusters]` error count per cluster
	///   - `uniqueness_counts`: `[total_neurons]` disagreement with majority count
	///   - `accuracy_counts`: `[total_neurons]` agreement with target count
	pub fn compute_all_stats(
		&self,
		packed_input: &[u64],
		words_per_example: usize,
		connections: &[i64],
		bits_per_neuron: &[usize],
		neurons_per_cluster: &[usize],
		storages: &[ClusterStorage],
		target_bits: &[u8],
		sample_indices: &[u32],
		num_clusters: usize,
		memory_mode: u8,
	) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
		let total_neurons: usize = neurons_per_cluster.iter().sum();
		let sample_size = sample_indices.len();

		if total_neurons == 0 || sample_size == 0 {
			return (
				vec![0; total_neurons],
				vec![0; total_neurons],
				vec![0; num_clusters],
				vec![0; total_neurons],
				vec![0; total_neurons],
			);
		}

		// =====================================================================
		// Data Marshaling: Build GPU buffers from heterogeneous ClusterStorage
		// =====================================================================

		// Connection offsets (per-neuron)
		let mut conn_offsets = Vec::with_capacity(total_neurons + 1);
		conn_offsets.push(0usize);
		for &b in bits_per_neuron {
			conn_offsets.push(conn_offsets.last().unwrap() + b);
		}

		// Flatten dense memory + build sparse export
		let mut all_dense_words: Vec<i64> = Vec::new();
		let mut all_sparse_keys: Vec<u64> = Vec::new();
		let mut all_sparse_values: Vec<u8> = Vec::new();
		let mut all_sparse_offsets: Vec<u32> = Vec::new();
		let mut all_sparse_counts: Vec<u32> = Vec::new();

		let mut neuron_metas: Vec<NeuronMeta> = Vec::with_capacity(total_neurons);
		let mut cluster_metas: Vec<ClusterMetaGpu> = Vec::with_capacity(num_clusters);

		let mut global_n = 0usize;
		let mut dense_word_cursor = 0u32;
		let mut sparse_neuron_cursor = 0u32;

		let empty_cell = match memory_mode {
			MODE_QUAD_BINARY | MODE_QUAD_WEIGHTED => 1u32, // QUAD_WEAK_FALSE
			_ => EMPTY_U8 as u32,
		};

		for cluster in 0..num_clusters {
			let n_neurons = neurons_per_cluster[cluster];
			cluster_metas.push(ClusterMetaGpu {
				first_neuron: global_n as u32,
				num_neurons: n_neurons as u32,
			});

			match &storages[cluster] {
				ClusterStorage::Dense { words, words_per_neuron, .. } => {
					let wpn = *words_per_neuron;
					// Append all words for this cluster
					all_dense_words.extend_from_slice(words);

					for local_n in 0..n_neurons {
						neuron_metas.push(NeuronMeta {
							cluster_id: cluster as u32,
							bits: bits_per_neuron[global_n] as u32,
							conn_offset: conn_offsets[global_n] as u32,
							is_sparse: 0,
							dense_word_offset: dense_word_cursor + (local_n * wpn) as u32,
							words_per_neuron: wpn as u32,
							sparse_neuron_idx: 0,
						});
						global_n += 1;
					}
					dense_word_cursor += words.len() as u32;
				}
				ClusterStorage::Sparse { neurons: maps, .. } => {
					for local_n in 0..n_neurons {
						// Export this neuron's sparse data
						let map = &maps[local_n];
						let sp_offset = all_sparse_keys.len() as u32;

						// Sort entries by key for binary search
						let mut entries: Vec<(u32, u8)> = map.iter().map(|(&k, &v)| (k, v)).collect();
						entries.sort_unstable_by_key(|(k, _)| *k);
						let count = entries.len() as u32;

						for (k, v) in entries {
							all_sparse_keys.push(k as u64);
							all_sparse_values.push(v);
						}
						all_sparse_offsets.push(sp_offset);
						all_sparse_counts.push(count);

						neuron_metas.push(NeuronMeta {
							cluster_id: cluster as u32,
							bits: bits_per_neuron[global_n] as u32,
							conn_offset: conn_offsets[global_n] as u32,
							is_sparse: 1,
							dense_word_offset: 0,
							words_per_neuron: 0,
							sparse_neuron_idx: sparse_neuron_cursor,
						});
						sparse_neuron_cursor += 1;
						global_n += 1;
					}
				}
			}
		}

		// Convert connections to i32 for GPU
		let connections_i32: Vec<i32> = connections.iter().map(|&c| c as i32).collect();

		// Ensure non-empty buffers (Metal doesn't like zero-length)
		if all_dense_words.is_empty() { all_dense_words.push(0); }
		if all_sparse_keys.is_empty() { all_sparse_keys.push(0); }
		if all_sparse_values.is_empty() { all_sparse_values.push(0); }
		if all_sparse_offsets.is_empty() { all_sparse_offsets.push(0); }
		if all_sparse_counts.is_empty() { all_sparse_counts.push(0); }

		let params = StatsParams {
			num_neurons: total_neurons as u32,
			sample_size: sample_size as u32,
			words_per_example: words_per_example as u32,
			num_clusters: num_clusters as u32,
			empty_cell_value: empty_cell,
		};

		// =====================================================================
		// Create Metal Buffers (StorageModeShared = unified memory, zero-copy)
		// =====================================================================

		let buf_packed = self.make_buffer(packed_input);
		let buf_connections = self.make_buffer(&connections_i32);
		let buf_targets = self.make_buffer(target_bits);
		let buf_samples = self.make_buffer(sample_indices);
		let buf_dense = self.make_buffer(&all_dense_words);
		let buf_sparse_keys = self.make_buffer(&all_sparse_keys);
		let buf_sparse_values = self.make_buffer(&all_sparse_values);
		let buf_sparse_offsets = self.make_buffer(&all_sparse_offsets);
		let buf_sparse_counts = self.make_buffer(&all_sparse_counts);
		let buf_neuron_meta = self.make_buffer(&neuron_metas);
		let buf_params = self.make_buffer_from_struct(&params);

		// Output buffers
		let buf_error_counts = self.make_zero_buffer::<u32>(total_neurons);
		let buf_filled_counts = self.make_zero_buffer::<u32>(total_neurons);
		let votes_len = total_neurons * sample_size;
		let buf_votes = self.make_zero_buffer::<u8>(votes_len.max(1));

		// =====================================================================
		// Pass 1: Per-Neuron Stats
		// =====================================================================

		let cmd_buf = self.command_queue.new_command_buffer();
		{
			let encoder = cmd_buf.new_compute_command_encoder();
			encoder.set_compute_pipeline_state(&self.neuron_stats_pipeline);

			encoder.set_buffer(0, Some(&buf_packed), 0);
			encoder.set_buffer(1, Some(&buf_connections), 0);
			encoder.set_buffer(2, Some(&buf_targets), 0);
			encoder.set_buffer(3, Some(&buf_samples), 0);
			encoder.set_buffer(4, Some(&buf_dense), 0);
			encoder.set_buffer(5, Some(&buf_sparse_keys), 0);
			encoder.set_buffer(6, Some(&buf_sparse_values), 0);
			encoder.set_buffer(7, Some(&buf_sparse_offsets), 0);
			encoder.set_buffer(8, Some(&buf_sparse_counts), 0);
			encoder.set_buffer(9, Some(&buf_neuron_meta), 0);
			encoder.set_buffer(10, Some(&buf_params), 0);
			encoder.set_buffer(11, Some(&buf_error_counts), 0);
			encoder.set_buffer(12, Some(&buf_filled_counts), 0);
			encoder.set_buffer(13, Some(&buf_votes), 0);

			let threads = MTLSize::new(total_neurons as u64, 1, 1);
			let max_tpg = self.neuron_stats_pipeline.max_total_threads_per_threadgroup();
			let tpg = MTLSize::new((total_neurons as u64).min(max_tpg), 1, 1);
			encoder.dispatch_threads(threads, tpg);
			encoder.end_encoding();
		}

		// =====================================================================
		// Pass 2: Cluster Stats (same command buffer — sequential on GPU)
		// =====================================================================

		let buf_cluster_meta = self.make_buffer(&cluster_metas);
		let buf_cluster_errors = self.make_zero_buffer::<u32>(num_clusters);
		let buf_uniqueness = self.make_zero_buffer::<u32>(total_neurons);
		let buf_accuracy = self.make_zero_buffer::<u32>(total_neurons);

		{
			let encoder = cmd_buf.new_compute_command_encoder();
			encoder.set_compute_pipeline_state(&self.cluster_stats_pipeline);

			encoder.set_buffer(0, Some(&buf_votes), 0);
			encoder.set_buffer(1, Some(&buf_targets), 0);
			encoder.set_buffer(2, Some(&buf_samples), 0);
			encoder.set_buffer(3, Some(&buf_cluster_meta), 0);
			encoder.set_buffer(4, Some(&buf_params), 0);
			encoder.set_buffer(5, Some(&buf_cluster_errors), 0);
			encoder.set_buffer(6, Some(&buf_uniqueness), 0);
			encoder.set_buffer(7, Some(&buf_accuracy), 0);

			let threads = MTLSize::new(num_clusters as u64, sample_size as u64, 1);
			let max_tpg = self.cluster_stats_pipeline.max_total_threads_per_threadgroup();
			// For 2D grid: try to balance x and y dimensions
			let tpg_x = (num_clusters as u64).min(max_tpg);
			let tpg_y = (sample_size as u64).min(max_tpg / tpg_x.max(1));
			let tpg = MTLSize::new(tpg_x, tpg_y.max(1), 1);
			encoder.dispatch_threads(threads, tpg);
			encoder.end_encoding();
		}

		cmd_buf.commit();
		cmd_buf.wait_until_completed();

		// =====================================================================
		// Read back results
		// =====================================================================

		let error_counts = self.read_buffer::<u32>(&buf_error_counts, total_neurons);
		let filled_counts = self.read_buffer::<u32>(&buf_filled_counts, total_neurons);
		let cluster_errors = self.read_buffer::<u32>(&buf_cluster_errors, num_clusters);
		let uniqueness_counts = self.read_buffer::<u32>(&buf_uniqueness, total_neurons);
		let accuracy_counts = self.read_buffer::<u32>(&buf_accuracy, total_neurons);

		(error_counts, filled_counts, cluster_errors, uniqueness_counts, accuracy_counts)
	}

	// =========================================================================
	// Buffer helpers
	// =========================================================================

	fn make_buffer<T>(&self, data: &[T]) -> Buffer {
		let bytes = (data.len() * mem::size_of::<T>()) as u64;
		if bytes == 0 {
			return self.device.new_buffer(64, MTLResourceOptions::StorageModeShared);
		}
		self.device.new_buffer_with_data(
			data.as_ptr() as *const _,
			bytes,
			MTLResourceOptions::StorageModeShared,
		)
	}

	fn make_buffer_from_struct<T>(&self, data: &T) -> Buffer {
		let bytes = mem::size_of::<T>() as u64;
		self.device.new_buffer_with_data(
			data as *const T as *const _,
			bytes,
			MTLResourceOptions::StorageModeShared,
		)
	}

	fn make_zero_buffer<T>(&self, count: usize) -> Buffer {
		let bytes = (count * mem::size_of::<T>()) as u64;
		let bytes = bytes.max(64); // Metal minimum
		let buf = self.device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
		// Zero-initialize
		unsafe {
			std::ptr::write_bytes(buf.contents() as *mut u8, 0, bytes as usize);
		}
		buf
	}

	fn read_buffer<T: Copy + Default>(&self, buffer: &Buffer, count: usize) -> Vec<T> {
		let ptr = buffer.contents() as *const T;
		let mut result = vec![T::default(); count];
		unsafe {
			std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), count);
		}
		result
	}
}
