//! Option B B2 — Metal GPU training kernel dispatch for per-genome training
//! on the marker-FSM AtomicHashTable.
//!
//! Each call trains one genome's memory entirely on GPU:
//!   - Inputs: packed_input, connections, neuron_meta (offsets + slot regions),
//!     train_targets, train_negatives, class_weights, params
//!   - Outputs: writes into the Metal-backed MarkerHashTable's buffers in
//!     place (via the marker FSM)
//!
//! Caller (Option B's dispatcher) is responsible for:
//!   - Allocating one MarkerHashTable::new_metal per neuron (or one merged
//!     table with per-neuron slot regions — current design uses the latter)
//!   - Computing per-neuron slot offsets + capacities
//!   - Binding the buffers and invoking this dispatch

#[cfg(target_os = "macos")]
pub mod metal_impl {

use metal::{
	CompileOptions, ComputePipelineState, Device, MTLLanguageVersion,
	MTLResourceOptions, MTLSize,
};

const SHADER_SOURCE: &str = include_str!("shaders/marker_train.metal");

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NeuronTrainMeta {
	pub bits: u32,
	pub conn_offset: u32,
	pub slot_offset: u32,
	pub slot_capacity: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TrainParams {
	pub num_examples: u32,
	pub num_negatives: u32,
	pub num_neurons: u32,
	pub num_genomes: u32,
	pub words_per_example: u32,
	pub num_classes: u32,
	pub memory_mode: u32,
	pub single_cluster: u32,
	pub normal_class: u32,
	pub conn_stride: u32,
}

pub struct MarkerTrainer {
	device: Device,
	command_queue: metal::CommandQueue,
	pipeline: ComputePipelineState,
}

impl MarkerTrainer {
	pub fn new() -> Result<Self, String> {
		let device = Device::system_default().ok_or("No Metal device available")?;
		let command_queue = device.new_command_queue();
		let opts = CompileOptions::new();
		opts.set_language_version(MTLLanguageVersion::V3_1);
		let library = device
			.new_library_with_source(SHADER_SOURCE, &opts)
			.map_err(|e| format!("marker_train.metal compile failed: {}", e))?;
		let kernel = library
			.get_function("marker_train", None)
			.map_err(|e| format!("get_function marker_train: {}", e))?;
		let pipeline = device
			.new_compute_pipeline_state_with_function(&kernel)
			.map_err(|e| format!("pipeline marker_train: {}", e))?;
		Ok(Self { device, command_queue, pipeline })
	}

	pub fn device(&self) -> &Device {
		&self.device
	}

	/// Run the marker-FSM training kernel for one genome's memory. All
	/// buffers must be Metal-backed (shared storage) and remain valid
	/// for the lifetime of this call.
	#[allow(clippy::too_many_arguments)]
	pub fn train(
		&self,
		packed_input: &metal::Buffer,
		connections: &metal::Buffer,
		neuron_meta: &[NeuronTrainMeta],
		train_targets: &metal::Buffer,
		train_negatives: &metal::Buffer,
		class_weights: &[u32],
		params: TrainParams,
		slot_markers: &metal::Buffer,
		slot_keys: &metal::Buffer,
		slot_values: &metal::Buffer,
	) -> Result<f64, String> {
		let t0 = std::time::Instant::now();

		let meta_bytes = (neuron_meta.len() * std::mem::size_of::<NeuronTrainMeta>()) as u64;
		let meta_buf = self.device.new_buffer_with_data(
			neuron_meta.as_ptr() as *const _,
			meta_bytes,
			MTLResourceOptions::StorageModeShared,
		);

		let weights_bytes = (class_weights.len().max(1) * 4) as u64;
		let cw_buf = if class_weights.is_empty() {
			let one = 1u32;
			self.device.new_buffer_with_data(
				&one as *const _ as *const _,
				4,
				MTLResourceOptions::StorageModeShared,
			)
		} else {
			self.device.new_buffer_with_data(
				class_weights.as_ptr() as *const _,
				weights_bytes,
				MTLResourceOptions::StorageModeShared,
			)
		};

		let params_buf = self.device.new_buffer_with_data(
			&params as *const _ as *const _,
			std::mem::size_of::<TrainParams>() as u64,
			MTLResourceOptions::StorageModeShared,
		);

		let cmd = self.command_queue.new_command_buffer();
		let enc = cmd.new_compute_command_encoder();
		enc.set_compute_pipeline_state(&self.pipeline);
		enc.set_buffer(0, Some(packed_input), 0);
		enc.set_buffer(1, Some(connections), 0);
		enc.set_buffer(2, Some(&meta_buf), 0);
		enc.set_buffer(3, Some(train_targets), 0);
		enc.set_buffer(4, Some(train_negatives), 0);
		enc.set_buffer(5, Some(&cw_buf), 0);
		enc.set_buffer(6, Some(&params_buf), 0);
		enc.set_buffer(7, Some(slot_markers), 0);
		enc.set_buffer(8, Some(slot_keys), 0);
		enc.set_buffer(9, Some(slot_values), 0);

		// 2D grid: x = neuron_idx, y = genome_idx. Each thread owns one
		// (genome, neuron) cell's slot region. Parallelism scales with
		// batch_size: 16 genomes × 100 neurons = 1600 threads ≈ full GPU.
		let n = params.num_neurons as u64;
		let g = params.num_genomes as u64;
		let max_threads = self.pipeline.max_total_threads_per_threadgroup();
		// Threadgroup: prefer x-major, since neurons are independent within
		// a genome and adjacent x threads share less state.
		let tg_x = 32u64.min(n).max(1);
		let tg_y = (max_threads / tg_x).min(g).max(1);
		let grid = MTLSize::new(n, g, 1);
		let tg = MTLSize::new(tg_x, tg_y, 1);
		enc.dispatch_threads(grid, tg);
		enc.end_encoding();
		cmd.commit();
		cmd.wait_until_completed();

		let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
		eprintln!(
			"[MARKER_TRAIN_BATCHED] {} genomes × {} neurons × {} examples in {:.2}ms",
			params.num_genomes, params.num_neurons, params.num_examples, elapsed_ms
		);
		Ok(elapsed_ms)
	}
}

}  // mod metal_impl

#[cfg(target_os = "macos")]
pub use metal_impl::{MarkerTrainer, NeuronTrainMeta, TrainParams};

// =============================================================================
// B4a — train_genome_via_marker: per-genome GPU training producing a
// SparseGpuExport-compatible output for the existing eval pipeline.
// =============================================================================

#[cfg(target_os = "macos")]
pub mod genome_path {

use super::metal_impl::{MarkerTrainer, NeuronTrainMeta, TrainParams};
use crate::atomic_hashtable::{MarkerHashTable, estimate_capacity};
use metal::MTLResourceOptions;
use std::sync::OnceLock;

/// Lazy-initialized global MarkerTrainer. Pipeline compilation is amortized
/// across all per-genome calls.
static GLOBAL_TRAINER: OnceLock<Result<MarkerTrainer, String>> = OnceLock::new();

fn get_trainer() -> Result<&'static MarkerTrainer, String> {
	let result = GLOBAL_TRAINER.get_or_init(MarkerTrainer::new);
	match result {
		Ok(t) => Ok(t),
		Err(e) => Err(e.clone()),
	}
}

/// Inputs for one genome's GPU-marker training run. Matches what the
/// existing CPU path's `train_genome_in_slot` consumes, but in a form
/// convenient for staging Metal buffers.
pub struct GenomeTrainInputs<'a> {
	/// Per-neuron bit count (length = total_neurons for this genome)
	pub per_neuron_bits: &'a [usize],
	/// Per-neuron connection offset into `connections` (length = total_neurons)
	pub neuron_conn_offsets: &'a [usize],
	/// Flat connections (i32 for GPU; CPU side uses i64 — cast at boundary)
	pub connections: &'a [i64],
	/// Packed input bits (u64 word per (example, word_idx))
	pub packed_input: &'a [u64],
	pub words_per_example: usize,
	pub train_targets: &'a [i64],
	pub train_negatives: &'a [i64],
	pub num_train: usize,
	pub num_negatives: usize,
	pub class_weights: Option<&'a [u32]>,
	pub num_classes: usize,
	pub single_cluster: bool,
	pub normal_class: usize,
	pub memory_mode: u8,
}

/// Output of a single genome's GPU training: the per-neuron sorted
/// (key, value) export ready for the existing Metal sparse eval kernel.
pub struct GenomeTrainOutput {
	pub keys: Vec<u64>,
	pub values: Vec<u8>,
	pub offsets: Vec<u32>,
	pub counts: Vec<u32>,
	pub num_neurons: usize,
	pub kernel_ms: f64,
}

/// Train one genome on GPU via the marker-FSM kernel. Returns a
/// SparseGpuExport-compatible tuple. The Metal buffers are allocated
/// transiently within this call.
///
/// For single-cluster (IDS binary) only in V1 — multi-cluster genomes
/// fall back to the existing CPU path.
pub fn train_genome_via_marker(inputs: &GenomeTrainInputs) -> Result<GenomeTrainOutput, String> {
	let trainer = get_trainer()?;
	let device = trainer.device();
	let num_neurons = inputs.per_neuron_bits.len();
	if num_neurons == 0 {
		return Err("zero neurons".into());
	}

	// Per-neuron slot capacity sized via existing heuristic. For 46M
	// flows: estimate_capacity(num_train) returns ~524K which is large;
	// scale down by expected unique-addresses-per-neuron (~sqrt(num_train) * 20).
	// For now: estimate_capacity reads ~3K base + sqrt(num_train) scaling.
	let slot_capacity_per_neuron = estimate_capacity(inputs.num_train);
	let total_slots = num_neurons * slot_capacity_per_neuron;

	// Default cell value depends on memory mode. QUAD_WEIGHTED → 1 (WEAK_FALSE).
	// Ternary mode (mode 0) → 2 (EMPTY).
	let default_value: u8 = match inputs.memory_mode {
		2 => 1,  // QUAD_WEIGHTED
		_ => 2,  // TERNARY default
	};

	// Allocate the genome's flat per-neuron table.
	let gpu_table = MarkerHashTable::new_metal(device, total_slots, default_value);
	let (markers_buf, keys_buf, values_buf) = gpu_table
		.metal_buffers()
		.ok_or("Metal-backed MarkerHashTable returned no buffers")?;

	// Per-neuron metadata. slot_offsets are contiguous (one neuron region
	// after another) inside the flat buffer.
	let neuron_meta: Vec<NeuronTrainMeta> = (0..num_neurons)
		.map(|n| NeuronTrainMeta {
			bits: inputs.per_neuron_bits[n] as u32,
			conn_offset: inputs.neuron_conn_offsets[n] as u32,
			slot_offset: (n * slot_capacity_per_neuron) as u32,
			slot_capacity: slot_capacity_per_neuron as u32,
		})
		.collect();

	// Convert i64 connections to i32 (Metal kernel expects int*)
	let connections_i32: Vec<i32> = inputs.connections.iter().map(|&c| c as i32).collect();

	// Allocate Metal buffers for inputs (per-genome; ~few MB each at scale)
	let packed_buf = device.new_buffer_with_data(
		inputs.packed_input.as_ptr() as *const _,
		(inputs.packed_input.len() * 8) as u64,
		MTLResourceOptions::StorageModeShared,
	);
	let conn_buf = device.new_buffer_with_data(
		connections_i32.as_ptr() as *const _,
		(connections_i32.len() * 4) as u64,
		MTLResourceOptions::StorageModeShared,
	);
	let targets_buf = device.new_buffer_with_data(
		inputs.train_targets.as_ptr() as *const _,
		(inputs.train_targets.len() * 8) as u64,
		MTLResourceOptions::StorageModeShared,
	);
	// train_negatives may be empty for single_cluster; provide a 1-element
	// stub buffer in that case (Metal doesn't allow zero-byte buffers).
	let negs_storage = if inputs.train_negatives.is_empty() { vec![0i64] } else { inputs.train_negatives.to_vec() };
	let negs_buf = device.new_buffer_with_data(
		negs_storage.as_ptr() as *const _,
		(negs_storage.len() * 8) as u64,
		MTLResourceOptions::StorageModeShared,
	);
	let cw_storage: Vec<u32> = inputs.class_weights
		.map(|cw| cw.to_vec())
		.unwrap_or_else(|| vec![1; inputs.num_classes.max(1)]);

	let conn_stride: u32 = inputs.per_neuron_bits.iter().sum::<usize>() as u32;
	let params = TrainParams {
		num_examples: inputs.num_train as u32,
		num_negatives: inputs.num_negatives as u32,
		num_neurons: num_neurons as u32,
		num_genomes: 1,  // single-genome via this entry point
		words_per_example: inputs.words_per_example as u32,
		num_classes: inputs.num_classes as u32,
		memory_mode: inputs.memory_mode as u32,
		single_cluster: if inputs.single_cluster { 1 } else { 0 },
		normal_class: inputs.normal_class as u32,
		conn_stride,
	};

	let kernel_ms = trainer.train(
		&packed_buf, &conn_buf, &neuron_meta,
		&targets_buf, &negs_buf, &cw_storage, params,
		&markers_buf, &keys_buf, &values_buf,
	)?;

	// Build the export
	let slot_offsets: Vec<u32> = (0..num_neurons as u32)
		.map(|n| n * slot_capacity_per_neuron as u32)
		.collect();
	let slot_capacities: Vec<u32> = vec![slot_capacity_per_neuron as u32; num_neurons];
	let (keys, values, offsets, counts) = gpu_table.export_per_neuron(&slot_offsets, &slot_capacities);

	Ok(GenomeTrainOutput {
		keys, values, offsets, counts,
		num_neurons,
		kernel_ms,
	})
}

}  // mod genome_path

#[cfg(target_os = "macos")]
pub use genome_path::{GenomeTrainInputs, GenomeTrainOutput, train_genome_via_marker};
