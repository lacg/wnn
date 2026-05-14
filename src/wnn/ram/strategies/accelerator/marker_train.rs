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
	pub words_per_example: u32,
	pub num_classes: u32,
	pub memory_mode: u32,
	pub single_cluster: u32,
	pub normal_class: u32,
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

		// 1D grid: one thread per neuron. Each thread iterates over all
		// examples sequentially within its neuron's slot region. This
		// avoids atomic contention and same-key duplicate inserts at the
		// cost of example-level parallelism within a neuron.
		let n = params.num_neurons as u64;
		let max_threads = self.pipeline.max_total_threads_per_threadgroup();
		let tg_x = max_threads.min(n).max(1);
		let grid = MTLSize::new(n, 1, 1);
		let tg = MTLSize::new(tg_x, 1, 1);
		enc.dispatch_threads(grid, tg);
		enc.end_encoding();
		cmd.commit();
		cmd.wait_until_completed();

		let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
		eprintln!(
			"[MARKER_TRAIN] {} neurons × {} examples in {:.2}ms",
			params.num_neurons, params.num_examples, elapsed_ms
		);
		Ok(elapsed_ms)
	}
}

}  // mod metal_impl

#[cfg(target_os = "macos")]
pub use metal_impl::{MarkerTrainer, NeuronTrainMeta, TrainParams};
