//! marker_probe.rs — eval-in-place probe dispatch (07/07/2026).
//!
//! Dispatches shaders/marker_probe_eval.metal against a trained-but-still-
//! resident MarkerHashTable: per (example, neuron-in-chunk) the kernel probes
//! the table (linear chain, inline oi_merge of duplicate-key slots, inline
//! oi_bin_to_cell) and atomically accumulates integer vote sums (cell weight
//! quantized ×4) into a persistent votes buffer. Used by
//! `marker_train::train_single_genome_chunked_scored` to skip the sorted
//! sparse export for fitness-only evaluations in the over-budget chunked
//! regime. OI mode only — the caller gates.

#[cfg(target_os = "macos")]
pub mod metal_impl
{

	use metal::{
		CompileOptions, ComputePipelineState, Device, MTLLanguageVersion, MTLResourceOptions, MTLSize,
	};
	use std::sync::OnceLock;

	use crate::marker_train::NeuronTrainMeta;

	const SHADER_SOURCE: &str = concat!(
		include_str!("core/shaders/common.metal"),
		"\n",
		include_str!("core/shaders/marker_slots.metal"),
		"\n",
		include_str!("shaders/marker_probe_eval.metal"),
	);

	/// Host-chunk size along the example axis. Bounds per-dispatch wall time so
	/// the cooperative SIGTERM cancel flag is polled at most one dispatch apart
	/// (same rationale as MarkerTrainer::train's 5M-example default).
	const PROBE_EXAMPLES_PER_DISPATCH: u32 = 5_000_000;

	#[repr(C)]
	#[derive(Clone, Copy, Debug)]
	struct ProbeParams
	{
		num_examples: u32,
		num_neurons: u32,
		words_per_example: u32,
		num_clusters: u32,
		example_offset: u32,
		examples_in_dispatch: u32,
	}

	pub struct MarkerProber
	{
		device: Device,
		command_queue: metal::CommandQueue,
		pipeline: ComputePipelineState,
	}

	impl MarkerProber
	{
		pub fn new(device: &Device) -> Result<Self, String>
		{
			let opts = CompileOptions::new();
			opts.set_language_version(MTLLanguageVersion::V3_1);
			let library = device
				.new_library_with_source(SHADER_SOURCE, &opts)
				.map_err(|e| format!("marker_probe_eval.metal compile failed: {}", e))?;
			let kernel = library
				.get_function("marker_probe_eval", None)
				.map_err(|e| format!("get_function marker_probe_eval: {}", e))?;
			let pipeline = device
				.new_compute_pipeline_state_with_function(&kernel)
				.map_err(|e| format!("pipeline marker_probe_eval: {}", e))?;
			Ok(Self {
				device: device.clone(),
				command_queue: device.new_command_queue(),
				pipeline,
			})
		}

		/// Probe the (still-resident) table for every (example, neuron) pair and
		/// accumulate integer votes into `votes` (u32 × num_examples × num_clusters,
		/// caller-zeroed once; accumulation persists across chunk calls). All
		/// buffers must live on this prober's device (= the trainer's device).
		#[allow(clippy::too_many_arguments)]
		pub fn probe_accumulate(
			&self,
			packed_examples: &metal::Buffer,
			connections: &metal::Buffer,
			neuron_meta: &[NeuronTrainMeta],
			slot_markers: &metal::Buffer,
			slot_keys: &metal::Buffer,
			slot_values: &metal::Buffer,
			votes: &metal::Buffer,
			num_examples: usize,
			words_per_example: usize,
			num_clusters: usize,
		) -> Result<(), String>
		{
			if num_examples == 0 || neuron_meta.is_empty()
			{
				return Ok(());
			}
			let meta_bytes = (neuron_meta.len() * std::mem::size_of::<NeuronTrainMeta>()) as u64;
			let meta_buf = self.device.new_buffer_with_data(
				neuron_meta.as_ptr() as *const _,
				meta_bytes,
				MTLResourceOptions::StorageModeShared,
			);

			let num_neurons = neuron_meta.len() as u32;
			let total = num_examples as u32;
			let mut chunk_start: u32 = 0;
			while chunk_start < total
			{
				// Cancel poll BEFORE each dispatch (mirrors MarkerTrainer::train).
				if ram_core::cancel::check_cancel()
				{
					return Err("cancelled".to_string());
				}
				let chunk_count = PROBE_EXAMPLES_PER_DISPATCH.min(total - chunk_start);
				let params = ProbeParams {
					num_examples: total,
					num_neurons,
					words_per_example: words_per_example as u32,
					num_clusters: num_clusters as u32,
					example_offset: chunk_start,
					examples_in_dispatch: chunk_count,
				};
				let params_buf = self.device.new_buffer_with_data(
					&params as *const _ as *const _,
					std::mem::size_of::<ProbeParams>() as u64,
					MTLResourceOptions::StorageModeShared,
				);

				// X = examples (SIMD-coalesced same-neuron table access), Y = neurons.
				let grid = MTLSize::new(chunk_count as u64, num_neurons as u64, 1);
				let max_threads = self.pipeline.max_total_threads_per_threadgroup();
				let tg_x = 256u64.min(max_threads).min((chunk_count as u64).max(1));
				let tg_y = (max_threads / tg_x).max(1).min((num_neurons as u64).max(1));
				let tg = MTLSize::new(tg_x, tg_y, 1);

				let cmd = self.command_queue.new_command_buffer();
				let enc = cmd.new_compute_command_encoder();
				enc.set_compute_pipeline_state(&self.pipeline);
				enc.set_buffer(0, Some(packed_examples), 0);
				enc.set_buffer(1, Some(connections), 0);
				enc.set_buffer(2, Some(&meta_buf), 0);
				enc.set_buffer(3, Some(&params_buf), 0);
				enc.set_buffer(4, Some(slot_markers), 0);
				enc.set_buffer(5, Some(slot_keys), 0);
				enc.set_buffer(6, Some(slot_values), 0);
				enc.set_buffer(7, Some(votes), 0);
				enc.dispatch_threads(grid, tg);
				enc.end_encoding();
				cmd.commit();
				cmd.wait_until_completed();
				chunk_start += chunk_count;
			}
			Ok(())
		}
	}

	/// Lazy global prober (pipeline compilation amortized). Initialized with the
	/// trainer's device so all probe buffers share one Metal device.
	static GLOBAL_PROBER: OnceLock<Result<MarkerProber, String>> = OnceLock::new();

	pub fn get_prober(device: &Device) -> Result<&'static MarkerProber, String>
	{
		let result = GLOBAL_PROBER.get_or_init(|| MarkerProber::new(device));
		match result
		{
			Ok(p) => Ok(p),
			Err(e) => Err(e.clone()),
		}
	}

	/// Allocate a zero-initialized u32 vote buffer of `len` entries.
	pub fn new_zeroed_vote_buffer(device: &Device, len: usize) -> metal::Buffer
	{
		let bytes = (len.max(1) * 4) as u64;
		let buf = device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
		unsafe {
			std::ptr::write_bytes(buf.contents() as *mut u8, 0, bytes as usize);
		}
		buf
	}

	/// Read a vote buffer back into a Vec<u32>.
	pub fn read_vote_buffer(buf: &metal::Buffer, len: usize) -> Vec<u32>
	{
		let ptr = buf.contents() as *const u32;
		unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
	}
} // mod metal_impl

#[cfg(target_os = "macos")]
pub use metal_impl::{get_prober, new_zeroed_vote_buffer, read_vote_buffer};
