//! Metal GPU Accelerator for Training Address Computation
//!
//! Computes memory addresses for all (neuron, example) pairs on the GPU,
//! then returns them to the CPU for vote accumulation or nudging.
//!
//! The address computation is the expensive part of training (random memory
//! lookups into packed input). Vote accumulation is sequential and fast on CPU.

use metal::*;
use std::mem;
use std::time::Instant;

use crate::neuron_memory::{NeuronTrainMeta, TrainAddressParams};

// =============================================================================
// Buffer Cache for Training
// =============================================================================

struct CachedTrainBuffer {
	buffer: Buffer,
	capacity_bytes: u64,
}

struct TrainBufferCache {
	input_buffer: Option<CachedTrainBuffer>,
	conn_buffer: Option<CachedTrainBuffer>,
	neuron_meta_buffer: Option<CachedTrainBuffer>,
	params_buffer: Option<CachedTrainBuffer>,
	address_buffer: Option<CachedTrainBuffer>,
}

impl TrainBufferCache {
	fn new() -> Self {
		Self {
			input_buffer: None,
			conn_buffer: None,
			neuron_meta_buffer: None,
			params_buffer: None,
			address_buffer: None,
		}
	}
}

/// Get or create a Metal buffer, reusing cached buffer if capacity is sufficient.
fn get_or_create_buffer<T>(
	device: &Device,
	cached: &mut Option<CachedTrainBuffer>,
	data: &[T],
) -> Buffer {
	let required_bytes = (data.len() * mem::size_of::<T>()) as u64;

	if let Some(ref cache) = cached {
		if cache.capacity_bytes >= required_bytes {
			let ptr = cache.buffer.contents() as *mut T;
			unsafe {
				std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
			}
			return cache.buffer.clone();
		}
	}

	// Allocate with 50% headroom
	let alloc_bytes = ((required_bytes as f64 * 1.5) as u64).max(1024);
	let buffer = device.new_buffer(alloc_bytes, MTLResourceOptions::StorageModeShared);

	let ptr = buffer.contents() as *mut T;
	unsafe {
		std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
	}

	*cached = Some(CachedTrainBuffer {
		buffer: buffer.clone(),
		capacity_bytes: alloc_bytes,
	});

	buffer
}

/// Get or create an output buffer (no data copy, just allocation).
fn get_or_create_output_buffer(
	device: &Device,
	cached: &mut Option<CachedTrainBuffer>,
	num_elements: usize,
	element_size: usize,
) -> Buffer {
	let required_bytes = (num_elements * element_size) as u64;

	if let Some(ref cache) = cached {
		if cache.capacity_bytes >= required_bytes {
			return cache.buffer.clone();
		}
	}

	let alloc_bytes = ((required_bytes as f64 * 1.5) as u64).max(1024);
	let buffer = device.new_buffer(alloc_bytes, MTLResourceOptions::StorageModeShared);

	*cached = Some(CachedTrainBuffer {
		buffer: buffer.clone(),
		capacity_bytes: alloc_bytes,
	});

	buffer
}

// =============================================================================
// MetalTrainer
// =============================================================================

pub struct MetalTrainer {
	device: Device,
	command_queue: CommandQueue,
	address_pipeline: ComputePipelineState,
	cache: TrainBufferCache,
}

impl MetalTrainer {
	pub fn new() -> Result<Self, String> {
		// Check WNN_NO_METAL env var
		if std::env::var("WNN_NO_METAL").is_ok() {
			return Err("Metal training disabled by WNN_NO_METAL".into());
		}

		let device = Device::system_default().ok_or("No Metal device found")?;
		let command_queue = device.new_command_queue();

		let shader_source = include_str!("shaders/train_address.metal");
		let library = device
			.new_library_with_source(shader_source, &CompileOptions::new())
			.map_err(|e| format!("Failed to compile train_address shader: {}", e))?;

		let kernel = library
			.get_function("train_compute_addresses", None)
			.map_err(|e| format!("Failed to get train_compute_addresses kernel: {}", e))?;

		let address_pipeline = device
			.new_compute_pipeline_state_with_function(&kernel)
			.map_err(|e| format!("Failed to create address pipeline: {}", e))?;

		eprintln!("[GPU_TRAIN] MetalTrainer initialized: {}", device.name());

		Ok(Self {
			device,
			command_queue,
			address_pipeline,
			cache: TrainBufferCache::new(),
		})
	}

	/// Compute addresses for all (neuron, example) pairs on GPU.
	///
	/// Args:
	///   packed_input: [num_examples * words_per_example] packed u64 input bits
	///   connections: [total_connections] flattened connection indices (i64, will be cast to i32)
	///   neuron_meta: [total_neurons] per-neuron metadata (bits, conn_offset)
	///   num_examples: number of training examples
	///   words_per_example: ceil(total_input_bits / 64)
	///
	/// Returns: [total_neurons * num_examples] u32 addresses
	pub fn compute_addresses(
		&mut self,
		packed_input: &[u64],
		connections: &[i64],
		neuron_meta: &[NeuronTrainMeta],
		num_examples: usize,
		words_per_example: usize,
	) -> Result<Vec<u32>, String> {
		let total_neurons = neuron_meta.len();

		if total_neurons == 0 || num_examples == 0 {
			return Ok(vec![]);
		}

		let t0 = Instant::now();

		// Convert i64 connections to i32 for GPU (Metal doesn't have 64-bit int support for connections)
		let connections_i32: Vec<i32> = connections.iter().map(|&c| c as i32).collect();

		let params = TrainAddressParams {
			num_examples: num_examples as u32,
			words_per_example: words_per_example as u32,
			total_neurons: total_neurons as u32,
			_pad: 0,
		};

		// Create/reuse buffers
		let input_buffer = get_or_create_buffer(
			&self.device, &mut self.cache.input_buffer, packed_input,
		);
		let conn_buffer = get_or_create_buffer(
			&self.device, &mut self.cache.conn_buffer, &connections_i32,
		);
		let meta_buffer = get_or_create_buffer(
			&self.device, &mut self.cache.neuron_meta_buffer, neuron_meta,
		);

		// Params buffer (small, always reallocate)
		let params_buffer = self.device.new_buffer_with_data(
			&params as *const _ as *const _,
			mem::size_of::<TrainAddressParams>() as u64,
			MTLResourceOptions::StorageModeShared,
		);

		let output_size = total_neurons * num_examples;
		let address_buffer = get_or_create_output_buffer(
			&self.device, &mut self.cache.address_buffer,
			output_size, mem::size_of::<u32>(),
		);

		// Dispatch GPU kernel
		let command_buffer = self.command_queue.new_command_buffer();
		let encoder = command_buffer.new_compute_command_encoder();

		encoder.set_compute_pipeline_state(&self.address_pipeline);
		encoder.set_buffer(0, Some(&input_buffer), 0);
		encoder.set_buffer(1, Some(&conn_buffer), 0);
		encoder.set_buffer(2, Some(&meta_buffer), 0);
		encoder.set_buffer(3, Some(&params_buffer), 0);
		encoder.set_buffer(4, Some(&address_buffer), 0);

		// Grid: (total_neurons, num_examples) — one thread per (neuron, example)
		let grid_size = MTLSize::new(total_neurons as u64, num_examples as u64, 1);
		let max_threads = self.address_pipeline.max_total_threads_per_threadgroup();
		// Choose threadgroup: neurons along x, examples along y
		let tg_x = 32u64.min(total_neurons as u64);
		let tg_y = (max_threads / tg_x).min(num_examples as u64).max(1);
		let thread_group_size = MTLSize::new(tg_x, tg_y, 1);

		encoder.dispatch_threads(grid_size, thread_group_size);
		encoder.end_encoding();
		command_buffer.commit();
		command_buffer.wait_until_completed();

		// Read back results
		let result_ptr = address_buffer.contents() as *const u32;
		let results: Vec<u32> = unsafe {
			std::slice::from_raw_parts(result_ptr, output_size).to_vec()
		};

		let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
		eprintln!(
			"[GPU_TRAIN] compute_addresses: {} neurons × {} examples = {} addresses in {:.2}ms",
			total_neurons, num_examples, output_size, elapsed_ms
		);

		Ok(results)
	}
}
