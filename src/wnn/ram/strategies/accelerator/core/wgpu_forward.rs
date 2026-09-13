//! `WgpuForward` — the portable GPU backend (Metal / Vulkan / DX12 / WebGPU)
//! for the sparse forward, feature `gpu-wgpu`.
//!
//! One WGSL kernel (`shaders/sparse_forward.wgsl`), the twin of
//! `sparse_forward.metal`; this file is the buffer plumbing. Every buffer is
//! uploaded per call (the public estimator's `predict` is one call per batch);
//! caching the memory upload across calls is the obvious next step once a
//! benchmark says it matters. Reads back through a staging buffer.
//!
//! Chunked dispatch: a compute dispatch is capped at 65 535 workgroups per
//! dimension, so (examples × clusters) is walked in chunks of 65 535 × 64
//! invocations with `base_idx` rewritten in the uniform between submits.

use crate::forward::{Forward, ForwardError, ForwardRequest};
use crate::neuron_memory::pack_packed_to_u64;
use wgpu::util::DeviceExt;

const WORKGROUP: u32 = 64;
const MAX_WORKGROUPS: u32 = 65_535;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params
{
	num_examples: u32,
	words_per_example: u32,
	num_neurons: u32,
	bits_per_neuron: u32,
	neurons_per_cluster: u32,
	num_clusters: u32,
	memory_mode: u32,
	default_cell: u32,
	run_seed_lo: u32,
	run_seed_hi: u32,
	empty_value: f32,
	base_idx: u32,
}

pub struct WgpuForward
{
	device: wgpu::Device,
	queue: wgpu::Queue,
	pipeline: wgpu::ComputePipeline,
	layout: wgpu::BindGroupLayout,
	backend: wgpu::Backend,
}

impl WgpuForward
{
	/// Any available adapter (discrete first, then integrated, then software).
	pub fn new() -> Result<Self, ForwardError>
	{
		pollster::block_on(Self::new_async())
	}

	async fn new_async() -> Result<Self, ForwardError>
	{
		let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
		let adapter = instance
			.request_adapter(&wgpu::RequestAdapterOptions {
				power_preference: wgpu::PowerPreference::HighPerformance,
				compatible_surface: None,
				force_fallback_adapter: false,
				apply_limit_buckets: false,
			})
			.await
			.map_err(|e| ForwardError::Unavailable(format!("no wgpu adapter: {e}")))?;
		let backend = adapter.get_info().backend;
		// Take the adapter's own limits so a large sorted-key buffer is not
		// capped at the 128 MiB WebGPU default.
		let limits = adapter.limits();
		let (device, queue) = adapter
			.request_device(&wgpu::DeviceDescriptor {
				label: Some("ram_core sparse forward"),
				required_features: wgpu::Features::empty(),
				required_limits: limits,
				experimental_features: wgpu::ExperimentalFeatures::default(),
				memory_hints: wgpu::MemoryHints::Performance,
				trace: wgpu::Trace::Off,
			})
			.await
			.map_err(|e| ForwardError::Unavailable(format!("wgpu device: {e}")))?;
		let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
			label: Some("sparse_forward.wgsl"),
			source: wgpu::ShaderSource::Wgsl(include_str!("shaders/sparse_forward.wgsl").into()),
		});
		let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
			binding,
			visibility: wgpu::ShaderStages::COMPUTE,
			ty: wgpu::BindingType::Buffer {
				ty: wgpu::BufferBindingType::Storage { read_only },
				has_dynamic_offset: false,
				min_binding_size: None,
			},
			count: None,
		};
		let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("sparse forward"),
			entries: &[
				wgpu::BindGroupLayoutEntry {
					binding: 0,
					visibility: wgpu::ShaderStages::COMPUTE,
					ty: wgpu::BindingType::Buffer {
						ty: wgpu::BufferBindingType::Uniform,
						has_dynamic_offset: false,
						min_binding_size: None,
					},
					count: None,
				},
				storage(1, true),
				storage(2, true),
				storage(3, true),
				storage(4, true),
				storage(5, true),
				storage(6, true),
				storage(7, false),
			],
		});
		let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
			label: Some("sparse forward"),
			bind_group_layouts: &[Some(&layout)],
			immediate_size: 0,
		});
		let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
			label: Some("sparse_forward"),
			layout: Some(&pipeline_layout),
			module: &module,
			entry_point: Some("sparse_forward"),
			compilation_options: wgpu::PipelineCompilationOptions::default(),
			cache: None,
		});
		Ok(Self {
			device,
			queue,
			pipeline,
			layout,
			backend,
		})
	}

	/// The wgpu backend actually in use ("Metal", "Vulkan", "Dx12", …).
	pub fn backend_name(&self) -> String
	{
		format!("{:?}", self.backend)
	}

	fn storage_buffer<T: bytemuck::Pod>(&self, label: &str, data: &[T], read_only: bool) -> wgpu::Buffer
	{
		// A zero-length binding is invalid; keep one element of padding.
		let bytes: Vec<u8> = if data.is_empty()
		{
			vec![0u8; std::mem::size_of::<T>().max(4)]
		}
		else
		{
			bytemuck::cast_slice(data).to_vec()
		};
		let usage = if read_only
		{
			wgpu::BufferUsages::STORAGE
		}
		else
		{
			wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
		};
		self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
			label: Some(label),
			contents: &bytes,
			usage,
		})
	}
}

/// u64 words → (lo, hi) u32 pairs, the shader's layout.
fn split_u64(words: &[u64]) -> Vec<u32>
{
	let mut out = Vec::with_capacity(words.len() * 2);
	for &w in words
	{
		out.push(w as u32);
		out.push((w >> 32) as u32);
	}
	out
}

/// u8 cells → 4 per u32, byte i = cell i (little-endian, like the shader reads).
fn pack_u8(values: &[u8]) -> Vec<u32>
{
	values
		.chunks(4)
		.map(|c| {
			let mut w = 0u32;
			for (i, &b) in c.iter().enumerate()
			{
				w |= (b as u32) << (8 * i);
			}
			w
		})
		.collect()
}

impl Forward for WgpuForward
{
	fn name(&self) -> &'static str
	{
		"wgpu"
	}

	fn forward(&self, req: &ForwardRequest) -> Result<Vec<f32>, ForwardError>
	{
		req.validate()?;
		let l = req.layout;
		let r = req.read;
		let num_examples = req.input.num_rows();
		let total = num_examples * l.num_clusters;
		if total == 0
		{
			return Ok(vec![]);
		}
		let (words, wpe) = pack_packed_to_u64(req.input);
		let input_u32 = split_u64(&words);
		let conns_i32: Vec<i32> = req.connections.iter().map(|&c| c as i32).collect();
		let keys_u32 = split_u64(&req.memory.keys);
		let values_u32 = pack_u8(&req.memory.values);

		let params = Params {
			num_examples: num_examples as u32,
			words_per_example: wpe as u32,
			num_neurons: l.num_neurons() as u32,
			bits_per_neuron: l.bits_per_neuron as u32,
			neurons_per_cluster: l.neurons_per_cluster as u32,
			num_clusters: l.num_clusters as u32,
			memory_mode: r.cell_mode.as_u8() as u32,
			default_cell: r.cell_mode.miss_cell(r.coverage_aware) as u32,
			run_seed_lo: r.run_seed as u32,
			run_seed_hi: (r.run_seed >> 32) as u32,
			empty_value: r.empty_value,
			base_idx: 0,
		};
		let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
			label: Some("params"),
			contents: bytemuck::bytes_of(&params),
			usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
		});
		let input_buf = self.storage_buffer("packed_input", &input_u32, true);
		let conn_buf = self.storage_buffer("connections", &conns_i32, true);
		let keys_buf = self.storage_buffer("keys", &keys_u32, true);
		let values_buf = self.storage_buffer("values", &values_u32, true);
		let offsets_buf = self.storage_buffer("offsets", &req.memory.offsets, true);
		let counts_buf = self.storage_buffer("counts", &req.memory.counts, true);
		let out_bytes = (total * std::mem::size_of::<f32>()) as u64;
		let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
			label: Some("probs_out"),
			size: out_bytes,
			usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
			mapped_at_creation: false,
		});
		let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
			label: Some("staging"),
			size: out_bytes,
			usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
			mapped_at_creation: false,
		});
		let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: Some("sparse forward"),
			layout: &self.layout,
			entries: &[
				wgpu::BindGroupEntry {
					binding: 0,
					resource: params_buf.as_entire_binding(),
				},
				wgpu::BindGroupEntry {
					binding: 1,
					resource: input_buf.as_entire_binding(),
				},
				wgpu::BindGroupEntry {
					binding: 2,
					resource: conn_buf.as_entire_binding(),
				},
				wgpu::BindGroupEntry {
					binding: 3,
					resource: keys_buf.as_entire_binding(),
				},
				wgpu::BindGroupEntry {
					binding: 4,
					resource: values_buf.as_entire_binding(),
				},
				wgpu::BindGroupEntry {
					binding: 5,
					resource: offsets_buf.as_entire_binding(),
				},
				wgpu::BindGroupEntry {
					binding: 6,
					resource: counts_buf.as_entire_binding(),
				},
				wgpu::BindGroupEntry {
					binding: 7,
					resource: out_buf.as_entire_binding(),
				},
			],
		});

		let chunk = (MAX_WORKGROUPS * WORKGROUP) as usize;
		let mut base = 0usize;
		while base < total
		{
			let n = (total - base).min(chunk);
			let groups = ((n as u32) + WORKGROUP - 1) / WORKGROUP;
			let p = Params {
				base_idx: base as u32,
				..params
			};
			self.queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&p));
			let mut encoder = self
				.device
				.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("sparse forward") });
			{
				let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
					label: Some("sparse_forward"),
					timestamp_writes: None,
				});
				pass.set_pipeline(&self.pipeline);
				pass.set_bind_group(0, &bind, &[]);
				pass.dispatch_workgroups(groups, 1, 1);
			}
			if base + n >= total
			{
				encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_bytes);
			}
			self.queue.submit(Some(encoder.finish()));
			base += n;
		}

		let slice = staging.slice(..);
		let (tx, rx) = std::sync::mpsc::channel();
		slice.map_async(wgpu::MapMode::Read, move |res| {
			let _ = tx.send(res);
		});
		self
			.device
			.poll(wgpu::PollType::wait_indefinitely())
			.map_err(|e| ForwardError::Backend(format!("wgpu poll: {e:?}")))?;
		rx.recv()
			.map_err(|_| ForwardError::Backend("wgpu map callback dropped".into()))?
			.map_err(|e| ForwardError::Backend(format!("wgpu map: {e:?}")))?;
		let view = slice
			.get_mapped_range()
			.map_err(|e| ForwardError::Backend(format!("wgpu mapped range: {e:?}")))?;
		let out: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
		drop(view);
		staging.unmap();
		Ok(out)
	}
}
