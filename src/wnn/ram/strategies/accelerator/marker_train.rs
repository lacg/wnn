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
	/// Which cluster this neuron belongs to. For single-cluster (binary IDS)
	/// this is always 0. For multi-cluster, set by the dispatcher from
	/// neurons_per_cluster[].
	pub cluster_idx: u32,
	/// Padding for natural alignment (24-byte struct). Reserved for future use.
	pub _pad: u32,
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
		let trace = std::env::var("WNN_OPTION_B_TRACE").is_ok();
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
		let t_after_aux_bufs = t0.elapsed().as_secs_f64() * 1000.0;

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
		if trace {
			eprintln!(
				"[OPT_B_TRACE] dispatch grid=({},{}) tg=({},{}) max_threads_per_tg={} max_total={}",
				n, g, tg_x, tg_y, max_threads,
				self.pipeline.thread_execution_width()
			);
		}
		enc.dispatch_threads(grid, tg);
		enc.end_encoding();
		let t_after_encode = t0.elapsed().as_secs_f64() * 1000.0;
		cmd.commit();
		let t_after_commit = t0.elapsed().as_secs_f64() * 1000.0;
		cmd.wait_until_completed();
		let t_after_wait = t0.elapsed().as_secs_f64() * 1000.0;

		let elapsed_ms = t_after_wait;
		eprintln!(
			"[MARKER_TRAIN_BATCHED] {} genomes × {} neurons × {} examples in {:.2}ms",
			params.num_genomes, params.num_neurons, params.num_examples, elapsed_ms
		);
		if trace {
			eprintln!(
				"[OPT_B_TRACE]   aux_buf={:.2}ms encode={:.2}ms commit_call={:.2}ms wait_completed={:.2}ms (kernel_only={:.2}ms)",
				t_after_aux_bufs,
				t_after_encode - t_after_aux_bufs,
				t_after_commit - t_after_encode,
				t_after_wait - t_after_commit,
				t_after_wait - t_after_commit
			);
		}
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
use crate::atomic_hashtable::MarkerHashTable;
use metal::MTLResourceOptions;
use std::sync::OnceLock;

/// Worst-case unique-address load-factor 0.5 sizing for MarkerHashTable
/// fixed-capacity Metal buffers. Use this instead of
/// `atomic_hashtable::estimate_capacity`, which was designed for the
/// growable AtomicHashTable (CPU). The marker variant has fixed-size
/// Metal buffers — undersizing causes the GPU kernel to spin in probe
/// loops indefinitely.
pub(super) fn marker_capacity_for_train(num_train: usize, max_bits: usize) -> usize {
	let upper = if max_bits >= 30 {
		num_train
	} else {
		num_train.min(1usize << max_bits)
	};
	let raw = (upper.saturating_mul(2)).max(256);
	raw.next_power_of_two()
}

/// Lazy-initialized global MarkerTrainer. Pipeline compilation is amortized
/// across all per-genome calls.
static GLOBAL_TRAINER: OnceLock<Result<MarkerTrainer, String>> = OnceLock::new();

pub(super) fn get_trainer() -> Result<&'static MarkerTrainer, String> {
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

	// Per-neuron slot capacity sized for worst-case unique-address load
	// at ~50% factor. Earlier versions used estimate_capacity() which is
	// for the growable AtomicHashTable — MarkerHashTable has fixed-size
	// Metal buffers and undersizing causes GPU probe-loop spinning.
	let max_bits = inputs.per_neuron_bits.iter().copied().max().unwrap_or(48);
	let slot_capacity_per_neuron = marker_capacity_for_train(inputs.num_train, max_bits);
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
	// after another) inside the flat buffer. single_cluster → cluster_idx=0.
	let neuron_meta: Vec<NeuronTrainMeta> = (0..num_neurons)
		.map(|n| NeuronTrainMeta {
			bits: inputs.per_neuron_bits[n] as u32,
			conn_offset: inputs.neuron_conn_offsets[n] as u32,
			slot_offset: (n * slot_capacity_per_neuron) as u32,
			slot_capacity: slot_capacity_per_neuron as u32,
			cluster_idx: 0,
			_pad: 0,
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

// =============================================================================
// B4b — batched_train_offspring: batched-dispatch version that takes the
// same inputs as evaluate_genomes_parallel_hybrid and returns Vec<GenomeExport>
// =============================================================================

#[cfg(target_os = "macos")]
pub mod batched_path {

use super::metal_impl::{NeuronTrainMeta, TrainParams};
use super::genome_path::{get_trainer, marker_capacity_for_train};
use crate::adaptive::{ConfigGroup, GenomeExport, SparseGpuExport, build_groups, per_cluster_max_bits};
use crate::atomic_hashtable::MarkerHashTable;
use metal::MTLResourceOptions;

/// Batched GPU training for a whole offspring population. Returns one
/// `GenomeExport` per genome, ready for the existing eval phase.
///
/// Supports both single-cluster (binary IDS) and multi-cluster (K-class).
/// Multi-cluster requires uniform per-cluster neuron counts across the batch
/// (all genomes must share the same `neurons_per_cluster[]` shape). The
/// kernel mirrors the CPU semantics at `adaptive.rs:2837-2940`:
///   - Positive: nudge TRUE for target cluster's neurons
///   - Negatives: for each train_negatives[ex_idx][k], nudge FALSE for that cluster's neurons
///   - Other clusters: skip
///
/// Returns Err on shape mismatch so the caller falls back to per-genome path.
#[allow(clippy::too_many_arguments)]
pub fn batched_train_offspring(
	genomes_bits_flat: &[usize],
	genomes_neurons_flat: &[usize],
	genomes_connections_flat: &[i64],
	num_genomes: usize,
	num_clusters: usize,
	train_input_bits: &crate::packed_bits::PackedBits,
	train_targets: &[i64],
	train_negatives: &[i64],
	num_train: usize,
	num_negatives: usize,
	total_input_bits: usize,
	empty_value: f32,
	_neuron_sample_rate: f32,
	_rng_seed: u64,
	class_weights: Option<&[u32]>,
) -> Result<Vec<GenomeExport>, String> {
	if num_genomes == 0 {
		return Ok(Vec::new());
	}

	// Verify uniform per-cluster neuron counts (need same neurons_per_cluster[]
	// shape across all genomes for the batched kernel to use a single dispatch).
	let first_neurons_per_cluster = &genomes_neurons_flat[0..num_clusters];
	let first_total_neurons: usize = first_neurons_per_cluster.iter().sum();
	for g in 1..num_genomes {
		let base = g * num_clusters;
		let this_npc = &genomes_neurons_flat[base..base + num_clusters];
		if this_npc != first_neurons_per_cluster {
			return Err(format!(
				"non-uniform neurons_per_cluster across batch: genome[0]={:?}, genome[{}]={:?}",
				first_neurons_per_cluster, g, this_npc
			));
		}
		let _ = first_total_neurons;
	}

	let num_neurons_per_genome = first_total_neurons;
	// total conns per genome = sum of bits across that genome's neurons
	// (must also be uniform — verify by sampling first vs rest)
	let mut conn_per_genome_max = 0usize;
	let mut genome_bpn_offsets: Vec<usize> = Vec::with_capacity(num_genomes + 1);
	genome_bpn_offsets.push(0);
	for g in 0..num_genomes {
		let bpn_start = genome_bpn_offsets[g];
		let bpn_end = bpn_start + num_neurons_per_genome;
		let conn_total: usize = genomes_bits_flat[bpn_start..bpn_end].iter().sum();
		conn_per_genome_max = conn_per_genome_max.max(conn_total);
		genome_bpn_offsets.push(bpn_end);
	}

	// For now require uniform conn_per_genome too (some flows may use
	// per-neuron variable bits; if so, fall back to per-genome dispatch)
	let conn_per_genome = {
		let bpn_start = genome_bpn_offsets[0];
		let bpn_end = bpn_start + num_neurons_per_genome;
		let total: usize = genomes_bits_flat[bpn_start..bpn_end].iter().sum();
		for g in 1..num_genomes {
			let bpn_start = genome_bpn_offsets[g];
			let bpn_end = bpn_start + num_neurons_per_genome;
			let t: usize = genomes_bits_flat[bpn_start..bpn_end].iter().sum();
			if t != total {
				return Err(format!(
					"non-uniform conn_per_genome: genome[0]={}, genome[{}]={}",
					total, g, t
				));
			}
		}
		total
	};
	let _ = conn_per_genome_max;

	// Slot capacity per neuron — sized for worst-case unique-address load
	// at ~50% factor via shared helper. Fixed-buffer Metal hashtable can't
	// grow, so undersizing causes GPU probe-loop spinning (was 62-sec
	// kernels under the old `estimate_capacity` path).
	let max_bits = genomes_bits_flat.iter().copied().max().unwrap_or(48);
	let slot_capacity_per_neuron = marker_capacity_for_train(num_train, max_bits);
	let slots_per_genome = num_neurons_per_genome * slot_capacity_per_neuron;
	let total_slots = num_genomes * slots_per_genome;

	// Memory budget check: each slot = 16 B (marker u32 + key u64 + value u32).
	// Cap the batched dispatch at 16 GB total (Mac Studio 64 GB unified, but
	// other GPU buffers + Rust heap + Python all share; 16 GB keeps headroom).
	// On overflow, return Err so the caller falls back to the per-genome path,
	// which trains one genome at a time (much smaller per-call footprint).
	const BATCH_MEMORY_BUDGET: usize = 16 * 1024 * 1024 * 1024;
	let total_buffer_bytes = total_slots.saturating_mul(16);
	if total_buffer_bytes > BATCH_MEMORY_BUDGET {
		return Err(format!(
			"batched_train_offspring would allocate {:.2} GB (cap/n={}, n={}/genome, g={}), exceeds {} GB budget — fall back to per-genome path",
			total_buffer_bytes as f64 / 1e9,
			slot_capacity_per_neuron,
			num_neurons_per_genome,
			num_genomes,
			BATCH_MEMORY_BUDGET / 1024 / 1024 / 1024
		));
	}
	let default_value: u8 = 1;  // QUAD_WEAK_FALSE (memory_mode=QUAD_WEIGHTED)
	let _ = empty_value;

	let trace = std::env::var("WNN_OPTION_B_TRACE").is_ok();
	let t_phase = std::time::Instant::now();
	if trace {
		eprintln!(
			"[OPT_B_TRACE] capacity: max_bits={} cap/n={} slots/genome={} total_slots={} buffer_size={:.2}GB",
			max_bits, slot_capacity_per_neuron,
			slots_per_genome, total_slots,
			total_buffer_bytes as f64 / 1e9
		);
	}

	let trainer = get_trainer()?;
	let device = trainer.device();

	// Allocate the genome-batch's flat marker hashtable
	let gpu_table = MarkerHashTable::new_metal(device, total_slots, default_value);
	let (markers_buf, keys_buf, values_buf) = gpu_table
		.metal_buffers()
		.ok_or("MarkerHashTable returned no Metal buffers")?;
	if trace {
		eprintln!(
			"[OPT_B_TRACE] batched_train_offspring: trainer+hashtable alloc={:.2}ms total_slots={} default_value={}",
			t_phase.elapsed().as_secs_f64() * 1000.0, total_slots, default_value
		);
	}
	let t_after_alloc = t_phase.elapsed().as_secs_f64() * 1000.0;

	// Pack train_input_bits to u64 once (shared across all genomes)
	let (packed_train_input, words_per_example) =
		crate::neuron_memory::pack_packed_to_u64(train_input_bits);
	if trace {
		eprintln!(
			"[OPT_B_TRACE]   pack_packed_to_u64={:.2}ms (words_per_example={}, packed_len={})",
			t_phase.elapsed().as_secs_f64() * 1000.0 - t_after_alloc,
			words_per_example, packed_train_input.len()
		);
	}

	// Pre-compute per-neuron cluster_idx within a genome by walking the
	// uniform `first_neurons_per_cluster[]` shape. neuron n in genome g
	// belongs to the cluster c such that prefix-sum(neurons_per_cluster[0..c])
	// <= n < prefix-sum(neurons_per_cluster[0..=c]).
	let mut neuron_cluster_within_genome: Vec<u32> = Vec::with_capacity(num_neurons_per_genome);
	for (c, &npc) in first_neurons_per_cluster.iter().enumerate() {
		for _ in 0..npc {
			neuron_cluster_within_genome.push(c as u32);
		}
	}
	debug_assert_eq!(neuron_cluster_within_genome.len(), num_neurons_per_genome);

	// Build per-(genome, neuron) NeuronTrainMeta. slot_offset is global
	// in the flat buffer (genome_idx * slots_per_genome + neuron_idx * cap).
	let mut neuron_meta: Vec<NeuronTrainMeta> = Vec::with_capacity(num_genomes * num_neurons_per_genome);
	for g in 0..num_genomes {
		let bpn_start = genome_bpn_offsets[g];
		let mut local_conn_offset: u32 = 0;
		for n in 0..num_neurons_per_genome {
			let bits = genomes_bits_flat[bpn_start + n] as u32;
			neuron_meta.push(NeuronTrainMeta {
				bits,
				conn_offset: local_conn_offset,
				slot_offset: ((g * num_neurons_per_genome + n) * slot_capacity_per_neuron) as u32,
				slot_capacity: slot_capacity_per_neuron as u32,
				cluster_idx: neuron_cluster_within_genome[n],
				_pad: 0,
			});
			local_conn_offset += bits;
		}
	}

	// Build flat connections (i32 for GPU). Genomes laid out
	// contiguously: genome 0's conns, then genome 1's, etc.
	// conn_stride = conn_per_genome.
	let provided_connections = !genomes_connections_flat.is_empty();
	let connections_i32: Vec<i32> = if provided_connections {
		// genomes_connections_flat is already laid out per-genome with
		// length num_genomes * conn_per_genome
		assert_eq!(genomes_connections_flat.len(), num_genomes * conn_per_genome,
			"connections layout mismatch");
		genomes_connections_flat.iter().map(|&c| c as i32).collect()
	} else {
		// Generate random connections per-genome (matches existing behavior)
		use rand::{Rng, SeedableRng};
		let mut all = Vec::with_capacity(num_genomes * conn_per_genome);
		for g in 0..num_genomes {
			let mut rng = rand::rngs::SmallRng::seed_from_u64((g * 12345) as u64);
			for _ in 0..conn_per_genome {
				all.push(rng.gen_range(0..total_input_bits as i64) as i32);
			}
		}
		all
	};

	// Metal buffers
	let packed_buf = device.new_buffer_with_data(
		packed_train_input.as_ptr() as *const _,
		(packed_train_input.len() * 8) as u64,
		MTLResourceOptions::StorageModeShared,
	);
	let conn_buf = device.new_buffer_with_data(
		connections_i32.as_ptr() as *const _,
		(connections_i32.len() * 4) as u64,
		MTLResourceOptions::StorageModeShared,
	);
	let targets_buf = device.new_buffer_with_data(
		train_targets.as_ptr() as *const _,
		(train_targets.len() * 8) as u64,
		MTLResourceOptions::StorageModeShared,
	);
	let negs_storage = if train_negatives.is_empty() { vec![0i64] } else { train_negatives.to_vec() };
	let negs_buf = device.new_buffer_with_data(
		negs_storage.as_ptr() as *const _,
		(negs_storage.len() * 8) as u64,
		MTLResourceOptions::StorageModeShared,
	);
	let num_classes_effective = num_clusters.max(2);
	let cw_storage: Vec<u32> = class_weights
		.map(|cw| cw.to_vec())
		.unwrap_or_else(|| vec![1; num_classes_effective]);

	let params = TrainParams {
		num_examples: num_train as u32,
		// Multi-cluster: pass actual num_negatives so the kernel walks
		// train_negatives[ex_idx][k]. Single-cluster: keep 0 (kernel skips
		// the negative loop entirely on the single_cluster path).
		num_negatives: if num_clusters == 1 { 0 } else { num_negatives as u32 },
		num_neurons: num_neurons_per_genome as u32,
		num_genomes: num_genomes as u32,
		words_per_example: words_per_example as u32,
		num_classes: num_classes_effective as u32,
		memory_mode: 2,  // QUAD_WEIGHTED
		single_cluster: if num_clusters == 1 { 1 } else { 0 },
		normal_class: 0,
		conn_stride: conn_per_genome as u32,
	};

	let t_pre_train = t_phase.elapsed().as_secs_f64() * 1000.0;
	if trace {
		eprintln!(
			"[OPT_B_TRACE]   meta+buf+conn build={:.2}ms (about to enter MarkerTrainer::train)",
			t_pre_train - t_after_alloc
		);
	}
	trainer.train(
		&packed_buf, &conn_buf, &neuron_meta, &targets_buf, &negs_buf,
		&cw_storage, params,
		&markers_buf, &keys_buf, &values_buf,
	)?;
	let t_after_train = t_phase.elapsed().as_secs_f64() * 1000.0;
	if trace {
		eprintln!(
			"[OPT_B_TRACE]   trainer.train returned (wall={:.2}ms since start)",
			t_after_train
		);
	}

	// Build per-genome GenomeExport from the flat buffer
	let mut exports: Vec<GenomeExport> = Vec::with_capacity(num_genomes);
	for g in 0..num_genomes {
		let bpn_start = genome_bpn_offsets[g];
		let neurons_slice = &genomes_neurons_flat[g * num_clusters..(g + 1) * num_clusters];
		let bpn_slice = &genomes_bits_flat[bpn_start..bpn_start + num_neurons_per_genome];

		// Per-cluster max bits → groups. V1 supports a single group (all
		// clusters share the same per-cluster max bits). Mixed-bits across
		// clusters would require splitting the sparse_export per group;
		// returning Err lets the caller fall back to the per-genome path.
		let bits_per_cluster = per_cluster_max_bits(bpn_slice, neurons_slice);
		let groups: Vec<ConfigGroup> = build_groups(&bits_per_cluster, neurons_slice);
		if groups.len() != 1 {
			return Err(format!(
				"batched_train_offspring V1 requires single group (all clusters same max bits); got {} groups for genome {}",
				groups.len(), g
			));
		}

		// Per-neuron slot offsets/capacities for THIS genome (subset of
		// the flat buffer corresponding to genome g's slots)
		let slot_offsets: Vec<u32> = (0..num_neurons_per_genome as u32)
			.map(|n| (g * slots_per_genome) as u32 + n * slot_capacity_per_neuron as u32)
			.collect();
		let slot_capacities: Vec<u32> = vec![slot_capacity_per_neuron as u32; num_neurons_per_genome];

		let (keys, values, offsets, counts) =
			gpu_table.export_per_neuron(&slot_offsets, &slot_capacities);

		let sparse_export = SparseGpuExport {
			keys,
			values,
			offsets,
			counts,
			num_neurons: num_neurons_per_genome,
		};

		// GenomeExport for single-cluster: 1 group, 1 sparse export
		let cluster_ids: Vec<usize> = (0..num_clusters).collect();
		let connections_genome = connections_i32[g * conn_per_genome..(g + 1) * conn_per_genome]
			.iter().map(|&c| c as i64).collect();
		let export = GenomeExport {
			connections: connections_genome,
			group_info: vec![(true, 0, cluster_ids)],
			dense_exports: vec![],
			sparse_exports: vec![sparse_export],
			groups,
		};
		exports.push(export);
	}

	if trace {
		eprintln!(
			"[OPT_B_TRACE]   export_per_neuron + GenomeExport build for {} genomes: {:.2}ms (wall_total={:.2}ms)",
			num_genomes,
			t_phase.elapsed().as_secs_f64() * 1000.0 - t_after_train,
			t_phase.elapsed().as_secs_f64() * 1000.0
		);
	}
	Ok(exports)
}

}  // mod batched_path

#[cfg(target_os = "macos")]
pub use batched_path::batched_train_offspring;
