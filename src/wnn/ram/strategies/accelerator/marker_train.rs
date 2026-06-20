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

const SHADER_SOURCE: &str = concat!(
	include_str!("core/shaders/common.metal"), "\n",
	include_str!("core/shaders/marker_slots.metal"), "\n",   // shared GPU cell-write primitives
	include_str!("shaders/marker_train.metal"),
);

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
	/// Probability of training each neuron per example (0.0-1.0). When
	/// < 1.0, kernel applies the same xorshift sampling as the CPU path
	/// at adaptive.rs:2867-2880 (per-(neuron_idx, ex_idx) deterministic).
	pub neuron_sample_rate: f32,
	/// RNG seed used by the sampling hash. Same value must be used on
	/// CPU and GPU paths for parity.
	pub rng_seed: u32,
	/// B10: number of threads along the example axis per (genome, neuron)
	/// cell. Default is 1 (the original 2D behavior — one thread loops
	/// over all examples). When ng×n doesn't saturate the GPU (e.g., grid
	/// search at ng=1), this value is raised so each (genome, neuron) has
	/// `num_example_chunks` threads sharing the example loop. Slot writes
	/// remain correct via the marker-FSM atomic CAS.
	pub num_example_chunks: u32,
	/// Order-independent training mode. 1 = packed (obs, net) counter per
	/// slot, replaces the clamped slot_nudge with slot_nudge_oi in MSL.
	/// Slot values must be binned to 2-bit cells via MarkerHashTable::commit_oi
	/// after the kernel completes.
	pub oi_mode: u32,
	/// 31/05/2026: example chunking for cooperative cancellation. The host
	/// loops over example chunks, calling the kernel once per chunk with the
	/// matching `example_offset`/`examples_in_dispatch`. Between chunks the
	/// host polls the cancel flag — a SIGTERM during a long training run
	/// (e.g. 46M CIC-IoT) now bails within ~1 chunk's wall time (~1s default
	/// vs the previous all-or-nothing 30+ seconds). For single-chunk dispatch
	/// (backwards-compatible) set example_offset=0 and
	/// examples_in_dispatch=num_examples.
	pub example_offset: u32,
	pub examples_in_dispatch: u32,
}

pub struct MarkerTrainer {
	device: Device,
	command_queue: metal::CommandQueue,
	pipeline: ComputePipelineState,
}

/// Snapshot of the system's GPU capabilities discovered at MarkerTrainer init.
/// Used to choose defaults for `GPU_TARGET_THREADS`, `AFFINITY_RATIO`, etc.
#[derive(Debug, Clone)]
pub struct GpuInfo {
	pub name: String,
	pub recommended_max_working_set_bytes: u64,
	pub max_threads_per_threadgroup: u32,
	/// Estimated SIMT capacity (≈ GPU cores × 32 SIMD lanes). Derived from
	/// chip-family heuristics on the device name (Metal doesn't expose
	/// core count directly).
	pub estimated_simt_lanes: u32,
}

impl GpuInfo {
	fn from_device(device: &Device) -> Self {
		let name = device.name().to_string();
		// Heuristic: Apple Silicon SIMT lanes from chip name. Each GPU core
		// has 32 SIMD lanes. Core counts: M1 (7-8), M1 Pro (14-16), M1 Max
		// (24-32), M2/M3 base (8-10), Pro (16-19), Max (30-40), Ultra (60-76),
		// M4 base (10), Pro (16-20), Max (32-40), Ultra (60-80).
		let lower = name.to_lowercase();
		let cores: u32 = if lower.contains("m4 max") { 40 }
			else if lower.contains("m4 pro") { 20 }
			else if lower.contains("m4 ultra") { 80 }
			else if lower.contains("m4") { 10 }
			else if lower.contains("m3 max") { 40 }
			else if lower.contains("m3 pro") { 18 }
			else if lower.contains("m3 ultra") { 76 }
			else if lower.contains("m3") { 10 }
			else if lower.contains("m2 max") { 38 }
			else if lower.contains("m2 pro") { 19 }
			else if lower.contains("m2 ultra") { 76 }
			else if lower.contains("m2") { 10 }
			else if lower.contains("m1 max") { 32 }
			else if lower.contains("m1 pro") { 16 }
			else if lower.contains("m1 ultra") { 64 }
			else if lower.contains("m1") { 8 }
			else { 8 };  // conservative fallback
		let max_tg = device.max_threads_per_threadgroup();
		// MTLSize -> use width (largest dim) as the upper bound for tg sizing
		let max_threads_per_threadgroup = max_tg.width as u32;
		Self {
			name,
			recommended_max_working_set_bytes: device.recommended_max_working_set_size(),
			max_threads_per_threadgroup,
			estimated_simt_lanes: cores * 32,
		}
	}
}

impl MarkerTrainer {
	pub fn new() -> Result<Self, String> {
		let device = Device::system_default().ok_or("No Metal device available")?;
		let gpu_info = GpuInfo::from_device(&device);
		eprintln!(
			"[MARKER_TRAINER] GPU init: {} | est_simt_lanes={} | max_threads/tg={} | recommended_working_set={:.1}GB",
			gpu_info.name,
			gpu_info.estimated_simt_lanes,
			gpu_info.max_threads_per_threadgroup,
			gpu_info.recommended_max_working_set_bytes as f64 / 1e9,
		);
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
		let trace = crate::adaptive::gpu_batched_train_trace();
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

		let t_after_aux_bufs = t0.elapsed().as_secs_f64() * 1000.0;

		// 31/05/2026: chunk examples into multiple Metal dispatches so the
		// host can poll the cooperative cancel flag between chunks. Default
		// EXAMPLES_PER_DISPATCH = 5M aims for roughly 5s wall-clock per
		// dispatch on typical (n, b) configurations — bound by Apple-GPU's
		// per-(neuron, example) throughput. 5s is well under the user-stated
		// cancel tolerance (1-3s acceptable, up to ~5s an "exception day")
		// while keeping the per-chunk command-buffer overhead at ~0.2%
		// (~10ms / 5000ms). Tune via WNN_MARKER_EXAMPLES_PER_DISPATCH env
		// var; set to a value ≥ num_examples to restore the original single-
		// dispatch behaviour.
		const DEFAULT_EXAMPLES_PER_DISPATCH: u32 = 5_000_000;
		let examples_per_dispatch: u32 = std::env::var("WNN_MARKER_EXAMPLES_PER_DISPATCH")
			.ok()
			.and_then(|s| s.parse::<u32>().ok())
			.filter(|&v| v > 0)
			.unwrap_or(DEFAULT_EXAMPLES_PER_DISPATCH);
		let num_examples_total = params.num_examples;
		let num_host_chunks = if examples_per_dispatch >= num_examples_total || num_examples_total == 0 {
			1
		} else {
			(num_examples_total + examples_per_dispatch - 1) / examples_per_dispatch
		};

		// Grid shape (unchanged across host chunks).
		let n = params.num_neurons as u64;
		let g = params.num_genomes as u64;
		let z_chunks = (params.num_example_chunks.max(1)) as u64;
		let max_threads = self.pipeline.max_total_threads_per_threadgroup();
		let tg_x = 32u64.min(n).max(1);
		let tg_z_cap = z_chunks.min(max_threads / tg_x.max(1));
		let tg_z = tg_z_cap.max(1);
		let tg_y_cap = (max_threads / (tg_x * tg_z)).max(1);
		let tg_y = tg_y_cap.min(g).max(1);
		let grid = MTLSize::new(n, g, z_chunks);
		let tg = MTLSize::new(tg_x, tg_y, tg_z);
		if trace {
			eprintln!(
				"[GPU_BATCHED_TRACE] host_chunks={} grid=({},{},{}) tg=({},{},{}) max_threads_per_tg={} max_total={}",
				num_host_chunks, n, g, z_chunks, tg_x, tg_y, tg_z, max_threads,
				self.pipeline.thread_execution_width()
			);
		}

		// Iterate host-side example chunks. Each iteration dispatches the
		// kernel for [chunk_start, chunk_start + chunk_count) examples. The
		// underlying marker/slot buffers persist across dispatches — each
		// chunk's writes are visible to subsequent chunks via Metal's shared
		// storage model.
		let mut chunk_start: u32 = 0;
		let mut cancelled = false;
		let mut last_wait = t0.elapsed().as_secs_f64() * 1000.0;
		while chunk_start < num_examples_total {
			// Cooperative SIGTERM cancellation. Polled BEFORE each dispatch
			// so a SIGTERM that arrives mid-chunk waits at most one
			// chunk's wall-clock (~1s default).
			if ram_core::cancel::check_cancel() {
				cancelled = true;
				break;
			}
			let chunk_count = examples_per_dispatch.min(num_examples_total - chunk_start);

			// Per-chunk params buffer (TrainParams is small; allocation
			// overhead is negligible vs the ~1s kernel work). Update only
			// the example range fields; everything else stays the same.
			let mut chunk_params = params;
			chunk_params.example_offset       = chunk_start;
			chunk_params.examples_in_dispatch = chunk_count;
			let params_buf = self.device.new_buffer_with_data(
				&chunk_params as *const _ as *const _,
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
			enc.dispatch_threads(grid, tg);
			enc.end_encoding();
			cmd.commit();
			cmd.wait_until_completed();
			last_wait = t0.elapsed().as_secs_f64() * 1000.0;
			chunk_start += chunk_count;
		}

		let elapsed_ms = last_wait;
		eprintln!(
			"[MARKER_TRAIN_BATCHED] {} genomes × {} neurons × {} examples ({}{} chunks of ≤{}) in {:.2}ms",
			params.num_genomes, params.num_neurons, num_examples_total,
			if cancelled { "cancelled after " } else { "" }, num_host_chunks, examples_per_dispatch, elapsed_ms
		);
		if trace {
			eprintln!(
				"[GPU_BATCHED_TRACE]   aux_buf={:.2}ms total_train={:.2}ms cancelled={}",
				t_after_aux_bufs, elapsed_ms - t_after_aux_bufs, cancelled
			);
		}
		if cancelled {
			Err("cancelled".to_string())
		} else {
			Ok(elapsed_ms)
		}
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

use super::metal_impl::MarkerTrainer;
use std::sync::OnceLock;

/// Sample-rate-aware load-factor sizing for MarkerHashTable's fixed-capacity
/// Metal buffers. Replaces `atomic_hashtable::estimate_capacity` (which was
/// designed for the growable AtomicHashTable).
///
/// B8: at production sample_rate=0.25, only ~25% of (neuron, example) pairs
/// produce a write — the rest are skipped by the kernel's `should_skip_sample`.
/// Sizing for `num_train * 2` was therefore over-allocating 4× at sr=0.25.
/// Measured actual LF at sr=0.25 was 6-11% with the old formula.
///
/// New formula: `effective_train = num_train * sample_rate`, then size for
/// 0.5 LF on the effective count. At sr=1.0 this is identical to the old
/// behavior; at sr=0.25 it gives a 4× memory + export speedup.
///
/// The `WNN_MARKER_OVERSIZE` env var (default 2.0) lets you tune further if
/// real data uniqueness diverges from the worst case.
pub(super) fn marker_capacity_for_train(
	num_train: usize,
	max_bits: usize,
	neuron_sample_rate: f32,
) -> usize {
	let oversize_factor = std::env::var("WNN_MARKER_OVERSIZE")
		.ok()
		.and_then(|s| s.parse::<f32>().ok())
		.unwrap_or(2.0)
		.max(1.05); // safety floor

	let effective_train = ((num_train as f32) * neuron_sample_rate.clamp(0.0, 1.0)).ceil() as usize;
	let effective_train = effective_train.max(1);

	let upper = if max_bits >= 30 {
		effective_train
	} else {
		effective_train.min(1usize << max_bits)
	};
	let raw = (((upper as f32) * oversize_factor).ceil() as usize).max(256);
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

}  // mod genome_path

#[cfg(target_os = "macos")]
#[allow(unused_imports)]  // re-exported for external callers; not used within this file

// =============================================================================
// B4b — batched_train_offspring: batched-dispatch version that takes the
// same inputs as evaluate_genomes_parallel_hybrid and returns Vec<GenomeExport>
// =============================================================================

#[cfg(target_os = "macos")]
pub mod batched_path {

use super::metal_impl::{NeuronTrainMeta, TrainParams};
use super::genome_path::{get_trainer, marker_capacity_for_train};
use crate::adaptive::{ConfigGroup, GenomeExport, SparseGpuExport, build_groups, per_cluster_max_bits, reorganize_connections_for_gpu};
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
	train_input_bits: &ram_core::packed_bits::PackedBits,
	train_targets: &[i64],
	train_negatives: &[i64],
	num_train: usize,
	num_negatives: usize,
	total_input_bits: usize,
	empty_value: f32,
	neuron_sample_rate: f32,
	rng_seed: u64,
	class_weights: Option<&[u32]>,
) -> Result<Vec<GenomeExport>, String> {
	if num_genomes == 0 {
		return Ok(Vec::new());
	}

	// Cooperative SIGTERM cancellation (added 31/05/2026): if cancel is set
	// before we even begin the big Metal training dispatch, bail immediately
	// with an Err. The caller (cpu_one_genome in adaptive::evaluate_genomes_*)
	// falls through to its own cancel check on the dense fallback path, then
	// returns a default GenomeExport so the outer batch loop short-circuits.
	if ram_core::cancel::check_cancel() {
		return Err("cancelled".to_string());
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

	// Detect heterogeneity (within or across genomes). When detected, the
	// training kernel still works correctly (each NeuronTrainMeta carries its
	// own `bits` field), but the export.connections layout must be PADDED to
	// N × max_bits to match what downstream evaluate_group_sparse_gpu expects.
	// CPU per-genome path uses reorganize_connections_for_gpu for this; we do
	// the equivalent padding inline below.
	//
	// has_heterogeneous_bpn is true if ANY genome has non-uniform bpn within
	// itself OR if genomes differ in their max bits. Both cases require the
	// padded-export path.
	let mut has_heterogeneous_bpn = false;
	let mut max_bits_in_batch: usize = 0;
	for g in 0..num_genomes {
		let bpn_start = genome_bpn_offsets[g];
		let bpn_end = bpn_start + num_neurons_per_genome;
		let slice = &genomes_bits_flat[bpn_start..bpn_end];
		if let (Some(&mn), Some(&mx)) = (slice.iter().min(), slice.iter().max()) {
			if mn != mx {
				has_heterogeneous_bpn = true;
			}
			max_bits_in_batch = max_bits_in_batch.max(mx);
		}
	}
	// Also detect heterogeneity across genomes (different max bits per genome)
	if !has_heterogeneous_bpn && num_genomes > 1 {
		let g0_max: usize = genomes_bits_flat[0..num_neurons_per_genome]
			.iter().copied().max().unwrap_or(0);
		for g in 1..num_genomes {
			let bpn_start = genome_bpn_offsets[g];
			let bpn_end = bpn_start + num_neurons_per_genome;
			let g_max: usize = genomes_bits_flat[bpn_start..bpn_end]
				.iter().copied().max().unwrap_or(0);
			if g_max != g0_max {
				has_heterogeneous_bpn = true;
				break;
			}
		}
	}

	// conn_per_genome: the total connection slots per genome that we'll lay out
	// in the connections_i32 buffer. For uniform bpn, this equals sum(bpn) per
	// genome (unpadded — fast path, no padding needed). For heterogeneous bpn,
	// we pad to N × max_bits_in_batch so each neuron has a fixed stride.
	let conn_per_genome: usize = if has_heterogeneous_bpn {
		num_neurons_per_genome * max_bits_in_batch
	} else {
		// Uniform case: verify sum(bpn) is the same across genomes (sanity)
		let bpn_start = genome_bpn_offsets[0];
		let bpn_end = bpn_start + num_neurons_per_genome;
		let total: usize = genomes_bits_flat[bpn_start..bpn_end].iter().sum();
		for g in 1..num_genomes {
			let bpn_start = genome_bpn_offsets[g];
			let bpn_end = bpn_start + num_neurons_per_genome;
			let t: usize = genomes_bits_flat[bpn_start..bpn_end].iter().sum();
			if t != total {
				return Err(format!(
					"non-uniform conn_per_genome (uniform path): genome[0]={}, genome[{}]={}",
					total, g, t
				));
			}
		}
		total
	};
	let _ = conn_per_genome_max;

	// B5g: per-cluster slot capacity based on actual cluster participation.
	//
	// For each example, exactly one cluster (the target) does a positive nudge
	// and `num_negatives` other clusters do negative nudges. With sparse
	// negative sampling (num_negatives < num_clusters - 1), some clusters see
	// fewer examples than others — minority clusters need less capacity.
	//
	// Compute the actual per-cluster example count by walking train_targets
	// + train_negatives. Single-cluster reduces to the previous B8 behavior
	// exactly (cluster 0 sees all num_train examples).
	let max_bits = genomes_bits_flat.iter().copied().max().unwrap_or(48);
	let cluster_example_count: Vec<usize> = {
		let mut counts = vec![0usize; num_clusters];
		if num_clusters == 1 {
			counts[0] = num_train;
		} else {
			for ex_idx in 0..num_train {
				let target = train_targets[ex_idx] as usize;
				if target < num_clusters {
					counts[target] += 1;
				}
				if num_negatives > 0 {
					let neg_start = ex_idx * num_negatives;
					for k in 0..num_negatives {
						let nc = train_negatives[neg_start + k] as usize;
						if nc < num_clusters && nc != target {
							counts[nc] += 1;
						}
					}
				}
			}
		}
		counts
	};

	// Per-cluster slot capacity (using the same sample-rate-aware formula
	// from B8 applied per cluster).
	let cluster_capacity: Vec<usize> = cluster_example_count.iter()
		.map(|&n| marker_capacity_for_train(n, max_bits, neuron_sample_rate))
		.collect();

	// Per-genome size = sum over clusters of (neurons_per_cluster * cluster_cap).
	// Per-cluster offset within a genome = cumulative size of prior clusters.
	let mut cluster_offset_in_genome: Vec<usize> = Vec::with_capacity(num_clusters + 1);
	cluster_offset_in_genome.push(0);
	let mut running = 0usize;
	for c in 0..num_clusters {
		running += first_neurons_per_cluster[c] * cluster_capacity[c];
		cluster_offset_in_genome.push(running);
	}
	let slots_per_genome = running;
	let total_slots = num_genomes * slots_per_genome;

	// For single-cluster (most production), slot_capacity_per_neuron is
	// still uniform — keep a scalar shadow for downstream code paths that
	// need a representative value (trace output, error messages).
	let slot_capacity_per_neuron = cluster_capacity[0];

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

	let trace = crate::adaptive::gpu_batched_train_trace();
	let t_phase = std::time::Instant::now();
	if trace {
		eprintln!(
			"[GPU_BATCHED_TRACE] capacity: max_bits={} cap/n={} slots/genome={} total_slots={} buffer_size={:.2}GB",
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
			"[GPU_BATCHED_TRACE] batched_train_offspring: trainer+hashtable alloc={:.2}ms total_slots={} default_value={}",
			t_phase.elapsed().as_secs_f64() * 1000.0, total_slots, default_value
		);
	}
	let t_after_alloc = t_phase.elapsed().as_secs_f64() * 1000.0;

	// Pack train_input_bits to u64 once (shared across all genomes)
	let (packed_train_input, words_per_example) =
		ram_core::neuron_memory::pack_packed_to_u64(train_input_bits);
	if trace {
		eprintln!(
			"[GPU_BATCHED_TRACE]   pack_packed_to_u64={:.2}ms (words_per_example={}, packed_len={})",
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

	// Build per-(genome, neuron) NeuronTrainMeta.
	// slot_offset = genome_g_base + cluster_offset_in_genome[c] + n_local * cluster_capacity[c]
	// where n_local is the neuron's index WITHIN its cluster.
	//
	// conn_offset semantics differ by path:
	//   - Uniform path (has_heterogeneous_bpn=false): contiguous, += actual bits
	//     after each neuron. Total per genome = sum(bpn) = conn_per_genome.
	//   - Heterogeneous path: per-neuron stride of max_bits_in_batch. Each
	//     neuron reads its `bits` actual connections; the kernel ignores the
	//     remaining (max_bits - bits) padding slots (filled with -1 below).
	//     Total per genome = N × max_bits_in_batch = conn_per_genome.
	let mut neuron_meta: Vec<NeuronTrainMeta> = Vec::with_capacity(num_genomes * num_neurons_per_genome);
	for g in 0..num_genomes {
		let bpn_start = genome_bpn_offsets[g];
		let genome_base = g * slots_per_genome;
		let mut local_conn_offset: u32 = 0;
		// Walk neurons within this genome by cluster
		let mut n_in_genome = 0usize;
		for c in 0..num_clusters {
			let cap_c = cluster_capacity[c];
			let n_in_cluster = first_neurons_per_cluster[c];
			let cluster_base = genome_base + cluster_offset_in_genome[c];
			for n_local in 0..n_in_cluster {
				let bits = genomes_bits_flat[bpn_start + n_in_genome] as u32;
				neuron_meta.push(NeuronTrainMeta {
					bits,
					conn_offset: local_conn_offset,
					slot_offset: (cluster_base + n_local * cap_c) as u32,
					slot_capacity: cap_c as u32,
					cluster_idx: c as u32,
					_pad: 0,
				});
				// Heterogeneous: fixed stride of max_bits per neuron.
				// Uniform: increment by actual bits (compact, sum(bpn) total).
				local_conn_offset += if has_heterogeneous_bpn { max_bits_in_batch as u32 } else { bits };
				n_in_genome += 1;
			}
		}
		debug_assert_eq!(n_in_genome, num_neurons_per_genome);
	}
	// neuron_cluster_within_genome is now redundant (cluster_idx set above);
	// drop it to silence unused warning if previously bound.
	let _ = neuron_cluster_within_genome;

	// Build flat connections (i32 for GPU). Genomes laid out contiguously.
	//
	// Uniform path: connections_i32 is sum(bpn) per genome (unpadded).
	// Heterogeneous path: connections_i32 is N × max_bits per genome (padded);
	// each neuron's actual bits are placed in the first `bits` slots, the
	// remaining (max_bits - bits) slots are -1 sentinels. The kernel reads
	// only `bits` slots per neuron via NeuronTrainMeta.bits and ignores the
	// padding. The padded layout also matches what downstream
	// evaluate_group_sparse_gpu expects (ConfigGroup.conn_size = N × max_bits).
	let provided_connections = !genomes_connections_flat.is_empty();
	let connections_i32: Vec<i32> = if provided_connections {
		// Caller passes unpadded layout (per-genome stride = sum(bpn) per
		// that genome). If we're on the heterogeneous path, we need to repack
		// into padded layout. If uniform, sum(bpn) == conn_per_genome so we
		// can pass through.
		if has_heterogeneous_bpn {
			// Repack: caller's buffer is laid out as concatenated unpadded
			// per-genome connection blocks. Length = sum_g(sum_n(bpn[g][n])).
			let mut out = vec![-1i32; num_genomes * conn_per_genome];
			let mut src_offset: usize = 0;
			for g in 0..num_genomes {
				let bpn_start = genome_bpn_offsets[g];
				for n in 0..num_neurons_per_genome {
					let bits = genomes_bits_flat[bpn_start + n];
					let dst_offset = g * conn_per_genome + n * max_bits_in_batch;
					for k in 0..bits {
						out[dst_offset + k] = genomes_connections_flat[src_offset + k] as i32;
					}
					src_offset += bits;
				}
			}
			out
		} else {
			// Uniform: caller's buffer length must equal num_genomes * conn_per_genome
			assert_eq!(genomes_connections_flat.len(), num_genomes * conn_per_genome,
				"connections layout mismatch (uniform path)");
			genomes_connections_flat.iter().map(|&c| c as i32).collect()
		}
	} else {
		// Generate random connections per-genome. For heterogeneous, fill only
		// the first `bits` slots per neuron; rest stays -1.
		use rand::{Rng, SeedableRng};
		let mut all = vec![-1i32; num_genomes * conn_per_genome];
		for g in 0..num_genomes {
			let mut rng = rand::rngs::SmallRng::seed_from_u64((g * 12345) as u64);
			if has_heterogeneous_bpn {
				let bpn_start = genome_bpn_offsets[g];
				for n in 0..num_neurons_per_genome {
					let bits = genomes_bits_flat[bpn_start + n];
					let dst_offset = g * conn_per_genome + n * max_bits_in_batch;
					for k in 0..bits {
						all[dst_offset + k] = rng.gen_range(0..total_input_bits as i64) as i32;
					}
				}
			} else {
				for k in 0..conn_per_genome {
					all[g * conn_per_genome + k] = rng.gen_range(0..total_input_bits as i64) as i32;
				}
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

	// B10: example-axis chunking. Splits the per-(genome, neuron) example
	// loop across multiple threads when ng × n is too small to saturate
	// the GPU. Atomic CAS handles slot contention.
	//
	// Heuristic tuning (15/05/2026 empirical, M4 Max 40-core GPU):
	//   - GPU_BUSY_THRESHOLD=640: don't chunk if ng×n is already at half
	//     of GPU target. Avoids contention overhead when GPU is already busy.
	//   - GPU_TARGET_THREADS=1280: target total threads for saturation.
	//   - MAX_EXAMPLE_CHUNKS=8: chunking past 8 hurts more than helps
	//     (CAS contention on shared slot region dominates).
	// Only chunk when ng×n is well below the GPU's effective parallel
	// capacity. At ng×n ≥ 320 the kernel already gets enough work per
	// thread that the contention from chunking exceeds the benefit.
	const GPU_BUSY_THRESHOLD: u64 = 320;
	const GPU_TARGET_THREADS: u64 = 1280;
	const MAX_EXAMPLE_CHUNKS: u64 = 8;
	let ng_n_product = (num_genomes as u64) * (num_neurons_per_genome as u64);
	let num_example_chunks: u32 = if ng_n_product == 0 || ng_n_product >= GPU_BUSY_THRESHOLD {
		1
	} else {
		let raw = (GPU_TARGET_THREADS + ng_n_product - 1) / ng_n_product;
		raw.next_power_of_two().min(MAX_EXAMPLE_CHUNKS) as u32
	};
	if trace {
		eprintln!(
			"[GPU_BATCHED_TRACE] B10: ng×n={} chunks={} (target_threads={})",
			ng_n_product, num_example_chunks, GPU_TARGET_THREADS
		);
	}

	// OI gating for the batched Metal path. memory_mode is always QUAD_WEIGHTED
	// here (hardcoded to 2 below), so only the env var matters.
	let use_oi = ram_core::neuron_memory::order_independent_training_enabled();
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
		neuron_sample_rate,
		// CPU uses u64 rng_seed; only low 32 bits used by the sampling hash
		// (matches xorshift32). Truncate consistently.
		rng_seed: (rng_seed as u32),
		num_example_chunks,
		oi_mode: if use_oi { 1 } else { 0 },
		// Set per-chunk inside the host loop below (in MarkerTrainer::train).
		example_offset: 0,
		examples_in_dispatch: num_train as u32,
	};

	let t_pre_train = t_phase.elapsed().as_secs_f64() * 1000.0;
	if trace {
		eprintln!(
			"[GPU_BATCHED_TRACE]   meta+buf+conn build={:.2}ms (about to enter MarkerTrainer::train)",
			t_pre_train - t_after_alloc
		);
	}
	trainer.train(
		&packed_buf, &conn_buf, &neuron_meta, &targets_buf, &negs_buf,
		&cw_storage, params,
		&markers_buf, &keys_buf, &values_buf,
	)?;
	// OI commit: bin packed counters → 2-bit cells across every slot in
	// the batched table BEFORE any per-genome export reads values.
	if use_oi {
		gpu_table.commit_oi_all();
	}
	let t_after_train = t_phase.elapsed().as_secs_f64() * 1000.0;
	if trace {
		eprintln!(
			"[GPU_BATCHED_TRACE]   trainer.train returned (wall={:.2}ms since start)",
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

		// Per-neuron slot offsets/capacities for THIS genome.
		// Layout mirrors NeuronTrainMeta construction above:
		//   slot[n] = g * slots_per_genome + cluster_offset_in_genome[c]
		//           + n_local * cluster_capacity[c]
		let genome_base = g * slots_per_genome;
		let mut slot_offsets: Vec<u32> = Vec::with_capacity(num_neurons_per_genome);
		let mut slot_capacities: Vec<u32> = Vec::with_capacity(num_neurons_per_genome);
		for c in 0..num_clusters {
			let cap_c = cluster_capacity[c] as u32;
			let cluster_base = (genome_base + cluster_offset_in_genome[c]) as u32;
			for n_local in 0..first_neurons_per_cluster[c] {
				slot_offsets.push(cluster_base + (n_local as u32) * cap_c);
				slot_capacities.push(cap_c);
			}
		}

		let (keys, values, offsets, counts) =
			gpu_table.export_per_neuron(&slot_offsets, &slot_capacities);

		let sparse_export = SparseGpuExport {
			keys,
			values,
			offsets,
			counts,
			num_neurons: num_neurons_per_genome,
		};

		// GenomeExport for single-cluster: 1 group, 1 sparse export.
		//
		// The downstream evaluate_group_sparse_gpu expects export.connections in
		// the "PREFIX-pad with -1, real connections at END" layout produced by
		// reorganize_connections_for_gpu (see adaptive.rs:914). Our internal
		// connections_i32 layout differs:
		//   - Uniform path: sum(bpn) per genome, no padding
		//   - Heterogeneous path: N × max_bits per genome, with PREFIX zeros
		//     followed by real conns? No — actually we wrote real conns at the
		//     FRONT (0..bits) and padding at the END. The GPU shader's address
		//     bit i = (max_bits-1-i) means it reads from the END first.
		// So in BOTH cases we need to produce the same end-padded layout as
		// reorganize_connections_for_gpu (real conns at slots [pad..max_bits]).
		// Easiest: always call reorganize_connections_for_gpu here on the
		// unpadded source.
		let cluster_ids: Vec<usize> = (0..num_clusters).collect();
		// Rebuild unpadded source for THIS genome (one slice from caller's input
		// or regenerate from our connections_i32 by trimming padding).
		let connections_genome: Vec<i64> = if has_heterogeneous_bpn {
			// Build an unpadded i64 slice from our padded i32 layout, then
			// hand to reorganize_connections_for_gpu to apply PREFIX-padding.
			let mut unpadded: Vec<i64> = Vec::with_capacity(
				bpn_slice.iter().sum::<usize>()
			);
			for n in 0..num_neurons_per_genome {
				let bits = bpn_slice[n];
				let src_offset = g * conn_per_genome + n * max_bits_in_batch;
				for k in 0..bits {
					unpadded.push(connections_i32[src_offset + k] as i64);
				}
			}
			reorganize_connections_for_gpu(&unpadded, bpn_slice, neurons_slice, &groups)
		} else {
			// Uniform path: connections_i32 already at sum(bpn) per genome,
			// but downstream still expects PREFIX-padded layout. For uniform
			// bpn this is a no-op padding (n_bits == max_bits → pad_size == 0),
			// but call through to keep behavior consistent.
			let unpadded: Vec<i64> = connections_i32[g * conn_per_genome..(g + 1) * conn_per_genome]
				.iter().map(|&c| c as i64).collect();
			reorganize_connections_for_gpu(&unpadded, bpn_slice, neurons_slice, &groups)
		};
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
			"[GPU_BATCHED_TRACE]   export_per_neuron + GenomeExport build for {} genomes: {:.2}ms (wall_total={:.2}ms)",
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
