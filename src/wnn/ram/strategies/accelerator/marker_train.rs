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
	/// 05/07/2026: neuron-axis chunking for over-budget single genomes. The
	/// sampling hash keys on the neuron's index within the dispatch; when a
	/// genome is trained in neuron chunks, each chunk passes its start index
	/// here so `should_skip_sample(neuron_idx + offset, ...)` sees the same
	/// GLOBAL neuron index as an unchunked dispatch — keeping chunked output
	/// bit-exact vs unchunked (and vs the CPU path). 0 for unchunked calls.
	pub neuron_index_offset: u32,
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
pub(crate) fn marker_capacity_for_train(
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
///
/// 05/07/2026: when a SINGLE single-cluster genome alone exceeds the batch
/// memory budget (46M-row datasets: 250n × 2^24 slots × 16 B = 67 GB), the
/// call is transparently routed through `train_single_genome_neuron_chunked`
/// — the neuron axis is split into budget-sized dispatches and the per-neuron
/// exports concatenated. Output is bit-exact vs an unchunked dispatch (the
/// sampling hash receives the global neuron index via
/// `TrainParams::neuron_index_offset`). Previously this case fell back to the
/// CPU DashMap path, whose OI counters held a ~61 GB heap per genome.
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
	memory_mode: u8,
) -> Result<Vec<GenomeExport>, String> {
	let result = batched_train_core(
		genomes_bits_flat, genomes_neurons_flat, genomes_connections_flat,
		num_genomes, num_clusters, train_input_bits, train_targets,
		train_negatives, num_train, num_negatives, total_input_bits,
		empty_value, neuron_sample_rate, rng_seed, class_weights, 0, memory_mode,
	);
	match result {
		Err(ref msg) if msg.starts_with(BUDGET_ERR_PREFIX)
			&& num_genomes == 1
			&& num_clusters == 1
			&& !genomes_connections_flat.is_empty() =>
		{
			train_single_genome_neuron_chunked(
				genomes_bits_flat, genomes_connections_flat, train_input_bits,
				train_targets, train_negatives, num_train, num_negatives,
				total_input_bits, empty_value, neuron_sample_rate, rng_seed,
				class_weights, CHUNK_MEMORY_BUDGET, memory_mode,
			)
		}
		other => other,
	}
}

/// Stable prefix of the budget-overflow error — the chunked-path routing in
/// `batched_train_offspring` matches on it. Keep format! below in sync.
const BUDGET_ERR_PREFIX: &str = "batched_train_offspring would allocate";

/// Batched-dispatch Metal buffer cap (16 GB total slots; Mac Studio 64 GB
/// unified, but other GPU buffers + Rust heap + Python all share; 16 GB keeps
/// headroom). On overflow the batched call returns Err so the caller falls
/// back to the per-genome path (and, for a single single-cluster genome, the
/// neuron-chunked path).
const BATCH_MEMORY_BUDGET: usize = 16 * 1024 * 1024 * 1024;

/// Per-dispatch Metal buffer target for the neuron-chunked path. Half the
/// 16 GB batch budget: the chunk buffer is transient (alloc → train → export
/// → free per chunk) but the worker shares 64 GB with the dataset memmap,
/// Python heap and any controller run — 8 GB keeps the peak civil while the
/// dispatch overhead stays negligible (per-chunk work is neurons × examples,
/// so total kernel work is identical at any chunk size).
const CHUNK_MEMORY_BUDGET: usize = 8 * 1024 * 1024 * 1024;

/// Train ONE single-cluster genome whose marker table exceeds the batch
/// budget by splitting its neurons into budget-sized chunks. Each chunk is a
/// normal `batched_train_core` dispatch (1 genome, chunk_n neurons) with
/// `neuron_index_offset` = the chunk's global start, so sampling matches an
/// unchunked dispatch bit-for-bit. Per-neuron sparse exports concatenate in
/// neuron order; groups/connections are rebuilt from the FULL genome so the
/// merged `GenomeExport` is indistinguishable from an unchunked one.
#[allow(clippy::too_many_arguments)]
fn train_single_genome_neuron_chunked(
	genomes_bits_flat: &[usize],
	genomes_connections_flat: &[i64],
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
	chunk_budget_bytes: usize,
	memory_mode: u8,
) -> Result<Vec<GenomeExport>, String> {
	let n_total = genomes_bits_flat.len();
	let max_bits = genomes_bits_flat.iter().copied().max().unwrap_or(48);
	let cap_per_neuron = marker_capacity_for_train(num_train, max_bits, neuron_sample_rate);
	let bytes_per_neuron = cap_per_neuron.saturating_mul(16);
	let chunk_n = (chunk_budget_bytes / bytes_per_neuron.max(1)).max(1).min(n_total);
	let num_chunks = (n_total + chunk_n - 1) / chunk_n;
	eprintln!(
		"[MARKER_CHUNK] single genome over batch budget ({} n × {:.0} MB/n = {:.2} GB) — {} neuron-chunked dispatches of ≤{} neurons ({:.2} GB each)",
		n_total, bytes_per_neuron as f64 / 1e6,
		(n_total * bytes_per_neuron) as f64 / 1e9,
		num_chunks, chunk_n, (chunk_n * bytes_per_neuron) as f64 / 1e9,
	);

	// Per-neuron connection prefix sums for slicing the unpadded flat buffer.
	let mut conn_prefix: Vec<usize> = Vec::with_capacity(n_total + 1);
	conn_prefix.push(0);
	for &b in genomes_bits_flat {
		conn_prefix.push(conn_prefix.last().unwrap() + b);
	}
	if *conn_prefix.last().unwrap() != genomes_connections_flat.len() {
		return Err(format!(
			"neuron-chunked path: connections length {} != sum(bits) {}",
			genomes_connections_flat.len(), conn_prefix.last().unwrap()
		));
	}

	// Train each chunk and concatenate the per-neuron sparse exports.
	let mut keys: Vec<u64> = Vec::new();
	let mut values: Vec<u8> = Vec::new();
	let mut offsets: Vec<u32> = Vec::with_capacity(n_total);
	let mut counts: Vec<u32> = Vec::with_capacity(n_total);
	let mut start = 0usize;
	while start < n_total {
		let end = (start + chunk_n).min(n_total);
		let chunk_bits = &genomes_bits_flat[start..end];
		let chunk_conns = &genomes_connections_flat[conn_prefix[start]..conn_prefix[end]];
		let chunk_neurons = [end - start];
		let mut chunk_exports = batched_train_core(
			chunk_bits, &chunk_neurons, chunk_conns, 1, 1, train_input_bits,
			train_targets, train_negatives, num_train, num_negatives,
			total_input_bits, empty_value, neuron_sample_rate, rng_seed,
			class_weights, start as u32, memory_mode,
		)?;
		let export = chunk_exports.pop()
			.ok_or_else(|| "neuron-chunked path: core returned empty Vec".to_string())?;
		let sparse = export.sparse_exports.into_iter().next()
			.ok_or_else(|| "neuron-chunked path: chunk export has no sparse_exports".to_string())?;
		let base = keys.len() as u32;
		for n in 0..sparse.num_neurons {
			offsets.push(base + sparse.offsets[n]);
			counts.push(sparse.counts[n]);
		}
		keys.extend_from_slice(&sparse.keys);
		values.extend_from_slice(&sparse.values);
		start = end;
	}

	// Rebuild groups/connections from the FULL genome (identical to what an
	// unchunked dispatch would have produced for g=0).
	let neurons_full = [n_total];
	let bits_per_cluster = per_cluster_max_bits(genomes_bits_flat, &neurons_full);
	let groups: Vec<ConfigGroup> = build_groups(&bits_per_cluster, &neurons_full);
	if groups.len() != 1 {
		return Err(format!(
			"neuron-chunked path expects a single group for a single-cluster genome; got {}",
			groups.len()
		));
	}
	let connections = reorganize_connections_for_gpu(
		genomes_connections_flat, genomes_bits_flat, &neurons_full, &groups,
	);
	let sparse_export = SparseGpuExport {
		keys,
		values,
		offsets,
		counts,
		num_neurons: n_total,
	};
	Ok(vec![GenomeExport {
		connections,
		group_info: vec![(true, 0, vec![0])],
		dense_exports: vec![],
		sparse_exports: vec![sparse_export],
		groups,
	}])
}

/// Eval-in-place eligibility: true when ONE single-cluster genome alone
/// exceeds the batched-dispatch budget — i.e. `batched_train_offspring` would
/// route it through `train_single_genome_neuron_chunked`. Mirrors the sizing
/// math there (marker_capacity_for_train × 16 B/slot vs BATCH_MEMORY_BUDGET)
/// so the fused path fires for EXACTLY the chunked regime and nothing else.
pub fn single_genome_exceeds_batch_budget(
	bits_flat: &[usize],
	num_train: usize,
	neuron_sample_rate: f32,
) -> bool {
	let max_bits = bits_flat.iter().copied().max().unwrap_or(48);
	let cap_per_neuron = marker_capacity_for_train(num_train, max_bits, neuron_sample_rate);
	bits_flat.len().saturating_mul(cap_per_neuron).saturating_mul(16) > BATCH_MEMORY_BUDGET
}

/// Per-example integer vote sums from the eval-in-place probe. One u32 per
/// (example, cluster) — single-cluster in the current fused path, so one per
/// example. votes/4.0 = the sum of QUAD cell weights the sorted-export eval
/// would have produced (F=0, wF=1, wT=3, T=4 quarters; a table MISS
/// contributes the QUAD default cell WEAK_FALSE = 1, identical to how the
/// export-path eval miss-defaults — which is also why the wF-filter never
/// changes scores).
pub struct ChunkedVoteSums {
	pub eval_votes: Vec<u32>,
	/// Present when the caller also needs train-set scores (threshold
	/// calibration on training data — the fitness path's default).
	pub train_votes: Option<Vec<u32>>,
}

/// Eval-in-place for the over-budget chunked regime: train ONE single-cluster
/// genome in neuron chunks (identical dispatches to
/// `train_single_genome_neuron_chunked` — same capacity, params and
/// neuron_index_offset), but after each chunk trains, probe the STILL-RESIDENT
/// GPU hash table with the eval (and optionally train) examples, accumulating
/// per-example integer vote sums. The sorted sparse export is never built.
///
/// OI mode only: the probe merges duplicate-key slots via an inline oi_merge
/// (duplicates exist by design — concurrent find-or-claim races), which is
/// exact because OI counters are commutative. Legacy mode keeps the export
/// path.
#[allow(clippy::too_many_arguments)]
pub fn train_single_genome_chunked_scored(
	genomes_bits_flat: &[usize],
	genomes_connections_flat: &[i64],
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
	eval_input_bits: &ram_core::packed_bits::PackedBits,
	num_eval: usize,
	score_train: bool,
	memory_mode: u8,
) -> Result<ChunkedVoteSums, String> {
	train_single_genome_chunked_scored_with_budget(
		genomes_bits_flat, genomes_connections_flat, train_input_bits,
		train_targets, train_negatives, num_train, num_negatives,
		total_input_bits, empty_value, neuron_sample_rate, rng_seed,
		class_weights, eval_input_bits, num_eval, score_train,
		CHUNK_MEMORY_BUDGET, memory_mode,
	)
}

/// Budget-parameterized body of `train_single_genome_chunked_scored`
/// (tests shrink the budget to force multiple chunks on small data).
#[allow(clippy::too_many_arguments)]
fn train_single_genome_chunked_scored_with_budget(
	genomes_bits_flat: &[usize],
	genomes_connections_flat: &[i64],
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
	eval_input_bits: &ram_core::packed_bits::PackedBits,
	num_eval: usize,
	score_train: bool,
	chunk_budget_bytes: usize,
	memory_mode: u8,
) -> Result<ChunkedVoteSums, String> {
	if !ram_core::neuron_memory::order_independent_training_enabled() {
		return Err("eval-in-place requires WNN_ORDER_INDEPENDENT_TRAIN=1 (probe merges OI counters)".to_string());
	}
	let n_total = genomes_bits_flat.len();
	let max_bits = genomes_bits_flat.iter().copied().max().unwrap_or(48);
	let cap_per_neuron = marker_capacity_for_train(num_train, max_bits, neuron_sample_rate);
	let bytes_per_neuron = cap_per_neuron.saturating_mul(16);
	let chunk_n = (chunk_budget_bytes / bytes_per_neuron.max(1)).max(1).min(n_total);
	let num_chunks = (n_total + chunk_n - 1) / chunk_n;
	eprintln!(
		"[MARKER_CHUNK_SCORED] eval-in-place: {} n × {:.0} MB/n — {} chunked dispatches of ≤{} neurons, probing {} eval{} examples per chunk (sorted export SKIPPED)",
		n_total, bytes_per_neuron as f64 / 1e6, num_chunks, chunk_n, num_eval,
		if score_train { format!(" + {} train", num_train) } else { String::new() },
	);

	// Per-neuron connection prefix sums (same slicing as the export chunked path).
	let mut conn_prefix: Vec<usize> = Vec::with_capacity(n_total + 1);
	conn_prefix.push(0);
	for &b in genomes_bits_flat {
		conn_prefix.push(conn_prefix.last().unwrap() + b);
	}
	if *conn_prefix.last().unwrap() != genomes_connections_flat.len() {
		return Err(format!(
			"eval-in-place path: connections length {} != sum(bits) {}",
			genomes_connections_flat.len(), conn_prefix.last().unwrap()
		));
	}

	let trainer = get_trainer()?;
	let device = trainer.device();
	let prober = crate::marker_probe::get_prober(device)?;

	// Probe-set buffers persist across chunks: votes accumulate atomically.
	let (packed_eval, eval_words) = ram_core::neuron_memory::pack_packed_to_u64(eval_input_bits);
	let eval_buf = device.new_buffer_with_data(
		packed_eval.as_ptr() as *const _,
		(packed_eval.len().max(1) * 8) as u64,
		MTLResourceOptions::StorageModeShared,
	);
	drop(packed_eval);
	let eval_votes_buf = crate::marker_probe::new_zeroed_vote_buffer(device, num_eval);
	let train_probe = if score_train {
		let (packed_train, train_words) = ram_core::neuron_memory::pack_packed_to_u64(train_input_bits);
		let train_buf = device.new_buffer_with_data(
			packed_train.as_ptr() as *const _,
			(packed_train.len().max(1) * 8) as u64,
			MTLResourceOptions::StorageModeShared,
		);
		let train_votes_buf = crate::marker_probe::new_zeroed_vote_buffer(device, num_train);
		Some((train_buf, train_votes_buf, train_words))
	} else {
		None
	};

	// Chunk loop: train (table resident) → probe → free. Eval probes ALL
	// (example, neuron) pairs — sampling is train-only by design.
	let trace = crate::adaptive::gpu_batched_train_trace();
	let (mut t_train_ms, mut t_eval_ms, mut t_trainprobe_ms) = (0.0f64, 0.0f64, 0.0f64);
	let mut start = 0usize;
	while start < n_total {
		let end = (start + chunk_n).min(n_total);
		let chunk_bits = &genomes_bits_flat[start..end];
		let chunk_conns = &genomes_connections_flat[conn_prefix[start]..conn_prefix[end]];
		let chunk_neurons = [end - start];
		let t0 = std::time::Instant::now();
		let tb = train_batch_to_table(
			chunk_bits, &chunk_neurons, chunk_conns, 1, 1, train_input_bits,
			train_targets, train_negatives, num_train, num_negatives,
			total_input_bits, empty_value, neuron_sample_rate, rng_seed,
			class_weights, start as u32, memory_mode,
		)?;
		t_train_ms += t0.elapsed().as_secs_f64() * 1000.0;
		let (markers_buf, keys_buf, values_buf) = tb.gpu_table
			.metal_buffers()
			.ok_or("eval-in-place: MarkerHashTable returned no Metal buffers")?;
		let t1 = std::time::Instant::now();
		prober.probe_accumulate(
			&eval_buf, &tb.conn_buf, &tb.neuron_meta,
			&markers_buf, &keys_buf, &values_buf, &eval_votes_buf,
			num_eval, eval_words, 1,
		)?;
		t_eval_ms += t1.elapsed().as_secs_f64() * 1000.0;
		if let Some((train_buf, train_votes_buf, train_words)) = &train_probe {
			let t2 = std::time::Instant::now();
			prober.probe_accumulate(
				train_buf, &tb.conn_buf, &tb.neuron_meta,
				&markers_buf, &keys_buf, &values_buf, train_votes_buf,
				num_train, *train_words, 1,
			)?;
			t_trainprobe_ms += t2.elapsed().as_secs_f64() * 1000.0;
		}
		// tb drops here — chunk's 8 GB table freed before the next alloc.
		start = end;
	}
	if trace {
		eprintln!(
			"[EVAL_IN_PLACE_PHASES] train={:.0}ms eval_probe={:.0}ms({} ex) train_probe={:.0}ms({} ex)",
			t_train_ms, t_eval_ms, num_eval, t_trainprobe_ms,
			if score_train { num_train } else { 0 },
		);
	}

	Ok(ChunkedVoteSums {
		eval_votes: crate::marker_probe::read_vote_buffer(&eval_votes_buf, num_eval),
		train_votes: train_probe
			.map(|(_, votes_buf, _)| crate::marker_probe::read_vote_buffer(&votes_buf, num_train)),
	})
}

/// A trained-but-not-yet-exported batch: the resident GPU MarkerHashTable plus
/// every piece of layout metadata the export loop (or the eval-in-place probe)
/// needs. Produced by `train_batch_to_table`; consumed either by
/// `export_trained_batch` (legacy sorted-export path) or by the probe-eval
/// dispatch in `train_single_genome_chunked_scored` (fused fitness path, which
/// skips the export entirely).
struct TrainedBatch {
	gpu_table: MarkerHashTable,
	use_oi: bool,
	/// Per-(genome, neuron) training metadata — the probe kernel reuses it
	/// verbatim (bits / conn_offset / slot_offset / slot_capacity / cluster_idx),
	/// so probe addressing + table regions are identical to training by
	/// construction.
	neuron_meta: Vec<NeuronTrainMeta>,
	/// The exact connections buffer the training kernel read (probe reuse).
	conn_buf: metal::Buffer,
	connections_i32: Vec<i32>,
	num_genomes: usize,
	num_clusters: usize,
	num_neurons_per_genome: usize,
	cluster_capacity: Vec<usize>,
	cluster_offset_in_genome: Vec<usize>,
	slots_per_genome: usize,
	genome_bpn_offsets: Vec<usize>,
	has_heterogeneous_bpn: bool,
	max_bits_in_batch: usize,
	conn_per_genome: usize,
}

/// The original batched dispatch body. `neuron_index_offset` is threaded to
/// the kernel's sampling hash (0 = unchunked; chunk start otherwise).
/// Split 07/07/2026 into train_batch_to_table + export_trained_batch so the
/// eval-in-place path can probe the resident table without materializing the
/// sorted export.
#[allow(clippy::too_many_arguments)]
fn batched_train_core(
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
	neuron_index_offset: u32,
	memory_mode: u8,
) -> Result<Vec<GenomeExport>, String> {
	if num_genomes == 0 {
		return Ok(Vec::new());
	}
	let trace = crate::adaptive::gpu_batched_train_trace();
	let t_phase = std::time::Instant::now();
	let tb = train_batch_to_table(
		genomes_bits_flat, genomes_neurons_flat, genomes_connections_flat,
		num_genomes, num_clusters, train_input_bits, train_targets,
		train_negatives, num_train, num_negatives, total_input_bits,
		empty_value, neuron_sample_rate, rng_seed, class_weights,
		neuron_index_offset, memory_mode,
	)?;
	let t_after_train = t_phase.elapsed().as_secs_f64() * 1000.0;
	let exports = export_trained_batch(&tb, genomes_bits_flat, genomes_neurons_flat)?;
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

/// Steps 1-12 of the former batched_train_core: shape checks, capacity sizing,
/// buffer builds and the training kernel dispatch. Returns the resident table
/// + layout; the caller decides between sorted export and probe-eval.
#[allow(clippy::too_many_arguments)]
fn train_batch_to_table(
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
	neuron_index_offset: u32,
	memory_mode: u8,
) -> Result<TrainedBatch, String> {
	if num_genomes == 0 {
		return Err("train_batch_to_table requires at least one genome".to_string());
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
		// BINARY (classical WiSARD) writes ONLY the TRUE direction — FALSE-
		// direction participants are skipped pre-claim (marker_train.metal:241).
		// Sizing capacity on all num_train (legacy) over-allocated the marker
		// table ~1/positive_rate (single) / ~1+num_negatives (multi) — the
		// BINARY GPU-buffer bloat. Safe: distinct written addrs ≤ positives.
		let is_binary = memory_mode == ram_core::neuron_memory::MODE_BINARY;
		if num_clusters == 1 {
			counts[0] = if is_binary {
				train_targets[..num_train].iter().filter(|&&t| t == 1).count()
			} else {
				num_train
			};
		} else {
			for ex_idx in 0..num_train {
				let target = train_targets[ex_idx] as usize;
				if target < num_clusters {
					counts[target] += 1;
				}
				if !is_binary && num_negatives > 0 {
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
	// (BATCH_MEMORY_BUDGET hoisted to module scope 07/07/2026 so the
	// eval-in-place eligibility check can reuse the identical threshold.)
	let total_buffer_bytes = total_slots.saturating_mul(16);
	if total_buffer_bytes > BATCH_MEMORY_BUDGET {
		return Err(format!(
			"{} {:.2} GB (cap/n={}, n={}/genome, g={}), exceeds {} GB budget — fall back to per-genome path",
			BUDGET_ERR_PREFIX,
			total_buffer_bytes as f64 / 1e9,
			slot_capacity_per_neuron,
			num_neurons_per_genome,
			num_genomes,
			BATCH_MEMORY_BUDGET / 1024 / 1024 / 1024
		));
	}
	// OI slots hold packed counters — they MUST start at OI_INITIAL (0):
	// the legacy cell default (1 = QUAD_WEAK_FALSE) reads as net=+1, adding
	// a phantom +1 to every trained cell's tally (skews weak-cell binning
	// one bin toward TRUE, and compounds per duplicate slot under merge).
	// Found 07/07/2026 by the oi_z_parity net-sum audit. Legacy mode keeps
	// the cell default.
	let default_value: u8 = match memory_mode {
		// TERNARY: claimed-but-unwritten shouldn't occur (every claim writes);
		// init to EMPTY so any such slot exports harmlessly.
		ram_core::neuron_memory::MODE_TERNARY => 2,
		// BINARY: FALSE — but note FALSE-direction work is skipped pre-claim.
		ram_core::neuron_memory::MODE_BINARY => 0,
		_ if ram_core::neuron_memory::order_independent_training_enabled() => 0,  // OI_INITIAL (QUAD only — earlier arms catch T/B)
		_ => 1,            // QUAD_WEAK_FALSE (memory_mode=QUAD_WEIGHTED)
	};
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
	// OI gating for the batched Metal path (hoisted above the chunk heuristic
	// — the z-axis policy depends on it). OI only exists for QUAD_WEIGHTED;
	// TERNARY/BINARY use lattice writes that are order-independent natively
	// (12/07/2026 GPU-train generalization).
	let use_oi = ram_core::neuron_memory::order_independent_training_enabled()
		&& memory_mode == ram_core::neuron_memory::MODE_QUAD_WEIGHTED;
	// WNN_EXAMPLE_CHUNKS overrides the z heuristic (tuning/bench escape hatch).
	let env_chunks = std::env::var("WNN_EXAMPLE_CHUNKS").ok()
		.and_then(|s| s.parse::<u32>().ok())
		.filter(|&v| v > 0);
	// Occupancy fix 07/07/2026: the old MAX_EXAMPLE_CHUNKS=8 cap left the
	// neuron-chunked single-genome path at 256 threads — latency-bound on
	// big hash tables (34s/dispatch; z=1024 measured 30× faster). High z is
	// exact ONLY in OI mode: duplicate-key claims from concurrent inserts
	// are merged at export (oi_merge — commutative counters). Legacy mode
	// keeps the old tuned heuristic (order-dependent anyway).
	const OI_TARGET_THREADS: u64 = 32768;
	const OI_MAX_EXAMPLE_CHUNKS: u64 = 1024;
	let num_example_chunks: u32 = if let Some(z) = env_chunks {
		z
	} else if use_oi {
		if ng_n_product == 0 || ng_n_product >= OI_TARGET_THREADS {
			1
		} else {
			let raw = (OI_TARGET_THREADS + ng_n_product - 1) / ng_n_product;
			// Keep ≥256 examples per thread so tiny datasets don't shred
			// into degenerate slices.
			let work_cap = ((num_train as u64) / 256).max(1).next_power_of_two();
			raw.next_power_of_two().min(OI_MAX_EXAMPLE_CHUNKS).min(work_cap) as u32
		}
	} else if ng_n_product == 0 || ng_n_product >= GPU_BUSY_THRESHOLD {
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
		memory_mode: memory_mode as u32,
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
		neuron_index_offset,
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
	// OI: no commit pass — slots keep RAW packed counters; the export merges
	// duplicate keys (high-z concurrent claims) and bins to cells in one go
	// (and the probe-eval path merges them inline on GPU).
	if trace {
		eprintln!(
			"[GPU_BATCHED_TRACE]   trainer.train returned (wall={:.2}ms since start)",
			t_phase.elapsed().as_secs_f64() * 1000.0
		);
	}

	Ok(TrainedBatch {
		gpu_table,
		use_oi,
		neuron_meta,
		conn_buf,
		connections_i32,
		num_genomes,
		num_clusters,
		num_neurons_per_genome,
		cluster_capacity,
		cluster_offset_in_genome,
		slots_per_genome,
		genome_bpn_offsets,
		has_heterogeneous_bpn,
		max_bits_in_batch,
		conn_per_genome,
	})
}

/// Step 13 of the former batched_train_core: walk the resident table and build
/// one `GenomeExport` per genome (sorted sparse export + reorganized
/// connections). Byte-identical output to the pre-split code.
fn export_trained_batch(
	tb: &TrainedBatch,
	genomes_bits_flat: &[usize],
	genomes_neurons_flat: &[usize],
) -> Result<Vec<GenomeExport>, String> {
	let TrainedBatch {
		gpu_table,
		use_oi,
		connections_i32,
		num_genomes,
		num_clusters,
		num_neurons_per_genome,
		cluster_capacity,
		cluster_offset_in_genome,
		slots_per_genome,
		genome_bpn_offsets,
		has_heterogeneous_bpn,
		max_bits_in_batch,
		conn_per_genome,
		..
	} = tb;
	let (num_genomes, num_clusters, num_neurons_per_genome) =
		(*num_genomes, *num_clusters, *num_neurons_per_genome);
	let (slots_per_genome, has_heterogeneous_bpn, max_bits_in_batch, conn_per_genome, use_oi) =
		(*slots_per_genome, *has_heterogeneous_bpn, *max_bits_in_batch, *conn_per_genome, *use_oi);

	// Build per-genome GenomeExport from the flat buffer
	let mut exports: Vec<GenomeExport> = Vec::with_capacity(num_genomes);
	for g in 0..num_genomes {
		let bpn_start = genome_bpn_offsets[g];
		let neurons_slice = &genomes_neurons_flat[g * num_clusters..(g + 1) * num_clusters];
		let bpn_slice = &genomes_bits_flat[bpn_start..bpn_start + num_neurons_per_genome];

		// Per-cluster max bits → groups. Multi-group supported since
		// 11/07/2026 (was V1 single-group with an Err fallback): the trained
		// slot layout is PER-CLUSTER (cluster_offset_in_genome /
		// cluster_capacity — group structure never touched training), so
		// groups only matter here, at export: each group emits its clusters'
		// neuron regions as its own SparseGpuExport in group-local order.
		// This is what K-cluster multiclass genomes produce (per-cluster
		// neurons/bits evolve independently → several (neurons, bits)
		// buckets); single-group genomes walk the exact same code and yield
		// the same export as before.
		let bits_per_cluster = per_cluster_max_bits(bpn_slice, neurons_slice);
		let groups: Vec<ConfigGroup> = build_groups(&bits_per_cluster, neurons_slice);

		// Per-neuron slot offsets/capacities, group by group.
		// Layout mirrors NeuronTrainMeta construction above:
		//   slot[n] = g * slots_per_genome + cluster_offset_in_genome[c]
		//           + n_local * cluster_capacity[c]
		// Within a group, each cluster contributes group.neurons slots (the
		// padded layout eval expects); coalesced groups pad clusters with
		// fewer actual neurons using capacity-0 regions, which
		// export_per_neuron turns into count-0 entries.
		let genome_base = g * slots_per_genome;
		let mut group_info: Vec<(bool, usize, Vec<usize>)> = Vec::with_capacity(groups.len());
		let mut sparse_exports: Vec<SparseGpuExport> = Vec::with_capacity(groups.len());
		for (group_idx, group) in groups.iter().enumerate() {
			let group_neuron_count = group.total_neurons();
			let mut slot_offsets: Vec<u32> = Vec::with_capacity(group_neuron_count);
			let mut slot_capacities: Vec<u32> = Vec::with_capacity(group_neuron_count);
			for (local_idx, &c) in group.cluster_ids.iter().enumerate() {
				let actual_n = group.actual_neurons.as_ref()
					.map_or(group.neurons, |a| a[local_idx] as usize);
				let cap_c = cluster_capacity[c] as u32;
				let cluster_base = (genome_base + cluster_offset_in_genome[c]) as u32;
				for n_local in 0..group.neurons {
					if n_local < actual_n {
						slot_offsets.push(cluster_base + (n_local as u32) * cap_c);
						slot_capacities.push(cap_c);
					} else {
						slot_offsets.push(0);
						slot_capacities.push(0);
					}
				}
			}

			let (keys, values, offsets, counts) =
				gpu_table.export_per_neuron(&slot_offsets, &slot_capacities, use_oi);

			sparse_exports.push(SparseGpuExport {
				keys,
				values,
				offsets,
				counts,
				num_neurons: group_neuron_count,
			});
			group_info.push((true, group_idx, group.cluster_ids.clone()));
		}

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
		// unpadded source (it is group-aware: output is group-major at each
		// group's conn_offset).
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
			group_info,
			dense_exports: vec![],
			sparse_exports,
			groups,
		};
		exports.push(export);
	}
	Ok(exports)
}

#[cfg(test)]
mod chunked_tests {
	use super::*;

	/// Multi-group export (11/07/2026): a K-cluster genome whose clusters
	/// differ in (neurons, bits) must produce a per-group export that scores
	/// IDENTICALLY to the per-genome CPU reference path
	/// (train_genome_in_slot + export_genome_for_gpu). Data is built so
	/// every (neuron, address) only ever receives AGREEING votes (class bits
	/// disjoint per class), making both training paths exactly deterministic
	/// in legacy AND OI modes — the score comparison is exact, not
	/// tolerance-based. Also asserts argmax separability, which would fail
	/// loudly if the group-local neuron→cluster mapping were scrambled.
	/// Mode-parameterized body — the GPU batched trainer must match the CPU
	/// per-genome reference for QUAD (nudge/OI), TERNARY (TRUE-wins lattice)
	/// and BINARY (classical one-shot own-class) — 12/07/2026 GPU-train
	/// generalization (Luiz order).
	fn run_multigroup_parity(memory_mode: u8) {
		use crate::adaptive::{
			build_neuron_metadata, compute_per_example_scores,
			export_genome_for_gpu, train_genome_in_slot, GroupMemory,
		};
		use ram_core::neuron_memory::pack_packed_to_u64;

		if get_trainer().is_err() {
			eprintln!("[multigroup] no Metal device — skipping");
			return;
		}
		let total_input_bits = 16usize;
		let num_train = 300usize;
		let num_classes = 3usize;
		let num_negatives = 2usize; // exhaustive for K=3

		// clusters: neurons [2,2,3], bits: c0/c1 = 8, c2 = 10 → groups
		// (2n,8b)×{c0,c1} + (3n,10b)×{c2} = 2 groups.
		let neurons_flat = vec![2usize, 2, 3];
		let mut bits_flat: Vec<usize> = Vec::new();
		bits_flat.extend([8usize; 4]);
		bits_flat.extend([10usize; 3]);
		// Every neuron sees bits [0..b): class-pair bits 0-5 + noise bits 6+.
		let mut conns: Vec<i64> = Vec::new();
		for &b in &bits_flat {
			for k in 0..b {
				conns.push(k as i64);
			}
		}

		// Example ex: class c = ex % 3 sets bits {2c, 2c+1}; bits 6.. carry
		// ex-derived noise. Distinct classes → distinct addresses, so every
		// (neuron, address) sees one class only → votes always agree.
		let mut bools = vec![false; num_train * total_input_bits];
		let mut targets: Vec<i64> = Vec::with_capacity(num_train);
		for ex in 0..num_train {
			let c = ex % num_classes;
			targets.push(c as i64);
			bools[ex * total_input_bits + 2 * c] = true;
			bools[ex * total_input_bits + 2 * c + 1] = true;
			for b in 6..total_input_bits {
				bools[ex * total_input_bits + b] = (ex >> (b - 6)) & 1 == 1;
			}
		}
		let packed = ram_core::packed_bits::PackedBits::from_bool_slice(&bools, total_input_bits);
		let mut negatives = vec![0i64; num_train * num_negatives];
		for (ex, &t) in targets.iter().enumerate() {
			let mut k = 0;
			for c in 0..num_classes as i64 {
				if c != t {
					negatives[ex * num_negatives + k] = c;
					k += 1;
				}
			}
		}

		// Batched GPU path (the path under test).
		let batched = batched_train_offspring(
			&bits_flat, &neurons_flat, &conns, 1, num_classes, &packed,
			&targets, &negatives, num_train, num_negatives, total_input_bits,
			0.5, 1.0, 42, None, memory_mode,
		).expect("batched multi-group train failed");
		assert_eq!(batched.len(), 1);
		let bexp = &batched[0];
		assert_eq!(bexp.groups.len(), 2, "expected 2 config groups");
		assert_eq!(bexp.sparse_exports.len(), 2);
		assert_eq!(bexp.group_info[0], (true, 0, vec![0, 1]));
		assert_eq!(bexp.group_info[1], (true, 1, vec![2]));
		assert_eq!(bexp.sparse_exports[0].num_neurons, 4);
		assert_eq!(bexp.sparse_exports[1].num_neurons, 3);

		// CPU reference: the per-genome path (same construction as
		// IDSGenomeStreamer), sequential for determinism.
		let bits_per_cluster = per_cluster_max_bits(&bits_flat, &neurons_flat);
		let groups = build_groups(&bits_per_cluster, &neurons_flat);
		let (cluster_neuron_starts, neuron_conn_offsets) =
			build_neuron_metadata(&bits_flat, &neurons_flat);
		let mut cluster_to_group = vec![(0usize, 0usize); num_classes];
		for (gi, gr) in groups.iter().enumerate() {
			for (li, &cid) in gr.cluster_ids.iter().enumerate() {
				cluster_to_group[cid] = (gi, li);
			}
		}
		let mut memories: Vec<GroupMemory> = groups.iter()
			.map(|gr| GroupMemory::new(gr.total_neurons(), gr.bits, memory_mode))
			.collect();
		train_genome_in_slot(
			&mut memories, &groups, &conns, &bits_flat,
			&cluster_neuron_starts, &neuron_conn_offsets, &cluster_to_group,
			&packed, &targets, &negatives, num_train, num_negatives,
			total_input_bits, None, 1.0, 42, memory_mode, None, false,
		);
		let gpu_conns = reorganize_connections_for_gpu(&conns, &bits_flat, &neurons_flat, &groups);
		let cexp = export_genome_for_gpu(&memories, &groups, &gpu_conns);

		// Score-level comparison on the training data (CPU eval on both
		// exports — implementation-independent of the Metal eval path).
		let (packed_u64, wpe) = pack_packed_to_u64(&packed);
		let sb = compute_per_example_scores(
			bexp, &packed, &packed_u64, wpe, num_train, num_classes,
			total_input_bits, 0.5, memory_mode, None, None,
		);
		let sc = compute_per_example_scores(
			&cexp, &packed, &packed_u64, wpe, num_train, num_classes,
			total_input_bits, 0.5, memory_mode, None, None,
		);
		for ex in 0..num_train {
			let argmax = |row: &Vec<f64>| row.iter().enumerate()
				.max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
				.map(|(i, _)| i as i64).unwrap();
			assert_eq!(argmax(&sb[ex]), targets[ex],
				"batched export failed to separate ex {} (scores {:?})", ex, sb[ex]);
			for c in 0..num_classes {
				assert!((sb[ex][c] - sc[ex][c]).abs() < 1e-12,
					"score mismatch ex {} cluster {}: batched={} cpu={}",
					ex, c, sb[ex][c], sc[ex][c]);
			}
		}
	}

	#[test]
	fn multigroup_batched_export_matches_cpu_reference() {
		run_multigroup_parity(ram_core::neuron_memory::MODE_QUAD_WEIGHTED);
	}

	#[test]
	fn multigroup_parity_ternary() {
		run_multigroup_parity(ram_core::neuron_memory::MODE_TERNARY);
	}

	#[test]
	fn multigroup_parity_binary() {
		run_multigroup_parity(ram_core::neuron_memory::MODE_BINARY);
	}

	/// Chunked-vs-unchunked parity. Data is built so every (neuron, example)
	/// write lands on a UNIQUE address (each example is a distinct 16-bit
	/// pattern observed through a per-neuron permutation of all 16 input
	/// bits), so slot content is order-independent by construction and the
	/// comparison is exact in both legacy and OI modes. sample_rate=0.5
	/// exercises `neuron_index_offset`: without it, chunk-local neuron
	/// indices would sample different example subsets and the test fails.
	#[test]
	fn neuron_chunked_matches_unchunked() {
		if get_trainer().is_err() {
			eprintln!("[chunked_tests] no Metal device — skipping");
			return;
		}
		let num_train = 1000usize;
		let total_input_bits = 16usize;
		let n_neurons = 6usize;
		let bits = 16usize;

		let mut bools = vec![false; num_train * total_input_bits];
		for ex in 0..num_train {
			for b in 0..total_input_bits {
				bools[ex * total_input_bits + b] = (ex >> b) & 1 == 1;
			}
		}
		let packed = ram_core::packed_bits::PackedBits::from_bool_slice(&bools, total_input_bits);
		let targets: Vec<i64> = (0..num_train).map(|ex| (ex % 2) as i64).collect();

		let bits_flat = vec![bits; n_neurons];
		let neurons_flat = vec![n_neurons];
		// Per-neuron connection = rotation of 0..16 (unique pattern → unique address)
		let mut conns: Vec<i64> = Vec::with_capacity(n_neurons * bits);
		for n in 0..n_neurons {
			for k in 0..bits {
				conns.push(((k + n) % total_input_bits) as i64);
			}
		}

		let unchunked = batched_train_offspring(
			&bits_flat, &neurons_flat, &conns, 1, 1, &packed, &targets, &[],
			num_train, 0, total_input_bits, 0.5, 0.5, 42, None,
		 ram_core::neuron_memory::MODE_QUAD_WEIGHTED).expect("unchunked train failed");

		// cap: effective=500 → ×2.0 oversize → 1024 slots × 16 B = 16 KB/neuron.
		// Budget of 2 neurons' worth forces 3 chunks of 2.
		let cap = marker_capacity_for_train(num_train, bits, 0.5);
		let chunked = train_single_genome_neuron_chunked(
			&bits_flat, &conns, &packed, &targets, &[], num_train, 0,
			total_input_bits, 0.5, 0.5, 42, None, cap * 16 * 2,
			ram_core::neuron_memory::MODE_QUAD_WEIGHTED,
		).expect("chunked train failed");

		assert_eq!(unchunked.len(), 1);
		assert_eq!(chunked.len(), 1);
		let a = &unchunked[0];
		let b = &chunked[0];
		assert_eq!(a.connections, b.connections, "connections differ");
		assert_eq!(a.group_info, b.group_info, "group_info differs");
		assert_eq!(a.groups.len(), 1);
		assert_eq!(b.groups.len(), 1);
		assert_eq!(a.sparse_exports.len(), 1);
		assert_eq!(b.sparse_exports.len(), 1);
		let sa = &a.sparse_exports[0];
		let sb = &b.sparse_exports[0];
		assert_eq!(sa.num_neurons, sb.num_neurons, "num_neurons differs");
		assert_eq!(sa.counts, sb.counts, "counts differ");
		assert_eq!(sa.offsets, sb.offsets, "offsets differ");
		assert_eq!(sa.keys, sb.keys, "keys differ");
		assert_eq!(sa.values, sb.values, "values differ");
		// Sampling really happened (sr=0.5 → roughly half the examples/neuron)
		let total: u32 = sa.counts.iter().sum();
		assert!(total > 0 && (total as usize) < num_train * n_neurons,
			"sampling did not thin writes: total={}", total);

		// Negative control: the same chunk trained with offset=0 instead of
		// its global start must sample DIFFERENT examples — proves the
		// parity above is earned by neuron_index_offset, not vacuous.
		let chunk_bits = &bits_flat[2..4];
		let chunk_conns = &conns[2 * bits..4 * bits];
		let with_offset = batched_train_core(
			chunk_bits, &[2], chunk_conns, 1, 1, &packed, &targets, &[],
			num_train, 0, total_input_bits, 0.5, 0.5, 42, None, 2,
		 ram_core::neuron_memory::MODE_QUAD_WEIGHTED).expect("offset chunk failed");
		let without_offset = batched_train_core(
			chunk_bits, &[2], chunk_conns, 1, 1, &packed, &targets, &[],
			num_train, 0, total_input_bits, 0.5, 0.5, 42, None, 0,
		 ram_core::neuron_memory::MODE_QUAD_WEIGHTED).expect("no-offset chunk failed");
		assert_ne!(
			with_offset[0].sparse_exports[0].keys,
			without_offset[0].sparse_exports[0].keys,
			"offset had no effect on sampling — negative control failed"
		);
	}

	/// High-z exactness under address collisions (opt-in: requires
	/// WNN_ORDER_INDEPENDENT_TRAIN=1 in the process env — OI is what makes
	/// z-parallel training commutative; skips otherwise).
	///
	/// Small address space (12-bit) + many examples ⇒ every address is
	/// first-touched near-simultaneously by many of the z=1024 threads ⇒
	/// duplicate-key claims are guaranteed. The export's oi_merge must
	/// reconstruct EXACTLY the z=1 result (z=1 has one thread per neuron —
	/// no same-table concurrency, so it is the race-free ground truth).
	/// Also catches slot_nudge_oi retry exhaustion (dropped nudges would
	/// change net tallies ⇒ different cells).
	/// Run: WNN_ORDER_INDEPENDENT_TRAIN=1 cargo test --release oi_z_parity -- --nocapture --test-threads=1
	#[test]
	fn oi_z_parity_with_collisions() {
		if !ram_core::neuron_memory::order_independent_training_enabled() {
			eprintln!("[oi_z_parity] WNN_ORDER_INDEPENDENT_TRAIN not set — skipping");
			return;
		}
		if get_trainer().is_err() {
			eprintln!("[oi_z_parity] no Metal device — skipping");
			return;
		}
		let num_train = 200_000usize;
		let total_input_bits = 16usize;
		let n_neurons = 8usize;
		let bits = 12usize;

		let mut bools = vec![false; num_train * total_input_bits];
		for ex in 0..num_train {
			// 256 distinct 8-bit patterns → ~780 examples per address: heavy
			// same-key concurrency at z=1024 (duplicate storm) while staying
			// well under the 4096-slot capacity even with duplicate slots
			// (table-full drops would make this a capacity test, not a merge
			// test — production runs at ~50% LF with 2× headroom).
			let v = (ex as u32).wrapping_mul(2654435761) >> 16 & 0xFF;
			for b in 0..total_input_bits {
				bools[ex * total_input_bits + b] = (v >> b) & 1 == 1;
			}
		}
		let packed = ram_core::packed_bits::PackedBits::from_bool_slice(&bools, total_input_bits);
		let targets: Vec<i64> = (0..num_train).map(|ex| (ex % 2) as i64).collect();

		let bits_flat = vec![bits; n_neurons];
		let mut conns: Vec<i64> = Vec::with_capacity(n_neurons * bits);
		for n in 0..n_neurons {
			for k in 0..bits {
				conns.push(((k + n) % total_input_bits) as i64);
			}
		}

		let mut runs = Vec::new();
		for z in ["1", "2", "8", "64", "1024"] {
			std::env::set_var("WNN_EXAMPLE_CHUNKS", z);
			let out = batched_train_core(
				&bits_flat, &[n_neurons], &conns, 1, 1, &packed, &targets, &[],
				num_train, 0, total_input_bits, 0.5, 1.0, 42, None, 0,
			 ram_core::neuron_memory::MODE_QUAD_WEIGHTED).expect("oi_z_parity train failed");
			let g = out.into_iter().next().unwrap();
			let mut hist = [0usize; 4];
			for &v in &g.sparse_exports[0].values {
				hist[(v as usize).min(3)] += 1;
			}
			eprintln!("[oi_z_parity] z={:4} keys={} cells F/wF/wT/T = {:?}",
				z, g.sparse_exports[0].keys.len(), hist);
			runs.push(g);
		}
		std::env::remove_var("WNN_EXAMPLE_CHUNKS");

		let a = &runs[0].sparse_exports[0];
		for (i, run) in runs.iter().enumerate().skip(1) {
			let b = &run.sparse_exports[0];
			assert_eq!(a.counts, b.counts, "run {} counts differ from z=1 — unmerged duplicates", i);
			assert_eq!(a.keys, b.keys, "run {} keys differ from z=1", i);
			assert_eq!(a.values, b.values, "run {} cell values differ from z=1 — lost/split tallies", i);
		}
		// The collision setup really collided: ≤256 distinct addresses per
		// neuron (~780 examples each).
		let total: u32 = a.counts.iter().sum();
		assert!((total as usize) <= 256 * n_neurons,
			"collision regime not reached: {} distinct addresses", total);
		eprintln!("[oi_z_parity] exact across z=1 vs z=1024 ({} distinct addresses)", total);
	}

	/// Pure-math gate check: eval-in-place must fire for exactly the shapes
	/// that batched_train_offspring routes through the neuron-chunked path.
	#[test]
	fn budget_eligibility_matches_chunked_regime() {
		// UNSW/CICIDS-scale shapes stay under budget → export path.
		assert!(!single_genome_exceeds_batch_budget(&vec![16; 250], 100_000, 0.25));
		assert!(!single_genome_exceeds_batch_budget(&vec![34; 500], 1_100_000, 0.25));
		// 46M-flow production shape (250n × 48-64b) is the chunked regime.
		assert!(single_genome_exceeds_batch_budget(&vec![48; 250], 37_000_000, 0.25));
		assert!(single_genome_exceeds_batch_budget(&vec![64; 250], 37_000_000, 0.25));
	}

	/// CPU reference for the eval-in-place probe: walk an exported
	/// SparseGpuExport per (example, neuron) with the ORIGINAL (unpadded,
	/// uniform-bits) connections and ACCUMULATE integer votes (F=0, wF=1,
	/// wT=3, T=4; binary-search MISS → QUAD default WEAK_FALSE = 1) into
	/// `votes`. Requires the export to have been produced with
	/// WNN_EXPORT_SKIP_WF=0 so present-wF vs miss-wF distinctions can't hide a
	/// probe bug behind the filter's no-op identity. Returns (hits, misses).
	#[cfg(test)]
	fn accumulate_reference_votes(
		export: &GenomeExport,
		input: &ram_core::packed_bits::PackedBits,
		num_examples: usize,
		n_neurons: usize,
		bits: usize,
		conns: &[i64],
		votes: &mut [u32],
	) -> (usize, usize) {
		const VOTES_X4: [u32; 4] = [0, 1, 3, 4];
		let sparse = &export.sparse_exports[0];
		let (mut hits, mut misses) = (0usize, 0usize);
		for ex in 0..num_examples {
			let row = input.packed_row(ex);
			let mut sum = 0u32;
			for n in 0..n_neurons {
				let addr = ram_core::neuron_memory::compute_address_packed_bytes_sparse(
					row, &conns[n * bits..(n + 1) * bits], bits,
				);
				let start = sparse.offsets[n] as usize;
				let count = sparse.counts[n] as usize;
				let cell = match sparse.keys[start..start + count].binary_search(&addr) {
					Ok(i) => { hits += 1; sparse.values[start + i] }
					Err(_) => { misses += 1; 1u8 }  // miss → QUAD_WEAK_FALSE
				};
				sum += VOTES_X4[(cell as usize).min(3)];
			}
			votes[ex] += sum;
		}
		(hits, misses)
	}

	/// Eval-in-place vote parity (opt-in like oi_z_parity_with_collisions:
	/// requires WNN_ORDER_INDEPENDENT_TRAIN=1; skips otherwise).
	///
	/// Trains one genome chunk-by-chunk (same dispatches as the chunked
	/// production path) and, for each chunk's SINGLE resident table, computes
	/// per-example vote sums BOTH ways:
	///   (a) the probe-eval kernel (marker_probe_eval.metal),
	///   (b) a CPU reference walking that same table's SparseGpuExport
	///       (WNN_EXPORT_SKIP_WF=0 so wF cells are present; misses default wF).
	/// Asserts EXACT integer equality across all eval AND train examples.
	///
	/// Probing the SAME table both ways is deliberate: two separate trainings
	/// of this contention-extreme data (a neuron whose 12-bit window sees only
	/// 4 distinct addresses × ~25K writes at z=1024) occasionally differ by a
	/// single dropped nudge (slot_nudge_oi 64-retry exhaustion) — a
	/// PRE-EXISTING cross-training nondeterminism that also flakes
	/// oi_z_parity_with_collisions ~4/20 isolated reruns. Same-table
	/// comparison isolates what this test is about: the Metal probe's
	/// chain-walk + inline oi_merge + oi_bin_to_cell + miss-default logic vs
	/// the Rust export's, on identical counters.
	///
	/// Data reuses the oi_z_parity collision trick (256 distinct 8-bit train
	/// patterns → address collisions + duplicate-key slots at high z), and the
	/// eval set contains both trained addresses (v < 256) and guaranteed
	/// misses (v ≥ 256 sets input bit 8, always 0 in training). sr=0.5
	/// exercises neuron_index_offset across chunks.
	#[test]
	fn eval_in_place_vote_parity() {
		if !ram_core::neuron_memory::order_independent_training_enabled() {
			eprintln!("[eval_in_place_parity] WNN_ORDER_INDEPENDENT_TRAIN not set — skipping");
			return;
		}
		if get_trainer().is_err() {
			eprintln!("[eval_in_place_parity] no Metal device — skipping");
			return;
		}
		let num_train = 200_000usize;
		let total_input_bits = 16usize;
		let n_neurons = 8usize;
		let bits = 12usize;
		let sample_rate = 0.5f32;

		let mut train_bools = vec![false; num_train * total_input_bits];
		for ex in 0..num_train {
			let v = (ex as u32).wrapping_mul(2654435761) >> 16 & 0xFF;
			for b in 0..total_input_bits {
				train_bools[ex * total_input_bits + b] = (v >> b) & 1 == 1;
			}
		}
		let packed_train = ram_core::packed_bits::PackedBits::from_bool_slice(&train_bools, total_input_bits);
		let targets: Vec<i64> = (0..num_train).map(|ex| (ex % 2) as i64).collect();

		// Eval: 4096 examples over 512 patterns — half hit trained addresses,
		// half (bit 8 set) miss every neuron's table.
		let num_eval = 4096usize;
		let mut eval_bools = vec![false; num_eval * total_input_bits];
		for ex in 0..num_eval {
			let v = (ex % 512) as u32;
			for b in 0..total_input_bits {
				eval_bools[ex * total_input_bits + b] = (v >> b) & 1 == 1;
			}
		}
		let packed_eval = ram_core::packed_bits::PackedBits::from_bool_slice(&eval_bools, total_input_bits);

		let bits_flat = vec![bits; n_neurons];
		let mut conns: Vec<i64> = Vec::with_capacity(n_neurons * bits);
		for n in 0..n_neurons {
			for k in 0..bits {
				conns.push(((k + n) % total_input_bits) as i64);
			}
		}

		// 3 neurons per chunk forces ceil(8/3) = 3 chunks.
		let chunk_n = 3usize;
		let trainer = get_trainer().expect("trainer");
		let device = trainer.device();
		let prober = crate::marker_probe::get_prober(device).expect("prober");

		// Persistent probe buffers (mirror train_single_genome_chunked_scored).
		let (packed_eval_u64, eval_words) = ram_core::neuron_memory::pack_packed_to_u64(&packed_eval);
		let eval_buf = device.new_buffer_with_data(
			packed_eval_u64.as_ptr() as *const _,
			(packed_eval_u64.len() * 8) as u64,
			MTLResourceOptions::StorageModeShared,
		);
		let (packed_train_u64, train_words) = ram_core::neuron_memory::pack_packed_to_u64(&packed_train);
		let train_buf = device.new_buffer_with_data(
			packed_train_u64.as_ptr() as *const _,
			(packed_train_u64.len() * 8) as u64,
			MTLResourceOptions::StorageModeShared,
		);
		let eval_votes_buf = crate::marker_probe::new_zeroed_vote_buffer(device, num_eval);
		let train_votes_buf = crate::marker_probe::new_zeroed_vote_buffer(device, num_train);
		let mut ref_eval = vec![0u32; num_eval];
		let mut ref_train = vec![0u32; num_train];
		let (mut eval_hits, mut eval_misses) = (0usize, 0usize);

		std::env::set_var("WNN_EXPORT_SKIP_WF", "0");
		let mut start = 0usize;
		while start < n_neurons {
			let end = (start + chunk_n).min(n_neurons);
			let chunk_bits = &bits_flat[start..end];
			let chunk_conns = &conns[start * bits..end * bits];
			let chunk_neurons = [end - start];
			let tb = train_batch_to_table(
				chunk_bits, &chunk_neurons, chunk_conns, 1, 1, &packed_train,
				&targets, &[], num_train, 0, total_input_bits, 0.5, sample_rate,
				42, None, start as u32, ram_core::neuron_memory::MODE_QUAD_WEIGHTED,
			).expect("chunk train failed");
			// (a) probe the resident table.
			let (markers_buf, keys_buf, values_buf) =
				tb.gpu_table.metal_buffers().expect("metal buffers");
			prober.probe_accumulate(
				&eval_buf, &tb.conn_buf, &tb.neuron_meta,
				&markers_buf, &keys_buf, &values_buf, &eval_votes_buf,
				num_eval, eval_words, 1,
			).expect("eval probe failed");
			prober.probe_accumulate(
				&train_buf, &tb.conn_buf, &tb.neuron_meta,
				&markers_buf, &keys_buf, &values_buf, &train_votes_buf,
				num_train, train_words, 1,
			).expect("train probe failed");
			// (b) export the SAME table, walk it on CPU.
			let exports = export_trained_batch(&tb, chunk_bits, &chunk_neurons)
				.expect("chunk export failed");
			let (h, m) = accumulate_reference_votes(
				&exports[0], &packed_eval, num_eval, end - start, bits,
				chunk_conns, &mut ref_eval,
			);
			eval_hits += h;
			eval_misses += m;
			accumulate_reference_votes(
				&exports[0], &packed_train, num_train, end - start, bits,
				chunk_conns, &mut ref_train,
			);
			start = end;
		}
		std::env::remove_var("WNN_EXPORT_SKIP_WF");

		assert!(eval_hits > 0 && eval_misses > 0,
			"eval set must exercise both hit and miss paths: hits={} misses={}", eval_hits, eval_misses);
		let probe_eval = crate::marker_probe::read_vote_buffer(&eval_votes_buf, num_eval);
		let probe_train = crate::marker_probe::read_vote_buffer(&train_votes_buf, num_train);
		assert_eq!(probe_eval, ref_eval, "eval vote sums differ from export-walk reference");
		assert_eq!(probe_train, ref_train, "train vote sums differ from export-walk reference");
		eprintln!(
			"[eval_in_place_parity] EXACT: {} eval ({} hits / {} misses) + {} train examples",
			num_eval, eval_hits, eval_misses, num_train
		);

		// Public-entry smoke: the wrapper composes the exact pieces verified
		// above; asserting only shape/bounds here because it RETRAINS (the
		// pre-existing cross-training drop nondeterminism, see doc comment).
		let cap = marker_capacity_for_train(num_train, bits, sample_rate);
		let scored = train_single_genome_chunked_scored_with_budget(
			&bits_flat, &conns, &packed_train, &targets, &[], num_train, 0,
			total_input_bits, 0.5, sample_rate, 42, None,
			&packed_eval, num_eval, true, cap * 16 * chunk_n,
		 ram_core::neuron_memory::MODE_QUAD_WEIGHTED).expect("scored wrapper failed");
		assert_eq!(scored.eval_votes.len(), num_eval);
		let train_votes = scored.train_votes.expect("train votes requested");
		assert_eq!(train_votes.len(), num_train);
		let max_vote = (4 * n_neurons) as u32;
		assert!(scored.eval_votes.iter().all(|&v| v <= max_vote), "eval vote exceeds 4×neurons");
		assert!(train_votes.iter().all(|&v| v <= max_vote), "train vote exceeds 4×neurons");
	}

	/// Production-shape full-genome cycle benchmark (opt-in: WNN_BENCH=1).
	/// 250n × 64b × 10M examples @ sr=0.25 — the 46M-flow offspring shape
	/// scaled 3× down on examples (attribution ratios hold; memory-safe next
	/// to a live worker). Run with WNN_GPU_BATCHED_TRAIN_TRACE=1 for the
	/// per-phase breakdown (alloc / kernel / export+build).
	/// Run: WNN_BENCH=1 WNN_ORDER_INDEPENDENT_TRAIN=1 WNN_GPU_BATCHED_TRAIN_TRACE=1 \
	///   cargo test --release bench_prod_genome -- --nocapture --test-threads=1
	#[test]
	fn bench_prod_genome() {
		if std::env::var("WNN_BENCH").ok().as_deref() != Some("1") {
			return;
		}
		if get_trainer().is_err() {
			eprintln!("[bench_prod] no Metal device — skipping");
			return;
		}
		let num_train = 10_000_000usize;
		let total_input_bits = 96usize;
		let n_neurons = 250usize;
		let bits = 64usize;

		let mut bools = vec![false; num_train * total_input_bits];
		let mut state = 0x9E3779B97F4A7C15u64;
		for chunk in bools.chunks_mut(64) {
			state ^= state << 13;
			state ^= state >> 7;
			state ^= state << 17;
			for (i, b) in chunk.iter_mut().enumerate() {
				*b = (state >> (i % 64)) & 1 == 1;
			}
		}
		let packed = ram_core::packed_bits::PackedBits::from_bool_slice(&bools, total_input_bits);
		drop(bools);
		let targets: Vec<i64> = (0..num_train).map(|ex| (ex % 2) as i64).collect();

		let bits_flat = vec![bits; n_neurons];
		let mut conns: Vec<i64> = Vec::with_capacity(n_neurons * bits);
		for n in 0..n_neurons {
			for k in 0..bits {
				conns.push((((n * 37 + k * 11) ^ (k * 5)) % total_input_bits) as i64);
			}
		}

		// 2M-example eval set (shared by both paths' scoring phases).
		let num_eval = 2_000_000usize;
		let mut eval_bools = vec![false; num_eval * total_input_bits];
		let mut estate = 0xD1B54A32D192ED03u64;
		for chunk in eval_bools.chunks_mut(64) {
			estate ^= estate << 13;
			estate ^= estate >> 7;
			estate ^= estate << 17;
			for (i, b) in chunk.iter_mut().enumerate() {
				*b = (estate >> (i % 64)) & 1 == 1;
			}
		}
		let packed_eval = ram_core::packed_bits::PackedBits::from_bool_slice(&eval_bools, total_input_bits);
		drop(eval_bools);

		// EXPORT PATH: train + sorted export, then the sparse-GPU eval pass it
		// owes before any metric exists (same kernel compute_per_example_scores
		// uses in production).
		let t0 = std::time::Instant::now();
		let out = batched_train_offspring(
			&bits_flat, &[n_neurons], &conns, 1, 1, &packed, &targets, &[],
			num_train, 0, total_input_bits, 0.5, 0.25, 42, None,
		 ram_core::neuron_memory::MODE_QUAD_WEIGHTED).expect("bench_prod train failed");
		let export_train_ms = t0.elapsed().as_secs_f64() * 1000.0;
		let keys: usize = out[0].sparse_exports.iter().map(|s| s.keys.len()).sum();
		let (packed_eval_u64, eval_words) = ram_core::neuron_memory::pack_packed_to_u64(&packed_eval);
		let t_score = std::time::Instant::now();
		let sparse_eval = ram_core::metal_sparse::MetalSparseEvaluator::new().expect("sparse evaluator");
		let scores = crate::adaptive::evaluate_group_sparse_gpu(
			&sparse_eval, &packed_eval_u64, &out[0].connections,
			&out[0].sparse_exports[0], &out[0].groups[0],
			num_eval, eval_words, 2, 0.5,
		).expect("export-path eval failed");
		let export_eval_ms = t_score.elapsed().as_secs_f64() * 1000.0;
		let export_total_ms = export_train_ms + export_eval_ms;
		eprintln!(
			"[bench_prod] EXPORT PATH 250n×64b×10M: train+export={:.0}ms + sparse-eval({}ex)={:.0}ms → total={:.0}ms  exported_keys={} ({:.1} GB export)",
			export_train_ms, num_eval, export_eval_ms, export_total_ms, keys, (keys * 9) as f64 / 1e9,
		);
		let export_scores_probe: Vec<f32> = scores.iter().take(4).copied().collect();
		drop(scores);
		drop(out);

		// FUSED PATH (OI only): same training dispatches, each chunk scored by
		// probing the resident table — no sorted export, no separate eval pass.
		if !ram_core::neuron_memory::order_independent_training_enabled() {
			eprintln!("[bench_prod] fused variant needs WNN_ORDER_INDEPENDENT_TRAIN=1 — skipping");
			return;
		}
		let t1 = std::time::Instant::now();
		let scored = train_single_genome_chunked_scored(
			&bits_flat, &conns, &packed, &targets, &[], num_train, 0,
			total_input_bits, 0.5, 0.25, 42, None,
			&packed_eval, num_eval, false,
		 ram_core::neuron_memory::MODE_QUAD_WEIGHTED).expect("bench_prod fused failed");
		let fused_ms = t1.elapsed().as_secs_f64() * 1000.0;
		let nonzero = scored.eval_votes.iter().filter(|&&v| v > 0).count();
		eprintln!(
			"[bench_prod] FUSED PATH  250n×64b×10M + {}-example probe: total={:.0}ms (export+eval skipped) → {:.2}x vs export-path total  nonzero_votes={}  (export scores head: {:?})",
			num_eval, fused_ms, export_total_ms / fused_ms.max(1.0), nonzero, export_scores_probe,
		);
	}

	/// Occupancy benchmark for the neuron-chunked path (opt-in: WNN_BENCH=1).
	/// Production-shaped: 32 neurons × 24 bits × 10M examples @ sr=0.25 —
	/// the 46M-flow chunk regime where ng×n=32 caps the grid at 256 threads.
	/// Sweeps WNN_EXAMPLE_CHUNKS to measure z-axis occupancy scaling.
	/// Run: WNN_BENCH=1 cargo test --release bench_chunked_z_sweep -- --nocapture
	#[test]
	fn bench_chunked_z_sweep() {
		if std::env::var("WNN_BENCH").ok().as_deref() != Some("1") {
			return;
		}
		if get_trainer().is_err() {
			eprintln!("[bench] no Metal device — skipping");
			return;
		}
		let num_train = 10_000_000usize;
		let total_input_bits = 96usize;
		let n_neurons = 32usize;
		let bits = 24usize;

		let mut bools = vec![false; num_train * total_input_bits];
		let mut state = 0x2545F4914F6CDD1Du64;
		for chunk in bools.chunks_mut(64) {
			state ^= state << 13;
			state ^= state >> 7;
			state ^= state << 17;
			for (i, b) in chunk.iter_mut().enumerate() {
				*b = (state >> (i % 64)) & 1 == 1;
			}
		}
		let packed = ram_core::packed_bits::PackedBits::from_bool_slice(&bools, total_input_bits);
		drop(bools);
		let targets: Vec<i64> = (0..num_train).map(|ex| (ex % 2) as i64).collect();

		let bits_flat = vec![bits; n_neurons];
		let mut conns: Vec<i64> = Vec::with_capacity(n_neurons * bits);
		for n in 0..n_neurons {
			for k in 0..bits {
				conns.push((((n * 37 + k * 11) ^ (k * 5)) % total_input_bits) as i64);
			}
		}

		for z in [8u32, 64, 256, 1024] {
			std::env::set_var("WNN_EXAMPLE_CHUNKS", z.to_string());
			let t0 = std::time::Instant::now();
			let out = batched_train_core(
				&bits_flat, &[n_neurons], &conns, 1, 1, &packed, &targets, &[],
				num_train, 0, total_input_bits, 0.5, 0.25, 42, None, 0,
			 ram_core::neuron_memory::MODE_QUAD_WEIGHTED).expect("bench train failed");
			let total: u32 = out[0].sparse_exports.iter().map(|s| s.counts.iter().sum::<u32>()).sum();
			eprintln!("[bench] z={:4}  wall={:8.1}ms  writes={}", z, t0.elapsed().as_secs_f64() * 1000.0, total);
		}
		std::env::remove_var("WNN_EXAMPLE_CHUNKS");
	}
}

}  // mod batched_path

#[cfg(target_os = "macos")]
pub use batched_path::{
	batched_train_offspring, single_genome_exceeds_batch_budget,
	train_single_genome_chunked_scored, ChunkedVoteSums,
};
