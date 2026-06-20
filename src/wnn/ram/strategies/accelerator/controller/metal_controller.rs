//! Metal GPU-batched closed-loop controller eval (drone attitude).
//!
//! Drives shaders/controller_rollout.metal: one thread = one (genome, episode)
//! rollout. Trains stay on CPU (branchy QSR solver + DashMap writes); this is
//! the forward-rollout EVAL only — the GPU half of the hybrid. All controller
//! genomes are shape-identical, so a single uniform kernel handles the whole
//! population. See controller.rs for the CPU reference being mirrored.

use metal::*;
use std::mem;

use pyo3::prelude::*;

use ram_core::cancel::check_cancel;
use crate::controller::WnnController;

/// Chunking heuristic for cooperative cancellation. We split a full
/// (num_genomes × num_episodes) dispatch into chunks of `EPISODES_PER_CHUNK`
/// episodes each, polling the cancel flag between chunks.
///
/// 31/05/2026 history:
///   - Initial ship at 5 (sub-100ms cancel) caused stalls under heavy GPU
///     contention from a co-resident IDS worker (commit 63b60a49 bumped
///     default to 100 = single dispatch).
///   - For solo runs (e.g. curriculum-GA, dedicated controller search) the
///     contention concern doesn't apply and tight chunking gives sub-100ms
///     cancel response. Default is now 25 (4 chunks per typical 100-episode
///     call) — middle ground between the original "stall under IDS load"
///     and "no cancel granularity at all". Override via WNN_CONTROLLER_CHUNK
///     for tighter cancel (set to 5) or larger chunks (set to 100+).
fn episodes_per_chunk() -> usize {
	std::env::var("WNN_CONTROLLER_CHUNK")
		.ok()
		.and_then(|v| v.parse::<usize>().ok())
		.filter(|&v| v > 0)
		.unwrap_or(25)
}

#[repr(C)]
// repr(C) guarantees field order/layout matches the Metal `Params` struct. All
// fields are 4-byte (u32/f32) so the two are tightly packed and identical.
#[repr(C)]
#[derive(Clone, Copy)]
struct RolloutParams {
	num_genomes: u32,
	num_episodes: u32,
	steps: u32,
	num_motors: u32,
	levels: u32,
	n_state: u32,
	sbpn: u32,
	obpn: u32,
	bpf: u32,
	window: u32,
	frame_bits: u32,
	sensor_total: u32,
	state_bits_in: u32,
	dt: f32,
	arm_length: f32,
	k_thrust: f32,
	k_drag: f32,
	inertia0: f32,
	inertia1: f32,
	inertia2: f32,
	gravity: f32,
	target0: f32,
	target1: f32,
	target2: f32,
	// Delta-control mode (must match the shader Params layout exactly). When
	// delta_control!=0 the kernel decodes to a per-step PWM delta + leaky
	// accumulator, matching controller.rs step() (was absolute-only before).
	delta_control: u32,
	delta_max: f32,
	delta_leak: f32,
	// H2 observation-feature config (layout must match the shader Params exactly).
	num_features: u32,
	obs_tilt_p: u32,
	obs_tilt_i: u32,
	obs_peraxis_p: u32,
	obs_peraxis_i: u32,
	obs_pwm: u32,
	integral_leak: f32,
	integral_scale: f32,
	decouple_outputs: u32,   // H3: 4 banks are controls [T,τr,τp,τy] → mix to motors
}

pub struct ControllerRolloutEvaluator {
	device: Device,
	queue: CommandQueue,
	pipeline: ComputePipelineState,
}

impl ControllerRolloutEvaluator {
	pub fn new() -> Result<Self, String> {
		let device = Device::system_default().ok_or("No Metal device found")?;
		let queue = device.new_command_queue();
		let src = concat!(
			include_str!("../core/shaders/common.metal"), "\n",
			include_str!("../core/shaders/marker_slots.metal"), "\n",  // GPU cell-write primitives (controller_train)
			include_str!("shaders/controller_rollout.metal"),
		);
		let library = device
			.new_library_with_source(src, &CompileOptions::new())
			.map_err(|e| format!("controller_rollout.metal compile failed: {e}"))?;
		let func = library
			.get_function("controller_rollout", None)
			.map_err(|e| format!("kernel controller_rollout not found: {e}"))?;
		let pipeline = device
			.new_compute_pipeline_state_with_function(&func)
			.map_err(|e| format!("pipeline creation failed: {e}"))?;
		Ok(Self { device, queue, pipeline })
	}

	fn buf<T>(&self, data: &[T]) -> Buffer {
		// Metal rejects zero-length buffers; pad to 1 element (never read — count=0).
		let n = data.len().max(1);
		let bytes = (n * mem::size_of::<T>()) as u64;
		if data.is_empty() {
			self.device.new_buffer(bytes, MTLResourceOptions::StorageModeShared)
		} else {
			self.device.new_buffer_with_data(
				data.as_ptr() as *const _, bytes, MTLResourceOptions::StorageModeShared)
		}
	}

	/// Score a whole population closed-loop. Returns per-genome
	/// (mean_reward, mean_attitude_error_rad, stable_rate). q0/omega0 are the
	/// per-episode initial conditions (shared across genomes), flat
	/// (num_episodes*4) and (num_episodes*3) — sampled host-side to match the
	/// CPU eval set for parity.
	///
	/// Cooperative cancellation (added 31/05/2026): the full
	/// (num_genomes × num_episodes) dispatch is split into chunks of
	/// `EPISODES_PER_CHUNK` episodes each. Between chunks we poll
	/// `check_cancel()`; if set, we stop early and return the aggregate over
	/// whatever episodes completed. Genomes that received zero completed
	/// episodes get (0.0, 0.0, 0.0) — sentinel for "cancelled before any
	/// data" — so callers can filter them out of population statistics if
	/// they choose. The connectivity + sparse exports buffers are uploaded
	/// ONCE up front (per-controller state is constant across chunks); only
	/// the per-chunk q0/omega0/output buffers are re-allocated. Per-chunk
	/// overhead is on the order of milliseconds — negligible vs the ~25ms
	/// kernel work per chunk.
	#[allow(clippy::too_many_arguments)]
	pub fn score(
		&self,
		controllers: &[PyRef<WnnController>],
		q0: &[f32],
		omega0: &[f32],
		num_episodes: usize,
		steps: usize,
		sim: (f32, f32, f32, [f32; 3], f32),   // (dt, arm, k_thrust, inertia, gravity) ... k_drag below
		k_drag: f32,
		target: [f32; 3],
	) -> Result<Vec<(f64, f64, f64, f64, f64)>, String> {
		let g = controllers.len();
		if g == 0 {
			return Ok(vec![]);
		}
		let (num_motors, levels, n_state, sbpn, obpn, bpf, window) = controllers[0].gpu_dims();
		// Delta-control mode (uniform across the population) so the kernel decodes
		// the SAME way step() does (was absolute-only → wrong for delta controllers).
		let (delta_control, delta_max, delta_leak) = controllers[0].delta_params();
		// H2 observation-feature config (uniform); num_features drives frame sizing
		// (was hardcoded 9 → ignored the H2 extras).
		let (num_features, obs_tilt_p, obs_tilt_i, obs_peraxis_p, obs_peraxis_i,
		     obs_pwm, integral_leak, integral_scale, decouple_outputs) = controllers[0].obs_params();
		let num_out = num_motors * levels;
		let frame_bits = num_features * bpf;
		let sensor_total = window * frame_bits;
		let state_bits_in = 2 * n_state;

		// Concatenate connectivity + sparse exports across the population,
		// re-basing per-neuron offsets into the global key arrays. This is
		// done ONCE; the buffers are reused across all chunks below.
		let mut state_conns: Vec<i32> = Vec::with_capacity(g * n_state * sbpn);
		let mut out_conns: Vec<i32> = Vec::with_capacity(g * num_out * obpn);
		let mut s_keys: Vec<u64> = Vec::new();
		let mut s_vals: Vec<u8> = Vec::new();
		let mut s_off: Vec<u32> = Vec::with_capacity(g * n_state);
		let mut s_cnt: Vec<u32> = Vec::with_capacity(g * n_state);
		let mut o_keys: Vec<u64> = Vec::new();
		let mut o_vals: Vec<u8> = Vec::new();
		let mut o_off: Vec<u32> = Vec::with_capacity(g * num_out);
		let mut o_cnt: Vec<u32> = Vec::with_capacity(g * num_out);

		for c in controllers {
			let (sc, oc, sexp, oexp) = c.gpu_export();
			state_conns.extend(sc.iter().map(|&x| x as i32));
			out_conns.extend(oc.iter().map(|&x| x as i32));
			// state memory: rebase offsets by current global key length
			let s_base = s_keys.len() as u32;
			s_keys.extend_from_slice(&sexp.keys);
			s_vals.extend_from_slice(&sexp.values);
			for n in 0..n_state {
				s_off.push(s_base + sexp.offsets[n]);
				s_cnt.push(sexp.counts[n]);
			}
			let o_base = o_keys.len() as u32;
			o_keys.extend_from_slice(&oexp.keys);
			o_vals.extend_from_slice(&oexp.values);
			for n in 0..num_out {
				o_off.push(o_base + oexp.offsets[n]);
				o_cnt.push(oexp.counts[n]);
			}
		}

		let (dt, arm, k_thrust, inertia, gravity) = sim;

		// Static input buffers — allocated once, reused across chunks.
		let b_sc = self.buf(&state_conns);
		let b_oc = self.buf(&out_conns);
		let b_sk = self.buf(&s_keys);
		let b_sv = self.buf(&s_vals);
		let b_so = self.buf(&s_off);
		let b_scn = self.buf(&s_cnt);
		let b_ok = self.buf(&o_keys);
		let b_ov = self.buf(&o_vals);
		let b_oo = self.buf(&o_off);
		let b_ocn = self.buf(&o_cnt);
		let b_th = self.buf(controllers[0].thresholds_ref());

		// Per-genome aggregators (initialised to "no data yet"). When a chunk
		// completes we accumulate; when cancellation hits we return the
		// aggregate over `completed_episodes` only.
		let mut sum_reward_per_g = vec![0.0f64; g];
		let mut sum_mean_err_per_g = vec![0.0f64; g];
		let mut stable_count_per_g = vec![0usize; g];
		let mut sum_jerk_per_g = vec![0.0f64; g];   // mean |Δpwm| per episode, summed
		let mut sum_mono_per_g = vec![0.0f64; g];   // last-step thermometer violations, summed
		let stable_thresh = (5.0_f64).to_radians();

		let chunk_size = episodes_per_chunk();
		let mut completed_episodes: usize = 0;
		let mut chunk_start: usize = 0;
		while chunk_start < num_episodes {
			// Poll cancellation flag at the chunk boundary. Cheap (relaxed
			// atomic load); how often this fires depends on chunk_size.
			if check_cancel() {
				break;
			}
			let chunk_end = (chunk_start + chunk_size).min(num_episodes);
			let chunk_ep_count = chunk_end - chunk_start;

			// Slice q0 / omega0 for this chunk. The kernel indexes them by
			// the chunk-local episode index, so we only pass the chunk slice.
			let q0_chunk = &q0[chunk_start * 4 .. chunk_end * 4];
			let w0_chunk = &omega0[chunk_start * 3 .. chunk_end * 3];

			let chunk_params = RolloutParams {
				num_genomes: g as u32, num_episodes: chunk_ep_count as u32, steps: steps as u32,
				num_motors: num_motors as u32, levels: levels as u32, n_state: n_state as u32,
				sbpn: sbpn as u32, obpn: obpn as u32, bpf: bpf as u32, window: window as u32,
				frame_bits: frame_bits as u32, sensor_total: sensor_total as u32,
				state_bits_in: state_bits_in as u32,
				dt, arm_length: arm, k_thrust, k_drag,
				inertia0: inertia[0], inertia1: inertia[1], inertia2: inertia[2], gravity,
				target0: target[0], target1: target[1], target2: target[2],
				delta_control: if delta_control { 1 } else { 0 }, delta_max, delta_leak,
				num_features: num_features as u32,
				obs_tilt_p: obs_tilt_p as u32, obs_tilt_i: obs_tilt_i as u32,
				obs_peraxis_p: obs_peraxis_p as u32, obs_peraxis_i: obs_peraxis_i as u32,
				obs_pwm: obs_pwm as u32,
				integral_leak, integral_scale,
				decouple_outputs: decouple_outputs as u32,
			};

			let b_q0 = self.buf(q0_chunk);
			let b_w0 = self.buf(w0_chunk);
			let b_par = self.device.new_buffer_with_data(
				&chunk_params as *const _ as *const _,
				mem::size_of::<RolloutParams>() as u64,
				MTLResourceOptions::StorageModeShared);

			let n_out_chunk = g * chunk_ep_count;
			let mk_out = |bytes: usize| {
				let b = self.device.new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
				unsafe { std::ptr::write_bytes(b.contents() as *mut u8, 0, bytes); }
				b
			};
			let b_reward = mk_out(n_out_chunk * mem::size_of::<f32>());
			let b_sumerr = mk_out(n_out_chunk * mem::size_of::<f32>());
			let b_steps = mk_out(n_out_chunk * mem::size_of::<u32>());
			let b_div = mk_out(n_out_chunk * mem::size_of::<u32>());
			let b_jerk = mk_out(n_out_chunk * mem::size_of::<f32>());
			let b_mono = mk_out(n_out_chunk * mem::size_of::<f32>());

			let cmd = self.queue.new_command_buffer();
			let enc = cmd.new_compute_command_encoder();
			enc.set_compute_pipeline_state(&self.pipeline);
			let bufs: [&Buffer; 20] = [
				&b_sc, &b_oc, &b_sk, &b_sv, &b_so, &b_scn, &b_ok, &b_ov, &b_oo, &b_ocn,
				&b_th, &b_q0, &b_w0, &b_par, &b_reward, &b_sumerr, &b_steps, &b_div,
				&b_jerk, &b_mono,
			];
			for (i, b) in bufs.iter().enumerate() {
				enc.set_buffer(i as u64, Some(b), 0);
			}
			let grid = MTLSize::new(g as u64, chunk_ep_count as u64, 1);
			let tg = MTLSize::new(8.min(g as u64), 8.min(chunk_ep_count as u64), 1);
			enc.dispatch_threads(grid, tg);
			enc.end_encoding();
			cmd.commit();
			cmd.wait_until_completed();

			// Accumulate this chunk's results into per-genome totals.
			let reward = unsafe { std::slice::from_raw_parts(b_reward.contents() as *const f32, n_out_chunk) };
			let sumerr = unsafe { std::slice::from_raw_parts(b_sumerr.contents() as *const f32, n_out_chunk) };
			let stepsv = unsafe { std::slice::from_raw_parts(b_steps.contents() as *const u32, n_out_chunk) };
			let divv = unsafe { std::slice::from_raw_parts(b_div.contents() as *const u32, n_out_chunk) };
			let jerkv = unsafe { std::slice::from_raw_parts(b_jerk.contents() as *const f32, n_out_chunk) };
			let monov = unsafe { std::slice::from_raw_parts(b_mono.contents() as *const f32, n_out_chunk) };
			for gi in 0..g {
				for ce in 0..chunk_ep_count {
					let idx = gi * chunk_ep_count + ce;
					let st = stepsv[idx].max(1) as f64;
					let mean_err = sumerr[idx] as f64 / st;
					sum_reward_per_g[gi] += reward[idx] as f64;
					sum_mean_err_per_g[gi] += mean_err;
					sum_jerk_per_g[gi] += jerkv[idx] as f64;
					sum_mono_per_g[gi] += monov[idx] as f64;
					if divv[idx] == 0 && mean_err <= stable_thresh {
						stable_count_per_g[gi] += 1;
					}
				}
			}
			completed_episodes = chunk_end;
			chunk_start = chunk_end;
		}

		// Aggregate per-genome over completed episodes only. If none completed
		// (cancellation hit before the first chunk), all genomes get the
		// sentinel (0.0, 0.0, 0.0).
		let mut out = Vec::with_capacity(g);
		if completed_episodes == 0 {
			for _ in 0..g {
				out.push((0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64));
			}
		} else {
			let n = completed_episodes as f64;
			for gi in 0..g {
				out.push((
					sum_reward_per_g[gi] / n,
					sum_mean_err_per_g[gi] / n,
					stable_count_per_g[gi] as f64 / n,
					sum_jerk_per_g[gi] / n,
					sum_mono_per_g[gi] / n,
				));
			}
		}
		Ok(out)
	}
}

/// PyO3 entry: score a population of controllers on the GPU. Returns per-genome
/// (mean_reward, mean_attitude_error_rad, stable_rate). Sim params default to
/// AttitudeSim's defaults so the rollout physics matches the CPU sim.
#[pyfunction]
#[pyo3(signature = (
	controllers, q0, omega0, num_episodes, steps,
	dt = 0.001, arm_length = 0.075, k_thrust = 2.4, k_drag = 0.05,
	inertia = [0.0023, 0.0023, 0.0046], gravity = 9.81,
	target = [0.0, 0.0, 0.0],
))]
#[allow(clippy::too_many_arguments)]
pub fn score_controllers_metal(
	controllers: Vec<PyRef<WnnController>>,
	q0: Vec<f32>,
	omega0: Vec<f32>,
	num_episodes: usize,
	steps: usize,
	dt: f32,
	arm_length: f32,
	k_thrust: f32,
	k_drag: f32,
	inertia: [f32; 3],
	gravity: f32,
	target: [f32; 3],
) -> PyResult<Vec<(f64, f64, f64, f64, f64)>> {
	let evaluator = ControllerRolloutEvaluator::new()
		.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
	evaluator
		.score(&controllers, &q0, &omega0, num_episodes, steps,
		       (dt, arm_length, k_thrust, inertia, gravity), k_drag, target)
		.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
}

// =============================================================================
// ControllerTrainer — GPU host for the controller_train kernel (split_retrain_output
// on GPU). Thread = genome. Uploads frozen state memory + connections + recorded
// trajectories + an EMPTY output marker table (values pre-set to the hover sentinel
// 2), dispatches, and reads back the trained output cells per (genome, neuron).
// =============================================================================
const MARKER_FINAL_U32: u32 = 0xFFFF_FFFFu32;

#[repr(C)]
#[derive(Clone, Copy)]
struct TrainParams {
	num_genomes: u32, n_state: u32, sbpn: u32, obpn: u32,
	num_motors: u32, levels: u32, bpf: u32, window: u32,
	frame_bits: u32, sensor_total: u32, num_features: u32,
	obs_tilt_p: u32, obs_tilt_i: u32, obs_peraxis_p: u32, obs_peraxis_i: u32, obs_pwm: u32,
	integral_leak: f32, integral_scale: f32,
	decouple_outputs: u32, delta_control: u32, selective: u32,
	target0: f32, target1: f32, target2: f32,
}

/// Per-genome recorded trajectory batch, flat across genomes (matches the kernel's
/// ep_base/ep_count/step_base/step_count layout). gyros/accels/targets are *3 per
/// step; pid_pwms is *4. All in CPU-collection order (episode 0..E, step 0..T).
pub struct TrainBatch<'a> {
	pub ep_base: &'a [u32],
	pub ep_count: &'a [u32],
	pub step_base: &'a [u32],
	pub step_count: &'a [u32],
	pub gyros: &'a [f32],
	pub accels: &'a [f32],
	pub targets: &'a [f32],
	pub pid_pwms: &'a [f32],
	pub selective: bool,
	pub target_rpy: [f32; 3],
}

pub struct ControllerTrainer {
	device: Device,
	queue: CommandQueue,
	pipeline: ComputePipelineState,         // controller_train
	record_pipeline: ComputePipelineState,  // controller_record (P2a)
	scan_pipeline: ComputePipelineState,    // controller_scan (P2b)
	sep_walk_pipeline: ComputePipelineState, // controller_sep_walk (P3 Type-1)
	sep_counts_pipeline: ComputePipelineState, // controller_sep_counts (P3 Type-2)
	sep_bidir_pipeline: ComputePipelineState, // controller_sep_bidir (P3 Type-2, strict-FP lib)
	plant_table_pipeline: ComputePipelineState, // controller_plant_table (P4 latch + counter)
	plant_bidir_pipeline: ComputePipelineState, // controller_plant_bidir (P4 bidir counter)
}

/// One conflict from the GPU scan (twin of controller_split::Conflict). `out_in`
/// is the COARSE bucket key; `instances` are record indices ASCENDING; `spread`
/// is the per-motor PWM spread.
pub struct ScanConflict {
	pub out_in: Vec<bool>,    // coarse bucket key (compared bit-for-bit in the parity test)
	pub instances: Vec<usize>,
	pub spread: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SepParams {
	num_conflicts: u32,
	num_bits: u32,
	sil: u32,
}

// Same layout as the Metal CountParams (P3 Type-2 window-count primitive).
#[repr(C)]
#[derive(Clone, Copy)]
struct CountParams {
	num_conflicts: u32,
	num_bits: u32,
	sil: u32,
}

// Same layout as the Metal PlantParams (P4 planting truth-table write).
#[repr(C)]
#[derive(Clone, Copy)]
struct PlantParams {
	num_records: u32,
	sbpn: u32,
	state_words: u32,
	slot_cap: u32,
	num_rel: u32,
}

// Same layout as the Metal BidirPlantParams (P4 dense bidir table).
#[repr(C)]
#[derive(Clone, Copy)]
struct BidirPlantParams {
	sbpn: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ScanParams {
	num_records: u32,
	out_input_len: u32,
	out_words: u32,
	frame_bits: u32,
	bpf: u32,
	num_features: u32,
	k: u32,
	key_words: u32,
	slot_capacity: u32,
}

impl ControllerTrainer {
	pub fn new() -> Result<Self, String> {
		let device = Device::system_default().ok_or("No Metal device found")?;
		let queue = device.new_command_queue();
		let src = concat!(
			include_str!("../core/shaders/common.metal"), "\n",
			include_str!("../core/shaders/marker_slots.metal"), "\n",
			include_str!("shaders/controller_rollout.metal"),
		);
		let library = device
			.new_library_with_source(src, &CompileOptions::new())
			.map_err(|e| format!("controller_train shader compile failed: {e}"))?;
		let mk = |name: &str| -> Result<ComputePipelineState, String> {
			let func = library.get_function(name, None)
				.map_err(|e| format!("kernel {name} not found: {e}"))?;
			device.new_compute_pipeline_state_with_function(&func)
				.map_err(|e| format!("{name} pipeline creation failed: {e}"))
		};
		let pipeline = mk("controller_train")?;
		let record_pipeline = mk("controller_record")?;
		let scan_pipeline = mk("controller_scan")?;
		let sep_walk_pipeline = mk("controller_sep_walk")?;
		let sep_counts_pipeline = mk("controller_sep_counts")?;
		let plant_table_pipeline = mk("controller_plant_table")?;
		let plant_bidir_pipeline = mk("controller_plant_bidir")?;

		// The bidir Pearson kernel needs STRICT FP (fast-math OFF → IEEE div/sqrt;
		// contract OFF via the in-file pragma → no FMA fusion) to bit-match Rust's
		// f32 two-pass pearson. Compile it in its OWN library so these flags can't
		// perturb the physics kernels' existing parity.
		let strict_opts = CompileOptions::new();
		strict_opts.set_fast_math_enabled(false);
		let sep_lib = device
			.new_library_with_source(include_str!("shaders/controller_sep.metal"), &strict_opts)
			.map_err(|e| format!("controller_sep.metal compile failed: {e}"))?;
		let sep_func = sep_lib.get_function("controller_sep_bidir", None)
			.map_err(|e| format!("kernel controller_sep_bidir not found: {e}"))?;
		let sep_bidir_pipeline = device.new_compute_pipeline_state_with_function(&sep_func)
			.map_err(|e| format!("controller_sep_bidir pipeline creation failed: {e}"))?;

		Ok(Self { device, queue, pipeline, record_pipeline, scan_pipeline,
			sep_walk_pipeline, sep_counts_pipeline, sep_bidir_pipeline, plant_table_pipeline, plant_bidir_pipeline })
	}

	fn buf<T>(&self, data: &[T]) -> Buffer {
		let n = data.len().max(1);
		let bytes = (n * mem::size_of::<T>()) as u64;
		if data.is_empty() {
			self.device.new_buffer(bytes, MTLResourceOptions::StorageModeShared)
		} else {
			self.device.new_buffer_with_data(
				data.as_ptr() as *const _, bytes, MTLResourceOptions::StorageModeShared)
		}
	}

	/// Train output cells on the GPU. Returns, per (genome, neuron), the sorted
	/// trained (address, cell) entries — the GPU twin of each genome's
	/// output_memory after split_retrain_output. Outer index = g*num_out + n.
	pub fn train(
		&self,
		controllers: &[&WnnController],
		batch: &TrainBatch,
	) -> Result<Vec<Vec<(u64, u8)>>, String> {
		let g = controllers.len();
		if g == 0 { return Ok(vec![]); }
		let (num_motors, levels, n_state, sbpn, obpn, bpf, window) = controllers[0].gpu_dims();
		let (num_features, obs_tilt_p, obs_tilt_i, obs_peraxis_p, obs_peraxis_i,
		     obs_pwm, integral_leak, integral_scale, decouple_outputs) = controllers[0].obs_params();
		let (delta_control, _dmax, _dleak) = controllers[0].delta_params();
		let num_out = num_motors * levels;
		let frame_bits = num_features * bpf;
		let sensor_total = window * frame_bits;

		// Concatenate frozen state memory + connections (rebased), as in score().
		let mut state_conns: Vec<i32> = Vec::with_capacity(g * n_state * sbpn);
		let mut out_conns: Vec<i32> = Vec::with_capacity(g * num_out * obpn);
		let mut s_keys: Vec<u64> = Vec::new();
		let mut s_vals: Vec<u8> = Vec::new();
		let mut s_off: Vec<u32> = Vec::with_capacity(g * n_state);
		let mut s_cnt: Vec<u32> = Vec::with_capacity(g * n_state);
		for c in controllers {
			let (sc, oc, sexp, _oexp) = c.gpu_export();
			state_conns.extend(sc.iter().map(|&x| x as i32));
			out_conns.extend(oc.iter().map(|&x| x as i32));
			let s_base = s_keys.len() as u32;
			s_keys.extend_from_slice(&sexp.keys);
			s_vals.extend_from_slice(&sexp.values);
			for n in 0..n_state { s_off.push(s_base + sexp.offsets[n]); s_cnt.push(sexp.counts[n]); }
		}

		// Output marker table: per (genome, neuron) a slot region sized for that
		// genome's step count (≤ one distinct address per step per neuron), 50% load.
		let mut slot_off: Vec<u32> = Vec::with_capacity(g * num_out);
		let mut slot_cap: Vec<u32> = Vec::with_capacity(g * num_out);
		let mut total_slots: u64 = 0;
		for gi in 0..g {
			let e0 = batch.ep_base[gi] as usize;
			let ne = batch.ep_count[gi] as usize;
			let steps_g: u64 = (e0..e0 + ne).map(|ep| batch.step_count[ep] as u64).sum();
			let cap = ((steps_g.saturating_mul(2)).max(16)).next_power_of_two() as u32;
			for _ in 0..num_out {
				slot_off.push(total_slots as u32);
				slot_cap.push(cap);
				total_slots += cap as u64;
			}
		}
		let total_slots = total_slots as usize;
		// markers init EMPTY=0 (new_buffer zeroes); keys init 0 (read only when FINAL);
		// values init 2 (EMPTY hover sentinel — see kernel HOST CONTRACT).
		let markers = vec![0u32; total_slots];
		let keys = vec![0u64; total_slots];
		let values = vec![2u32; total_slots];

		let p = TrainParams {
			num_genomes: g as u32, n_state: n_state as u32, sbpn: sbpn as u32, obpn: obpn as u32,
			num_motors: num_motors as u32, levels: levels as u32, bpf: bpf as u32, window: window as u32,
			frame_bits: frame_bits as u32, sensor_total: sensor_total as u32, num_features: num_features as u32,
			obs_tilt_p: obs_tilt_p as u32, obs_tilt_i: obs_tilt_i as u32,
			obs_peraxis_p: obs_peraxis_p as u32, obs_peraxis_i: obs_peraxis_i as u32, obs_pwm: obs_pwm as u32,
			integral_leak, integral_scale,
			decouple_outputs: decouple_outputs as u32, delta_control: if delta_control { 1 } else { 0 },
			selective: if batch.selective { 1 } else { 0 },
			target0: batch.target_rpy[0], target1: batch.target_rpy[1], target2: batch.target_rpy[2],
		};

		let b_sc = self.buf(&state_conns);
		let b_oc = self.buf(&out_conns);
		let b_sk = self.buf(&s_keys);
		let b_sv = self.buf(&s_vals);
		let b_so = self.buf(&s_off);
		let b_scn = self.buf(&s_cnt);
		let b_th = self.buf(controllers[0].thresholds_ref());
		let b_epb = self.buf(batch.ep_base);
		let b_epc = self.buf(batch.ep_count);
		let b_stb = self.buf(batch.step_base);
		let b_stc = self.buf(batch.step_count);
		let b_gy = self.buf(batch.gyros);
		let b_ac = self.buf(batch.accels);
		let b_tg = self.buf(batch.targets);
		let b_pp = self.buf(batch.pid_pwms);
		let b_mk = self.buf(&markers);
		let b_ky = self.buf(&keys);
		let b_vl = self.buf(&values);
		let b_soff = self.buf(&slot_off);
		let b_scap = self.buf(&slot_cap);
		let b_par = self.device.new_buffer_with_data(
			&p as *const _ as *const _, mem::size_of::<TrainParams>() as u64,
			MTLResourceOptions::StorageModeShared);
		let writes = vec![0u32; g];
		let b_wr = self.buf(&writes);

		let cmd = self.queue.new_command_buffer();
		let enc = cmd.new_compute_command_encoder();
		enc.set_compute_pipeline_state(&self.pipeline);
		let bufs: [&Buffer; 22] = [
			&b_sc, &b_oc, &b_sk, &b_sv, &b_so, &b_scn, &b_th,
			&b_epb, &b_epc, &b_stb, &b_stc, &b_gy, &b_ac, &b_tg, &b_pp,
			&b_mk, &b_ky, &b_vl, &b_soff, &b_scap, &b_par, &b_wr,
		];
		for (i, b) in bufs.iter().enumerate() { enc.set_buffer(i as u64, Some(b), 0); }
		let tw = self.pipeline.max_total_threads_per_threadgroup().min(g as u64).max(1);
		enc.dispatch_threads(MTLSize::new(g as u64, 1, 1), MTLSize::new(tw, 1, 1));
		enc.end_encoding();
		cmd.commit();
		cmd.wait_until_completed();

		// Read back the marker table → sorted (addr, cell) per (genome, neuron).
		let mk = unsafe { std::slice::from_raw_parts(b_mk.contents() as *const u32, total_slots) };
		let ky = unsafe { std::slice::from_raw_parts(b_ky.contents() as *const u64, total_slots) };
		let vl = unsafe { std::slice::from_raw_parts(b_vl.contents() as *const u32, total_slots) };
		let mut out: Vec<Vec<(u64, u8)>> = Vec::with_capacity(g * num_out);
		for gn in 0..g * num_out {
			let off = slot_off[gn] as usize;
			let cap = slot_cap[gn] as usize;
			let mut entries: Vec<(u64, u8)> = Vec::new();
			for s in off..off + cap {
				if mk[s] == MARKER_FINAL_U32 { entries.push((ky[s], (vl[s] & 0xFF) as u8)); }
			}
			entries.sort_by_key(|&(k, _)| k);
			out.push(entries);
		}
		Ok(out)
	}

	/// P2: GPU split_record. Returns, per global step (in step_base order), the
	/// (out_ins, state_ins, pid_pwm) record the conflict scan + separator consume.
	pub fn record(
		&self,
		controllers: &[&WnnController],
		batch: &TrainBatch,
	) -> Result<Vec<(Vec<bool>, Vec<bool>, [f32; 4])>, String> {
		let g = controllers.len();
		if g == 0 { return Ok(vec![]); }
		let (num_motors, levels, n_state, sbpn, obpn, bpf, window) = controllers[0].gpu_dims();
		let (num_features, obs_tilt_p, obs_tilt_i, obs_peraxis_p, obs_peraxis_i,
		     obs_pwm, integral_leak, integral_scale, decouple_outputs) = controllers[0].obs_params();
		let (delta_control, _dmax, _dleak) = controllers[0].delta_params();
		let _ = (num_motors, levels, obpn);
		let frame_bits = num_features * bpf;
		let sensor_total = window * frame_bits;
		let out_input_len = frame_bits + n_state;
		let state_input_len = sensor_total + n_state;
		let out_words = (out_input_len + 31) / 32;
		let state_words = (state_input_len + 31) / 32;
		let total_steps: usize = batch.step_count.iter().map(|&s| s as usize).sum();

		// Frozen state memory + connections (rebased), as in train().
		let mut state_conns: Vec<i32> = Vec::new();
		let mut s_keys: Vec<u64> = Vec::new();
		let mut s_vals: Vec<u8> = Vec::new();
		let mut s_off: Vec<u32> = Vec::new();
		let mut s_cnt: Vec<u32> = Vec::new();
		for c in controllers {
			let (sc, _oc, sexp, _oexp) = c.gpu_export();
			state_conns.extend(sc.iter().map(|&x| x as i32));
			let s_base = s_keys.len() as u32;
			s_keys.extend_from_slice(&sexp.keys);
			s_vals.extend_from_slice(&sexp.values);
			for n in 0..n_state { s_off.push(s_base + sexp.offsets[n]); s_cnt.push(sexp.counts[n]); }
		}

		let p = TrainParams {
			num_genomes: g as u32, n_state: n_state as u32, sbpn: sbpn as u32, obpn: obpn as u32,
			num_motors: num_motors as u32, levels: levels as u32, bpf: bpf as u32, window: window as u32,
			frame_bits: frame_bits as u32, sensor_total: sensor_total as u32, num_features: num_features as u32,
			obs_tilt_p: obs_tilt_p as u32, obs_tilt_i: obs_tilt_i as u32,
			obs_peraxis_p: obs_peraxis_p as u32, obs_peraxis_i: obs_peraxis_i as u32, obs_pwm: obs_pwm as u32,
			integral_leak, integral_scale,
			decouple_outputs: decouple_outputs as u32, delta_control: if delta_control { 1 } else { 0 },
			selective: 0, target0: batch.target_rpy[0], target1: batch.target_rpy[1], target2: batch.target_rpy[2],
		};

		let rec_out = vec![0u32; total_steps * out_words];
		let rec_state = vec![0u32; total_steps * state_words];
		let rec_pwm = vec![0f32; total_steps * 4];

		let b_sc = self.buf(&state_conns);
		let b_sk = self.buf(&s_keys); let b_sv = self.buf(&s_vals);
		let b_so = self.buf(&s_off); let b_scn = self.buf(&s_cnt);
		let b_th = self.buf(controllers[0].thresholds_ref());
		let b_epb = self.buf(batch.ep_base); let b_epc = self.buf(batch.ep_count);
		let b_stb = self.buf(batch.step_base); let b_stc = self.buf(batch.step_count);
		let b_gy = self.buf(batch.gyros); let b_ac = self.buf(batch.accels);
		let b_tg = self.buf(batch.targets); let b_pp = self.buf(batch.pid_pwms);
		let b_ro = self.buf(&rec_out); let b_rs = self.buf(&rec_state); let b_rp = self.buf(&rec_pwm);
		let b_par = self.device.new_buffer_with_data(
			&p as *const _ as *const _, mem::size_of::<TrainParams>() as u64,
			MTLResourceOptions::StorageModeShared);

		let cmd = self.queue.new_command_buffer();
		let enc = cmd.new_compute_command_encoder();
		enc.set_compute_pipeline_state(&self.record_pipeline);
		let bufs: [&Buffer; 18] = [
			&b_sc, &b_sk, &b_sv, &b_so, &b_scn, &b_th,
			&b_epb, &b_epc, &b_stb, &b_stc, &b_gy, &b_ac, &b_tg, &b_pp,
			&b_ro, &b_rs, &b_rp, &b_par,
		];
		for (i, b) in bufs.iter().enumerate() { enc.set_buffer(i as u64, Some(b), 0); }
		let max_ep = batch.ep_count.iter().copied().max().unwrap_or(0) as u64;
		let tw = 8u64.min(g as u64).max(1);
		let th = 8u64.min(max_ep).max(1);
		enc.dispatch_threads(MTLSize::new(g as u64, max_ep, 1), MTLSize::new(tw, th, 1));
		enc.end_encoding();
		cmd.commit();
		cmd.wait_until_completed();

		let ro = unsafe { std::slice::from_raw_parts(b_ro.contents() as *const u32, total_steps * out_words) };
		let rs = unsafe { std::slice::from_raw_parts(b_rs.contents() as *const u32, total_steps * state_words) };
		let rp = unsafe { std::slice::from_raw_parts(b_rp.contents() as *const f32, total_steps * 4) };
		let unpack = |buf: &[u32], base_w: usize, len: usize| -> Vec<bool> {
			(0..len).map(|pos| (buf[base_w + (pos >> 5)] >> (pos & 31)) & 1 != 0).collect()
		};
		let mut out = Vec::with_capacity(total_steps);
		for r in 0..total_steps {
			out.push((
				unpack(ro, r * out_words, out_input_len),
				unpack(rs, r * state_words, state_input_len),
				[rp[r*4], rp[r*4+1], rp[r*4+2], rp[r*4+3]],
			));
		}
		Ok(out)
	}

	/// P2b: GPU scan_conflicts_coarse. The GPU computes each record's COARSE
	/// `coarse_key` and hash-claims a bucket slot for it (controller_scan kernel);
	/// the host then groups records by slot (ascending index = bit-exact instance
	/// order), computes PWM spread, filters >tau, and runs the adaptive-k loop —
	/// the bit-exact twin of controller_split::scan_conflicts_coarse. Returns
	/// (conflicts sorted by descending spread, chosen_k). `out_ins` are the
	/// per-record output-layer input vectors (length frame_bits + n_state).
	#[allow(clippy::too_many_arguments)]
	pub fn scan(
		&self,
		out_ins: &[Vec<bool>],
		pwms: &[[f32; 4]],
		tau: f32,
		bpf: usize,
		num_features: usize,
		frame_bits: usize,
		target_min: usize,
	) -> Result<(Vec<ScanConflict>, usize), String> {
		let n = out_ins.len();
		if bpf == 0 || n == 0 {
			return Ok((Vec::new(), bpf));
		}
		let out_input_len = out_ins[0].len();
		let n_state = out_input_len - frame_bits;
		let out_words = (out_input_len + 31) / 32;

		// Pack out_ins → words (matches the controller_record packing convention).
		let mut packed = vec![0u32; n * out_words];
		for (i, oi) in out_ins.iter().enumerate() {
			let base = i * out_words;
			for (pos, &b) in oi.iter().enumerate() {
				if b { packed[base + (pos >> 5)] |= 1u32 << (pos & 31); }
			}
		}
		let b_recs = self.buf(&packed);

		// Adaptive coarseness: largest k whose conflict count reaches target_min.
		for k in (1..=bpf).rev() {
			let key_bits = num_features * k + n_state;
			let key_words = key_bits.div_ceil(64).max(1);
			let cap = (n.saturating_mul(2)).max(16).next_power_of_two();

			let markers = vec![0u32; cap];               // MARKER_EMPTY
			let keys = vec![0u64; cap * key_words];
			let slot_of = vec![0u32; n];
			let b_mk = self.buf(&markers);
			let b_ky = self.buf(&keys);
			let b_so = self.buf(&slot_of);
			let p = ScanParams {
				num_records: n as u32, out_input_len: out_input_len as u32, out_words: out_words as u32,
				frame_bits: frame_bits as u32, bpf: bpf as u32, num_features: num_features as u32,
				k: k as u32, key_words: key_words as u32, slot_capacity: cap as u32,
			};
			let b_par = self.device.new_buffer_with_data(
				&p as *const _ as *const _, mem::size_of::<ScanParams>() as u64,
				MTLResourceOptions::StorageModeShared);

			let cmd = self.queue.new_command_buffer();
			let enc = cmd.new_compute_command_encoder();
			enc.set_compute_pipeline_state(&self.scan_pipeline);
			let bufs: [&Buffer; 5] = [&b_recs, &b_mk, &b_ky, &b_so, &b_par];
			for (i, b) in bufs.iter().enumerate() { enc.set_buffer(i as u64, Some(b), 0); }
			let tw = self.scan_pipeline.max_total_threads_per_threadgroup().min(n as u64).max(1);
			enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(tw, 1, 1));
			enc.end_encoding();
			cmd.commit();
			cmd.wait_until_completed();

			let so = unsafe { std::slice::from_raw_parts(b_so.contents() as *const u32, n) };
			if so.iter().any(|&s| s == 0xFFFF_FFFF) {
				return Err(format!("controller_scan: hash table full at k={k} (cap={cap})"));
			}

			// Group by slot — ascending i preserves CPU insertion order within a
			// bucket; slot↔distinct-key is a bijection (full-key compare), so this
			// reproduces the CPU HashMap<coarse_key> grouping exactly.
			let mut buckets: std::collections::HashMap<u32, Vec<usize>> = std::collections::HashMap::new();
			for (i, &s) in so.iter().enumerate() { buckets.entry(s).or_default().push(i); }
			let mut conflicts: Vec<ScanConflict> = buckets
				.into_iter()
				.filter(|(_, idxs)| idxs.len() >= 2)
				.filter_map(|(_, idxs)| {
					let spread = crate::controller_split::pwm_spread(&idxs, pwms);
					(spread > tau).then(|| ScanConflict {
						out_in: crate::controller_split::coarse_key(
							&out_ins[idxs[0]], k, bpf, num_features, frame_bits),
						instances: idxs, spread,
					})
				})
				.collect();
			conflicts.sort_by(|a, b| b.spread.partial_cmp(&a.spread).unwrap_or(std::cmp::Ordering::Equal));
			if conflicts.len() >= target_min || k == 1 {
				return Ok((conflicts, k));
			}
		}
		Ok((Vec::new(), 1))
	}

	/// P3 (Type-1): GPU discriminative_walk, batched over conflicts. For each
	/// conflict the GPU scores every (candidate bit, lag) over the HIGH/LOW-labelled
	/// instances; the host reduces over bits in candidate order with the CPU's
	/// `gain>best || (gain==best && lag<best.lag)` rule → the exact global Separator
	/// (or None). `conflicts`/`labels` are per-conflict (instances already sampled,
	/// labels from label_high_low); `max_lags` is per-conflict. Bit-exact twin of
	/// controller_split::discriminative_walk.
	#[allow(clippy::too_many_arguments)]
	pub fn sep_walk(
		&self,
		conflicts: &[Vec<usize>],
		labels: &[Vec<bool>],
		ep_of: &[usize],
		step_of: &[usize],
		ep_start: &[usize],
		state_ins_flat: &[bool],
		sil: usize,
		candidate_bits: &[usize],
		max_lags: &[usize],
	) -> Result<Vec<Option<crate::controller_split::Separator>>, String> {
		let c = conflicts.len();
		let b = candidate_bits.len();
		if c == 0 || b == 0 {
			return Ok((0..c).map(|_| None).collect());
		}

		// Flatten per-conflict instances + labels (aligned).
		let mut conf_inst_base = Vec::with_capacity(c);
		let mut conf_inst_count = Vec::with_capacity(c);
		let mut conf_inst: Vec<u32> = Vec::new();
		let mut labels_flat: Vec<u8> = Vec::new();
		for (insts, labs) in conflicts.iter().zip(labels.iter()) {
			conf_inst_base.push(conf_inst.len() as u32);
			conf_inst_count.push(insts.len() as u32);
			conf_inst.extend(insts.iter().map(|&x| x as u32));
			labels_flat.extend(labs.iter().map(|&x| x as u8));
		}
		let ep_of_u: Vec<u32> = ep_of.iter().map(|&x| x as u32).collect();
		let step_of_u: Vec<u32> = step_of.iter().map(|&x| x as u32).collect();
		let ep_start_u: Vec<u32> = ep_start.iter().map(|&x| x as u32).collect();
		let state_u: Vec<u8> = state_ins_flat.iter().map(|&x| x as u8).collect();
		let cb_u: Vec<u32> = candidate_bits.iter().map(|&x| x as u32).collect();
		let maxlag_u: Vec<u32> = max_lags.iter().map(|&x| x as u32).collect();

		let out_correct = vec![0xFFFF_FFFFu32; c * b];
		let out_lag = vec![0u32; c * b];
		let out_high = vec![0u32; c * b];

		let b_base = self.buf(&conf_inst_base);
		let b_cnt = self.buf(&conf_inst_count);
		let b_inst = self.buf(&conf_inst);
		let b_lab = self.buf(&labels_flat);
		let b_epof = self.buf(&ep_of_u);
		let b_stof = self.buf(&step_of_u);
		let b_epst = self.buf(&ep_start_u);
		let b_st = self.buf(&state_u);
		let b_cb = self.buf(&cb_u);
		let b_ml = self.buf(&maxlag_u);
		let b_og = self.buf(&out_correct);
		let b_ol = self.buf(&out_lag);
		let b_oh = self.buf(&out_high);
		let p = SepParams { num_conflicts: c as u32, num_bits: b as u32, sil: sil as u32 };
		let b_par = self.device.new_buffer_with_data(
			&p as *const _ as *const _, mem::size_of::<SepParams>() as u64,
			MTLResourceOptions::StorageModeShared);

		let cmd = self.queue.new_command_buffer();
		let enc = cmd.new_compute_command_encoder();
		enc.set_compute_pipeline_state(&self.sep_walk_pipeline);
		let bufs: [&Buffer; 14] = [
			&b_base, &b_cnt, &b_inst, &b_lab, &b_epof, &b_stof, &b_epst, &b_st,
			&b_cb, &b_ml, &b_og, &b_ol, &b_oh, &b_par,
		];
		for (i, bf) in bufs.iter().enumerate() { enc.set_buffer(i as u64, Some(bf), 0); }
		let tw = 8u64.min(c as u64).max(1);
		let th = 8u64.min(b as u64).max(1);
		enc.dispatch_threads(MTLSize::new(c as u64, b as u64, 1), MTLSize::new(tw, th, 1));
		enc.end_encoding();
		cmd.commit();
		cmd.wait_until_completed();

		let oc = unsafe { std::slice::from_raw_parts(b_og.contents() as *const u32, c * b) };
		let ol = unsafe { std::slice::from_raw_parts(b_ol.contents() as *const u32, c * b) };
		let oh = unsafe { std::slice::from_raw_parts(b_oh.contents() as *const u32, c * b) };

		// Compute gain on the host (exact CPU separation_score) and reduce over bits
		// in candidate order — reproduces the CPU global winner bit-for-bit.
		let mut out: Vec<Option<crate::controller_split::Separator>> = Vec::with_capacity(c);
		for ci in 0..c {
			let n = conf_inst_count[ci] as f32;
			let mut best: Option<crate::controller_split::Separator> = None;
			for bi in 0..b {
				let correct = oc[ci * b + bi];
				if correct == 0xFFFF_FFFF { continue; }   // no valid lag
				let purity = correct as f32 / n;
				let g = (2.0 * (purity - 0.5)).clamp(0.0, 1.0);
				if g <= 0.0 { continue; }                 // CPU `gain > 0.0` guard
				let lag = ol[ci * b + bi] as usize;
				let high_on = oh[ci * b + bi] != 0;
				let better = match &best {
					None => true,
					Some(s) => g > s.gain || (g == s.gain && lag < s.lag),
				};
				if better {
					best = Some(crate::controller_split::Separator {
						bit: candidate_bits[bi], lag, gain: g, high_on });
				}
			}
			out.push(best);
		}
		Ok(out)
	}

	/// P3 (Type-2): GPU window-counts + host pearson for both the increment
	/// (detect_accumulator) and bidirectional (detect_accumulator_bidir) searches.
	/// The GPU computes per-(conflict, bit) per-instance window counts (integer,
	/// exact); the host runs the EXACT CPU pearson + argmax on them, so the
	/// correlation path is bit-exact (no Metal fast-math div/sqrt). `scalars` is the
	/// per-conflict disagreeing-motor PWM per instance (host best_m). Returns, per
	/// conflict, (best Accumulator, best BidirAccumulator) — the latter None unless
	/// `do_bidir`. Bit-exact twin of detect_accumulator / detect_accumulator_bidir.
	#[allow(clippy::too_many_arguments)]
	pub fn accumulator_search(
		&self,
		conflicts: &[Vec<usize>],
		scalars: &[Vec<f32>],
		ep_of: &[usize],
		step_of: &[usize],
		ep_start: &[usize],
		state_ins_flat: &[bool],
		sil: usize,
		candidate_bits: &[usize],
		max_lags: &[usize],
		do_bidir: bool,
	) -> Result<Vec<(Option<crate::controller_split::Accumulator>, Option<crate::controller_split::BidirAccumulator>)>, String> {
		let c = conflicts.len();
		let b = candidate_bits.len();
		if c == 0 || b == 0 {
			return Ok((0..c).map(|_| (None, None)).collect());
		}

		// Flatten instances + per-conflict count-block offsets (block_base = prefix
		// sum of B*n_c). counts[block_base[c] + bi*n_c + j].
		let mut conf_inst_base = Vec::with_capacity(c);
		let mut conf_inst_count = Vec::with_capacity(c);
		let mut conf_inst: Vec<u32> = Vec::new();
		let mut block_base: Vec<u32> = Vec::with_capacity(c);
		let mut total_counts: usize = 0;
		for insts in conflicts.iter() {
			conf_inst_base.push(conf_inst.len() as u32);
			conf_inst_count.push(insts.len() as u32);
			conf_inst.extend(insts.iter().map(|&x| x as u32));
			block_base.push(total_counts as u32);
			total_counts += b * insts.len();
		}
		let ep_of_u: Vec<u32> = ep_of.iter().map(|&x| x as u32).collect();
		let step_of_u: Vec<u32> = step_of.iter().map(|&x| x as u32).collect();
		let ep_start_u: Vec<u32> = ep_start.iter().map(|&x| x as u32).collect();
		let state_u: Vec<u8> = state_ins_flat.iter().map(|&x| x as u8).collect();
		let cb_u: Vec<u32> = candidate_bits.iter().map(|&x| x as u32).collect();
		let maxlag_u: Vec<u32> = max_lags.iter().map(|&x| x as u32).collect();
		let counts = vec![0.0f32; total_counts.max(1)];

		let b_base = self.buf(&conf_inst_base);
		let b_cnt = self.buf(&conf_inst_count);
		let b_inst = self.buf(&conf_inst);
		let b_epof = self.buf(&ep_of_u);
		let b_stof = self.buf(&step_of_u);
		let b_epst = self.buf(&ep_start_u);
		let b_st = self.buf(&state_u);
		let b_cb = self.buf(&cb_u);
		let b_ml = self.buf(&maxlag_u);
		let b_bb = self.buf(&block_base);
		let b_co = self.buf(&counts);
		let p = CountParams { num_conflicts: c as u32, num_bits: b as u32, sil: sil as u32 };
		let b_par = self.device.new_buffer_with_data(
			&p as *const _ as *const _, mem::size_of::<CountParams>() as u64,
			MTLResourceOptions::StorageModeShared);

		let cmd = self.queue.new_command_buffer();
		let enc = cmd.new_compute_command_encoder();
		enc.set_compute_pipeline_state(&self.sep_counts_pipeline);
		let bufs: [&Buffer; 12] = [
			&b_base, &b_cnt, &b_inst, &b_epof, &b_stof, &b_epst, &b_st,
			&b_cb, &b_ml, &b_bb, &b_co, &b_par,
		];
		for (i, bf) in bufs.iter().enumerate() { enc.set_buffer(i as u64, Some(bf), 0); }
		let tw = 8u64.min(c as u64).max(1);
		let th = 8u64.min(b as u64).max(1);
		enc.dispatch_threads(MTLSize::new(c as u64, b as u64, 1), MTLSize::new(tw, th, 1));
		enc.end_encoding();
		cmd.commit();
		cmd.wait_until_completed();

		let co = unsafe { std::slice::from_raw_parts(b_co.contents() as *const f32, total_counts.max(1)) };

		// Bidir: the heavy O(B²·instances) pair-correlation runs on the GPU in the
		// strict-FP library (one thread = one (conflict, up, dn) pair → corr table
		// [C·B·B]); the host argmaxes the small table. corr is bit-exact with the CPU
		// pearson thanks to fast-math-off + contract-off in controller_sep.metal.
		let bidir_corr: Vec<f32> = if do_bidir {
			let scalar_flat: Vec<f32> = scalars.iter().flatten().copied().collect();
			let b_sc = self.buf(&scalar_flat);
			let out_corr = vec![0.0f32; c * b * b];
			let b_oc = self.buf(&out_corr);
			let bp = CountParams { num_conflicts: c as u32, num_bits: b as u32, sil: 0 };
			let b_bp = self.device.new_buffer_with_data(
				&bp as *const _ as *const _, mem::size_of::<CountParams>() as u64,
				MTLResourceOptions::StorageModeShared);
			let cmd2 = self.queue.new_command_buffer();
			let enc2 = cmd2.new_compute_command_encoder();
			enc2.set_compute_pipeline_state(&self.sep_bidir_pipeline);
			let bufs2: [&Buffer; 7] = [&b_cnt, &b_base, &b_bb, &b_co, &b_sc, &b_oc, &b_bp];
			for (i, bf) in bufs2.iter().enumerate() { enc2.set_buffer(i as u64, Some(bf), 0); }
			let g = MTLSize::new(c as u64, b as u64, b as u64);
			let tg = MTLSize::new(4u64.min(c as u64).max(1), 4u64.min(b as u64).max(1), 4u64.min(b as u64).max(1));
			enc2.dispatch_threads(g, tg);
			enc2.end_encoding();
			cmd2.commit();
			cmd2.wait_until_completed();
			unsafe { std::slice::from_raw_parts(b_oc.contents() as *const f32, c * b * b) }.to_vec()
		} else {
			Vec::new()
		};

		// increment search on host (cheap O(B·instances), exact CPU pearson); bidir
		// argmax over the GPU corr table.
		use crate::controller_split::{pearson, Accumulator, BidirAccumulator};
		let mut out = Vec::with_capacity(c);
		for ci in 0..c {
			let n = conf_inst_count[ci] as usize;
			let bb = block_base[ci] as usize;
			let scalar = &scalars[ci];
			let bit_counts = |bi: usize| -> &[f32] { &co[bb + bi * n..bb + bi * n + n] };

			let mut acc: Option<Accumulator> = None;
			for bi in 0..b {
				let corr = pearson(bit_counts(bi), scalar);
				if corr.abs() > acc.as_ref().map(|a| a.corr).unwrap_or(0.0) {
					acc = Some(Accumulator { bit: candidate_bits[bi], up: corr >= 0.0, corr: corr.abs() });
				}
			}

			let mut bid: Option<BidirAccumulator> = None;
			if do_bidir {
				for ai in 0..b {
					for bi in 0..b {
						if ai == bi { continue; }
						let corr = bidir_corr[(ci * b + ai) * b + bi];
						if corr > bid.as_ref().map(|x| x.corr).unwrap_or(0.0) {
							bid = Some(BidirAccumulator { up: candidate_bits[ai], dn: candidate_bits[bi], corr });
						}
					}
				}
			}
			out.push((acc, bid));
		}
		Ok(out)
	}

	/// Pack a flat bool state-input record stream into the controller_record word
	/// layout (pos p → word p>>5, bit p&31).
	fn pack_state_ins(state_ins_flat: &[bool], sil: usize, num_records: usize, state_words: usize) -> Vec<u32> {
		let mut packed = vec![0u32; num_records * state_words];
		for r in 0..num_records {
			let base_w = r * state_words;
			for pos in 0..sil {
				if state_ins_flat[r * sil + pos] { packed[base_w + (pos >> 5)] |= 1u32 << (pos & 31); }
			}
		}
		packed
	}

	/// Shared P4 dispatch: scan `num_records` packed state-input records through one
	/// neuron's connections and write its sparse truth table (combo_vals over the
	/// rel_pos bits) into a fresh MarkerHashTable. Returns the sorted (addr, cell)
	/// entries. Drives controller_plant_table — the common core of latch + counter.
	fn plant_table_dispatch(
		&self,
		packed: &[u32],
		conns: &[i32],
		sbpn: usize,
		state_words: usize,
		num_records: usize,
		rel_pos: &[u32],
		combo_vals: &[u8],
	) -> Vec<(u64, u8)> {
		let ncombo = combo_vals.len();
		// ≤ one distinct addr per record × ncombo combos, 50% load.
		let cap = ((num_records.saturating_mul(2 * ncombo)).max(16)).next_power_of_two();
		let markers = vec![0u32; cap];
		let keys = vec![0u64; cap];
		let values = vec![2u32; cap];   // EMPTY=2 default (unwritten reads as hover)

		let b_si = self.buf(packed);
		let b_cn = self.buf(conns);
		let b_rp = self.buf(rel_pos);
		let b_cv = self.buf(combo_vals);
		let b_mk = self.buf(&markers);
		let b_ky = self.buf(&keys);
		let b_vl = self.buf(&values);
		let p = PlantParams {
			num_records: num_records as u32, sbpn: sbpn as u32, state_words: state_words as u32,
			slot_cap: cap as u32, num_rel: rel_pos.len() as u32,
		};
		let b_par = self.device.new_buffer_with_data(
			&p as *const _ as *const _, mem::size_of::<PlantParams>() as u64,
			MTLResourceOptions::StorageModeShared);

		let cmd = self.queue.new_command_buffer();
		let enc = cmd.new_compute_command_encoder();
		enc.set_compute_pipeline_state(&self.plant_table_pipeline);
		let bufs: [&Buffer; 8] = [&b_si, &b_cn, &b_rp, &b_cv, &b_mk, &b_ky, &b_vl, &b_par];
		for (i, bf) in bufs.iter().enumerate() { enc.set_buffer(i as u64, Some(bf), 0); }
		let tw = self.plant_table_pipeline.max_total_threads_per_threadgroup().min(num_records.max(1) as u64).max(1);
		enc.dispatch_threads(MTLSize::new(num_records.max(1) as u64, 1, 1), MTLSize::new(tw, 1, 1));
		enc.end_encoding();
		cmd.commit();
		cmd.wait_until_completed();

		let mk = unsafe { std::slice::from_raw_parts(b_mk.contents() as *const u32, cap) };
		let ky = unsafe { std::slice::from_raw_parts(b_ky.contents() as *const u64, cap) };
		let vl = unsafe { std::slice::from_raw_parts(b_vl.contents() as *const u32, cap) };
		let mut entries: Vec<(u64, u8)> = Vec::new();
		for s in 0..cap {
			if mk[s] == MARKER_FINAL_U32 { entries.push((ky[s], (vl[s] & 0xFF) as u8)); }
		}
		entries.sort_by_key(|&(k, _)| k);
		entries
	}

	/// P4 (Type-1 planting): GPU split_plant_latch. Selects the latch neuron on the
	/// host (plant_latch_neuron — shared with the CPU planter), then the GPU writes
	/// the 4-combo set/hold latch truth table (combo bit0=trigger@tp, bit1=self@sp;
	/// on = sv || (tv==high_on)). Returns (planted neuron, sorted (addr, cell)) or
	/// (None, []). Bit-exact twin of WnnController::split_plant_latch.
	pub fn plant_latch(
		&self,
		controller: &WnnController,
		state_ins_flat: &[bool],
		sil: usize,
		bit: usize,
		high_on: bool,
		used: &[bool],
	) -> Result<(Option<usize>, Vec<(u64, u8)>), String> {
		let (c, tp, sp) = match controller.plant_latch_neuron(bit, used) {
			Some(t) => t,
			None => return Ok((None, Vec::new())),
		};
		let (_nm, _lv, _ns, sbpn, _ob, _bpf, _w) = controller.gpu_dims();
		let (sc, _oc, _se, _oe) = controller.gpu_export();
		let conns: Vec<i32> = sc[c * sbpn..(c + 1) * sbpn].iter().map(|&x| x as i32).collect();
		let num_records = if sil == 0 { 0 } else { state_ins_flat.len() / sil };
		let state_words = (sil + 31) / 32;
		let packed = Self::pack_state_ins(state_ins_flat, sil, num_records, state_words);

		// combo bit0=tv@tp, bit1=sv@sp; on = sv==1 || (tv==1)==high_on.
		let combo_vals: Vec<u8> = (0..4).map(|combo| {
			let (tv, sv) = (combo & 1, (combo >> 1) & 1);
			let on = sv == 1 || (tv == 1) == high_on;
			if on { 3 } else { 1 }
		}).collect();
		let entries = self.plant_table_dispatch(&packed, &conns, sbpn, state_words, num_records, &[tp as u32, sp as u32], &combo_vals);
		Ok((Some(c), entries))
	}

	/// P4 (Type-2 increment planting): GPU split_install_counter. Selects the trigger
	/// chain on the host (plant_counter_chain — shared with the CPU planter); needs
	/// ≥2 levels. Per level k the GPU writes the gated increment+hold truth table:
	/// level 0 = latch (on = self||trigger); level k>0 = on = self||(trigger&&lower).
	/// Returns (chain, per-neuron sorted (addr, cell) entries) or (None, []) if no
	/// ≥2-level chain exists or a level isn't wired (trigger/self/lower not observed).
	/// Bit-exact twin of WnnController::split_install_counter.
	#[allow(clippy::type_complexity)]
	pub fn plant_counter(
		&self,
		controller: &WnnController,
		state_ins_flat: &[bool],
		sil: usize,
		trigger: usize,
		max_levels: usize,
		used: &[bool],
	) -> Result<(Option<Vec<usize>>, Vec<Vec<(u64, u8)>>), String> {
		let chain = controller.plant_counter_chain(trigger, max_levels, used);
		if chain.len() < 2 {
			return Ok((None, Vec::new()));
		}
		let (_nm, _lv, _ns, sbpn, _ob, _bpf, _w) = controller.gpu_dims();
		let (num_features, ..) = controller.obs_params();
		let (_, _, _, _, _, bpf, window) = controller.gpu_dims();
		let sensor_window = window * num_features * bpf;
		let (sc, _oc, _se, _oe) = controller.gpu_export();
		let num_records = if sil == 0 { 0 } else { state_ins_flat.len() / sil };
		let state_words = (sil + 31) / 32;
		let packed = Self::pack_state_ins(state_ins_flat, sil, num_records, state_words);

		let pos = |conns: &[i64], target: usize| -> Option<usize> {
			conns.iter().position(|&x| x as usize == target)
		};
		let mut per_neuron: Vec<Vec<(u64, u8)>> = Vec::with_capacity(chain.len());
		for k in 0..chain.len() {
			let c = chain[k];
			let conns_i64 = &sc[c * sbpn..(c + 1) * sbpn];
			let conns: Vec<i32> = conns_i64.iter().map(|&x| x as i32).collect();
			let tp = pos(conns_i64, trigger).ok_or("counter: trigger not observed")? as u32;
			let sp = pos(conns_i64, sensor_window + c).ok_or("counter: self not observed")? as u32;
			let (rel_pos, combo_vals): (Vec<u32>, Vec<u8>) = if k == 0 {
				// level 0 = latch: on = sv || tv (combo bit0=tv@tp, bit1=sv@sp).
				let cv: Vec<u8> = (0..4).map(|combo| {
					let (tv, sv) = (combo & 1, (combo >> 1) & 1);
					if sv == 1 || tv == 1 { 3 } else { 1 }
				}).collect();
				(vec![tp, sp], cv)
			} else {
				// level k>0: on = sv || (tv && lv); combo bit0=tv@tp, bit1=lv@lp, bit2=sv@sp.
				let lp = pos(conns_i64, sensor_window + chain[k - 1]).ok_or("counter: lower not observed")? as u32;
				let cv: Vec<u8> = (0..8).map(|combo| {
					let (tv, lv, sv) = (combo & 1, (combo >> 1) & 1, (combo >> 2) & 1);
					if sv == 1 || (tv == 1 && lv == 1) { 3 } else { 1 }
				}).collect();
				(vec![tp, lp, sp], cv)
			};
			per_neuron.push(self.plant_table_dispatch(&packed, &conns, sbpn, state_words, num_records, &rel_pos, &combo_vals));
		}
		Ok((Some(chain), per_neuron))
	}

	/// P4 (Type-2 bidirectional planting): GPU split_install_counter_bidir. Verifies
	/// the up/down chain wiring on the host (plant_counter_bidir_ok — shared with the
	/// CPU planter); on success the GPU computes the DENSE 2^sbpn on-table (a pure
	/// function of the address bits up/dn/lower/self/upper), which every chain neuron
	/// 0..n_levels shares. Returns (levels, per-neuron (addr, cell)) or (None, []).
	/// Bit-exact twin of WnnController::split_install_counter_bidir.
	#[allow(clippy::type_complexity)]
	pub fn plant_counter_bidir(
		&self,
		controller: &WnnController,
		up: usize,
		dn: usize,
		n_levels: usize,
		used: &[bool],
	) -> Result<(Option<Vec<usize>>, Vec<Vec<(u64, u8)>>), String> {
		if !controller.plant_counter_bidir_ok(up, dn, n_levels, used) {
			return Ok((None, Vec::new()));
		}
		let (_nm, _lv, _ns, sbpn, _ob, _bpf, _w) = controller.gpu_dims();
		let naddr = 1usize << sbpn;
		let out_vals = vec![0u8; naddr];
		let b_ov = self.buf(&out_vals);
		let p = BidirPlantParams { sbpn: sbpn as u32 };
		let b_par = self.device.new_buffer_with_data(
			&p as *const _ as *const _, mem::size_of::<BidirPlantParams>() as u64,
			MTLResourceOptions::StorageModeShared);
		let cmd = self.queue.new_command_buffer();
		let enc = cmd.new_compute_command_encoder();
		enc.set_compute_pipeline_state(&self.plant_bidir_pipeline);
		enc.set_buffer(0, Some(&b_ov), 0);
		enc.set_buffer(1, Some(&b_par), 0);
		let tw = self.plant_bidir_pipeline.max_total_threads_per_threadgroup().min(naddr as u64).max(1);
		enc.dispatch_threads(MTLSize::new(naddr as u64, 1, 1), MTLSize::new(tw, 1, 1));
		enc.end_encoding();
		cmd.commit();
		cmd.wait_until_completed();

		let ov = unsafe { std::slice::from_raw_parts(b_ov.contents() as *const u8, naddr) };
		// Every chain neuron shares the same dense table.
		let table: Vec<(u64, u8)> = (0..naddr).map(|a| (a as u64, ov[a])).collect();
		let per_neuron: Vec<Vec<(u64, u8)>> = (0..n_levels).map(|_| table.clone()).collect();
		Ok((Some((0..n_levels).collect()), per_neuron))
	}
}

/// PyO3: self-contained bit-exact parity test for the GPU controller_train kernel
/// vs the CPU split_retrain_output. Builds a controller (seeded), plants state so
/// the selective gate is exercised, generates deterministic trajectories, runs GPU
/// THEN CPU (CPU mutates output_memory; GPU read the frozen state first), and
/// compares the cell FUNCTION over every touched address. Returns a list of
/// (name, passed, detail) tuples (mirrors run_marker_train_parity_test).
#[pyfunction]
pub fn run_controller_train_parity_test() -> Vec<(String, bool, String)> {
	let mut results = Vec::new();
	if Device::system_default().is_none() {
		results.push(("controller_train_parity".to_string(), true, "skipped: no Metal device".to_string()));
		return results;
	}
	for &selective in &[false, true] {
		match controller_train_parity_once(selective) {
			Ok((mismatches, cpu_cells, addrs)) => {
				let name = format!("controller_train_parity(selective={selective})");
				results.push((name, mismatches == 0,
					format!("addresses={addrs}, cpu_nonempty_cells={cpu_cells}, mismatches={mismatches}")));
			}
			Err(e) => results.push((format!("controller_train_parity(selective={selective})"), false, e)),
		}
	}
	results
}

// Deterministic xorshift for the self-contained fixture (no Math.random in tests).
fn xs(state: &mut u64) -> u64 { let mut x = *state; x ^= x << 13; x ^= x >> 7; x ^= x << 17; *state = x; x }
fn xf(state: &mut u64) -> f32 { (xs(state) >> 40) as f32 / (1u64 << 24) as f32 } // [0,1)

/// Shared deterministic fixture for the GPU-training parity tests (P1, P2, …):
/// a seeded controller (absolute + decouple, the bug-prone config) with planted
/// state, and trajectories in BOTH the CPU nested form and the GPU flat form from
/// the same RNG. num_out + dims derive from the controller.
struct ParityFixture {
	c: WnnController,
	num_out: usize,
	cpu_g: Vec<Vec<[f32; 3]>>, cpu_a: Vec<Vec<[f32; 3]>>, cpu_t: Vec<Vec<[f32; 3]>>, cpu_p: Vec<Vec<[f32; 4]>>,
	gyros: Vec<f32>, accels: Vec<f32>, targets: Vec<f32>, pids: Vec<f32>,
	ep_base: Vec<u32>, ep_count: Vec<u32>, step_base: Vec<u32>, step_count: Vec<u32>,
}

fn build_parity_fixture(seed_salt: u64) -> Result<ParityFixture, String> {
	let (num_motors, levels, n_state, sbpn, obpn, bpf, window) = (4usize, 8usize, 8usize, 12usize, 12usize, 4usize, 4usize);
	let num_features = 9usize; // no H2 extras
	let frame_bits = num_features * bpf;
	let sensor_total = window * frame_bits;
	let total_state_in = sensor_total + n_state;
	let num_out = num_motors * levels;
	let total_out_in = frame_bits + n_state;

	let mut rng = 0x9E3779B97F4A7C15u64 ^ seed_salt.wrapping_mul(0xD1B54A32D192ED03);
	let thresholds: Vec<f32> = (0..num_features * bpf).map(|_| xf(&mut rng) - 0.5).collect();
	let state_conns: Vec<i64> = (0..n_state * sbpn).map(|_| (xs(&mut rng) % total_state_in as u64) as i64).collect();
	let output_conns: Vec<i64> = (0..num_out * obpn).map(|_| (xs(&mut rng) % total_out_in as u64) as i64).collect();

	let c = WnnController::new(
		num_motors, levels, bpf, window, n_state, sbpn, obpn,
		thresholds, state_conns, output_conns,
		false, 0.1, 0.95,
		false, false, false, false, false, 0.99, 1.0,
		true,
	).map_err(|e| format!("{e}"))?;
	for _ in 0..(n_state * 4) {
		let n = (xs(&mut rng) % n_state as u64) as usize;
		let addr = xs(&mut rng) % (1u64 << sbpn);
		c.plant_state_cell(n, addr, 3u8);
	}

	let (e_count, t_steps) = (3usize, 40usize);
	let (mut gyros, mut accels, mut targets, mut pids) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
	let (mut step_base, mut step_count) = (Vec::new(), Vec::new());
	let (mut cpu_g, mut cpu_a, mut cpu_t, mut cpu_p) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
	let mut sbase = 0u32;
	for _ in 0..e_count {
		step_base.push(sbase); step_count.push(t_steps as u32); sbase += t_steps as u32;
		let (mut eg, mut ea, mut et, mut ep): (Vec<[f32;3]>, Vec<[f32;3]>, Vec<[f32;3]>, Vec<[f32;4]>) =
			(Vec::new(), Vec::new(), Vec::new(), Vec::new());
		for _ in 0..t_steps {
			let gy = [xf(&mut rng)-0.5, xf(&mut rng)-0.5, xf(&mut rng)-0.5];
			let ac = [xf(&mut rng)-0.5, xf(&mut rng)-0.5, xf(&mut rng)+0.5];
			let tg = [0.0f32, 0.0, 0.0];
			let pw = [xf(&mut rng), xf(&mut rng)*2.0-1.0, xf(&mut rng)*2.0-1.0, xf(&mut rng)*2.0-1.0];
			gyros.extend_from_slice(&gy); accels.extend_from_slice(&ac); targets.extend_from_slice(&tg); pids.extend_from_slice(&pw);
			eg.push(gy); ea.push(ac); et.push(tg); ep.push(pw);
		}
		cpu_g.push(eg); cpu_a.push(ea); cpu_t.push(et); cpu_p.push(ep);
	}
	Ok(ParityFixture {
		c, num_out, cpu_g, cpu_a, cpu_t, cpu_p, gyros, accels, targets, pids,
		ep_base: vec![0u32], ep_count: vec![e_count as u32], step_base, step_count,
	})
}

fn controller_train_parity_once(selective: bool) -> Result<(usize, usize, usize), String> {
	let f = build_parity_fixture(selective as u64)?;
	let num_out = f.num_out;
	let mut c = f.c;

	// GPU FIRST (reads frozen state + empty output table).
	let trainer = ControllerTrainer::new()?;
	let batch = TrainBatch {
		ep_base: &f.ep_base, ep_count: &f.ep_count, step_base: &f.step_base, step_count: &f.step_count,
		gyros: &f.gyros, accels: &f.accels, targets: &f.targets, pid_pwms: &f.pids,
		selective, target_rpy: [0.0, 0.0, 0.0],
	};
	let gpu = trainer.train(&[&c], &batch)?;

	// CPU reference (mutates c.output_memory).
	let _writes = c.split_retrain_output_pub(&f.cpu_g, &f.cpu_a, &f.cpu_t, &f.cpu_p, selective);

	// Compare the cell FUNCTION over the union of touched addresses per neuron.
	let mut mismatches = 0usize;
	let mut cpu_cells = 0usize;
	let mut addrs = 0usize;
	for n in 0..num_out {
		let gpu_entries = &gpu[n];
		let cpu_entries = c.output_entries(n);
		cpu_cells += cpu_entries.iter().filter(|&&(_, v)| v != 2).count();
		// Union of addresses either side touched.
		let mut all: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
		for &(a, _) in gpu_entries { all.insert(a); }
		for &(a, _) in &cpu_entries { all.insert(a); }
		addrs += all.len();
		let gpu_map: std::collections::HashMap<u64, u8> = gpu_entries.iter().copied().collect();
		for a in all {
			let gv = *gpu_map.get(&a).unwrap_or(&2u8);   // GPU miss → EMPTY=2
			let cv = c.output_cell(n, a);                 // CPU read_cell (miss → EMPTY=2)
			if gv != cv { mismatches += 1; }
		}
	}
	Ok((mismatches, cpu_cells, addrs))
}

/// PyO3: bit-exact parity for the GPU controller_record kernel (P2) vs CPU
/// split_record — compares out_ins, state_ins, and pwm per record.
#[pyfunction]
pub fn run_controller_record_parity_test() -> Vec<(String, bool, String)> {
	let mut results = Vec::new();
	if Device::system_default().is_none() {
		results.push(("controller_record_parity".to_string(), true, "skipped: no Metal device".to_string()));
		return results;
	}
	match controller_record_parity_once() {
		Ok((records, mism_out, mism_state, mism_pwm)) => {
			let ok = mism_out == 0 && mism_state == 0 && mism_pwm == 0;
			results.push(("controller_record_parity".to_string(), ok, format!(
				"records={records}, out_ins_mismatch={mism_out}, state_ins_mismatch={mism_state}, pwm_mismatch={mism_pwm}")));
		}
		Err(e) => results.push(("controller_record_parity".to_string(), false, e)),
	}
	results
}

fn controller_record_parity_once() -> Result<(usize, usize, usize, usize), String> {
	let f = build_parity_fixture(0x5EC0_0DE5u64)?;
	let mut c = f.c;

	// GPU records (reads frozen state).
	let trainer = ControllerTrainer::new()?;
	let batch = TrainBatch {
		ep_base: &f.ep_base, ep_count: &f.ep_count, step_base: &f.step_base, step_count: &f.step_count,
		gyros: &f.gyros, accels: &f.accels, targets: &f.targets, pid_pwms: &f.pids,
		selective: false, target_rpy: [0.0, 0.0, 0.0],
	};
	let gpu = trainer.record(&[&c], &batch)?;

	// CPU reference.
	let (cpu_out_ins, cpu_pwms, cpu_state_flat, state_len) =
		c.split_record_pub(f.cpu_g.clone(), f.cpu_a.clone(), f.cpu_t.clone(), f.cpu_p.clone());

	let records = gpu.len();
	if records != cpu_out_ins.len() {
		return Err(format!("record count mismatch: gpu={records} cpu={}", cpu_out_ins.len()));
	}
	let (mut mism_out, mut mism_state, mut mism_pwm) = (0usize, 0usize, 0usize);
	for r in 0..records {
		let (g_out, g_state, g_pwm) = &gpu[r];
		if *g_out != cpu_out_ins[r] { mism_out += 1; }
		let cpu_state = &cpu_state_flat[r * state_len .. (r + 1) * state_len];
		if g_state.as_slice() != cpu_state { mism_state += 1; }
		for m in 0..4 { if (g_pwm[m] - cpu_pwms[r][m]).abs() > 1e-6 { mism_pwm += 1; break; } }
	}
	Ok((records, mism_out, mism_state, mism_pwm))
}

/// PyO3: bit-exact parity for the GPU controller_scan (P2b) vs CPU
/// scan_conflicts_coarse. Two cases: (A) single-word key with adaptive
/// coarsening, (B) multi-word key (out_input_len > 64). Compares chosen_k and the
/// SET of conflicts (keyed by ascending-instance tuple, with matching spread) —
/// set comparison is required because both sides sort by descending spread and
/// inherit HashMap tie order, so a strict sequence compare would be flaky.
#[pyfunction]
pub fn run_controller_scan_parity_test() -> Vec<(String, bool, String)> {
	let mut results = Vec::new();
	if Device::system_default().is_none() {
		results.push(("controller_scan_parity".to_string(), true, "skipped: no Metal device".to_string()));
		return results;
	}
	for (case, label) in [(0u32, "single_word"), (1u32, "multi_word")] {
		match controller_scan_parity_once(case) {
			Ok((cpu_k, gpu_k, n_conf, mism)) => {
				let ok = cpu_k == gpu_k && mism == 0;
				results.push((format!("controller_scan_parity({label})"), ok, format!(
					"cpu_k={cpu_k}, gpu_k={gpu_k}, conflicts={n_conf}, mismatches={mism}")));
			}
			Err(e) => results.push((format!("controller_scan_parity({label})"), false, e)),
		}
	}
	results
}

/// Deterministic synthetic record fixture for the scan parity test. Returns
/// (out_ins, pwms, bpf, num_features, frame_bits, tau, target_min). Case 0:
/// random records, single-word key, forces the adaptive-k coarsening loop.
/// Case 1: planted groups of identical out_in (88-bit → 2 key words) with split
/// PWM → conflicts at k=bpf, exercising the multi-word claim/grouping.
fn build_scan_fixture(case: u32) -> (Vec<Vec<bool>>, Vec<[f32; 4]>, usize, usize, usize, f32, usize) {
	let mut rng = 0xC0FFEEu64 ^ (case as u64).wrapping_mul(0x9E3779B97F4A7C15);
	if case == 0 {
		let (num_features, bpf, n_state) = (4usize, 4usize, 4usize);
		let frame_bits = num_features * bpf;          // 16
		let total = frame_bits + n_state;             // 20
		let (mut out_ins, mut pwms) = (Vec::new(), Vec::new());
		for _ in 0..300usize {
			out_ins.push((0..total).map(|_| xf(&mut rng) < 0.5).collect());
			pwms.push([xf(&mut rng), xf(&mut rng), xf(&mut rng), xf(&mut rng)]);
		}
		(out_ins, pwms, bpf, num_features, frame_bits, 0.4f32, 8usize)
	} else {
		let (num_features, bpf, n_state) = (8usize, 8usize, 24usize);
		let frame_bits = num_features * bpf;          // 64
		let total = frame_bits + n_state;             // 88 → 2 key words at k=8
		let (mut out_ins, mut pwms) = (Vec::new(), Vec::new());
		for _ in 0..12 {                              // 12 conflict groups
			let oi: Vec<bool> = (0..total).map(|_| xf(&mut rng) < 0.5).collect();
			for r in 0..4 {
				out_ins.push(oi.clone());
				let lvl = if r < 2 { 0.1f32 } else { 0.9f32 };   // motor-0 spread ≈ 0.8 > tau
				pwms.push([lvl, xf(&mut rng), xf(&mut rng), xf(&mut rng)]);
			}
		}
		for _ in 0..100 {                             // singleton noise
			out_ins.push((0..total).map(|_| xf(&mut rng) < 0.5).collect());
			pwms.push([xf(&mut rng), xf(&mut rng), xf(&mut rng), xf(&mut rng)]);
		}
		(out_ins, pwms, bpf, num_features, frame_bits, 0.5f32, 5usize)
	}
}

fn controller_scan_parity_once(case: u32) -> Result<(usize, usize, usize, usize), String> {
	let (out_ins, pwms, bpf, num_features, frame_bits, tau, target_min) = build_scan_fixture(case);

	let trainer = ControllerTrainer::new()?;
	let (gpu_conf, gpu_k) = trainer.scan(&out_ins, &pwms, tau, bpf, num_features, frame_bits, target_min)?;
	let (cpu_conf, cpu_k) = crate::controller_split::scan_conflicts_coarse(
		&out_ins, &pwms, tau, bpf, num_features, frame_bits, target_min);

	// Canonical set keyed by ascending-instance tuple → (spread bits exact f32,
	// coarse key). Including out_in validates the GPU-side coarse_key reconstruction,
	// not just the bucketing — a wrong key would still group identically here but
	// reconstruct a different bucket signature.
	let cpu_set: std::collections::HashMap<Vec<usize>, (u32, Vec<bool>)> = cpu_conf
		.iter().map(|c| (c.instances.clone(), (c.spread.to_bits(), c.out_in.clone()))).collect();
	let gpu_set: std::collections::HashMap<Vec<usize>, (u32, Vec<bool>)> = gpu_conf
		.iter().map(|c| (c.instances.clone(), (c.spread.to_bits(), c.out_in.clone()))).collect();

	let mut mism = 0usize;
	for (inst, sp) in &cpu_set {
		if gpu_set.get(inst) != Some(sp) { mism += 1; }
	}
	for (inst, sp) in &gpu_set {
		if cpu_set.get(inst) != Some(sp) { mism += 1; }
	}
	Ok((cpu_k, gpu_k, cpu_conf.len(), mism))
}

/// PyO3: bit-exact parity for the GPU controller_sep_walk (P3 Type-1) vs CPU
/// discriminative_walk. Builds synthetic conflicts over a deterministic record
/// stream and compares the best Separator per conflict (bit, lag, gain bits,
/// high_on, and None-vs-Some).
#[pyfunction]
pub fn run_controller_sep_walk_parity_test() -> Vec<(String, bool, String)> {
	let mut results = Vec::new();
	if Device::system_default().is_none() {
		results.push(("controller_sep_walk_parity".to_string(), true, "skipped: no Metal device".to_string()));
		return results;
	}
	match controller_sep_walk_parity_once() {
		Ok((n_conf, n_some, mism)) => {
			results.push(("controller_sep_walk_parity".to_string(), mism == 0, format!(
				"conflicts={n_conf}, separators_found={n_some}, mismatches={mism}")));
		}
		Err(e) => results.push(("controller_sep_walk_parity".to_string(), false, e)),
	}
	results
}

fn controller_sep_walk_parity_once() -> Result<(usize, usize, usize), String> {
	let mut rng = 0x5E9A_5EA70123u64 ^ 0x9E3779B97F4A7C15u64;
	let (e_count, t_steps, sil, num_bits, n_conf) = (6usize, 40usize, 24usize, 16usize, 20usize);
	let r = e_count * t_steps;

	// Episode-major record stream: ep_of/step_of/ep_start.
	let ep_of: Vec<usize> = (0..r).map(|rec| rec / t_steps).collect();
	let step_of: Vec<usize> = (0..r).map(|rec| rec % t_steps).collect();
	let ep_start: Vec<usize> = (0..e_count).map(|ep| ep * t_steps).collect();

	let state_ins_flat: Vec<bool> = (0..r * sil).map(|_| xf(&mut rng) < 0.5).collect();
	let pwms: Vec<[f32; 4]> = (0..r)
		.map(|_| [xf(&mut rng), xf(&mut rng) * 2.0 - 1.0, xf(&mut rng) * 2.0 - 1.0, xf(&mut rng) * 2.0 - 1.0])
		.collect();
	let candidate_bits: Vec<usize> = (0..num_bits).collect();

	// Synthetic conflicts: 4..12 instances each, steps in [5, T) so max_lag ≥ 5.
	let (mut conflicts, mut labels, mut max_lags) = (Vec::new(), Vec::new(), Vec::new());
	for _ in 0..n_conf {
		let m = 4 + (xs(&mut rng) % 9) as usize;
		let insts: Vec<usize> = (0..m)
			.map(|_| {
				let ep = (xs(&mut rng) % e_count as u64) as usize;
				let step = 5 + (xs(&mut rng) % (t_steps as u64 - 5)) as usize;
				ep * t_steps + step
			})
			.collect();
		let labs = crate::controller_split::label_high_low(&insts, &pwms);
		let ml = insts.iter().map(|&i| step_of[i]).min().unwrap_or(0).min(48);
		conflicts.push(insts);
		labels.push(labs);
		max_lags.push(ml);
	}

	// GPU.
	let trainer = ControllerTrainer::new()?;
	let gpu = trainer.sep_walk(&conflicts, &labels, &ep_of, &step_of, &ep_start,
		&state_ins_flat, sil, &candidate_bits, &max_lags)?;

	// CPU reference, per conflict.
	let mut mism = 0usize;
	let mut n_some = 0usize;
	for ci in 0..n_conf {
		let cpu = crate::controller_split::discriminative_walk(
			&conflicts[ci], &labels[ci], &ep_of, &step_of, &ep_start,
			&state_ins_flat, sil, &candidate_bits, max_lags[ci]);
		if cpu.is_some() { n_some += 1; }
		let eq = match (&cpu, &gpu[ci]) {
			(None, None) => true,
			(Some(a), Some(b)) =>
				a.bit == b.bit && a.lag == b.lag && a.gain.to_bits() == b.gain.to_bits() && a.high_on == b.high_on,
			_ => false,
		};
		if !eq { mism += 1; }
	}
	Ok((n_conf, n_some, mism))
}

/// PyO3: bit-exact parity for the GPU controller_sep_counts + host pearson (P3
/// Type-2) vs CPU detect_accumulator AND detect_accumulator_bidir. Compares the
/// best Accumulator (bit, up, corr bits) and BidirAccumulator (up, dn, corr bits)
/// per conflict, plus None-vs-Some.
#[pyfunction]
pub fn run_controller_accumulator_parity_test() -> Vec<(String, bool, String)> {
	let mut results = Vec::new();
	if Device::system_default().is_none() {
		results.push(("controller_accumulator_parity".to_string(), true, "skipped: no Metal device".to_string()));
		return results;
	}
	match controller_accumulator_parity_once() {
		Ok((n_conf, n_inc, n_bid, mism_inc, mism_bid)) => {
			let ok = mism_inc == 0 && mism_bid == 0;
			results.push(("controller_accumulator_parity".to_string(), ok, format!(
				"conflicts={n_conf}, inc_found={n_inc}, bidir_found={n_bid}, inc_mismatch={mism_inc}, bidir_mismatch={mism_bid}")));
		}
		Err(e) => results.push(("controller_accumulator_parity".to_string(), false, e)),
	}
	results
}

fn controller_accumulator_parity_once() -> Result<(usize, usize, usize, usize, usize), String> {
	let mut rng = 0xACC0_11DE_5EED_0001u64 ^ 0x9E3779B97F4A7C15u64;
	let (e_count, t_steps, sil, num_bits, n_conf) = (6usize, 40usize, 24usize, 12usize, 20usize);
	let r = e_count * t_steps;

	let ep_of: Vec<usize> = (0..r).map(|rec| rec / t_steps).collect();
	let step_of: Vec<usize> = (0..r).map(|rec| rec % t_steps).collect();
	let ep_start: Vec<usize> = (0..e_count).map(|ep| ep * t_steps).collect();
	let state_ins_flat: Vec<bool> = (0..r * sil).map(|_| xf(&mut rng) < 0.5).collect();
	let pwms: Vec<[f32; 4]> = (0..r)
		.map(|_| [xf(&mut rng), xf(&mut rng) * 2.0 - 1.0, xf(&mut rng) * 2.0 - 1.0, xf(&mut rng) * 2.0 - 1.0])
		.collect();
	let candidate_bits: Vec<usize> = (0..num_bits).collect();

	let (mut conflicts, mut scalars, mut max_lags) = (Vec::new(), Vec::new(), Vec::new());
	for _ in 0..n_conf {
		let m = 4 + (xs(&mut rng) % 9) as usize;
		let insts: Vec<usize> = (0..m)
			.map(|_| {
				let ep = (xs(&mut rng) % e_count as u64) as usize;
				let step = 5 + (xs(&mut rng) % (t_steps as u64 - 5)) as usize;
				ep * t_steps + step
			})
			.collect();
		// disagreeing motor = max-spread motor (mirrors split_resolve_conflict).
		let (mut best_m, mut best_s) = (0usize, -1.0f32);
		for mo in 0..4 {
			let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
			for &i in &insts { lo = lo.min(pwms[i][mo]); hi = hi.max(pwms[i][mo]); }
			if hi - lo > best_s { best_s = hi - lo; best_m = mo; }
		}
		let scalar: Vec<f32> = insts.iter().map(|&i| pwms[i][best_m]).collect();
		let ml = insts.iter().map(|&i| step_of[i]).min().unwrap_or(0).min(48);
		conflicts.push(insts);
		scalars.push(scalar);
		max_lags.push(ml);
	}

	let trainer = ControllerTrainer::new()?;
	let gpu = trainer.accumulator_search(&conflicts, &scalars, &ep_of, &step_of, &ep_start,
		&state_ins_flat, sil, &candidate_bits, &max_lags, true)?;

	let (mut n_inc, mut n_bid, mut mism_inc, mut mism_bid) = (0usize, 0usize, 0usize, 0usize);
	for ci in 0..n_conf {
		let cpu_acc = crate::controller_split::detect_accumulator(
			&conflicts[ci], &scalars[ci], &ep_of, &step_of, &ep_start,
			&state_ins_flat, sil, &candidate_bits, max_lags[ci]);
		let cpu_bid = crate::controller_split::detect_accumulator_bidir(
			&conflicts[ci], &scalars[ci], &ep_of, &step_of, &ep_start,
			&state_ins_flat, sil, &candidate_bits, max_lags[ci]);
		if cpu_acc.is_some() { n_inc += 1; }
		if cpu_bid.is_some() { n_bid += 1; }
		let (g_acc, g_bid) = &gpu[ci];
		let acc_eq = match (&cpu_acc, g_acc) {
			(None, None) => true,
			(Some(a), Some(b)) => a.bit == b.bit && a.up == b.up && a.corr.to_bits() == b.corr.to_bits(),
			_ => false,
		};
		let bid_eq = match (&cpu_bid, g_bid) {
			(None, None) => true,
			(Some(a), Some(b)) => a.up == b.up && a.dn == b.dn && a.corr.to_bits() == b.corr.to_bits(),
			_ => false,
		};
		if !acc_eq { mism_inc += 1; }
		if !bid_eq { mism_bid += 1; }
	}
	Ok((n_conf, n_inc, n_bid, mism_inc, mism_bid))
}

/// PyO3: bit-exact parity for the GPU controller_plant_latch (P4) vs CPU
/// split_plant_latch. Compares the planted neuron and the cell FUNCTION over every
/// touched state address, for both high_on directions.
#[pyfunction]
pub fn run_controller_plant_latch_parity_test() -> Vec<(String, bool, String)> {
	let mut results = Vec::new();
	if Device::system_default().is_none() {
		results.push(("controller_plant_latch_parity".to_string(), true, "skipped: no Metal device".to_string()));
		return results;
	}
	for high_on in [false, true] {
		match controller_plant_latch_parity_once(high_on) {
			Ok((addrs, cpu_cells, mism)) => {
				results.push((format!("controller_plant_latch_parity(high_on={high_on})"), mism == 0, format!(
					"addresses={addrs}, cpu_nonempty_cells={cpu_cells}, mismatches={mism}")));
			}
			Err(e) => results.push((format!("controller_plant_latch_parity(high_on={high_on})"), false, e)),
		}
	}
	results
}

fn controller_plant_latch_parity_once(high_on: bool) -> Result<(usize, usize, usize), String> {
	let (num_motors, levels, n_state, sbpn, obpn, bpf, window) = (4usize, 8usize, 8usize, 12usize, 12usize, 4usize, 4usize);
	let num_features = 9usize;
	let frame_bits = num_features * bpf;            // 36
	let sensor_window = window * frame_bits;        // 144
	let state_input_len = sensor_window + n_state;  // 152
	let total_out_in = frame_bits + n_state;        // 44
	let num_out = num_motors * levels;
	let trigger = 5usize;
	let self0 = sensor_window;                        // neuron 0's self bit = 144

	let mut rng = 0x9E3779B97F4A7C15u64 ^ (high_on as u64).wrapping_mul(0xD1B54A32D192ED03);
	let thresholds: Vec<f32> = (0..num_features * bpf).map(|_| xf(&mut rng) - 0.5).collect();

	// Neuron 0 observes trigger@pos0 + self@pos1 (the ONLY qualifying neuron);
	// neurons 1.. use a band [50,152) that excludes the trigger bit.
	let mut state_conns = vec![0i64; n_state * sbpn];
	state_conns[0] = trigger as i64;
	state_conns[1] = self0 as i64;
	for i in 2..sbpn { state_conns[i] = (20 + i) as i64; }
	for c in 1..n_state {
		for i in 0..sbpn {
			state_conns[c * sbpn + i] = (50 + ((c * sbpn + i) % (state_input_len - 50))) as i64;
		}
	}
	let output_conns: Vec<i64> = (0..num_out * obpn).map(|_| (xs(&mut rng) % total_out_in as u64) as i64).collect();

	let c = WnnController::new(
		num_motors, levels, bpf, window, n_state, sbpn, obpn,
		thresholds, state_conns, output_conns,
		false, 0.1, 0.95,
		false, false, false, false, false, 0.99, 1.0,
		true,
	).map_err(|e| format!("{e}"))?;

	// Synthetic state-layer input records (the scan source for visited bases).
	let num_records = 200usize;
	let sif: Vec<bool> = (0..num_records * state_input_len).map(|_| xf(&mut rng) < 0.5).collect();

	// GPU plant (reads records, writes the latch cells).
	let trainer = ControllerTrainer::new()?;
	let used = vec![false; n_state];
	let (gpu_n, gpu_cells) = trainer.plant_latch(&c, &sif, state_input_len, trigger, high_on, &used)?;

	// CPU reference (mutates state_memory).
	let cpu_n = c.split_plant_latch_pub(trigger, high_on, &sif, state_input_len);
	if gpu_n != cpu_n {
		return Err(format!("planted-neuron mismatch: gpu={gpu_n:?} cpu={cpu_n:?}"));
	}
	let n = cpu_n.ok_or_else(|| "no neuron planted".to_string())?;

	// Compare the cell FUNCTION over the union of touched addresses (miss → EMPTY=2).
	let cpu_entries = c.state_entries(n);
	let cpu_cells = cpu_entries.iter().filter(|&&(_, v)| v != 2).count();
	let mut all: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
	for &(a, _) in &gpu_cells { all.insert(a); }
	for &(a, _) in &cpu_entries { all.insert(a); }
	let gpu_map: std::collections::HashMap<u64, u8> = gpu_cells.iter().copied().collect();
	let mut mism = 0usize;
	for a in &all {
		let gv = *gpu_map.get(a).unwrap_or(&2u8);
		let cv = c.state_cell(n, *a);
		if gv != cv { mism += 1; }
	}
	Ok((all.len(), cpu_cells, mism))
}

/// PyO3: bit-exact parity for the GPU controller_plant_table COUNTER path (P4) vs
/// CPU split_install_counter. Wires a 3-level increment chain (neurons 0,1,2 observe
/// trigger + self + lower) and compares the planted chain + per-neuron cell function.
#[pyfunction]
pub fn run_controller_plant_counter_parity_test() -> Vec<(String, bool, String)> {
	let mut results = Vec::new();
	if Device::system_default().is_none() {
		results.push(("controller_plant_counter_parity".to_string(), true, "skipped: no Metal device".to_string()));
		return results;
	}
	match controller_plant_counter_parity_once() {
		Ok((levels, addrs, mism)) => {
			results.push(("controller_plant_counter_parity".to_string(), mism == 0, format!(
				"chain_levels={levels}, addresses={addrs}, mismatches={mism}")));
		}
		Err(e) => results.push(("controller_plant_counter_parity".to_string(), false, e)),
	}
	results
}

fn controller_plant_counter_parity_once() -> Result<(usize, usize, usize), String> {
	let (num_motors, levels, n_state, sbpn, obpn, bpf, window) = (4usize, 8usize, 8usize, 12usize, 12usize, 4usize, 4usize);
	let num_features = 9usize;
	let frame_bits = num_features * bpf;            // 36
	let sensor_window = window * frame_bits;        // 144
	let state_input_len = sensor_window + n_state;  // 152
	let total_out_in = frame_bits + n_state;
	let num_out = num_motors * levels;
	let trigger = 5usize;

	let mut rng = 0xC0117E20F1A57u64 ^ 0x9E3779B97F4A7C15u64;
	let thresholds: Vec<f32> = (0..num_features * bpf).map(|_| xf(&mut rng) - 0.5).collect();

	// 3-level increment chain. Neuron k observes trigger@0, lower@1 (self of k-1, or
	// trigger for k=0), self@2; neurons 3.. avoid the trigger so the chain is exactly
	// [0,1,2]. self(k)=sensor_window+k; lower(k>0)=sensor_window+(k-1).
	let mut state_conns = vec![0i64; n_state * sbpn];
	let set = |sc: &mut [i64], c: usize, vals: &[i64]| {
		for (i, &v) in vals.iter().enumerate() { sc[c * sbpn + i] = v; }
		for i in vals.len()..sbpn { sc[c * sbpn + i] = (60 + c * sbpn + i) as i64; } // fillers (<152, ≠5)
	};
	set(&mut state_conns, 0, &[trigger as i64, (sensor_window) as i64]);                       // n0: trig, self144
	set(&mut state_conns, 1, &[trigger as i64, (sensor_window) as i64, (sensor_window + 1) as i64]); // n1: trig, lower144, self145
	set(&mut state_conns, 2, &[trigger as i64, (sensor_window + 1) as i64, (sensor_window + 2) as i64]); // n2: trig, lower145, self146
	for c in 3..n_state {
		for i in 0..sbpn { state_conns[c * sbpn + i] = (50 + ((c * sbpn + i) % (state_input_len - 50))) as i64; }
	}
	let output_conns: Vec<i64> = (0..num_out * obpn).map(|_| (xs(&mut rng) % total_out_in as u64) as i64).collect();

	let c = WnnController::new(
		num_motors, levels, bpf, window, n_state, sbpn, obpn,
		thresholds, state_conns, output_conns,
		false, 0.1, 0.95,
		false, false, false, false, false, 0.99, 1.0,
		true,
	).map_err(|e| format!("{e}"))?;

	let num_records = 200usize;
	let sif: Vec<bool> = (0..num_records * state_input_len).map(|_| xf(&mut rng) < 0.5).collect();

	let trainer = ControllerTrainer::new()?;
	let used = vec![false; n_state];
	let (gpu_chain, gpu_cells) = trainer.plant_counter(&c, &sif, state_input_len, trigger, n_state, &used)?;

	let cpu_chain = c.split_install_counter_pub(trigger, n_state, &sif, state_input_len);
	if gpu_chain != cpu_chain {
		return Err(format!("chain mismatch: gpu={gpu_chain:?} cpu={cpu_chain:?}"));
	}
	let chain = cpu_chain.ok_or_else(|| "no counter chain installed".to_string())?;

	let mut total_addrs = 0usize;
	let mut mism = 0usize;
	for (k, &n) in chain.iter().enumerate() {
		let cpu_entries = c.state_entries(n);
		let mut all: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
		for &(a, _) in &gpu_cells[k] { all.insert(a); }
		for &(a, _) in &cpu_entries { all.insert(a); }
		let gpu_map: std::collections::HashMap<u64, u8> = gpu_cells[k].iter().copied().collect();
		total_addrs += all.len();
		for a in &all {
			let gv = *gpu_map.get(a).unwrap_or(&2u8);
			let cv = c.state_cell(n, *a);
			if gv != cv { mism += 1; }
		}
	}
	Ok((chain.len(), total_addrs, mism))
}

/// PyO3: bit-exact parity for the GPU controller_plant_bidir (P4) vs CPU
/// split_install_counter_bidir. Wires a 3-level bidirectional chain and compares
/// the planted levels + per-neuron dense 2^sbpn truth table.
#[pyfunction]
pub fn run_controller_plant_bidir_parity_test() -> Vec<(String, bool, String)> {
	let mut results = Vec::new();
	if Device::system_default().is_none() {
		results.push(("controller_plant_bidir_parity".to_string(), true, "skipped: no Metal device".to_string()));
		return results;
	}
	match controller_plant_bidir_parity_once() {
		Ok((levels, addrs, mism)) => {
			results.push(("controller_plant_bidir_parity".to_string(), mism == 0, format!(
				"levels={levels}, addresses={addrs}, mismatches={mism}")));
		}
		Err(e) => results.push(("controller_plant_bidir_parity".to_string(), false, e)),
	}
	results
}

fn controller_plant_bidir_parity_once() -> Result<(usize, usize, usize), String> {
	let (num_motors, levels, n_state, sbpn, obpn, bpf, window) = (4usize, 8usize, 8usize, 12usize, 12usize, 4usize, 4usize);
	let num_features = 9usize;
	let frame_bits = num_features * bpf;
	let sensor_window = window * frame_bits;        // 144
	let state_input_len = sensor_window + n_state;  // 152
	let total_out_in = frame_bits + n_state;
	let num_out = num_motors * levels;
	let (up, dn) = (5usize, 6usize);
	let n_levels = 3usize;

	let mut rng = 0xB1D14_C0117E2u64 ^ 0x9E3779B97F4A7C15u64;
	let thresholds: Vec<f32> = (0..num_features * bpf).map(|_| xf(&mut rng) - 0.5).collect();

	// Wire neurons 0..n_levels as the bidir chain: [up, dn, lower, self, upper].
	// lower(k)=up if k==0 else sensor_window+(k-1); self=sensor_window+k;
	// upper=sensor_window+(k+1) for non-top (top's upper is unchecked).
	let mut state_conns = vec![0i64; n_state * sbpn];
	for k in 0..n_levels {
		let lower = if k == 0 { up } else { sensor_window + (k - 1) };
		let upper = sensor_window + (k + 1);   // for top this is unchecked; any value is fine
		let vals = [up as i64, dn as i64, lower as i64, (sensor_window + k) as i64, upper as i64];
		for (i, &v) in vals.iter().enumerate() { state_conns[k * sbpn + i] = v; }
		for i in 5..sbpn { state_conns[k * sbpn + i] = (60 + k * sbpn + i) as i64; }
	}
	for c in n_levels..n_state {
		for i in 0..sbpn { state_conns[c * sbpn + i] = (50 + ((c * sbpn + i) % (state_input_len - 50))) as i64; }
	}
	let output_conns: Vec<i64> = (0..num_out * obpn).map(|_| (xs(&mut rng) % total_out_in as u64) as i64).collect();

	let c = WnnController::new(
		num_motors, levels, bpf, window, n_state, sbpn, obpn,
		thresholds, state_conns, output_conns,
		false, 0.1, 0.95,
		false, false, false, false, false, 0.99, 1.0,
		true,
	).map_err(|e| format!("{e}"))?;

	let trainer = ControllerTrainer::new()?;
	let used = vec![false; n_state];
	let (gpu_levels, gpu_cells) = trainer.plant_counter_bidir(&c, up, dn, n_levels, &used)?;

	let cpu_levels = c.split_install_counter_bidir_pub(up, dn, n_levels);
	if gpu_levels != cpu_levels {
		return Err(format!("levels mismatch: gpu={gpu_levels:?} cpu={cpu_levels:?}"));
	}
	let lv = cpu_levels.ok_or_else(|| "no bidir counter installed".to_string())?;

	let mut total_addrs = 0usize;
	let mut mism = 0usize;
	for (k, &n) in lv.iter().enumerate() {
		let gpu_map: std::collections::HashMap<u64, u8> = gpu_cells[k].iter().copied().collect();
		// CPU wrote the full 2^sbpn table; compare every address.
		for a in 0..(1u64 << sbpn) {
			total_addrs += 1;
			let gv = *gpu_map.get(&a).unwrap_or(&2u8);
			let cv = c.state_cell(n, a);
			if gv != cv { mism += 1; }
		}
	}
	Ok((lv.len(), total_addrs, mism))
}

#[cfg(test)]
mod tests {
	use super::*;

	/// Force runtime compilation of controller_rollout.metal with the common.metal
	/// preamble (now sourced from ../core/shaders after the 2026-06-19 crate split).
	/// A bad include_str! path or a common.metal/controller_rollout collision fails
	/// HERE, not mid-run.
	#[test]
	fn controller_rollout_shader_compiles() {
		if Device::system_default().is_none() {
			eprintln!("skipping: no Metal device");
			return;
		}
		ControllerRolloutEvaluator::new().expect("controller_rollout.metal");
	}
}
