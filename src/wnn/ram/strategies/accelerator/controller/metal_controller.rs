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
	record_pipeline: ComputePipelineState,  // controller_record (P2)
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
		Ok(Self { device, queue, pipeline, record_pipeline })
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
