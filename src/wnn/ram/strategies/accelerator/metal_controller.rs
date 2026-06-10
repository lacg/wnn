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

use crate::cancel::check_cancel;
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
		let src = concat!(include_str!("shaders/common.metal"), "\n", include_str!("shaders/controller_rollout.metal"));
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
		let num_out = num_motors * levels;
		let frame_bits = 9 * bpf;
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
