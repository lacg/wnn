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

/// E5 residual-hybrid config for the GPU rollout: analytic PID baseline gains +
/// the residual scale/clamp. `pid` = [kp_rp, ki_rp, kd_rp, iclamp_rp, kp_yaw,
/// ki_yaw, kd_yaw, iclamp_yaw, hover, authority] (mirror AttitudePidRs fields).
#[derive(Clone, Copy)]
pub struct ResidualCfg {
	pub scale: f32,
	pub clamp: f32,
	pub pid: [f32; 10],
	/// Firmware cascade for the residual baseline. `None` = the legacy single-loop PID
	/// (`pid` above). When set, the kernel runs the same cascade
	/// `crate::pid_firmware::AttitudePidFirmwareRs` runs on the CPU.
	pub cascade: Option<PidFwCfg>,
}

/// The firmware PID cascade as the kernel needs it: SI gains + host-precomputed filter
/// coefficients. Mirrors the `pidfw_*` tail of `RolloutParams`.
#[derive(Clone, Copy)]
pub struct PidFwCfg {
	/// [roll, pitch, yaw] x [kp, ki, kd, i_limit], rad -> rad/s.
	pub att: [f32; 12],
	/// [roll, pitch, yaw] x [kp, ki, kd, i_limit], rad/s -> newtons.
	pub rate: [f32; 12],
	pub out_limit_n: f32,
	pub hover_n: f32,
	pub k_thrust: f32,
	/// main_loop_hz / attitude_hz — the cascade updates every Nth physical step.
	pub decimation: u32,
	/// b0, b1, b2, a1, a2 from filter.c's lpf2pSetCutoffFreq.
    pub lpf: [f32; 5],
	pub filter_on: u32,
}

/// Kernel-side rotor-table width — lockstep with MAX_ROTORS in
/// controller_rollout.metal (octo-X is the widest preset).
const MAX_ROTORS_GPU: usize = 8;

/// One rotor of the geometry buffer(27) — 48 B, all-f32, field-for-field
/// lockstep with the Metal `RotorGpu` struct. `k_thrust` is the EFFECTIVE
/// coefficient (nominal × per-rotor asym baked at build time; tilt/position
/// error arrive as a perturbed table), so the kernel needs no per-rotor
/// disturbance fields.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RotorGpu {
	px: f32, py: f32, pz: f32,   // position, body frame (m)
	ax: f32, ay: f32, az: f32,   // unit thrust axis, body frame
	spin: f32,                   // +1 CCW / -1 CW (drag-torque sign)
	k_thrust: f32,               // N per pwm² (effective)
	k_drag: f32,                 // drag-torque/thrust ratio
	_pad0: f32, _pad1: f32, _pad2: f32,
}

/// Build the GPU rotor table from 9-float geometry rows
/// [px,py,pz, ax,ay,az, spin, k_thrust, k_drag] — the SAME row contract as
/// AttitudeSim::set_geometry (axis normalized here, mirroring set_geometry_core
/// so CPU and GPU see identical unit axes). `rotor_asym` (the N-rotor D3 twin)
/// is baked into the effective k_thrust: the CPU model computes
/// (k_thrust * asym) * p * p, so k_eff = k_thrust * asym is value-identical.
pub fn build_rotor_table(rows: &[[f32; 9]], rotor_asym: Option<&[f32]>) -> Result<Vec<RotorGpu>, String> {
	if rows.is_empty() || rows.len() > MAX_ROTORS_GPU {
		return Err(format!(
			"geometry needs 1..={MAX_ROTORS_GPU} rotors, got {}", rows.len()));
	}
	if let Some(a) = rotor_asym {
		if a.len() != rows.len() {
			return Err(format!("rotor_asym len {} != num_rotors {}", a.len(), rows.len()));
		}
	}
	Ok(rows.iter().enumerate().map(|(i, r)| {
		let n = (r[3] * r[3] + r[4] * r[4] + r[5] * r[5]).sqrt().max(1e-9);
		let asym = rotor_asym.map_or(1.0, |a| a[i]);
		RotorGpu {
			px: r[0], py: r[1], pz: r[2],
			ax: r[3] / n, ay: r[4] / n, az: r[5] / n,
			spin: r[6],
			k_thrust: r[7] * asym,
			k_drag: r[8],
			_pad0: 0.0, _pad1: 0.0, _pad2: 0.0,
		}
	}).collect())
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
	obs_peraxis_yaw: u32,    // per-axis carries yaw (1) or only roll+pitch (0)
	obs_yaw_err: u32,        // yaw-anchor: clean scalar (target_yaw − anchored heading)
	obs_yaw_err_i: u32,      // yaw-anchor: leaky integral of the yaw error
	obs_pwm: u32,
	integral_leak: f32,
	integral_scale: f32,
	decouple_outputs: u32,   // H3: 4 banks are controls [T,τr,τp,τy] → mix to motors
	// Action-repeat N (arm R): decide every Nth step, HOLD the PWM in between.
	// APPENDED at the END — layout must match the Metal Params struct exactly.
	action_repeat: u32,
	// --- W2 disturbances (layout must match the Metal Params exactly). ---
	// dist_enabled=0 ⇒ the pre-W2 clean rollout. Semantics mirror
	// controller.rs `Disturbance`; seed is PRE-FOLDED via dist_seed32.
	dist_enabled: u32,
	dist_tau_bias0: f32,
	dist_tau_bias1: f32,
	dist_tau_bias2: f32,
	dist_gust_sigma: f32,
	dist_gust_tau_c: f32,
	dist_motor_asym0: f32,
	dist_motor_asym1: f32,
	dist_motor_asym2: f32,
	dist_motor_asym3: f32,
	dist_gyro_sigma: f32,
	dist_gyro_bias_walk: f32,
	dist_accel_sigma: f32,
	dist_seed: u32,
	// Global index of episode 0 in this dispatch — score() chunks episodes;
	// per-episode seeds (channel-15 hash) must not depend on the chunk size.
	dist_ep_offset: u32,
	// --- E5 residual hybrid (APPENDED at END; layout must match Metal Params). ---
	// residual_enabled=0 ⇒ pure-WNN (pre-E5). When 1, the WNN output is a signed
	// residual composed on an analytic PID baseline (compose_residual), all in-kernel.
	residual_enabled: u32,
	pid_kp_rp: f32,
	pid_ki_rp: f32,
	pid_kd_rp: f32,
	pid_iclamp_rp: f32,
	pid_kp_yaw: f32,
	pid_ki_yaw: f32,
	pid_kd_yaw: f32,
	pid_iclamp_yaw: f32,
	pid_hover: f32,
	pid_authority: f32,
	residual_scale: f32,
	residual_clamp: f32,
	// --- Overactuated Phase 2 (APPENDED at END; layout must match Metal Params).
	// 1 ⇒ the residual composes on the allocator-LQR baseline (buffer 28)
	// instead of the quad PID.
	alloc_baseline: u32,
	// Residual neutral anchor (ABI 11) — mode-derived neutral since ABI 12
	// (cell_mode::neutral_decode; also the delta-control neutral in-kernel).
	residual_neutral: f32,
	// Memory mode of the population's cells (ABI 12): 0 TERNARY / 1-2 QUAD /
	// 3 BINARY (antagonist-pair output decode). APPENDED at END — layout must
	// match the Metal Params exactly.
	memory_mode: u32,
	// Output decode TOPOLOGY (03/08/2026): WNN_DECODE_CUMULATIVE/ANTAGONIST.
	// Orthogonal to memory_mode — see cell_mode.rs. APPENDED at END.
	output_decode: u32,
	// --- W2.4 D5/D6/D7 (APPENDED at END; layout must match the Metal Params
	//     exactly). 0 = exactly-off = the bit-identical pre-W2.4 rollout. ---
	dist_dropout_prob: f32,
	dist_dropout_len_steps: u32,
	dist_obs_delay_steps: u32,
	dist_torque_scale_jitter: f32,
	// --- FIRMWARE PID CASCADE (APPENDED at END; layout must match the Metal Params
	//     exactly — every field here is 4 bytes, so the two structs stay sequential
	//     with no padding to reason about). pidfw_on = 0 keeps the legacy single-loop
	//     pid_step, which is the bit-identical pre-cascade rollout. ---
	// Gains are ALREADY SI (rad, rad/s, newtons): Python's _SiGains.from_firmware is
	// the one place degrees and actuator counts exist. Layout per loop is
	// [roll, pitch, yaw] x [kp, ki, kd, i_limit].
	pidfw_on: u32,
	pidfw_att: [f32; 12],
	pidfw_rate: [f32; 12],
	pidfw_out_limit_n: f32,
	pidfw_hover_n: f32,
	pidfw_k_thrust: f32,
	pidfw_decimation: u32,
	// lpf2p coefficients b0,b1,b2,a1,a2 PRECOMPUTED ON THE HOST. Deliberate: deriving
	// them in-shader from tan/cos would let the GPU and CPU disagree about the filter
	// in the last bits, and the filter is the term that decides stability.
	pidfw_lpf: [f32; 5],
	pidfw_filter_on: u32,
	// L1 d̂ observer (06/08/2026), appended at the END of BOTH structs in lockstep.
	// dhat_on=0 ⇒ the pre-L1 feature set, bit-identical.
	dhat_on: u32,
	dhat_b: [f32; 3],
	dhat_l_gain: f32,
}

pub struct ControllerRolloutEvaluator {
	device: Device,
	queue: CommandQueue,
	library: Library,
	pipeline: ComputePipelineState,
	// Overactuated Phase 1: specialized pipelines keyed by rotor count N
	// (HAS_GEOMETRY=true is implied — the legacy quad is `pipeline` above).
	// RefCell: score() is &self and the evaluator is single-threaded per call.
	geom_pipelines: std::cell::RefCell<std::collections::HashMap<u32, ComputePipelineState>>,
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
		// controller_rollout declares function constants (overactuated Phase 1:
		// FC_HAS_GEOMETRY/FC_NUM_ROTORS) — Metal then REQUIRES specialized
		// function creation even for the default (quad) pipeline. An empty
		// constant set leaves both undefined → is_function_constant_defined
		// fallbacks give USE_GEOMETRY=false / NUM_ROTORS=4, i.e. the exact
		// legacy pipeline (generic path dead-stripped).
		let fcv = metal::FunctionConstantValues::new();
		let func = library
			.get_function("controller_rollout", Some(fcv))
			.map_err(|e| format!("kernel controller_rollout not found: {e}"))?;
		let pipeline = device
			.new_compute_pipeline_state_with_function(&func)
			.map_err(|e| format!("pipeline creation failed: {e}"))?;
		Ok(Self {
			device, queue, library, pipeline,
			geom_pipelines: std::cell::RefCell::new(std::collections::HashMap::new()),
		})
	}

	/// Get-or-create the specialized rollout pipeline for an N-rotor geometry:
	/// FC_HAS_GEOMETRY(0)=true + FC_NUM_ROTORS(1)=n → the compiler dead-strips
	/// the quad torque block and unrolls the generic loop at N. Cached per
	/// evaluator (ComputePipelineState clones are ObjC retains — cheap).
	fn geometry_pipeline(&self, n: u32) -> Result<ComputePipelineState, String> {
		if let Some(p) = self.geom_pipelines.borrow().get(&n) {
			return Ok(p.clone());
		}
		let fcv = metal::FunctionConstantValues::new();
		let has_geometry = true;
		fcv.set_constant_value_at_index(
			&has_geometry as *const bool as *const _, MTLDataType::Bool, 0);
		fcv.set_constant_value_at_index(
			&n as *const u32 as *const _, MTLDataType::UInt, 1);
		let func = self.library
			.get_function("controller_rollout", Some(fcv))
			.map_err(|e| format!("controller_rollout specialization (N={n}) failed: {e}"))?;
		let p = self.device
			.new_compute_pipeline_state_with_function(&func)
			.map_err(|e| format!("geometry pipeline creation (N={n}) failed: {e}"))?;
		self.geom_pipelines.borrow_mut().insert(n, p.clone());
		Ok(p)
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
		controllers: &[&WnnController],
		q0: &[f32],
		omega0: &[f32],
		num_episodes: usize,
		steps: usize,
		sim: (f32, f32, f32, [f32; 3], f32),   // (dt, arm, k_thrust, inertia, gravity) ... k_drag below
		k_drag: f32,
		target: [f32; 3],
		// W2 disturbances: None = clean rollout (pre-W2 behavior). The
		// Disturbance seed is the BASE seed; the kernel derives per-episode
		// seeds via the channel-15 hash on the GLOBAL episode index.
		dist: Option<crate::controller::Disturbance>,
		// E5 residual hybrid: None = pure-WNN. Some ⇒ the WNN output is composed as
		// a signed residual on an in-kernel baseline (quad PID, or the
		// allocator-LQR when alloc_baseline is set).
		residual: Option<ResidualCfg>,
		// Overactuated Phase 1: None = the legacy quad sim (bit-identical
		// pre-geometry pipeline). Some(table) ⇒ the CPU step_n twin: generic
		// r×F + spin-drag torque over the table's rotors on a specialized
		// pipeline. NOTE: dist.motor_asym is IGNORED on this path (mirrors
		// step_n_core — per-rotor asym is baked into the table's k_thrust).
		geometry: Option<&[RotorGpu]>,
		// Overactuated Phase 2: allocator-LQR residual baseline (requires
		// `residual` for scale/clamp; its pid gains are then unused). Built
		// from the NOMINAL geometry — pass the PERTURBED table via `geometry`.
		alloc_baseline: Option<&crate::optimal::AllocBaseline>,
	) -> Result<Vec<Vec<f64>>, String> {
		let g = controllers.len();
		if g == 0 {
			return Ok(vec![]);
		}
		let (num_motors, levels, n_state, sbpn, obpn, bpf, window) = controllers[0].gpu_dims();
		if let Some(ab) = alloc_baseline {
			if residual.is_none() {
				return Err("score_controllers_metal: alloc_baseline requires residual \
				            (scale/clamp) — pass residual cfg.".to_string());
			}
			if ab.num_rotors() != num_motors {
				return Err(format!(
					"score_controllers_metal: alloc baseline has {} rotors but controllers \
					 emit num_motors={} PWMs — they must match.", ab.num_rotors(), num_motors));
			}
		}
		if let Some(rotors) = geometry {
			// The controller must emit exactly one PWM per rotor, and the
			// kernel's fixed-width pwm arrays cap at MAX_ROTORS.
			if rotors.len() != num_motors {
				return Err(format!(
					"score_controllers_metal: geometry has {} rotors but controllers emit \
					 num_motors={} PWMs — they must match.", rotors.len(), num_motors));
			}
			// The in-kernel quad-only blocks (decouple mix reads pwm[0..3]; the E5
			// residual PID writes base[4]) are undefined for N≠4. decouple is already
			// impossible at the controller level (new_core enforces num_motors==4);
			// the quad-PID residual must be refused loudly here (the allocator-LQR
			// baseline is the N-rotor path).
			if num_motors != 4 && residual.is_some() && alloc_baseline.is_none() {
				return Err(format!(
					"score_controllers_metal: residual hybrid (quad PID baseline) is not \
					 supported with an N={num_motors} geometry — pass an alloc baseline \
					 or CPU fallback."));
			}
		}
		// GUARD (09/07/2026): the kernel's thread-private prev_state/new_state arrays are
		// MAX_STATE_NEURONS-sized (64). A controller with more state neurons would overflow
		// them (UB → silent zero/garbage metrics + adjacent-thread corruption). Refuse
		// loudly so the caller (Python _score_population_gpu) CPU-falls-back instead. Keep
		// this in lockstep with MAX_STATE_NEURONS in controller_rollout.metal.
		const MAX_STATE_NEURONS_GPU: usize = 64;
		if n_state > MAX_STATE_NEURONS_GPU {
			return Err(format!(
				"score_controllers_metal: state_neurons={} exceeds GPU MAX_STATE_NEURONS={} \
				 (would overflow thread-private arrays) — CPU fallback required.",
				n_state, MAX_STATE_NEURONS_GPU));
		}
		// Delta-control mode (uniform across the population) so the kernel decodes
		// the SAME way step() does (was absolute-only → wrong for delta controllers).
		let (delta_control, delta_max, delta_leak) = controllers[0].delta_params();
		// H2 observation-feature config (uniform); num_features drives frame sizing
		// (was hardcoded 9 → ignored the H2 extras).
		let (num_features, obs_tilt_p, obs_tilt_i, obs_peraxis_p, obs_peraxis_i,
		     obs_pwm, integral_leak, integral_scale, decouple_outputs,
		     obs_peraxis_yaw, obs_yaw_err, obs_yaw_err_i, _ctrl_dt) = controllers[0].obs_params();
		// Action-repeat N (uniform across the population, like delta/obs config).
		let action_repeat = controllers[0].action_repeat_n();
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

		// Pipeline: legacy quad (function constants undefined → generic path
		// dead-stripped, bit-identical pre-geometry kernel) or the N-rotor
		// specialization. Selected ONCE — uniform across all chunks.
		let geom_pipeline = match geometry {
			Some(rotors) => Some(self.geometry_pipeline(rotors.len() as u32)?),
			None => None,
		};
		let pipeline: &ComputePipelineState = geom_pipeline.as_ref().unwrap_or(&self.pipeline);

		// Static input buffers — allocated once, reused across chunks.
		// The rotor table binds at buffer(27) even on the legacy pipeline
		// (1-element pad; the slot is dead-stripped there and never read).
		let b_rot = self.buf(geometry.unwrap_or(&[]));
		// Alloc-LQR baseline blob at buffer(28) — 1-float pad when disabled
		// (runtime flag; the kernel never reads it then).
		let alloc_blob = alloc_baseline.map(|ab| ab.to_gpu_blob()).unwrap_or_default();
		let b_alloc = self.buf(&alloc_blob);
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
		let mut sum_steady_per_g = vec![0.0f64; g]; // mean attitude err over last-20% window, summed
		// Transient-speed metrics (rise/settle/ITAE), summed over episodes per genome.
		let mut sum_rise_per_g = vec![0.0f64; g];
		let mut sum_settleab_per_g = vec![0.0f64; g];
		let mut sum_settlere_per_g = vec![0.0f64; g];
		let mut sum_itae_per_g = vec![0.0f64; g];
		let mut sum_iae_per_g = vec![0.0f64; g];
		let mut sum_ise_per_g = vec![0.0f64; g];
		// Allocation-effort (Phase 3): per-episode mean Σ_m pwm², summed.
		let mut sum_effort_per_g = vec![0.0f64; g];
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
				obs_peraxis_yaw: obs_peraxis_yaw as u32,
				obs_yaw_err: obs_yaw_err as u32, obs_yaw_err_i: obs_yaw_err_i as u32,
				obs_pwm: obs_pwm as u32,
				integral_leak, integral_scale,
				decouple_outputs: decouple_outputs as u32,
				action_repeat: action_repeat as u32,
				dist_enabled: if dist.is_some() { 1 } else { 0 },
				dist_tau_bias0: dist.map_or(0.0, |d| d.tau_bias[0]),
				dist_tau_bias1: dist.map_or(0.0, |d| d.tau_bias[1]),
				dist_tau_bias2: dist.map_or(0.0, |d| d.tau_bias[2]),
				dist_gust_sigma: dist.map_or(0.0, |d| d.gust_sigma),
				dist_gust_tau_c: dist.map_or(0.1, |d| d.gust_tau_c),
				dist_motor_asym0: dist.map_or(1.0, |d| d.motor_asym[0]),
				dist_motor_asym1: dist.map_or(1.0, |d| d.motor_asym[1]),
				dist_motor_asym2: dist.map_or(1.0, |d| d.motor_asym[2]),
				dist_motor_asym3: dist.map_or(1.0, |d| d.motor_asym[3]),
				dist_gyro_sigma: dist.map_or(0.0, |d| d.gyro_sigma),
				dist_gyro_bias_walk: dist.map_or(0.0, |d| d.gyro_bias_walk),
				dist_accel_sigma: dist.map_or(0.0, |d| d.accel_sigma),
				dist_seed: dist.map_or(0, |d| crate::controller::dist_seed32(d.seed)),
				dist_ep_offset: chunk_start as u32,
				// E5 residual hybrid: PID baseline gains + residual scale/clamp.
				residual_enabled: if residual.is_some() { 1 } else { 0 },
				pid_kp_rp: residual.map_or(0.0, |r| r.pid[0]),
				pid_ki_rp: residual.map_or(0.0, |r| r.pid[1]),
				pid_kd_rp: residual.map_or(0.0, |r| r.pid[2]),
				pid_iclamp_rp: residual.map_or(0.0, |r| r.pid[3]),
				pid_kp_yaw: residual.map_or(0.0, |r| r.pid[4]),
				pid_ki_yaw: residual.map_or(0.0, |r| r.pid[5]),
				pid_kd_yaw: residual.map_or(0.0, |r| r.pid[6]),
				pid_iclamp_yaw: residual.map_or(0.0, |r| r.pid[7]),
				pid_hover: residual.map_or(0.5, |r| r.pid[8]),
				pid_authority: residual.map_or(0.4, |r| r.pid[9]),
				residual_scale: residual.map_or(1.0, |r| r.scale),
				residual_clamp: residual.map_or(0.4, |r| r.clamp),
				alloc_baseline: if alloc_baseline.is_some() { 1 } else { 0 },
				residual_neutral: controllers[0].neutral_f32(),
				memory_mode: controllers[0].memory_mode_u8() as u32,
				output_decode: controllers[0].output_decode_u8() as u32,
				// W2.4 D5/D6/D7 — 0 when dist is None (exactly-off).
				dist_dropout_prob: dist.map_or(0.0, |d| d.dropout_prob),
				dist_dropout_len_steps: dist.map_or(0, |d| d.dropout_len_steps),
				dist_obs_delay_steps: dist.map_or(0, |d| d.obs_delay_steps),
				dist_torque_scale_jitter: dist.map_or(0.0, |d| d.torque_scale_jitter),
				// Firmware cascade — off unless the residual config carries it.
				pidfw_on: if residual.and_then(|r| r.cascade).is_some() { 1 } else { 0 },
				pidfw_att: residual.and_then(|r| r.cascade)
					.map_or([0.0; 12], |c| c.att),
				pidfw_rate: residual.and_then(|r| r.cascade)
					.map_or([0.0; 12], |c| c.rate),
				pidfw_out_limit_n: residual.and_then(|r| r.cascade)
					.map_or(0.0, |c| c.out_limit_n),
				pidfw_hover_n: residual.and_then(|r| r.cascade)
					.map_or(0.0, |c| c.hover_n),
				pidfw_k_thrust: residual.and_then(|r| r.cascade)
					.map_or(0.0, |c| c.k_thrust),
				pidfw_decimation: residual.and_then(|r| r.cascade)
					.map_or(1, |c| c.decimation.max(1)),
				pidfw_lpf: residual.and_then(|r| r.cascade)
					.map_or([0.0; 5], |c| c.lpf),
				pidfw_filter_on: residual.and_then(|r| r.cascade)
					.map_or(0, |c| c.filter_on),
				// L1 d̂ observer — read from the controller itself (uniform across the
				// population, like every other obs_* flag), so the kernel cannot use a
				// b that disagrees with the one the CPU path computes features from.
				dhat_on: if controllers[0].dhat_params().is_some() { 1 } else { 0 },
				dhat_b: controllers[0].dhat_params().map_or([0.0; 3], |(b, _)| b),
				dhat_l_gain: controllers[0].dhat_params().map_or(0.05, |(_, g)| g),
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
			let b_steady = mk_out(n_out_chunk * mem::size_of::<f32>());
			// Transient-speed metric buffers (rise/settle_abs/settle_rel/itae/iae/ise).
			let b_rise = mk_out(n_out_chunk * mem::size_of::<f32>());
			let b_settleab = mk_out(n_out_chunk * mem::size_of::<f32>());
			let b_settlere = mk_out(n_out_chunk * mem::size_of::<f32>());
			let b_itae = mk_out(n_out_chunk * mem::size_of::<f32>());
			let b_iae = mk_out(n_out_chunk * mem::size_of::<f32>());
			let b_ise = mk_out(n_out_chunk * mem::size_of::<f32>());
			let b_effort = mk_out(n_out_chunk * mem::size_of::<f32>());

			let cmd = self.queue.new_command_buffer();
			let enc = cmd.new_compute_command_encoder();
			enc.set_compute_pipeline_state(pipeline);
			let bufs: [&Buffer; 30] = [
				&b_sc, &b_oc, &b_sk, &b_sv, &b_so, &b_scn, &b_ok, &b_ov, &b_oo, &b_ocn,
				&b_th, &b_q0, &b_w0, &b_par, &b_reward, &b_sumerr, &b_steps, &b_div,
				&b_jerk, &b_mono, &b_steady,
				&b_rise, &b_settleab, &b_settlere, &b_itae, &b_iae, &b_ise,
				&b_rot, &b_alloc, &b_effort,
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

			// ROBUSTNESS (09/07/2026): a failed/timed-out command buffer leaves the
			// output buffers at their zero-init state (mk_out zeroes them). Reading
			// those back silently yielded all-zero metrics (err=0/stable=0) for the
			// WHOLE chunk — which then poisoned the GA (0-viable stall / degenerate
			// stable=0% winner). This is INTERMITTENT under heavy GPU load (large
			// batches, or a second process contending with the IDS worker's Metal
			// kernels). NEVER return a silently-zeroed result: fail loud so the caller
			// (Python _score_population_gpu) falls back to the CPU per-step eval.
			if cmd.status() != MTLCommandBufferStatus::Completed {
				return Err(format!(
					"score_controllers_metal: Metal command buffer did not complete \
					 (status={:?}) on chunk g={} chunk_eps={} — refusing to read back a \
					 silently-zeroed buffer; caller must CPU-fallback.",
					cmd.status(), g, chunk_ep_count));
			}

			// Accumulate this chunk's results into per-genome totals.
			let reward = unsafe { std::slice::from_raw_parts(b_reward.contents() as *const f32, n_out_chunk) };
			let sumerr = unsafe { std::slice::from_raw_parts(b_sumerr.contents() as *const f32, n_out_chunk) };
			let stepsv = unsafe { std::slice::from_raw_parts(b_steps.contents() as *const u32, n_out_chunk) };
			let divv = unsafe { std::slice::from_raw_parts(b_div.contents() as *const u32, n_out_chunk) };
			let jerkv = unsafe { std::slice::from_raw_parts(b_jerk.contents() as *const f32, n_out_chunk) };
			let monov = unsafe { std::slice::from_raw_parts(b_mono.contents() as *const f32, n_out_chunk) };
			let steadyv = unsafe { std::slice::from_raw_parts(b_steady.contents() as *const f32, n_out_chunk) };
			let risev = unsafe { std::slice::from_raw_parts(b_rise.contents() as *const f32, n_out_chunk) };
			let settleabv = unsafe { std::slice::from_raw_parts(b_settleab.contents() as *const f32, n_out_chunk) };
			let settlerev = unsafe { std::slice::from_raw_parts(b_settlere.contents() as *const f32, n_out_chunk) };
			let itaev = unsafe { std::slice::from_raw_parts(b_itae.contents() as *const f32, n_out_chunk) };
			let iaev = unsafe { std::slice::from_raw_parts(b_iae.contents() as *const f32, n_out_chunk) };
			let isev = unsafe { std::slice::from_raw_parts(b_ise.contents() as *const f32, n_out_chunk) };
			let effortv = unsafe { std::slice::from_raw_parts(b_effort.contents() as *const f32, n_out_chunk) };
			for gi in 0..g {
				for ce in 0..chunk_ep_count {
					let idx = gi * chunk_ep_count + ce;
					let st = stepsv[idx].max(1) as f64;
					let mean_err = sumerr[idx] as f64 / st;
					sum_reward_per_g[gi] += reward[idx] as f64;
					sum_mean_err_per_g[gi] += mean_err;
					sum_jerk_per_g[gi] += jerkv[idx] as f64;
					sum_mono_per_g[gi] += monov[idx] as f64;
					sum_steady_per_g[gi] += steadyv[idx] as f64;
					sum_rise_per_g[gi] += risev[idx] as f64;
					sum_settleab_per_g[gi] += settleabv[idx] as f64;
					sum_settlere_per_g[gi] += settlerev[idx] as f64;
					sum_itae_per_g[gi] += itaev[idx] as f64;
					sum_iae_per_g[gi] += iaev[idx] as f64;
					sum_ise_per_g[gi] += isev[idx] as f64;
					sum_effort_per_g[gi] += effortv[idx] as f64;
					if divv[idx] == 0 && mean_err <= stable_thresh {
						stable_count_per_g[gi] += 1;
					}
				}
			}
			completed_episodes = chunk_end;
			chunk_start = chunk_end;
		}

		// Aggregate per-genome over completed episodes only. Each row is 13 metrics:
		// [reward, err_rad, stable, jerk, mono, steady_rad, rise_s, settle_abs_s,
		//  settle_rel_s, itae, iae, ise, effort]. Vec<Vec> (not a tuple) so more
		// metrics can be appended without hitting PyO3's 12-arity tuple ceiling.
		// If none completed (cancellation before the first chunk) → all-zero sentinel.
		let mut out = Vec::with_capacity(g);
		if completed_episodes == 0 {
			for _ in 0..g {
				out.push(vec![0.0_f64; 13]);
			}
		} else {
			let n = completed_episodes as f64;
			for gi in 0..g {
				out.push(vec![
					sum_reward_per_g[gi] / n,
					sum_mean_err_per_g[gi] / n,
					stable_count_per_g[gi] as f64 / n,
					sum_jerk_per_g[gi] / n,
					sum_mono_per_g[gi] / n,
					sum_steady_per_g[gi] / n,
					sum_rise_per_g[gi] / n,
					sum_settleab_per_g[gi] / n,
					sum_settlere_per_g[gi] / n,
					sum_itae_per_g[gi] / n,
					sum_iae_per_g[gi] / n,
					sum_ise_per_g[gi] / n,
					sum_effort_per_g[gi] / n,
				]);
			}
		}
		Ok(out)
	}
}

/// PyO3 entry: score a population of controllers on the GPU. Returns per-genome
/// (mean_reward, mean_attitude_error_rad, stable_rate, mean_jerk, mean_mono,
/// mean_steady_error_rad). Sim params default to AttitudeSim's defaults so the
/// rollout physics matches the CPU sim.
#[pyfunction]
#[pyo3(signature = (
	controllers, q0, omega0, num_episodes, steps,
	dt = 0.001, arm_length = 0.075, k_thrust = 2.4, k_drag = 0.05,
	inertia = [0.0023, 0.0023, 0.0046], gravity = 9.81,
	target = [0.0, 0.0, 0.0],
	dist_enabled = false,
	dist_tau_bias = [0.0, 0.0, 0.0],
	dist_gust_sigma = 0.0,
	dist_gust_tau_c = 0.1,
	dist_motor_asym = [1.0, 1.0, 1.0, 1.0],
	dist_gyro_sigma = 0.0,
	dist_gyro_bias_walk = 0.0,
	dist_accel_sigma = 0.0,
	dist_seed = 0,
	// W2.4 D5 dropout/freeze + D6 latency + D7 torque-scale jitter —
	// 0-defaults = exactly-off (bit-identical legacy rollout).
	dist_dropout_prob = 0.0,
	dist_dropout_len_steps = 0,
	dist_obs_delay_steps = 0,
	dist_torque_scale_jitter = 0.0,
	// E5 residual hybrid — default disabled = pure-WNN. pid_gains =
	// [kp_rp, ki_rp, kd_rp, iclamp_rp, kp_yaw, ki_yaw, kd_yaw, iclamp_yaw, hover, authority].
	residual_enabled = false,
	residual_scale = 1.0,
	residual_clamp = 0.4,
	pid_gains = [1.2, 0.0, 0.30, 0.5, 0.6, 0.0, 0.20, 0.5, 0.5, 0.4],
	// Overactuated Phase 1 — None = legacy quad sim. Rows are
	// [px,py,pz, ax,ay,az, spin, k_thrust, k_drag] (the set_geometry row
	// contract; pass the PERTURBED table for tilt/position error).
	// rotor_asym = per-rotor thrust multipliers (N-rotor D3 twin), baked
	// into the effective k_thrust at upload.
	geometry = None,
	rotor_asym = None,
	// Overactuated Phase 2 — allocator-LQR residual baseline. alloc_rows =
	// the NOMINAL geometry (the allocator's model; perturb only `geometry`).
	// Requires residual_enabled=true (scale/clamp; pid_gains then unused).
	alloc_rows = None,
	alloc_q_att = 12.0,
	alloc_q_rate = 1.0,
	alloc_r_ctrl = 1.0,
	alloc_tau_max = 0.144,
	alloc_f_hover = None,
	alloc_lambda = 1e-6,
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
	// W2 disturbances — defaults = disabled = pre-W2 behavior. dist_seed is
	// the BASE seed; per-episode seeds derive in-kernel (disturbance_episode_seed).
	dist_enabled: bool,
	dist_tau_bias: [f32; 3],
	dist_gust_sigma: f32,
	dist_gust_tau_c: f32,
	dist_motor_asym: [f32; 4],
	dist_gyro_sigma: f32,
	dist_gyro_bias_walk: f32,
	dist_accel_sigma: f32,
	dist_seed: u64,
	dist_dropout_prob: f32,
	dist_dropout_len_steps: u32,
	dist_obs_delay_steps: u32,
	dist_torque_scale_jitter: f32,
	residual_enabled: bool,
	residual_scale: f32,
	residual_clamp: f32,
	pid_gains: [f32; 10],
	geometry: Option<Vec<[f32; 9]>>,
	rotor_asym: Option<Vec<f32>>,
	alloc_rows: Option<Vec<[f32; 9]>>,
	alloc_q_att: f64,
	alloc_q_rate: f64,
	alloc_r_ctrl: f64,
	alloc_tau_max: f64,
	alloc_f_hover: Option<f64>,
	alloc_lambda: f32,
) -> PyResult<Vec<Vec<f64>>> {
	let evaluator = ControllerRolloutEvaluator::new()
		.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
	let alloc = match &alloc_rows {
		Some(rows) => Some(
			crate::optimal::AllocBaseline::build(
				rows, inertia, alloc_q_att, alloc_q_rate, alloc_r_ctrl,
				alloc_tau_max, alloc_f_hover, alloc_lambda,
			).map_err(pyo3::exceptions::PyValueError::new_err)?,
		),
		None => None,
	};
	let rotor_table = match &geometry {
		Some(rows) => Some(
			build_rotor_table(rows, rotor_asym.as_deref())
				.map_err(pyo3::exceptions::PyValueError::new_err)?,
		),
		None => {
			if rotor_asym.is_some() {
				return Err(pyo3::exceptions::PyValueError::new_err(
					"rotor_asym requires geometry (the quad path models motor asymmetry \
					 via dist_motor_asym instead)".to_string()));
			}
			None
		}
	};
	let residual = if residual_enabled {
		Some(ResidualCfg { scale: residual_scale, clamp: residual_clamp, pid: pid_gains,
			cascade: None })
	} else {
		None
	};
	let dist = if dist_enabled {
		Some(crate::controller::Disturbance {
			tau_bias: dist_tau_bias,
			gust_sigma: dist_gust_sigma,
			gust_tau_c: dist_gust_tau_c,
			motor_asym: dist_motor_asym,
			gyro_sigma: dist_gyro_sigma,
			gyro_bias_walk: dist_gyro_bias_walk,
			accel_sigma: dist_accel_sigma,
			dropout_prob: dist_dropout_prob,
			dropout_len_steps: dist_dropout_len_steps,
			obs_delay_steps: dist_obs_delay_steps,
			torque_scale_jitter: dist_torque_scale_jitter,
			seed: dist_seed,
		})
	} else {
		None
	};
	let refs: Vec<&WnnController> = controllers.iter().map(|c| &**c).collect();
	evaluator
		.score(&refs, &q0, &omega0, num_episodes, steps,
		       (dt, arm_length, k_thrust, inertia, gravity), k_drag, target, dist, residual,
		       rotor_table.as_deref(), alloc.as_ref())
		.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
}

// =============================================================================
// ControllerTrainer — GPU host for the controller_train kernel (split_retrain_output
// on GPU). Thread = genome. Uploads frozen state memory + connections + recorded
// trajectories + an EMPTY output marker table (values pre-set to the hover sentinel
// 2), dispatches, and reads back the trained output cells per (genome, neuron).
// =============================================================================
const MARKER_FINAL_U32: u32 = 0xFFFF_FFFFu32;

/// Host mirror of marker_slots.metal `slot_hash` / atomic_hashtable.rs `Inner::hash`
/// — the Murmur3 finalizer masked to the table size. Used to seed the output marker
/// table (train_seeded) at the exact home index find_or_claim_slot would probe from.
#[inline]
fn host_slot_hash(key: u64, mask: u64) -> usize {
	let mut x = key;
	x ^= x >> 33;
	x = x.wrapping_mul(0xff51afd7ed558ccd);
	x ^= x >> 33;
	x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
	x ^= x >> 33;
	(x & mask) as usize
}

/// Place `addr` into the [off, off+cap) marker region at the first EMPTY slot in its
/// linear-probe chain (the slot find_or_claim_slot would claim). Single-threaded host
/// seeding — keys are assumed distinct (one cell per address). cap is a power of two.
#[inline]
fn host_seed_slot(markers: &[u32], off: usize, cap: usize, addr: u64) -> usize {
	let mask = (cap - 1) as u64;
	let mut idx = host_slot_hash(addr, mask);
	for _ in 0..cap {
		let slot = off + idx;
		if markers[slot] == 0u32 { return slot; } // MARKER_EMPTY
		idx = ((idx as u64 + 1) & mask) as usize;
	}
	off // unreachable: cap sized for seeded+new with <0.5 load
}

/// Same layout as the Metal StateCommitParams.
#[repr(C)]
#[derive(Clone, Copy)]
struct StateCommitParams {
	num_genomes: u32,
	n_state: u32,
	num_commits: u32,
	memory_mode: u32,
}

/// One section-(c) state commit: nudge (genome, neuron)'s cell at `addr` toward
/// `target_true`. The walk emits these in record order; the ORDER IS SEMANTIC, because
/// nudge is a read-modify-write.
#[derive(Clone, Copy, Debug)]
pub struct StateCommit {
	pub genome: u32,
	pub neuron: u32,
	pub addr: u64,
	pub target_true: bool,
}

/// Find `addr` in the [off, off+cap) region, following the same linear probe
/// `find_or_claim_slot` uses. Returns the slot, or None if absent. The read twin of
/// `host_seed_slot` — needed to verify a resident table host-side without a readback
/// pass that re-sorts everything.
#[inline]
fn host_lookup_slot(markers: &[u32], keys: &[u64], off: usize, cap: usize, addr: u64) -> Option<usize> {
	let mask = (cap - 1) as u64;
	let mut idx = host_slot_hash(addr, mask);
	for _ in 0..cap {
		let slot = off + idx;
		if markers[slot] == 0u32 { return None; }              // EMPTY ⇒ not present
		if markers[slot] == MARKER_FINAL_U32 && keys[slot] == addr { return Some(slot); }
		idx = ((idx as u64 + 1) & mask) as usize;
	}
	None
}

/// A resident, in-kernel-WRITABLE cell layer: one open-addressing slot region per
/// (genome, neuron), addressed by `find_or_claim_slot(markers, keys, off[gn], cap[gn], …)`
/// and mutated by `slot_nudge` / `slot_set_direct`.
///
/// 04/08/2026: extracted from the output-layer construction in `train_impl` so the STATE
/// layer can reuse the identical, parity-proven layout instead of a second hand-rolled
/// copy. The bptt window walk commits to BOTH layers — section (c) to state, (d) to
/// output — so state cannot stay a read-only sorted export the way it is today
/// (docs/gpu_solve_port_design.md).
struct SlotTable {
	off: Vec<u32>,
	cap: Vec<u32>,
	markers: Vec<u32>,
	keys: Vec<u64>,
	values: Vec<u32>,
}

impl SlotTable {
	/// `steps_per_genome[gi]` bounds the NEW addresses a genome can touch (≤ one
	/// distinct address per step per neuron). `seed[gi]`, when given, is that genome's
	/// existing cells for this layer: the region must also hold them, so it is sized
	/// for (seeded + new) using the worst-case neuron, and they are replayed at exactly
	/// the slot `find_or_claim_slot` would claim so the kernel nudges FROM the
	/// accumulated value rather than from EMPTY (mirrors read_cell).
	fn build(
		steps_per_genome: &[u64],
		num_neurons: usize,
		seed: Option<&[ram_core::sparse_memory::SparseGpuExport]>,
	) -> Self {
		let g = steps_per_genome.len();
		let (mut off, mut cap) = (Vec::with_capacity(g * num_neurons), Vec::with_capacity(g * num_neurons));
		let mut total: u64 = 0;
		for (gi, &steps_g) in steps_per_genome.iter().enumerate() {
			let max_existing: u64 = seed
				.map(|s| (0..num_neurons).map(|n| s[gi].counts[n] as u64).max().unwrap_or(0))
				.unwrap_or(0);
			// 50% load factor, power-of-two so the probe mask is (cap-1).
			let c = ((steps_g + max_existing).saturating_mul(2).max(16)).next_power_of_two() as u32;
			for _ in 0..num_neurons {
				off.push(total as u32);
				cap.push(c);
				total += c as u64;
			}
		}
		let total = total as usize;
		// markers EMPTY=0; keys read only when FINAL; values 2 = EMPTY hover sentinel
		// (the kernel's HOST CONTRACT).
		let mut t = SlotTable {
			off, cap,
			markers: vec![0u32; total],
			keys: vec![0u64; total],
			values: vec![2u32; total],
		};
		if let Some(exports) = seed {
			for gi in 0..g {
				for n in 0..num_neurons {
					let gn = gi * num_neurons + n;
					let (o, c) = (t.off[gn] as usize, t.cap[gn] as usize);
					let e = &exports[gi];
					let (e0, ec) = (e.offsets[n] as usize, e.counts[n] as usize);
					for i in 0..ec {
						let (addr, val) = (e.keys[e0 + i], e.values[e0 + i] as u32);
						let slot = host_seed_slot(&t.markers, o, c, addr);
						t.markers[slot] = MARKER_FINAL_U32;
						t.keys[slot] = addr;
						t.values[slot] = val;
					}
				}
			}
		}
		t
	}

	/// Cells a (genome, neuron) region holds — the readback shape the CPU compares
	/// against. Only FINAL slots carry a real cell.
	fn entries(&self, gn: usize) -> Vec<(u64, u8)> {
		let (o, c) = (self.off[gn] as usize, self.cap[gn] as usize);
		(o..o + c)
			.filter(|&s| self.markers[s] == MARKER_FINAL_U32)
			.map(|s| (self.keys[s], (self.values[s] & 0xFF) as u8))
			.collect()
	}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct TrainParams {
	num_genomes: u32, n_state: u32, sbpn: u32, obpn: u32,
	num_motors: u32, levels: u32, bpf: u32, window: u32,
	frame_bits: u32, sensor_total: u32, num_features: u32,
	obs_tilt_p: u32, obs_tilt_i: u32, obs_peraxis_p: u32, obs_peraxis_i: u32, obs_peraxis_yaw: u32, obs_pwm: u32,
	obs_yaw_err: u32, obs_yaw_err_i: u32,
	integral_leak: f32, integral_scale: f32,
	dt: f32,   // yaw-anchor: gyro-z integration step (train/record recompute yaw_heading)
	decouple_outputs: u32, delta_control: u32, selective: u32,
	target0: f32, target1: f32, target2: f32,
	// Action-repeat N (arm R): trainer re-forward decision mask. APPENDED at the
	// END — layout must match the Metal TrainParams struct exactly.
	action_repeat: u32,
	// Memory mode (ABI 12): decode/fire-bit/nudge semantics. APPENDED at END.
	memory_mode: u32,
	// Output decode topology (03/08/2026). APPENDED at END.
	output_decode: u32,
	// L1 d̂ observer (06/08/2026) — APPENDED at END of BOTH TrainParams structs.
	dhat_on: u32,
	dhat_b: [f32; 3],
	dhat_l_gain: f32,
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
	// Yaw-anchor: per-episode initial quaternion (w,x,y,z), [num_episodes*4]. The
	// train/record kernels derive init_yaw = yaw_from_quat(init_q[ep]) to seed the
	// yaw heading — they have no q0 otherwise (they replay recorded sensor traces).
	pub init_q: &'a [f32],
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
	mht_populate_pipeline: ComputePipelineState, // controller_mht_populate (P5b)
	mht_probe_pipeline: ComputePipelineState,    // controller_mht_probe (P5b)
	state_commit_pipeline: ComputePipelineState, // controller_state_commit — bptt section (c)
	nudge_dist_pipeline: ComputePipelineState,   // controller_nudge_distance_probe — solver cost twin
	proj_addr_pipeline: ComputePipelineState,    // controller_projected_address_probe — MSB-first twin
	cand_rank_pipeline: ComputePipelineState,    // controller_candidate_rank_probe — phase-1 ordering key
	phase1_pipeline: ComputePipelineState,       // controller_phase1_topk — reachable enumeration
	beam_pipeline: ComputePipelineState,         // controller_beam_search — phase 2
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
	state_words: u32,   // stride into PACKED state_ins
}

// Same layout as the Metal CountParams (P3 Type-2 window-count primitive).
#[repr(C)]
#[derive(Clone, Copy)]
struct CountParams {
	num_conflicts: u32,
	num_bits: u32,
	state_words: u32,   // stride into PACKED state_ins
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
	on_val: u32,  // planted cell for on(a) — cell_mode::plant_cell(true, mode)
	off_val: u32, // planted cell for !on(a) — cell_mode::plant_cell(false, mode)
}

// Same layout as the Metal MhtParams (P5b resident-cell read path).
#[repr(C)]
#[derive(Clone, Copy)]
struct MhtParams {
	slot_cap: u32,
	num_cells: u32,
	sorted_count: u32,
	num_q: u32,
}

/// Resident GPU record buffers from record_dispatch (P5a) — the packed out_ins /
/// state_ins / pid_pwm the scan + search consume WITHOUT a host round-trip.
struct RecordBuffers {
	b_ro: Buffer,   // rec_out_ins   [total_steps * out_words]
	b_rs: Buffer,   // rec_state_ins [total_steps * state_words]
	b_rp: Buffer,   // rec_pwm       [total_steps * 4]
	total_steps: usize,
	out_words: usize,
	state_words: usize,
	out_input_len: usize,
	state_input_len: usize,
	n_state: usize,
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

pub(crate) struct SolveLayer<'a> {
	pub keys: &'a [u64],
	pub values: &'a [u8],
	pub offsets: &'a [u32],
	pub counts: &'a [u32],
	pub conns: &'a [i64],          // instance-major, neuron-minor
	pub num_inst: usize,
	pub neurons_per_inst: usize,
	pub n_bits: usize,
	pub total_input_bits: usize,
	pub input_bits: &'a [bool],    // shared across this layer's instances
	pub target_bits: &'a [bool],   // [num_inst * neurons_per_inst]
	pub n_immutable_bits: usize,
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
			// Empty constant set: these kernels don't reference the Phase-1
			// function constants, but specialized creation is harmless and
			// future-proofs against Metal's strict validation.
			let func = library.get_function(name, Some(metal::FunctionConstantValues::new()))
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
		let mht_populate_pipeline = mk("controller_mht_populate")?;
		let mht_probe_pipeline = mk("controller_mht_probe")?;
		let state_commit_pipeline = mk("controller_state_commit")?;
		let nudge_dist_pipeline = mk("controller_nudge_distance_probe")?;
		let proj_addr_pipeline = mk("controller_projected_address_probe")?;
		let cand_rank_pipeline = mk("controller_candidate_rank_probe")?;
		let phase1_pipeline = mk("controller_phase1_topk")?;
		let beam_pipeline = mk("controller_beam_search")?;

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
			sep_walk_pipeline, sep_counts_pipeline, sep_bidir_pipeline, plant_table_pipeline, plant_bidir_pipeline,
			mht_populate_pipeline, mht_probe_pipeline, state_commit_pipeline, nudge_dist_pipeline, proj_addr_pipeline, cand_rank_pipeline, phase1_pipeline, beam_pipeline })
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

	/// Train output cells on the GPU from an EMPTY table (round-1 semantics).
	/// Returns, per (genome, neuron), the sorted trained (address, cell) entries —
	/// One layer's worth of independent solves: `num_inst` instances of
	/// `neurons_per_inst` neurons each, sharing an export and a conn array that is
	/// already instance-major / neuron-minor.
	///
	/// A struct rather than fifteen positional arguments, and instance-COUNTED rather
	/// than motor-specific, because the next batching step (genomes) adds instances to
	/// exactly this axis — 50 genomes x 4 motors is one layer with num_inst=200, not a
	/// new entry point.

	/// Solve SEVERAL layers in ONE command buffer — one CPU↔GPU sync for all of them.
	///
	/// The sync is the cost. Measured on the batched motor path, dispatch+sync is ~81%
	/// of GPU time (bbe70f50) and ~0.9 ms is an order of magnitude above Metal's launch
	/// overhead, i.e. it is pipeline flushes, not compute. Records cannot stop waiting —
	/// record d's result feeds d-1 — so the only lever is putting MORE WORK inside each
	/// wait.
	///
	/// The walk's (a) per-motor output solve and (b) state transition solve are
	/// independent within a record: (b) consumes d_next from record d+1, not (a)'s
	/// output, and they meet only in the aggregation afterwards. So both belong in one
	/// command buffer, halving syncs per record.
	pub(crate) fn solve_layers(
		&self,
		layers: &[SolveLayer],
		topk_per_neuron: usize,
		memory_mode: u8,
	) -> Result<Vec<Vec<Option<Vec<bool>>>>, String> {
		let cmd = self.queue.new_command_buffer();
		// Buffers must outlive the encoders, so they are held here until the wait.
		let mut held: Vec<(Buffer, Buffer, usize, usize, usize)> = Vec::with_capacity(layers.len());

		for l in layers {
			let k_top = topk_per_neuron.min(1usize << l.n_bits.min(31)).min(8);
			let all_neurons = l.num_inst * l.neurons_per_inst;
			let conns_i32: Vec<i32> = l.conns.iter().map(|&c| c as i32).collect();
			let ib: Vec<u8> = l.input_bits.iter().map(|&b| b as u8).collect();
			let tb: Vec<u8> = l.target_bits.iter().map(|&b| b as u8).collect();
			let mut ib_rep = Vec::with_capacity(l.num_inst * l.total_input_bits);
			for _ in 0..l.num_inst { ib_rep.extend_from_slice(&ib); }

			let (b_k, b_v) = (self.buf(l.keys), self.buf(l.values));
			let (b_of, b_cn) = (self.buf(l.offsets), self.buf(l.counts));
			let (b_co, b_ib, b_tb) = (self.buf(&conns_i32), self.buf(&ib), self.buf(&tb));
			let b_oa = self.buf(&vec![0u64; all_neurons * k_top]);
			let b_or = self.buf(&vec![0u32; all_neurons * k_top]);
			let b_oc = self.buf(&vec![0u32; all_neurons]);
			let p1: [u32; 4] = [all_neurons as u32, l.n_bits as u32, k_top as u32, memory_mode as u32];
			let b_p1 = self.device.new_buffer_with_data(p1.as_ptr() as *const _, 16, MTLResourceOptions::StorageModeShared);
			let enc = cmd.new_compute_command_encoder();
			enc.set_compute_pipeline_state(&self.phase1_pipeline);
			for (i, b) in [&b_k, &b_v, &b_of, &b_cn, &b_co, &b_ib, &b_tb, &b_oa, &b_or, &b_oc, &b_p1]
				.iter().enumerate() { enc.set_buffer(i as u64, Some(b), 0); }
			let tg = self.phase1_pipeline.max_total_threads_per_threadgroup().min(all_neurons as u64).max(1);
			enc.dispatch_threads(MTLSize::new(all_neurons as u64, 1, 1), MTLSize::new(tg, 1, 1));
			enc.end_encoding();

			let b_ibr = self.buf(&ib_rep);
			let b_scr = self.buf(&vec![0u32; l.num_inst * 2 * 64 * 16]);
			let b_ob = self.buf(&vec![0u8; l.num_inst * l.total_input_bits]);
			let b_ok = self.buf(&vec![0u32; l.num_inst]);
			let p2: [u32; 4] = [l.num_inst as u32, l.neurons_per_inst as u32, l.n_bits as u32, k_top as u32];
			let q2: [u32; 2] = [l.total_input_bits as u32, l.n_immutable_bits as u32];
			let b_p2 = self.device.new_buffer_with_data(p2.as_ptr() as *const _, 16, MTLResourceOptions::StorageModeShared);
			let b_q2 = self.device.new_buffer_with_data(q2.as_ptr() as *const _, 8, MTLResourceOptions::StorageModeShared);
			let enc2 = cmd.new_compute_command_encoder();
			enc2.set_compute_pipeline_state(&self.beam_pipeline);
			for (i, b) in [&b_oa, &b_or, &b_oc, &b_co, &b_ibr, &b_scr, &b_ob, &b_ok, &b_p2, &b_q2]
				.iter().enumerate() { enc2.set_buffer(i as u64, Some(b), 0); }
			let tg2 = self.beam_pipeline.max_total_threads_per_threadgroup().min(l.num_inst as u64).max(1);
			enc2.dispatch_threads(MTLSize::new(l.num_inst as u64, 1, 1), MTLSize::new(tg2, 1, 1));
			enc2.end_encoding();

			held.push((b_ob, b_ok, l.num_inst, l.total_input_bits, 0));
		}

		cmd.commit();
		cmd.wait_until_completed();   // ONE sync for every layer
		if cmd.status() != MTLCommandBufferStatus::Completed {
			return Err(format!("solve_layers failed: {:?}", cmd.status()));
		}
		Ok(held.iter().map(|(b_ob, b_ok, n_inst, tib, _)| {
			let ok = unsafe { std::slice::from_raw_parts(b_ok.contents() as *const u32, *n_inst) };
			let bits = unsafe { std::slice::from_raw_parts(b_ob.contents() as *const u8, n_inst * tib) };
			(0..*n_inst).map(|i| {
				if ok[i] == 0 { None }
				else { Some(bits[i * tib..(i + 1) * tib].iter().map(|&b| b != 0).collect()) }
			}).collect()
		}).collect())
	}

	/// BATCHED solve — every motor of one record in TWO dispatches instead of 2 per motor.
	///
	/// This is the only batching the algorithm admits at this level. Records are strictly
	/// sequential (record d's commits are read by d-1's solve), but the motors WITHIN a
	/// record are independent: each addresses its own bank and they combine only
	/// afterwards, in the vote. Batching them is therefore free of any ordering question.
	///
	/// The layouts already suit it and nothing has to be rearranged:
	///   * all motors share `input_bits` (the record's output-layer input),
	///   * `output_connections` is already motor-major / neuron-minor, so it IS the
	///     batched conn array — instance m, neuron n lives at (m*levels + n)*obpn,
	///   * the whole output layer exports once; motor m simply owns neurons
	///     [m*levels, (m+1)*levels).
	///
	/// Phase 1 needs no instance concept at all — every neuron's top-k is independent, so
	/// one dispatch covers all `num_motors * levels` of them. Phase 2 runs `num_motors`
	/// beam instances, one per motor.
	pub(crate) fn solve_qsr_reachable_motors(
		&self,
		keys: &[u64], values: &[u8], offsets: &[u32], counts: &[u32],
		connections: &[i64],       // full output_connections, motor-major
		num_motors: usize,
		levels: usize,             // neurons per motor
		n_bits: usize,
		total_input_bits: usize,
		input_bits: &[bool],       // shared by every motor
		target_bits: &[bool],      // [num_motors * levels], motor-major
		n_immutable_bits: usize,
		topk_per_neuron: usize,
		memory_mode: u8,
	) -> Result<Vec<Option<Vec<bool>>>, String> {
		let k_top = topk_per_neuron.min(1usize << n_bits.min(31)).min(8);
		let all_neurons = num_motors * levels;
		let conns_i32: Vec<i32> = connections.iter().map(|&c| c as i32).collect();
		let ib: Vec<u8> = input_bits.iter().map(|&b| b as u8).collect();
		let tb: Vec<u8> = target_bits.iter().map(|&b| b as u8).collect();
		// The beam reads input_bits at a per-instance stride; every motor wants the same
		// record, so it is replicated rather than special-casing a zero stride in-kernel.
		let mut ib_rep = Vec::with_capacity(num_motors * total_input_bits);
		for _ in 0..num_motors { ib_rep.extend_from_slice(&ib); }

		let (b_k, b_v) = (self.buf(keys), self.buf(values));
		let (b_of, b_cn) = (self.buf(offsets), self.buf(counts));
		let (b_co, b_ib, b_tb) = (self.buf(&conns_i32), self.buf(&ib), self.buf(&tb));
		let b_oa = self.buf(&vec![0u64; all_neurons * k_top]);
		let b_or = self.buf(&vec![0u32; all_neurons * k_top]);
		let b_oc = self.buf(&vec![0u32; all_neurons]);
		// Phase 1: ALL motors' neurons in ONE dispatch.
		let p1: [u32; 4] = [all_neurons as u32, n_bits as u32, k_top as u32, memory_mode as u32];
		let b_p1 = self.device.new_buffer_with_data(p1.as_ptr() as *const _, 16, MTLResourceOptions::StorageModeShared);
		let cmd = self.queue.new_command_buffer();
		let enc = cmd.new_compute_command_encoder();
		enc.set_compute_pipeline_state(&self.phase1_pipeline);
		for (i, b) in [&b_k, &b_v, &b_of, &b_cn, &b_co, &b_ib, &b_tb, &b_oa, &b_or, &b_oc, &b_p1]
			.iter().enumerate() { enc.set_buffer(i as u64, Some(b), 0); }
		let tg1 = self.phase1_pipeline.max_total_threads_per_threadgroup().min(all_neurons as u64).max(1);
		enc.dispatch_threads(MTLSize::new(all_neurons as u64, 1, 1), MTLSize::new(tg1, 1, 1));
		enc.end_encoding();

		// Phase 2: one beam instance per motor, same command buffer.
		let b_ibr = self.buf(&ib_rep);
		let b_scr = self.buf(&vec![0u32; num_motors * 2 * 64 * 16]);
		let b_ob = self.buf(&vec![0u8; num_motors * total_input_bits]);
		let b_ok = self.buf(&vec![0u32; num_motors]);
		let p2: [u32; 4] = [num_motors as u32, levels as u32, n_bits as u32, k_top as u32];
		let q2: [u32; 2] = [total_input_bits as u32, n_immutable_bits as u32];
		let b_p2 = self.device.new_buffer_with_data(p2.as_ptr() as *const _, 16, MTLResourceOptions::StorageModeShared);
		let b_q2 = self.device.new_buffer_with_data(q2.as_ptr() as *const _, 8, MTLResourceOptions::StorageModeShared);
		let enc2 = cmd.new_compute_command_encoder();
		enc2.set_compute_pipeline_state(&self.beam_pipeline);
		for (i, b) in [&b_oa, &b_or, &b_oc, &b_co, &b_ibr, &b_scr, &b_ob, &b_ok, &b_p2, &b_q2]
			.iter().enumerate() { enc2.set_buffer(i as u64, Some(b), 0); }
		enc2.dispatch_threads(MTLSize::new(num_motors as u64, 1, 1), MTLSize::new(num_motors as u64, 1, 1));
		enc2.end_encoding();
		cmd.commit();
		cmd.wait_until_completed();
		if cmd.status() != MTLCommandBufferStatus::Completed {
			return Err(format!("solve_qsr_reachable_motors failed: {:?}", cmd.status()));
		}
		let ok = unsafe { std::slice::from_raw_parts(b_ok.contents() as *const u32, num_motors) };
		let bits = unsafe { std::slice::from_raw_parts(b_ob.contents() as *const u8, num_motors * total_input_bits) };
		Ok((0..num_motors).map(|m| {
			if ok[m] == 0 { None }
			else { Some(bits[m * total_input_bits..(m + 1) * total_input_bits].iter().map(|&b| b != 0).collect()) }
		}).collect())
	}

	/// GPU twin of `solve_partial_connectivity_qsr_reachable` — phase 1 + phase 2 for ONE
	/// solve. Same signature shape as the CPU function so the walk can swap between them.
	///
	/// `entries` is the neuron-major sorted export of the layer being solved against
	/// (keys/values/offsets/counts), which is exactly what `SparseLayerMemory::
	/// export_for_gpu` produces.
	pub(crate) fn solve_qsr_reachable(
		&self,
		keys: &[u64], values: &[u8], offsets: &[u32], counts: &[u32],
		connections: &[i64],
		num_neurons: usize,
		n_bits: usize,
		total_input_bits: usize,
		input_bits: &[bool],
		target_bits: &[bool],
		n_immutable_bits: usize,
		topk_per_neuron: usize,
		memory_mode: u8,
	) -> Result<Option<Vec<bool>>, String> {
		let k_top = topk_per_neuron.min(1usize << n_bits.min(31)).min(8);
		let conns_i32: Vec<i32> = connections.iter().map(|&c| c as i32).collect();
		let ib: Vec<u8> = input_bits.iter().map(|&b| b as u8).collect();
		let tb: Vec<u8> = target_bits.iter().map(|&b| b as u8).collect();

		// --- phase 1 ---------------------------------------------------------------
		let (b_k, b_v) = (self.buf(keys), self.buf(values));
		let (b_of, b_cn) = (self.buf(offsets), self.buf(counts));
		let (b_co, b_ib, b_tb) = (self.buf(&conns_i32), self.buf(&ib), self.buf(&tb));
		let b_oa = self.buf(&vec![0u64; num_neurons * k_top]);
		let b_or = self.buf(&vec![0u32; num_neurons * k_top]);
		let b_oc = self.buf(&vec![0u32; num_neurons]);
		let p1: [u32; 4] = [num_neurons as u32, n_bits as u32, k_top as u32, memory_mode as u32];
		let b_p1 = self.device.new_buffer_with_data(
			p1.as_ptr() as *const _, 16, MTLResourceOptions::StorageModeShared);
		let cmd = self.queue.new_command_buffer();
		let enc = cmd.new_compute_command_encoder();
		enc.set_compute_pipeline_state(&self.phase1_pipeline);
		for (i, b) in [&b_k, &b_v, &b_of, &b_cn, &b_co, &b_ib, &b_tb, &b_oa, &b_or, &b_oc, &b_p1]
			.iter().enumerate() { enc.set_buffer(i as u64, Some(b), 0); }
		let tg1 = self.phase1_pipeline.max_total_threads_per_threadgroup().min(num_neurons as u64).max(1);
		enc.dispatch_threads(MTLSize::new(num_neurons as u64, 1, 1), MTLSize::new(tg1, 1, 1));
		enc.end_encoding();

		// --- phase 2 (same command buffer — no host round-trip between stages) ------
		let b_scr = self.buf(&vec![0u32; 2 * 64 * 16]);
		let b_ob = self.buf(&vec![0u8; total_input_bits]);
		let b_ok = self.buf(&vec![0u32; 1]);
		let p2: [u32; 4] = [1, num_neurons as u32, n_bits as u32, k_top as u32];
		let q2: [u32; 2] = [total_input_bits as u32, n_immutable_bits as u32];
		let b_p2 = self.device.new_buffer_with_data(p2.as_ptr() as *const _, 16, MTLResourceOptions::StorageModeShared);
		let b_q2 = self.device.new_buffer_with_data(q2.as_ptr() as *const _, 8, MTLResourceOptions::StorageModeShared);
		let enc2 = cmd.new_compute_command_encoder();
		enc2.set_compute_pipeline_state(&self.beam_pipeline);
		for (i, b) in [&b_oa, &b_or, &b_oc, &b_co, &b_ib, &b_scr, &b_ob, &b_ok, &b_p2, &b_q2]
			.iter().enumerate() { enc2.set_buffer(i as u64, Some(b), 0); }
		enc2.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
		enc2.end_encoding();
		cmd.commit();
		cmd.wait_until_completed();
		if cmd.status() != MTLCommandBufferStatus::Completed {
			return Err(format!("solve_qsr_reachable failed: {:?}", cmd.status()));
		}
		let ok = unsafe { *(b_ok.contents() as *const u32) };
		if ok == 0 { return Ok(None); }
		let bits = unsafe { std::slice::from_raw_parts(b_ob.contents() as *const u8, total_input_bits) };
		Ok(Some(bits.iter().map(|&b| b != 0).collect()))
	}

	/// GPU twin of the bptt walk's section (c): apply `commits` IN ORDER to a resident,
	/// writable STATE table, returning each (genome, neuron) region's resulting cells.
	///
	/// Order is semantic — nudge is a read-modify-write, so the kernel puts exactly ONE
	/// thread on each (genome, neuron) region and that thread replays the list in index
	/// order. Concurrency across regions is free because regions are disjoint.
	pub(crate) fn state_commit(
		&self,
		table: &SlotTable,
		commits: &[StateCommit],
		num_genomes: usize,
		n_state: usize,
		memory_mode: u8,
	) -> Result<Vec<Vec<(u64, u8)>>, String> {
		let (cg, cn): (Vec<u32>, Vec<u32>) = commits.iter().map(|c| (c.genome, c.neuron)).unzip();
		let ca: Vec<u64> = commits.iter().map(|c| c.addr).collect();
		let ct: Vec<u8> = commits.iter().map(|c| c.target_true as u8).collect();
		let (b_cg, b_cn, b_ca, b_ct) = (self.buf(&cg), self.buf(&cn), self.buf(&ca), self.buf(&ct));
		let b_mk = self.buf(&table.markers);
		let b_ky = self.buf(&table.keys);
		let b_vl = self.buf(&table.values);
		let (b_off, b_cap) = (self.buf(&table.off), self.buf(&table.cap));
		let writes = vec![0u32; num_genomes];
		let b_w = self.buf(&writes);
		let p = StateCommitParams {
			num_genomes: num_genomes as u32, n_state: n_state as u32,
			num_commits: commits.len() as u32, memory_mode: memory_mode as u32,
		};
		let b_par = self.device.new_buffer_with_data(
			&p as *const _ as *const _, mem::size_of::<StateCommitParams>() as u64,
			MTLResourceOptions::StorageModeShared);

		let cmd = self.queue.new_command_buffer();
		let enc = cmd.new_compute_command_encoder();
		enc.set_compute_pipeline_state(&self.state_commit_pipeline);
		let bufs: [&Buffer; 11] = [&b_cg, &b_cn, &b_ca, &b_ct, &b_mk, &b_ky, &b_vl, &b_off, &b_cap, &b_par, &b_w];
		for (i, b) in bufs.iter().enumerate() { enc.set_buffer(i as u64, Some(b), 0); }
		let total = num_genomes * n_state;
		let tg = self.state_commit_pipeline.max_total_threads_per_threadgroup().min(total as u64).max(1);
		enc.dispatch_threads(MTLSize::new(total as u64, 1, 1), MTLSize::new(tg, 1, 1));
		enc.end_encoding();
		cmd.commit();
		cmd.wait_until_completed();
		if cmd.status() != MTLCommandBufferStatus::Completed {
			return Err(format!("controller_state_commit failed: {:?}", cmd.status()));
		}

		let mk = unsafe { std::slice::from_raw_parts(b_mk.contents() as *const u32, table.markers.len()) };
		let ky = unsafe { std::slice::from_raw_parts(b_ky.contents() as *const u64, table.keys.len()) };
		let vl = unsafe { std::slice::from_raw_parts(b_vl.contents() as *const u32, table.values.len()) };
		Ok((0..total)
			.map(|gn| {
				let (o, c) = (table.off[gn] as usize, table.cap[gn] as usize);
				let mut e: Vec<(u64, u8)> = (o..o + c)
					.filter(|&s| mk[s] == MARKER_FINAL_U32)
					.map(|s| (ky[s], (vl[s] & 0xFF) as u8))
					.collect();
				e.sort_unstable();
				e
			})
			.collect())
	}

	/// the GPU twin of each genome's output_memory after split_retrain_output.
	/// Outer index = g*num_out + n.
	pub fn train(
		&self,
		controllers: &[&WnnController],
		batch: &TrainBatch,
	) -> Result<Vec<Vec<(u64, u8)>>, String> {
		self.train_impl(controllers, batch, false)
	}

	/// Train output cells on the GPU SEEDED from each controller's CURRENT output
	/// cells. split_retrain_output ACCUMULATES (each round's nudge starts from
	/// read_cell of the existing cell), so rounds 2+ must seed the marker table
	/// with the controller's present cells before nudging. The seed is host-side:
	/// replay the existing cells into the marker buffers with the same slot_hash +
	/// linear probe find_or_claim_slot uses, so the kernel finds each and nudges
	/// from its accumulated value. Returns the same per-(genome,neuron) layout.
	pub fn train_seeded(
		&self,
		controllers: &[&WnnController],
		batch: &TrainBatch,
	) -> Result<Vec<Vec<(u64, u8)>>, String> {
		self.train_impl(controllers, batch, true)
	}

	fn train_impl(
		&self,
		controllers: &[&WnnController],
		batch: &TrainBatch,
		seed: bool,
	) -> Result<Vec<Vec<(u64, u8)>>, String> {
		let g = controllers.len();
		if g == 0 { return Ok(vec![]); }
		let (num_motors, levels, n_state, sbpn, obpn, bpf, window) = controllers[0].gpu_dims();
		let (num_features, obs_tilt_p, obs_tilt_i, obs_peraxis_p, obs_peraxis_i,
		     obs_pwm, integral_leak, integral_scale, decouple_outputs,
		     obs_peraxis_yaw, obs_yaw_err, obs_yaw_err_i, ctrl_dt) = controllers[0].obs_params();
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
		// When seeding (round 2+), the region must ALSO hold the genome's existing
		// output cells, so size it for (seeded + new) with the worst-case neuron.
		let steps_per_genome: Vec<u64> = (0..g)
			.map(|gi| {
				let (e0, ne) = (batch.ep_base[gi] as usize, batch.ep_count[gi] as usize);
				(e0..e0 + ne).map(|ep| batch.step_count[ep] as u64).sum()
			})
			.collect();
		let seed_exports: Option<Vec<_>> = seed.then(|| {
			controllers.iter().map(|c| { let (_, _, _, oexp) = c.gpu_export(); oexp }).collect()
		});
		let table = SlotTable::build(&steps_per_genome, num_out, seed_exports.as_deref());
		let SlotTable { off: slot_off, cap: slot_cap, markers, keys, values } = table;
		let total_slots = markers.len();

		let p = TrainParams {
			num_genomes: g as u32, n_state: n_state as u32, sbpn: sbpn as u32, obpn: obpn as u32,
			num_motors: num_motors as u32, levels: levels as u32, bpf: bpf as u32, window: window as u32,
			frame_bits: frame_bits as u32, sensor_total: sensor_total as u32, num_features: num_features as u32,
			obs_tilt_p: obs_tilt_p as u32, obs_tilt_i: obs_tilt_i as u32,
			obs_peraxis_p: obs_peraxis_p as u32, obs_peraxis_i: obs_peraxis_i as u32, obs_peraxis_yaw: obs_peraxis_yaw as u32, obs_pwm: obs_pwm as u32,
			obs_yaw_err: obs_yaw_err as u32, obs_yaw_err_i: obs_yaw_err_i as u32,
			integral_leak, integral_scale,
			dt: ctrl_dt,
			decouple_outputs: decouple_outputs as u32, delta_control: if delta_control { 1 } else { 0 },
			selective: if batch.selective { 1 } else { 0 },
			target0: batch.target_rpy[0], target1: batch.target_rpy[1], target2: batch.target_rpy[2],
			action_repeat: controllers[0].action_repeat_n() as u32,
			memory_mode: controllers[0].memory_mode_u8() as u32,
				output_decode: controllers[0].output_decode_u8() as u32,
			// L1 d̂ — from the controller, same source as the score path.
			dhat_on: if controllers[0].dhat_params().is_some() { 1 } else { 0 },
			dhat_b: controllers[0].dhat_params().map_or([0.0; 3], |(b, _)| b),
			dhat_l_gain: controllers[0].dhat_params().map_or(0.05, |(_, g)| g),
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
		let b_iy = self.buf(batch.init_q);   // yaw-anchor: per-episode q0 (buffer 22)

		let cmd = self.queue.new_command_buffer();
		let enc = cmd.new_compute_command_encoder();
		enc.set_compute_pipeline_state(&self.pipeline);
		let bufs: [&Buffer; 23] = [
			&b_sc, &b_oc, &b_sk, &b_sv, &b_so, &b_scn, &b_th,
			&b_epb, &b_epc, &b_stb, &b_stc, &b_gy, &b_ac, &b_tg, &b_pp,
			&b_mk, &b_ky, &b_vl, &b_soff, &b_scap, &b_par, &b_wr, &b_iy,
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

	/// P2/P5a: GPU split_record dispatch — forward-rolls the batch and emits the
	/// packed records to GPU buffers, returning them RESIDENT (no readback). The
	/// public `record()` reads them back; `record_and_scan()` (P5a) feeds them
	/// straight to the scan kernel.
	fn record_dispatch(&self, controllers: &[&WnnController], batch: &TrainBatch) -> Result<RecordBuffers, String> {
		let g = controllers.len();
		let (num_motors, levels, n_state, sbpn, obpn, bpf, window) = controllers[0].gpu_dims();
		let (num_features, obs_tilt_p, obs_tilt_i, obs_peraxis_p, obs_peraxis_i,
		     obs_pwm, integral_leak, integral_scale, decouple_outputs,
		     obs_peraxis_yaw, obs_yaw_err, obs_yaw_err_i, ctrl_dt) = controllers[0].obs_params();
		let (delta_control, _dmax, _dleak) = controllers[0].delta_params();
		let _ = (num_motors, levels, obpn);
		let frame_bits = num_features * bpf;
		let sensor_total = window * frame_bits;
		let out_input_len = frame_bits + n_state;
		let state_input_len = sensor_total + n_state;
		let out_words = (out_input_len + 31) / 32;
		let state_words = (state_input_len + 31) / 32;
		// Action-repeat: records exist only at DECISION steps → the record layout
		// is decision-space. rec_base[ep] = first record index per episode (prefix
		// sums of ceil(T/N)); the kernel writes rec = rec_base[ep] + decision_idx.
		// N=1 ⇒ rec_base == step_base and total_records == total_steps (identical).
		let action_repeat = controllers[0].action_repeat_n().max(1);
		let mut rec_base: Vec<u32> = Vec::with_capacity(batch.step_count.len());
		let mut total_records: usize = 0;
		for &s in batch.step_count.iter() {
			rec_base.push(total_records as u32);
			total_records += (s as usize).div_ceil(action_repeat);
		}

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
			obs_peraxis_p: obs_peraxis_p as u32, obs_peraxis_i: obs_peraxis_i as u32, obs_peraxis_yaw: obs_peraxis_yaw as u32, obs_pwm: obs_pwm as u32,
			obs_yaw_err: obs_yaw_err as u32, obs_yaw_err_i: obs_yaw_err_i as u32,
			integral_leak, integral_scale,
			dt: ctrl_dt,
			decouple_outputs: decouple_outputs as u32, delta_control: if delta_control { 1 } else { 0 },
			selective: 0, target0: batch.target_rpy[0], target1: batch.target_rpy[1], target2: batch.target_rpy[2],
			action_repeat: action_repeat as u32,
			memory_mode: controllers[0].memory_mode_u8() as u32,
				output_decode: controllers[0].output_decode_u8() as u32,
			// L1 d̂ — from the controller, same source as the score path.
			dhat_on: if controllers[0].dhat_params().is_some() { 1 } else { 0 },
			dhat_b: controllers[0].dhat_params().map_or([0.0; 3], |(b, _)| b),
			dhat_l_gain: controllers[0].dhat_params().map_or(0.05, |(_, g)| g),
		};

		let rec_out = vec![0u32; total_records * out_words];
		let rec_state = vec![0u32; total_records * state_words];
		let rec_pwm = vec![0f32; total_records * 4];

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
		let b_iy = self.buf(batch.init_q);   // yaw-anchor: per-episode q0 (buffer 18)
		let b_rb = self.buf(&rec_base);      // action-repeat: per-episode record base (buffer 19)

		let cmd = self.queue.new_command_buffer();
		let enc = cmd.new_compute_command_encoder();
		enc.set_compute_pipeline_state(&self.record_pipeline);
		let bufs: [&Buffer; 20] = [
			&b_sc, &b_sk, &b_sv, &b_so, &b_scn, &b_th,
			&b_epb, &b_epc, &b_stb, &b_stc, &b_gy, &b_ac, &b_tg, &b_pp,
			&b_ro, &b_rs, &b_rp, &b_par, &b_iy, &b_rb,
		];
		for (i, b) in bufs.iter().enumerate() { enc.set_buffer(i as u64, Some(b), 0); }
		let max_ep = batch.ep_count.iter().copied().max().unwrap_or(0) as u64;
		let tw = 8u64.min(g as u64).max(1);
		let th = 8u64.min(max_ep).max(1);
		enc.dispatch_threads(MTLSize::new(g as u64, max_ep, 1), MTLSize::new(tw, th, 1));
		enc.end_encoding();
		cmd.commit();
		cmd.wait_until_completed();

		Ok(RecordBuffers {
			// total_steps is the RECORD count (decision-space; == physical steps at N=1).
			b_ro, b_rs, b_rp, total_steps: total_records, out_words, state_words,
			out_input_len, state_input_len, n_state,
		})
	}

	/// P2: GPU split_record. Returns, per global step (in step_base order), the
	/// (out_ins, state_ins, pid_pwm) record the conflict scan + separator consume.
	pub fn record(
		&self,
		controllers: &[&WnnController],
		batch: &TrainBatch,
	) -> Result<Vec<(Vec<bool>, Vec<bool>, [f32; 4])>, String> {
		if controllers.is_empty() { return Ok(vec![]); }
		let rb = self.record_dispatch(controllers, batch)?;
		let ro = unsafe { std::slice::from_raw_parts(rb.b_ro.contents() as *const u32, rb.total_steps * rb.out_words) };
		let rs = unsafe { std::slice::from_raw_parts(rb.b_rs.contents() as *const u32, rb.total_steps * rb.state_words) };
		let rp = unsafe { std::slice::from_raw_parts(rb.b_rp.contents() as *const f32, rb.total_steps * 4) };
		let unpack = |buf: &[u32], base_w: usize, len: usize| -> Vec<bool> {
			(0..len).map(|pos| (buf[base_w + (pos >> 5)] >> (pos & 31)) & 1 != 0).collect()
		};
		let mut out = Vec::with_capacity(rb.total_steps);
		for r in 0..rb.total_steps {
			out.push((
				unpack(ro, r * rb.out_words, rb.out_input_len),
				unpack(rs, r * rb.state_words, rb.state_input_len),
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
	/// P2b/P5a core: scan a packed records buffer (`b_recs`, [num_records*out_words])
	/// for conflicts. Same as scan() but operates on a buffer that may be RESIDENT
	/// (from record_dispatch) — no host out_ins required. `out_ins` is Some only to
	/// reconstruct the diagnostic coarse key (Conflict.out_in); None leaves it empty
	/// (the resident chain — downstream resolve uses only instances).
	#[allow(clippy::too_many_arguments)]
	fn scan_buffer(
		&self,
		b_recs: &Buffer,
		num_records: usize,
		out_input_len: usize,
		out_words: usize,
		n_state: usize,
		pwms: &[[f32; 4]],
		tau: f32,
		bpf: usize,
		num_features: usize,
		frame_bits: usize,
		target_min: usize,
		out_ins: Option<&[Vec<bool>]>,
	) -> Result<(Vec<ScanConflict>, usize), String> {
		let n = num_records;
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
			let bufs: [&Buffer; 5] = [b_recs, &b_mk, &b_ky, &b_so, &b_par];
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
						out_in: out_ins.map_or_else(Vec::new, |oi| crate::controller_split::coarse_key(
							&oi[idxs[0]], k, bpf, num_features, frame_bits)),
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

	/// P2b: GPU scan_conflicts_coarse. The GPU computes each record's COARSE
	/// `coarse_key` and hash-claims a bucket slot for it (controller_scan kernel);
	/// the host then groups records by slot (ascending index = bit-exact instance
	/// order), computes PWM spread, filters >tau, and runs the adaptive-k loop —
	/// the bit-exact twin of controller_split::scan_conflicts_coarse. `out_ins` are
	/// the per-record output-layer input vectors (length frame_bits + n_state).
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
		self.scan_buffer(&b_recs, n, out_input_len, out_words, n_state, pwms, tau, bpf, num_features, frame_bits, target_min, Some(out_ins))
	}

	/// P5a: resident record→scan chain. Forward-rolls the batch (record_dispatch),
	/// keeps the packed out_ins buffer RESIDENT on the GPU, reads back only the small
	/// pid_pwm (needed host-side for spread), and feeds the resident buffer straight
	/// to the scan kernel — NO out_ins/state_ins round-trip. Returns the conflicts
	/// (instances + spread; out_in left empty — diagnostic only) + chosen_k. Single
	/// controller (the per-genome live call). Bit-exact composition of P2a+P2b.
	#[allow(clippy::too_many_arguments)]
	pub fn record_and_scan(
		&self,
		controller: &WnnController,
		batch: &TrainBatch,
		tau: f32,
		bpf: usize,
		num_features: usize,
		frame_bits: usize,
		target_min: usize,
	) -> Result<(Vec<ScanConflict>, usize), String> {
		let rb = self.record_dispatch(&[controller], batch)?;
		if rb.total_steps == 0 {
			return Ok((Vec::new(), bpf));
		}
		// Only the pwm reads back (small); out_ins stays resident in rb.b_ro.
		let rp = unsafe { std::slice::from_raw_parts(rb.b_rp.contents() as *const f32, rb.total_steps * 4) };
		let pwms: Vec<[f32; 4]> = (0..rb.total_steps).map(|r| [rp[r*4], rp[r*4+1], rp[r*4+2], rp[r*4+3]]).collect();
		self.scan_buffer(&rb.b_ro, rb.total_steps, rb.out_input_len, rb.out_words, rb.n_state,
			&pwms, tau, bpf, num_features, frame_bits, target_min, None)
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
		state_ins_flat: &[u32],
		sil: usize,
		candidate_bits: &[usize],
		max_lags: &[usize],
	) -> Result<Vec<Option<crate::controller_split::Separator>>, String> {
		// Pack host state_ins → resident word layout, then run the buffer core.
		let state_words = (sil + 31) / 32;
		let state_packed = state_ins_flat;   // already in kernel word layout
		let b_state = self.buf(&state_packed);
		self.sep_walk_buffer(&b_state, state_words, conflicts, labels, ep_of, step_of, ep_start, candidate_bits, max_lags)
	}

	/// P5a.2: sep_walk core operating on a PACKED state buffer that may be RESIDENT
	/// (record's b_rs) — no host state_ins round-trip. `b_state` is
	/// [num_records*state_words] packed; everything else as sep_walk.
	#[allow(clippy::too_many_arguments)]
	pub fn sep_walk_buffer(
		&self,
		b_state: &Buffer,
		state_words: usize,
		conflicts: &[Vec<usize>],
		labels: &[Vec<bool>],
		ep_of: &[usize],
		step_of: &[usize],
		ep_start: &[usize],
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
		let b_st = b_state;
		let b_cb = self.buf(&cb_u);
		let b_ml = self.buf(&maxlag_u);
		let b_og = self.buf(&out_correct);
		let b_ol = self.buf(&out_lag);
		let b_oh = self.buf(&out_high);
		let p = SepParams { num_conflicts: c as u32, num_bits: b as u32, state_words: state_words as u32 };
		let b_par = self.device.new_buffer_with_data(
			&p as *const _ as *const _, mem::size_of::<SepParams>() as u64,
			MTLResourceOptions::StorageModeShared);

		let cmd = self.queue.new_command_buffer();
		let enc = cmd.new_compute_command_encoder();
		enc.set_compute_pipeline_state(&self.sep_walk_pipeline);
		let bufs: [&Buffer; 14] = [
			&b_base, &b_cnt, &b_inst, &b_lab, &b_epof, &b_stof, &b_epst, b_st,
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
		state_ins_flat: &[u32],
		sil: usize,
		candidate_bits: &[usize],
		max_lags: &[usize],
		do_bidir: bool,
	) -> Result<Vec<(Option<crate::controller_split::Accumulator>, Option<crate::controller_split::BidirAccumulator>)>, String> {
		// Pack host state_ins → resident word layout, then run the buffer core.
		let state_words = (sil + 31) / 32;
		let state_packed = state_ins_flat;   // already in kernel word layout
		let b_state = self.buf(&state_packed);
		self.accumulator_search_buffer(&b_state, state_words, conflicts, scalars, ep_of, step_of, ep_start, candidate_bits, max_lags, do_bidir)
	}

	/// P5a.2: accumulator_search core on a PACKED state buffer that may be RESIDENT
	/// (record's b_rs) — no host state_ins round-trip.
	#[allow(clippy::too_many_arguments)]
	pub fn accumulator_search_buffer(
		&self,
		b_state: &Buffer,
		state_words: usize,
		conflicts: &[Vec<usize>],
		scalars: &[Vec<f32>],
		ep_of: &[usize],
		step_of: &[usize],
		ep_start: &[usize],
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
		let cb_u: Vec<u32> = candidate_bits.iter().map(|&x| x as u32).collect();
		let maxlag_u: Vec<u32> = max_lags.iter().map(|&x| x as u32).collect();
		let counts = vec![0.0f32; total_counts.max(1)];

		let b_base = self.buf(&conf_inst_base);
		let b_cnt = self.buf(&conf_inst_count);
		let b_inst = self.buf(&conf_inst);
		let b_epof = self.buf(&ep_of_u);
		let b_stof = self.buf(&step_of_u);
		let b_epst = self.buf(&ep_start_u);
		let b_st = b_state;
		let b_cb = self.buf(&cb_u);
		let b_ml = self.buf(&maxlag_u);
		let b_bb = self.buf(&block_base);
		let b_co = self.buf(&counts);
		let p = CountParams { num_conflicts: c as u32, num_bits: b as u32, state_words: state_words as u32 };
		let b_par = self.device.new_buffer_with_data(
			&p as *const _ as *const _, mem::size_of::<CountParams>() as u64,
			MTLResourceOptions::StorageModeShared);

		let cmd = self.queue.new_command_buffer();
		let enc = cmd.new_compute_command_encoder();
		enc.set_compute_pipeline_state(&self.sep_counts_pipeline);
		let bufs: [&Buffer; 12] = [
			&b_base, &b_cnt, &b_inst, &b_epof, &b_stof, &b_epst, b_st,
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
			let bp = CountParams { num_conflicts: c as u32, num_bits: b as u32, state_words: 0 };
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
		state_ins_flat: &[u32],
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
		let state_words = (sil + 31) / 32;
		let num_records = if state_words == 0 { 0 } else { state_ins_flat.len() / state_words };
		let packed = state_ins_flat;   // already in kernel word layout

		// combo bit0=tv@tp, bit1=sv@sp; on = sv==1 || (tv==1)==high_on.
		let mode = controller.memory_mode_u8();
		let combo_vals: Vec<u8> = (0..4).map(|combo| {
			let (tv, sv) = (combo & 1, (combo >> 1) & 1);
			let on = sv == 1 || (tv == 1) == high_on;
			crate::cell_mode::plant_cell(on, mode)
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
		state_ins_flat: &[u32],
		sil: usize,
		trigger: usize,
		max_levels: usize,
		used: &[bool],
	) -> Result<(Option<Vec<usize>>, Vec<usize>, Vec<Vec<(u64, u8)>>), String> {
		let (_nm, _lv, _ns, sbpn, _ob, _bpf, _w) = controller.gpu_dims();
		if sbpn < 2 {
			return Ok((None, Vec::new(), Vec::new()));
		}
		let chain = controller.plant_counter_chain(trigger, max_levels, used);
		if chain.len() < 2 {
			return Ok((None, Vec::new(), Vec::new()));
		}
		let (num_features, ..) = controller.obs_params();
		let (_, _, _, _, _, bpf, window) = controller.gpu_dims();
		let sensor_window = window * num_features * bpf;
		let (sc, _oc, _se, _oe) = controller.gpu_export();
		let state_words = (sil + 31) / 32;
		let num_records = if state_words == 0 { 0 } else { state_ins_flat.len() / state_words };
		let packed = state_ins_flat;   // already in kernel word layout

		let pos = |conns: &[i64], target: usize| -> Option<usize> {
			conns.iter().position(|&x| x as usize == target)
		};
		// `written` stays parallel to `per_neuron`. The position lookups mirror the
		// CPU split_install_counter's `?` EXACTLY: a missing trigger/self/lower bit
		// STOPS the chain (returns None) but KEEPS the levels already planted (CPU
		// has already written them to state_memory before its `?` fires). The caller
		// applies `written`/`per_neuron` regardless, so partial planting matches.
		let mut written: Vec<usize> = Vec::with_capacity(chain.len());
		let mut per_neuron: Vec<Vec<(u64, u8)>> = Vec::with_capacity(chain.len());
		for k in 0..chain.len() {
			let c = chain[k];
			let conns_i64 = &sc[c * sbpn..(c + 1) * sbpn];
			let conns: Vec<i32> = conns_i64.iter().map(|&x| x as i32).collect();
			let tp = match pos(conns_i64, trigger) { Some(p) => p as u32, None => return Ok((None, written, per_neuron)) };
			let sp = match pos(conns_i64, sensor_window + c) { Some(p) => p as u32, None => return Ok((None, written, per_neuron)) };
			let mode = controller.memory_mode_u8();
			let (rel_pos, combo_vals): (Vec<u32>, Vec<u8>) = if k == 0 {
				// level 0 = latch: on = sv || tv (combo bit0=tv@tp, bit1=sv@sp).
				let cv: Vec<u8> = (0..4).map(|combo| {
					let (tv, sv) = (combo & 1, (combo >> 1) & 1);
					crate::cell_mode::plant_cell(sv == 1 || tv == 1, mode)
				}).collect();
				(vec![tp, sp], cv)
			} else {
				// level k>0: on = sv || (tv && lv); combo bit0=tv@tp, bit1=lv@lp, bit2=sv@sp.
				let lp = match pos(conns_i64, sensor_window + chain[k - 1]) { Some(p) => p as u32, None => return Ok((None, written, per_neuron)) };
				let cv: Vec<u8> = (0..8).map(|combo| {
					let (tv, lv, sv) = (combo & 1, (combo >> 1) & 1, (combo >> 2) & 1);
					crate::cell_mode::plant_cell(sv == 1 || (tv == 1 && lv == 1), mode)
				}).collect();
				(vec![tp, lp, sp], cv)
			};
			per_neuron.push(self.plant_table_dispatch(&packed, &conns, sbpn, state_words, num_records, &rel_pos, &combo_vals));
			written.push(c);
		}
		Ok((Some(chain), written, per_neuron))
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
		let mode = controller.memory_mode_u8();
		let p = BidirPlantParams {
			sbpn: sbpn as u32,
			on_val: crate::cell_mode::plant_cell(true, mode) as u32,
			off_val: crate::cell_mode::plant_cell(false, mode) as u32,
		};
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

	/// P5-integrate: GPU twin of WnnController::split_resolve_conflict. Mirrors the
	/// CPU decision exactly — subsample → label → Type-1 (discriminative_walk →
	/// plant_latch); else Type-2 (best-spread motor → bidir then increment
	/// accumulator → plant_counter_bidir / plant_counter) — but the search + planting
	/// run on the GPU, and the planted cells are applied to the controller's
	/// state_memory (the hybrid apply-back). Returns (mode, neurons) like the CPU.
	#[allow(clippy::too_many_arguments)]
	pub fn resolve_conflict_gpu(
		&self,
		controller: &WnnController,
		instances: &[usize],
		pwms: &[[f32; 4]],
		ep_of: &[usize],
		step_of: &[usize],
		ep_start: &[usize],
		state_ins_flat: &[u32],
		sil: usize,
		candidate_bits: &[usize],
		clean_gain: f32,
		accum_corr: f32,
		used: &[bool],
	) -> Result<(i64, Vec<usize>), String> {
		let (num_motors, _lv, n_state, sbpn, _ob, _bpf, _w) = controller.gpu_dims();
		let sampled = crate::controller::subsample_instances(instances, crate::controller::SPLIT_INST_CAP);
		let labels = crate::controller_split::label_high_low(&sampled, pwms);
		let max_lag = sampled.iter().map(|&i| step_of[i]).min().unwrap_or(0).min(crate::controller::SPLIT_LAG_CAP);

		// TYPE-1: discriminative_walk → set/hold latch.
		let seps = self.sep_walk(&[sampled.clone()], &[labels], ep_of, step_of, ep_start, state_ins_flat, sil, candidate_bits, &[max_lag])?;
		if let Some(s) = seps[0].as_ref().filter(|s| s.gain >= clean_gain) {
			let (n_opt, cells) = self.plant_latch(controller, state_ins_flat, sil, s.bit, s.high_on, used)?;
			if let Some(n) = n_opt {
				for (addr, val) in cells { controller.plant_state_cell(n, addr, val); }
				return Ok((1, vec![n]));
			}
		}

		// TYPE-2: disagreeing motor → window-count correlation.
		let mut best_m = 0usize;
		let mut best_s = -1.0f32;
		for m in 0..num_motors {
			let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
			for &i in &sampled { lo = lo.min(pwms[i][m]); hi = hi.max(pwms[i][m]); }
			if hi - lo > best_s { best_s = hi - lo; best_m = m; }
		}
		let scalar: Vec<f32> = sampled.iter().map(|&i| pwms[i][best_m]).collect();

		let do_bidir = sbpn >= 5;
		let accs = self.accumulator_search(&[sampled], &[scalar], ep_of, step_of, ep_start, state_ins_flat, sil, candidate_bits, &[max_lag], do_bidir)?;
		let (acc, bid) = &accs[0];

		// BIDIRECTIONAL (mode 3) first when the chain can hold an up/down counter.
		if do_bidir {
			if let Some(b) = bid.as_ref().filter(|b| b.corr >= accum_corr) {
				let (lv_opt, per_neuron) = self.plant_counter_bidir(controller, b.up, b.dn, n_state, used)?;
				if let Some(levels) = lv_opt {
					for (k, &nn) in levels.iter().enumerate() {
						for &(addr, val) in &per_neuron[k] { controller.plant_state_cell(nn, addr, val); }
					}
					return Ok((3, levels));
				}
			}
		}

		// INCREMENT-only (mode 2). Apply whatever levels were planted (CPU persists
		// partial chains too), then commit as mode 2 only if the full chain succeeded.
		if let Some(a) = acc.as_ref().filter(|a| a.corr >= accum_corr) {
			let (chain_opt, written, per_neuron) = self.plant_counter(controller, state_ins_flat, sil, a.bit, n_state, used)?;
			for (k, &nn) in written.iter().enumerate() {
				for &(addr, val) in &per_neuron[k] { controller.plant_state_cell(nn, addr, val); }
			}
			if let Some(chain) = chain_opt {
				return Ok((2, chain));
			}
		}
		Ok((0, vec![]))
	}

	/// P5 THE END: GPU twin of WnnController::split_train_loop — the full multi-round
	/// state-splitting loop, every phase on GPU. Per round: GPU record → GPU scan →
	/// resolve_conflict_gpu per conflict (committed/k cap + `used` guard) → GPU
	/// train_seeded, applied back so the next round seeds from the accumulated output.
	/// State plants are applied back inside resolve_conflict_gpu. The CPU only
	/// sequences the rounds + holds the small conflict metadata (mirrors the design's
	/// "orchestration stays CPU"). batch.selective is the retrain selective_output.
	/// Returns (rounds_run, conflicts_final, planted_total, committed_per_round).
	#[allow(clippy::too_many_arguments)]
	pub fn split_train_loop_gpu(
		&self,
		controller: &WnnController,
		batch: &TrainBatch,
		tau: f32,
		clean_gain: f32,
		accum_corr: f32,
		max_rounds: usize,
		k_start: usize,
		coarse_target: usize,
	) -> Result<(usize, usize, usize, Vec<usize>), String> {
		let (_nm, _lv, n_state, _sbpn, _ob, bpf, window) = controller.gpu_dims();
		let (num_features, ..) = controller.obs_params();
		let frame_bits = num_features * bpf;
		let sensor_window = window * frame_bits;
		let num_out = {
			let (nm, lv, ..) = controller.gpu_dims();
			nm * lv
		};

		// candidate bits = frame bits (< sensor_window) some state neuron observes.
		let (sc, _oc, _se, _oe) = controller.gpu_export();
		let mut candidate_bits: Vec<usize> = sc.iter().map(|&x| x as usize).filter(|&b| b < sensor_window).collect();
		candidate_bits.sort_unstable();
		candidate_bits.dedup();

		// Episode metadata for this single genome (episode-major record order),
		// reconstructed from the flat batch — the GPU record() returns records in the
		// same rec_base ordering split_record uses. Action-repeat: records exist only
		// at decision steps, so lengths are ceil(T/N) (== T at N=1) and ep_start /
		// step_of are DECISION-space — matching the CPU split_record indexing.
		let action_repeat = controller.action_repeat_n().max(1);
		let e0 = batch.ep_base[0] as usize;
		let ne = batch.ep_count[0] as usize;
		let epl: Vec<usize> = (0..ne).map(|j| (batch.step_count[e0 + j] as usize).div_ceil(action_repeat)).collect();
		let mut ep_start = vec![0usize; ne];
		let mut acc = 0usize;
		for (e, &len) in epl.iter().enumerate() { ep_start[e] = acc; acc += len; }
		let mut ep_of: Vec<usize> = Vec::new();
		let mut step_of: Vec<usize> = Vec::new();
		for (ej, &len) in epl.iter().enumerate() {
			for t in 0..len { ep_of.push(ej); step_of.push(t); }
		}

		let mut used = vec![false; n_state];
		let mut planted_total = 0usize;
		let mut per_round: Vec<usize> = Vec::new();
		let mut rounds_run = 0usize;

		for round in 0..max_rounds {
			// 1+2. GPU record → host out_ins/state_ins/pwm, then GPU scan.
			let recs = self.record(&[controller], batch)?;
			if recs.is_empty() { break; }
			let sil = recs[0].1.len();
			let out_ins: Vec<Vec<bool>> = recs.iter().map(|r| r.0.clone()).collect();
			let pwms: Vec<[f32; 4]> = recs.iter().map(|r| r.2).collect();
			let mut state_ins_bools: Vec<bool> = Vec::with_capacity(recs.len() * sil);
			for r in &recs { state_ins_bools.extend_from_slice(&r.1); }
			let state_ins_flat = crate::controller_split::pack_sif(&state_ins_bools, sil);
			drop(state_ins_bools);

			let (conflicts, _k) = self.scan(&out_ins, &pwms, tau, bpf, num_features, frame_bits, coarse_target)?;
			if conflicts.is_empty() { break; } // converged
			rounds_run = round + 1;

			// 3. resolve up to k(round) conflicts, worst-first, honoring `used`.
			let k = k_start + round; // greedy → batch anneal
			let mut committed = 0usize;
			for c in conflicts.iter() {
				if committed >= k { break; }
				let (mode, neurons) = self.resolve_conflict_gpu(
					controller, &c.instances, &pwms, &ep_of, &step_of, &ep_start,
					&state_ins_flat, sil, &candidate_bits, clean_gain, accum_corr, &used,
				)?;
				if mode != 0 {
					for n in neurons { if n < used.len() { used[n] = true; } }
					committed += 1;
					planted_total += 1;
				}
			}
			per_round.push(committed);
			if committed == 0 { break; } // stalled

			// 4. GPU output retrain (seeded from current cells) → apply back.
			let cells = self.train_seeded(&[controller], batch)?;
			for n in 0..num_out {
				for &(addr, val) in &cells[n] { controller.set_output_cell(n, addr, val); }
			}
		}

		// Final scan for conflicts_final (no retrain — mirrors split_train_loop).
		let recs = self.record(&[controller], batch)?;
		let conflicts_final = if recs.is_empty() {
			0
		} else {
			let out_ins: Vec<Vec<bool>> = recs.iter().map(|r| r.0.clone()).collect();
			let pwms: Vec<[f32; 4]> = recs.iter().map(|r| r.2).collect();
			self.scan(&out_ins, &pwms, tau, bpf, num_features, frame_bits, coarse_target)?.0.len()
		};

		Ok((rounds_run, conflicts_final, planted_total, per_round))
	}

	/// P5b foundation: parity of the read-only MHT cell lookup (mht_lookup) vs the
	/// sorted-array bsearch (bsearch_cell) the forward uses today. Populates an MHT
	/// from `cells`, builds the sorted arrays, then probes both for every query addr
	/// and counts disagreements. This is the cell-read path the resident-cell forward
	/// (P5b) will use; gating it here de-risks the shared-forward change. Returns the
	/// number of mismatching queries.
	pub fn mht_lookup_parity(&self, cells: &[(u64, u8)], queries: &[u64]) -> Result<usize, String> {
		let nc = cells.len();
		let nq = queries.len();
		if nq == 0 { return Ok(0); }

		// Sorted bsearch arrays (the gpu_export format).
		let mut sorted: Vec<(u64, u8)> = cells.to_vec();
		sorted.sort_by_key(|&(a, _)| a);
		let sorted_keys: Vec<u64> = sorted.iter().map(|&(a, _)| a).collect();
		let sorted_vals: Vec<u8> = sorted.iter().map(|&(_, v)| v).collect();
		let cell_addrs: Vec<u64> = cells.iter().map(|&(a, _)| a).collect();
		let cell_vals: Vec<u8> = cells.iter().map(|&(_, v)| v).collect();

		let cap = ((nc.saturating_mul(2)).max(16)).next_power_of_two();
		let markers = vec![0u32; cap];
		let keys = vec![0u64; cap];
		let values = vec![2u32; cap];   // EMPTY=2 default

		let b_ca = self.buf(&cell_addrs);
		let b_cv = self.buf(&cell_vals);
		let b_mk = self.buf(&markers);
		let b_ky = self.buf(&keys);
		let b_vl = self.buf(&values);
		let p = MhtParams { slot_cap: cap as u32, num_cells: nc as u32, sorted_count: nc as u32, num_q: nq as u32 };
		let b_par = self.device.new_buffer_with_data(
			&p as *const _ as *const _, mem::size_of::<MhtParams>() as u64,
			MTLResourceOptions::StorageModeShared);

		// 1) populate the MHT.
		let cmd = self.queue.new_command_buffer();
		let enc = cmd.new_compute_command_encoder();
		enc.set_compute_pipeline_state(&self.mht_populate_pipeline);
		for (i, bf) in [&b_ca, &b_cv, &b_mk, &b_ky, &b_vl, &b_par].iter().enumerate() {
			enc.set_buffer(i as u64, Some(bf), 0);
		}
		let tw = 8u64.min(nc.max(1) as u64).max(1);
		enc.dispatch_threads(MTLSize::new(nc.max(1) as u64, 1, 1), MTLSize::new(tw, 1, 1));
		enc.end_encoding();
		cmd.commit();
		cmd.wait_until_completed();

		// 2) probe both paths.
		let b_sk = self.buf(&sorted_keys);
		let b_sv = self.buf(&sorted_vals);
		let b_q = self.buf(queries);
		let out_bsearch = vec![0u32; nq];
		let out_mht = vec![0u32; nq];
		let b_ob = self.buf(&out_bsearch);
		let b_om = self.buf(&out_mht);
		let cmd2 = self.queue.new_command_buffer();
		let enc2 = cmd2.new_compute_command_encoder();
		enc2.set_compute_pipeline_state(&self.mht_probe_pipeline);
		for (i, bf) in [&b_sk, &b_sv, &b_mk, &b_ky, &b_vl, &b_q, &b_ob, &b_om, &b_par].iter().enumerate() {
			enc2.set_buffer(i as u64, Some(bf), 0);
		}
		let tw2 = 8u64.min(nq as u64).max(1);
		enc2.dispatch_threads(MTLSize::new(nq as u64, 1, 1), MTLSize::new(tw2, 1, 1));
		enc2.end_encoding();
		cmd2.commit();
		cmd2.wait_until_completed();

		let ob = unsafe { std::slice::from_raw_parts(b_ob.contents() as *const u32, nq) };
		let om = unsafe { std::slice::from_raw_parts(b_om.contents() as *const u32, nq) };
		Ok((0..nq).filter(|&q| ob[q] != om[q]).count())
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
	init_q: Vec<f32>,   // per-episode q0 (identity here; fixtures run anchor-off)
}

fn build_parity_fixture(seed_salt: u64) -> Result<ParityFixture, String> {
	build_parity_fixture_mode(seed_salt, 2, None) // QUAD + mode default: the bit-identical anchor
}

/// Mode-parameterized fixture (ABI 12 split-trainer T/B support): identical RNG
/// stream to the QUAD anchor; only the controller's memory_mode, the planted
/// fixture cells' encoding (true_cell), and the output-decode TOPOLOGY differ.
///
/// `output_decode` is None for every historical case, which resolves to
/// cell_mode::default_output_decode and keeps those fixtures bit-identical. Passing
/// Some(DECODE_ANTAGONIST) with a QUAD mode is the 03/08/2026 combination — the one
/// where the shader's `bin_half` and `ctrl_output_target_bit` had to stop keying on
/// memory_mode, and therefore the one a mode-only fixture could never have caught.
fn build_parity_fixture_mode(seed_salt: u64, memory_mode: u8, output_decode: Option<u8>) -> Result<ParityFixture, String> {
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
		false, false, false, false, true, false, false, false, 0.99, 1.0, 0.001,
		true,
		1,   // action_repeat: parity fixtures stay at N=1 (bit-identical anchor)
		memory_mode,
		output_decode,
		None, 0.05,   // dhat_b: parity fixtures keep obs_dhat OFF (bit-identical anchor)
	).map_err(|e| format!("{e}"))?;
	for _ in 0..(n_state * 4) {
		let n = (xs(&mut rng) % n_state as u64) as usize;
		let addr = xs(&mut rng) % (1u64 << sbpn);
		c.plant_state_cell(n, addr, crate::cell_mode::true_cell(memory_mode));
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
	// Identity quaternion per episode (anchor-off fixtures never read this).
	let init_q: Vec<f32> = (0..e_count).flat_map(|_| [1.0f32, 0.0, 0.0, 0.0]).collect();
	Ok(ParityFixture {
		c, num_out, cpu_g, cpu_a, cpu_t, cpu_p, gyros, accels, targets, pids,
		ep_base: vec![0u32], ep_count: vec![e_count as u32], step_base, step_count,
		init_q,
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
		gyros: &f.gyros, accels: &f.accels, targets: &f.targets, pid_pwms: &f.pids, init_q: &f.init_q,
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

/// PyO3: bit-exact parity for the GPU train_seeded (round 2+ accumulating output
/// retrain) vs the CPU split_retrain_output called TWICE. split_retrain_output
/// accumulates (each round's nudge starts from the existing cell), so round 2 on
/// GPU must SEED the marker table from round 1's cells. Sequence: CPU round 1 →
/// GPU train_seeded (reads the round-1 cells) → CPU round 2 → compare round 2.
#[pyfunction]
pub fn run_controller_train_seeded_parity_test() -> Vec<(String, bool, String)> {
	let mut results = Vec::new();
	if Device::system_default().is_none() {
		results.push(("controller_train_seeded_parity".to_string(), true, "skipped: no Metal device".to_string()));
		return results;
	}
	for &selective in &[false, true] {
		match controller_train_seeded_parity_once(selective) {
			Ok((mismatches, seeded, cpu_cells, addrs)) => {
				let name = format!("controller_train_seeded_parity(selective={selective})");
				results.push((name, mismatches == 0,
					format!("addresses={addrs}, round1_seeded_cells={seeded}, round2_cpu_cells={cpu_cells}, mismatches={mismatches}")));
			}
			Err(e) => results.push((format!("controller_train_seeded_parity(selective={selective})"), false, e)),
		}
	}
	results
}

fn controller_train_seeded_parity_once(selective: bool) -> Result<(usize, usize, usize, usize), String> {
	let f = build_parity_fixture(0xA11CE_u64 ^ selective as u64)?;
	let num_out = f.num_out;
	let mut c = f.c;

	let trainer = ControllerTrainer::new()?;
	let batch = TrainBatch {
		ep_base: &f.ep_base, ep_count: &f.ep_count, step_base: &f.step_base, step_count: &f.step_count,
		gyros: &f.gyros, accels: &f.accels, targets: &f.targets, pid_pwms: &f.pids, init_q: &f.init_q,
		selective, target_rpy: [0.0, 0.0, 0.0],
	};

	// ROUND 1 (CPU) — establish the accumulated state the GPU round 2 must seed from.
	let _ = c.split_retrain_output_pub(&f.cpu_g, &f.cpu_a, &f.cpu_t, &f.cpu_p, selective);
	let seeded: usize = (0..num_out).map(|n| c.output_entries(n).len()).sum();

	// ROUND 2 (GPU, SEEDED) — reads c's round-1 cells, returns round-2 GPU cells.
	// Must run before the CPU round 2 mutates c.output_memory.
	let gpu = trainer.train_seeded(&[&c], &batch)?;

	// ROUND 2 (CPU) — accumulates onto round 1 in place.
	let _ = c.split_retrain_output_pub(&f.cpu_g, &f.cpu_a, &f.cpu_t, &f.cpu_p, selective);

	// Compare the cell FUNCTION over the union of touched addresses per neuron.
	let mut mismatches = 0usize;
	let mut cpu_cells = 0usize;
	let mut addrs = 0usize;
	for n in 0..num_out {
		let gpu_entries = &gpu[n];
		let cpu_entries = c.output_entries(n);
		cpu_cells += cpu_entries.iter().filter(|&&(_, v)| v != 2).count();
		let mut all: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
		for &(a, _) in gpu_entries { all.insert(a); }
		for &(a, _) in &cpu_entries { all.insert(a); }
		addrs += all.len();
		let gpu_map: std::collections::HashMap<u64, u8> = gpu_entries.iter().copied().collect();
		for a in all {
			let gv = *gpu_map.get(&a).unwrap_or(&2u8);
			let cv = c.output_cell(n, a);
			if gv != cv { mismatches += 1; }
		}
	}
	Ok((mismatches, seeded, cpu_cells, addrs))
}

/// The split-train parity matrix. ONE definition, shared by the pyfunction and the
/// `cargo test` wrapper, so the two can never drift into testing different things.
///
/// Columns: (selective, coarse_target, memory_mode, output_decode).
/// coarse_target>0 keeps BOTH sides on the coarse scan path (the parity-proven twin);
/// larger targets coarsen harder → surface conflicts → exercise planting + output
/// retrain across the full loop.
///
/// output_decode = None is the mode's historical default, so the first eight rows are
/// the exact pre-03/08/2026 cases. The two QUAD+ANTAGONIST rows are the new
/// combination and the reason this matrix gained a column: they are the only cases
/// where the shader's ctrl_output_target_bit takes the E/I branch under a NON-BINARY
/// mode, which is precisely what the old mode-keyed twin would have got wrong.
const SPLIT_TRAIN_PARITY_CASES: &[(bool, usize, u8, Option<u8>)] = &[
	(false, 8, 2, None), (true, 8, 2, None),
	(false, 12, 2, None), (true, 16, 2, None),
	(false, 8, 0, None), (true, 8, 0, None),
	(false, 8, 3, None), (true, 8, 3, None),
	(false, 8, 2, Some(crate::cell_mode::DECODE_ANTAGONIST)),
	(true, 8, 2, Some(crate::cell_mode::DECODE_ANTAGONIST)),
];

/// PyO3: THE END — full-loop parity for the GPU split_train_loop_gpu vs the CPU
/// split_train_loop. Two identical controllers (same fixture seed) train through
/// the whole multi-round state-splitting loop — one on CPU, one on GPU — and the
/// END-TO-END CPU/GPU parity for the beam solve — phase 1 + phase 2 together.
///
/// This is the acceptance gate for the port of the controller's last CPU island. It runs
/// `controller_phase1_topk` then `controller_beam_search` and compares the SOLVED INPUT
/// BITS against `solve_partial_connectivity_qsr_reachable` — the actual function
/// `bptt_train_window` calls. Not a component, not a re-implementation: the same entry
/// point, the same fixture memory, the same connections.
///
/// It also silently proves the integer-cost claim end to end. The CPU accumulates f64
/// `rank + saturation` across neurons; the GPU accumulates integer ranks only. Matching
/// solved bits over many neurons and modes means the per-neuron saturations really do
/// cancel from every comparison, at every beam step and at the argmin — which is what
/// lets a device without f64 reproduce this at all.
#[pyfunction]
pub fn run_controller_beam_solve_parity_test() -> Vec<(String, bool, String)> {
	let mut results = Vec::new();
	if Device::system_default().is_none() {
		results.push(("controller_beam_solve_parity".to_string(), true, "skipped: no Metal device".to_string()));
		return results;
	}
	let t = match ControllerTrainer::new() {
		Ok(t) => t,
		Err(e) => { results.push(("controller_beam_solve_parity".to_string(), false, e)); return results; }
	};
	const K_TOP: usize = 4;
	for &mode in &[2u8, 0u8, 3u8] {
		let tag = format!("beam_solve_mode{mode}");
		let f = match build_parity_fixture_mode(0xBEA3_0000u64, mode, None) {
			Ok(f) => f,
			Err(e) => { results.push((tag, false, format!("fixture: {e}"))); continue; }
		};
		let n_state = f.c.state_neurons_pub();
		let (sconn, _, sexp, _) = f.c.gpu_export();
		let n_bits = f.c.state_bits_per_neuron_pub();
		let tib = f.c.state_input_len_pub();
		let conns_i32: Vec<i32> = sconn.iter().map(|&c| c as i32).collect();

		// Several independent input/target draws per mode — one solve can agree by luck.
		const INST: usize = 16;
		let mut rng = 0xBEA3_1234_5678_9ABCu64;
		let mut next = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng };
		let mut input_bits = vec![0u8; INST * tib];
		let mut target_bits = vec![0u8; INST * n_state];
		for i in 0..INST {
			for j in 0..tib { input_bits[i * tib + j] = (next() % 2) as u8; }
			for j in 0..n_state { target_bits[i * n_state + j] = (next() % 2) as u8; }
		}

		// --- phase 1, once per instance -------------------------------------------
		let mut tk_addr = vec![0u64; INST * n_state * K_TOP];
		let mut tk_rank = vec![0u32; INST * n_state * K_TOP];
		let mut tk_cnt = vec![0u32; INST * n_state];
		let (b_k, b_v) = (t.buf(&sexp.keys), t.buf(&sexp.values));
		let (b_of, b_cn) = (t.buf(&sexp.offsets), t.buf(&sexp.counts));
		let b_co = t.buf(&conns_i32);
		for i in 0..INST {
			let b_ib = t.buf(&input_bits[i * tib..(i + 1) * tib]);
			let b_tb = t.buf(&target_bits[i * n_state..(i + 1) * n_state]);
			let b_oa = t.buf(&vec![0u64; n_state * K_TOP]);
			let b_or = t.buf(&vec![0u32; n_state * K_TOP]);
			let b_oc = t.buf(&vec![0u32; n_state]);
			let p: [u32; 4] = [n_state as u32, n_bits as u32, K_TOP as u32, mode as u32];
			let b_p = t.device.new_buffer_with_data(
				p.as_ptr() as *const _, 16, MTLResourceOptions::StorageModeShared);
			let cmd = t.queue.new_command_buffer();
			let enc = cmd.new_compute_command_encoder();
			enc.set_compute_pipeline_state(&t.phase1_pipeline);
			for (bi, b) in [&b_k, &b_v, &b_of, &b_cn, &b_co, &b_ib, &b_tb, &b_oa, &b_or, &b_oc, &b_p]
				.iter().enumerate() { enc.set_buffer(bi as u64, Some(b), 0); }
			enc.dispatch_threads(MTLSize::new(n_state as u64, 1, 1), MTLSize::new(n_state as u64, 1, 1));
			enc.end_encoding(); cmd.commit(); cmd.wait_until_completed();
			let a = unsafe { std::slice::from_raw_parts(b_oa.contents() as *const u64, n_state * K_TOP) };
			let r = unsafe { std::slice::from_raw_parts(b_or.contents() as *const u32, n_state * K_TOP) };
			let c = unsafe { std::slice::from_raw_parts(b_oc.contents() as *const u32, n_state) };
			tk_addr[i * n_state * K_TOP..(i + 1) * n_state * K_TOP].copy_from_slice(a);
			tk_rank[i * n_state * K_TOP..(i + 1) * n_state * K_TOP].copy_from_slice(r);
			tk_cnt[i * n_state..(i + 1) * n_state].copy_from_slice(c);
		}

		// --- phase 2, all instances in one dispatch -------------------------------
		let (b_ta, b_tr, b_tc) = (t.buf(&tk_addr), t.buf(&tk_rank), t.buf(&tk_cnt));
		let b_ib = t.buf(&input_bits);
		let b_scr = t.buf(&vec![0u32; INST * 2 * 64 * 16]);
		let b_ob = t.buf(&vec![0u8; INST * tib]);
		let b_ok = t.buf(&vec![0u32; INST]);
		let p: [u32; 4] = [INST as u32, n_state as u32, n_bits as u32, K_TOP as u32];
		let q: [u32; 2] = [tib as u32, 0];
		let b_p = t.device.new_buffer_with_data(p.as_ptr() as *const _, 16, MTLResourceOptions::StorageModeShared);
		let b_q = t.device.new_buffer_with_data(q.as_ptr() as *const _, 8, MTLResourceOptions::StorageModeShared);
		let cmd = t.queue.new_command_buffer();
		let enc = cmd.new_compute_command_encoder();
		enc.set_compute_pipeline_state(&t.beam_pipeline);
		for (bi, b) in [&b_ta, &b_tr, &b_tc, &b_co, &b_ib, &b_scr, &b_ob, &b_ok, &b_p, &b_q]
			.iter().enumerate() { enc.set_buffer(bi as u64, Some(b), 0); }
		enc.dispatch_threads(MTLSize::new(INST as u64, 1, 1), MTLSize::new(INST as u64, 1, 1));
		enc.end_encoding(); cmd.commit(); cmd.wait_until_completed();
		if cmd.status() != MTLCommandBufferStatus::Completed {
			results.push((tag, false, format!("dispatch: {:?}", cmd.status()))); continue;
		}
		let gb = unsafe { std::slice::from_raw_parts(b_ob.contents() as *const u8, INST * tib) };
		let gok = unsafe { std::slice::from_raw_parts(b_ok.contents() as *const u32, INST) };

		// --- CPU reference: the REAL solver bptt_train_window calls ---------------
		let mut bad = 0usize;
		let mut solved_ok = 0usize;   // guards against a vacuous pass: if BOTH sides
		let mut first = String::new(); // returned None everywhere, nothing was compared
		for i in 0..INST {
			let ibits: Vec<bool> = input_bits[i * tib..(i + 1) * tib].iter().map(|&b| b != 0).collect();
			let tbits: Vec<bool> = target_bits[i * n_state..(i + 1) * n_state].iter().map(|&b| b != 0).collect();
			let want = crate::controller_training::solve_partial_connectivity_qsr_reachable(
				|n| f.c.state_entries(n), ram_core::neuron_memory::EMPTY_U8,
				sconn, n_state, n_bits, tib, &ibits, &tbits, 0, K_TOP, mode);
			match (want, gok[i] == 1) {
				(Some(w), true) => {
					solved_ok += 1;
					let g: Vec<bool> = gb[i * tib..(i + 1) * tib].iter().map(|&b| b != 0).collect();
					if w != g {
						bad += 1;
						if first.is_empty() {
							let k = (0..tib).find(|&j| w[j] != g[j]).unwrap_or(0);
							first = format!("inst {i}: first bit diff at {k} (cpu={} gpu={})", w[k], g[k]);
						}
					}
				}
				(None, false) => {}
				(w, ok) => {
					bad += 1;
					if first.is_empty() {
						first = format!("inst {i}: solvability differs — cpu={} gpu={ok}", w.is_some());
					}
				}
			}
		}
		// A sweep where every solve returned None on both sides would report zero
		// mismatches while having compared nothing — require real solutions.
		results.push((tag, bad == 0 && solved_ok > 0,
			if bad == 0 && solved_ok > 0 {
				format!("{solved_ok}/{INST} solves succeeded on BOTH sides x {n_state} neurons: solved input bits identical")
			} else if solved_ok == 0 {
				format!("VACUOUS: no instance solved on both sides ({INST} attempted)")
			} else { format!("{bad}/{INST} differ; first {first}") }));
	}
	results
}

/// CPU/GPU parity for PHASE 1 of the beam solve — reachable-address enumeration.
///
/// This is the expensive half of the walk's section (a): per neuron, offer every trained
/// cell as a candidate, then climb Hamming radii from the projected address collecting
/// UNTRAINED cells until k_top are held, and keep the best k_top by (cost, address).
///
/// The comparison is the SELECTED ADDRESS SEQUENCE, which is what the beam consumes —
/// not the cost values, which are f64 on the CPU and integer on the GPU by design (see
/// candidate_rank). Both sides are driven from the SAME fixture memory and the CPU side
/// calls reachable_topk_for_neuron itself, so this cannot pass against a re-implementation.
#[pyfunction]
pub fn run_controller_phase1_topk_parity_test() -> Vec<(String, bool, String)> {
	let mut results = Vec::new();
	if Device::system_default().is_none() {
		results.push(("controller_phase1_topk_parity".to_string(), true, "skipped: no Metal device".to_string()));
		return results;
	}
	let t = match ControllerTrainer::new() {
		Ok(t) => t,
		Err(e) => { results.push(("controller_phase1_topk_parity".to_string(), false, e)); return results; }
	};
	// QUAD, TERNARY and BINARY differ in nudge_distance AND in how densely they write,
	// which changes how far the radius climb has to go — the two behaviours this
	// kernel has to get right.
	for &mode in &[2u8, 0u8, 3u8] {
		for &k_top in &[1usize, 4] {
			let tag = format!("phase1_topk_mode{mode}_k{k_top}");
			let f = match build_parity_fixture_mode(0xF1A5_E000u64, mode, None) {
				Ok(f) => f,
				Err(e) => { results.push((tag, false, format!("fixture: {e}"))); continue; }
			};
			let n_state = f.c.state_neurons_pub();
			let (sconn, _, sexp, _) = f.c.gpu_export();
			let n_bits = f.c.state_bits_per_neuron_pub();
			let total_input_bits = f.c.state_input_len_pub();

			let mut rng = 0xA5A5_1234_5678_9ABCu64;
			let mut next = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng };
			let input_bits: Vec<u8> = (0..total_input_bits).map(|_| (next() % 2) as u8).collect();
			let target_bits: Vec<u8> = (0..n_state).map(|_| (next() % 2) as u8).collect();
			let conns_i32: Vec<i32> = sconn.iter().map(|&c| c as i32).collect();

			let (b_k, b_v) = (t.buf(&sexp.keys), t.buf(&sexp.values));
			let (b_of, b_cn) = (t.buf(&sexp.offsets), t.buf(&sexp.counts));
			let (b_co, b_ib, b_tb) = (t.buf(&conns_i32), t.buf(&input_bits), t.buf(&target_bits));
			let b_oa = t.buf(&vec![0u64; n_state * k_top]);
			let b_or = t.buf(&vec![0u32; n_state * k_top]);
			let b_oc = t.buf(&vec![0u32; n_state]);
			let p: [u32; 4] = [n_state as u32, n_bits as u32, k_top as u32, mode as u32];
			let b_p = t.device.new_buffer_with_data(
				p.as_ptr() as *const _, (4 * mem::size_of::<u32>()) as u64,
				MTLResourceOptions::StorageModeShared);

			let cmd = t.queue.new_command_buffer();
			let enc = cmd.new_compute_command_encoder();
			enc.set_compute_pipeline_state(&t.phase1_pipeline);
			for (i, b) in [&b_k, &b_v, &b_of, &b_cn, &b_co, &b_ib, &b_tb, &b_oa, &b_or, &b_oc, &b_p]
				.iter().enumerate() { enc.set_buffer(i as u64, Some(b), 0); }
			enc.dispatch_threads(MTLSize::new(n_state as u64, 1, 1), MTLSize::new(n_state as u64, 1, 1));
			enc.end_encoding();
			cmd.commit();
			cmd.wait_until_completed();
			if cmd.status() != MTLCommandBufferStatus::Completed {
				results.push((tag, false, format!("dispatch: {:?}", cmd.status())));
				continue;
			}
			let ga = unsafe { std::slice::from_raw_parts(b_oa.contents() as *const u64, n_state * k_top) };
			let gc = unsafe { std::slice::from_raw_parts(b_oc.contents() as *const u32, n_state) };

			let bits_bool: Vec<bool> = input_bits.iter().map(|&b| b != 0).collect();
			let mut bad = 0usize;
			let mut first = String::new();
			for n in 0..n_state {
				let entries = f.c.state_entries(n);
				let target_true = target_bits[n] != 0;
				// The SAME closure the QSR solver builds (QSR_DISTANCE_COST = 7.0).
				let base = |val: u8| -> Option<f64> {
					Some(7.0 * crate::cell_mode::nudge_distance(val, target_true, mode) as f64)
				};
				let want = crate::controller_training::reachable_topk_for_neuron(
					&entries, ram_core::neuron_memory::EMPTY_U8,
					&sconn[n * n_bits..(n + 1) * n_bits], &bits_bool, n_bits,
					k_top, 0.0, &base, &[]);
				let want_addrs: Vec<u64> = want.iter().map(|&(a, _)| a as u64).collect();
				let got_addrs: Vec<u64> = (0..gc[n] as usize).map(|i| ga[n * k_top + i]).collect();
				if want_addrs != got_addrs {
					bad += 1;
					if first.is_empty() {
						first = format!("neuron {n}: cpu {want_addrs:?} vs gpu {got_addrs:?}");
					}
				}
			}
			results.push((tag, bad == 0,
				if bad == 0 { format!("{n_state} neurons, k_top={k_top}: selected address sequences identical") }
				else { format!("{bad}/{n_state} neurons differ; first {first}") }));
		}
	}
	results
}

/// CPU/GPU parity for the phase-1 candidate ranking key, AND the claim that makes the
/// port possible: that the integer key orders candidates identically to the CPU's f64
/// cost.
///
/// Two things are checked, and the second is the important one:
///   1. the key itself agrees, over random (cell, target, mode, addr, proj);
///   2. SORTING a candidate set by the integer key produces the SAME ORDER as sorting it
///      by the CPU's f64 `7.0*d + 1.0*h + saturation` — including the address tie-break —
///      for a range of saturation values. If that ever fails, the port's premise is void
///      and no amount of kernel correctness saves it.
#[pyfunction]
pub fn run_controller_candidate_rank_parity_test() -> Vec<(String, bool, String)> {
	let mut results = Vec::new();
	if Device::system_default().is_none() {
		results.push(("controller_candidate_rank_parity".to_string(), true, "skipped: no Metal device".to_string()));
		return results;
	}
	let t = match ControllerTrainer::new() {
		Ok(t) => t,
		Err(e) => { results.push(("controller_candidate_rank_parity".to_string(), false, e)); return results; }
	};
	let mut rng = 0xC4CE_B9FE_1A85_EC53u64;
	let mut next = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng };

	let n = 4096usize;
	let (mut cells, mut targets, mut modes) = (Vec::new(), Vec::new(), Vec::new());
	let (mut addrs, mut projs) = (Vec::new(), Vec::new());
	for _ in 0..n {
		cells.push((next() % 4) as u8);
		targets.push((next() % 2) as u8);
		modes.push((next() % 6) as u8);
		addrs.push(next() & ((1u64 << 30) - 1));
		projs.push(next() & ((1u64 << 30) - 1));
	}
	let (b_c, b_t, b_m) = (t.buf(&cells), t.buf(&targets), t.buf(&modes));
	let (b_a, b_p) = (t.buf(&addrs), t.buf(&projs));
	let b_o = t.buf(&vec![0u32; n]);
	let nn = n as u32;
	let b_n = t.device.new_buffer_with_data(
		&nn as *const _ as *const _, mem::size_of::<u32>() as u64, MTLResourceOptions::StorageModeShared);

	let cmd = t.queue.new_command_buffer();
	let enc = cmd.new_compute_command_encoder();
	enc.set_compute_pipeline_state(&t.cand_rank_pipeline);
	for (i, b) in [&b_c, &b_t, &b_m, &b_a, &b_p, &b_o, &b_n].iter().enumerate() {
		enc.set_buffer(i as u64, Some(b), 0);
	}
	let tg = t.cand_rank_pipeline.max_total_threads_per_threadgroup().min(n as u64);
	enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(tg, 1, 1));
	enc.end_encoding();
	cmd.commit();
	cmd.wait_until_completed();
	if cmd.status() != MTLCommandBufferStatus::Completed {
		results.push(("controller_candidate_rank_parity".to_string(), false, format!("dispatch: {:?}", cmd.status())));
		return results;
	}
	let got = unsafe { std::slice::from_raw_parts(b_o.contents() as *const u32, n) };

	let mut bad = 0usize;
	let mut first = String::new();
	for i in 0..n {
		let want = crate::controller_training::candidate_rank(
			cells[i], targets[i] != 0, modes[i], addrs[i] as usize, projs[i] as usize);
		if want != got[i] {
			bad += 1;
			if first.is_empty() {
				first = format!("i={i} cell={} tgt={} mode={} cpu={want} gpu={}",
					cells[i], targets[i], modes[i], got[i]);
			}
		}
	}
	results.push(("candidate_rank_values".to_string(), bad == 0,
		if bad == 0 { format!("{n} random candidates agree") } else { format!("{bad}/{n} differ; first {first}") }));

	// THE PREMISE: integer key orders identically to the f64 cost, for any saturation.
	// Saturation is a per-neuron constant, so it must not be able to reorder anything.
	let mut order_bad = 0usize;
	let mut order_first = String::new();
	for &sat in &[0.0f64, 0.37, 7.5, 14.999, 15.0] {
		let mut by_int: Vec<usize> = (0..n).collect();
		by_int.sort_by_key(|&i| (got[i], addrs[i]));
		let mut by_f64: Vec<usize> = (0..n).collect();
		by_f64.sort_by(|&a, &b| {
			let ca = 7.0f64 * crate::cell_mode::nudge_distance(cells[a], targets[a] != 0, modes[a]) as f64
				+ (addrs[a] ^ projs[a]).count_ones() as f64 + sat;
			let cb = 7.0f64 * crate::cell_mode::nudge_distance(cells[b], targets[b] != 0, modes[b]) as f64
				+ (addrs[b] ^ projs[b]).count_ones() as f64 + sat;
			ca.partial_cmp(&cb).unwrap().then(addrs[a].cmp(&addrs[b]))
		});
		if by_int != by_f64 {
			order_bad += 1;
			if order_first.is_empty() {
				let k = (0..n).find(|&i| by_int[i] != by_f64[i]).unwrap_or(0);
				order_first = format!("sat={sat}: first divergence at position {k}");
			}
		}
	}
	results.push(("candidate_rank_orders_like_f64_cost".to_string(), order_bad == 0,
		if order_bad == 0 { format!("integer key reproduces the f64 ordering of {n} candidates at 5 saturation values") }
		else { format!("{order_bad}/5 saturations reorder; {order_first}") }));
	results
}

/// CPU/GPU parity for the beam solver's projected address (MSB-first bit order).
///
/// Every candidate's Hamming term is `popcount(addr ^ proj)`, so a wrong `proj` does not
/// fail loudly — it ranks a different, plausible-looking set of addresses. Bit order is
/// the classic silent divergence here (27dabcf8), so this sweeps several widths and a
/// dense random input, and includes the all-zero and all-one inputs where an LSB-first
/// twin would AGREE by symmetry and hide the bug.
#[pyfunction]
pub fn run_controller_projected_address_parity_test() -> Vec<(String, bool, String)> {
	let mut results = Vec::new();
	if Device::system_default().is_none() {
		results.push(("controller_projected_address_parity".to_string(), true, "skipped: no Metal device".to_string()));
		return results;
	}
	let t = match ControllerTrainer::new() {
		Ok(t) => t,
		Err(e) => { results.push(("controller_projected_address_parity".to_string(), false, e)); return results; }
	};
	let mut rng = 0x9E37_79B9_7F4A_7C15u64;
	let mut next = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng };

	for &n_bits in &[4usize, 12, 24, 30] {
		for (label, fill) in [("random", 2u8), ("all_zero", 0), ("all_one", 1)] {
			let tag = format!("proj_addr_b{n_bits}_{label}");
			let num_neurons = 24usize;
			let total_input_bits = 512usize;
			let input_bits: Vec<u8> = (0..total_input_bits)
				.map(|_| if fill == 2 { (next() % 2) as u8 } else { fill })
				.collect();
			let conns: Vec<i32> = (0..num_neurons * n_bits)
				.map(|_| (next() % total_input_bits as u64) as i32)
				.collect();

			let (b_c, b_i) = (t.buf(&conns), t.buf(&input_bits));
			let b_o = t.buf(&vec![0u64; num_neurons]);
			let p: [u32; 2] = [num_neurons as u32, n_bits as u32];
			let b_p = t.device.new_buffer_with_data(
				p.as_ptr() as *const _, (2 * mem::size_of::<u32>()) as u64,
				MTLResourceOptions::StorageModeShared);

			let cmd = t.queue.new_command_buffer();
			let enc = cmd.new_compute_command_encoder();
			enc.set_compute_pipeline_state(&t.proj_addr_pipeline);
			for (i, b) in [&b_c, &b_i, &b_o, &b_p].iter().enumerate() { enc.set_buffer(i as u64, Some(b), 0); }
			enc.dispatch_threads(MTLSize::new(num_neurons as u64, 1, 1), MTLSize::new(num_neurons as u64, 1, 1));
			enc.end_encoding();
			cmd.commit();
			cmd.wait_until_completed();
			if cmd.status() != MTLCommandBufferStatus::Completed {
				results.push((tag, false, format!("dispatch: {:?}", cmd.status())));
				continue;
			}
			let got = unsafe { std::slice::from_raw_parts(b_o.contents() as *const u64, num_neurons) };

			// CPU side calls the REAL function the solver uses, not a copy.
			let bits_bool: Vec<bool> = input_bits.iter().map(|&b| b != 0).collect();
			let conns_i64: Vec<i64> = conns.iter().map(|&c| c as i64).collect();
			let mut bad = 0usize;
			let mut first = String::new();
			for n in 0..num_neurons {
				let want = crate::controller_training::projected_address(
					&conns_i64[n * n_bits..(n + 1) * n_bits], &bits_bool, n_bits) as u64;
				if want != got[n] {
					bad += 1;
					if first.is_empty() { first = format!("neuron {n}: cpu={want:#x} gpu={:#x}", got[n]); }
				}
			}
			results.push((tag, bad == 0,
				if bad == 0 { format!("{num_neurons} neurons agree at n_bits={n_bits} ({label})") }
				else { format!("{bad}/{num_neurons} differ; first {first}") }));
		}
	}
	results
}

/// EXHAUSTIVE CPU/GPU parity for the beam solver's innermost cost term.
///
/// `nudge_distance` ranks every candidate address in section (a) of the bptt walk
/// (cost = QSR_DISTANCE_COST × nudge_distance + HAMMING × popcount + saturation). If the
/// GPU twin disagrees anywhere, the beam is silently reordered and the GPU walk trains on
/// different addresses than the CPU — no crash, no obviously wrong output, just worse
/// training. So it is proven over the WHOLE input space (4 cells × 2 targets × 6 modes =
/// 48 combinations), not sampled.
#[pyfunction]
pub fn run_controller_nudge_distance_parity_test() -> Vec<(String, bool, String)> {
	let mut results = Vec::new();
	if Device::system_default().is_none() {
		results.push(("controller_nudge_distance_parity".to_string(), true, "skipped: no Metal device".to_string()));
		return results;
	}
	let t = match ControllerTrainer::new() {
		Ok(t) => t,
		Err(e) => { results.push(("controller_nudge_distance_parity".to_string(), false, e)); return results; }
	};
	let (mut cells, mut targets, mut modes) = (Vec::new(), Vec::new(), Vec::new());
	for mode in 0u8..6 {
		for cell in 0u8..4 {
			for tgt in 0u8..2 {
				cells.push(cell); targets.push(tgt); modes.push(mode);
			}
		}
	}
	let n = cells.len() as u32;
	let (b_c, b_t, b_m) = (t.buf(&cells), t.buf(&targets), t.buf(&modes));
	let b_o = t.buf(&vec![0u32; cells.len()]);
	let b_n = t.device.new_buffer_with_data(
		&n as *const _ as *const _, mem::size_of::<u32>() as u64, MTLResourceOptions::StorageModeShared);

	let cmd = t.queue.new_command_buffer();
	let enc = cmd.new_compute_command_encoder();
	enc.set_compute_pipeline_state(&t.nudge_dist_pipeline);
	for (i, b) in [&b_c, &b_t, &b_m, &b_o, &b_n].iter().enumerate() {
		enc.set_buffer(i as u64, Some(b), 0);
	}
	enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(n.min(48) as u64, 1, 1));
	enc.end_encoding();
	cmd.commit();
	cmd.wait_until_completed();
	if cmd.status() != MTLCommandBufferStatus::Completed {
		results.push(("controller_nudge_distance_parity".to_string(), false,
		              format!("dispatch: {:?}", cmd.status())));
		return results;
	}
	let got = unsafe { std::slice::from_raw_parts(b_o.contents() as *const u32, cells.len()) };

	let mut bad = Vec::new();
	for i in 0..cells.len() {
		let want = crate::cell_mode::nudge_distance(cells[i], targets[i] != 0, modes[i]) as u32;
		if want != got[i] {
			bad.push(format!("(cell={} target={} mode={}) cpu={} gpu={}",
				cells[i], targets[i], modes[i], want, got[i]));
		}
	}
	results.push((
		"controller_nudge_distance_parity".to_string(),
		bad.is_empty(),
		if bad.is_empty() { format!("all {} (cell,target,mode) combinations agree", cells.len()) }
		else { format!("{} of {} disagree: {}", bad.len(), cells.len(), bad.join("; ")) },
	));
	results
}

/// CPU/GPU parity for the bptt walk's section (c) — the STATE-layer commit.
///
/// Section (c) is the half of the walk that had no GPU path at all: controller_train
/// already writes the OUTPUT layer into a resident table, but the state layer was a
/// read-only sorted export, so a GPU walk could not commit to it
/// (docs/gpu_solve_port_design.md). This proves the new controller_state_commit kernel
/// reproduces the CPU's cell function exactly, INCLUDING the read-modify-write ordering
/// that makes nudge order-sensitive.
///
/// Coverage: QUAD (±1 lattice nudge) and TERNARY + BINARY (direct set, last write wins).
/// Repeated addresses are generated ON PURPOSE — a commit list of distinct addresses
/// would pass under any ordering and could not detect a race.
#[pyfunction]
pub fn run_controller_state_commit_parity_test() -> Vec<(String, bool, String)> {
	let mut results = Vec::new();
	if Device::system_default().is_none() {
		results.push(("controller_state_commit_parity".to_string(), true, "skipped: no Metal device".to_string()));
		return results;
	}
	let trainer = match ControllerTrainer::new() {
		Ok(t) => t,
		Err(e) => { results.push(("controller_state_commit_parity".to_string(), false, e)); return results; }
	};
	for &mode in &[2u8, 0u8, 3u8] {
		let tag = format!("state_commit_mode{mode}");
		let f = match build_parity_fixture_mode(0x5C0_0000u64, mode, None) {
			Ok(f) => f,
			Err(e) => { results.push((tag, false, format!("fixture: {e}"))); continue; }
		};
		let n_state = f.c.state_neurons_pub();
		let (_, _, sexp, _) = f.c.gpu_export();

		// Commits over a SMALL address pool so addresses repeat and the nudge lattice
		// is actually walked (QUAD needs repeats to move past one step).
		let mut rng = 0xC0FF_EE00_1234_5678u64;
		let mut next = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng };
		let commits: Vec<StateCommit> = (0..400)
			.map(|_| {
				let n = (next() % n_state as u64) as u32;
				StateCommit {
					genome: 0,
					neuron: n,
					addr: next() % 24,                 // tiny pool ⇒ heavy repetition
					target_true: next() % 2 == 0,
				}
			})
			.collect();

		// CPU reference: replay onto the fixture's controller via its own write path.
		let mut cpu = f.c.clone();
		for c in &commits {
			let cur = cpu.state_cell(c.neuron as usize, c.addr);
			let nv = crate::cell_mode::nudge_cell(cur, c.target_true, mode);
			cpu.write_state_cell_internal(c.neuron as usize, c.addr, nv);
		}

		let table = SlotTable::build(&[commits.len() as u64], n_state, Some(std::slice::from_ref(&sexp)));
		let gpu = match trainer.state_commit(&table, &commits, 1, n_state, mode) {
			Ok(g) => g,
			Err(e) => { results.push((tag, false, format!("dispatch: {e}"))); continue; }
		};

		// Compare the cell FUNCTION, not the stored entry set. A cell nudged back to
		// EMPTY(2) reads identically to one that was never visited, and the CPU may
		// drop it while the GPU keeps a FINAL slot holding 2 — the same function, a
		// different representation. This is the convention the existing sweeps use and
		// that `output_cell`'s doc states outright; comparing entries instead makes the
		// test fail on a difference that does not exist.
		let mut mism = 0usize;
		let mut first = String::new();
		let mut checked = 0usize;
		for n in 0..n_state {
			let got: std::collections::HashMap<u64, u8> = gpu[n].iter().copied().collect();
			for addr in 0u64..24 {
				let want = cpu.state_cell(n, addr);                 // miss → EMPTY
				let have = got.get(&addr).copied().unwrap_or(ram_core::neuron_memory::EMPTY_U8);
				checked += 1;
				// Compare the DECODED weight, which is what the controller actually
				// reads. Raw bytes can differ harmlessly: the CPU's sparse layer skips
				// storing a write equal to its mode default, so a BINARY FALSE write
				// leaves a MISS that reads EMPTY(2), while the GPU keeps a FINAL slot
				// holding 0 — and under BINARY cell_to_weight maps BOTH to 0.0
				// (neuron_memory.rs: `if cell == TRUE {1.0} else {0.0}`). Comparing
				// bytes would fail on a difference the substrate defines as none.
				// Under QUAD and TERNARY the weights stay distinct per cell state, so
				// this does not weaken those arms.
				let (wf, hf) = (crate::cell_mode::cell_weight(want, mode),
				                crate::cell_mode::cell_weight(have, mode));
				if wf != hf {
					mism += 1;
					if first.is_empty() {
						first = format!("neuron {n} addr {addr}: cpu cell={want} w={wf} vs gpu cell={have} w={hf}");
					}
				}
			}
		}
		results.push((tag, mism == 0, if mism == 0 {
			format!("{} commits, {checked} (neuron,addr) cell reads identical across {n_state} neurons", commits.len())
		} else { format!("{mism}/{checked} cell reads differ; first {first}") }));
	}
	results
}

/// Fingerprint of a controller's WHOLE cell function — both layers, every planted
/// entry, order-independent. The GPU walk must reproduce this exactly, so it is the
/// comparison target the bptt port is written against.
fn memory_digest(c: &WnnController, n_state: usize, num_out: usize) -> u64 {
	// FNV-1a over a canonical (layer, neuron, addr, value) ordering. Entries come back
	// from a DashMap in arbitrary order, so they are SORTED first — otherwise the
	// digest would be nondeterministic for reasons that have nothing to do with the
	// walk, and the determinism assertion below would be measuring hash iteration.
	let mut h: u64 = 0xcbf2_9ce4_8422_2325;
	let mut feed = |v: u64| {
		for b in v.to_le_bytes() {
			h ^= b as u64;
			h = h.wrapping_mul(0x1000_0000_01b3);
		}
	};
	for (layer, n_neurons) in [(0u64, n_state), (1u64, num_out)] {
		for n in 0..n_neurons {
			let mut e = if layer == 0 { c.state_entries(n) } else { c.output_entries(n) };
			e.sort_unstable();
			feed(layer);
			feed(n as u64);
			feed(e.len() as u64);
			for (addr, val) in e {
				feed(addr);
				feed(val as u64);
			}
		}
	}
	h
}

/// bptt window-walk reference sweep — the ACCEPTANCE GATE for the GPU port of the
/// controller's last CPU island (docs/gpu_solve_port_design.md).
///
/// Written BEFORE the kernel, deliberately. Today it has no GPU side to compare
/// against, so it pins the two properties the port depends on and would otherwise
/// discover the expensive way:
///
///   1. DETERMINISM. The same fixture walked twice must produce a bit-identical cell
///      function. If the CPU walk were not deterministic, bit-exact GPU parity would be
///      unachievable and the whole port premise is void. This is the precondition, and
///      nothing asserted it.
///   2. ORDER DEPENDENCE. Record d commits (c)/(d) and record d-1 then SOLVES against
///      the memory those commits changed. Walking the same records in a different order
///      must therefore yield a DIFFERENT cell function. That is the constraint forcing
///      the whole walk in-kernel rather than a per-record dispatch — so it is pinned
///      here, and a future "optimisation" that parallelises records will fail loudly.
///
/// When the GPU walk lands, add a third case per mode: run both, compare digests.
#[pyfunction]
pub fn run_controller_bptt_window_parity_test() -> Vec<(String, bool, String)> {
	let mut results = Vec::new();
	// Mode coverage mirrors SPLIT_TRAIN_PARITY_CASES: QUAD is the anchor, TERNARY and
	// BINARY exercise the mode-aware plant/decode paths. BINARY takes the antagonist
	// decode (its default and its only legal one).
	for &(mode, decode) in &[(2u8, None::<u8>), (0u8, None), (3u8, Some(1u8))] {
		let tag = format!("bptt_window_mode{mode}");
		let f = match build_parity_fixture_mode(0xB971_0000u64, mode, decode) {
			Ok(f) => f,
			Err(e) => { results.push((tag, false, format!("fixture: {e}"))); continue; }
		};
		let n_state = f.c.state_neurons_pub();
		let num_out = f.num_out;

		// Walk one episode's records, twice, from the SAME starting controller.
		let walk = |rev: bool| -> (u64, usize, usize) {
			let mut c = f.c.clone();
			let (mut g, mut a, mut t, mut p) =
				(f.cpu_g[0].clone(), f.cpu_a[0].clone(), f.cpu_t[0].clone(), f.cpu_p[0].clone());
			if rev { g.reverse(); a.reverse(); t.reverse(); p.reverse(); }
			let (sw, ow) = c.bptt_train_window(g, a, t, p, 4, true, false, None, 0.0);
			(memory_digest(&c, n_state, num_out), sw, ow)
		};

		let (d1, sw1, ow1) = walk(false);
		let (d2, sw2, ow2) = walk(false);
		results.push((
			format!("{tag}_deterministic"),
			d1 == d2 && sw1 == sw2 && ow1 == ow2,
			format!("digest {d1:#018x} vs {d2:#018x}, writes s={sw1}/{sw2} o={ow1}/{ow2}"),
		));

		// A walk that wrote nothing would make every other assertion here vacuous.
		results.push((
			format!("{tag}_writes_something"),
			ow1 > 0,
			format!("state_writes={sw1} output_writes={ow1} (output must be >0)"),
		));

		// THE PORT'S ACCEPTANCE GATE: the same walk, once with the CPU solve and once
		// with the GPU solve, must produce an IDENTICAL cell function. This is the
		// whole-walk statement — phase 1, the beam, and both commit paths, exercised
		// through bptt_train_window itself rather than component-by-component.
		//
		// The env var is read by gpu_solver()'s OnceLock, so it must be set before the
		// first solve of the process. Setting it here and re-walking is only meaningful
		// when the handle has not yet been initialised by an earlier case, so the walk
		// is compared against the SAME d1 either way: if the GPU path is inactive this
		// degenerates to re-running the CPU walk, which still asserts determinism and
		// cannot report a false pass.
		let gpu_active = crate::metal_controller::gpu_solver().is_some();
		// If the flag is set the GPU path MUST be live. Without this, a run with
		// WNN_CONTROLLER_GPU_SOLVE=1 that silently fell back to the CPU would report a
		// clean pass while having proven nothing about the GPU at all — the arm would
		// be comparing the CPU walk against itself.
		let requested = std::env::var("WNN_CONTROLLER_GPU_SOLVE").map(|v| v == "1").unwrap_or(false);
		let (d_gpu, sw_g, ow_g) = walk(false);
		results.push((
			format!("{tag}_gpu_walk_matches_cpu"),
			d_gpu == d1 && sw_g == sw1 && ow_g == ow1 && (gpu_active || !requested),
			format!("gpu_solve={} digest {d_gpu:#018x} vs cpu {d1:#018x}, writes s={sw_g}/{sw1} o={ow_g}/{ow1}{}",
				gpu_active,
				if gpu_active { "" } else { " (WNN_CONTROLLER_GPU_SOLVE unset — this arm re-ran the CPU walk)" }),
		));

		let (d_rev, _, _) = walk(true);
		results.push((
			format!("{tag}_order_dependent"),
			d_rev != d1,
			format!("forward {d1:#018x} vs reversed {d_rev:#018x} — these MUST differ: \
			         record d's commits are read by record d-1's solve, which is why the \
			         GPU port cannot dispatch per record"),
		));
	}
	results
}

/// final state + output memory must agree cell-for-cell. This composes the four
/// parity-gated phases (record, scan, resolve, train_seeded) into one round loop;
/// passing it retires the CPU path to the parity oracle.
#[pyfunction]
pub fn run_controller_split_train_loop_parity_test() -> Vec<(String, bool, String)> {
	let mut results = Vec::new();
	if Device::system_default().is_none() {
		results.push(("controller_split_train_loop_parity".to_string(), true, "skipped: no Metal device".to_string()));
		return results;
	}
	// coarse_target>0 keeps BOTH sides on the coarse scan path (the parity-proven
	// twin); larger targets coarsen harder → surface conflicts → exercise planting
	// + output retrain across the full loop (coarse=8/12/16 all plant on the fixture).
	// Modes: QUAD(2) is the bit-identical anchor across all 4 configs; TERNARY(0)
	// and BINARY(3) each get one selective + one non-selective config (split-trainer
	// T/B support, Luiz 12/07/2026 — plant_cell + mode-aware retrain must twin).
	for &(selective, coarse_target, mode, decode) in SPLIT_TRAIN_PARITY_CASES {
		match controller_split_train_loop_parity_once(selective, coarse_target, 0xF0_0DAEu64 ^ ((coarse_target as u64) << 8), mode, decode) {
			Ok((s_mism, o_mism, planted, s_addr, o_addr)) => {
				let dec = match decode { Some(1) => "antagonist", Some(0) => "cumulative", _ => "mode-default" };
				let name = format!("controller_split_train_loop_parity(selective={selective}, coarse={coarse_target}, mode={mode}, decode={dec})");
				results.push((name, s_mism == 0 && o_mism == 0, format!(
					"planted={planted}, state_addrs={s_addr} state_mismatch={s_mism}, output_addrs={o_addr} output_mismatch={o_mism}")));
			}
			Err(e) => results.push((format!("controller_split_train_loop_parity(selective={selective}, coarse={coarse_target}, mode={mode}, decode={decode:?})"), false, e)),
		}
	}
	results
}

fn controller_split_train_loop_parity_once(selective: bool, coarse_target: usize, salt: u64,
                                           memory_mode: u8, output_decode: Option<u8>)
	-> Result<(usize, usize, usize, usize, usize), String> {
	// Loosened clean_gain/accum_corr (vs production 0.999/0.9) so the synthetic
	// random fixture actually plants latches/counters/bidir chains — exercising the
	// full resolve+retrain machinery the parity must cover. Parity is threshold-
	// agnostic (it's the same decision both sides).
	let (tau, clean_gain, accum_corr, max_rounds, k_start) = (0.1f32, 0.7f32, 0.6f32, 6usize, 1usize);
	// Two identical controllers from the same deterministic fixture seed.
	let f_cpu = build_parity_fixture_mode(salt, memory_mode, output_decode)?;
	let f_gpu = build_parity_fixture_mode(salt, memory_mode, output_decode)?;
	let mut c_cpu = f_cpu.c;
	let c_gpu = f_gpu.c;
	let (num_motors, levels, n_state, ..) = c_gpu.gpu_dims();
	let num_out = num_motors * levels;

	// GPU loop (reads/writes c_gpu via interior mutability + resolve apply-back).
	let trainer = ControllerTrainer::new()?;
	let batch = TrainBatch {
		ep_base: &f_gpu.ep_base, ep_count: &f_gpu.ep_count, step_base: &f_gpu.step_base, step_count: &f_gpu.step_count,
		gyros: &f_gpu.gyros, accels: &f_gpu.accels, targets: &f_gpu.targets, pid_pwms: &f_gpu.pids, init_q: &f_gpu.init_q,
		selective, target_rpy: [0.0, 0.0, 0.0],
	};
	let (_rr, _cf, planted, _pr) = trainer.split_train_loop_gpu(
		&c_gpu, &batch, tau, clean_gain, accum_corr, max_rounds, k_start, coarse_target)?;

	// CPU reference loop.
	let _ = c_cpu.split_train_loop(
		f_cpu.cpu_g, f_cpu.cpu_a, f_cpu.cpu_t, f_cpu.cpu_p,
		tau, clean_gain, accum_corr, max_rounds, k_start, coarse_target, selective, vec![]);

	// Compare STATE memory (cell function over union of touched addresses per neuron).
	let mut s_mism = 0usize; let mut s_addr = 0usize;
	for n in 0..n_state {
		let mut all: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
		for (a, _) in c_cpu.state_entries(n) { all.insert(a); }
		for (a, _) in c_gpu.state_entries(n) { all.insert(a); }
		s_addr += all.len();
		for a in all { if c_cpu.state_cell(n, a) != c_gpu.state_cell(n, a) { s_mism += 1; } }
	}
	// Compare OUTPUT memory.
	let mut o_mism = 0usize; let mut o_addr = 0usize;
	for n in 0..num_out {
		let mut all: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
		for (a, _) in c_cpu.output_entries(n) { all.insert(a); }
		for (a, _) in c_gpu.output_entries(n) { all.insert(a); }
		o_addr += all.len();
		for a in all { if c_cpu.output_cell(n, a) != c_gpu.output_cell(n, a) { o_mism += 1; } }
	}
	Ok((s_mism, o_mism, planted, s_addr, o_addr))
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
		gyros: &f.gyros, accels: &f.accels, targets: &f.targets, pid_pwms: &f.pids, init_q: &f.init_q,
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
		// cpu_state_flat is packed (u32 words); the GPU reads back per-bit bools.
		// Compare bit-for-bit — same test as the old slice equality, just across
		// the two representations.
		if (0..state_len).any(|b| g_state[b] != crate::controller_split::sif_bit(&cpu_state_flat, r, state_len, b)) {
			mism_state += 1;
		}
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

	let state_ins_bools: Vec<bool> = (0..r * sil).map(|_| xf(&mut rng) < 0.5).collect();
	let state_ins_flat = crate::controller_split::pack_sif(&state_ins_bools, sil);
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
	let state_ins_bools: Vec<bool> = (0..r * sil).map(|_| xf(&mut rng) < 0.5).collect();
	let state_ins_flat = crate::controller_split::pack_sif(&state_ins_bools, sil);
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
		false, false, false, false, true, false, false, false, 0.99, 1.0, 0.001,
		true,
		1,   // action_repeat: parity fixtures stay at N=1 (bit-identical anchor)
		2,   // memory_mode: parity fixtures are QUAD (bit-identical anchor)
		None, // output_decode: default for the mode — fixtures must not move
		None, 0.05, // dhat_b: obs_dhat OFF for fixtures (bit-identical anchor)
	).map_err(|e| format!("{e}"))?;

	// Synthetic state-layer input records (the scan source for visited bases).
	let num_records = 200usize;
	let sif_bools: Vec<bool> = (0..num_records * state_input_len).map(|_| xf(&mut rng) < 0.5).collect();
	let sif = crate::controller_split::pack_sif(&sif_bools, state_input_len);

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
		false, false, false, false, true, false, false, false, 0.99, 1.0, 0.001,
		true,
		1,   // action_repeat: parity fixtures stay at N=1 (bit-identical anchor)
		2,   // memory_mode: parity fixtures are QUAD (bit-identical anchor)
		None, // output_decode: default for the mode — fixtures must not move
		None, 0.05, // dhat_b: obs_dhat OFF for fixtures (bit-identical anchor)
	).map_err(|e| format!("{e}"))?;

	let num_records = 200usize;
	let sif_bools: Vec<bool> = (0..num_records * state_input_len).map(|_| xf(&mut rng) < 0.5).collect();
	let sif = crate::controller_split::pack_sif(&sif_bools, state_input_len);

	let trainer = ControllerTrainer::new()?;
	let used = vec![false; n_state];
	let (gpu_chain, _written, gpu_cells) = trainer.plant_counter(&c, &sif, state_input_len, trigger, n_state, &used)?;

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
		false, false, false, false, true, false, false, false, 0.99, 1.0, 0.001,
		true,
		1,   // action_repeat: parity fixtures stay at N=1 (bit-identical anchor)
		2,   // memory_mode: parity fixtures are QUAD (bit-identical anchor)
		None, // output_decode: default for the mode — fixtures must not move
		None, 0.05, // dhat_b: obs_dhat OFF for fixtures (bit-identical anchor)
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

/// PyO3: parity for the P5b read-only MHT cell lookup (mht_lookup) vs bsearch_cell.
/// Builds a random cell set + a query mix (present + absent addresses) and checks
/// the two read paths agree on every query.
#[pyfunction]
pub fn run_controller_mht_lookup_parity_test() -> Vec<(String, bool, String)> {
	let mut results = Vec::new();
	if Device::system_default().is_none() {
		results.push(("controller_mht_lookup_parity".to_string(), true, "skipped: no Metal device".to_string()));
		return results;
	}
	match controller_mht_lookup_parity_once() {
		Ok((nc, nq, mism)) => {
			results.push(("controller_mht_lookup_parity".to_string(), mism == 0, format!(
				"cells={nc}, queries={nq}, mismatches={mism}")));
		}
		Err(e) => results.push(("controller_mht_lookup_parity".to_string(), false, e)),
	}
	results
}

fn controller_mht_lookup_parity_once() -> Result<(usize, usize, usize), String> {
	let mut rng = 0x4D17_10F0_0CA7u64 ^ 0x9E3779B97F4A7C15u64;
	// Random distinct cells (addr in a 20-bit space, val in 0..4).
	let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
	let mut cells: Vec<(u64, u8)> = Vec::new();
	while cells.len() < 500 {
		let a = xs(&mut rng) % (1u64 << 20);
		if seen.insert(a) {
			cells.push((a, (xs(&mut rng) % 4) as u8));
		}
	}
	// Queries: half present (some addresses we stored), half random (mostly absent).
	let mut queries: Vec<u64> = cells.iter().take(300).map(|&(a, _)| a).collect();
	for _ in 0..300 { queries.push(xs(&mut rng) % (1u64 << 20)); }

	let trainer = ControllerTrainer::new()?;
	let mism = trainer.mht_lookup_parity(&cells, &queries)?;
	Ok((cells.len(), queries.len(), mism))
}

/// PyO3: parity for the P5a resident record→scan chain (record_and_scan) vs the CPU
/// split_record → scan_conflicts_coarse. Validates that keeping the records GPU-
/// resident (only pwm reads back) produces the same conflicts as the host-roundtrip
/// path. Compares chosen_k + the conflict set (instances → spread bits).
#[pyfunction]
pub fn run_controller_record_and_scan_parity_test() -> Vec<(String, bool, String)> {
	let mut results = Vec::new();
	if Device::system_default().is_none() {
		results.push(("controller_record_and_scan_parity".to_string(), true, "skipped: no Metal device".to_string()));
		return results;
	}
	match controller_record_and_scan_parity_once() {
		Ok((cpu_k, gpu_k, n_conf, mism)) => {
			let ok = cpu_k == gpu_k && mism == 0;
			results.push(("controller_record_and_scan_parity".to_string(), ok, format!(
				"cpu_k={cpu_k}, gpu_k={gpu_k}, conflicts={n_conf}, mismatches={mism}")));
		}
		Err(e) => results.push(("controller_record_and_scan_parity".to_string(), false, e)),
	}
	results
}

fn controller_record_and_scan_parity_once() -> Result<(usize, usize, usize, usize), String> {
	let f = build_parity_fixture(0x2EC0_5CA2u64)?;
	let (bpf, num_features, frame_bits, target_min, tau) = (4usize, 9usize, 36usize, 2usize, 0.05f32);

	let trainer = ControllerTrainer::new()?;
	let batch = TrainBatch {
		ep_base: &f.ep_base, ep_count: &f.ep_count, step_base: &f.step_base, step_count: &f.step_count,
		gyros: &f.gyros, accels: &f.accels, targets: &f.targets, pid_pwms: &f.pids, init_q: &f.init_q,
		selective: false, target_rpy: [0.0, 0.0, 0.0],
	};
	let (gpu_conf, gpu_k) = trainer.record_and_scan(&f.c, &batch, tau, bpf, num_features, frame_bits, target_min)?;

	let mut c = f.c;
	let (out_ins, pwms, _sf, _sl) = c.split_record_pub(f.cpu_g.clone(), f.cpu_a.clone(), f.cpu_t.clone(), f.cpu_p.clone());
	let (cpu_conf, cpu_k) = crate::controller_split::scan_conflicts_coarse(
		&out_ins, &pwms, tau, bpf, num_features, frame_bits, target_min);

	let cpu_set: std::collections::HashMap<Vec<usize>, u32> =
		cpu_conf.iter().map(|c| (c.instances.clone(), c.spread.to_bits())).collect();
	let gpu_set: std::collections::HashMap<Vec<usize>, u32> =
		gpu_conf.iter().map(|c| (c.instances.clone(), c.spread.to_bits())).collect();
	let mut mism = 0usize;
	for (k, v) in &cpu_set { if gpu_set.get(k) != Some(v) { mism += 1; } }
	for (k, v) in &gpu_set { if cpu_set.get(k) != Some(v) { mism += 1; } }
	Ok((cpu_k, gpu_k, cpu_conf.len(), mism))
}

/// PyO3: parity for the P5a.2 resident record→search chain. record_dispatch keeps
/// the packed state_ins (b_rs) RESIDENT; sep_walk_buffer + accumulator_search_buffer
/// read it directly (no state_ins round-trip) and must match the CPU search
/// (discriminative_walk / detect_accumulator(_bidir)) on the CPU split_record state.
#[pyfunction]
pub fn run_controller_record_search_parity_test() -> Vec<(String, bool, String)> {
	let mut results = Vec::new();
	if Device::system_default().is_none() {
		results.push(("controller_record_search_parity".to_string(), true, "skipped: no Metal device".to_string()));
		return results;
	}
	match controller_record_search_parity_once() {
		Ok((n, mism_sep, mism_acc)) => {
			let ok = mism_sep == 0 && mism_acc == 0;
			results.push(("controller_record_search_parity".to_string(), ok, format!(
				"conflicts={n}, sep_mismatch={mism_sep}, accum_mismatch={mism_acc}")));
		}
		Err(e) => results.push(("controller_record_search_parity".to_string(), false, e)),
	}
	results
}

fn controller_record_search_parity_once() -> Result<(usize, usize, usize), String> {
	use crate::controller_split::{discriminative_walk, detect_accumulator, detect_accumulator_bidir};
	let f = build_parity_fixture(0x5EA2_C4D5u64)?;
	let trainer = ControllerTrainer::new()?;
	let batch = TrainBatch {
		ep_base: &f.ep_base, ep_count: &f.ep_count, step_base: &f.step_base, step_count: &f.step_count,
		gyros: &f.gyros, accels: &f.accels, targets: &f.targets, pid_pwms: &f.pids, init_q: &f.init_q,
		selective: false, target_rpy: [0.0, 0.0, 0.0],
	};
	// record → RESIDENT packed state_ins (b_rs).
	let rb = trainer.record_dispatch(&[&f.c], &batch)?;
	let (sil, total_steps, state_words) = (rb.state_input_len, rb.total_steps, rb.state_words);

	// episode/step maps from the batch (cheap, derived — not a GPU readback).
	let e_count = f.ep_count[0] as usize;
	let (step_base, step_count): (Vec<usize>, Vec<usize>) =
		(f.step_base.iter().map(|&x| x as usize).collect(), f.step_count.iter().map(|&x| x as usize).collect());
	let mut ep_of = vec![0usize; total_steps];
	let mut step_of = vec![0usize; total_steps];
	let mut ep_start = vec![0usize; e_count];
	for ep in 0..e_count {
		ep_start[ep] = step_base[ep];
		for s in 0..step_count[ep] { ep_of[step_base[ep] + s] = ep; step_of[step_base[ep] + s] = s; }
	}

	// CPU split_record state_ins (== GPU b_rs by P2a record parity) — the reference.
	let mut c = f.c;
	let (_oi, _pw, cpu_state_flat, cpu_sil) = c.split_record_pub(f.cpu_g.clone(), f.cpu_a.clone(), f.cpu_t.clone(), f.cpu_p.clone());
	// state_flat is bit-PACKED into u32 words (the 20/07/2026 Metal word-layout
	// packing, worth 985 -> 758 B/cell), so its length is total_steps * state_words —
	// NOT total_steps * state_input_len, which is a count of BITS. This assertion
	// compared a packed length against a bit count and had been failing since the
	// packing landed; it went unnoticed because the sweep was reachable only from
	// Python and nothing ran it. state_words is already unpacked above.
	if cpu_sil != sil || cpu_state_flat.len() != total_steps * state_words {
		return Err(format!(
			"state shape mismatch: cpu_sil={cpu_sil} sil={sil} words={state_words} \
			 len={} expected={}", cpu_state_flat.len(), total_steps * state_words));
	}

	// Synthetic conflicts within single episodes (steps ≥5 so max_lag ≥5).
	let mut rng = 0x515E_A2C4u64 ^ 0x9E3779B97F4A7C15u64;
	let candidate_bits: Vec<usize> = (0..16).collect();
	let (mut conflicts, mut labels, mut scalars, mut max_lags) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
	for _ in 0..12 {
		let ep = (xs(&mut rng) % e_count as u64) as usize;
		let t = step_count[ep];
		let m = 4 + (xs(&mut rng) % 5) as usize;
		let insts: Vec<usize> = (0..m).map(|_| step_base[ep] + 5 + (xs(&mut rng) % (t as u64 - 5)) as usize).collect();
		let labs: Vec<bool> = insts.iter().map(|_| xf(&mut rng) < 0.5).collect();
		let scal: Vec<f32> = insts.iter().map(|_| xf(&mut rng) * 2.0 - 1.0).collect();
		let ml = insts.iter().map(|&i| step_of[i]).min().unwrap_or(0).min(48);
		conflicts.push(insts); labels.push(labs); scalars.push(scal); max_lags.push(ml);
	}

	// GPU resident search (reads b_rs directly).
	let gpu_sep = trainer.sep_walk_buffer(&rb.b_rs, state_words, &conflicts, &labels, &ep_of, &step_of, &ep_start, &candidate_bits, &max_lags)?;
	let gpu_acc = trainer.accumulator_search_buffer(&rb.b_rs, state_words, &conflicts, &scalars, &ep_of, &step_of, &ep_start, &candidate_bits, &max_lags, true)?;

	let n = conflicts.len();
	let (mut mism_sep, mut mism_acc) = (0usize, 0usize);
	for ci in 0..n {
		let cpu_sep = discriminative_walk(&conflicts[ci], &labels[ci], &ep_of, &step_of, &ep_start, &cpu_state_flat, sil, &candidate_bits, max_lags[ci]);
		let sep_eq = match (&cpu_sep, &gpu_sep[ci]) {
			(None, None) => true,
			(Some(a), Some(b)) => a.bit == b.bit && a.lag == b.lag && a.gain.to_bits() == b.gain.to_bits() && a.high_on == b.high_on,
			_ => false,
		};
		if !sep_eq { mism_sep += 1; }

		let cpu_acc = detect_accumulator(&conflicts[ci], &scalars[ci], &ep_of, &step_of, &ep_start, &cpu_state_flat, sil, &candidate_bits, max_lags[ci]);
		let cpu_bid = detect_accumulator_bidir(&conflicts[ci], &scalars[ci], &ep_of, &step_of, &ep_start, &cpu_state_flat, sil, &candidate_bits, max_lags[ci]);
		let (g_acc, g_bid) = &gpu_acc[ci];
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
		if !acc_eq || !bid_eq { mism_acc += 1; }
	}
	Ok((n, mism_sep, mism_acc))
}

/// PyO3: parity for the P5-integrate resolve glue — resolve_conflict_gpu vs the CPU
/// split_resolve_conflict. Two identical controllers (same fixture seed) resolve the
/// SAME conflict; we compare the returned (mode, neurons) AND the resulting state
/// memory (the apply-back must reproduce the CPU planting exactly).
#[pyfunction]
pub fn run_controller_resolve_conflict_parity_test() -> Vec<(String, bool, String)> {
	let mut results = Vec::new();
	if Device::system_default().is_none() {
		results.push(("controller_resolve_conflict_parity".to_string(), true, "skipped: no Metal device".to_string()));
		return results;
	}
	match controller_resolve_conflict_parity_once() {
		Ok((cpu_mode, gpu_mode, neurons_eq, cell_mism)) => {
			let ok = cpu_mode == gpu_mode && neurons_eq && cell_mism == 0;
			results.push(("controller_resolve_conflict_parity".to_string(), ok, format!(
				"cpu_mode={cpu_mode}, gpu_mode={gpu_mode}, neurons_match={neurons_eq}, cell_mismatch={cell_mism}")));
		}
		Err(e) => results.push(("controller_resolve_conflict_parity".to_string(), false, e)),
	}
	results
}

fn controller_resolve_conflict_parity_once() -> Result<(i64, i64, bool, usize), String> {
	let salt = 0x2EC0_5CA2u64;   // a seed whose records produce conflicts (see record_and_scan)
	let f = build_parity_fixture(salt)?;
	let (_nm, _lv, n_state, _sb, _ob, bpf, window) = f.c.gpu_dims();
	let (num_features, ..) = f.c.obs_params();
	let frame_bits = num_features * bpf;
	let sensor_window = window * frame_bits;

	// Records (+ episode maps + candidate_bits) from a controller at its initial state.
	let mut c_rec = f.c;
	let (out_ins, pwms, state_flat, sil) = c_rec.split_record_pub(f.cpu_g.clone(), f.cpu_a.clone(), f.cpu_t.clone(), f.cpu_p.clone());
	let total_steps = out_ins.len();
	let e_count = f.ep_count[0] as usize;
	let step_base: Vec<usize> = f.step_base.iter().map(|&x| x as usize).collect();
	let step_count: Vec<usize> = f.step_count.iter().map(|&x| x as usize).collect();
	let mut ep_of = vec![0usize; total_steps];
	let mut step_of = vec![0usize; total_steps];
	let mut ep_start = vec![0usize; e_count];
	for ep in 0..e_count {
		ep_start[ep] = step_base[ep];
		for s in 0..step_count[ep] { ep_of[step_base[ep] + s] = ep; step_of[step_base[ep] + s] = s; }
	}
	let (sc, _oc, _se, _oe) = c_rec.gpu_export();
	let mut candidate_bits: Vec<usize> = sc.iter().map(|&x| x as usize).filter(|&b| b < sensor_window).collect();
	candidate_bits.sort_unstable();
	candidate_bits.dedup();

	// A conflict to resolve (worst by spread).
	let (conflicts, _k) = crate::controller_split::scan_conflicts_coarse(&out_ins, &pwms, 0.05, bpf, num_features, frame_bits, 2);
	if conflicts.is_empty() {
		return Ok((0, 0, true, 0));   // no conflict — trivial agreement
	}
	let instances = &conflicts[0].instances;
	// Thresholds at 0 force BOTH sides through the full Type-1 → Type-2 plant-attempt
	// chain (every search result clears the filter), maximizing exercised paths while
	// still demanding exact agreement on the decision + the resulting memory.
	let (clean_gain, accum_corr) = (0.0f32, 0.0f32);
	let used = vec![false; n_state];

	// CPU + GPU resolve on two identical fresh controllers.
	let c_cpu = build_parity_fixture(salt)?.c;
	let (cpu_mode, cpu_neurons) = c_cpu.split_resolve_conflict_pub(
		instances, &pwms, &ep_of, &step_of, &ep_start, &state_flat, sil, &candidate_bits, clean_gain, accum_corr, &used);

	let c_gpu = build_parity_fixture(salt)?.c;
	let trainer = ControllerTrainer::new()?;
	let (gpu_mode, gpu_neurons) = trainer.resolve_conflict_gpu(
		&c_gpu, instances, &pwms, &ep_of, &step_of, &ep_start, &state_flat, sil, &candidate_bits, clean_gain, accum_corr, &used)?;

	// Compare decision + the resulting state memory over every touched address.
	let neurons_eq = cpu_neurons == gpu_neurons;
	let mut cell_mism = 0usize;
	for n in 0..n_state {
		let mut addrs: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
		for (a, _) in c_cpu.state_entries(n) { addrs.insert(a); }
		for (a, _) in c_gpu.state_entries(n) { addrs.insert(a); }
		for a in addrs {
			if c_cpu.state_cell(n, a) != c_gpu.state_cell(n, a) { cell_mism += 1; }
		}
	}
	Ok((cpu_mode, gpu_mode, neurons_eq, cell_mism))
}

#[cfg(test)]
mod tests {
	use super::*;

	// ---- CPU/GPU parity sweeps as first-class `cargo test` cases -----------------
	//
	// 04/08/2026. These 14 sweeps existed as pure-Rust functions returning
	// Vec<(case, ok, detail)> but were reachable ONLY from Python, via the
	// pyfunctions registered in lib.rs. That was carried in the notes as a hard
	// constraint — "parity is Python-only, a #[test] cannot link (pyo3
	// extension-module, no libpython in a test binary)" — and it is simply FALSE:
	// every sweep is pure Rust with no PyO3 type in its signature, and this module
	// already ran 11 Metal #[test]s against the same device. Nobody had wired them.
	//
	// The cost of that gap was real: CPU/GPU parity only ran when a human remembered
	// to invoke it from Python after an install, so a regression could land and sit.
	// As #[test]s they run on every `cargo test -p ram_controller --lib`.
	//
	// A sweep self-skips (ok=true, "skipped: no Metal device") on a machine without
	// Metal, so this stays green on CI boxes that have no GPU.
	macro_rules! parity_sweep_test {
		($name:ident, $sweep:path) => {
			#[test]
			fn $name() {
				let results = $sweep();
				assert!(!results.is_empty(),
				        concat!(stringify!($sweep), " returned no cases — a sweep that ",
				                "asserts nothing passes vacuously, which is worse than failing"));
				let failed: Vec<&(String, bool, String)> =
					results.iter().filter(|(_, ok, _)| !ok).collect();
				assert!(failed.is_empty(),
				        "{}/{} parity cases FAILED: {:#?}",
				        failed.len(), results.len(), failed);
			}
		};
	}

	parity_sweep_test!(parity_controller_train, run_controller_train_parity_test);
	parity_sweep_test!(parity_controller_train_seeded, run_controller_train_seeded_parity_test);
	parity_sweep_test!(parity_split_train_loop, run_controller_split_train_loop_parity_test);
	parity_sweep_test!(parity_bptt_window, run_controller_bptt_window_parity_test);
	parity_sweep_test!(parity_state_commit, run_controller_state_commit_parity_test);
	parity_sweep_test!(parity_nudge_distance, run_controller_nudge_distance_parity_test);
	parity_sweep_test!(parity_projected_address, run_controller_projected_address_parity_test);
	parity_sweep_test!(parity_candidate_rank, run_controller_candidate_rank_parity_test);
	parity_sweep_test!(parity_phase1_topk, run_controller_phase1_topk_parity_test);
	parity_sweep_test!(parity_beam_solve, run_controller_beam_solve_parity_test);

	/// The coalescer's real failure mode is not "slow" — it is a result reaching the
	/// WRONG THREAD, which would silently train genomes on each other's solves and
	/// produce no error anywhere. Single-threaded tests cannot see it, so this drives
	/// many threads through it concurrently with DISTINCT inputs and requires every one
	/// to get back exactly what it would have got alone.
	#[test]
	fn coalescer_never_crosses_results_between_threads() {
		if Device::system_default().is_none() { return; }
		let t = match ControllerTrainer::new() { Ok(t) => t, Err(_) => return };
		let f = build_parity_fixture_mode(0xC0A1_E5CE, 2, None).expect("fixture");
		let n_state = f.c.state_neurons_pub();
		let (sconn, _, sexp, _) = f.c.gpu_export();
		let n_bits = f.c.state_bits_per_neuron_pub();
		let tib = f.c.state_input_len_pub();

		// N distinct jobs: each gets its own input/target pattern, so each has its own
		// expected answer and a crossed result cannot coincidentally match.
		const N: usize = 12;
		let jobs: Vec<OwnedLayer> = (0..N).map(|j| {
			let mut rng = 0xAB_CDEFu64.wrapping_mul(j as u64 + 1) | 1;
			let mut next = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng };
			OwnedLayer {
				keys: sexp.keys.clone(), values: sexp.values.clone(),
				offsets: sexp.offsets.clone(), counts: sexp.counts.clone(),
				conns: sconn.to_vec(),
				num_inst: 1, neurons_per_inst: n_state,
				n_bits, total_input_bits: tib,
				input_bits: (0..tib).map(|_| next() % 2 == 0).collect(),
				target_bits: (0..n_state).map(|_| next() % 2 == 0).collect(),
				n_immutable_bits: 0,
			}
		}).collect();

		// Ground truth, computed serially through the same path.
		let expect: Vec<SolveOut> = jobs.iter()
			.map(|l| solve_coalescer().solve(&t, vec![l.clone()], 4, 2).expect("serial solve"))
			.collect();

		// Now all at once. If the leader/follower bookkeeping is wrong this either
		// deadlocks (the test times out) or returns another thread's answer.
		let got: Vec<SolveOut> = std::thread::scope(|sc| {
			let hs: Vec<_> = jobs.iter().map(|l| sc.spawn(|| {
				solve_coalescer().solve(&t, vec![l.clone()], 4, 2).expect("concurrent solve")
			})).collect();
			hs.into_iter().map(|h| h.join().expect("thread panicked")).collect()
		});

		for j in 0..N {
			assert_eq!(expect[j], got[j],
				"job {j} got a different answer under concurrency — results crossed threads");
		}
		// And the jobs must not all be identical, or crossing would be undetectable.
		assert!(expect.iter().any(|r| *r != expect[0]),
			"all jobs produced the same result — this test cannot detect crossing");
	}

	/// Dispatch-cost measurement for the three solve paths. Not an assertion about
	/// speed — it PRINTS, and the numbers are read by a human, because a timing
	/// threshold baked into a test either flakes or gets loosened until meaningless.
	/// Run with:  cargo test ... bench_solve_paths -- --nocapture --ignored
	#[test]
	#[ignore]
	fn bench_solve_paths() {
		use std::time::Instant;
		if Device::system_default().is_none() { println!("no Metal device"); return; }
		let t = ControllerTrainer::new().expect("trainer");
		let f = build_parity_fixture_mode(0xB0BB_1E00, 2, None).expect("fixture");
		let n_state = f.c.state_neurons_pub();
		let (sconn, _, sexp, _) = f.c.gpu_export();
		let n_bits = f.c.state_bits_per_neuron_pub();
		let tib = f.c.state_input_len_pub();
		let mut rng = 0x1234_5678_9ABC_DEF0u64;
		let mut next = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng };
		let ib: Vec<bool> = (0..tib).map(|_| next() % 2 == 0).collect();
		let tb: Vec<bool> = (0..n_state).map(|_| next() % 2 == 0).collect();
		// Treat the state layer as M "motors" of L neurons so the batched entry point
		// is exercised with the same shape the walk gives it.
		let (m_cnt, lv) = (4usize, n_state / 4);
		const REPS: usize = 40;

		let t0 = Instant::now();
		for _ in 0..REPS {
			let _ = crate::controller_training::solve_partial_connectivity_qsr_reachable(
				|n| f.c.state_entries(n), ram_core::neuron_memory::EMPTY_U8,
				sconn, n_state, n_bits, tib, &ib, &tb, 0, 4, 2);
		}
		let cpu = t0.elapsed().as_secs_f64() / REPS as f64 * 1e3;

		let t1 = Instant::now();
		for _ in 0..REPS {
			for m in 0..m_cnt {
				let (mut k, mut v, mut o, mut c) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
				for nn in 0..lv {
					let mut e = f.c.state_entries(m * lv + nn); e.sort_unstable();
					o.push(k.len() as u32); c.push(e.len() as u32);
					for (a, val) in e { k.push(a); v.push(val); }
				}
				let _ = t.solve_qsr_reachable(&k, &v, &o, &c,
					&sconn[m * lv * n_bits..(m + 1) * lv * n_bits], lv, n_bits, tib,
					&ib, &tb[m * lv..(m + 1) * lv], 0, 4, 2);
			}
		}
		let per_motor = t1.elapsed().as_secs_f64() / REPS as f64 * 1e3;

		let t2 = Instant::now();
		for _ in 0..REPS {
			let (mut k, mut v, mut o, mut c) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
			for n in 0..n_state {
				let mut e = f.c.state_entries(n); e.sort_unstable();
				o.push(k.len() as u32); c.push(e.len() as u32);
				for (a, val) in e { k.push(a); v.push(val); }
			}
			let _ = t.solve_qsr_reachable_motors(&k, &v, &o, &c, sconn,
				m_cnt, lv, n_bits, tib, &ib, &tb, 0, 4, 2);
		}
		let batched = t2.elapsed().as_secs_f64() / REPS as f64 * 1e3;

		// Where does the batched GPU time actually go? Split the export GATHER (walking
		// the DashMap per solve to build keys/values/offsets/counts) from the DISPATCH.
		// If the gather dominates, residency is the whole remaining win and the kernel
		// is not the problem.
		let t3 = Instant::now();
		for _ in 0..REPS {
			let (mut k, mut v, mut o, mut c) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
			for n in 0..n_state {
				let mut e = f.c.state_entries(n); e.sort_unstable();
				o.push(k.len() as u32); c.push(e.len() as u32);
				for (a, val) in e { k.push(a); v.push(val); }
			}
			std::hint::black_box(&k);
		}
		let gather = t3.elapsed().as_secs_f64() / REPS as f64 * 1e3;

		println!("\n  export gather (CPU)  {gather:9.3} ms/record   = {:.0}% of the batched GPU path",
			gather / batched * 100.0);
		println!("  solve path            ms/record   vs CPU");
		println!("  CPU (rayon)          {cpu:9.3}    1.00x");
		println!("  GPU per-motor (x{m_cnt})  {per_motor:9.3}   {:.2}x", cpu / per_motor);
		println!("  GPU batched          {batched:9.3}   {:.2}x   ({:.2}x vs per-motor)\n",
			cpu / batched, per_motor / batched);
	}

	/// The STATE layer must become an in-kernel-WRITABLE resident table, because the
	/// bptt walk's section (c) commits to it — today it is only a read-only sorted
	/// export, which is why the walk cannot run on the GPU
	/// (docs/gpu_solve_port_design.md). This proves the state layer lays out and seeds
	/// through the SAME parity-proven SlotTable the output layer uses, so no second
	/// hand-rolled table is introduced.
	#[test]
	fn state_layer_slot_table_seeds_and_round_trips() {
		let f = build_parity_fixture(0x57A7_E000).expect("fixture");
		let (_, _, sexp, _) = f.c.gpu_export();
		let n_state = f.c.state_neurons_pub();
		let planted: usize = (0..n_state).map(|n| sexp.counts[n] as usize).sum();
		assert!(planted > 0, "fixture planted no state cells — the test would be vacuous");

		// One genome, sized like a real window (steps bound the NEW addresses).
		let t = SlotTable::build(&[40u64], n_state, Some(std::slice::from_ref(&sexp)));

		// Layout invariants the kernel's probe mask depends on.
		assert_eq!(t.off.len(), n_state, "one region per (genome, state neuron)");
		for gn in 0..n_state {
			let c = t.cap[gn];
			assert!(c.is_power_of_two(), "cap {c} must be a power of two — the probe uses cap-1 as a mask");
			assert!(t.off[gn] as usize + c as usize <= t.markers.len(), "region {gn} overruns the table");
			if gn > 0 {
				assert!(t.off[gn] >= t.off[gn - 1] + t.cap[gn - 1], "regions {} and {gn} overlap", gn - 1);
			}
		}

		// Every planted cell must be findable at the slot find_or_claim_slot would
		// probe to, carrying its value — otherwise the kernel would nudge from EMPTY
		// instead of from the accumulated cell, silently changing what training means.
		let mut found = 0usize;
		for n in 0..n_state {
			let (e0, ec) = (sexp.offsets[n] as usize, sexp.counts[n] as usize);
			for i in 0..ec {
				let (addr, val) = (sexp.keys[e0 + i], sexp.values[e0 + i]);
				let slot = host_lookup_slot(&t.markers, &t.keys, t.off[n] as usize, t.cap[n] as usize, addr)
					.unwrap_or_else(|| panic!("state cell (n={n}, addr={addr:#x}) not found after seeding"));
				assert_eq!(t.values[slot] & 0xFF, val as u32, "state cell (n={n}, addr={addr:#x}) seeded with the wrong value");
				found += 1;
			}
		}
		assert_eq!(found, planted, "seeded {found} of {planted} state cells");

		// entries() is the readback shape the CPU comparison will use.
		let total: usize = (0..n_state).map(|n| t.entries(n).len()).sum();
		assert_eq!(total, planted, "entries() must enumerate exactly the planted cells");

		// An address that was never planted must MISS, or the walk would read cells
		// that do not exist.
		let absent = sexp.keys.iter().max().copied().unwrap_or(0).wrapping_add(0x5EED);
		assert!(
			host_lookup_slot(&t.markers, &t.keys, t.off[0] as usize, t.cap[0] as usize, absent).is_none(),
			"an unplanted address must not resolve to a slot"
		);
	}
	parity_sweep_test!(parity_controller_record, run_controller_record_parity_test);
	parity_sweep_test!(parity_controller_scan, run_controller_scan_parity_test);
	parity_sweep_test!(parity_sep_walk, run_controller_sep_walk_parity_test);
	parity_sweep_test!(parity_accumulator, run_controller_accumulator_parity_test);
	parity_sweep_test!(parity_plant_latch, run_controller_plant_latch_parity_test);
	parity_sweep_test!(parity_plant_counter, run_controller_plant_counter_parity_test);
	parity_sweep_test!(parity_plant_bidir, run_controller_plant_bidir_parity_test);
	parity_sweep_test!(parity_mht_lookup, run_controller_mht_lookup_parity_test);
	parity_sweep_test!(parity_record_and_scan, run_controller_record_and_scan_parity_test);
	parity_sweep_test!(parity_record_search, run_controller_record_search_parity_test);
	parity_sweep_test!(parity_resolve_conflict, run_controller_resolve_conflict_parity_test);

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

	/// W2 layout guard: RolloutParams is all-4-byte tightly packed and must
	/// stay in field-for-field lockstep with the Metal `Params` struct
	/// (71 fields as of the ABI-12 memory_mode; verified against
	/// shaders/controller_rollout.metal `struct Params` 12/07/2026). A size
	/// change here without the matching Metal edit is the layout-drift bug
	/// class this pins — when it fires, count the Metal fields FIRST, then
	/// update both sides together.
	#[test]
	fn rollout_params_size_lockstep() {
		// 71 pre-W2.4 + 4 (D5 dropout_prob/len, D6 obs_delay, D7 torque jitter)
		// + 1 (output_decode, 03/08/2026) = 76, both sides counted field-for-field
		// against shaders/controller_rollout.metal `struct Params`.
		//
		// output_decode was inserted right after memory_mode rather than appended
		// after the W2.4 block — in BOTH structs, at the same position, so the
		// layouts still agree. The "append at END" convention is about keeping the
		// two in step, not about any absolute ordering; what matters is that the
		// edits are paired. This assert is what makes that enforceable.
		// + 35 (firmware PID cascade, 05/08/2026) = 111. The cascade block is
		// pidfw_on(1) + att(12) + rate(12) + out_limit_n(1) + hover_n(1) +
		// k_thrust(1) + decimation(1) + lpf(5) + filter_on(1), appended at the END of
		// BOTH structs in the same order. Every field is 4 bytes, so the two layouts
		// stay sequential with no padding to reason about — and THIS assert is what
		// caught the mismatch the moment only one side had been edited.
		// + 5 (L1 d̂ observer, 06/08/2026) = 116. The count is in 4-byte WORDS, not
		// fields: dhat_on(1) + dhat_b[3](3) + dhat_l_gain(1). Appended at the END of
		// BOTH structs in the same order.
		assert_eq!(mem::size_of::<RolloutParams>(), 116 * 4);
	}

	// ===== Overactuated Phase 1 (step 2): geometry rollout parity ============
	//
	// CPU oracle = the SAME closed loop the production CPU eval runs, but on
	// step_n (generic N-rotor torque). The kernel's geometry branch mirrors
	// body_torque_asym's accumulation order exactly, so CPU↔GPU on the SAME
	// path gets a tight tolerance; quad-geometry vs the legacy quad expression
	// computes the same value with DIFFERENT float rounding order (cross
	// products vs the closed-form mixer), so that cross-check is loose.

	use crate::controller::{compute_reward, monotonicity_violations_core, yaw_from_quat_rs, AttitudeSim};
	use crate::overactuated::RotorGeometry;
	use rand::{rngs::SmallRng, Rng, SeedableRng};

	const SIM_DT: f32 = 0.001;
	const SIM_ARM: f32 = 0.075;
	const SIM_KT: f32 = 2.4;
	const SIM_KD: f32 = 0.05;
	const SIM_INERTIA: [f32; 3] = [0.0023, 0.0023, 0.0046];
	const SIM_G: f32 = 9.81;

	/// 9-float geometry rows (the set_geometry contract) from a RotorGeometry.
	fn rows_from(geo: &RotorGeometry) -> Vec<[f32; 9]> {
		geo.rotors.iter().map(|r| [
			r.position[0], r.position[1], r.position[2],
			r.axis[0], r.axis[1], r.axis[2],
			r.spin, r.k_thrust, r.k_drag,
		]).collect()
	}

	/// Tiny controller (9 base features, no extras, absolute PWM). `plant`
	/// fills EVERY address of both memories with pseudorandom QUAD cells so
	/// the rollout exercises real state/output dynamics; false leaves them
	/// EMPTY (decode = hover 0.5 → constant-PWM, physics-only rollout).
	fn test_controller(num_motors: usize, seed: u64, plant: bool) -> WnnController {
		test_controller_mode(num_motors, seed, plant, 2)
	}

	/// Mode-parameterized twin (ABI 12): plants MODE-NATIVE cells — QUAD draws
	/// 0..3; TERNARY/BINARY draw {FALSE=0, TRUE=1} (2 is the EMPTY/unwritten
	/// sentinel and 3 is invalid outside QUAD).
	fn test_controller_mode(num_motors: usize, seed: u64, plant: bool, memory_mode: u8) -> WnnController {
		let (levels, bpf, window, n_state, sbpn, obpn) = (4usize, 3usize, 2usize, 8usize, 8usize, 8usize);
		let num_features = 9usize;
		let frame_bits = num_features * bpf;
		let mut rng = SmallRng::seed_from_u64(seed);
		let thresholds: Vec<f32> = (0..frame_bits).map(|_| rng.gen_range(-5.0f32..5.0)).collect();
		// Connections stay inside the POPULATED input regions ([K frames |
		// prev-state MSBs] for state; [current frame | new-state MSBs] for output).
		let state_in = window * frame_bits + n_state;
		let state_connections: Vec<i64> =
			(0..n_state * sbpn).map(|_| rng.gen_range(0..state_in) as i64).collect();
		let num_out = num_motors * levels;
		let out_in = frame_bits + n_state;
		let output_connections: Vec<i64> =
			(0..num_out * obpn).map(|_| rng.gen_range(0..out_in) as i64).collect();
		let mut c = WnnController::new_core(
			num_motors, levels, bpf, window, n_state, sbpn, obpn,
			thresholds, state_connections, output_connections,
			false, 0.15, 0.98,                 // delta-control off (absolute PWM)
			false, false, false, false, false, // H2 obs extras off
			false, false, false,
			0.99, 1.0, SIM_DT, false, 1,       // decouple off, action_repeat 1
			memory_mode,
			None,                              // output_decode: mode default (anchor)
			None, 0.05,                        // dhat_b: obs_dhat OFF (anchor)
		).expect("test controller");
		if plant {
			let cell_hi = if crate::cell_mode::is_quad(memory_mode) { 4u8 } else { 2u8 };
			let mut state_cells = Vec::new();
			for n in 0..n_state {
				for a in 0..(1u64 << sbpn) {
					state_cells.push((n, a, rng.gen_range(0u8..cell_hi)));
				}
			}
			let mut output_cells = Vec::new();
			for n in 0..num_out {
				for a in 0..(1u64 << obpn) {
					output_cells.push((n, a, rng.gen_range(0u8..cell_hi)));
				}
			}
			c.restore_cells(state_cells, output_cells);
		}
		c
	}

	/// Per-episode initial conditions: random small tilts (≤ ~17°, w-first
	/// quats, normalized) + modest body rates.
	fn test_episodes(seed: u64, n: usize) -> (Vec<f32>, Vec<f32>) {
		let mut rng = SmallRng::seed_from_u64(seed);
		let mut q0 = Vec::with_capacity(n * 4);
		let mut w0 = Vec::with_capacity(n * 3);
		for _ in 0..n {
			let ax = [rng.gen_range(-1.0f32..1.0), rng.gen_range(-1.0f32..1.0), rng.gen_range(-1.0f32..1.0)];
			let norm = (ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2]).sqrt().max(1e-6);
			let half = rng.gen_range(-0.3f32..0.3) * 0.5;
			let (s, c) = half.sin_cos();
			q0.extend_from_slice(&[c, ax[0] / norm * s, ax[1] / norm * s, ax[2] / norm * s]);
			for _ in 0..3 { w0.push(rng.gen_range(-1.0f32..1.0)); }
		}
		(q0, w0)
	}

	/// CPU oracle: the cpu_score closed loop on step_n (geometry edition).
	/// Aggregation mirrors the GPU host exactly: per-episode means, averaged
	/// over episodes; jerk normalized per episode; mono = LAST decision step's
	/// violations (the kernel's mono_last). Returns [reward, err, stable, jerk, mono].
	fn cpu_oracle_geometry(
		c: &mut WnnController, rows: &[[f32; 9]], asym: Option<Vec<f32>>,
		q0: &[f32], omega0: &[f32], num_eps: usize, steps: usize,
	) -> [f64; 5] {
		let mut sim = AttitudeSim::new(SIM_DT, SIM_ARM, SIM_KT, SIM_KD, SIM_INERTIA, SIM_G);
		sim.set_geometry_core(rows.to_vec()).expect("oracle geometry");
		sim.set_rotor_asym_core(asym).expect("oracle asym");
		let (num_motors, levels, ..) = c.gpu_dims();
		let stable_thresh = 5.0_f64.to_radians();
		let (mut sum_reward, mut sum_err, mut sum_jerk, mut sum_mono) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
		let mut n_stable = 0usize;
		for ep in 0..num_eps {
			let q = [q0[ep * 4], q0[ep * 4 + 1], q0[ep * 4 + 2], q0[ep * 4 + 3]];
			let om = [omega0[ep * 3], omega0[ep * 3 + 1], omega0[ep * 3 + 2]];
			c.reset(yaw_from_quat_rs(q));
			sim.reset(Some(q), Some(om));
			let mut ep_sum_err = 0.0f64;
			let mut ep_jerk = 0.0f64;
			let mut ep_jerk_count = 0usize;
			let mut prev_pwm = vec![0.5f32; num_motors];
			let mut first = true;
			let mut ep_steps = 0usize;
			let mut diverged = false;
			let mut mono_last = 0.0f64;
			for _t in 0..steps {
				if sim.is_unstable() { diverged = true; break; }
				let (gyro, accel) = sim.read_imu();
				let pwm = c.step(gyro, accel, [0.0, 0.0, 0.0]);
				if !first {
					let mut dj = 0.0f64;
					for m in 0..num_motors {
						let d = (pwm[m] - prev_pwm[m]) as f64;
						dj += d * d;
					}
					ep_jerk += dj.sqrt();
					ep_jerk_count += 1;
				}
				prev_pwm.copy_from_slice(&pwm);
				first = false;
				mono_last = monotonicity_violations_core(&c.get_last_output_cells(), levels, num_motors,
					c.memory_mode_u8(), c.output_decode_u8()).expect("mono") as f64;
				sim.step_n_core(&pwm).expect("step_n");
				let err = sim.attitude_error(None);
				sum_reward += compute_reward(err, 0.0, 0, 0.0, 0.0) as f64;
				ep_sum_err += err as f64;
				ep_steps += 1;
			}
			let mean_err = ep_sum_err / ep_steps.max(1) as f64;
			sum_err += mean_err;
			sum_jerk += if ep_jerk_count > 0 { ep_jerk / ep_jerk_count as f64 } else { 0.0 };
			sum_mono += mono_last;
			if !diverged && mean_err <= stable_thresh { n_stable += 1; }
		}
		let n = num_eps as f64;
		[sum_reward / n, sum_err / n, n_stable as f64 / n, sum_jerk / n, sum_mono / n]
	}

	// ===== Firmware PID cascade: GPU (pidfw_step) vs CPU (AttitudePidFirmwareRs) ====
	//
	// The cascade's oracle is the Rust twin, NOT cpu_score::rollout_one — that scorer has
	// no quad-PID residual path at all (alloc baseline only), which is also why Metal's
	// LEGACY pid_step never had a Rust parity test. See pid_firmware.rs for the twin's own
	// golden test against the Python reference; this test closes the last link in the
	// chain Python -> Rust -> Metal.
	//
	// Plant is cf21_brushless (fixed moment arm) so the loop is genuinely stable —
	// parity on a diverging trajectory is meaningless, because chaos amplifies the
	// f32-kernel / f64-CPU difference without any bug being present.
	const BL_ARM: f32 = 0.070710678;
	const BL_KT: f32 = 0.2;
	const BL_KD: f32 = 0.0056927884437141703;
	const BL_INERTIA: [f32; 3] = [3.003982457e-05, 3.019189704e-05, 5.304310132e-05];
	const BL_HOVER_N: f32 = 0.096383250;

	/// SI gains from wnn.control.pid_firmware `_SiGains.from_firmware(cf21_brushless)`,
	/// and lpf2p coefficients precomputed exactly as the host does at 500 Hz / 30 Hz.
	fn bl_pidfw_cfg(decimation: u32) -> PidFwCfg {
		// Filter coefficients come FROM the Rust filter rather than a hardcoded copy, so
		// the two implementations cannot drift and so a non-500 Hz cascade is handed the
		// right filter automatically.
		let lpf = bl_pidfw_rs(decimation).rate_lpf_coeffs().expect("filter on");
		PidFwCfg {
			att: [
				6.0, 3.0, 0.0, 0.34906585,
				6.0, 3.0, 0.0, 0.34906585,
				6.0, 1.0, 0.35, 6.28318531,
			],
			rate: [
				0.0349711022, 0.0699422043, 0.000437138777, 0.581194641,
				0.0349711022, 0.0699422043, 0.000437138777, 0.581194641,
				0.0209826613, 0.00292008703, 0.0, 2.90946386,
			],
			out_limit_n: 0.0999984741,
			hover_n: BL_HOVER_N,
			k_thrust: BL_KT,
			decimation,   // 1 kHz sim / ATTITUDE_RATE
			lpf: [lpf[0] as f32, lpf[1] as f32, lpf[2] as f32,
			      lpf[3] as f32, lpf[4] as f32],
			filter_on: 1,
		}
	}

	/// SI gains as above. attitude_hz = 1000 / decimation, so decimation 2 is the real
	/// 500 Hz firmware rate and larger values are deliberate probes of the HOLD path.
	fn bl_pidfw_rs(decimation: u32) -> crate::pid_firmware::AttitudePidFirmwareRs {
		const ATT: [f64; 12] = [
			6.0, 3.0, 0.0, 0.34906585,
			6.0, 3.0, 0.0, 0.34906585,
			6.0, 1.0, 0.35, 6.28318531,
		];
		const RATE: [f64; 12] = [
			0.0349711022, 0.0699422043, 0.000437138777, 0.581194641,
			0.0349711022, 0.0699422043, 0.000437138777, 0.581194641,
			0.0209826613, 0.00292008703, 0.0, 2.90946386,
		];
		crate::pid_firmware::AttitudePidFirmwareRs::from_si_arrays(
			ATT, RATE, 0.0999984741, BL_HOVER_N as f64, BL_KT as f64,
			1000, 1000.0 / decimation as f64, 30.0,
		).expect("cascade builds")
	}

	/// CPU oracle: mirrors the kernel's composed loop exactly — cascade base, WNN signed
	/// residual on top, jerk measured on the COMPOSED pwm.
	fn cpu_oracle_pidfw(
		c: &mut WnnController, q0: &[f32], omega0: &[f32], num_eps: usize, steps: usize,
		scale: f32, clamp_r: f32, decimation: u32,
	) -> [f64; 5] {
		let mut sim = AttitudeSim::new(SIM_DT, BL_ARM, BL_KT, BL_KD, BL_INERTIA, SIM_G);
		let (num_motors, levels, ..) = c.gpu_dims();
		let stable_thresh = 5.0_f64.to_radians();
		let (mut sum_reward, mut sum_err, mut sum_jerk, mut sum_mono) =
			(0.0f64, 0.0f64, 0.0f64, 0.0f64);
		let mut n_stable = 0usize;
		for ep in 0..num_eps {
			let q = [q0[ep * 4], q0[ep * 4 + 1], q0[ep * 4 + 2], q0[ep * 4 + 3]];
			let om = [omega0[ep * 3], omega0[ep * 3 + 1], omega0[ep * 3 + 2]];
			c.reset(yaw_from_quat_rs(q));
			sim.reset(Some(q), Some(om));
			let mut pid = bl_pidfw_rs(decimation);
			pid.reset();
			let (mut ep_sum_err, mut ep_jerk) = (0.0f64, 0.0f64);
			let mut ep_jerk_count = 0usize;
			let mut prev_pwm = [0.5f32; 4];
			let (mut first, mut ep_steps, mut diverged) = (true, 0usize, false);
			let mut mono_last = 0.0f64;
			for _t in 0..steps {
				if sim.is_unstable() { diverged = true; break; }
				let (gyro, accel) = sim.read_imu();
				let wnn = c.step(gyro, accel, [0.0, 0.0, 0.0]);
				// Same q/gyro the kernel's pidfw_step sees: true attitude, noisy gyro.
				let base = pid.step_f32(sim.quaternion(), gyro, [0.0, 0.0, 0.0]);
				let neutral = c.neutral_f32();
				let mut pwm = [0.0f32; 4];
				for m in 0..4 {
					let r = ((wnn[m] - neutral) * scale).clamp(-clamp_r, clamp_r);
					pwm[m] = (base[m] + r).clamp(0.0, 1.0);
				}
				if !first {
					let mut dj = 0.0f64;
					for m in 0..num_motors {
						let d = (pwm[m] - prev_pwm[m]) as f64;
						dj += d * d;
					}
					ep_jerk += dj.sqrt();
					ep_jerk_count += 1;
				}
				prev_pwm = pwm;
				first = false;
				mono_last = monotonicity_violations_core(
					&c.get_last_output_cells(), levels, num_motors,
					c.memory_mode_u8(), c.output_decode_u8()).expect("mono") as f64;
				sim.step(pwm);
				let err = sim.attitude_error(None);
				sum_reward += compute_reward(err, 0.0, 0, 0.0, 0.0) as f64;
				ep_sum_err += err as f64;
				ep_steps += 1;
			}
			let mean_err = ep_sum_err / ep_steps.max(1) as f64;
			sum_err += mean_err;
			sum_jerk += if ep_jerk_count > 0 { ep_jerk / ep_jerk_count as f64 } else { 0.0 };
			sum_mono += mono_last;
			if !diverged && mean_err <= stable_thresh { n_stable += 1; }
		}
		let n = num_eps as f64;
		[sum_reward / n, sum_err / n, n_stable as f64 / n, sum_jerk / n, sum_mono / n]
	}

	#[test]
	fn gpu_pidfw_cascade_matches_cpu_twin() {
		if Device::system_default().is_none() {
			eprintln!("skipping: no Metal device");
			return;
		}
		let (num_eps, steps) = (8usize, 2000usize);
		let (q0, w0) = test_episodes(0xF19FA, num_eps);
		let ev = ControllerRolloutEvaluator::new().expect("evaluator");
		// TWO decimations on purpose. 2 is the real firmware rate (500 Hz). 10 is a probe
		// of the HOLD path: the cascade output is then reused for 10 physical steps, so a
		// kernel that failed to hold — or held the wrong value, or used the wrong dt —
		// diverges far outside tolerance. At decimation 2 alone the difference between
		// holding and not holding hides inside the tolerance, because the output moves
		// slowly against a 1 ms step; that was a measured gap in the first version of
		// this test, and this is the fix for it.
		for decimation in [2u32, 10u32] {
			// EMPTY memories ⇒ the WNN decodes to neutral ⇒ residual is exactly 0, so the
			// composed action IS the cascade. That isolates pidfw_step, which is the point.
			let mut c = test_controller(4, 0xC5, false);
			let residual = ResidualCfg {
				scale: 1.0, clamp: 0.4, pid: [0.0; 10],
				cascade: Some(bl_pidfw_cfg(decimation)),
			};
			let gpu = ev.score(
				&[&c], &q0, &w0, num_eps, steps,
				(SIM_DT, BL_ARM, BL_KT, BL_INERTIA, SIM_G), BL_KD,
				[0.0, 0.0, 0.0], None, Some(residual), None, None,
			).expect("gpu score");
			let cpu = cpu_oracle_pidfw(&mut c, &q0, &w0, num_eps, steps, 1.0, 0.4, decimation);
			// f32 kernel vs f64 CPU over 2000 steps through a decimated cascade with a
			// 2-pole filter: the filter's recursion accumulates the rounding difference,
			// so this is looser than the memoryless alloc parity. Anything structural (a
			// swapped mixer sign, a dropped hold, an unfiltered D) is orders of magnitude
			// larger.
			for (i, name) in ["reward", "err", "stable", "jerk", "mono"].iter().enumerate() {
				assert_rel_close(gpu[0][i], cpu[i], 2e-2, 1e-3,
					&format!("pidfw dec={decimation} {name}"));
			}
			// Guard against a vacuous pass: the cascade must actually be flying.
			assert!(gpu[0][1] > 1e-4,
				"vacuous rollout at dec={decimation} (err={})", gpu[0][1]);
		}
		// At the real firmware rate the cascade must also STABILIZE the vehicle — a
		// parity test that agreed on a diverging trajectory would be worthless.
		let mut c = test_controller(4, 0xC5, false);
		let residual = ResidualCfg {
			scale: 1.0, clamp: 0.4, pid: [0.0; 10], cascade: Some(bl_pidfw_cfg(2)),
		};
		let gpu = ev.score(
			&[&c], &q0, &w0, num_eps, steps,
			(SIM_DT, BL_ARM, BL_KT, BL_INERTIA, SIM_G), BL_KD,
			[0.0, 0.0, 0.0], None, Some(residual), None, None,
		).expect("gpu score");
		let _ = &mut c;
		assert_eq!(gpu[0][2], 1.0, "cascade failed to stabilize (stable={})", gpu[0][2]);
	}
	// SENSITIVITY, MEASURED BY MUTATION (05/08/2026) — recorded so nobody assumes more
	// coverage than this test has:
	//   - GPU filter_on 1 -> 0 (CPU oracle unchanged): FAILS. Good; the filter is the term
	//     that decides stability and it is covered.
	//   - GPU decimation mismatched against the oracle: FAILS at dec=10 (it passed when the
	//     test only ran dec=2, which is why dec=10 is here). The HOLD path is covered.

	fn assert_rel_close(gpu: f64, cpu: f64, rel: f64, abs_floor: f64, what: &str) {
		let tol = abs_floor.max(cpu.abs() * rel);
		assert!(
			(gpu - cpu).abs() <= tol,
			"{what}: gpu={gpu} cpu={cpu} (|Δ|={} > tol={tol})", (gpu - cpu).abs()
		);
	}

	/// The perturbed octo the physics tests share: baked per-rotor asym +
	/// tilt/position error — every new field of the RotorGpu table is live.
	fn perturbed_octo() -> (Vec<[f32; 9]>, Vec<f32>) {
		let tilt = [0.8f32, -1.2, 0.5, -0.3, 1.0, -0.7, 0.2, -0.9]
			.map(|d: f32| d.to_radians());
		let pos_err: Vec<[f32; 3]> = (0..8)
			.map(|i| [0.001 * (i as f32 - 3.5), -0.0008 * (i as f32 - 3.5), 0.0005])
			.collect();
		let geo = RotorGeometry::octo_x(SIM_ARM, SIM_KT, SIM_KD).perturbed(&tilt, &pos_err);
		let asym = vec![0.98f32, 1.02, 1.01, 0.99, 1.015, 0.985, 1.005, 0.995];
		(rows_from(&geo), asym)
	}

	#[test]
	fn rotor_table_builder_validates_and_bakes() {
		// Length gates.
		assert!(build_rotor_table(&[], None).is_err());
		assert!(build_rotor_table(&vec![[0.0f32; 9]; 9], None).is_err());
		let rows = rows_from(&RotorGeometry::quad_plus(SIM_ARM, SIM_KT, SIM_KD));
		assert!(build_rotor_table(&rows, Some(&[1.0, 1.0])).is_err(), "asym len mismatch");
		// Axis normalization mirrors set_geometry_core; asym bakes into k_thrust.
		let raw = [[0.1f32, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, SIM_KT, SIM_KD]];
		let t = build_rotor_table(&raw, Some(&[0.9])).unwrap();
		assert!((t[0].az - 1.0).abs() < 1e-6, "axis must normalize: {}", t[0].az);
		assert!((t[0].k_thrust - SIM_KT * 0.9).abs() < 1e-6, "asym must bake: {}", t[0].k_thrust);
	}

	/// Physics-only tight parity: EMPTY memory ⇒ constant hover PWM ⇒ the
	/// trajectory is pure step_n integration (no discrete controller feedback
	/// to amplify float drift). Perturbed octo + asym makes the torque loop,
	/// table upload, and specialized pipeline all load-bearing.
	#[test]
	fn gpu_octo_geometry_matches_cpu_step_n_hover() {
		if Device::system_default().is_none() {
			eprintln!("skipping: no Metal device");
			return;
		}
		let (rows, asym) = perturbed_octo();
		let table = build_rotor_table(&rows, Some(&asym)).unwrap();
		let (num_eps, steps) = (16usize, 300usize);
		let (q0, w0) = test_episodes(0xA11CE, num_eps);
		let mut c = test_controller(8, 0xBEEF, false);
		let oracle = cpu_oracle_geometry(&mut c, &rows, Some(asym), &q0, &w0, num_eps, steps);
		let ev = ControllerRolloutEvaluator::new().expect("evaluator");
		let rows_gpu = ev.score(
			&[&c], &q0, &w0, num_eps, steps,
			(SIM_DT, SIM_ARM, SIM_KT, SIM_INERTIA, SIM_G), SIM_KD,
			[0.0, 0.0, 0.0], None, None, Some(&table), None,
		).expect("gpu score");
		assert_rel_close(rows_gpu[0][0], oracle[0], 1e-3, 1e-6, "reward");
		assert_rel_close(rows_gpu[0][1], oracle[1], 1e-3, 1e-6, "err");
		assert_eq!(rows_gpu[0][2], oracle[2], "stable rate");
		assert_rel_close(rows_gpu[0][3], oracle[3], 1e-3, 1e-7, "jerk");
		assert_eq!(rows_gpu[0][4], oracle[4], "mono");
		// Non-vacuity: the tilted ICs + perturbed/asym table must produce real
		// attitude error, or the parity above proves nothing about the torque path.
		assert!(oracle[1] > 1e-3, "hover rollout has no attitude error (err={})", oracle[1]);
	}

	/// Closed-loop parity on the octo: planted pseudorandom cells drive real
	/// state/output dynamics through the N=8 decode + torque path. Discrete
	/// thermometer decisions amplify float drift near thresholds, so the
	/// tolerance is looser than the hover test — this is the wiring gate
	/// (wrong buffer/pipeline/N would be off by far more than 2%).
	#[test]
	fn gpu_octo_geometry_matches_cpu_step_n_closed_loop() {
		if Device::system_default().is_none() {
			eprintln!("skipping: no Metal device");
			return;
		}
		let (rows, asym) = perturbed_octo();
		let table = build_rotor_table(&rows, Some(&asym)).unwrap();
		let (num_eps, steps) = (24usize, 300usize);
		let (q0, w0) = test_episodes(0x0C70, num_eps);
		let mut c = test_controller(8, 0xD00D, true);
		let oracle = cpu_oracle_geometry(&mut c, &rows, Some(asym), &q0, &w0, num_eps, steps);
		let ev = ControllerRolloutEvaluator::new().expect("evaluator");
		let rows_gpu = ev.score(
			&[&c], &q0, &w0, num_eps, steps,
			(SIM_DT, SIM_ARM, SIM_KT, SIM_INERTIA, SIM_G), SIM_KD,
			[0.0, 0.0, 0.0], None, None, Some(&table), None,
		).expect("gpu score");
		assert_rel_close(rows_gpu[0][0], oracle[0], 2e-2, 1e-4, "reward");
		assert_rel_close(rows_gpu[0][1], oracle[1], 2e-2, 1e-4, "err");
		assert_rel_close(rows_gpu[0][2], oracle[2], 0.0, 1.0 / num_eps as f64 + 1e-9, "stable rate");
		assert_rel_close(rows_gpu[0][3], oracle[3], 2e-2, 1e-4, "jerk");
		// Non-vacuity: planted cells must actually vary the PWM (jerk > 0) and
		// the tilted ICs must produce real attitude error — an all-EMPTY memory
		// or a zeroed rotor table would pass the parity above trivially.
		assert!(oracle[3] > 1e-4, "closed-loop rollout is trivially constant (jerk={})", oracle[3]);
		assert!(oracle[1] > 1e-3, "closed-loop rollout has no attitude error (err={})", oracle[1]);
		// PRODUCTION CPU scorer parity (mono unification 12/07/2026): the rayon
		// batch scorer's rollout_one must agree with the kernel on ALL 5 fitness
		// metrics — including mono (last decision step) and per-episode jerk.
		let mut c2 = test_controller(8, 0xD00D, true);
		let (rows2, asym2) = perturbed_octo();
		let cpu_row = crate::cpu_score::rollout_one(
			&mut c2, &q0, &w0, num_eps, steps,
			SIM_DT, SIM_ARM, SIM_KT, SIM_KD, SIM_INERTIA, SIM_G, [0.0, 0.0, 0.0],
			false, [0.0; 3], 0.0, 0.1, [1.0; 4], 0.0, 0.0, 0.0, 0, 0.0, 0, 0, 0.0,
			4, 8, Some(&rows2), Some(&asym2), None, 1.0, 0.4,
			None,
		);
		for (i, name) in ["reward", "err", "stable", "jerk", "mono"].iter().enumerate() {
			assert_rel_close(rows_gpu[0][i], cpu_row[i], 2e-2, 1e-4,
				&format!("cpu_score {name}"));
		}
		// Allocation-effort metric (row index 12): GPU ↔ CPU scorer parity +
		// non-vacuity (an 8-rotor hoverish rollout has effort ≈ 8·0.25 = 2).
		assert_rel_close(rows_gpu[0][12], cpu_row[12], 2e-2, 1e-3, "cpu_score effort");
		assert!(cpu_row[12] > 0.5, "effort metric vacuous: {}", cpu_row[12]);
		// Transient/display metrics (indices 5..12, implemented on CPU 20/07/2026 —
		// they used to be hardcoded 0.0, so CPU-scored runs reported steady=0.00°).
		// Same kernel definitions ⇒ same tolerance family as the fitness metrics.
		for (i, name) in [(5usize, "steady"), (6, "rise"), (7, "settle_abs"),
		                  (8, "settle_rel"), (9, "itae"), (10, "iae"), (11, "ise")] {
			assert_rel_close(rows_gpu[0][i], cpu_row[i], 3e-2, 1e-4,
				&format!("cpu_score {name}"));
		}
		// Non-vacuity: the OLD behavior (all seven hardcoded 0.0) must fail here.
		// A tilted-IC rollout has real steady-state error and non-zero integrals.
		assert!(cpu_row[5] > 1e-4, "steady vacuous (regressed to hardcoded 0?): {}", cpu_row[5]);
		assert!(cpu_row[9] > 1e-4, "itae vacuous: {}", cpu_row[9]);
		assert!(cpu_row[10] > 1e-4, "iae vacuous: {}", cpu_row[10]);
		assert!(cpu_row[11] > 1e-6, "ise vacuous: {}", cpu_row[11]);
	}

	/// Quad-as-geometry tracks the legacy quad pipeline on the SAME controller
	/// and episodes. Same physics, different float op order (generic r×F vs
	/// the closed-form mixer) ⇒ loose tolerance. Also proves geometry=None
	/// still dispatches the legacy pipeline after the plumbing.
	#[test]
	fn gpu_quad_geometry_tracks_legacy_quad() {
		if Device::system_default().is_none() {
			eprintln!("skipping: no Metal device");
			return;
		}
		let rows = rows_from(&RotorGeometry::quad_plus(SIM_ARM, SIM_KT, SIM_KD));
		let table = build_rotor_table(&rows, None).unwrap();
		let (num_eps, steps) = (24usize, 300usize);
		let (q0, w0) = test_episodes(0x5EED, num_eps);
		let c = test_controller(4, 0xF00D, true);
		let ev = ControllerRolloutEvaluator::new().expect("evaluator");
		let legacy = ev.score(
			&[&c], &q0, &w0, num_eps, steps,
			(SIM_DT, SIM_ARM, SIM_KT, SIM_INERTIA, SIM_G), SIM_KD,
			[0.0, 0.0, 0.0], None, None, None, None,
		).expect("legacy score");
		let geom = ev.score(
			&[&c], &q0, &w0, num_eps, steps,
			(SIM_DT, SIM_ARM, SIM_KT, SIM_INERTIA, SIM_G), SIM_KD,
			[0.0, 0.0, 0.0], None, None, Some(&table), None,
		).expect("geometry score");
		assert_rel_close(geom[0][1], legacy[0][1], 2e-2, 1e-4, "err quad-geom vs legacy");
		assert_rel_close(geom[0][0], legacy[0][0], 2e-2, 1e-4, "reward quad-geom vs legacy");
	}

	/// L1 (06/08/2026): a controller with `obs_dhat` ON must roll out identically on
	/// GPU and CPU. The d̂ observer is a RECURRENCE over the gyro finite-difference and
	/// the throttle accumulator, so a kernel that dropped the state, seeded it wrong, or
	/// mis-ordered the feature would diverge — the exact class the yaw-anchor bug was.
	/// Quad path (obs_dhat requires the '+' mixer inverse), planted cells so the decode
	/// is live, and BOTH the kernel and the production rayon scorer are checked.
	#[test]
	fn gpu_dhat_feature_matches_cpu_closed_loop() {
		if Device::system_default().is_none() {
			eprintln!("skipping: no Metal device");
			return;
		}
		let (num_eps, steps) = (16usize, 300usize);
		let (q0, w0) = test_episodes(0x0D8A, num_eps);
		let b = crate::controller::calibrate_control_gains_rs(
			SIM_DT, SIM_ARM, SIM_KT, SIM_KD, SIM_INERTIA, SIM_G, 0.5, 0.05);
		let mut c = test_controller_dhat(0xDEA7, Some(b));
		// Sanity: the feature is actually on — 9 base + 3 d̂.
		assert_eq!(c.obs_params().0, 12, "obs_dhat must add exactly 3 features");
		let oracle = cpu_oracle_quad(&mut c, &q0, &w0, num_eps, steps);
		let ev = ControllerRolloutEvaluator::new().expect("evaluator");
		let rows_gpu = ev.score(
			&[&c], &q0, &w0, num_eps, steps,
			(SIM_DT, SIM_ARM, SIM_KT, SIM_INERTIA, SIM_G), SIM_KD,
			[0.0, 0.0, 0.0], None, None, None, None,
		).expect("gpu score");
		assert_rel_close(rows_gpu[0][1], oracle[1], 2e-2, 1e-4, "dhat err");
		assert_rel_close(rows_gpu[0][0], oracle[0], 2e-2, 1e-4, "dhat reward");
		assert_rel_close(rows_gpu[0][3], oracle[3], 2e-2, 1e-4, "dhat jerk");
		// Production rayon scorer must agree too (it shares rollout_one with the
		// held-out path, so a divergence here would only surface in a real run).
		let mut c2 = test_controller_dhat(0xDEA7, Some(b));
		let cpu_row = crate::cpu_score::rollout_one(
			&mut c2, &q0, &w0, num_eps, steps,
			SIM_DT, SIM_ARM, SIM_KT, SIM_KD, SIM_INERTIA, SIM_G, [0.0, 0.0, 0.0],
			false, [0.0; 3], 0.0, 0.1, [1.0; 4], 0.0, 0.0, 0.0, 0, 0.0, 0, 0, 0.0,
			4, 4, None, None, None, 1.0, 0.4,
			None,
		);
		assert_rel_close(cpu_row[1], oracle[1], 1e-9, 1e-12, "rollout_one vs oracle err");
		// Non-vacuity: planted cells must move the PWM, and the rollout must show
		// real attitude error — otherwise the parity above proves nothing.
		assert!(oracle[3] > 1e-4, "trivially constant rollout (jerk={})", oracle[3]);
		assert!(oracle[1] > 1e-3, "no attitude error (err={})", oracle[1]);
		// MUTATION GUARD: the d̂ features must actually CHANGE the trajectory. A
		// controller identical but for obs_dhat OFF has a different feature count
		// (9 vs 12) and therefore different addresses — if these agreed, the feature
		// would be inert and every parity assert above would be vacuous.
		let mut c_off = test_controller_dhat(0xDEA7, None);
		assert_eq!(c_off.obs_params().0, 9, "control arm must be the 9-feature anchor");
		let off = cpu_oracle_quad(&mut c_off, &q0, &w0, num_eps, steps);
		assert!((off[1] - oracle[1]).abs() > 1e-6,
			"obs_dhat did not change the rollout (on={} off={}) — feature is inert",
			oracle[1], off[1]);
	}

	/// Quad (no-geometry) twin of cpu_oracle_geometry — the legacy sim.step path.
	fn cpu_oracle_quad(
		c: &mut WnnController, q0: &[f32], omega0: &[f32], num_eps: usize, steps: usize,
	) -> [f64; 5] {
		let mut sim = AttitudeSim::new(SIM_DT, SIM_ARM, SIM_KT, SIM_KD, SIM_INERTIA, SIM_G);
		let (num_motors, levels, ..) = c.gpu_dims();
		let stable_thresh = 5.0_f64.to_radians();
		let (mut sum_reward, mut sum_err, mut sum_jerk, mut sum_mono) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
		let mut n_stable = 0usize;
		for ep in 0..num_eps {
			let q = [q0[ep * 4], q0[ep * 4 + 1], q0[ep * 4 + 2], q0[ep * 4 + 3]];
			let om = [omega0[ep * 3], omega0[ep * 3 + 1], omega0[ep * 3 + 2]];
			c.reset(yaw_from_quat_rs(q));
			sim.reset(Some(q), Some(om));
			let (mut ep_sum_err, mut ep_jerk) = (0.0f64, 0.0f64);
			let mut ep_jerk_count = 0usize;
			let mut prev_pwm = vec![0.5f32; num_motors];
			let (mut first, mut ep_steps, mut diverged) = (true, 0usize, false);
			let mut mono_last = 0.0f64;
			for _t in 0..steps {
				if sim.is_unstable() { diverged = true; break; }
				let (gyro, accel) = sim.read_imu();
				let pwm = c.step(gyro, accel, [0.0, 0.0, 0.0]);
				if !first {
					let mut dj = 0.0f64;
					for m in 0..num_motors {
						let d = (pwm[m] - prev_pwm[m]) as f64;
						dj += d * d;
					}
					ep_jerk += dj.sqrt();
					ep_jerk_count += 1;
				}
				prev_pwm.copy_from_slice(&pwm);
				first = false;
				if let Ok(mv) = monotonicity_violations_core(
					&c.get_last_output_cells(), levels, num_motors,
					c.memory_mode_u8(), c.output_decode_u8()) {
					mono_last = mv as f64;
				}
				sim.step([pwm[0], pwm[1], pwm[2], pwm[3]]);
				let err = sim.attitude_error(None);
				sum_reward += crate::controller::compute_reward(err, 0.0, 0, 0.0, 0.0) as f64;
				ep_sum_err += err as f64;
				ep_steps += 1;
			}
			let mean_err = ep_sum_err / ep_steps.max(1) as f64;
			sum_err += mean_err;
			sum_jerk += if ep_jerk_count > 0 { ep_jerk / ep_jerk_count as f64 } else { 0.0 };
			sum_mono += mono_last;
			if !diverged && mean_err <= stable_thresh { n_stable += 1; }
		}
		let n = num_eps.max(1) as f64;
		[sum_reward / n, sum_err / n, n_stable as f64 / n, sum_jerk / n, sum_mono / n]
	}

	/// Quad test controller with obs_dhat switchable. Same shape as
	/// test_controller_mode but 4 motors and the d̂ feature under test.
	fn test_controller_dhat(seed: u64, dhat_b: Option<[f64; 3]>) -> WnnController {
		let (num_motors, levels, bpf, window, n_state, sbpn, obpn) =
			(4usize, 4usize, 3usize, 2usize, 8usize, 8usize, 8usize);
		let num_features = 9usize + if dhat_b.is_some() { 3 } else { 0 };
		let frame_bits = num_features * bpf;
		let mut rng = SmallRng::seed_from_u64(seed);
		// d̂ is small (rate-accel residual), so its thresholds must straddle a small
		// range or all 3 features would be constant-TRUE and the feature inert.
		let thresholds: Vec<f32> = (0..frame_bits).map(|i| {
			if i >= 9 * bpf { rng.gen_range(-0.5f32..0.5) } else { rng.gen_range(-5.0f32..5.0) }
		}).collect();
		let state_in = window * frame_bits + n_state;
		let state_connections: Vec<i64> =
			(0..n_state * sbpn).map(|_| rng.gen_range(0..state_in) as i64).collect();
		let num_out = num_motors * levels;
		let out_in = frame_bits + n_state;
		let output_connections: Vec<i64> =
			(0..num_out * obpn).map(|_| rng.gen_range(0..out_in) as i64).collect();
		let mut c = WnnController::new_core(
			num_motors, levels, bpf, window, n_state, sbpn, obpn,
			thresholds, state_connections, output_connections,
			false, 0.15, 0.98,
			false, false, false, false, false,
			false, false, false,
			0.99, 1.0, SIM_DT, false, 1,
			2, None,
			dhat_b, 0.05,
		).expect("dhat test controller");
		let mut state_cells = Vec::new();
		for n in 0..n_state {
			for a in 0..(1u64 << sbpn) {
				state_cells.push((n, a, rng.gen_range(0u8..4u8)));
			}
		}
		let mut out_cells = Vec::new();
		for n in 0..num_out {
			for a in 0..(1u64 << obpn) {
				out_cells.push((n, a, rng.gen_range(0u8..4u8)));
			}
		}
		for (n, a, v) in state_cells { let _ = c.write_state_cell_internal(n, a, v); }
		for (n, a, v) in out_cells { let _ = c.write_output_cell_internal(n, a, v); }
		c
	}

	/// ABI 12 mode parity: TERNARY and BINARY controllers (planted mode-native
	/// cells) must roll out identically on GPU and CPU — same fire-bit rule,
	/// same cell weights (TERNARY empty=0.5), same BINARY antagonist decode.
	/// Uses the octo geometry harness so the whole decode+torque path is live.
	#[test]
	fn gpu_mode_parity_ternary_binary_closed_loop() {
		if Device::system_default().is_none() {
			eprintln!("skipping: no Metal device");
			return;
		}
		for mode in [0u8, 3u8] {
			let (rows, asym) = perturbed_octo();
			let table = build_rotor_table(&rows, Some(&asym)).unwrap();
			let (num_eps, steps) = (24usize, 300usize);
			let (q0, w0) = test_episodes(0x0C71 + mode as u64, num_eps);
			let mut c = test_controller_mode(8, 0xD11D + mode as u64, true, mode);
			let oracle = cpu_oracle_geometry(&mut c, &rows, Some(asym), &q0, &w0, num_eps, steps);
			let ev = ControllerRolloutEvaluator::new().expect("evaluator");
			let rows_gpu = ev.score(
				&[&c], &q0, &w0, num_eps, steps,
				(SIM_DT, SIM_ARM, SIM_KT, SIM_INERTIA, SIM_G), SIM_KD,
				[0.0, 0.0, 0.0], None, None, Some(&table), None,
			).expect("gpu score");
			assert_rel_close(rows_gpu[0][0], oracle[0], 2e-2, 1e-4, &format!("mode {mode} reward"));
			assert_rel_close(rows_gpu[0][1], oracle[1], 2e-2, 1e-4, &format!("mode {mode} err"));
			assert_rel_close(rows_gpu[0][2], oracle[2], 0.0, 1.0 / num_eps as f64 + 1e-9,
				&format!("mode {mode} stable rate"));
			assert_rel_close(rows_gpu[0][3], oracle[3], 2e-2, 1e-4, &format!("mode {mode} jerk"));
			// Non-vacuity: planted cells must actually vary the PWM, else the
			// parity proves nothing about the mode decode path.
			assert!(oracle[3] > 1e-4, "mode {mode}: closed loop trivially constant (jerk={})", oracle[3]);
			assert!(oracle[1] > 1e-3, "mode {mode}: no attitude error (err={})", oracle[1]);
		}
	}

	/// ABI 12 neutral invariant: an UNTRAINED TERNARY (empty=0.5) or BINARY
	/// (antagonist ΣE−ΣI=0) controller decodes EXACTLY 0.5 — absolute-mode
	/// hover, and residual anchor 0 by construction.
	#[test]
	fn untrained_ternary_binary_decode_exact_hover() {
		for mode in [0u8, 3u8] {
			let mut c = test_controller_mode(4, 0xAB, false, mode);
			let pwm = c.step([0.01, -0.02, 0.005], [0.1, -0.05, 9.7], [0.0, 0.0, 0.0]);
			for (m, &p) in pwm.iter().enumerate() {
				assert_eq!(p, 0.5,
					"mode {mode} motor {m}: untrained absolute decode must be exactly neutral");
			}
		}
	}

	/// Phase 2 residual→0 sanity: an all-EMPTY controller decodes exactly 0.5
	/// ⇒ residual = 0 ⇒ the composed rollout IS the allocator-LQR baseline.
	/// GPU (in-kernel alloc_step) vs the production CPU scorer must agree, and
	/// the baseline must actually FLY (stable=1 from small tilts) — the
	/// nominal-geometry sanity gate Phase 3's first run rests on.
	#[test]
	fn gpu_alloc_residual_zero_equals_teacher() {
		if Device::system_default().is_none() {
			eprintln!("skipping: no Metal device");
			return;
		}
		let (rows, asym) = perturbed_octo();
		let nominal = rows_from(&RotorGeometry::octo_x(SIM_ARM, SIM_KT, SIM_KD));
		let table = build_rotor_table(&rows, Some(&asym)).unwrap();
		let ab = crate::optimal::AllocBaseline::build(
			&nominal, SIM_INERTIA, 12.0, 1.0, 1.0, 0.144, None, 1e-6).unwrap();
		// 2500 steps (2.5 s): the mean err must be dominated by the SETTLED
		// phase, not the 17°-tilt transient, for the stable=1.0 gate below.
		let (num_eps, steps) = (12usize, 2500usize);
		let (q0, w0) = test_episodes(0xA110C, num_eps);
		let mut c = test_controller(8, 0xE0, false);   // EMPTY memories → residual 0
		let residual = ResidualCfg { scale: 1.0, clamp: 0.4, pid: [0.0; 10], cascade: None };
		let ev = ControllerRolloutEvaluator::new().expect("evaluator");
		let gpu = ev.score(
			&[&c], &q0, &w0, num_eps, steps,
			(SIM_DT, SIM_ARM, SIM_KT, SIM_INERTIA, SIM_G), SIM_KD,
			[0.0, 0.0, 0.0], None, Some(residual), Some(&table), Some(&ab),
		).expect("gpu score");
		let cpu = crate::cpu_score::rollout_one(
			&mut c, &q0, &w0, num_eps, steps,
			SIM_DT, SIM_ARM, SIM_KT, SIM_KD, SIM_INERTIA, SIM_G, [0.0, 0.0, 0.0],
			false, [0.0; 3], 0.0, 0.1, [1.0; 4], 0.0, 0.0, 0.0, 0, 0.0, 0, 0, 0.0,
			4, 8, Some(&rows), Some(&asym), Some(&ab), 1.0, 0.4,
			None,
		);
		// f32 kernel euler vs f64 CPU euler drifts ~0.006° over 2500 steps —
		// the 3e-4 abs floor absorbs that; anything structural is far larger.
		for (i, name) in ["reward", "err", "stable", "jerk", "mono"].iter().enumerate() {
			assert_rel_close(gpu[0][i], cpu[i], 1e-2, 3e-4, &format!("alloc-zero {name}"));
		}
		// The baseline must stabilize the perturbed vehicle from ≤17° tilts.
		assert_eq!(gpu[0][2], 1.0, "alloc-LQR baseline failed to stabilize (stable={})", gpu[0][2]);
		assert!(gpu[0][1] > 1e-4, "vacuous rollout (err={})", gpu[0][1]);
	}

	/// Phase 2 residual parity with a LIVE residual: planted cells make the
	/// WNN push nonzero Δu on the alloc baseline — GPU vs the production CPU
	/// scorer on all 5 fitness metrics (discrete decisions ⇒ loose 2%).
	#[test]
	fn gpu_alloc_residual_matches_cpu_scorer() {
		if Device::system_default().is_none() {
			eprintln!("skipping: no Metal device");
			return;
		}
		let (rows, asym) = perturbed_octo();
		let nominal = rows_from(&RotorGeometry::octo_x(SIM_ARM, SIM_KT, SIM_KD));
		let table = build_rotor_table(&rows, Some(&asym)).unwrap();
		let ab = crate::optimal::AllocBaseline::build(
			&nominal, SIM_INERTIA, 12.0, 1.0, 1.0, 0.144, None, 1e-6).unwrap();
		let (num_eps, steps) = (24usize, 300usize);
		let (q0, w0) = test_episodes(0xA111, num_eps);
		let mut c = test_controller(8, 0xA5, true);    // planted → live residual
		let residual = ResidualCfg { scale: 0.5, clamp: 0.15, pid: [0.0; 10], cascade: None };
		let ev = ControllerRolloutEvaluator::new().expect("evaluator");
		let gpu = ev.score(
			&[&c], &q0, &w0, num_eps, steps,
			(SIM_DT, SIM_ARM, SIM_KT, SIM_INERTIA, SIM_G), SIM_KD,
			[0.0, 0.0, 0.0], None, Some(residual), Some(&table), Some(&ab),
		).expect("gpu score");
		let cpu = crate::cpu_score::rollout_one(
			&mut c, &q0, &w0, num_eps, steps,
			SIM_DT, SIM_ARM, SIM_KT, SIM_KD, SIM_INERTIA, SIM_G, [0.0, 0.0, 0.0],
			false, [0.0; 3], 0.0, 0.1, [1.0; 4], 0.0, 0.0, 0.0, 0, 0.0, 0, 0, 0.0,
			4, 8, Some(&rows), Some(&asym), Some(&ab), 0.5, 0.15,
			None,
		);
		for (i, name) in ["reward", "err", "stable", "jerk", "mono"].iter().enumerate() {
			assert_rel_close(gpu[0][i], cpu[i], 2e-2, 2e-3, &format!("alloc-residual {name}"));
		}
		assert!(cpu[3] > 1e-4, "residual never moved the PWM (jerk={})", cpu[3]);
	}

	/// Geometry misuse fails loudly (CPU-fallback contract): rotor-count vs
	/// num_motors mismatch, and residual hybrid on a non-quad geometry.
	#[test]
	fn geometry_guards_fail_loudly() {
		if Device::system_default().is_none() {
			eprintln!("skipping: no Metal device");
			return;
		}
		let (rows, _) = perturbed_octo();
		let table = build_rotor_table(&rows, None).unwrap();
		let (q0, w0) = test_episodes(1, 2);
		let ev = ControllerRolloutEvaluator::new().expect("evaluator");
		// 8-rotor table, 4-motor controller → mismatch.
		let c4 = test_controller(4, 42, false);
		assert!(ev.score(
			&[&c4], &q0, &w0, 2, 10,
			(SIM_DT, SIM_ARM, SIM_KT, SIM_INERTIA, SIM_G), SIM_KD,
			[0.0, 0.0, 0.0], None, None, Some(&table), None,
		).is_err(), "rotor/motor mismatch must be refused");
		// Residual hybrid (quad PID baseline) on an N=8 geometry → refused.
		let c8 = test_controller(8, 43, false);
		let residual = ResidualCfg { scale: 1.0, clamp: 0.4, pid: [1.2, 0.0, 0.3, 0.5, 0.6, 0.0, 0.2, 0.5, 0.5, 0.4], cascade: None };
		assert!(ev.score(
			&[&c8], &q0, &w0, 2, 10,
			(SIM_DT, SIM_ARM, SIM_KT, SIM_INERTIA, SIM_G), SIM_KD,
			[0.0, 0.0, 0.0], None, Some(residual), Some(&table), None,
		).is_err(), "residual + N≠4 geometry must be refused");
	}
}

// ---------------------------------------------------------------------------
// GPU SOLVE HANDLE for the bptt walk.
//
// bptt_train_window calls the solver once per (record, motor); constructing a Metal
// device + pipelines per call would dwarf the work. One handle is built lazily on
// first use and reused for the process. It is NOT mutable state — it is a device
// handle cache, immutable after construction, which is why a OnceLock is the right
// shape rather than the mutable global the style rules rule out.
//
// GATED, DEFAULT OFF (WNN_CONTROLLER_GPU_SOLVE=1). Per-solve dispatch is the known
// anti-pattern — thousands of tiny launches against a serial dependency — so this is
// correctness-complete, not yet throughput-complete: it proves the walk produces a
// bit-identical cell function on the GPU, and batching across (genome x motor x record)
// is the performance step that follows. Enabling it by default before that would be a
// regression, so it stays opt-in and the default path is untouched.
static GPU_SOLVER: std::sync::OnceLock<Option<ControllerTrainer>> = std::sync::OnceLock::new();

pub(crate) fn gpu_solver() -> Option<&'static ControllerTrainer> {
	GPU_SOLVER.get_or_init(|| {
		if std::env::var("WNN_CONTROLLER_GPU_SOLVE").map(|s| s == "1").unwrap_or(false) {
			match ControllerTrainer::new() {
				Ok(t) => Some(t),
				Err(e) => { eprintln!("[controller] GPU solve requested but unavailable: {e}"); None }
			}
		} else { None }
	}).as_ref()
}

// ---------------------------------------------------------------------------
// GENOME-LEVEL COALESCING — the last factor of sync amortisation.
//
// Measured ladder: dispatch+sync is ~81% of GPU time; batching the 4 motors bought
// 3.79x, and issuing (a)+(b) in one command buffer halved syncs again. Everything
// available WITHIN a record is now taken. The next factor can only come from outside
// it, because records are serially dependent (d's result feeds d-1).
//
// It does NOT need bptt_train_window restructured into a lockstep driver. The genomes
// are ALREADY concurrent — dagger_train_batch_inplace runs them under rayon par_iter —
// so several are inside their walks at any instant, each currently paying its own
// commit+wait. Coalescing those into one command buffer is the same win as lockstep
// with none of the control-flow inversion, and it degrades gracefully: with one genome
// in flight a "batch" is just that genome, exactly today's behaviour.
//
// LEADER/FOLLOWER, NO WAITING. A thread enqueues its work and then either finds the GPU
// idle and dispatches EVERYTHING queued at that moment (itself included), or waits for
// whoever is already dispatching. Nobody ever waits *for more work to arrive*, which is
// what makes this deadlock-free regardless of how many genomes are live: a batch of one
// is legal and immediate. Batch size is therefore emergent — however many threads
// happened to be queued — rather than a tuned constant that could stall on a small pop.
#[derive(Clone)]
pub(crate) struct OwnedLayer {
	pub keys: Vec<u64>,
	pub values: Vec<u8>,
	pub offsets: Vec<u32>,
	pub counts: Vec<u32>,
	pub conns: Vec<i64>,
	pub num_inst: usize,
	pub neurons_per_inst: usize,
	pub n_bits: usize,
	pub total_input_bits: usize,
	pub input_bits: Vec<bool>,
	pub target_bits: Vec<bool>,
	pub n_immutable_bits: usize,
}

type SolveOut = Vec<Vec<Option<Vec<bool>>>>;

/// A queued solve. The RETURN PATH TRAVELS WITH THE REQUEST — `tx` is this submitter's
/// own channel, created by it and moved in with its data.
///
/// This is the structural fix for the coalescer's one dangerous failure mode. An earlier
/// design routed results through a `HashMap<id, result>`, which is correct only while the
/// id bookkeeping is: a mismatched key would deliver one genome's solved bits to another,
/// training it on a foreign answer with no error raised anywhere. Here there is no id and
/// no lookup — a result can only reach the submitter whose sender came attached to the
/// job. Delivering to the wrong thread is not a bug that has been tested for; it is a
/// state that cannot be represented.
struct Job {
	layers: Vec<OwnedLayer>,
	topk: usize,
	mode: u8,
	tx: std::sync::mpsc::Sender<SolveOut>,
}

struct CoalescerInner {
	queue: Vec<Job>,
	busy: bool,
}

pub(crate) struct SolveCoalescer {
	m: std::sync::Mutex<CoalescerInner>,
}

impl SolveCoalescer {
	fn new() -> Self {
		SolveCoalescer { m: std::sync::Mutex::new(CoalescerInner { queue: Vec::new(), busy: false }) }
	}

	/// Submit and block until this submitter's own result arrives.
	///
	/// Enqueue and the leader test happen under ONE lock acquisition, which is what makes
	/// the handoff race-free: a thread either finds the GPU idle and takes leadership, or
	/// it is already in a queue that a live leader is guaranteed to drain. The leader
	/// keeps draining until the queue is empty before releasing the flag, so a job that
	/// arrives mid-batch is picked up by the next iteration rather than stranded.
	///
	/// Nobody ever waits for MORE work to arrive — a batch of one dispatches immediately.
	/// That is what keeps a small or single-genome population at exactly today's latency.
	pub(crate) fn solve(
		&self, t: &ControllerTrainer, layers: Vec<OwnedLayer>, topk: usize, mode: u8,
	) -> Result<SolveOut, String> {
		let (tx, rx) = std::sync::mpsc::channel();
		let lead = {
			let mut g = self.m.lock().map_err(|e| format!("coalescer poisoned: {e}"))?;
			g.queue.push(Job { layers, topk, mode, tx });
			if g.busy { false } else { g.busy = true; true }
		};
		if lead {
			loop {
				let batch = {
					let mut g = self.m.lock().map_err(|e| format!("coalescer poisoned: {e}"))?;
					if g.queue.is_empty() { g.busy = false; break; }
					std::mem::take(&mut g.queue)
				};
				Self::dispatch_batch(t, batch);
			}
		}
		// Leader included: everyone collects from their OWN channel.
		rx.recv().map_err(|e| format!("coalescer sender dropped before delivering: {e}"))
	}

	/// Flatten every job's layers into ONE command buffer, then hand each job its slice
	/// back down its own sender. One sync for the whole batch — the entire point.
	fn dispatch_batch(t: &ControllerTrainer, batch: Vec<Job>) {
		// topk/mode are run-level constants in practice, so this is normally ONE group.
		// Partitioning anyway keeps a mixed batch correct instead of silently applying
		// one job's parameters to another's data.
		let mut groups: std::collections::HashMap<(usize, u8), Vec<Job>> =
			std::collections::HashMap::new();
		for j in batch { groups.entry((j.topk, j.mode)).or_default().push(j); }

		for ((topk, mode), jobs) in groups {
			let mut flat: Vec<SolveLayer> = Vec::new();
			for j in &jobs {
				for l in &j.layers {
					flat.push(SolveLayer {
						keys: &l.keys, values: &l.values, offsets: &l.offsets, counts: &l.counts,
						conns: &l.conns, num_inst: l.num_inst, neurons_per_inst: l.neurons_per_inst,
						n_bits: l.n_bits, total_input_bits: l.total_input_bits,
						input_bits: &l.input_bits, target_bits: &l.target_bits,
						n_immutable_bits: l.n_immutable_bits,
					});
				}
			}
			let res = t.solve_layers(&flat, topk, mode);
			let mut k = 0usize;
			for j in &jobs {
				let n = j.layers.len();
				let out = match &res {
					Ok(r) => r[k..k + n].to_vec(),
					// A failed batch degrades every member to "no GPU result" (empty layer
					// vecs), which the walk reads as "use the CPU path". One failure must
					// not poison its batch-mates, and must not strand them either.
					Err(_) => vec![Vec::new(); n],
				};
				k += n;
				// Send failure means the submitter is gone; nothing to do and nothing to
				// leak — the result is simply dropped.
				let _ = j.tx.send(out);
			}
		}
	}
}

pub(crate) fn solve_coalescer() -> &'static SolveCoalescer {
	static C: std::sync::OnceLock<SolveCoalescer> = std::sync::OnceLock::new();
	C.get_or_init(SolveCoalescer::new)
}
