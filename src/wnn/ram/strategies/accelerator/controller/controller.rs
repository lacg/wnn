//! Drone-controller hot-path: attitude simulator, WNN controller wrapper,
//! Strategy-5 QSR-weighted decoder, monotonicity violation count, and
//! reward computation.
//!
//! Per the design memory `project_drone_controller_paper1.md`:
//! - `AttitudeSim`: rigid-body rotational dynamics + motor torques (no
//!   translation; attitude stabilization only for paper #1).
//! - `WnnController`: stateful per-step wrapper that thermometer-encodes
//!   IMU + setpoint, forwards through a `RAMRecurrentNetwork`, decodes
//!   Strategy-5, and emits 4 normalized motor PWM in [0, 1].
//! - `strategy_5_qsr_weighted`: per-motor decode of 256 raw QSR cell
//!   values (0..3) into a bucket index in [0, 255] via the weighted-sum
//!   mapping FALSE=0.0, WEAK_FALSE=0.25, WEAK_TRUE=0.75, TRUE=1.0.
//! - `monotonicity_violations`: counts thermometer-pattern violations
//!   for the soft regularizer in the training reward.
//! - `compute_reward`: scalar reward = -attitude_error²
//!                       - λ_smooth × motor_command_jerk²
//!                       - λ_mono × monotonicity_violations
//!
//! This file is the in-Rust hot path. The Python side
//! (`src/wnn/control/`) holds the GA orchestration + episode runner.
//!
//! STATUS: sim physics LIVE; controller forward + thermometer encoding
//! still TODO. AttitudeSim integrates real rigid-body rotational dynamics
//! via RK4. Decoders + reward + monotonicity are pure-data utilities that
//! work end-to-end.

use std::collections::VecDeque;

use pyo3::prelude::*;

use ram_core::neuron_memory::compute_address_sparse;
use ram_core::sparse_memory::SparseLayerMemory;
use crate::controller_training::{solve_partial_connectivity_qsr_reachable, nudge_toward_value};

// Strategy-5 QSR weight lookup = the canonical QUAD table (single source of
// truth in neuron_memory.rs; the GPU twin lives in shaders/common.metal).
use ram_core::neuron_memory::QUAD_WEIGHTS as QSR_WEIGHTS;

// Neutral decode = the untrained-cell decode value, DERIVED from the active
// cell semantics (Luiz 12/07/2026): unwritten sparse cells read EMPTY_U8 and
// decode to QSR_WEIGHTS[EMPTY_U8] — 0.75 under QUAD; a TERNARY substrate
// (empty→0.5) would yield 0.5 automatically. Single anchor for BOTH
// delta-control (delta=0 ⇒ untrained controller HOLDS throttle — the stable
// bootstrap) and residual composition (residual=0 ⇒ untrained hybrid IS the
// analytic baseline EXACTLY — pre-ABI-11 anchored at a hardcoded 0.5, which
// silently composed a +clamp collective offset).
pub(crate) const NEUTRAL_DECODE: f32 =
	QSR_WEIGHTS[ram_core::neuron_memory::EMPTY_U8 as usize];

/// Map a Strategy-5 decode in [0,1] to a per-step PWM delta in
/// [-delta_max, +delta_max], piecewise-linear with neutral at NEUTRAL_DECODE
/// (so both decode halves reach the full ± range).
#[inline]
fn decoded_to_delta(decoded: f32, delta_max: f32) -> f32 {
	let n = NEUTRAL_DECODE;
	if decoded >= n {
		(decoded - n) / (1.0 - n) * delta_max
	} else {
		(decoded - n) / n * delta_max
	}
}

/// Inverse of decoded_to_delta: the decode target that yields a desired delta.
/// Used to turn the teacher's (target_pwm - current_pwm) into an output target.
#[inline]
fn delta_to_decoded(delta: f32, delta_max: f32) -> f32 {
	let n = NEUTRAL_DECODE;
	let d = delta.clamp(-delta_max, delta_max);
	if d >= 0.0 {
		n + d / delta_max * (1.0 - n)
	} else {
		n + d / delta_max * n
	}
}

// Controller input feature layout used by WnnController::step():
//   features 0..3 = body-frame gyro (rad/s)
//   features 3..6 = body-frame specific force (m/s²)
//   features 6..9 = target attitude RPY (rad)
const NUM_FEATURES: usize = 9;

// =============================================================================
// Quaternion + vector math helpers (private; not exposed via PyO3).
//
// Convention (right-hand frame, z-up body):
//   q = (w, x, y, z) unit quaternion, body-to-world.
//   q = identity (1, 0, 0, 0) means body is level with world.
// Body axes:
//   +x = forward, +y = left, +z = up (right-handed).
// =============================================================================

#[inline]
fn q_normalize(q: [f32; 4]) -> [f32; 4] {
	let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
	if n > 0.0 { [q[0] / n, q[1] / n, q[2] / n, q[3] / n] } else { [1.0, 0.0, 0.0, 0.0] }
}

#[inline]
fn q_conjugate(q: [f32; 4]) -> [f32; 4] {
	[q[0], -q[1], -q[2], -q[3]]
}

/// Hamilton product a ⊗ b (in w, x, y, z order).
#[inline]
fn q_multiply(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
	let (aw, ax, ay, az) = (a[0], a[1], a[2], a[3]);
	let (bw, bx, by, bz) = (b[0], b[1], b[2], b[3]);
	[
		aw * bw - ax * bx - ay * by - az * bz,
		aw * bx + ax * bw + ay * bz - az * by,
		aw * by - ax * bz + ay * bw + az * bx,
		aw * bz + ax * by - ay * bx + az * bw,
	]
}

/// Rotate a 3-vector v in WORLD frame to BODY frame using q (body-to-world).
/// v_body = q* · v_world · q  (with v promoted to pure quaternion (0, vx, vy, vz)).
#[inline]
fn rotate_world_to_body(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
	let v_q = [0.0, v[0], v[1], v[2]];
	let tmp = q_multiply(q_conjugate(q), v_q);
	let res = q_multiply(tmp, q);
	[res[1], res[2], res[3]]
}

#[inline]
fn vec_add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
	[a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
fn vec_scale3(v: [f32; 3], s: f32) -> [f32; 3] {
	[v[0] * s, v[1] * s, v[2] * s]
}

#[inline]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
	[
		a[1] * b[2] - a[2] * b[1],
		a[2] * b[0] - a[0] * b[2],
		a[0] * b[1] - a[1] * b[0],
	]
}

#[inline]
fn q_add(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
	[a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}

#[inline]
fn q_scale(q: [f32; 4], s: f32) -> [f32; 4] {
	[q[0] * s, q[1] * s, q[2] * s, q[3] * s]
}

// =============================================================================
// Disturbances (W2 — weather). SINGLE SOURCE OF TRUTH for the counter-based
// noise RNG; the Metal twin lives in shaders/controller_rollout.metal and MUST
// mirror the integer path bit-for-bit (the `should_skip_sample` precedent,
// shaders/marker_train.metal:87).
//
// Every stochastic draw is a PURE FUNCTION of (seed, step_idx, axis, channel)
// — no stateful RNG to sync between CPU and GPU. Gaussian draws use Box-Muller
// over two consecutive hashed uniforms (sub-draw k ∈ {0, 1}).
//
// CHANNEL IDS (canonical registry — do not renumber; Metal mirrors these):
//   0 = D2 gust OU innovation            (axis 0..2, drawn post-step)
//   1 = D4 gyro white noise              (axis 0..2, drawn at read_imu)
//   2 = D4 gyro bias random walk         (axis 0..2, drawn post-step)
//   3 = D4 accel white noise             (axis 0..2, drawn at read_imu)
//  15 = per-episode seed derivation      (axis 0, k 0; step = episode index)
// =============================================================================

pub const DIST_CH_GUST: u32 = 0;
pub const DIST_CH_GYRO: u32 = 1;
pub const DIST_CH_GYRO_BIAS: u32 = 2;
pub const DIST_CH_ACCEL: u32 = 3;
pub const DIST_CH_EP_SEED: u32 = 15;

// 2π as the SAME f32 literal used in the Metal twin (DIST_TWO_PI) so Box-Muller
// is computed from identical constants on both paths.
const DIST_TWO_PI: f32 = 6.283_185_5;

/// Fold the u64 disturbance seed into the u32 the counter hash consumes.
/// Hosts that pre-fold (metal_controller) and the sim itself must agree.
#[inline]
pub fn dist_seed32(seed: u64) -> u32 {
	(seed ^ (seed >> 32)) as u32
}

/// xorshift32 counter hash of (seed, step, axis, channel, sub-draw k) — same
/// style as `should_skip_sample` (marker_train.metal:87 / tiered_sparse.rs):
/// linear index mixing with large odd constants, then xorshift32.
#[inline]
pub fn dist_hash_u32(seed: u32, step: u32, axis: u32, channel: u32, k: u32) -> u32 {
	let mut rng = seed
		.wrapping_add(step.wrapping_mul(2_654_435_761))
		.wrapping_add(axis.wrapping_mul(1_000_003))
		.wrapping_add(channel.wrapping_mul(97_780_813))
		.wrapping_add(k.wrapping_mul(668_265_263));
	if rng == 0 { rng = 1; }
	rng ^= rng << 13;
	rng ^= rng >> 17;
	rng ^= rng << 5;
	rng
}

/// Uniform in [0, 1) from the counter hash (top 24 bits, like should_skip_sample).
#[inline]
fn dist_uniform(seed: u32, step: u32, axis: u32, channel: u32, k: u32) -> f32 {
	(dist_hash_u32(seed, step, axis, channel, k) >> 8) as f32 / 16_777_216.0
}

/// Standard-normal draw via Box-Muller over sub-draws k=0 (radius) and k=1
/// (angle). u1 is clamped away from 0 so ln() is finite (same clamp in Metal).
#[inline]
pub fn dist_gauss(seed: u32, step: u32, axis: u32, channel: u32) -> f32 {
	let u1 = dist_uniform(seed, step, axis, channel, 0).max(1e-7);
	let u2 = dist_uniform(seed, step, axis, channel, 1);
	(-2.0 * u1.ln()).sqrt() * (DIST_TWO_PI * u2).cos()
}

/// Per-episode disturbance seed from a base seed + episode index — the SAME
/// derivation the Metal rollout kernel uses (channel 15), exposed to Python so
/// hosts/tests never re-implement the hash. Batched paths (Metal scoring,
/// eval_ensemble) use this; run_episode instead draws its per-episode seed from
/// the numpy episode rng (documented there).
#[pyfunction]
pub fn disturbance_episode_seed(seed: u64, episode_idx: u64) -> u64 {
	dist_hash_u32(dist_seed32(seed), episode_idx as u32, 0, DIST_CH_EP_SEED, 0) as u64
}

/// W2 disturbance parameters (all four primitives). Plain data; `AttitudeSim`
/// holds an `Option<Disturbance>` — None ⇒ the bit-identical clean sim.
#[derive(Clone, Copy, Debug)]
pub struct Disturbance {
	// D1: constant body-frame torque bias (N·m) — the integrator test.
	pub tau_bias: [f32; 3],
	// D2: Ornstein-Uhlenbeck gust torque per axis:
	//     gust += -gust/tau_c·dt + sigma·sqrt(dt)·ξ  (updated AFTER use).
	pub gust_sigma: f32,
	pub gust_tau_c: f32,
	// D3: per-motor k_thrust multipliers (1.0 = clean motor).
	pub motor_asym: [f32; 4],
	// D4: sensor noise — gyro white σ + slow bias walk; accel white σ.
	pub gyro_sigma: f32,
	pub gyro_bias_walk: f32,
	pub accel_sigma: f32,
	// Base seed for the counter RNG (folded to u32 via dist_seed32).
	pub seed: u64,
}

// =============================================================================
// AttitudeSim
// =============================================================================
//
// '+' quadcopter motor layout (body frame, z-up, x forward, y left):
//   Motor 0 = front, at  ( +L,  0, 0)
//   Motor 1 = right, at  (  0, -L, 0)
//   Motor 2 = rear,  at  ( -L,  0, 0)
//   Motor 3 = left,  at  (  0, +L, 0)
//
// Each motor produces upward (+z) thrust = k_thrust × pwm². Body-frame torque
// per motor i is r_i × F_i. Yaw drag torque comes from prop counter-rotation:
//   Motors 0, 2 spin CCW (+z aerodynamic drag torque on the airframe)
//   Motors 1, 3 spin CW  (-z aerodynamic drag torque on the airframe)
// Net yaw torque ∝ k_drag × (T_0 - T_1 + T_2 - T_3).

#[pyclass]
pub struct AttitudeSim {
	// Unit quaternion (w, x, y, z), body-to-world. Identity = level.
	q: [f32; 4],
	// Angular velocity in body frame (rad/s).
	omega: [f32; 3],
	// Simulator time (s).
	t: f32,
	// Integration step (s). Default 1 ms = 1 kHz update.
	dt: f32,

	// Physical parameters (defaults model a ~250 g class quadcopter).
	arm_length: f32,        // motor-to-CG distance L (m)
	k_thrust: f32,          // N per pwm² unit (so pwm=1.0 → k_thrust N per motor)
	k_drag: f32,            // yaw-drag-torque to thrust ratio (dimensionless)
	inertia: [f32; 3],      // diagonal inertia tensor (Ixx, Iyy, Izz) in kg·m²
	gravity: f32,           // m/s² (default 9.81)

	// --- W2 disturbances (None = bit-identical clean sim; the hot loops
	//     branch on the Option BEFORE touching torque/IMU floats). ---
	dist: Option<Disturbance>,
	// D2 OU gust torque state (N·m, body frame). Zeroed at reset().
	gust: [f32; 3],
	// D4 gyro bias-walk state (rad/s). Zeroed at reset().
	gyro_bias: [f32; 3],
	// Physical-step counter driving the counter-based noise RNG. Incremented
	// once per step(); read_imu() at step t and the post-step updates of step t
	// both draw with step_idx = t. Zeroed at reset().
	step_idx: u64,

	// --- Overactuated Phase 1 (None = legacy quad, bit-identical path). ---
	// N-rotor geometry consumed by step_n(); step() never reads it. Persists
	// across reset() like `dist`. See docs/OVERACTUATED_RESIDUAL_DESIGN.md.
	geometry: Option<crate::overactuated::RotorGeometry>,
	// Per-rotor thrust multipliers for the geometry path (N-rotor D3 twin of
	// Disturbance.motor_asym, which stays quad-only).
	rotor_asym: Option<Vec<f32>>,
}

// Overactuated Phase-1 core (plain Rust, String errors): keeps the fallible
// logic out of the PyO3 error machinery so cargo tests can exercise it
// without linking libpython (house pattern — pymethods are thin wrappers).
impl AttitudeSim {
	pub(crate) fn set_geometry_core(&mut self, rotors: Vec<[f32; 9]>) -> Result<(), String> {
		let geo = crate::overactuated::RotorGeometry::from_rows(&rotors)?;
		if self.rotor_asym.as_ref().is_some_and(|a| a.len() != geo.num_rotors()) {
			self.rotor_asym = None;
		}
		self.geometry = Some(geo);
		Ok(())
	}

	pub(crate) fn perturb_geometry_core(&mut self, tilt_err_deg: Vec<f32>, pos_err: Vec<[f32; 3]>) -> Result<(), String> {
		let Some(geo) = &self.geometry else {
			return Err("no geometry set".into());
		};
		let tilt_rad: Vec<f32> = tilt_err_deg.iter().map(|d| d.to_radians()).collect();
		self.geometry = Some(geo.perturbed(&tilt_rad, &pos_err));
		Ok(())
	}

	pub(crate) fn set_rotor_asym_core(&mut self, asym: Option<Vec<f32>>) -> Result<(), String> {
		if let (Some(a), Some(g)) = (&asym, &self.geometry) {
			if a.len() != g.num_rotors() {
				return Err(format!("rotor_asym len {} != num_rotors {}", a.len(), g.num_rotors()));
			}
		}
		self.rotor_asym = asym;
		Ok(())
	}

	pub(crate) fn step_n_core(&mut self, motor_pwm: &[f32]) -> Result<(), String> {
		let Some(geo) = &self.geometry else {
			if motor_pwm.len() != 4 {
				return Err(format!("no geometry set: expected 4 PWMs, got {}", motor_pwm.len()));
			}
			self.step([motor_pwm[0], motor_pwm[1], motor_pwm[2], motor_pwm[3]]);
			return Ok(());
		};
		if motor_pwm.len() != geo.num_rotors() {
			return Err(format!("expected {} PWMs, got {}", geo.num_rotors(), motor_pwm.len()));
		}
		// Torque: generic geometry model (pwm clamped inside), then the same
		// disturbance composition as step() (D3-twin rotor_asym is folded
		// into the thrust model; D1 bias + D2 gust add on top).
		let base = geo.body_torque_asym(motor_pwm, self.rotor_asym.as_deref());
		let torque = match self.dist {
			None => base,
			Some(d) => [
				base[0] + d.tau_bias[0] + self.gust[0],
				base[1] + d.tau_bias[1] + self.gust[1],
				base[2] + d.tau_bias[2] + self.gust[2],
			],
		};
		let dt = self.dt;

		// RK4 on state y = (omega: 3, q: 4) — lockstep copy of step()'s body;
		// any integrator change must be applied to both.
		let (k1o, k1q) = self.derivatives(self.omega, self.q, torque);
		let omega2 = vec_add3(self.omega, vec_scale3(k1o, dt * 0.5));
		let q2 = q_add(self.q, q_scale(k1q, dt * 0.5));
		let (k2o, k2q) = self.derivatives(omega2, q2, torque);
		let omega3 = vec_add3(self.omega, vec_scale3(k2o, dt * 0.5));
		let q3 = q_add(self.q, q_scale(k2q, dt * 0.5));
		let (k3o, k3q) = self.derivatives(omega3, q3, torque);
		let omega4 = vec_add3(self.omega, vec_scale3(k3o, dt));
		let q4 = q_add(self.q, q_scale(k3q, dt));
		let (k4o, k4q) = self.derivatives(omega4, q4, torque);
		let omega_delta = vec_scale3(
			vec_add3(
				vec_add3(k1o, vec_scale3(k2o, 2.0)),
				vec_add3(vec_scale3(k3o, 2.0), k4o),
			),
			dt / 6.0,
		);
		let q_delta = q_scale(
			q_add(
				q_add(k1q, q_scale(k2q, 2.0)),
				q_add(q_scale(k3q, 2.0), k4q),
			),
			dt / 6.0,
		);
		self.omega = vec_add3(self.omega, omega_delta);
		self.q = q_normalize(q_add(self.q, q_delta));
		self.t += dt;

		// W2 post-step updates — lockstep copy of step()'s tail.
		if let Some(d) = self.dist {
			let s32 = dist_seed32(d.seed);
			let t32 = self.step_idx as u32;
			let sqrt_dt = dt.sqrt();
			for a in 0..3 {
				if d.gust_sigma > 0.0 {
					let xi = dist_gauss(s32, t32, a as u32, DIST_CH_GUST);
					self.gust[a] += -self.gust[a] / d.gust_tau_c * dt
						+ d.gust_sigma * sqrt_dt * xi;
				}
				if d.gyro_bias_walk > 0.0 {
					let xi = dist_gauss(s32, t32, a as u32, DIST_CH_GYRO_BIAS);
					self.gyro_bias[a] += d.gyro_bias_walk * sqrt_dt * xi;
				}
			}
		}
		self.step_idx += 1;
		Ok(())
	}
}

#[pymethods]
impl AttitudeSim {
	#[new]
	#[pyo3(signature = (
		dt = 0.001,
		arm_length = 0.075,
		k_thrust = 2.4,
		k_drag = 0.05,
		inertia = [0.0023, 0.0023, 0.0046],
		gravity = 9.81,
	))]
	pub fn new(
		dt: f32,
		arm_length: f32,
		k_thrust: f32,
		k_drag: f32,
		inertia: [f32; 3],
		gravity: f32,
	) -> Self {
		Self {
			q: [1.0, 0.0, 0.0, 0.0],
			omega: [0.0, 0.0, 0.0],
			t: 0.0,
			dt,
			arm_length,
			k_thrust,
			k_drag,
			inertia,
			gravity,
			dist: None,
			gust: [0.0, 0.0, 0.0],
			gyro_bias: [0.0, 0.0, 0.0],
			step_idx: 0,
			geometry: None,
			rotor_asym: None,
		}
	}

	/// Reset the simulator. Optional initial quaternion (defaults to identity)
	/// and initial angular velocity (defaults to zero). Disturbance PARAMS
	/// (dist) persist across reset; the disturbance STATE (gust, gyro bias,
	/// step counter) is zeroed so every episode starts clean-and-reproducible.
	#[pyo3(signature = (q = None, omega = None))]
	pub fn reset(&mut self, q: Option<[f32; 4]>, omega: Option<[f32; 3]>) {
		self.q = q_normalize(q.unwrap_or([1.0, 0.0, 0.0, 0.0]));
		self.omega = omega.unwrap_or([0.0, 0.0, 0.0]);
		self.t = 0.0;
		self.gust = [0.0, 0.0, 0.0];
		self.gyro_bias = [0.0, 0.0, 0.0];
		self.step_idx = 0;
	}

	/// Enable W2 disturbances (D1 τ-bias, D2 OU gusts, D3 motor asymmetry,
	/// D4 sensor noise). Explicit typed params — see `Disturbance` for units.
	/// Resets the disturbance state (gust/bias/step counter) so the noise
	/// stream restarts from (seed, step 0). `seed` should be a PER-EPISODE
	/// seed (hosts derive it via `disturbance_episode_seed` or the episode rng).
	#[allow(clippy::too_many_arguments)]
	#[pyo3(signature = (
		tau_bias = [0.0, 0.0, 0.0],
		gust_sigma = 0.0,
		gust_tau_c = 0.1,
		motor_asym = [1.0, 1.0, 1.0, 1.0],
		gyro_sigma = 0.0,
		gyro_bias_walk = 0.0,
		accel_sigma = 0.0,
		seed = 0,
	))]
	pub fn set_disturbance(
		&mut self,
		tau_bias: [f32; 3],
		gust_sigma: f32,
		gust_tau_c: f32,
		motor_asym: [f32; 4],
		gyro_sigma: f32,
		gyro_bias_walk: f32,
		accel_sigma: f32,
		seed: u64,
	) {
		self.dist = Some(Disturbance {
			tau_bias, gust_sigma, gust_tau_c, motor_asym,
			gyro_sigma, gyro_bias_walk, accel_sigma, seed,
		});
		self.gust = [0.0, 0.0, 0.0];
		self.gyro_bias = [0.0, 0.0, 0.0];
		self.step_idx = 0;
	}

	/// Disable disturbances — back to the bit-identical clean sim.
	pub fn clear_disturbance(&mut self) {
		self.dist = None;
		self.gust = [0.0, 0.0, 0.0];
		self.gyro_bias = [0.0, 0.0, 0.0];
		self.step_idx = 0;
	}

	/// Advance one timestep under the given 4-motor PWM (each clipped to [0, 1]).
	/// Uses RK4 integration of Euler's rotational equation + quaternion update.
	pub fn step(&mut self, motor_pwm: [f32; 4]) {
		let pwm = [
			motor_pwm[0].clamp(0.0, 1.0),
			motor_pwm[1].clamp(0.0, 1.0),
			motor_pwm[2].clamp(0.0, 1.0),
			motor_pwm[3].clamp(0.0, 1.0),
		];
		// W2 GUARD: None ⇒ the exact legacy torque path (no extra float ops).
		// Some ⇒ D3 per-motor thrust asymmetry + D1 constant bias + D2 OU gust,
		// all held constant over the RK4 step (same convention as motor torque).
		let torque = match self.dist {
			None => self.body_torque(pwm),
			Some(d) => {
				let base = self.body_torque_asym(pwm, d.motor_asym);
				[
					base[0] + d.tau_bias[0] + self.gust[0],
					base[1] + d.tau_bias[1] + self.gust[1],
					base[2] + d.tau_bias[2] + self.gust[2],
				]
			}
		};
		let dt = self.dt;

		// RK4 on state y = (omega: 3, q: 4). torque is held constant over the step.
		let (k1o, k1q) = self.derivatives(self.omega, self.q, torque);
		let omega2 = vec_add3(self.omega, vec_scale3(k1o, dt * 0.5));
		let q2 = q_add(self.q, q_scale(k1q, dt * 0.5));
		let (k2o, k2q) = self.derivatives(omega2, q2, torque);
		let omega3 = vec_add3(self.omega, vec_scale3(k2o, dt * 0.5));
		let q3 = q_add(self.q, q_scale(k2q, dt * 0.5));
		let (k3o, k3q) = self.derivatives(omega3, q3, torque);
		let omega4 = vec_add3(self.omega, vec_scale3(k3o, dt));
		let q4 = q_add(self.q, q_scale(k3q, dt));
		let (k4o, k4q) = self.derivatives(omega4, q4, torque);

		// y_{n+1} = y_n + (dt/6)(k1 + 2 k2 + 2 k3 + k4)
		let omega_delta = vec_scale3(
			vec_add3(
				vec_add3(k1o, vec_scale3(k2o, 2.0)),
				vec_add3(vec_scale3(k3o, 2.0), k4o),
			),
			dt / 6.0,
		);
		let q_delta = q_scale(
			q_add(
				q_add(k1q, q_scale(k2q, 2.0)),
				q_add(q_scale(k3q, 2.0), k4q),
			),
			dt / 6.0,
		);
		self.omega = vec_add3(self.omega, omega_delta);
		self.q = q_normalize(q_add(self.q, q_delta));
		self.t += dt;

		// W2 post-step state updates (once per physical step, AFTER the gust
		// was used in this step's torque): D2 OU gust innovation + D4 gyro
		// bias walk. Counter-based draws at step_idx = the step just taken.
		if let Some(d) = self.dist {
			let s32 = dist_seed32(d.seed);
			let t32 = self.step_idx as u32;
			let sqrt_dt = dt.sqrt();
			for a in 0..3 {
				if d.gust_sigma > 0.0 {
					let xi = dist_gauss(s32, t32, a as u32, DIST_CH_GUST);
					self.gust[a] += -self.gust[a] / d.gust_tau_c * dt
						+ d.gust_sigma * sqrt_dt * xi;
				}
				if d.gyro_bias_walk > 0.0 {
					let xi = dist_gauss(s32, t32, a as u32, DIST_CH_GYRO_BIAS);
					self.gyro_bias[a] += d.gyro_bias_walk * sqrt_dt * xi;
				}
			}
		}
		self.step_idx += 1;
	}

	/// Set an N-rotor geometry for step_n(). Rows are
	/// [px, py, pz, ax, ay, az, spin, k_thrust, k_drag] (body frame; axis
	/// need not be pre-normalized — it is normalized here). Clears rotor_asym
	/// if its length no longer matches. Persists across reset(), like `dist`.
	pub fn set_geometry(&mut self, rotors: Vec<[f32; 9]>) -> PyResult<()> {
		self.set_geometry_core(rotors).map_err(pyo3::exceptions::PyValueError::new_err)
	}

	/// Preset: flat octo-X (8 rotors, alternating spin).
	pub fn set_geometry_octo_x(&mut self, arm: f32, k_thrust: f32, k_drag: f32) {
		self.geometry = Some(crate::overactuated::RotorGeometry::octo_x(arm, k_thrust, k_drag));
		self.rotor_asym = None;
	}

	/// Preset: canted hex (Voliro-style fixed tilt, `cant_deg` about each arm).
	pub fn set_geometry_canted_hex(&mut self, arm: f32, k_thrust: f32, k_drag: f32, cant_deg: f32) {
		self.geometry = Some(crate::overactuated::RotorGeometry::canted_hex(arm, k_thrust, k_drag, cant_deg));
		self.rotor_asym = None;
	}

	/// Preset: the legacy '+' quad as a geometry (for parity tests; the
	/// production quad path stays step() with geometry=None).
	pub fn set_geometry_quad_plus(&mut self, arm: f32, k_thrust: f32, k_drag: f32) {
		self.geometry = Some(crate::overactuated::RotorGeometry::quad_plus(arm, k_thrust, k_drag));
		self.rotor_asym = None;
	}

	/// Perturb the CURRENT geometry in place: per-rotor tilt error (deg,
	/// about each arm direction) + position error (m). This is the
	/// true-vehicle-vs-nominal-allocator mismatch the residual must learn.
	#[pyo3(signature = (tilt_err_deg = vec![], pos_err = vec![]))]
	pub fn perturb_geometry(&mut self, tilt_err_deg: Vec<f32>, pos_err: Vec<[f32; 3]>) -> PyResult<()> {
		self.perturb_geometry_core(tilt_err_deg, pos_err).map_err(pyo3::exceptions::PyValueError::new_err)
	}

	/// Export the CURRENT geometry as 9-float rows
	/// [px,py,pz, ax,ay,az, spin, k_thrust, k_drag] — the set_geometry /
	/// score_controllers_* row contract. Lets Python build the presets +
	/// perturbations HERE (single implementation) and hand the resulting
	/// table to GeometryConfig / the scorers. None ⇒ no geometry set.
	pub fn geometry_rows(&self) -> Option<Vec<[f32; 9]>> {
		self.geometry.as_ref().map(|g| g.rotors.iter().map(|r| [
			r.position[0], r.position[1], r.position[2],
			r.axis[0], r.axis[1], r.axis[2],
			r.spin, r.k_thrust, r.k_drag,
		]).collect())
	}

	/// Back to the legacy quad-only sim (step_n then requires 4 PWMs).
	pub fn clear_geometry(&mut self) {
		self.geometry = None;
		self.rotor_asym = None;
	}

	/// Per-rotor thrust multipliers for the geometry path (N-rotor D3 twin).
	/// None resets to clean motors.
	#[pyo3(signature = (asym = None))]
	pub fn set_rotor_asym(&mut self, asym: Option<Vec<f32>>) -> PyResult<()> {
		self.set_rotor_asym_core(asym).map_err(pyo3::exceptions::PyValueError::new_err)
	}

	/// Rotor count of the active geometry (4 when running the legacy quad).
	pub fn num_rotors(&self) -> usize {
		self.geometry.as_ref().map_or(4, |g| g.num_rotors())
	}

	/// N-rotor twin of step(). With geometry=None it REQUIRES 4 PWMs and
	/// delegates to the legacy (bit-identical) quad path. With a geometry it
	/// computes torque via the generic r x F + spin-drag model (per-rotor
	/// asym + shared D1 bias/D2 gust), then runs the SAME RK4 + post-step
	/// noise updates as step() — kept in lockstep with step()'s body; any
	/// integrator change must be applied to both.
	pub fn step_n(&mut self, motor_pwm: Vec<f32>) -> PyResult<()> {
		self.step_n_core(&motor_pwm).map_err(pyo3::exceptions::PyValueError::new_err)
	}

	/// Read the simulated IMU: (gyro_xyz, accel_xyz) in body frame.
	///   gyro  = body-frame angular velocity (rad/s)
	///   accel = body-frame specific force (m/s²) — the negative of gravity
	///           rotated into body frame. At rest with q=identity, this reads
	///           (0, 0, +g) (the support force pushing UP through the IMU).
	pub fn read_imu(&self) -> ([f32; 3], [f32; 3]) {
		let gyro = self.omega;
		// gravity in WORLD frame points DOWN: (0, 0, -g)
		let gravity_world = [0.0, 0.0, -self.gravity];
		// rotate to body frame; specific force = -gravity_body (support force)
		let gravity_body = rotate_world_to_body(self.q, gravity_world);
		let accel = [-gravity_body[0], -gravity_body[1], -gravity_body[2]];
		// W2 D4: gyro bias + white noise; accel white noise. Pure function of
		// (seed, step_idx, axis) → idempotent (a second read_imu at the same
		// step returns the same values); the bias WALK advances in step().
		// None ⇒ the exact legacy return (no extra float ops).
		match self.dist {
			None => (gyro, accel),
			Some(d) => {
				let s32 = dist_seed32(d.seed);
				let t32 = self.step_idx as u32;
				let mut g = gyro;
				let mut a2 = accel;
				for a in 0..3 {
					g[a] += self.gyro_bias[a];
					if d.gyro_sigma > 0.0 {
						g[a] += d.gyro_sigma * dist_gauss(s32, t32, a as u32, DIST_CH_GYRO);
					}
					if d.accel_sigma > 0.0 {
						a2[a] += d.accel_sigma * dist_gauss(s32, t32, a as u32, DIST_CH_ACCEL);
					}
				}
				(g, a2)
			}
		}
	}

	/// Geodesic angle (rad) between current attitude and target attitude.
	/// Target defaults to identity (level). Uses 2·acos(|q·t|) on the
	/// quaternion dot product (the standard geodesic metric on SO(3)).
	#[pyo3(signature = (target = None))]
	pub fn attitude_error(&self, target: Option<[f32; 4]>) -> f32 {
		let t = q_normalize(target.unwrap_or([1.0, 0.0, 0.0, 0.0]));
		let dot = self.q[0] * t[0] + self.q[1] * t[1] + self.q[2] * t[2] + self.q[3] * t[3];
		// Clamp for numerical safety; acos domain is [-1, 1].
		let dot_abs = dot.abs().min(1.0);
		2.0 * dot_abs.acos()
	}

	/// True if the simulator state has diverged (omega above safety threshold
	/// or NaN in state).
	pub fn is_unstable(&self) -> bool {
		for v in self.omega.iter() {
			if !v.is_finite() || v.abs() > 50.0 {
				return true;
			}
		}
		for v in self.q.iter() {
			if !v.is_finite() {
				return true;
			}
		}
		false
	}

	#[getter]
	fn time(&self) -> f32 {
		self.t
	}

	#[getter]
	pub fn quaternion(&self) -> [f32; 4] {
		self.q
	}

	#[getter]
	fn angular_velocity(&self) -> [f32; 3] {
		self.omega
	}
}

// =============================================================================
// WnnController GPU export (crate-visible; consumed by metal_controller.rs).
// =============================================================================

impl WnnController {
	/// Export this controller's connectivity + trained memory for the GPU
	/// rollout kernel. Connections are flat i64; memories export to sorted
	/// (key,value) arrays for in-kernel binary search (untrained → EMPTY=2).
	pub(crate) fn gpu_export(&self) -> (
		&[i64], &[i64],
		ram_core::sparse_memory::SparseGpuExport,
		ram_core::sparse_memory::SparseGpuExport,
	) {
		(
			&self.state_connections,
			&self.output_connections,
			self.state_memory.export_for_gpu(),
			self.output_memory.export_for_gpu(),
		)
	}

	/// Shape dims the kernel needs (uniform across a population).
	pub(crate) fn gpu_dims(&self) -> (usize, usize, usize, usize, usize, usize, usize) {
		// (num_motors, levels, n_state, sbpn, obpn, bpf, window)
		(self.num_motors, self.levels_per_motor, self.state_neurons,
		 self.state_bits_per_neuron, self.output_bits_per_neuron,
		 self.bits_per_feature, self.input_window_k)
	}

	pub(crate) fn thresholds_ref(&self) -> &[f32] { &self.thresholds }

	/// Delta-control mode params the GPU scorer needs so it matches step()'s
	/// delta-mode decode (previously the kernel was absolute-only, scoring
	/// delta controllers in the WRONG mode). Uniform across a population.
	pub(crate) fn delta_params(&self) -> (bool, f32, f32) {
		(self.delta_control, self.delta_max, self.delta_leak)
	}

	/// num_features (9 + enabled H2 extras) + obs-feature config the GPU scorer
	/// needs to mirror compute_features. Uniform across a population.
	pub(crate) fn obs_params(&self) -> (usize, bool, bool, bool, bool, bool, f32, f32, bool, bool, bool, bool, f32) {
		(self.num_features, self.obs_tilt_p, self.obs_tilt_i,
		 self.obs_peraxis_p, self.obs_peraxis_i, self.obs_pwm,
		 self.integral_leak, self.integral_scale, self.decouple_outputs,
		 self.obs_peraxis_yaw,
		 self.obs_yaw_err, self.obs_yaw_err_i, self.dt)
	}

	/// H3: true when the 4 output banks are controls [T,τr,τp,τy]. Used by the
	/// DAGGER collector to un-mix teacher/student MOTOR targets into CONTROL targets.
	pub(crate) fn decouple_outputs_flag(&self) -> bool { self.decouple_outputs }

	/// Action-repeat N (arm R; uniform across a population). The GPU score /
	/// train / record hosts read it so the kernels mirror step()'s decision mask.
	pub(crate) fn action_repeat_n(&self) -> usize { self.action_repeat }

	/// Plain-Rust constructor twin of the pymethod `new` (house pattern: String
	/// errors keep cargo tests off the libpython link path — the GPU rollout
	/// parity suite builds controllers through THIS). The pymethod is a thin
	/// wrapper mapping to PyValueError.
	#[allow(clippy::too_many_arguments)]
	pub(crate) fn new_core(
		num_motors: usize,
		levels_per_motor: usize,
		bits_per_feature: usize,
		input_window_k: usize,
		state_neurons: usize,
		state_bits_per_neuron: usize,
		output_bits_per_neuron: usize,
		thresholds: Vec<f32>,
		state_connections: Vec<i64>,
		output_connections: Vec<i64>,
		delta_control: bool,
		delta_max: f32,
		delta_leak: f32,
		obs_tilt_p: bool,
		obs_tilt_i: bool,
		obs_peraxis_p: bool,
		obs_peraxis_i: bool,
		obs_peraxis_yaw: bool,
		obs_pwm: bool,
		obs_yaw_err: bool,
		obs_yaw_err_i: bool,
		integral_leak: f32,
		integral_scale: f32,
		dt: f32,
		decouple_outputs: bool,
		action_repeat: usize,
	) -> Result<Self, String> {
		// H3 needs exactly 4 control banks [T, τ_roll, τ_pitch, τ_yaw] → 4 motors.
		if decouple_outputs && num_motors != 4 {
			return Err("decouple_outputs requires num_motors == 4 (T + 3 torques → 4 motors)".to_string());
		}
		// num_features = base 9 + enabled extras (canonical order). All-off ⇒ 9.
		// Per-axis features carry 3 channels (roll/pitch/yaw) or 2 when yaw is dropped.
		let peraxis_n = if obs_peraxis_yaw { 3 } else { 2 };
		let num_extra = (obs_tilt_p as usize) + (obs_tilt_i as usize)
			+ (obs_peraxis_p as usize) * peraxis_n + (obs_peraxis_i as usize) * peraxis_n
			+ (obs_pwm as usize) * num_motors
			+ (obs_yaw_err as usize) + (obs_yaw_err_i as usize);  // clean scalar yaw channel
		let num_features = NUM_FEATURES + num_extra;
		// One integral accumulator per enabled "_i" feature (tilt_i + peraxis_n×peraxis_i + yaw_err_i).
		let num_integral = (obs_tilt_i as usize) + (obs_peraxis_i as usize) * peraxis_n
			+ (obs_yaw_err_i as usize);
		let expected_thresholds = num_features * bits_per_feature;
		if thresholds.len() != expected_thresholds {
			return Err(format!(
				"thresholds length {} != num_features * bits_per_feature = {} ({} base + {} extra features)",
				thresholds.len(), expected_thresholds, NUM_FEATURES, num_extra
			));
		}
		let expected_state_conn = state_neurons * state_bits_per_neuron;
		if state_connections.len() != expected_state_conn {
			return Err(format!(
				"state_connections length {} != state_neurons * state_bits_per_neuron = {}",
				state_connections.len(), expected_state_conn
			));
		}
		let num_output_neurons = num_motors * levels_per_motor;
		let expected_output_conn = num_output_neurons * output_bits_per_neuron;
		if output_connections.len() != expected_output_conn {
			return Err(format!(
				"output_connections length {} != num_motors * levels_per_motor * output_bits_per_neuron = {}",
				output_connections.len(), expected_output_conn
			));
		}

		Ok(Self {
			num_motors,
			levels_per_motor,
			bits_per_feature,
			input_window_k,
			state_neurons,
			state_bits_per_neuron,
			state_memory: SparseLayerMemory::new(state_neurons, state_bits_per_neuron),
			state_connections,
			output_bits_per_neuron,
			output_memory: SparseLayerMemory::new(num_output_neurons, output_bits_per_neuron),
			output_connections,
			thresholds,
			prev_state: vec![0u8; state_neurons],
			input_history: VecDeque::with_capacity(input_window_k),
			last_output_cells: vec![0u8; num_output_neurons],
			last_state_layer_input: Vec::new(),
			last_output_layer_input: Vec::new(),
			delta_control,
			delta_max,
			delta_leak,
			// Accumulator neutral: hover 0.5 per motor, OR (decouple) T→0.5, torques→0.
			pwm: (0..num_motors).map(|m| if decouple_outputs && m >= 1 { 0.0 } else { 0.5 }).collect(),
			pwm_prev: (0..num_motors).map(|m| if decouple_outputs && m >= 1 { 0.0 } else { 0.5 }).collect(),
			obs_tilt_p,
			obs_tilt_i,
			obs_peraxis_p,
			obs_peraxis_i,
			obs_peraxis_yaw,
			obs_pwm,
			obs_yaw_err,
			obs_yaw_err_i,
			integral_leak,
			integral_scale,
			dt,
			decouple_outputs,
			num_features,
			integral_acc: vec![0.0f32; num_integral],
			yaw_heading: 0.0,
			pending_init_yaws: Vec::new(),
			last_feature_vector: vec![0.0f32; num_features],
			// Action-repeat: N<1 makes no sense; normalize to 1 (= no repeat).
			action_repeat: action_repeat.max(1),
			step_counter: 0,
			last_pwm: (0..num_motors).map(|m| if decouple_outputs && m >= 1 { 0.0 } else { 0.5 }).collect(),
		})
	}

	// ---- GPU-train parity helpers (pub(crate); used by metal_controller's
	//      run_controller_train_parity_test to compare against the CPU reference) ----

	/// CPU reference: the LIVE production output trainer (split_retrain_output).
	pub(crate) fn split_retrain_output_pub(
		&mut self, gyros: &[Vec<[f32; 3]>], accels: &[Vec<[f32; 3]>],
		targets: &[Vec<[f32; 3]>], pid_pwms: &[Vec<[f32; 4]>], selective: bool,
	) -> usize {
		self.split_retrain_output(gyros, accels, targets, pid_pwms, selective)
	}

	/// Plant a STATE cell (so state_active varies → exercises the selective gate).
	pub(crate) fn plant_state_cell(&self, neuron: usize, addr: u64, v: u8) {
		self.state_memory.write_cell(neuron, addr, v, true);
	}

	/// Apply a trained OUTPUT cell (used by the GPU split_train_loop wrapper to
	/// write train_seeded's results back, so the next round seeds from them).
	pub(crate) fn set_output_cell(&self, neuron: usize, addr: u64, v: u8) {
		self.output_memory.write_cell(neuron, addr, v, true);
	}

	/// CPU reference for the GPU plant-latch parity (P4): runs split_plant_latch and
	/// returns the neuron it planted (or None). Mutates state_memory.
	pub(crate) fn split_plant_latch_pub(&self, bit: usize, high_on: bool, sif: &[bool], sil: usize) -> Option<usize> {
		self.split_plant_latch(bit, high_on, &vec![false; self.state_neurons], sif, sil)
	}

	/// CPU reference for the GPU plant-counter parity (P4): runs split_install_counter
	/// and returns the planted chain (or None). Mutates state_memory.
	pub(crate) fn split_install_counter_pub(&self, trigger: usize, max_levels: usize, sif: &[bool], sil: usize) -> Option<Vec<usize>> {
		self.split_install_counter(trigger, max_levels, &vec![false; self.state_neurons], sif, sil)
	}

	/// CPU reference for the GPU plant-counter-bidir parity (P4): runs
	/// split_install_counter_bidir and returns the planted levels (or None).
	pub(crate) fn split_install_counter_bidir_pub(&self, up: usize, dn: usize, n_levels: usize) -> Option<Vec<usize>> {
		self.split_install_counter_bidir(up, dn, n_levels, &vec![false; self.state_neurons])
	}

	/// CPU reference for the GPU resolve-conflict parity (P5-integrate): runs
	/// split_resolve_conflict and returns (mode, neurons). Mutates state_memory.
	#[allow(clippy::too_many_arguments)]
	pub(crate) fn split_resolve_conflict_pub(
		&self, instances: &[usize], pwms: &[[f32; 4]], ep_of: &[usize], step_of: &[usize],
		ep_start: &[usize], sif: &[bool], sil: usize, candidate_bits: &[usize],
		clean_gain: f32, accum_corr: f32, used: &[bool],
	) -> (i64, Vec<usize>) {
		self.split_resolve_conflict(instances, pwms, ep_of, step_of, ep_start, sif, sil, candidate_bits, clean_gain, accum_corr, used)
	}

	/// Effective state cell value (CPU read_cell, miss → EMPTY=2) — for the P4 parity
	/// comparison over the whole cell FUNCTION.
	pub(crate) fn state_cell(&self, neuron: usize, addr: u64) -> u8 {
		self.state_memory.read_cell(neuron, addr)
	}

	/// All planted state cells (neuron → (addr, value)).
	pub(crate) fn state_entries(&self, neuron: usize) -> Vec<(u64, u8)> {
		self.state_memory.neuron_entries(neuron)
	}

	/// Effective output cell value for (neuron, addr) — the CPU read_cell (miss →
	/// EMPTY=2), so the parity comparison is over the whole cell FUNCTION (a cell
	/// nudged back to 2 reads identically to an unvisited one).
	pub(crate) fn output_cell(&self, neuron: usize, addr: u64) -> u8 {
		self.output_memory.read_cell(neuron, addr)
	}

	/// All trained output cells (neuron → (addr, value)), for enumerating the
	/// addresses the CPU touched.
	pub(crate) fn output_entries(&self, neuron: usize) -> Vec<(u64, u8)> {
		self.output_memory.neuron_entries(neuron)
	}

	/// CPU reference for the GPU controller_record parity (P2): returns
	/// (out_ins per record, pid pwm per record, state_ins FLAT, state_input_len).
	pub(crate) fn split_record_pub(
		&mut self, gyros: Vec<Vec<[f32; 3]>>, accels: Vec<Vec<[f32; 3]>>,
		targets: Vec<Vec<[f32; 3]>>, pid_pwms: Vec<Vec<[f32; 4]>>,
	) -> (Vec<Vec<bool>>, Vec<[f32; 4]>, Vec<bool>, usize) {
		let (out_ins, pwms, _ep, _st, state_flat, state_len, _epl) =
			self.split_record(gyros, accels, targets, pid_pwms);
		(out_ins, pwms, state_flat, state_len)
	}
}

// =============================================================================
// AttitudeSim private helpers.
// =============================================================================

impl AttitudeSim {
	/// Body-frame torque vector from 4-motor PWM. See top-of-file convention.
	#[inline]
	fn body_torque(&self, pwm: [f32; 4]) -> [f32; 3] {
		// Per-motor thrust (N), quadratic in PWM.
		let t0 = self.k_thrust * pwm[0] * pwm[0];
		let t1 = self.k_thrust * pwm[1] * pwm[1];
		let t2 = self.k_thrust * pwm[2] * pwm[2];
		let t3 = self.k_thrust * pwm[3] * pwm[3];
		let l = self.arm_length;
		let k = self.k_drag;
		// roll  (about +x): -L*T1 + L*T3   (right motor → -x torque; left motor → +x torque)
		// pitch (about +y): -L*T0 + L*T2   (front motor → -y; rear motor → +y)
		// yaw   (about +z): +k(T0 - T1 + T2 - T3)
		[
			l * (-t1 + t3),
			l * (-t0 + t2),
			k * (t0 - t1 + t2 - t3),
		]
	}

	/// W2 D3 twin of body_torque: per-motor thrust = (k_thrust × asym_i) × pwm².
	/// Kept SEPARATE from body_torque so the clean path stays bit-identical
	/// (no ×1.0 passes through the hot loop). Same torque combination.
	#[inline]
	fn body_torque_asym(&self, pwm: [f32; 4], asym: [f32; 4]) -> [f32; 3] {
		let t0 = self.k_thrust * asym[0] * pwm[0] * pwm[0];
		let t1 = self.k_thrust * asym[1] * pwm[1] * pwm[1];
		let t2 = self.k_thrust * asym[2] * pwm[2] * pwm[2];
		let t3 = self.k_thrust * asym[3] * pwm[3] * pwm[3];
		let l = self.arm_length;
		let k = self.k_drag;
		[
			l * (-t1 + t3),
			l * (-t0 + t2),
			k * (t0 - t1 + t2 - t3),
		]
	}

	/// Compute (dω/dt, dq/dt) given (omega, q, body_torque).
	///   dω/dt = I⁻¹ (τ - ω × (I ω))         (Euler's equation, diag I)
	///   dq/dt = 0.5 q ⊗ [0, ω]
	#[inline]
	fn derivatives(&self, omega: [f32; 3], q: [f32; 4], torque: [f32; 3]) -> ([f32; 3], [f32; 4]) {
		let i = self.inertia;
		let i_omega = [omega[0] * i[0], omega[1] * i[1], omega[2] * i[2]];
		let coriolis = cross3(omega, i_omega);
		let net_torque = [torque[0] - coriolis[0], torque[1] - coriolis[1], torque[2] - coriolis[2]];
		let domega_dt = [net_torque[0] / i[0], net_torque[1] / i[1], net_torque[2] / i[2]];

		let omega_q = [0.0, omega[0], omega[1], omega[2]];
		let dq_dt_raw = q_multiply(q, omega_q);
		let dq_dt = q_scale(dq_dt_raw, 0.5);

		(domega_dt, dq_dt)
	}
}

// =============================================================================
// WnnController
// =============================================================================

/// Stateful WNN controller. One per drone / one per training episode.
///
/// Input pipeline per step:
///   sensors (9 floats) → thermometer-encode against per-feature thresholds
///   → push into K-step sliding window → concat with QSR-graded recurrent
///   state bits → state-layer forward → next state buffer → output-layer
///   forward → Strategy-5 decode → 4 motor PWMs in [0, 1].
///
/// Memory cells are stored in SparseLayerMemory (the same primitive used
/// by IDS training). A freshly-constructed controller has empty memory →
/// every neuron returns the EMPTY default (WEAK_FALSE = 1) → Strategy-5
/// produces a deterministic mid-throttle output. Training fills the
/// memories via write_state_cell / write_output_cell.
#[pyclass]
#[derive(Clone)]
pub struct WnnController {
	num_motors: usize,
	levels_per_motor: usize,
	bits_per_feature: usize,
	input_window_k: usize,

	state_neurons: usize,
	state_bits_per_neuron: usize,
	state_memory: SparseLayerMemory,
	state_connections: Vec<i64>,  // flat: state_neurons * state_bits_per_neuron

	// Output layer: one neuron per (motor, level). num_motors * levels_per_motor neurons total.
	output_bits_per_neuron: usize,
	output_memory: SparseLayerMemory,
	output_connections: Vec<i64>, // flat: (num_motors * levels_per_motor) * output_bits_per_neuron

	// Per-feature thermometer thresholds, flat: NUM_FEATURES * bits_per_feature.
	// thresholds[f*bits_per_feature + b] is the b-th threshold for feature f.
	thresholds: Vec<f32>,

	// Runtime state (mutable across step()).
	// One u8 QSR value per state neuron from the previous timestep.
	prev_state: Vec<u8>,
	// Last K thermometer-encoded sensor frames. Each frame is
	// NUM_FEATURES * bits_per_feature bools. Oldest is at .front(), newest at .back().
	input_history: VecDeque<Vec<bool>>,
	// Cached output cells from last step (for monotonicity-reward inspection).
	last_output_cells: Vec<u8>,
	// Cached state-layer input that step() used (windowed sensors + prev_state
	// at step-time). Needed for train_state_step to compute the SAME addresses
	// step() read from. Cleared on reset().
	last_state_layer_input: Vec<bool>,

	// Cached output-layer input step() used: [current sensor frame | new_state]
	// (Mealy: the output sees the input + the FULL state). edra_train_step
	// solves only the state bits (the frame is immutable). Cleared on reset().
	last_output_layer_input: Vec<bool>,

	// --- Delta-control mode ---
	// When true, the output decodes to a per-step PWM DELTA rather than an
	// absolute throttle: pwm[m] += delta, clamped to [0,1]. The accumulator IS
	// the integrator (it offloads PID's integral term out of the learned state).
	// The neutral decode point is the UNTRAINED value (QSR EMPTY -> 0.75), so an
	// untrained controller HOLDS throttle (delta 0) — a stable bootstrap that
	// doesn't tumble, which behavioral cloning of absolute PWM could not give.
	delta_control: bool,
	delta_max: f32,
	// Leaky integrator: each step the accumulator's deviation from hover (0.5)
	// decays by delta_leak before the delta is added (pwm = 0.5 + leak*(pwm-0.5)
	// + delta). leak=1.0 = pure integrator (bias can run away to saturation);
	// leak<1.0 bounds the steady-state offset to delta/(1-leak), preventing the
	// runaway drift that made raw delta-control tumble.
	delta_leak: f32,
	pwm: Vec<f32>,       // delta-accumulator: per-motor throttle, OR (decouple_outputs)
	                     // per-CONTROL [T, τ_roll, τ_pitch, τ_yaw] (Option A)
	pwm_prev: Vec<f32>,  // accumulator at the START of the current step (train baseline)

	// --- H3: decoupled outputs (18/06/2026) ---
	// When true, the 4 output banks are CONTROLS [common thrust T, τ_roll, τ_pitch,
	// τ_yaw] instead of 4 motor PWMs; a fixed mixing matrix converts controls→motors
	// AFTER the (per-control) delta accumulator (Option A). Gives the net an orthogonal
	// action space (one knob per physical axis) so each axis is a near-independent
	// SISO problem — and makes H4's per-axis curriculum warm-start cleanly. Requires
	// num_motors==4. delta accumulator neutral: T→0.5 (hover), torques→0 (no rotation).
	decouple_outputs: bool,

	// --- H2: configurable error/integral observation features (18/06/2026) ---
	// The base sensor frame is always the 9 raw values (NUM_FEATURES). These
	// toggles APPEND derived error/integral features in a FIXED canonical order
	// (tilt_p, tilt_i, roll_p, pitch_p, yaw_p, roll_i, pitch_i, yaw_i); only the
	// enabled ones are present, so `num_features` = 9 + enabled count (max 17).
	// All-off (the default) ⇒ num_features==9 ⇒ behaviour is bit-identical to the
	// pre-H2 controller (the parity anchor). The "_p" features are PROPORTIONAL
	// (the error itself); the "_i" features are LEAKY INTEGRALS of that error,
	// fed to the net as observations (the integral term as a SENSOR, not a control
	// law) — directly attacking the steady-state offset (H1 ruled out saturation).
	obs_tilt_p: bool,     // scalar tilt-to-vertical error (gravity ref, accel-only)
	obs_tilt_i: bool,     // leaky integral of the tilt error
	obs_peraxis_p: bool,  // per-axis roll/pitch/yaw error (3 features, or 2 if yaw off)
	obs_peraxis_i: bool,  // leaky integrals of the 3-axis error (3 features, or 2 if yaw off)
	// When false, the per-axis features push only roll+pitch (both gravity-observable
	// from accel) and DROP yaw — whose estimate is dead-reckoned (gyro-z integral, no
	// absolute reference → drifts) and was shown to poison the controller. Default true
	// preserves the original 3-axis behaviour (parity anchor).
	obs_peraxis_yaw: bool,
	// obs_pwm: expose the RAW throttle accumulator (current pwm, num_motors values)
	// as observations. This is the DIRECT fix for delta-mode's hidden state — the
	// optimal delta depends on the accumulator, which is otherwise unobservable
	// (∫error via obs_tilt_i is only a proxy and decorrelates from it in the
	// untrained regime — confirmed 18/06: delta+tilt_i pinned at 1% like bare delta).
	obs_pwm: bool,        // current throttle accumulator (num_motors features)
	integral_leak: f32,   // leaky-integral decay for the "_i" features (DISTINCT
	                      // from delta_leak, which is the OUTPUT accumulator's leak)
	integral_scale: f32,  // pre-threshold scale applied to integral features
	// --- Yaw-anchor (Phase A, 26/06/2026): a CLEAN scalar yaw-error channel ---
	// obs_yaw_err exposes (target_yaw − anchored yaw_heading) as ONE proportional
	// feature; obs_yaw_err_i its leaky integral. Distinct from obs_peraxis (which
	// degenerates on roll/pitch atan2 at large tilt) — yaw rides a dedicated scalar.
	// When EITHER is on the controller is "yaw-anchored": yaw_heading is SEEDED to
	// the episode's true initial yaw (init_yaw, from q0) at reset() and integrated
	// with the real dt (yaw_heading += gyro_z·dt), giving an absolute yaw reference.
	// Both off (default) ⇒ legacy behaviour (yaw_heading=0 seed, += gyro_z, no dt).
	obs_yaw_err: bool,    // scalar target_yaw − yaw_heading (1 feature)
	obs_yaw_err_i: bool,  // leaky integral of the yaw error (1 feature)
	dt: f32,              // physics timestep (s); used to scale gyro-z when anchored
	num_features: usize,  // 9 + enabled extra features (drives all frame sizing)
	// Per-step integral accumulators, one per enabled "_i" feature, in canonical
	// order (tilt_i first, then roll_i/pitch_i/yaw_i). Zeroed in reset().
	integral_acc: Vec<f32>,
	// Gyro-integrated yaw heading estimate (rad), for the yaw error feature. The
	// IMU gives no absolute yaw, so we integrate gyro-z. Zeroed in reset().
	yaw_heading: f32,
	// Yaw-anchor: per-episode initial yaw (rad) for the CURRENT training batch, set by
	// split_train_loop (and the dagger before bptt) so split_record/split_retrain_output
	// re-seed yaw_heading to match score-time when they replay recorded traces. Empty ⇒
	// legacy 0.0 seed (anchor-off). Transient batch state — NOT part of the genome.
	pending_init_yaws: Vec<f32>,
	// Raw (pre-threshold) feature values from the last compute_features() call,
	// length num_features. Exposed via get_last_feature_vector() so the Python
	// threshold-fitter calibrates the thermometer on the SAME values step() sees.
	last_feature_vector: Vec<f32>,

	// --- Action-repeat (arm R, 02/07/2026 — Sajus frame-skip adapted) ---
	// Decide every Nth physical step, HOLD the PWM in between. 1 = today's
	// behavior (bit-identical; the hold branch is guarded behind >1). Decision
	// steps are step_counter % N == 0, counter zeroed at reset() so episodes
	// align at t=0. Hold steps still tick compute_features (integ[]/yaw_heading
	// are physical-time) but do NO frame-encode/ring-push/forward/decode — the
	// K-window then holds the last K DECISION frames (spans K·N physical steps).
	// The SAME mask threads through every trainer re-forward (bptt_train_window,
	// split_record, split_retrain_output + the GPU twins) via step_counter, or
	// trainer addresses would diverge from deploy (the +55pp mode-mismatch class).
	action_repeat: usize,
	// Physical-step counter driving the decision mask. Shared by step() AND the
	// bptt forward roll (reset_state=false chunks carry it, keeping W-chunked
	// replays episode-aligned even when W % action_repeat != 0).
	step_counter: usize,
	// The motor PWM emitted at the last DECISION step, returned verbatim on hold
	// steps. Init/reset to the accumulator-neutral hover expression (never read
	// before the first decision — t=0 is always a decision step).
	last_pwm: Vec<f32>,
}

#[pymethods]
impl WnnController {
	/// Construct a controller. All connectivity and thresholds must be supplied
	/// up front. Memory cells start empty (EMPTY=WEAK_FALSE) — call
	/// write_state_cell / write_output_cell to populate during training.
	#[new]
	#[pyo3(signature = (
		num_motors,
		levels_per_motor,
		bits_per_feature,
		input_window_k,
		state_neurons,
		state_bits_per_neuron,
		output_bits_per_neuron,
		thresholds,
		state_connections,
		output_connections,
		delta_control = false,
		delta_max = 0.1,
		delta_leak = 1.0,
		obs_tilt_p = false,
		obs_tilt_i = false,
		obs_peraxis_p = false,
		obs_peraxis_i = false,
		obs_peraxis_yaw = true,
		obs_pwm = false,
		obs_yaw_err = false,
		obs_yaw_err_i = false,
		integral_leak = 0.99,
		integral_scale = 1.0,
		dt = 0.001,
		decouple_outputs = false,
		action_repeat = 1,
	))]
	#[allow(clippy::too_many_arguments)]
	pub fn new(
		num_motors: usize,
		levels_per_motor: usize,
		bits_per_feature: usize,
		input_window_k: usize,
		state_neurons: usize,
		state_bits_per_neuron: usize,
		output_bits_per_neuron: usize,
		thresholds: Vec<f32>,
		state_connections: Vec<i64>,
		output_connections: Vec<i64>,
		delta_control: bool,
		delta_max: f32,
		delta_leak: f32,
		obs_tilt_p: bool,
		obs_tilt_i: bool,
		obs_peraxis_p: bool,
		obs_peraxis_i: bool,
		obs_peraxis_yaw: bool,
		obs_pwm: bool,
		obs_yaw_err: bool,
		obs_yaw_err_i: bool,
		integral_leak: f32,
		integral_scale: f32,
		dt: f32,
		decouple_outputs: bool,
		action_repeat: usize,
	) -> PyResult<Self> {
		Self::new_core(
			num_motors, levels_per_motor, bits_per_feature, input_window_k,
			state_neurons, state_bits_per_neuron, output_bits_per_neuron,
			thresholds, state_connections, output_connections,
			delta_control, delta_max, delta_leak,
			obs_tilt_p, obs_tilt_i, obs_peraxis_p, obs_peraxis_i, obs_peraxis_yaw,
			obs_pwm, obs_yaw_err, obs_yaw_err_i,
			integral_leak, integral_scale, dt, decouple_outputs, action_repeat,
		).map_err(pyo3::exceptions::PyValueError::new_err)
	}

	/// Zero the recurrent state buffer and clear the input history. In
	/// delta-control mode also reset the throttle accumulator to hover (0.5).
	/// `init_yaw` seeds the yaw-heading estimate when yaw-anchored (obs_yaw_err[_i]) —
	/// the episode's true initial yaw, giving an absolute reference. Un-anchored ⇒
	/// the legacy 0.0 seed (bit-identical to the pre-anchor controller).
	#[pyo3(signature = (init_yaw = 0.0))]
	pub fn reset(&mut self, init_yaw: f32) {
		for v in self.prev_state.iter_mut() { *v = 0; }
		self.input_history.clear();
		for v in self.last_output_cells.iter_mut() { *v = 0; }
		self.last_state_layer_input.clear();
		self.last_output_layer_input.clear();
		// Reset to accumulator neutral (decouple: T→0.5, torques→0; else all hover 0.5).
		for (m, v) in self.pwm.iter_mut().enumerate() { *v = if self.decouple_outputs && m >= 1 { 0.0 } else { 0.5 }; }
		for (m, v) in self.pwm_prev.iter_mut().enumerate() { *v = if self.decouple_outputs && m >= 1 { 0.0 } else { 0.5 }; }
		// H2: zero the error-integral accumulators so each episode starts with no
		// accumulated error (matches per-episode reset).
		for v in self.integral_acc.iter_mut() { *v = 0.0; }
		// Yaw-anchored ⇒ seed heading to the episode's true initial yaw (absolute
		// reference for the obs_yaw_err channel). Un-anchored ⇒ legacy 0.0 seed.
		self.yaw_heading = if self.obs_yaw_err || self.obs_yaw_err_i { init_yaw } else { 0.0 };
		// Action-repeat: episodes align decisions at t=0; held PWM back to hover.
		self.step_counter = 0;
		for (m, v) in self.last_pwm.iter_mut().enumerate() { *v = if self.decouple_outputs && m >= 1 { 0.0 } else { 0.5 }; }
	}

	/// Snapshot all trained cells (state + output) — for best-checkpoint
	/// selection: the reward-gated trainer snapshots the controller at its best
	/// inner round and restores it at the end (the chaotic final round is often
	/// worse than the best). Returns (state_cells, output_cells) as
	/// (neuron_idx, address, value) triples.
	pub fn export_cells(&self) -> (Vec<(usize, u64, u8)>, Vec<(usize, u64, u8)>) {
		(self.state_memory.export(), self.output_memory.export())
	}

	/// (neuron_idx, address) the STATE layer read on the last step() call.
	/// Computed from the cached step-input — used by GA-Memory (paradigm B) to
	/// record the visited-address universe along reference rollouts (the cells
	/// whose QSR values the GA will evolve; unvisited addresses stay EMPTY).
	fn last_state_addresses(&self) -> Vec<(usize, u64)> {
		let mut v = Vec::with_capacity(self.state_neurons);
		if self.last_state_layer_input.is_empty() { return v; }
		for n in 0..self.state_neurons {
			let cs = n * self.state_bits_per_neuron;
			let ce = cs + self.state_bits_per_neuron;
			let addr = compute_address_sparse(
				&self.last_state_layer_input, &self.state_connections[cs..ce], self.state_bits_per_neuron);
			v.push((n, addr));
		}
		v
	}

	/// (neuron_idx, address) the OUTPUT layer read on the last step() call.
	fn last_output_addresses(&self) -> Vec<(usize, u64)> {
		let num_out = self.num_motors * self.levels_per_motor;
		let mut v = Vec::with_capacity(num_out);
		if self.last_output_layer_input.is_empty() { return v; }
		for n in 0..num_out {
			let cs = n * self.output_bits_per_neuron;
			let ce = cs + self.output_bits_per_neuron;
			let addr = compute_address_sparse(
				&self.last_output_layer_input, &self.output_connections[cs..ce], self.output_bits_per_neuron);
			v.push((n, addr));
		}
		v
	}

	/// The cached STATE-layer input bit-vector from the last step() —
	/// [K sensor frames | prev_state bits]. Exposed so Python can profile
	/// per-input-bit activation ENTROPY over a reference rollout, which drives
	/// stats-guided axonogenesis (rewire connections off near-constant input bits
	/// toward high-entropy ones). Entropy is a property of the input distribution,
	/// so one reference rollout serves all genomes.
	fn last_state_layer_input(&self) -> Vec<bool> {
		self.last_state_layer_input.clone()
	}

	/// The cached OUTPUT-layer input bit-vector from the last step() —
	/// [current sensor frame | new_state bits].
	fn last_output_layer_input(&self) -> Vec<bool> {
		self.last_output_layer_input.clone()
	}

	/// Restore a snapshot from export_cells: clear both memories and re-import.
	pub fn restore_cells(&mut self, state_cells: Vec<(usize, u64, u8)>, output_cells: Vec<(usize, u64, u8)>) {
		self.state_memory.reset();
		self.output_memory.reset();
		self.state_memory.import(&state_cells);
		self.output_memory.import(&output_cells);
	}

	/// Overwrite the throttle accumulator (delta-control integrator). DAGGER
	/// calls this when the EXPERT drives the sim, so the controller's
	/// integrator stays synced to the actually-applied PWM and the next step's
	/// delta is computed from the correct baseline.
	fn set_pwm(&mut self, pwm: Vec<f32>) {
		let n = self.num_motors.min(pwm.len());
		self.pwm[..n].copy_from_slice(&pwm[..n]);
	}

	/// Write a single cell into the state layer memory.
	fn write_state_cell(&self, neuron_idx: usize, address: u64, value: u8) -> PyResult<bool> {
		if neuron_idx >= self.state_neurons {
			return Err(pyo3::exceptions::PyValueError::new_err(format!(
				"state neuron_idx {} >= state_neurons {}", neuron_idx, self.state_neurons
			)));
		}
		Ok(self.state_memory.write_cell(neuron_idx, address, value & 0x3, true))
	}

	/// Write a single cell into the output layer memory.
	fn write_output_cell(&self, neuron_idx: usize, address: u64, value: u8) -> PyResult<bool> {
		let n_out = self.num_motors * self.levels_per_motor;
		if neuron_idx >= n_out {
			return Err(pyo3::exceptions::PyValueError::new_err(format!(
				"output neuron_idx {} >= num_motors * levels_per_motor = {}", neuron_idx, n_out
			)));
		}
		Ok(self.output_memory.write_cell(neuron_idx, address, value & 0x3, true))
	}

	/// Non-PyResult versions for crate-internal use (e.g. dagger_train batch
	/// path). Silently drop out-of-range writes — bad init-cell triples would
	/// already have been filtered upstream by `_filter_inherited_cells`.
	pub fn write_state_cell_internal(&self, neuron_idx: usize, address: u64, value: u8) -> bool {
		if neuron_idx >= self.state_neurons { return false; }
		self.state_memory.write_cell(neuron_idx, address, value & 0x3, true)
	}
	pub fn write_output_cell_internal(&self, neuron_idx: usize, address: u64, value: u8) -> bool {
		let n_out = self.num_motors * self.levels_per_motor;
		if neuron_idx >= n_out { return false; }
		self.output_memory.write_cell(neuron_idx, address, value & 0x3, true)
	}

	/// Seed the STATE layer as a fixed random reservoir: fill every cell of
	/// every state neuron with a deterministic pseudo-random QSR value (0..3)
	/// derived from (seed, neuron, address). This turns the state layer into a
	/// diverse fixed nonlinear projection of the input (echo-state / reservoir
	/// computing) so that DIFFERENT sensor inputs map to DIFFERENT states —
	/// which is the precondition for the output layer to learn an
	/// input-dependent 4-motor mapping. The output layer is then the only
	/// trained part (via train_output_step). This sidesteps the
	/// state-target problem that sank both the per-motor and identity EDRA
	/// approaches (state never became discriminative).
	fn seed_state_reservoir(&mut self, seed: u64) {
		let memory_size = 1usize << self.state_bits_per_neuron;
		for n in 0..self.state_neurons {
			for addr in 0..memory_size {
				// SplitMix64-style hash of (seed, n, addr) → 0..3.
				let mut z = seed
					.wrapping_add(0x9E3779B97F4A7C15u64.wrapping_mul(n as u64 + 1))
					.wrapping_add(0x6C8E9CF570932BD5u64.wrapping_mul(addr as u64 + 1));
				z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
				z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
				z ^= z >> 31;
				let val = (z & 0x3) as u8;
				self.state_memory.write_cell(n, addr as u64, val, true);
			}
		}
	}

	/// Read the raw output cells from the last step() call (or zeros if step
	/// has not yet been called this episode). Length = num_motors * levels_per_motor.
	/// Each entry is a QSR value in [0, 3]. Pass to monotonicity_violations()
	/// or strategy_5_qsr_weighted() to derive auxiliary signals.
	pub fn get_last_output_cells(&self) -> Vec<u8> {
		self.last_output_cells.clone()
	}

	/// Raw (pre-threshold) observation feature values from the last compute_features
	/// call — length num_features (9 base + enabled H2 extras). The Python threshold-
	/// fitter drives an untrained controller and collects these so the thermometer is
	/// calibrated on the SAME values step() encodes (single source of truth, no Python
	/// re-derivation of the stateful integral). Zeros before the first step().
	pub fn get_last_feature_vector(&self) -> Vec<f32> {
		self.last_feature_vector.clone()
	}

	/// Number of observation features (9 base + enabled H2 error/integral extras).
	pub fn num_features(&self) -> usize {
		self.num_features
	}

	/// QSR-aware single-step training of the OUTPUT layer.
	///
	/// Call AFTER step() — this uses the state buffer that step() just
	/// produced as the input to the output layer, and nudges the output
	/// cells at the addresses that step() just read toward the supplied
	/// target PWM (thermometer-encoded: bit i of motor m is "TRUE" iff
	/// target_pwm[m] * levels_per_motor >= i + 1).
	///
	/// This is the simplified one-step "EDRA on output only" — the state
	/// layer is NOT trained here. It's the minimum viable QSR-aware
	/// training: enough to validate that PID-supervised cells actually
	/// produce PID-like PWM at inference time.
	///
	/// SINGLE SOURCE OF TRUTH for encoding a per-motor ABSOLUTE commit target
	/// (motor PWM, or under decouple a control: throttle∈[0,1] / torque∈[-1,1])
	/// into the [0,1] raw-decode target that `decode_outputs` inverts. Under
	/// absolute+decouple, torque banks (m>=1) decode as (raw-0.5)*2 ∈ [-1,1], so
	/// the inverse is raw = τ/2 + 0.5 — without it, neutral torque (0) would train
	/// toward raw 0 = MAX NEGATIVE torque, making hover unlearnable. Throttle,
	/// non-decouple, and the delta path (which encodes via delta_to_decoded at its
	/// own call sites) all map directly with a clamp. EVERY training-commit site
	/// MUST call this so the encode can never drift from decode_outputs again
	/// (the absolute+decouple torque bug, 20/06/2026, was this drift across 6
	/// duplicated sites).
	#[inline]
	fn output_decode_target(&self, motor: usize, target: f32) -> f32 {
		if !self.delta_control && self.decouple_outputs && motor >= 1 {
			(target * 0.5 + 0.5).clamp(0.0, 1.0)
		} else {
			target.clamp(0.0, 1.0)
		}
	}

	/// Returns the number of cells modified.
	fn train_output_step(&mut self, target_pwm: [f32; 4]) -> usize {
		// Use the EXACT output-layer input step() built (Mealy: [frame | state]),
		// so we nudge the cells at the addresses step() actually read.
		let output_input = self.last_output_layer_input.clone();
		if output_input.is_empty() {
			return 0; // step() not called yet
		}

		let num_out = self.num_motors * self.levels_per_motor;
		let levels = self.levels_per_motor;
		let mut writes = 0usize;
		for n in 0..num_out {
			let motor = n / levels;
			let level_idx = n % levels;
			// Delta-control: target the DELTA decode level (from pwm_prev), not
			// the absolute PWM — matches what step() decodes.
			let d_target = if self.delta_control {
				delta_to_decoded(target_pwm[motor] - self.pwm_prev[motor], self.delta_max)
			} else {
				self.output_decode_target(motor, target_pwm[motor])
			};
			let target_true = (d_target * levels as f32) as usize > level_idx;

			let conn_start = n * self.output_bits_per_neuron;
			let conn_end = conn_start + self.output_bits_per_neuron;
			let address = ram_core::neuron_memory::compute_address_sparse(
				&output_input,
				&self.output_connections[conn_start..conn_end],
				self.output_bits_per_neuron,
			);
			let current = self.output_memory.read_cell(n, address);
			let new_value = crate::controller_training::nudge_toward_pub(current, target_true);
			if new_value != current {
				self.output_memory.write_cell(n, address, new_value, true);
				writes += 1;
			}
		}
		writes
	}

	/// QSR-aware training of the STATE layer. Requires
	/// `state_neurons == num_motors * levels_per_motor` (matches the
	/// identity-mapping assumption from Python's _solve_output).
	///
	/// Uses the LAST sensor frame + recurrent state that step() built
	/// (cached internally in `input_history` and `prev_state`). For each
	/// state neuron, nudges its cell at the recently-read address toward
	/// the target PWM bit corresponding to (this neuron's index).
	///
	/// Returns the number of cells modified (or 0 if the architecture
	/// constraint is violated).
	fn train_state_step(&mut self, target_pwm: [f32; 4]) -> usize {
		let total_target_bits = self.num_motors * self.levels_per_motor;
		if self.state_neurons != total_target_bits {
			return 0;
		}
		if self.last_state_layer_input.is_empty() {
			// step() must run first to populate last_state_layer_input.
			return 0;
		}
		let input_bits = &self.last_state_layer_input;

		let levels = self.levels_per_motor;
		let mut writes = 0usize;
		for n in 0..self.state_neurons {
			let motor = n / levels;
			let level_idx = n % levels;
			let p = self.output_decode_target(motor, target_pwm[motor]);
			let target_true = (p * levels as f32) as usize > level_idx;

			let conn_start = n * self.state_bits_per_neuron;
			let conn_end = conn_start + self.state_bits_per_neuron;
			let address = ram_core::neuron_memory::compute_address_sparse(
				input_bits,
				&self.state_connections[conn_start..conn_end],
				self.state_bits_per_neuron,
			);
			let current = self.state_memory.read_cell(n, address);
			let new_value = crate::controller_training::nudge_toward_pub(current, target_true);
			if new_value != current {
				self.state_memory.write_cell(n, address, new_value, true);
				writes += 1;
			}
		}
		writes
	}

	/// Combined: do one step() + one train_state_step() + one
	/// train_output_step() in a single Python call. This is the
	/// per-timestep hot path for training. The `target_pwm` is the PID
	/// teacher's action at this step.
	///
	/// When `state_neurons == num_motors * levels_per_motor`, both
	/// layers get trained (the "identity assumption" path). Otherwise
	/// only the output layer is trained (state-layer training is a
	/// no-op pending the full constraint-solver port).
	///
	/// Returns (pwm_from_controller, total_cells_written).
	fn step_and_train(
		&mut self,
		gyro: [f32; 3],
		accel: [f32; 3],
		target_attitude: [f32; 3],
		target_pwm: [f32; 4],
	) -> (Vec<f32>, usize) {
		let pwm = self.step(gyro, accel, target_attitude);
		let s_writes = self.train_state_step(target_pwm);
		let o_writes = self.train_output_step(target_pwm);
		(pwm, s_writes + o_writes)
	}

	/// Real per-motor EDRA-BPTT training step (lifts the n_state==n_output
	/// identity-assumption restriction of step_and_train).
	///
	/// Call AFTER step(). For each motor independently (256 neurons over the
	/// 2·state_neurons state-bit input space — tractable, unlike the joint
	/// 1024-neuron solve), the QSR constraint solver finds the desired
	/// state-layer output that would make that motor's thermometer PWM
	/// correct. The 4 per-motor desired states are vote-aggregated into a
	/// single state target; then:
	///   - the output layer is committed toward the target PWM (direct nudge), and
	///   - the state layer is committed toward the aggregated desired state.
	///
	/// Returns (state_cells_written, output_cells_written).
	fn edra_train_step(&mut self, target_pwm: [f32; 4], topk_per_neuron: usize) -> (usize, usize) {
		let levels = self.levels_per_motor;
		let obpn = self.output_bits_per_neuron;
		let state_bits_in = self.state_neurons; // 1 bit (QSR MSB) per state neuron (was 2·)

		// Output-layer input step() used (Mealy): [current frame | new_state].
		// The frame bits are immutable inputs; the solve adjusts only the state
		// bits (the tail), so n_immutable = frame_bits.
		let output_input = self.last_output_layer_input.clone();
		if output_input.is_empty() {
			return (0, 0); // step() not called yet
		}
		let frame_bits = self.num_features * self.bits_per_feature;
		let out_input_len = output_input.len(); // frame_bits + state_bits_in

		// Per-motor output solve -> vote per STATE bit (the solvable tail).
		// vote[i] > 0 -> motors want state bit i TRUE; < 0 -> FALSE; 0 -> keep.
		let mut vote = vec![0i32; state_bits_in];
		for m in 0..self.num_motors {
			// Output target: in delta-control mode the teacher's absolute
			// target_pwm becomes the DELTA needed from the pre-step throttle
			// (pwm_prev), encoded back to a decode level; otherwise it is the
			// absolute PWM. Then thermometer-encode it.
			let d_target = if self.delta_control {
				delta_to_decoded(target_pwm[m] - self.pwm_prev[m], self.delta_max)
			} else {
				self.output_decode_target(m, target_pwm[m])
			};
			let n_true = (d_target * levels as f32) as usize;
			let motor_target: Vec<bool> = (0..levels).map(|i| i < n_true).collect();

			let conn_start = m * levels * obpn;
			let conn_end = (m + 1) * levels * obpn;
			let motor_conns = &self.output_connections[conn_start..conn_end];

			let base = m * levels;
			// Reachable-address solver: enumerate trained cells from the sparse
			// memory + low-Hamming untrained, instead of scanning 2^obpn. Exact
			// (verified vs exhaustive) and ~1000x fewer address evaluations.
			let entries_fn = |nn: usize| self.output_memory.neuron_entries(base + nn);
			let solved = solve_partial_connectivity_qsr_reachable(
				entries_fn, ram_core::neuron_memory::EMPTY_U8,
				motor_conns, levels, obpn, out_input_len,
				&output_input, &motor_target, frame_bits, topk_per_neuron,
			);
			if let Some(sol) = solved {
				for i in 0..state_bits_in {
					vote[i] += if sol[frame_bits + i] { 1 } else { -1 };
				}
			}
		}

		let desired_state_bits: Vec<bool> = (0..state_bits_in)
			.map(|i| if vote[i] > 0 { true } else if vote[i] < 0 { false } else { output_input[frame_bits + i] })
			.collect();

		// Commit OUTPUT layer toward target PWM at the current state address.
		let o_writes = self.train_output_step(target_pwm);

		// Commit STATE layer toward the aggregated desired state, at the
		// state-layer input step() cached. desired QSR value per neuron =
		// 2·bit_msb + bit_lsb (matches step()'s QSR encode/decode).
		let input = self.last_state_layer_input.clone();
		let mut s_writes = 0usize;
		if !input.is_empty() {
			for n in 0..self.state_neurons {
				// desired_state_bits is now sn bits (MSB/side per neuron) → drive the
				// cell fully to that side (TRUE=3 / FALSE=0).
				let target_val: u8 = if desired_state_bits[n] { 3 } else { 0 };
				let conn_start = n * self.state_bits_per_neuron;
				let conn_end = conn_start + self.state_bits_per_neuron;
				let address = compute_address_sparse(
					&input,
					&self.state_connections[conn_start..conn_end],
					self.state_bits_per_neuron,
				);
				let current = self.state_memory.read_cell(n, address);
				let new_value = nudge_toward_value(current, target_val);
				if new_value != current {
					self.state_memory.write_cell(n, address, new_value, true);
					s_writes += 1;
				}
			}
		}
		(s_writes, o_writes)
	}

	/// Multi-step EDRA-BPTT over a teacher-forced window (absolute-PWM mode).
	///
	/// Given W steps of (gyro, accel, target, pid_pwm) -- typically a PID-driven
	/// rollout segment -- this:
	///   1. Forward-rolls the controller W steps, RECORDING each step's
	///      state-layer input, output-layer input, and recurrent state.
	///   2. Propagates desired states BACKWARD through the recurrence. Each
	///      step's desired state combines (a) the state that makes its output
	///      match PID (per-motor output solve + vote) and (b) the state that
	///      TRANSITIONS into the next step's desired state (solve the state layer
	///      for the prev-state that yields d_s[t+1], sensor bits immutable).
	///   3. Commits both layers per step toward those targets.
	///
	/// This trains the recurrence at the TRAJECTORY level -- the thing per-step
	/// EDRA cannot -- so the recurrent state can carry a stable integral instead
	/// of accumulating per-step-imitation noise. Resets the recurrent buffers at
	/// window start. Returns (state_writes, output_writes).
	#[pyo3(signature = (gyros, accels, targets, pid_pwms, topk_per_neuron = 4, reset_state = true, protect_learned = false, state_integral_targets = None, init_yaw = 0.0))]
	#[allow(clippy::too_many_arguments)]
	pub fn bptt_train_window(
		&mut self,
		gyros: Vec<[f32; 3]>,
		accels: Vec<[f32; 3]>,
		targets: Vec<[f32; 3]>,
		pid_pwms: Vec<[f32; 4]>,
		topk_per_neuron: usize,
		reset_state: bool,
		protect_learned: bool,
		// Option A: per-step NORMALIZED PID integral (roll,pitch,yaw)∈[-1,1]. When
		// provided AND WNN_STATE_INTEGRAL_TARGET=1, the state layer is committed
		// toward a thermometer encoding of this integral (a DIRECT, dense target)
		// instead of the fragile indirect output∧transition solve — so the
		// recurrent state actually learns to be the integrator. Default None →
		// unchanged behaviour.
		state_integral_targets: Option<Vec<[f32; 3]>>,
		// Yaw-anchor: this window's trajectory initial yaw (rad); seeds yaw_heading on
		// the reset_state window. 0.0 ⇒ legacy. Stashed so the shared reader picks it up.
		init_yaw: f32,
	) -> (usize, usize) {
		// Yaw-anchor: single-window bptt ⇒ pending_init_yaws holds just this traj's yaw.
		self.pending_init_yaws = vec![init_yaw];
		let w = gyros.len();
		if w == 0 {
			return (0, 0);
		}
		// Option A gate: env flag + targets present + length matches the window.
		let use_integral_target = state_integral_targets
			.as_ref()
			.map(|v| v.len() == w)
			.unwrap_or(false)
			&& std::env::var("WNN_STATE_INTEGRAL_TARGET").map(|s| s == "1").unwrap_or(false);
		let bpf = self.bits_per_feature;
		let frame_bits = self.num_features * bpf;
		let state_bits_in = self.state_neurons; // 1 bit (QSR MSB) per state neuron (was 2·)
		let sensor_window = self.input_window_k * frame_bits;
		let state_input_len = sensor_window + state_bits_in;
		let out_input_len = frame_bits + state_bits_in;
		let levels = self.levels_per_motor;
		let obpn = self.output_bits_per_neuron;
		let num_out = self.num_motors * levels;

		// ---- Forward roll, recording per-step inputs ----
		// reset_state=true: independent window from hover (deployment-consistent
		// for episode-start windows). false: carry recurrent state across windows
		// (truncated BPTT within an episode).
		if reset_state {
			// Yaw-anchor: bptt trains ONE trajectory → its init yaw is pending_init_yaws[0].
			let iy = self.pending_init_yaws.first().copied().unwrap_or(0.0);
			self.reset(iy);
		}
		let mut rec_state_input: Vec<Vec<bool>> = Vec::with_capacity(w);
		let mut rec_out_input: Vec<Vec<bool>> = Vec::with_capacity(w);
		// Action-repeat: physical step index of each record (records exist only
		// at decision steps), so the backward commit reads pid_pwms / integral
		// targets at the right PHYSICAL step. N=1 ⇒ rec_step[d] == d.
		let mut rec_step: Vec<usize> = Vec::with_capacity(w);
		// 31/05/2026: cooperative SIGTERM cancellation. Poll at the top of
		// every per-step iteration in the forward roll. ~1 ns/step Relaxed
		// atomic load, negligible vs the per-step Rust work (~100 µs-1 ms).
		// Returns the already-recorded prefix so the caller's bookkeeping
		// sees consistent state.
		for t in 0..w {
			if ram_core::cancel::check_cancel() {
				return (0, 0);
			}
			let feats = self.compute_features(gyros[t], accels[t], targets[t]);
			// Action-repeat hold: tick the accumulators only — no ring push, no
			// forward, no record (deploy visits NO addresses on hold steps, so
			// training one would write cells deploy never reads). The persistent
			// step_counter keeps W-chunked windows episode-aligned (reset_state
			// zeroes it via reset(); carry-chunks continue it), mirroring step().
			if self.action_repeat > 1 {
				let hold = self.step_counter % self.action_repeat != 0;
				self.step_counter += 1;
				if hold {
					continue;
				}
			}
			let mut frame = vec![false; frame_bits];
			for f in 0..self.num_features {
				let v = feats[f];
				let row = f * bpf;
				for b in 0..bpf {
					frame[row + b] = v >= self.thresholds[row + b];
				}
			}
			if self.input_history.len() == self.input_window_k {
				self.input_history.pop_front();
			}
			self.input_history.push_back(frame.clone());

			let mut in_state = vec![false; state_input_len];
			let pad = self.input_window_k - self.input_history.len();
			for (i, fr) in self.input_history.iter().enumerate() {
				let slot = (pad + i) * frame_bits;
				in_state[slot..slot + frame_bits].copy_from_slice(fr);
			}
			for (n, &v) in self.prev_state.iter().enumerate() {
				in_state[sensor_window + n] = (v >> 1) & 1 != 0; // MSB only
			}

			let mut new_state = vec![0u8; self.state_neurons];
			for n in 0..self.state_neurons {
				let cs = n * self.state_bits_per_neuron;
				let ce = cs + self.state_bits_per_neuron;
				let addr = compute_address_sparse(&in_state, &self.state_connections[cs..ce], self.state_bits_per_neuron);
				new_state[n] = self.state_memory.read_cell(n, addr);
			}

			let mut in_out = vec![false; out_input_len];
			in_out[0..frame_bits].copy_from_slice(&frame);
			for (n, &v) in new_state.iter().enumerate() {
				in_out[frame_bits + n] = (v >> 1) & 1 != 0; // MSB only
			}

			rec_state_input.push(in_state);
			rec_out_input.push(in_out);
			rec_step.push(t);
			self.prev_state = new_state;
		}

		// ---- Backward pass + commit ----
		// Action-repeat: the walk is over RECORDS (= decision steps); d indexes
		// records, rec_step[d] the matching physical step for pid/integral
		// targets. N=1 ⇒ d == t, bit-identical to the pre-repeat walk.
		let n_rec = rec_out_input.len();
		let mut s_writes = 0usize;
		let mut o_writes = 0usize;
		let mut d_next: Option<Vec<bool>> = None; // desired state bits for record d+1
		// 31/05/2026: cancel poll at the head of each backward BPTT step.
		// The backward step does the per-(t, neuron) QSR solving — by far the
		// most expensive part of bptt_train_window — so this is the polling
		// site that actually shortens SIGTERM response for long windows.
		for d in (0..n_rec).rev() {
			if ram_core::cancel::check_cancel() {
				return (s_writes, o_writes);
			}
			let t = rec_step[d];
			// (a) Output constraint: desired state bits that make o[t] match PID.
			let mut vote = vec![0i32; state_bits_in];
			for m in 0..self.num_motors {
				// Absolute + decouple: torque banks (m>=1) decode as (raw-0.5)*2 ∈
				// [-1,1], so the un-mixed torque CONTROL target (∈[-1,1]) must be
				// encoded back to raw-decode space as τ/2+0.5 (the inverse of
				// decode_outputs). Throttle bank, non-decouple, and the delta path
				// keep the direct target. THIS is the live DAGGER path (the per-step
				// train_output_step/edra_train_step are not what dagger_train calls).
				let p = self.output_decode_target(m, pid_pwms[t][m]);
				let n_true = (p * levels as f32) as usize;
				let motor_target: Vec<bool> = (0..levels).map(|i| i < n_true).collect();
				let cs = m * levels * obpn;
				let ce = (m + 1) * levels * obpn;
				let motor_conns = &self.output_connections[cs..ce];
				let base = m * levels;
				let entries_fn = |nn: usize| self.output_memory.neuron_entries(base + nn);
				let solved = solve_partial_connectivity_qsr_reachable(
					entries_fn, ram_core::neuron_memory::EMPTY_U8,
					motor_conns, levels, obpn, out_input_len,
					&rec_out_input[d], &motor_target, frame_bits, topk_per_neuron,
				);
				if let Some(sol) = solved {
					for i in 0..state_bits_in {
						vote[i] += if sol[frame_bits + i] { 1 } else { -1 };
					}
				}
			}
			let d_out: Vec<bool> = (0..state_bits_in)
				.map(|i| if vote[i] > 0 { true } else if vote[i] < 0 { false } else { rec_out_input[d][frame_bits + i] })
				.collect();

			// (b) Transition constraint (all but last step): the state at t should
			//     transition INTO d_next via the state layer at t+1. Solve the
			//     state layer for the prev-state bits (sensor bits immutable).
			let d_s: Vec<bool> = if let Some(ref dn) = d_next {
				// d_next is now sn bits (one MSB/side per state neuron, post 1-bit state).
				let target_sides: Vec<bool> = (0..self.state_neurons).map(|n| dn[n]).collect();
				let entries_fn = |nn: usize| self.state_memory.neuron_entries(nn);
				let solved = solve_partial_connectivity_qsr_reachable(
					entries_fn, ram_core::neuron_memory::EMPTY_U8,
					&self.state_connections, self.state_neurons, self.state_bits_per_neuron,
					state_input_len, &rec_state_input[d + 1], &target_sides, sensor_window, topk_per_neuron,
				);
				let d_trans: Vec<bool> = match solved {
					Some(sol) => (0..state_bits_in).map(|i| sol[sensor_window + i]).collect(),
					None => d_out.clone(),
				};
				// Aggregate: where output and transition agree, use it; on conflict
				// keep the current state bit (don't train an over-constrained bit).
				(0..state_bits_in)
					.map(|i| if d_out[i] == d_trans[i] { d_out[i] } else { rec_state_input[d][sensor_window + i] })
					.collect()
			} else {
				d_out.clone()
			};

			// (c) Commit STATE layer toward the desired state at the recorded address.
			// Option A: when integral-target mode is on, the desired state for neuron
			// n is a thermometer encoding of the PID integral — 3 axes (roll/pitch/yaw)
			// split across state_neurons/3 neurons each, each a TRUE/FALSE level. This
			// DIRECT dense target replaces d_s (the fragile output∧transition solve),
			// so the state actually learns to be the integrator. Extra neurons
			// (state_neurons % 3) fall back to d_s.
			let npa = self.state_neurons / 3;
			for n in 0..self.state_neurons {
				let target_val: u8 = if use_integral_target && npa > 0 && n < 3 * npa {
					let integ = &state_integral_targets.as_ref().unwrap()[t];
					let axis = n / npa;          // 0=roll,1=pitch,2=yaw
					let level = n % npa;
					let norm = ((integ[axis] + 1.0) * 0.5).clamp(0.0, 1.0); // [-1,1]→[0,1]
					if norm * (npa as f32) > (level as f32) { 3 } else { 0 }
				} else {
					// d_s is now sn bits (desired MSB/side per neuron) → drive the
					// cell fully to that side (TRUE=3 / FALSE=0).
					if d_s[n] { 3 } else { 0 }
				};
				let cs = n * self.state_bits_per_neuron;
				let ce = cs + self.state_bits_per_neuron;
				let addr = compute_address_sparse(&rec_state_input[d], &self.state_connections[cs..ce], self.state_bits_per_neuron);
				let cur = self.state_memory.read_cell(n, addr);
				// don't-punish: skip if cur is explicitly learned (not EMPTY) and
				// the target is on the opposite side (would erode learned behavior).
				if protect_learned && cur != ram_core::neuron_memory::EMPTY_U8
					&& (cur >= 2) != (target_val >= 2)
				{
					continue;
				}
				let nv = nudge_toward_value(cur, target_val);
				if nv != cur {
					self.state_memory.write_cell(n, addr, nv, true);
					s_writes += 1;
				}
			}

			// (d) Commit OUTPUT layer toward PID's PWM at the recorded out address.
			for n in 0..num_out {
				let motor = n / levels;
				let level_idx = n % levels;
				let p = self.output_decode_target(motor, pid_pwms[t][motor]);
				let target_true = (p * levels as f32) as usize > level_idx;
				let cs = n * obpn;
				let ce = cs + obpn;
				let addr = compute_address_sparse(&rec_out_input[d], &self.output_connections[cs..ce], obpn);
				let cur = self.output_memory.read_cell(n, addr);
				if protect_learned && cur != ram_core::neuron_memory::EMPTY_U8
					&& (cur >= 2) != target_true
				{
					continue;
				}
				let nv = crate::controller_training::nudge_toward_pub(cur, target_true);
				if nv != cur {
					self.output_memory.write_cell(n, addr, nv, true);
					o_writes += 1;
				}
			}

			d_next = Some(d_s);
		}
		(s_writes, o_writes)
	}

	// =========================================================================
	// State-splitting trainer (WNN_STATE_SPLIT). Design doc:
	// .claude/plans/controller_state_splitting_design.md
	//
	// A sibling to bptt_train_window — it does NOT replace it. Conflict-driven
	// constructive state induction: roll episodes on the CURRENT memory, find
	// where the SAME output-layer input is forced to DIFFERENT PWMs (a conflict
	// the memoryless output cannot satisfy), then plant a state distinction that
	// separates the histories. Phase 2 = scan + discriminative walk (Type-1).
	// =========================================================================

	/// Roll the given episodes on the current memory WITHOUT modifying it, and
	/// record per-step (output-layer input, PID PWM target) plus the per-episode
	/// state-layer-input history the backward walk needs. Returns the records as
	/// flat arrays. This is the Phase-2 recording pass (read-only).
	///
	/// Returns: (out_ins, pwm_targets, ep_of, step_of, state_ins_flat,
	///           state_in_len, ep_lengths) — where state_ins_flat is the
	///           concatenation of every step's state-layer input vector (each
	///           `state_in_len` bools), indexable per record.
	#[allow(clippy::type_complexity)]
	fn split_record(
		&mut self,
		gyros: Vec<Vec<[f32; 3]>>,
		accels: Vec<Vec<[f32; 3]>>,
		targets: Vec<Vec<[f32; 3]>>,
		pid_pwms: Vec<Vec<[f32; 4]>>,
	) -> (Vec<Vec<bool>>, Vec<[f32; 4]>, Vec<usize>, Vec<usize>, Vec<bool>, usize, Vec<usize>) {
		let bpf = self.bits_per_feature;
		let frame_bits = self.num_features * bpf;
		let sensor_window = self.input_window_k * frame_bits;
		let state_bits_in = self.state_neurons;
		let state_input_len = sensor_window + state_bits_in;
		let out_input_len = frame_bits + state_bits_in;

		let mut out_ins: Vec<Vec<bool>> = Vec::new();
		let mut pwms: Vec<[f32; 4]> = Vec::new();
		let mut ep_of: Vec<usize> = Vec::new();
		let mut step_of: Vec<usize> = Vec::new();
		let mut state_ins_flat: Vec<bool> = Vec::new();
		let mut ep_lengths: Vec<usize> = Vec::new();

		for ep in 0..gyros.len() {
			let iy = self.pending_init_yaws.get(ep).copied().unwrap_or(0.0); // yaw-anchor seed (0.0 ⇒ legacy)
			self.reset(iy);
			let w = gyros[ep].len();
			// Action-repeat: records exist only at DECISION steps. step_of is the
			// per-episode RECORD (decision) index because ep_start + step_of index
			// the record arrays downstream (walk lags become decision-space —
			// consistent with "window = last K decision frames"). ep_lengths is the
			// per-episode RECORD count (pushed after the loop). N=1 ⇒ identical.
			let mut dec = 0usize;
			for t in 0..w {
				let feats = self.compute_features(gyros[ep][t], accels[ep][t], targets[ep][t]);
				// Hold step: accumulators tick; no ring push / forward / record.
				if self.action_repeat > 1 {
					let hold = self.step_counter % self.action_repeat != 0;
					self.step_counter += 1;
					if hold {
						continue;
					}
				}
				let mut frame = vec![false; frame_bits];
				for f in 0..self.num_features {
					let row = f * bpf;
					for b in 0..bpf {
						frame[row + b] = feats[f] >= self.thresholds[row + b];
					}
				}
				if self.input_history.len() == self.input_window_k {
					self.input_history.pop_front();
				}
				self.input_history.push_back(frame.clone());

				let mut in_state = vec![false; state_input_len];
				let pad = self.input_window_k - self.input_history.len();
				for (i, fr) in self.input_history.iter().enumerate() {
					let slot = (pad + i) * frame_bits;
					in_state[slot..slot + frame_bits].copy_from_slice(fr);
				}
				for (n, &v) in self.prev_state.iter().enumerate() {
					in_state[sensor_window + n] = (v >> 1) & 1 != 0;
				}

				let mut new_state = vec![0u8; self.state_neurons];
				for n in 0..self.state_neurons {
					let cs = n * self.state_bits_per_neuron;
					let ce = cs + self.state_bits_per_neuron;
					let addr = compute_address_sparse(&in_state, &self.state_connections[cs..ce], self.state_bits_per_neuron);
					new_state[n] = self.state_memory.read_cell(n, addr);
				}

				let mut in_out = vec![false; out_input_len];
				in_out[0..frame_bits].copy_from_slice(&frame);
				for (n, &v) in new_state.iter().enumerate() {
					in_out[frame_bits + n] = (v >> 1) & 1 != 0;
				}

				out_ins.push(in_out);
				pwms.push(pid_pwms[ep][t]);
				ep_of.push(ep);
				step_of.push(dec);
				dec += 1;
				state_ins_flat.extend_from_slice(&in_state);
				self.prev_state = new_state;
			}
			ep_lengths.push(dec);
		}
		(out_ins, pwms, ep_of, step_of, state_ins_flat, state_input_len, ep_lengths)
	}

	/// Phase-2 scan: roll + record + bucket by output-layer input + flag PWM
	/// disagreement beyond `tau`. Read-only. Returns
	/// (num_records, [(spread, [(ep, step), ...]), ...]) worst-conflict-first —
	/// for inspection and the Phase-2 test.
	#[pyo3(signature = (gyros, accels, targets, pid_pwms, tau = 0.1))]
	#[allow(clippy::type_complexity)]
	fn split_scan(
		&mut self,
		gyros: Vec<Vec<[f32; 3]>>,
		accels: Vec<Vec<[f32; 3]>>,
		targets: Vec<Vec<[f32; 3]>>,
		pid_pwms: Vec<Vec<[f32; 4]>>,
		tau: f32,
	) -> (usize, Vec<(f32, Vec<(usize, usize)>)>) {
		let (out_ins, pwms, ep_of, step_of, _sif, _sil, _epl) =
			self.split_record(gyros, accels, targets, pid_pwms);
		let conflicts = crate::controller_split::scan_conflicts(&out_ins, &pwms, tau);
		let report = conflicts
			.iter()
			.map(|c| {
				let coords: Vec<(usize, usize)> =
					c.instances.iter().map(|&i| (ep_of[i], step_of[i])).collect();
				(c.spread, coords)
			})
			.collect();
		(out_ins.len(), report)
	}

	/// Phase-2 state-splitting: ONE greedy round (k=1). Roll + scan; take the
	/// worst conflict; run the discriminative backward walk; if a clean (Type-1)
	/// separator is found plant a latch (TYPE-1); else if the conflict is
	/// explained by an accumulated count install a thermometer counter (TYPE-2);
	/// retrain the output; re-scan to confirm resolution. Design doc §10 Phase 2-3.
	/// Returns:
	///   (conflicts_before, conflicts_after, mode, bit, lag_or_levels, score,
	///    direction, n_planted)
	/// where mode: 0 none / 1 TYPE-1 latch / 2 TYPE-2 counter; bit is the
	/// separator/accumulator feature (-1 if none); lag_or_levels is the TYPE-1 lag
	/// or TYPE-2 level count; score is gain (TYPE-1) or |corr| (TYPE-2); direction
	/// is high_on (TYPE-1) or count-up (TYPE-2); n_planted is state neurons written.
	#[pyo3(signature = (gyros, accels, targets, pid_pwms, tau = 0.1, clean_gain = 0.999, accum_corr = 0.9))]
	#[allow(clippy::type_complexity)]
	fn split_train(
		&mut self,
		gyros: Vec<Vec<[f32; 3]>>,
		accels: Vec<Vec<[f32; 3]>>,
		targets: Vec<Vec<[f32; 3]>>,
		pid_pwms: Vec<Vec<[f32; 4]>>,
		tau: f32,
		clean_gain: f32,
		accum_corr: f32,
	) -> (usize, usize, i64, i64, i64, f32, bool, i64) {
		// 1. record (bootstrap roll on current memory)
		let (out_ins, pwms, ep_of, step_of, sif, sil, epl) =
			self.split_record(gyros.clone(), accels.clone(), targets.clone(), pid_pwms.clone());
		// 2. scan
		let conflicts = crate::controller_split::scan_conflicts(&out_ins, &pwms, tau);
		let conflicts_before = conflicts.len();
		if conflicts.is_empty() {
			return (0, 0, 0, -1, -1, 0.0, false, 0);
		}
		// episode-major record starts
		let mut ep_start = vec![0usize; epl.len()];
		let mut acc = 0usize;
		for (e, &len) in epl.iter().enumerate() {
			ep_start[e] = acc;
			acc += len;
		}
		// candidate bits = frame bits (< sensor_window) some state neuron observes
		let frame_bits = self.num_features * self.bits_per_feature;
		let sensor_window = self.input_window_k * frame_bits;
		let mut candidate_bits: Vec<usize> = self
			.state_connections
			.iter()
			.map(|&x| x as usize)
			.filter(|&b| b < sensor_window)
			.collect();
		candidate_bits.sort_unstable();
		candidate_bits.dedup();

		// worst conflict (greedy k=1); label high/low by the max-spread motor
		let c = &conflicts[0];
		let labels = crate::controller_split::label_high_low(&c.instances, &pwms);
		let max_lag = c.instances.iter().map(|&i| step_of[i]).min().unwrap_or(0);

		// 3. TYPE-1: discriminative walk for a clean single-(bit,lag) separator
		let sep = crate::controller_split::discriminative_walk(
			&c.instances, &labels, &ep_of, &step_of, &ep_start, &sif, sil, &candidate_bits, max_lag,
		);
		let (mut mode, mut sbit, mut slag_lv, mut sscore, mut sdir, mut n_planted) =
			(0i64, -1i64, -1i64, 0.0f32, false, 0i64);

		let no_used = vec![false; self.state_neurons];
		if let Some(s) = sep.as_ref().filter(|s| s.gain >= clean_gain) {
			if let Some(neuron) = self.split_plant_latch(s.bit, s.high_on, &no_used, &sif, sil) {
				mode = 1;
				sbit = s.bit as i64;
				slag_lv = s.lag as i64;
				sscore = s.gain;
				sdir = s.high_on;
				n_planted = 1;
				let _ = neuron;
				self.split_retrain_output(&gyros, &accels, &targets, &pid_pwms, false);
			}
		}

		// 4. TYPE-2: no clean stump → is the conflict explained by an accumulated
		//    count? (the integral signal). Correlate each feature's window-count
		//    with the disagreeing motor's PWM.
		if mode == 0 {
			// disagreeing motor = the one with the largest spread across instances
			let mut best_m = 0usize;
			let mut best_s = -1.0f32;
			for m in 0..self.num_motors {
				let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
				for &i in &c.instances {
					lo = lo.min(pwms[i][m]);
					hi = hi.max(pwms[i][m]);
				}
				if hi - lo > best_s {
					best_s = hi - lo;
					best_m = m;
				}
			}
			let pwm_scalar: Vec<f32> = c.instances.iter().map(|&i| pwms[i][best_m]).collect();
			// 4a. BIDIRECTIONAL (mode 3): a signed net count that must UNWIND — try
			//     first when the chain is wide enough (sbpn>=5) to hold an up/down
			//     counter. Resolves conflicts an increment-only integral cannot.
			if self.state_bits_per_neuron >= 5 {
				let bi = crate::controller_split::detect_accumulator_bidir(
					&c.instances, &pwm_scalar, &ep_of, &step_of, &ep_start, &sif, sil, &candidate_bits, max_lag,
				);
				if let Some(b) = bi.filter(|b| b.corr >= accum_corr) {
					if let Some(neurons) = self.split_install_counter_bidir(b.up, b.dn, self.state_neurons, &no_used) {
						mode = 3;
						sbit = b.up as i64;
						slag_lv = neurons.len() as i64;
						sscore = b.corr;
						sdir = true;
						n_planted = neurons.len() as i64;
						self.split_retrain_output(&gyros, &accels, &targets, &pid_pwms, false);
					}
				}
			}
			// 4b. INCREMENT-only (mode 2): the saturating integral.
			let accum = crate::controller_split::detect_accumulator(
				&c.instances, &pwm_scalar, &ep_of, &step_of, &ep_start, &sif, sil, &candidate_bits, max_lag,
			);
			if mode == 0 {
				if let Some(a) = accum.filter(|a| a.corr >= accum_corr) {
					if let Some(neurons) = self.split_install_counter(a.bit, self.state_neurons, &no_used, &sif, sil) {
						mode = 2;
						sbit = a.bit as i64;
						slag_lv = neurons.len() as i64;
						sscore = a.corr;
						sdir = a.up;
						n_planted = neurons.len() as i64;
						self.split_retrain_output(&gyros, &accels, &targets, &pid_pwms, false);
					}
				}
			}
		}

		// 5. re-scan
		let (out_ins2, pwms2, _e2, _s2, _f2, _l2, _p2) =
			self.split_record(gyros, accels, targets, pid_pwms);
		let conflicts_after = crate::controller_split::scan_conflicts(&out_ins2, &pwms2, tau).len();
		(conflicts_before, conflicts_after, mode, sbit, slag_lv, sscore, sdir, n_planted)
	}

	/// Phase-4 state-splitting: the MULTI-ROUND consistency loop. Bootstraps from
	/// the memoryless controller and, each round, scans conflicts and commits up
	/// to k(round) of them (the worst first) before re-rolling + retraining the
	/// output. k(round) = k_start + round implements the greedy→batch anneal
	/// (design §7): few splits/round early when distinctions interact strongly,
	/// more later when residual conflicts are independent. A `used` guard enforces
	/// the collision rule (one distinction per neuron). Converges when no conflict
	/// exceeds tau, or stalls when a round resolves nothing. Returns
	///   (rounds_run, conflicts_final, planted_total, committed_per_round,
	///    saturation_pressure, connectivity_wish_bits)
	/// where the last two are the trainer's half of the GA handshake (design §8):
	/// saturation_pressure = unresolved conflicts whose separator IS observed
	/// (grow state_neurons); connectivity_wish_bits = state-input positions a
	/// separator wanted but no neuron observes (route a neuron there).
	#[pyo3(signature = (gyros, accels, targets, pid_pwms, tau = 0.1, clean_gain = 0.999, accum_corr = 0.9, max_rounds = 8, k_start = 1, coarse_target = 0, selective_output = false, init_yaws = vec![]))]
	#[allow(clippy::type_complexity, clippy::too_many_arguments)]
	pub fn split_train_loop(
		&mut self,
		gyros: Vec<Vec<[f32; 3]>>,
		accels: Vec<Vec<[f32; 3]>>,
		targets: Vec<Vec<[f32; 3]>>,
		pid_pwms: Vec<Vec<[f32; 4]>>,
		tau: f32,
		clean_gain: f32,
		accum_corr: f32,
		max_rounds: usize,
		k_start: usize,
		coarse_target: usize,
		selective_output: bool,
		// Yaw-anchor: per-episode initial yaw (rad), parallel to gyros' episodes. Empty
		// ⇒ legacy 0.0 seed. Stashed so split_record/split_retrain_output re-seed yaw.
		init_yaws: Vec<f32>,
	) -> (usize, usize, usize, Vec<usize>, usize, Vec<usize>) {
		self.pending_init_yaws = init_yaws;
		// Adaptive coarse-signature bucketing when coarse_target>0 (real
		// trajectories); exact full-frame when 0 (synthetic fixtures). Closure
		// captures the frame layout so both scan sites stay consistent.
		let frame_bits_c = self.num_features * self.bits_per_feature;
		let bpf_c = self.bits_per_feature;
		// Capture num_features as a Copy local so the closure does NOT borrow
		// &self (else it conflicts with the &mut self split_record/retrain calls).
		let num_features_c = self.num_features;
		let scan = |outs: &[Vec<bool>], pw: &[[f32; 4]]| -> Vec<crate::controller_split::Conflict> {
			if coarse_target > 0 {
				crate::controller_split::scan_conflicts_coarse(
					outs, pw, tau, bpf_c, num_features_c, frame_bits_c, coarse_target,
				)
				.0
			} else {
				crate::controller_split::scan_conflicts(outs, pw, tau)
			}
		};
		let frame_bits = self.num_features * self.bits_per_feature;
		let sensor_window = self.input_window_k * frame_bits;
		let mut candidate_bits: Vec<usize> = self
			.state_connections
			.iter()
			.map(|&x| x as usize)
			.filter(|&b| b < sensor_window)
			.collect();
		candidate_bits.sort_unstable();
		candidate_bits.dedup();

		let mut used = vec![false; self.state_neurons];
		let mut planted_total = 0usize;
		let mut per_round: Vec<usize> = Vec::new();
		let mut rounds_run = 0usize;
		let profile = std::env::var("WNN_SPLIT_PROFILE").map(|s| s == "1").unwrap_or(false);

		for round in 0..max_rounds {
			let t_rec = std::time::Instant::now();
			let (out_ins, pwms, ep_of, step_of, sif, sil, epl) =
				self.split_record(gyros.clone(), accels.clone(), targets.clone(), pid_pwms.clone());
			let d_rec = t_rec.elapsed();
			let t_scan = std::time::Instant::now();
			let conflicts = scan(&out_ins, &pwms);
			let d_scan = t_scan.elapsed();
			if conflicts.is_empty() {
				break; // converged
			}
			rounds_run = round + 1;
			let mut ep_start = vec![0usize; epl.len()];
			let mut acc = 0usize;
			for (e, &len) in epl.iter().enumerate() {
				ep_start[e] = acc;
				acc += len;
			}
			if profile {
				let tot_inst: usize = conflicts.iter().map(|c| c.instances.len()).sum();
				let max_inst = conflicts.iter().map(|c| c.instances.len()).max().unwrap_or(0);
				eprintln!(
					"[SPLIT_PROFILE] round {round}: records={} conflicts={} tot_inst={} max_inst={} candidate_bits={} | record={:.2?} scan={:.2?}",
					out_ins.len(), conflicts.len(), tot_inst, max_inst, candidate_bits.len(), d_rec, d_scan
				);
			}
			let t_res = std::time::Instant::now();
			let k = k_start + round; // greedy → batch anneal
			let mut committed = 0usize;
			let mut attempts = 0usize;
			for c in conflicts.iter() {
				if committed >= k {
					break;
				}
				attempts += 1;
				let (mode, neurons) = self.split_resolve_conflict(
					&c.instances, &pwms, &ep_of, &step_of, &ep_start, &sif, sil,
					&candidate_bits, clean_gain, accum_corr, &used,
				);
				if mode != 0 {
					for n in neurons {
						if n < used.len() {
							used[n] = true;
						}
					}
					committed += 1;
					planted_total += 1;
				}
			}
			per_round.push(committed);
			if profile {
				eprintln!(
					"[SPLIT_PROFILE] round {round}: resolve_attempts={attempts} committed={committed} | resolve={:.2?}",
					t_res.elapsed()
				);
			}
			if committed == 0 {
				break; // stalled: no resolvable conflict this round
			}
			let t_rt = std::time::Instant::now();
			self.split_retrain_output(&gyros, &accels, &targets, &pid_pwms, selective_output);
			if profile {
				eprintln!("[SPLIT_PROFILE] round {round}: retrain={:.2?}", t_rt.elapsed());
			}
		}

		// Final scan + PRESSURE analysis (Phase 5a): for each conflict the trainer
		// could NOT resolve, ask the discriminative walk over ALL frame bits what
		// WOULD have separated it. If that wish bit is one no neuron observes, it
		// is CONNECTIVITY pressure (route a neuron there); if it IS observed yet
		// the conflict stayed unresolved, the trainer ran out of free/wired neurons
		// → SATURATION pressure (grow state_neurons). These wishes are the trainer's
		// half of the GA handshake (design §8).
		let (out_ins, pwms, ep_of, step_of, sif, sil, epl) =
			self.split_record(gyros, accels, targets, pid_pwms);
		let conflicts = scan(&out_ins, &pwms);
		let conflicts_final = conflicts.len();
		let mut ep_start = vec![0usize; epl.len()];
		let mut acc = 0usize;
		for (e, &len) in epl.iter().enumerate() {
			ep_start[e] = acc;
			acc += len;
		}
		let t_wish = std::time::Instant::now();
		let all_bits: Vec<usize> = (0..sensor_window).collect();
		let observed: std::collections::HashSet<usize> = candidate_bits.iter().copied().collect();
		let mut saturation = 0usize;
		let mut wish_bits: Vec<usize> = Vec::new();
		for c in conflicts.iter() {
			let sampled = subsample_instances(&c.instances, SPLIT_INST_CAP);
			let labels = crate::controller_split::label_high_low(&sampled, &pwms);
			let max_lag = sampled
				.iter()
				.map(|&i| step_of[i])
				.min()
				.unwrap_or(0)
				.min(SPLIT_LAG_CAP);
			if let Some(s) = crate::controller_split::discriminative_walk(
				&sampled, &labels, &ep_of, &step_of, &ep_start, &sif, sil, &all_bits, max_lag,
			)
			.filter(|s| s.gain >= clean_gain)
			{
				if observed.contains(&s.bit) {
					saturation += 1; // a separator exists & is seen, but no free neuron
				} else {
					wish_bits.push(s.bit); // route a neuron to this currently-unseen bit
				}
			}
		}
		wish_bits.sort_unstable();
		wish_bits.dedup();
		if profile {
			eprintln!(
				"[SPLIT_PROFILE] wish-analysis: final_conflicts={} all_bits={} | {:.2?}",
				conflicts_final, all_bits.len(), t_wish.elapsed()
			);
		}
		(rounds_run, conflicts_final, planted_total, per_round, saturation, wish_bits)
	}

	/// Combined step() + per-motor EDRA train in one call. The real-EDRA
	/// analog of step_and_train. Returns (pwm, total_cells_written).
	#[pyo3(signature = (gyro, accel, target_attitude, target_pwm, topk_per_neuron = 4))]
	fn step_and_edra_train(
		&mut self,
		gyro: [f32; 3],
		accel: [f32; 3],
		target_attitude: [f32; 3],
		target_pwm: [f32; 4],
		topk_per_neuron: usize,
	) -> (Vec<f32>, usize) {
		let pwm = self.step(gyro, accel, target_attitude);
		let (s, o) = self.edra_train_step(target_pwm, topk_per_neuron);
		(pwm, s + o)
	}

	/// One controller cycle. Returns 4 motor PWMs in [0, 1].
	pub fn step(&mut self,
	        gyro: [f32; 3],
	        accel: [f32; 3],
	        target_attitude: [f32; 3]) -> Vec<f32> {
		// 1. Build the observation feature vector (9 base + enabled H2 extras) and
		//    thermometer-encode it. compute_features runs EVERY physical step
		//    (integ[]/yaw_heading are physical-time accumulators) — even on
		//    action-repeat hold steps.
		let bpf = self.bits_per_feature;
		let feats = self.compute_features(gyro, accel, target_attitude);
		// Action-repeat hold: between decision steps return the held PWM. No
		// frame-encode, no ring push, no forward, no decode; prev_state and the
		// last_* caches stay the DECISION step's (intentional). Guarded behind
		// >1 so the N=1 hot path is untouched (step_counter stays 0).
		if self.action_repeat > 1 {
			let hold = self.step_counter % self.action_repeat != 0;
			self.step_counter += 1;
			if hold {
				return self.last_pwm.clone();
			}
		}
		let mut frame = vec![false; self.num_features * bpf];
		for f in 0..self.num_features {
			let v = feats[f];
			let row_start = f * bpf;
			for b in 0..bpf {
				let t = self.thresholds[row_start + b];
				frame[row_start + b] = v >= t;
			}
		}

		// 2. Push into the K-step ring buffer.
		if self.input_history.len() == self.input_window_k {
			self.input_history.pop_front();
		}
		self.input_history.push_back(frame);

		// 3. Assemble the full state-layer input: K frames (pad front with zeros
		//    if we don't have K yet) then 2 bits per recurrent-state neuron.
		let frame_bits = self.num_features * bpf;
		let sensor_total = self.input_window_k * frame_bits;
		let state_bits_in = self.state_neurons; // 1 bit (QSR MSB) per state neuron (was 2·)
		let total_input_bits = sensor_total + state_bits_in;
		let mut input_bits = vec![false; total_input_bits];

		// Frames: oldest first. If history has fewer than K, the missing oldest
		// slots stay zero (paddding with no past observation).
		let pad = self.input_window_k - self.input_history.len();
		for (i, frame) in self.input_history.iter().enumerate() {
			let slot = (pad + i) * frame_bits;
			input_bits[slot..slot + frame_bits].copy_from_slice(frame);
		}

		// Recurrent state = 1 bit/neuron = the QSR MSB ((v>>1)&1 = the SIDE,
		// fired/not). The LSB was training-confidence, semantically wrong to feed
		// back into the address (08/06/2026).
		for (n, &v) in self.prev_state.iter().enumerate() {
			input_bits[sensor_total + n] = (v >> 1) & 1 != 0;
		}

		// Cache the state-layer input AS-OF this step so train_state_step
		// can compute the same addresses step() read from.
		self.last_state_layer_input = input_bits.clone();

		// 4. State-layer forward: compute address per neuron, read its cell.
		let mut new_state = vec![0u8; self.state_neurons];
		for n in 0..self.state_neurons {
			let conn_start = n * self.state_bits_per_neuron;
			let conn_end = conn_start + self.state_bits_per_neuron;
			let address = compute_address_sparse(
				&input_bits,
				&self.state_connections[conn_start..conn_end],
				self.state_bits_per_neuron,
			);
			new_state[n] = self.state_memory.read_cell(n, address);
		}

		// 5. Output-layer (Mealy) input: [current sensor frame | new_state].
		//    The output neurons are FULLY connected to the state (every state
		//    bit) so each motor bank knows the GLOBAL state, plus they sample
		//    some current-input bits (Mealy: react to the current observation,
		//    not only the state). State bits occupy the high indices.
		let frame_bits = self.num_features * bpf;
		let out_input_len = frame_bits + state_bits_in;
		let mut output_input = vec![false; out_input_len];
		if let Some(cur_frame) = self.input_history.back() {
			output_input[0..frame_bits].copy_from_slice(cur_frame);
		}
		for (n, &v) in new_state.iter().enumerate() {
			output_input[frame_bits + n] = (v >> 1) & 1 != 0; // MSB only
		}
		// Cache for edra_train_step (it solves the state bits; frame is immutable).
		self.last_output_layer_input = output_input.clone();

		// 6. Output-layer forward.
		let num_out = self.num_motors * self.levels_per_motor;
		for n in 0..num_out {
			let conn_start = n * self.output_bits_per_neuron;
			let conn_end = conn_start + self.output_bits_per_neuron;
			let address = compute_address_sparse(
				&output_input,
				&self.output_connections[conn_start..conn_end],
				self.output_bits_per_neuron,
			);
			self.last_output_cells[n] = self.output_memory.read_cell(n, address);
		}

		// 7. Strategy-5 decode → motor PWMs. delta-control applies the accumulator;
		//    decouple_outputs (H3) decodes 4 CONTROLS then mixes to motors. See
		//    decode_outputs() — the single source of truth (mirrored by the shader).
		let pwm = self.decode_outputs();

		// 8. Update recurrent state for next step.
		self.prev_state = new_state;

		// Action-repeat: remember this decision's PWM for the upcoming hold
		// steps (clone gated behind >1 to keep the N=1 hot path unchanged).
		if self.action_repeat > 1 {
			self.last_pwm.clone_from(&pwm);
		}

		pwm
	}

	#[getter]
	pub fn num_motors(&self) -> usize { self.num_motors }
	#[getter]
	pub fn levels_per_motor(&self) -> usize { self.levels_per_motor }
	#[getter]
	fn state_neurons(&self) -> usize { self.state_neurons }
	#[getter]
	fn input_window_k(&self) -> usize { self.input_window_k }
	#[getter]
	fn bits_per_feature(&self) -> usize { self.bits_per_feature }
}

// =============================================================================
// H2 observation features (18/06/2026) — plain impl, NOT Python-exposed.
// THE single source of truth for the feature vector: step() and every training
// rollout call this, and controller_rollout.metal mirrors it bit-for-bit (the
// cpu/gpu parity test guards the equivalence). Stateful: updates the leaky
// integral accumulators + yaw heading, so call EXACTLY ONCE per timestep, in
// rollout order, after reset() at episode start.
// =============================================================================
impl WnnController {
	/// Build the length-`num_features` observation vector: the 9 raw sensors
	/// followed by the enabled error/integral extras in canonical order
	/// [tilt_p, tilt_i, roll_p, pitch_p, yaw_p, roll_i, pitch_i, yaw_i].
	/// accel = -gravity_body, so at level accel=(0,0,+g): tilt grows from 0.
	/// Integrals are per-step leaky (acc = leak·acc + err; constant dt folded into
	/// integral_scale) so the Metal twin needs no dt. Caches into last_feature_vector.
	fn compute_features(&mut self, gyro: [f32; 3], accel: [f32; 3], target: [f32; 3]) -> Vec<f32> {
		let mut feats = Vec::with_capacity(self.num_features);
		feats.extend_from_slice(&[
			gyro[0], gyro[1], gyro[2],
			accel[0], accel[1], accel[2],
			target[0], target[1], target[2],
		]);
		// Parity anchor: all toggles off ⇒ exactly the original 9 features.
		if self.num_features == NUM_FEATURES {
			self.last_feature_vector.clone_from(&feats);
			return feats;
		}
		// Derived errors from the IMU (accel-only attitude; yaw dead-reckoned).
		let (ax, ay, az) = (accel[0], accel[1], accel[2]);
		let tilt = (ax * ax + ay * ay).sqrt().atan2(az); // angle-to-up, 0 at level
		let roll_est = ay.atan2(az);
		let pitch_est = (-ax).atan2((ay * ay + az * az).sqrt());
		// Yaw-anchored: integrate gyro-z with the REAL dt (yaw_heading was seeded to
		// the episode's true initial yaw in reset) → an ABSOLUTE yaw estimate.
		// Un-anchored: legacy gyro-z sum (constant dt absorbed) feeding only the
		// dead-reckoned peraxis yaw. Gated so the legacy feature stays bit-unchanged.
		if self.obs_yaw_err || self.obs_yaw_err_i {
			self.yaw_heading += gyro[2] * self.dt;
		} else {
			self.yaw_heading += gyro[2];
		}
		let roll_err = target[0] - roll_est;
		let pitch_err = target[1] - pitch_est;
		let yaw_err = target[2] - self.yaw_heading;
		// Append enabled features (canonical order); update integrals in lockstep.
		let mut iacc = 0usize;
		if self.obs_tilt_p { feats.push(tilt); }
		if self.obs_tilt_i {
			self.integral_acc[iacc] = self.integral_leak * self.integral_acc[iacc] + tilt;
			feats.push(self.integral_acc[iacc] * self.integral_scale);
			iacc += 1;
		}
		if self.obs_peraxis_p {
			feats.push(roll_err);
			feats.push(pitch_err);
			if self.obs_peraxis_yaw { feats.push(yaw_err); }  // yaw dropped when ref is unobservable
		}
		if self.obs_peraxis_i {
			// roll/pitch always; yaw only when its (dead-reckoned) reference is enabled.
			let errs: &[f32] = if self.obs_peraxis_yaw { &[roll_err, pitch_err, yaw_err] }
			                   else { &[roll_err, pitch_err] };
			for &e in errs {
				self.integral_acc[iacc] = self.integral_leak * self.integral_acc[iacc] + e;
				feats.push(self.integral_acc[iacc] * self.integral_scale);
				iacc += 1;
			}
		}
		if self.obs_pwm {
			// The throttle accumulator AS-OF step start (self.pwm is updated only
			// AFTER the output decode), i.e. the hidden state the optimal delta
			// depends on. Direct fix for delta's partial observability — unlike
			// ∫error (obs_tilt_i), this is an EXACT readout in every regime.
			for m in 0..self.num_motors {
				feats.push(self.pwm[m]);
			}
		}
		// Yaw-anchor (clean scalar channel, canonical order LAST): proportional yaw
		// error + its leaky integral. yaw_heading is the ABSOLUTE anchored estimate
		// (seeded to init_yaw, dt-integrated above). A dedicated scalar — NOT via
		// obs_peraxis — sidesteps the roll/pitch atan2 degeneracy at large tilt.
		if self.obs_yaw_err { feats.push(yaw_err); }
		if self.obs_yaw_err_i {
			self.integral_acc[iacc] = self.integral_leak * self.integral_acc[iacc] + yaw_err;
			feats.push(self.integral_acc[iacc] * self.integral_scale);
		}
		self.last_feature_vector.clone_from(&feats);
		feats
	}

	/// Decode the last output cells → 4 motor PWMs. Reads self.last_output_cells
	/// (already populated by the output-layer forward pass), applies the delta
	/// accumulator, and — under decouple_outputs (H3) — treats the 4 banks as
	/// CONTROLS [T, τ_roll, τ_pitch, τ_yaw] (Option A: accumulate per control with
	/// neutral T→0.5 / torque→0, THEN mix to motors). Mirrored by the Metal shader.
	/// decouple_outputs==false ⇒ the original per-motor decode (parity anchor).
	fn decode_outputs(&mut self) -> Vec<f32> {
		if self.delta_control {
			self.pwm_prev.copy_from_slice(&self.pwm);
		}
		let denom = self.levels_per_motor as f32;
		let mut out = Vec::with_capacity(self.num_motors);
		for m in 0..self.num_motors {
			let start = m * self.levels_per_motor;
			let mut sum = 0.0f32;
			for &cell in &self.last_output_cells[start..start + self.levels_per_motor] {
				sum += QSR_WEIGHTS[(cell & 0x3) as usize];
			}
			let decoded = (sum / denom).clamp(0.0, 1.0);
			if self.decouple_outputs {
				// banks: 0=T (neutral 0.5, [0,1]); 1..=3 = torques (neutral 0, [-1,1]).
				let is_torque = m >= 1;
				let neutral = if is_torque { 0.0 } else { 0.5 };
				let lo = if is_torque { -1.0 } else { 0.0 };
				self.pwm[m] = if self.delta_control {
					let delta = decoded_to_delta(decoded, self.delta_max);
					(neutral + self.delta_leak * (self.pwm[m] - neutral) + delta).clamp(lo, 1.0)
				} else if is_torque {
					(decoded - 0.5) * 2.0   // absolute: map [0,1] → [-1,1]
				} else {
					decoded
				};
				out.push(self.pwm[m]);
			} else if self.delta_control {
				let leaked = 0.5 + self.delta_leak * (self.pwm[m] - 0.5);
				self.pwm[m] = (leaked + decoded_to_delta(decoded, self.delta_max)).clamp(0.0, 1.0);
				out.push(self.pwm[m]);
			} else {
				out.push(decoded);
			}
		}
		if self.decouple_outputs { self.mix_controls_to_motors() } else { out }
	}

	/// Fixed control-allocation mix: controls [T, τ_roll, τ_pitch, τ_yaw] (= self.pwm)
	/// → 4 motor PWMs, signs matching the sim's torque convention
	/// (roll=−th1+th3, pitch=−th0+th2, yaw=th0−th1+th2−th3). Motors clamped [0,1].
	fn mix_controls_to_motors(&self) -> Vec<f32> {
		let (t, tr, tp, ty) = (self.pwm[0], self.pwm[1], self.pwm[2], self.pwm[3]);
		vec![
			(t - tp + ty).clamp(0.0, 1.0),  // 0 front
			(t - tr - ty).clamp(0.0, 1.0),  // 1 left
			(t + tp + ty).clamp(0.0, 1.0),  // 2 back
			(t + tr - ty).clamp(0.0, 1.0),  // 3 right
		]
	}
}

// =============================================================================
// State-splitting helpers (NOT Python-exposed — plain impl so they can take
// slice refs). Called from split_train in the #[pymethods] block above.
// =============================================================================

// Perf caps for the discriminative walk / accumulator detection (Phase 5d perf
// fix). The walk/detect cost is O(conflicts × candidate_bits × instances ×
// max_lag); on real hovering trajectories a single coarse bucket can hold
// hundreds of instances and span the whole episode, making one resolve take
// tens of seconds. The separator scores (purity / Pearson) are STATISTICS — a
// bounded sample of instances gives the same answer — and useful Type-1
// separators are RECENT (long-range memory is the integral's job, Type-2). So
// we cap both. Synthetic fixtures (tiny instances, lag ≤ ~5) are far below these
// caps → unaffected.
pub(crate) const SPLIT_INST_CAP: usize = 128; // max instances used for separator statistics
pub(crate) const SPLIT_LAG_CAP: usize = 48; // max lookback for the Type-1 walk / counts

/// Deterministically subsample instance indices to at most `cap` (even stride),
/// so separator statistics stay O(cap) regardless of bucket size.
pub(crate) fn subsample_instances(instances: &[usize], cap: usize) -> Vec<usize> {
	if instances.len() <= cap {
		return instances.to_vec();
	}
	let stride = (instances.len() / cap).max(1);
	instances.iter().step_by(stride).take(cap).copied().collect()
}

impl WnnController {
	/// Unique base addresses neuron `c` reads across the recorded state-layer
	/// inputs, with the `relevant` connection positions MASKED OUT. Lets the
	/// caller write a truth table over only the few relevant bits at the patterns
	/// the controller ACTUALLY visits — `2^relevant × visited` cells instead of
	/// `2^sbpn` (catastrophic for realistic neurons, sbpn≈35). The other bits take
	/// their visited values, so the addresses the forward-ripple reaches (same
	/// sensor patterns, flipped self/relevant bits) are covered.
	fn split_visited_bases(&self, c: usize, sif: &[bool], sil: usize, relevant: &[usize]) -> Vec<u64> {
		let sbpn = self.state_bits_per_neuron;
		let conns = &self.state_connections[c * sbpn..(c + 1) * sbpn];
		let mut mask: u64 = if sbpn >= 64 { u64::MAX } else { (1u64 << sbpn) - 1 };
		for &p in relevant {
			mask &= !(1u64 << (sbpn - 1 - p));
		}
		let n_rec = if sil == 0 { 0 } else { sif.len() / sil };
		let mut set: std::collections::HashSet<u64> = std::collections::HashSet::new();
		for r in 0..n_rec {
			let addr = compute_address_sparse(&sif[r * sil..(r + 1) * sil], conns, sbpn);
			set.insert(addr & mask);
		}
		set.into_iter().collect()
	}

	/// Plant a set/hold latch realizing the walk-found distinction. Finds a state
	/// neuron that observes BOTH the separator `bit` AND its own self-loop bit (so
	/// the latch can hold via the recurrence — Departure 3 connectivity gate), then
	/// writes ON where self-loop is set (hold) or the trigger is in the SET
	/// direction (`high_on`). SPARSE: only at visited sensor patterns × the 4
	/// (trigger, self) combos — NOT all 2^sbpn addresses. Returns the neuron used,
	/// or None if no connected neuron can express it (→ Phase-5 GA pressure).
	/// Select the latch neuron for `bit`: the first UNUSED state neuron that observes
	/// BOTH the trigger `bit` AND its own self-loop bit. Returns (neuron, trig_pos,
	/// self_pos), or None. Shared by the CPU planter and the GPU port so selection
	/// can't drift between them.
	pub(crate) fn plant_latch_neuron(&self, bit: usize, used: &[bool]) -> Option<(usize, usize, usize)> {
		let frame_bits = self.num_features * self.bits_per_feature;
		let sensor_window = self.input_window_k * frame_bits;
		let sbpn = self.state_bits_per_neuron;
		for c in 0..self.state_neurons {
			if used.get(c).copied().unwrap_or(false) {
				continue; // collision rule: one distinction per neuron (design §5)
			}
			let conns = &self.state_connections[c * sbpn..(c + 1) * sbpn];
			let self_idx = sensor_window + c;
			let trig_pos = conns.iter().position(|&x| x as usize == bit);
			let self_pos = conns.iter().position(|&x| x as usize == self_idx);
			if let (Some(tp), Some(sp)) = (trig_pos, self_pos) {
				return Some((c, tp, sp));
			}
		}
		None
	}

	fn split_plant_latch(&self, bit: usize, high_on: bool, used: &[bool], sif: &[bool], sil: usize) -> Option<usize> {
		let sbpn = self.state_bits_per_neuron;
		let (c, tp, sp) = self.plant_latch_neuron(bit, used)?;
		let tmask = 1u64 << (sbpn - 1 - tp);
		let smask = 1u64 << (sbpn - 1 - sp);
		for base in self.split_visited_bases(c, sif, sil, &[tp, sp]) {
			for tv in 0..2u64 {
				for sv in 0..2u64 {
					let addr = base | (tv * tmask) | (sv * smask);
					let on = sv == 1 || (tv == 1) == high_on; // hold OR set-direction
					self.state_memory.write_cell(c, addr, if on { 3 } else { 1 }, true);
				}
			}
		}
		Some(c)
	}

	/// Install a TYPE-2 integral as a gated thermometer counter on `trigger`.
	/// CONNECTIVITY-AGNOSTIC (Phase 6a): instead of requiring a hand-wired
	/// positional chain, it (1) picks up to `max_levels` FREE neurons that observe
	/// the trigger (each becomes one level — level k's "lower" = level k-1's
	/// feedback bit, always observable via the forced full-state prefix), (2)
	/// position-FINDS {trigger, lower, self} in each neuron's actual connections,
	/// (3) writes the gated increment+hold table SPARSELY — visited bases × the
	/// few relevant bits, NOT 2^sbpn. level 0 is a plain latch (on = self OR
	/// trigger); level k>0 increments when the level below is already on
	/// (on = self OR (trigger AND lower)) so each fire advances one level via the
	/// recurrence. Needs ≥2 trigger-observing free neurons (else None → saturation
	/// pressure). Direction is handled by the output retrain. v1 increment-only.
	/// Select the increment-counter chain for `trigger`: up to `max_levels` UNUSED
	/// state neurons that observe the trigger, in index order. Shared by the CPU
	/// planter and the GPU port. May return <2 (caller rejects).
	pub(crate) fn plant_counter_chain(&self, trigger: usize, max_levels: usize, used: &[bool]) -> Vec<usize> {
		let sbpn = self.state_bits_per_neuron;
		let mut chain: Vec<usize> = Vec::new();
		for c in 0..self.state_neurons {
			if used.get(c).copied().unwrap_or(false) {
				continue;
			}
			let conns = &self.state_connections[c * sbpn..(c + 1) * sbpn];
			if conns.iter().any(|&x| x as usize == trigger) {
				chain.push(c);
				if chain.len() >= max_levels {
					break;
				}
			}
		}
		chain
	}

	fn split_install_counter(&self, trigger: usize, max_levels: usize, used: &[bool], sif: &[bool], sil: usize) -> Option<Vec<usize>> {
		let frame_bits = self.num_features * self.bits_per_feature;
		let sensor_window = self.input_window_k * frame_bits;
		let sbpn = self.state_bits_per_neuron;
		if sbpn < 2 {
			return None;
		}
		let chain = self.plant_counter_chain(trigger, max_levels, used);
		if chain.len() < 2 {
			return None; // need ≥2 levels for an integral (1 = just a latch)
		}
		for k in 0..chain.len() {
			let c = chain[k];
			let conns = &self.state_connections[c * sbpn..(c + 1) * sbpn];
			let tp = conns.iter().position(|&x| x as usize == trigger)?;
			let sp = conns.iter().position(|&x| x as usize == sensor_window + c)?;
			let tmask = 1u64 << (sbpn - 1 - tp);
			let smask = 1u64 << (sbpn - 1 - sp);
			if k == 0 {
				// level 0 = latch: on = self OR trigger
				for base in self.split_visited_bases(c, sif, sil, &[tp, sp]) {
					for tv in 0..2u64 {
						for sv in 0..2u64 {
							let addr = base | (tv * tmask) | (sv * smask);
							self.state_memory.write_cell(c, addr, if sv == 1 || tv == 1 { 3 } else { 1 }, true);
						}
					}
				}
			} else {
				// level k>0: on = self OR (trigger AND lower=level k-1's feedback bit)
				let lp = conns.iter().position(|&x| x as usize == sensor_window + chain[k - 1])?;
				let lmask = 1u64 << (sbpn - 1 - lp);
				for base in self.split_visited_bases(c, sif, sil, &[tp, lp, sp]) {
					for tv in 0..2u64 {
						for lv in 0..2u64 {
							for sv in 0..2u64 {
								let addr = base | (tv * tmask) | (lv * lmask) | (sv * smask);
								let on = sv == 1 || (tv == 1 && lv == 1);
								self.state_memory.write_cell(c, addr, if on { 3 } else { 1 }, true);
							}
						}
					}
				}
			}
		}
		Some(chain)
	}

	/// Install a BIDIRECTIONAL integral (up/down thermometer counter) on the
	/// (up, dn) trigger pair. Verifies state neurons 0..n_levels are wired as the
	/// bidirectional chain — level k observes [up, dn, lower, self, upper] where
	/// lower = up (k=0) or level k-1's self, upper = level k+1's self (k<top); the
	/// TOP level's upper must be wired to a constant-0 bit (the GA/connectivity
	/// provisions this, Phase 5) so the top always "sees nothing above" and can
	/// unwind. Writes the truth table:
	///   on = 0  if (dn AND self AND NOT upper)   # decrement: I'm the top -> unwind
	///        1  elif self                          # hold
	///        1  elif (up AND lower)                # increment
	///        0  else
	/// Returns the neurons used, or None if the chain isn't bidirectional-wired.
	/// Verify neurons 0..n_levels are wired as the bidirectional up/down chain
	/// ([up, dn, lower, self, (upper)] at positions 0..4). Shared by the CPU planter
	/// and the GPU port. (Sbpn≥5, level range, and unused gating included.)
	pub(crate) fn plant_counter_bidir_ok(&self, up: usize, dn: usize, n_levels: usize, used: &[bool]) -> bool {
		let frame_bits = self.num_features * self.bits_per_feature;
		let sensor_window = self.input_window_k * frame_bits;
		let sbpn = self.state_bits_per_neuron;
		if sbpn < 5 || n_levels == 0 || n_levels > self.state_neurons {
			return false;
		}
		if (0..n_levels).any(|k| used.get(k).copied().unwrap_or(false)) {
			return false;
		}
		for k in 0..n_levels {
			let conns = &self.state_connections[k * sbpn..(k + 1) * sbpn];
			let lower = if k == 0 { up } else { sensor_window + (k - 1) };
			let self_k = sensor_window + k;
			if conns[0] as usize != up
				|| conns[1] as usize != dn
				|| conns[2] as usize != lower
				|| conns[3] as usize != self_k
			{
				return false;
			}
			// non-top: upper must be the level above; top: trust it's a const-0 bit
			if k + 1 < n_levels && conns[4] as usize != sensor_window + (k + 1) {
				return false;
			}
		}
		true
	}

	fn split_install_counter_bidir(
		&self,
		up: usize,
		dn: usize,
		n_levels: usize,
		used: &[bool],
	) -> Option<Vec<usize>> {
		let sbpn = self.state_bits_per_neuron;
		if !self.plant_counter_bidir_ok(up, dn, n_levels, used) {
			return None;
		}
		for k in 0..n_levels {
			for a in 0..(1usize << sbpn) {
				let up_b = (a >> (sbpn - 1)) & 1; // pos 0
				let dn_b = (a >> (sbpn - 2)) & 1; // pos 1
				let lower_b = (a >> (sbpn - 3)) & 1; // pos 2
				let self_b = (a >> (sbpn - 4)) & 1; // pos 3
				let upper_b = (a >> (sbpn - 5)) & 1; // pos 4
				let on = if dn_b == 1 && self_b == 1 && upper_b == 0 {
					false // decrement (top active, err_dn)
				} else if self_b == 1 {
					true // hold
				} else {
					up_b == 1 && lower_b == 1 // increment
				};
				self.state_memory.write_cell(k, a as u64, if on { 3 } else { 1 }, true);
			}
		}
		Some((0..n_levels).collect())
	}

	/// Roll the episodes and commit the OUTPUT layer toward the PID PWM at each
	/// step's (now state-aware) output address — the mechanical half of a split
	/// round. Reuses the same nudge primitive as bptt's output commit. Returns
	/// the number of output cells written.
	fn split_retrain_output(
		&mut self,
		gyros: &[Vec<[f32; 3]>],
		accels: &[Vec<[f32; 3]>],
		targets: &[Vec<[f32; 3]>],
		pid_pwms: &[Vec<[f32; 4]>],
		selective: bool,
	) -> usize {
		let bpf = self.bits_per_feature;
		let frame_bits = self.num_features * bpf;
		let sensor_window = self.input_window_k * frame_bits;
		let state_bits_in = self.state_neurons;
		let state_input_len = sensor_window + state_bits_in;
		let out_input_len = frame_bits + state_bits_in;
		let levels = self.levels_per_motor;
		let obpn = self.output_bits_per_neuron;
		let num_out = self.num_motors * levels;
		let mut writes = 0usize;
		// Diagnostic (Phase 6c probe): skip ALL output retrain → test whether the
		// PLANTED STATE alone (output left at hover-hold) preserves stability, i.e.
		// whether the destabilizer is output imitation vs the state recurrence.
		if std::env::var("WNN_SPLIT_SKIP_OUTPUT").map(|s| s == "1").unwrap_or(false) {
			return 0;
		}

		for ep in 0..gyros.len() {
			let iy = self.pending_init_yaws.get(ep).copied().unwrap_or(0.0); // yaw-anchor seed (0.0 ⇒ legacy)
			self.reset(iy);
			for t in 0..gyros[ep].len() {
				let feats = self.compute_features(gyros[ep][t], accels[ep][t], targets[ep][t]);
				// Action-repeat hold: accumulators tick; no ring push / forward /
				// output commit (deploy reads NO addresses on hold steps — training
				// them would write cells deploy never reads). prev_state unchanged.
				if self.action_repeat > 1 {
					let hold = self.step_counter % self.action_repeat != 0;
					self.step_counter += 1;
					if hold {
						continue;
					}
				}
				let mut frame = vec![false; frame_bits];
				for f in 0..self.num_features {
					let row = f * bpf;
					for b in 0..bpf {
						frame[row + b] = feats[f] >= self.thresholds[row + b];
					}
				}
				if self.input_history.len() == self.input_window_k {
					self.input_history.pop_front();
				}
				self.input_history.push_back(frame.clone());

				let mut in_state = vec![false; state_input_len];
				let pad = self.input_window_k - self.input_history.len();
				for (i, fr) in self.input_history.iter().enumerate() {
					let slot = (pad + i) * frame_bits;
					in_state[slot..slot + frame_bits].copy_from_slice(fr);
				}
				for (n, &v) in self.prev_state.iter().enumerate() {
					in_state[sensor_window + n] = (v >> 1) & 1 != 0;
				}
				let mut new_state = vec![0u8; self.state_neurons];
				for n in 0..self.state_neurons {
					let cs = n * self.state_bits_per_neuron;
					let ce = cs + self.state_bits_per_neuron;
					let addr = compute_address_sparse(&in_state, &self.state_connections[cs..ce], self.state_bits_per_neuron);
					new_state[n] = self.state_memory.read_cell(n, addr);
				}
				let mut in_out = vec![false; out_input_len];
				in_out[0..frame_bits].copy_from_slice(&frame);
				for (n, &v) in new_state.iter().enumerate() {
					in_out[frame_bits + n] = (v >> 1) & 1 != 0;
				}
				// SELECTIVE retrain (Phase 6c): when on, only deviate the output where
				// the recurrent state is ACTIVE (some state bit set) — i.e. where a
				// planted distinction is doing something. At state=0 (the hover-hold
				// default), leave the output's empty cells alone, so the stable
				// constant-hover the untrained seed gives is PRESERVED instead of
				// overwritten by destabilizing wholesale PID imitation. The state's
				// targeted corrections then ADD to hover rather than replace it.
				let state_active = new_state.iter().any(|&v| (v >> 1) & 1 != 0);
				if selective && !state_active {
					self.prev_state = new_state;
					continue;
				}
				// output commit toward PID (mirrors bptt_train_window step d)
				for n in 0..num_out {
					let motor = n / levels;
					let level_idx = n % levels;
					let p = self.output_decode_target(motor, pid_pwms[ep][t][motor]);
					let target_true = (p * levels as f32) as usize > level_idx;
					let cs = n * obpn;
					let ce = cs + obpn;
					let addr = compute_address_sparse(&in_out, &self.output_connections[cs..ce], obpn);
					let cur = self.output_memory.read_cell(n, addr);
					let nv = crate::controller_training::nudge_toward_pub(cur, target_true);
					if nv != cur {
						self.output_memory.write_cell(n, addr, nv, true);
						writes += 1;
					}
				}
				self.prev_state = new_state;
			}
		}
		writes
	}

	/// Resolve ONE conflict: TYPE-1 discriminative walk (plant a latch) else
	/// TYPE-2 accumulator (install a counter), honoring the `used` neuron guard.
	/// Returns (mode, neurons_planted): mode 0 none / 1 latch / 2 counter. Writes
	/// state cells; the caller marks the returned neurons used and retrains output.
	#[allow(clippy::too_many_arguments)]
	fn split_resolve_conflict(
		&self,
		instances: &[usize],
		pwms: &[[f32; 4]],
		ep_of: &[usize],
		step_of: &[usize],
		ep_start: &[usize],
		sif: &[bool],
		sil: usize,
		candidate_bits: &[usize],
		clean_gain: f32,
		accum_corr: f32,
		used: &[bool],
	) -> (i64, Vec<usize>) {
		// Perf caps: bound the separator statistics to a sample of instances and a
		// recent lookback (see SPLIT_INST_CAP / SPLIT_LAG_CAP).
		let sampled = subsample_instances(instances, SPLIT_INST_CAP);
		let instances = &sampled[..];
		let labels = crate::controller_split::label_high_low(instances, pwms);
		let max_lag = instances
			.iter()
			.map(|&i| step_of[i])
			.min()
			.unwrap_or(0)
			.min(SPLIT_LAG_CAP);
		// TYPE-1
		let sep = crate::controller_split::discriminative_walk(
			instances, &labels, ep_of, step_of, ep_start, sif, sil, candidate_bits, max_lag,
		);
		if let Some(s) = sep.filter(|s| s.gain >= clean_gain) {
			if let Some(n) = self.split_plant_latch(s.bit, s.high_on, used, sif, sil) {
				return (1, vec![n]);
			}
		}
		// TYPE-2: disagreeing motor → window-count correlation
		let mut best_m = 0usize;
		let mut best_s = -1.0f32;
		for m in 0..self.num_motors {
			let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
			for &i in instances {
				lo = lo.min(pwms[i][m]);
				hi = hi.max(pwms[i][m]);
			}
			if hi - lo > best_s {
				best_s = hi - lo;
				best_m = m;
			}
		}
		let scalar: Vec<f32> = instances.iter().map(|&i| pwms[i][best_m]).collect();
		// BIDIRECTIONAL (mode 3) first when the chain can hold an up/down counter
		if self.state_bits_per_neuron >= 5 {
			let bi = crate::controller_split::detect_accumulator_bidir(
				instances, &scalar, ep_of, step_of, ep_start, sif, sil, candidate_bits, max_lag,
			);
			if let Some(b) = bi.filter(|b| b.corr >= accum_corr) {
				if let Some(neurons) = self.split_install_counter_bidir(b.up, b.dn, self.state_neurons, used) {
					return (3, neurons);
				}
			}
		}
		// INCREMENT-only (mode 2)
		let accum = crate::controller_split::detect_accumulator(
			instances, &scalar, ep_of, step_of, ep_start, sif, sil, candidate_bits, max_lag,
		);
		if let Some(a) = accum.filter(|a| a.corr >= accum_corr) {
			if let Some(neurons) = self.split_install_counter(a.bit, self.state_neurons, used, sif, sil) {
				return (2, neurons);
			}
		}
		(0, vec![])
	}
}

// =============================================================================
// Decoders + reward
// =============================================================================

/// Strategy-5 QSR-weighted decode (paper #1 primary decoder).
///
/// Reads `output_cells` shaped (num_motors * levels_per_motor,), where each
/// entry is a raw QSR cell value in [0, 3]. Returns one bucket index per
/// motor in [0, levels_per_motor - 1].
///
/// ZYX (Tait-Bryan) yaw angle (rad) extracted from a (w,x,y,z) unit quaternion —
/// the inverse of `_euler_to_quat_xyz` for the yaw component. Seeds the yaw-anchor:
/// the harness derives each episode's true initial yaw from q0 and passes it to
/// `WnnController.reset(init_yaw=…)`. The Metal twin (`yaw_from_quat` in
/// controller_rollout.metal) mirrors this bit-for-bit so the GPU score/train/record
/// kernels seed identically from their own q0.
/// Plain Rust entry (callable from the Rust dagger/trainer, where there is no
/// Python). The #[pyfunction] yaw_from_quat below just wraps this.
pub(crate) fn yaw_from_quat_rs(q: [f32; 4]) -> f32 {
	let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
	(2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y * y + z * z))
}

#[pyfunction]
pub fn yaw_from_quat(q: [f32; 4]) -> f32 { yaw_from_quat_rs(q) }

#[pyfunction]
#[pyo3(signature = (output_cells, levels_per_motor = 256, num_motors = 4))]
pub fn strategy_5_qsr_weighted(
	output_cells: Vec<u8>,
	levels_per_motor: usize,
	num_motors: usize,
) -> PyResult<Vec<u32>> {
	if output_cells.len() != num_motors * levels_per_motor {
		return Err(pyo3::exceptions::PyValueError::new_err(format!(
			"output_cells length {} does not match num_motors * levels_per_motor = {}",
			output_cells.len(),
			num_motors * levels_per_motor
		)));
	}
	let mut buckets = Vec::with_capacity(num_motors);
	for m in 0..num_motors {
		let mut sum: f32 = 0.0;
		let start = m * levels_per_motor;
		for &cell in &output_cells[start..start + levels_per_motor] {
			let idx = (cell & 0x3) as usize;
			sum += QSR_WEIGHTS[idx];
		}
		let bucket = sum.round() as i64;
		let max_bucket = (levels_per_motor - 1) as i64;
		let clamped = bucket.clamp(0, max_bucket) as u32;
		buckets.push(clamped);
	}
	Ok(buckets)
}

/// Strategy-1 count-TRUE decode (ablation baseline). Counts cells with
/// QSR value ≥ 2 (WEAK_TRUE or TRUE) per motor.
#[pyfunction]
#[pyo3(signature = (output_cells, levels_per_motor = 256, num_motors = 4))]
pub fn strategy_1_count_true(
	output_cells: Vec<u8>,
	levels_per_motor: usize,
	num_motors: usize,
) -> PyResult<Vec<u32>> {
	if output_cells.len() != num_motors * levels_per_motor {
		return Err(pyo3::exceptions::PyValueError::new_err(format!(
			"output_cells length {} does not match num_motors * levels_per_motor = {}",
			output_cells.len(),
			num_motors * levels_per_motor
		)));
	}
	let mut buckets = Vec::with_capacity(num_motors);
	for m in 0..num_motors {
		let start = m * levels_per_motor;
		let count: u32 = output_cells[start..start + levels_per_motor]
			.iter()
			.filter(|&&c| (c & 0x3) >= 2)
			.count() as u32;
		buckets.push(count);
	}
	Ok(buckets)
}

/// Count thermometer-pattern violations across all motors. A violation is
/// a 0→1 transition AFTER any 1 has been observed (which would break the
/// thermometer monotonicity assumption that 1s come first, then 0s).
///
/// Used as a soft regularizer in the training reward:
///   reward += -lambda_mono * monotonicity_violations(output)
#[pyfunction]
#[pyo3(signature = (output_cells, levels_per_motor = 256, num_motors = 4))]
pub fn monotonicity_violations(
	output_cells: Vec<u8>,
	levels_per_motor: usize,
	num_motors: usize,
) -> PyResult<u32> {
	monotonicity_violations_core(&output_cells, levels_per_motor, num_motors)
		.map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Plain-Rust twin of `monotonicity_violations` (house pattern: String errors
/// keep cargo tests — the GPU rollout parity oracle — off the libpython link path).
pub(crate) fn monotonicity_violations_core(
	output_cells: &[u8],
	levels_per_motor: usize,
	num_motors: usize,
) -> Result<u32, String> {
	if output_cells.len() != num_motors * levels_per_motor {
		return Err(format!(
			"output_cells length {} does not match num_motors * levels_per_motor = {}",
			output_cells.len(),
			num_motors * levels_per_motor
		));
	}
	let mut violations: u32 = 0;
	for m in 0..num_motors {
		let start = m * levels_per_motor;
		let cells = &output_cells[start..start + levels_per_motor];
		let mut seen_one = false;
		let mut prev_was_zero = false;
		for &c in cells {
			let bit = (c & 0x3) >= 2;
			if bit {
				if prev_was_zero && seen_one {
					violations += 1;
				}
				seen_one = true;
				prev_was_zero = false;
			} else {
				prev_was_zero = true;
			}
		}
	}
	Ok(violations)
}

/// Compute the scalar reward for a single timestep.
///
/// Args:
///   attitude_error_rad: geodesic angle between current and target attitude (rad).
///   motor_command_jerk: sum of squared deltas between this and previous PWM.
///   mono_violations:    output from monotonicity_violations()
///   lambda_smooth:      weight on the jerk term (default 0.0 → off)
///   lambda_mono:        weight on the monotonicity penalty (default 0.0 → off)
///
/// Returns:
///   reward = -attitude_error²  - λ_smooth × jerk  - λ_mono × violations
#[pyfunction]
#[pyo3(signature = (attitude_error_rad, motor_command_jerk = 0.0, mono_violations = 0, lambda_smooth = 0.0, lambda_mono = 0.0))]
pub fn compute_reward(
	attitude_error_rad: f32,
	motor_command_jerk: f32,
	mono_violations: u32,
	lambda_smooth: f32,
	lambda_mono: f32,
) -> f32 {
	-(attitude_error_rad * attitude_error_rad)
		- lambda_smooth * motor_command_jerk
		- lambda_mono * (mono_violations as f32)
}

// =============================================================================
// AttitudePidRs — Rust port of src/wnn/control/pid.py (AttitudePID).
//
// The PID is the imitation teacher in the DAGGER training loop. Porting it to
// Rust lets the whole rollout (sim + controller + EDRA + PID) run with ZERO
// per-step Python<->Rust crossings — the prerequisite for GA-scale controller
// training (project_drone_controller_paper1.md). pid.py stays as the spec; a
// parity test (tests/test_pid_parity.py) checks Rust == Python step-for-step.
//
// Internal math is f64 to match Python floats; inputs/outputs are f32 to match
// AttitudeSim. Gains/mixing default to pid.py's hand-tuned AttitudePIDConfig.
// =============================================================================

#[inline]
pub(crate) fn clamp_f64(v: f64, lo: f64, hi: f64) -> f64 {
	if v < lo { lo } else if v > hi { hi } else { v }
}

#[inline]
pub(crate) fn wrap_angle_f64(a: f64) -> f64 {
	let mut x = a;
	while x > std::f64::consts::PI { x -= 2.0 * std::f64::consts::PI; }
	while x <= -std::f64::consts::PI { x += 2.0 * std::f64::consts::PI; }
	x
}

/// '+' quad mixing (roll/pitch/yaw normalized controls → 4 motor PWMs), motors
/// clamped [0,1]. Bit-identical to AttitudePidRs::step_rs mixing and optimal.py::
/// mix_to_motors. Shared by the Rust LQR/MPC teachers (optimal.rs).
#[inline]
pub(crate) fn mix_to_motors_f64(hover: f64, u_roll: f64, u_pitch: f64, u_yaw: f64) -> [f64; 4] {
	[
		clamp_f64(hover - u_pitch + u_yaw, 0.0, 1.0),  // M0 front
		clamp_f64(hover - u_roll  - u_yaw, 0.0, 1.0),  // M1 right
		clamp_f64(hover + u_pitch + u_yaw, 0.0, 1.0),  // M2 rear
		clamp_f64(hover + u_roll  - u_yaw, 0.0, 1.0),  // M3 left
	]
}

/// Calibrate normalized-control → angular-acceleration gains b=[b_roll,b_pitch,b_yaw]
/// by stepping a clean AttitudeSim once per axis from rest (mirrors optimal.py::
/// calibrate_control_gains). The Rust LQR/MPC teachers use this so their linear
/// plant model matches the EXACT sim they will control (sim params passed in).
pub(crate) fn calibrate_control_gains_rs(
	dt: f32, arm_length: f32, k_thrust: f32, k_drag: f32,
	inertia: [f32; 3], gravity: f32, hover: f64, u_probe: f64,
) -> [f64; 3] {
	let mut b = [0.0f64; 3];
	for axis in 0..3 {
		let mut sim = AttitudeSim::new(dt, arm_length, k_thrust, k_drag, inertia, gravity);
		sim.reset(Some([1.0, 0.0, 0.0, 0.0]), Some([0.0, 0.0, 0.0]));
		let mut u = [0.0f64; 3];
		u[axis] = u_probe;
		let m = mix_to_motors_f64(hover, u[0], u[1], u[2]);
		sim.step([m[0] as f32, m[1] as f32, m[2] as f32, m[3] as f32]);
		let omega = sim.omega;                       // [p,q,r] after one step from rest
		b[axis] = (omega[axis] as f64 / dt as f64) / u_probe;   // ω̇ / u
	}
	b
}

/// Body-to-world unit quaternion (w, x, y, z) -> (roll, pitch, yaw) radians.
/// Matches pid.py::_quat_to_euler exactly (Z-Y-X Tait-Bryan).
pub(crate) fn quat_to_euler_f64(q: [f32; 4]) -> (f64, f64, f64) {
	let w = q[0] as f64;
	let x = q[1] as f64;
	let y = q[2] as f64;
	let z = q[3] as f64;
	let sinr_cosp = 2.0 * (w * x + y * z);
	let cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
	let roll = sinr_cosp.atan2(cosr_cosp);
	let sinp = 2.0 * (w * y - z * x);
	let pitch = if sinp >= 1.0 {
		std::f64::consts::FRAC_PI_2
	} else if sinp <= -1.0 {
		-std::f64::consts::FRAC_PI_2
	} else {
		sinp.asin()
	};
	let siny_cosp = 2.0 * (w * z + x * y);
	let cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
	let yaw = siny_cosp.atan2(cosy_cosp);
	(roll, pitch, yaw)
}

/// 3-axis attitude PID teacher. See pid.py for the design rationale (D-term on
/// gyro for damping, anti-windup I-clamp, '+' quad mixing).
#[pyclass]
pub struct AttitudePidRs {
	// Per-axis gains (roll/pitch share by symmetry; yaw weaker).
	kp_rp: f64, ki_rp: f64, kd_rp: f64, i_clamp_rp: f64,
	kp_yaw: f64, ki_yaw: f64, kd_yaw: f64, i_clamp_yaw: f64,
	hover_throttle: f64,
	max_axis_authority: f64,
	dt: f64,
	// Mutable I-term accumulators.
	integral_roll: f64,
	integral_pitch: f64,
	integral_yaw: f64,
}

#[pymethods]
impl AttitudePidRs {
	#[new]
	#[pyo3(signature = (
		kp_rp = 1.2, ki_rp = 0.05, kd_rp = 0.30, i_clamp_rp = 0.5,
		kp_yaw = 0.6, ki_yaw = 0.02, kd_yaw = 0.20, i_clamp_yaw = 0.5,
		hover_throttle = 0.5, max_axis_authority = 0.4, dt = 0.001
	))]
	pub fn new(
		kp_rp: f64, ki_rp: f64, kd_rp: f64, i_clamp_rp: f64,
		kp_yaw: f64, ki_yaw: f64, kd_yaw: f64, i_clamp_yaw: f64,
		hover_throttle: f64, max_axis_authority: f64, dt: f64,
	) -> Self {
		AttitudePidRs {
			kp_rp, ki_rp, kd_rp, i_clamp_rp,
			kp_yaw, ki_yaw, kd_yaw, i_clamp_yaw,
			hover_throttle, max_axis_authority, dt,
			integral_roll: 0.0, integral_pitch: 0.0, integral_yaw: 0.0,
		}
	}

	/// Zero the I-term accumulators (call between episodes).
	pub fn reset(&mut self) {
		self.integral_roll = 0.0;
		self.integral_pitch = 0.0;
		self.integral_yaw = 0.0;
	}

	/// One PID cycle. Returns 4 motor PWMs in [0, 1] for the '+' quad layout
	/// (M0 front, M1 right, M2 rear, M3 left) matching AttitudeSim.
	fn step(
		&mut self,
		q: [f32; 4],
		gyro: [f32; 3],
		target_rpy: [f32; 3],
	) -> [f32; 4] {
		let pwm = self.step_rs(q, gyro, target_rpy);
		[pwm[0] as f32, pwm[1] as f32, pwm[2] as f32, pwm[3] as f32]
	}
}

impl AttitudePidRs {
	/// The teacher's current I-term accumulators (roll, pitch, yaw), each in
	/// [-i_clamp, i_clamp]. Used by option A to give the recurrent STATE layer a
	/// DIRECT integral target during BPTT training (project_controller_stability_
	/// diagnosis): the state learns to encode the PID integral so the policy
	/// becomes history-aware instead of memoryless-proportional.
	pub fn integrals(&self) -> [f32; 3] {
		[self.integral_roll as f32, self.integral_pitch as f32, self.integral_yaw as f32]
	}
	/// Clamp magnitudes for normalizing the integral to [-1, 1] (roll/pitch share,
	/// yaw separate). i_clamp_rp/yaw default 0.5.
	pub fn i_clamps(&self) -> [f32; 3] {
		[self.i_clamp_rp as f32, self.i_clamp_rp as f32, self.i_clamp_yaw as f32]
	}

	/// Native f64 step used by the in-crate DAGGER loop (no PyO3 conversion).
	pub fn step_rs(&mut self, q: [f32; 4], gyro: [f32; 3], target_rpy: [f32; 3]) -> [f64; 4] {
		let (roll, pitch, yaw) = quat_to_euler_f64(q);
		let err_roll = wrap_angle_f64(target_rpy[0] as f64 - roll);
		let err_pitch = wrap_angle_f64(target_rpy[1] as f64 - pitch);
		let err_yaw = wrap_angle_f64(target_rpy[2] as f64 - yaw);

		self.integral_roll = clamp_f64(
			self.integral_roll + err_roll * self.dt, -self.i_clamp_rp, self.i_clamp_rp);
		self.integral_pitch = clamp_f64(
			self.integral_pitch + err_pitch * self.dt, -self.i_clamp_rp, self.i_clamp_rp);
		self.integral_yaw = clamp_f64(
			self.integral_yaw + err_yaw * self.dt, -self.i_clamp_yaw, self.i_clamp_yaw);

		let gx = gyro[0] as f64;
		let gy = gyro[1] as f64;
		let gz = gyro[2] as f64;

		let mut u_roll = self.kp_rp * err_roll + self.ki_rp * self.integral_roll - self.kd_rp * gx;
		let mut u_pitch = self.kp_rp * err_pitch + self.ki_rp * self.integral_pitch - self.kd_rp * gy;
		let mut u_yaw = self.kp_yaw * err_yaw + self.ki_yaw * self.integral_yaw - self.kd_yaw * gz;

		let a = self.max_axis_authority;
		u_roll = clamp_f64(u_roll, -a, a);
		u_pitch = clamp_f64(u_pitch, -a, a);
		u_yaw = clamp_f64(u_yaw, -a, a);

		let base = self.hover_throttle;
		[
			clamp_f64(base - u_pitch + u_yaw, 0.0, 1.0),  // M0 front
			clamp_f64(base - u_roll  - u_yaw, 0.0, 1.0),  // M1 right
			clamp_f64(base + u_pitch + u_yaw, 0.0, 1.0),  // M2 rear
			clamp_f64(base + u_roll  - u_yaw, 0.0, 1.0),  // M3 left
		]
	}
}

// =============================================================================
// W2 disturbance unit tests (cargo test -p ram_controller).
// =============================================================================

#[cfg(test)]
mod dist_tests {
	use super::*;

	fn sim() -> AttitudeSim {
		AttitudeSim::new(0.001, 0.075, 2.4, 0.05, [0.0023, 0.0023, 0.0046], 9.81)
	}

	/// Run N steps under a fixed slightly-asymmetric PWM; return final state.
	fn run(s: &mut AttitudeSim, n: usize, pwm: [f32; 4]) -> ([f32; 4], [f32; 3]) {
		for _ in 0..n {
			let _imu = s.read_imu();  // exercise the IMU path each step too
			s.step(pwm);
		}
		(s.quaternion(), s.omega)
	}

	const PWM: [f32; 4] = [0.52, 0.48, 0.50, 0.51];

	/// (a) OFF is deterministic AND set+clear_disturbance returns to the exact
	/// clean trajectory — the None branch adds nothing.
	#[test]
	fn off_matches_clean_golden() {
		// Golden: a sim that never touched the disturbance API.
		let mut a = sim();
		let (qa, oa) = run(&mut a, 500, PWM);
		// Determinism: a second clean sim reproduces it exactly.
		let mut b = sim();
		let (qb, ob) = run(&mut b, 500, PWM);
		assert_eq!(qa, qb);
		assert_eq!(oa, ob);
		// set_disturbance → clear_disturbance → reset ⇒ bit-identical to golden.
		let mut c = sim();
		c.set_disturbance([0.01, 0.0, 0.0], 0.05, 0.1, [1.03, 0.97, 1.0, 1.0],
		                  0.02, 0.001, 0.1, 42);
		c.clear_disturbance();
		c.reset(None, None);
		let (qc, oc) = run(&mut c, 500, PWM);
		assert_eq!(qa, qc);
		assert_eq!(oa, oc);
	}

	/// (a') All-neutral disturbance fields (bias 0, sigmas 0, asym 1.0) also
	/// reproduce the clean trajectory value-for-value (×1.0 and +0.0 are exact).
	#[test]
	fn neutral_disturbance_matches_clean() {
		let mut a = sim();
		let (qa, oa) = run(&mut a, 500, PWM);
		let mut b = sim();
		b.set_disturbance([0.0; 3], 0.0, 0.1, [1.0; 4], 0.0, 0.0, 0.0, 7);
		let (qb, ob) = run(&mut b, 500, PWM);
		assert_eq!(qa, qb);
		assert_eq!(oa, ob);
	}

	/// (b) D1-only: a constant torque bias rotates a level, uncontrolled body
	/// (clean sim stays exactly level under symmetric zero torque).
	#[test]
	fn d1_bias_changes_steady_state() {
		let hover = [0.5, 0.5, 0.5, 0.5];
		let mut clean = sim();
		run(&mut clean, 1000, hover);
		let clean_err = clean.attitude_error(None);
		assert!(clean_err.abs() < 1e-6, "clean level sim drifted: {clean_err}");

		let mut biased = sim();
		biased.set_disturbance([0.002, 0.0, 0.0], 0.0, 0.1, [1.0; 4], 0.0, 0.0, 0.0, 0);
		run(&mut biased, 1000, hover);
		let biased_err = biased.attitude_error(None);
		assert!(biased_err > 1e-3, "D1 bias produced no attitude change: {biased_err}");
		// Positive roll bias ⇒ positive roll rate about +x.
		assert!(biased.omega[0] > 0.0);
	}

	/// (c) Same seed ⇒ identical trajectory; different seed ⇒ different.
	#[test]
	fn seed_reproducibility() {
		let dist = |s: &mut AttitudeSim, seed: u64| {
			s.set_disturbance([0.001, 0.0, 0.0], 0.02, 0.1, [1.02, 0.98, 1.01, 0.99],
			                  0.01, 0.0005, 0.05, seed);
		};
		let mut a = sim(); dist(&mut a, 1234);
		let (qa, oa) = run(&mut a, 400, PWM);
		let mut b = sim(); dist(&mut b, 1234);
		let (qb, ob) = run(&mut b, 400, PWM);
		assert_eq!(qa, qb);
		assert_eq!(oa, ob);
		let mut c = sim(); dist(&mut c, 5678);
		let (qc, _oc) = run(&mut c, 400, PWM);
		assert_ne!(qa, qc, "different seeds produced identical trajectories");
	}

	/// (d) OU stationarity sanity: gust variance stays bounded near the
	/// stationary value sigma²·tau_c/2 (loose 3× band; 20k steps at 1 kHz).
	#[test]
	fn ou_gust_stationary_variance() {
		let sigma = 0.05_f32;
		let tau_c = 0.1_f32;
		let mut s = sim();
		s.set_disturbance([0.0; 3], sigma, tau_c, [1.0; 4], 0.0, 0.0, 0.0, 99);
		let mut sum = 0.0_f64;
		let mut sum_sq = 0.0_f64;
		let mut n = 0_usize;
		for i in 0..20_000 {
			s.step([0.5; 4]);
			if i >= 2_000 {   // skip transient
				let g = s.gust[0] as f64;
				sum += g;
				sum_sq += g * g;
				n += 1;
			}
			// Keep the sim itself sane (gust torque swings it, that's fine).
			if s.is_unstable() { s.omega = [0.0; 3]; }
		}
		let mean = sum / n as f64;
		let var = sum_sq / n as f64 - mean * mean;
		let theory = (sigma as f64) * (sigma as f64) * (tau_c as f64) / 2.0;
		assert!(var > theory / 3.0 && var < theory * 3.0,
		        "OU variance {var:.3e} outside 3x band of theory {theory:.3e}");
		assert!(mean.abs() < 10.0 * theory.sqrt(), "OU mean {mean:.3e} not near zero");
	}

	/// D4: sensor noise perturbs read_imu but is idempotent at a fixed step,
	/// and the clean sim's IMU is untouched.
	#[test]
	fn d4_imu_noise_idempotent() {
		let mut s = sim();
		s.set_disturbance([0.0; 3], 0.0, 0.1, [1.0; 4], 0.02, 0.0, 0.1, 3);
		let (g1, a1) = s.read_imu();
		let (g2, a2) = s.read_imu();
		assert_eq!(g1, g2, "read_imu not idempotent at fixed step");
		assert_eq!(a1, a2);
		// Level, at rest: clean gyro would be exactly 0 — noise must move it.
		assert!(g1[0] != 0.0 || g1[1] != 0.0 || g1[2] != 0.0, "gyro noise absent");
		s.step([0.5; 4]);
		let (g3, _a3) = s.read_imu();
		assert_ne!(g1, g3, "noise did not advance with step_idx");
	}

	/// Episode-seed derivation is stable (regression-pins the channel-15 hash
	/// the Metal kernel mirrors).
	#[test]
	fn episode_seed_derivation_stable() {
		let a = disturbance_episode_seed(42, 0);
		let b = disturbance_episode_seed(42, 1);
		let c = disturbance_episode_seed(43, 0);
		assert_ne!(a, b);
		assert_ne!(a, c);
		assert_eq!(a, disturbance_episode_seed(42, 0));
		assert_eq!(a, dist_hash_u32(dist_seed32(42), 0, 0, DIST_CH_EP_SEED, 0) as u64);
	}
}

// =============================================================================
// Overactuated Phase-1 sim tests (step_n / geometry; the legacy step() path
// must stay bit-identical — docs/OVERACTUATED_RESIDUAL_DESIGN.md).
// =============================================================================

#[cfg(test)]
mod overactuated_sim_tests {
	use super::*;

	const ARM: f32 = 0.075;
	const KT: f32 = 2.4;
	const KD: f32 = 0.05;

	fn sim() -> AttitudeSim {
		AttitudeSim::new(0.001, ARM, KT, KD, [0.0023, 0.0023, 0.0046], 9.81)
	}

	const PWM4: [f32; 4] = [0.52, 0.48, 0.50, 0.51];

	/// Bit-identity gate: step_n with NO geometry is the legacy step().
	#[test]
	fn step_n_without_geometry_is_bitwise_legacy() {
		let mut a = sim();
		let mut b = sim();
		for _ in 0..500 {
			a.step(PWM4);
			b.step_n_core(&PWM4).unwrap();
		}
		assert_eq!(a.quaternion(), b.quaternion());
		assert_eq!(a.omega, b.omega);
	}

	/// Golden gate: the quad expressed AS a geometry tracks the legacy mixer
	/// to float-accumulation tolerance over a long rollout.
	#[test]
	fn quad_geometry_tracks_legacy_step() {
		let mut a = sim();
		let mut b = sim();
		b.set_geometry_quad_plus(ARM, KT, KD);
		for _ in 0..1000 {
			a.step(PWM4);
			b.step_n_core(&PWM4).unwrap();
		}
		let (qa, qb) = (a.quaternion(), b.quaternion());
		for i in 0..4 {
			assert!((qa[i] - qb[i]).abs() < 1e-4, "q[{i}]: {} vs {}", qa[i], qb[i]);
		}
		for i in 0..3 {
			assert!((a.omega[i] - b.omega[i]).abs() < 1e-3, "omega[{i}]");
		}
	}

	/// D1/D2-composition parity: with disturbances armed, the no-geometry
	/// step_n still matches step() bit-for-bit (same torque + noise stream).
	#[test]
	fn step_n_disturbance_composition_matches_step() {
		let mut a = sim();
		let mut b = sim();
		for s in [&mut a, &mut b] {
			s.set_disturbance([0.001, -0.002, 0.0005], 0.02, 0.1,
				[1.0, 1.0, 1.0, 1.0], 0.0, 0.0, 0.0, 1234);
		}
		for _ in 0..300 {
			a.step(PWM4);
			b.step_n_core(&PWM4).unwrap();
		}
		assert_eq!(a.quaternion(), b.quaternion());
		assert_eq!(a.omega, b.omega);
	}

	/// Octo hover balance: equal PWM on the symmetric octo-X leaves attitude
	/// level (near-zero torque), and wrong PWM count errors cleanly.
	#[test]
	fn octo_equal_pwm_stays_level() {
		let mut s = sim();
		s.set_geometry_octo_x(ARM, KT, KD);
		assert_eq!(s.num_rotors(), 8);
		assert!(s.step_n_core(&vec![0.5; 4]).is_err(), "wrong PWM count must error");
		for _ in 0..500 {
			s.step_n_core(&vec![0.5; 8]).unwrap();
		}
		let err = s.attitude_error(None);
		assert!(err < 1e-3, "octo hover drifted: {err} rad");
	}

	/// Perturbation: zero-perturb is a no-op; a real tilt error changes the
	/// trajectory (the mismatch signal the residual will learn) and a
	/// per-rotor asymmetry does too.
	#[test]
	fn perturbed_geometry_changes_dynamics() {
		let run = |tilt: Vec<f32>, asym: Option<Vec<f32>>| {
			let mut s = sim();
			s.set_geometry_octo_x(ARM, KT, KD);
			if !tilt.is_empty() {
				s.perturb_geometry_core(tilt, vec![]).unwrap();
			}
			s.set_rotor_asym_core(asym).unwrap();
			for _ in 0..500 {
				s.step_n_core(&vec![0.5; 8]).unwrap();
			}
			s.attitude_error(None)
		};
		let clean = run(vec![], None);
		let zero_tilt = run(vec![0.0; 8], None);
		assert!((clean - zero_tilt).abs() < 1e-7, "zero perturb must be a no-op");
		let tilted = run(vec![3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], None);
		assert!(tilted > clean + 1e-3, "3-deg tilt error must disturb attitude: {tilted}");
		let weak_motor = run(vec![], Some(vec![0.85, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]));
		assert!(weak_motor > clean + 1e-3, "weak rotor must disturb attitude: {weak_motor}");
	}
}
