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

use pyo3::prelude::*;

// Strategy-5 QSR weight lookup table. Index by raw cell value (0..3).
// FALSE=0=0.0, WEAK_FALSE=1=0.25, WEAK_TRUE=2=0.75, TRUE=3=1.0.
const QSR_WEIGHTS: [f32; 4] = [0.0, 0.25, 0.75, 1.0];

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
	fn new(
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
		}
	}

	/// Reset the simulator. Optional initial quaternion (defaults to identity)
	/// and initial angular velocity (defaults to zero).
	#[pyo3(signature = (q = None, omega = None))]
	fn reset(&mut self, q: Option<[f32; 4]>, omega: Option<[f32; 3]>) {
		self.q = q_normalize(q.unwrap_or([1.0, 0.0, 0.0, 0.0]));
		self.omega = omega.unwrap_or([0.0, 0.0, 0.0]);
		self.t = 0.0;
	}

	/// Advance one timestep under the given 4-motor PWM (each clipped to [0, 1]).
	/// Uses RK4 integration of Euler's rotational equation + quaternion update.
	fn step(&mut self, motor_pwm: [f32; 4]) {
		let pwm = [
			motor_pwm[0].clamp(0.0, 1.0),
			motor_pwm[1].clamp(0.0, 1.0),
			motor_pwm[2].clamp(0.0, 1.0),
			motor_pwm[3].clamp(0.0, 1.0),
		];
		let torque = self.body_torque(pwm);
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
	}

	/// Read the simulated IMU: (gyro_xyz, accel_xyz) in body frame.
	///   gyro  = body-frame angular velocity (rad/s)
	///   accel = body-frame specific force (m/s²) — the negative of gravity
	///           rotated into body frame. At rest with q=identity, this reads
	///           (0, 0, +g) (the support force pushing UP through the IMU).
	fn read_imu(&self) -> ([f32; 3], [f32; 3]) {
		let gyro = self.omega;
		// gravity in WORLD frame points DOWN: (0, 0, -g)
		let gravity_world = [0.0, 0.0, -self.gravity];
		// rotate to body frame; specific force = -gravity_body (support force)
		let gravity_body = rotate_world_to_body(self.q, gravity_world);
		let accel = [-gravity_body[0], -gravity_body[1], -gravity_body[2]];
		(gyro, accel)
	}

	/// Geodesic angle (rad) between current attitude and target attitude.
	/// Target defaults to identity (level). Uses 2·acos(|q·t|) on the
	/// quaternion dot product (the standard geodesic metric on SO(3)).
	#[pyo3(signature = (target = None))]
	fn attitude_error(&self, target: Option<[f32; 4]>) -> f32 {
		let t = q_normalize(target.unwrap_or([1.0, 0.0, 0.0, 0.0]));
		let dot = self.q[0] * t[0] + self.q[1] * t[1] + self.q[2] * t[2] + self.q[3] * t[3];
		// Clamp for numerical safety; acos domain is [-1, 1].
		let dot_abs = dot.abs().min(1.0);
		2.0 * dot_abs.acos()
	}

	/// True if the simulator state has diverged (omega above safety threshold
	/// or NaN in state).
	fn is_unstable(&self) -> bool {
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
	fn quaternion(&self) -> [f32; 4] {
		self.q
	}

	#[getter]
	fn angular_velocity(&self) -> [f32; 3] {
		self.omega
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

/// Stateful WNN controller wrapper. Holds the recurrent state buffer +
/// input history window across step() calls.
///
/// The actual network forward pass + thermometer encoding will plug into
/// the existing RAM neuron primitives (Memory / RAMRecurrentNetwork). For
/// the stub, step() returns mid-throttle on every motor (a baseline that
/// won't cause the sim to immediately destabilize).
#[pyclass]
pub struct WnnController {
	num_motors: usize,
	levels_per_motor: usize,
	state_bits: usize,
	input_window_k: usize,
	// TODO: hold a reference to a trained RAMRecurrentNetwork (or its
	// serialized form). For the stub we don't need it.
}

#[pymethods]
impl WnnController {
	#[new]
	#[pyo3(signature = (num_motors = 4, levels_per_motor = 256, state_bits = 200, input_window_k = 4))]
	fn new(num_motors: usize, levels_per_motor: usize, state_bits: usize, input_window_k: usize) -> Self {
		Self { num_motors, levels_per_motor, state_bits, input_window_k }
	}

	fn reset(&mut self) {
		// TODO: zero the recurrent state buffer + clear input history window
	}

	/// Run one controller cycle. Returns 4 PWM commands in [0, 1].
	/// TODO: implement thermometer encode → forward → Strategy 5 decode.
	fn step(&mut self,
	        _gyro: [f32; 3],
	        _accel: [f32; 3],
	        _target_attitude: [f32; 3]) -> Vec<f32> {
		// Stub: emit mid-throttle on every motor. Safe default.
		vec![0.5; self.num_motors]
	}

	#[getter]
	fn num_motors(&self) -> usize { self.num_motors }
	#[getter]
	fn levels_per_motor(&self) -> usize { self.levels_per_motor }
	#[getter]
	fn state_bits(&self) -> usize { self.state_bits }
	#[getter]
	fn input_window_k(&self) -> usize { self.input_window_k }
}

// =============================================================================
// Decoders + reward
// =============================================================================

/// Strategy-5 QSR-weighted decode (paper #1 primary decoder).
///
/// Reads `output_cells` shaped (num_motors * levels_per_motor,), where each
/// entry is a raw QSR cell value in [0, 3]. Returns one bucket index per
/// motor in [0, levels_per_motor - 1].
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
	if output_cells.len() != num_motors * levels_per_motor {
		return Err(pyo3::exceptions::PyValueError::new_err(format!(
			"output_cells length {} does not match num_motors * levels_per_motor = {}",
			output_cells.len(),
			num_motors * levels_per_motor
		)));
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
