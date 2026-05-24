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

use crate::neuron_memory::compute_address_sparse;
use crate::sparse_memory::SparseLayerMemory;
use crate::controller_training::{solve_partial_connectivity_qsr, nudge_toward_value};

// Strategy-5 QSR weight lookup table. Index by raw cell value (0..3).
// FALSE=0=0.0, WEAK_FALSE=1=0.25, WEAK_TRUE=2=0.75, TRUE=3=1.0.
const QSR_WEIGHTS: [f32; 4] = [0.0, 0.25, 0.75, 1.0];

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
	))]
	#[allow(clippy::too_many_arguments)]
	fn new(
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
	) -> PyResult<Self> {
		let expected_thresholds = NUM_FEATURES * bits_per_feature;
		if thresholds.len() != expected_thresholds {
			return Err(pyo3::exceptions::PyValueError::new_err(format!(
				"thresholds length {} != NUM_FEATURES * bits_per_feature = {}",
				thresholds.len(), expected_thresholds
			)));
		}
		let expected_state_conn = state_neurons * state_bits_per_neuron;
		if state_connections.len() != expected_state_conn {
			return Err(pyo3::exceptions::PyValueError::new_err(format!(
				"state_connections length {} != state_neurons * state_bits_per_neuron = {}",
				state_connections.len(), expected_state_conn
			)));
		}
		let num_output_neurons = num_motors * levels_per_motor;
		let expected_output_conn = num_output_neurons * output_bits_per_neuron;
		if output_connections.len() != expected_output_conn {
			return Err(pyo3::exceptions::PyValueError::new_err(format!(
				"output_connections length {} != num_motors * levels_per_motor * output_bits_per_neuron = {}",
				output_connections.len(), expected_output_conn
			)));
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
		})
	}

	/// Zero the recurrent state buffer and clear the input history.
	fn reset(&mut self) {
		for v in self.prev_state.iter_mut() { *v = 0; }
		self.input_history.clear();
		for v in self.last_output_cells.iter_mut() { *v = 0; }
		self.last_state_layer_input.clear();
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
	fn get_last_output_cells(&self) -> Vec<u8> {
		self.last_output_cells.clone()
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
	/// Returns the number of cells modified.
	fn train_output_step(&mut self, target_pwm: [f32; 4]) -> usize {
		// Recreate the output-layer-input bits the way step() built them
		// (2 bits per state neuron, MSB-LSB from QSR value).
		let state_bits_in = 2 * self.state_neurons;
		let mut output_input = vec![false; state_bits_in];
		for (n, &v) in self.prev_state.iter().enumerate() {
			output_input[2 * n] = (v >> 1) & 1 != 0;
			output_input[2 * n + 1] = v & 1 != 0;
		}

		let num_out = self.num_motors * self.levels_per_motor;
		let levels = self.levels_per_motor;
		let mut writes = 0usize;
		for n in 0..num_out {
			let motor = n / levels;
			let level_idx = n % levels;
			let p = target_pwm[motor].clamp(0.0, 1.0);
			let target_true = (p * levels as f32) as usize > level_idx;

			let conn_start = n * self.output_bits_per_neuron;
			let conn_end = conn_start + self.output_bits_per_neuron;
			let address = crate::neuron_memory::compute_address_sparse(
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
			let p = target_pwm[motor].clamp(0.0, 1.0);
			let target_true = (p * levels as f32) as usize > level_idx;

			let conn_start = n * self.state_bits_per_neuron;
			let conn_end = conn_start + self.state_bits_per_neuron;
			let address = crate::neuron_memory::compute_address_sparse(
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
		let state_bits_in = 2 * self.state_neurons;

		// Current output-layer input = QSR-encoded state that step() produced.
		let mut output_input = vec![false; state_bits_in];
		for (n, &v) in self.prev_state.iter().enumerate() {
			output_input[2 * n] = (v >> 1) & 1 != 0;
			output_input[2 * n + 1] = v & 1 != 0;
		}

		// Per-motor output solve → vote per state-input bit.
		// vote[i] > 0 → motors want bit i TRUE; < 0 → FALSE; 0 → keep current.
		let mut vote = vec![0i32; state_bits_in];
		for m in 0..self.num_motors {
			let p = target_pwm[m].clamp(0.0, 1.0);
			let n_true = (p * levels as f32) as usize;
			let motor_target: Vec<bool> = (0..levels).map(|i| i < n_true).collect();

			let conn_start = m * levels * obpn;
			let conn_end = (m + 1) * levels * obpn;
			let motor_conns = &self.output_connections[conn_start..conn_end];

			let base = m * levels;
			let read = |nn: usize, addr: usize| self.output_memory.read_cell(base + nn, addr as u64);
			let solved = solve_partial_connectivity_qsr(
				read, motor_conns, levels, obpn, state_bits_in,
				&output_input, &motor_target, 0, topk_per_neuron,
			);
			if let Some(sol) = solved {
				for i in 0..state_bits_in {
					vote[i] += if sol[i] { 1 } else { -1 };
				}
			}
		}

		let desired_state_bits: Vec<bool> = (0..state_bits_in)
			.map(|i| if vote[i] > 0 { true } else if vote[i] < 0 { false } else { output_input[i] })
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
				let target_val =
					((desired_state_bits[2 * n] as u8) << 1) | (desired_state_bits[2 * n + 1] as u8);
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
	fn step(&mut self,
	        gyro: [f32; 3],
	        accel: [f32; 3],
	        target_attitude: [f32; 3]) -> Vec<f32> {
		// 1. Build the current-frame sensor vector and thermometer-encode it.
		let sensors = [
			gyro[0], gyro[1], gyro[2],
			accel[0], accel[1], accel[2],
			target_attitude[0], target_attitude[1], target_attitude[2],
		];
		let bpf = self.bits_per_feature;
		let mut frame = vec![false; NUM_FEATURES * bpf];
		for f in 0..NUM_FEATURES {
			let v = sensors[f];
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
		let frame_bits = NUM_FEATURES * bpf;
		let sensor_total = self.input_window_k * frame_bits;
		let state_bits_in = 2 * self.state_neurons;
		let total_input_bits = sensor_total + state_bits_in;
		let mut input_bits = vec![false; total_input_bits];

		// Frames: oldest first. If history has fewer than K, the missing oldest
		// slots stay zero (paddding with no past observation).
		let pad = self.input_window_k - self.input_history.len();
		for (i, frame) in self.input_history.iter().enumerate() {
			let slot = (pad + i) * frame_bits;
			input_bits[slot..slot + frame_bits].copy_from_slice(frame);
		}

		// QSR state encoding: cell value 0..3 → 2 bits (MSB, LSB) = ((v>>1)&1, v&1).
		// This preserves the QSR ordering (00 < 01 < 10 < 11) so connections
		// targeting either pair bit get a meaningful gradient of "how confident
		// was that state neuron".
		for (n, &v) in self.prev_state.iter().enumerate() {
			let base = sensor_total + 2 * n;
			input_bits[base] = (v >> 1) & 1 != 0;
			input_bits[base + 1] = v & 1 != 0;
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

		// 5. Output-layer input: 2 bits per new-state neuron (QSR encoding).
		let mut output_input = vec![false; state_bits_in];
		for (n, &v) in new_state.iter().enumerate() {
			output_input[2 * n] = (v >> 1) & 1 != 0;
			output_input[2 * n + 1] = v & 1 != 0;
		}

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

		// 7. Strategy-5 decode → PWM in [0, 1].
		let mut pwm = Vec::with_capacity(self.num_motors);
		let denom = self.levels_per_motor as f32;
		for m in 0..self.num_motors {
			let start = m * self.levels_per_motor;
			let mut sum: f32 = 0.0;
			for &cell in &self.last_output_cells[start..start + self.levels_per_motor] {
				sum += QSR_WEIGHTS[(cell & 0x3) as usize];
			}
			// sum ranges [0, levels_per_motor]; map directly to [0, 1].
			pwm.push((sum / denom).clamp(0.0, 1.0));
		}

		// 8. Update recurrent state for next step.
		self.prev_state = new_state;

		pwm
	}

	#[getter]
	fn num_motors(&self) -> usize { self.num_motors }
	#[getter]
	fn levels_per_motor(&self) -> usize { self.levels_per_motor }
	#[getter]
	fn state_neurons(&self) -> usize { self.state_neurons }
	#[getter]
	fn input_window_k(&self) -> usize { self.input_window_k }
	#[getter]
	fn bits_per_feature(&self) -> usize { self.bits_per_feature }
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
