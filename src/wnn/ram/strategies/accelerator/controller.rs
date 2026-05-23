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
//! STATUS: stubs. Each function returns a safe default so the build +
//! PyO3 bindings can be validated. Physics + thermometer encoding +
//! actual network forward are TODO and tracked in
//! `project_drone_controller_paper1.md`.

use pyo3::prelude::*;

// Strategy-5 QSR weight lookup table. Index by raw cell value (0..3).
// FALSE=0=0.0, WEAK_FALSE=1=0.25, WEAK_TRUE=2=0.75, TRUE=3=1.0.
const QSR_WEIGHTS: [f32; 4] = [0.0, 0.25, 0.75, 1.0];

// =============================================================================
// AttitudeSim
// =============================================================================

#[pyclass]
pub struct AttitudeSim {
	// Unit quaternion (w, x, y, z), body-to-world. Identity = level attitude.
	q: [f32; 4],
	// Angular velocity in body frame (rad/s).
	omega: [f32; 3],
	// Simulator time (s).
	t: f32,
	// Integration step (s). Default 1 ms = 1 kHz update.
	dt: f32,
}

#[pymethods]
impl AttitudeSim {
	#[new]
	#[pyo3(signature = (dt = 0.001))]
	fn new(dt: f32) -> Self {
		Self {
			q: [1.0, 0.0, 0.0, 0.0],
			omega: [0.0, 0.0, 0.0],
			t: 0.0,
			dt,
		}
	}

	/// Reset the simulator. Optional initial quaternion (defaults to identity)
	/// and initial angular velocity (defaults to zero).
	#[pyo3(signature = (q = None, omega = None))]
	fn reset(&mut self, q: Option<[f32; 4]>, omega: Option<[f32; 3]>) {
		self.q = q.unwrap_or([1.0, 0.0, 0.0, 0.0]);
		self.omega = omega.unwrap_or([0.0, 0.0, 0.0]);
		self.t = 0.0;
	}

	/// Advance one timestep under the given 4-motor PWM (each in [0, 1]).
	/// TODO: implement RK4 integration of Euler's rotational equation +
	/// quaternion update. Current stub: no-op except for time bookkeeping.
	fn step(&mut self, _motor_pwm: [f32; 4]) {
		// TODO: motor_thrust = k_thrust * pwm²
		// TODO: body_torque = motor_mixing @ motor_thrust
		// TODO: dω/dt = I⁻¹ (τ - ω × (I ω))     (Euler's equation)
		// TODO: dq/dt = 0.5 (q ⊗ ω_quat)
		// TODO: RK4 integrate at self.dt
		self.t += self.dt;
	}

	/// Read the simulated IMU: (gyro_xyz, accel_xyz) in body frame.
	/// TODO: derive accel from rotation + gravity in body frame.
	fn read_imu(&self) -> ([f32; 3], [f32; 3]) {
		// gyro = body-frame angular velocity (current omega)
		// accel = body-frame specific force; at rest, points up at +1 g (with gravity)
		let gyro = self.omega;
		let accel = [0.0, 0.0, 9.81]; // TODO: rotate gravity into body frame
		(gyro, accel)
	}

	/// Geodesic angle (rad) between current attitude and target attitude.
	/// Target defaults to identity (level).
	#[pyo3(signature = (target = None))]
	fn attitude_error(&self, target: Option<[f32; 4]>) -> f32 {
		let _t = target.unwrap_or([1.0, 0.0, 0.0, 0.0]);
		// TODO: angle = 2 * acos(|q · t|)
		0.0
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
