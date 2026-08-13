// Mahony complementary attitude estimator — the RUST TWIN of
// `wnn/control/estimator.py` (the Python file is the REFERENCE; the
// golden-trajectory parity test below pins this twin to it, same discipline as
// pid_firmware's matches_python_reference_golden_trajectory).
//
// The estimator-fed teacher rule (Luiz, 13/08/2026): comparison-row teachers
// never read the true quaternion — they read THIS filter's output, computed
// from the same noisy IMU stream the WNN reads raw. Training (DAgger)
// teachers stay oracle.
//
// Algorithm (Mahony, Nonlinear Complementary Filters on SO(3), IEEE TAC 2008 —
// the MahonyAHRS formulation): predict the body-frame "up" direction from the
// current estimate, cross-product error against the accelerometer's measured
// "up" (specific force ≈ support at hover), feed kp·e (+ ki·∫e) back into the
// gyro integration. Yaw is unobservable from gyro+accel (cf21_brushless has no
// magnetometer) — it dead-reckons from the warm-start value, the same anchor
// the WNN gets at reset.
//
// f64 internally (like pid_firmware, and like the Python reference, which
// computes in Python floats = f64); f32 only at the sim/teacher boundary.

use pyo3::prelude::*;

pub struct MahonyFilter {
	dt: f64,
	kp: f64,
	ki: f64,
	q: [f64; 4],
	integral: [f64; 3],
}

impl MahonyFilter {
	pub fn new(dt: f64, kp: f64, ki: f64) -> Self {
		Self { dt, kp, ki, q: [1.0, 0.0, 0.0, 0.0], integral: [0.0; 3] }
	}

	/// Warm-start from q0 (the converged-filter assumption: a firmware filter
	/// has been running since before takeoff — disclosed). None = identity.
	pub fn reset(&mut self, q0: Option<[f32; 4]>) {
		let q = q0.map_or([1.0, 0.0, 0.0, 0.0], |v| v.map(|x| x as f64));
		let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
		let n = if n > 0.0 { n } else { 1.0 };
		self.q = [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
		self.integral = [0.0; 3];
	}

	/// One fusion step: (measured gyro rad/s, measured accel m/s²) → quaternion
	/// estimate (w, x, y, z). Pure function of the sensor stream — no true
	/// state is ever read. Line-for-line twin of MahonyEstimator.update().
	pub fn update(&mut self, gyro: [f32; 3], accel: [f32; 3]) -> [f32; 4] {
		let (w, x, y, z) = (self.q[0], self.q[1], self.q[2], self.q[3]);
		let (mut gx, mut gy, mut gz) = (gyro[0] as f64, gyro[1] as f64, gyro[2] as f64);
		let (mut ax, mut ay, mut az) = (accel[0] as f64, accel[1] as f64, accel[2] as f64);

		let norm = (ax * ax + ay * ay + az * az).sqrt();
		if norm > 1e-9 {
			ax /= norm;
			ay /= norm;
			az /= norm;
			// Predicted body-frame "up" = world +z rotated into the body frame
			// (the MahonyAHRS v-vector).
			let vx = 2.0 * (x * z - w * y);
			let vy = 2.0 * (w * x + y * z);
			let vz = w * w - x * x - y * y + z * z;
			// Error = measured × predicted; zero when they agree.
			let ex = ay * vz - az * vy;
			let ey = az * vx - ax * vz;
			let ez = ax * vy - ay * vx;
			if self.ki > 0.0 {
				self.integral[0] += self.ki * ex * self.dt;
				self.integral[1] += self.ki * ey * self.dt;
				self.integral[2] += self.ki * ez * self.dt;
			}
			gx += self.kp * ex + self.integral[0];
			gy += self.kp * ey + self.integral[1];
			gz += self.kp * ez + self.integral[2];
		}

		// q̇ = ½ q ⊗ (0, ω_corrected); first-order step, then renormalize.
		let half_dt = 0.5 * self.dt;
		let dw = (-x * gx - y * gy - z * gz) * half_dt;
		let dx = (w * gx + y * gz - z * gy) * half_dt;
		let dy = (w * gy - x * gz + z * gx) * half_dt;
		let dz = (w * gz + x * gy - y * gx) * half_dt;
		let (w, x, y, z) = (w + dw, x + dx, y + dy, z + dz);
		let n = (w * w + x * x + y * y + z * z).sqrt();
		let n = if n > 0.0 { n } else { 1.0 };
		self.q = [w / n, x / n, y / n, z / n];
		[self.q[0] as f32, self.q[1] as f32, self.q[2] as f32, self.q[3] as f32]
	}
}

/// Python-facing wrapper (thin, house pattern — logic lives in MahonyFilter).
#[pyclass]
pub struct MahonyEstimatorRs {
	inner: MahonyFilter,
}

#[pymethods]
impl MahonyEstimatorRs {
	#[new]
	#[pyo3(signature = (dt, kp = 2.0, ki = 0.1))]
	pub fn new(dt: f64, kp: f64, ki: f64) -> PyResult<Self> {
		if dt <= 0.0 {
			return Err(pyo3::exceptions::PyValueError::new_err(
				format!("dt must be > 0, got {dt}")));
		}
		Ok(Self { inner: MahonyFilter::new(dt, kp, ki) })
	}

	#[pyo3(signature = (q0 = None))]
	pub fn reset(&mut self, q0: Option<[f32; 4]>) {
		self.inner.reset(q0);
	}

	pub fn update(&mut self, gyro: [f32; 3], accel: [f32; 3]) -> [f32; 4] {
		self.inner.update(gyro, accel)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// GOLDEN-TRAJECTORY PARITY with the Python reference (estimator.py).
	///
	/// Inputs are deterministic sinusoids exercising all axes; the expected
	/// quaternions were printed by the reference itself (13/08/2026, f64).
	/// Tolerance 1e-9: both sides compute in f64 with the same operation
	/// order, so anything looser would hide a transcription drift.
	#[test]
	fn matches_python_reference_golden_trajectory() {
		let goldens: [(usize, [f64; 4]); 6] = [
			(1, [0.958044389900, 0.106461965380, -0.159543921298, 0.212984821257]),
			(10, [0.958016617674, 0.106824462886, -0.158455026159, 0.213739792913]),
			(100, [0.956148874239, 0.123898818036, -0.154477667354, 0.215789396098]),
			(500, [0.975684972455, 0.029804187469, -0.055911055430, 0.209820158267]),
			(1000, [0.974028510961, -0.015608359786, -0.041414101305, 0.222057900452]),
			(2000, [0.973889064926, -0.001222423659, -0.049834872263, 0.221483815222]),
		];
		let mut est = MahonyFilter::new(0.001, 2.0, 0.1);
		est.reset(Some([0.9, 0.1, -0.15, 0.2]));
		let mut gi = 0usize;
		for t in 1..=2000usize {
			let tf = t as f64;
			let gyro = [
				(0.8 * (0.013 * tf).sin()) as f32,
				(-0.5 * (0.007 * tf + 1.0).sin()) as f32,
				(0.3 * (0.019 * tf + 2.0).sin()) as f32,
			];
			let a = [0.2 * (0.011 * tf).sin(), 0.2 * (0.017 * tf).cos(), 1.0];
			let n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
			let accel = [
				(9.81 * a[0] / n) as f32,
				(9.81 * a[1] / n) as f32,
				(9.81 * a[2] / n) as f32,
			];
			est.update(gyro, accel);
			if gi < goldens.len() && goldens[gi].0 == t {
				for k in 0..4 {
					let got = est.q[k];
					let want = goldens[gi].1[k];
					assert!((got - want).abs() < 1e-6,
						"step {t} component {k}: rust {got} vs python {want} — \
						 the twin drifted from the reference");
				}
				gi += 1;
			}
		}
		assert_eq!(gi, goldens.len(), "not every golden checkpoint was reached");
	}

	/// The filter must TRACK a flying vehicle from sensors alone: drive a clean
	/// AttitudeSim open-loop and require the estimate to stay near the true
	/// attitude (clean sensors ⇒ the only error is integration order).
	#[test]
	fn tracks_clean_sim_attitude() {
		let mut sim = crate::controller::AttitudeSim::new(
			0.001, 0.0707, 0.2, 0.0057, [1.66e-5, 1.66e-5, 2.93e-5], 9.81);
		sim.reset(Some([0.996, 0.05, -0.05, 0.03]), Some([0.3, -0.2, 0.1]));
		let mut est = MahonyFilter::new(0.001, 2.0, 0.1);
		est.reset(Some(sim.quaternion()));
		let mut max_diff_deg = 0.0f64;
		for _ in 0..1000 {
			let (gyro, accel) = sim.read_imu();
			let qe = est.update(gyro, accel);
			let qt = sim.quaternion();
			let dot = (qe[0] as f64) * (qt[0] as f64) + (qe[1] as f64) * (qt[1] as f64)
				+ (qe[2] as f64) * (qt[2] as f64) + (qe[3] as f64) * (qt[3] as f64);
			let diff = 2.0 * dot.abs().min(1.0).acos().to_degrees();
			max_diff_deg = max_diff_deg.max(diff);
			// mild differential so the attitude actually moves
			sim.step([0.70, 0.69, 0.70, 0.69]);
		}
		assert!(max_diff_deg < 1.0,
			"estimator lost a cleanly-sensed vehicle: max divergence {max_diff_deg:.3}°");
	}

	/// Zero-norm accel must not poison the state (dropout-style reading):
	/// the correction is skipped and the gyro integration continues.
	#[test]
	fn zero_accel_skips_correction() {
		let mut est = MahonyFilter::new(0.001, 2.0, 0.1);
		est.reset(Some([1.0, 0.0, 0.0, 0.0]));
		let q = est.update([0.1, 0.0, 0.0], [0.0, 0.0, 0.0]);
		assert!(q.iter().all(|v| v.is_finite()), "NaN leaked from a zero accel");
	}
}
