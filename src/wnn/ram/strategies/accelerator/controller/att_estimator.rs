// AttEstimator — the WNN's on-board attitude estimate (ABI 32, 31/10/2026).
//
// WHY. Under the ABI-32 specific-force IMU (Luiz option A) a free-flying
// accelerometer reads ≈ (0, 0, T/m): it carries no tilt, so the WNN's legacy
// ACCEL-ONLY attitude features (roll_est = atan2(ay, az), …) would be blind.
// A real flight controller gets tilt from gyro integration corrected by the
// accelerometer on average — an attitude estimator. This wraps the SAME Mahony
// filter the classical rivals are fed by (estimator.rs MahonyFilter, kp/ki
// defaults 2.0/0.1 = TeacherFeed / the scorers' est_kp/est_ki), warm-started
// from the episode's initial attitude exactly like the rivals (the
// converged-filter assumption, disclosed).
//
// WHAT THE FEATURES READ. The estimate's predicted body-frame "up" (world +z
// rotated into the body frame) — the vector Mahony itself compares against the
// accelerometer. compute_features plugs it into the unchanged accel-only
// formulas (atan2 is scale-free), so tilt/roll/pitch keep their definitions.
//
// REPLAY. A replay trainer has no live filter: the rollout records `last_up()`
// per step (TrajectoryRs.att_up → ReplayObs.att_up) and the replay hands it back
// through `set_override`, so CPU BPTT and the GPU split replay address on
// exactly what deploy saw (the ReplayObs.pwm_dev pattern).
//
// FAIL LOUD. A live step on an UNSEEDED estimator panics: a host that forgot to
// seed it would fly the filter from identity — a silent wrong-attitude start.

use crate::estimator::MahonyFilter;

#[derive(Clone)]
pub struct AttEstimator
{
	filter: MahonyFilter,
	kp: f32,
	ki: f32,
	seeded: bool,
	last_up: [f32; 3],
	override_up: Option<[f32; 3]>,
}

impl AttEstimator
{
	pub fn new(dt: f32, kp: f32, ki: f32) -> Result<Self, String>
	{
		if !(dt > 0.0) || !kp.is_finite() || kp < 0.0 || !ki.is_finite() || ki < 0.0
		{
			return Err(format!(
				"obs_att_estimator: need dt > 0 and finite kp, ki >= 0 (got dt={dt}, kp={kp}, ki={ki})"
			));
		}
		Ok(Self {
			filter: MahonyFilter::new(dt as f64, kp as f64, ki as f64),
			kp,
			ki,
			seeded: false,
			last_up: [0.0, 0.0, 1.0],
			override_up: None,
		})
	}

	/// (kp, ki) — what the Metal twin needs.
	pub fn gains(&self) -> (f32, f32)
	{
		(self.kp, self.ki)
	}

	/// Episode start: forget the previous episode (the host must re-seed).
	pub fn unseed(&mut self)
	{
		self.seeded = false;
		self.override_up = None;
		self.last_up = [0.0, 0.0, 1.0];
	}

	/// Warm-start from the episode's initial attitude (the rivals' reset).
	pub fn seed(&mut self, q0: [f32; 4])
	{
		self.filter.reset(Some(q0));
		self.seeded = true;
	}

	/// Replay: the NEXT `up()` returns this recorded value (one-shot).
	pub fn set_override(&mut self, up: [f32; 3])
	{
		self.override_up = Some(up);
	}

	/// The "up" vector the last `up()` produced — what the rollout records.
	pub fn last_up(&self) -> [f32; 3]
	{
		self.last_up
	}

	/// This step's body "up": the replay override when set, else one live
	/// Mahony update on (gyro, accel). Called exactly once per physical step.
	pub fn up(&mut self, gyro: [f32; 3], accel: [f32; 3]) -> [f32; 3]
	{
		let up = match self.override_up.take()
		{
			Some(u) => u,
			None =>
			{
				assert!(
					self.seeded,
					"obs_att_estimator: live step on an UNSEEDED estimator — the host \
					 must call seed_att_estimator(q0) after reset (the filter would \
					 otherwise start from identity: a silent wrong-attitude start)"
				);
				up_from_quat(self.filter.update(gyro, accel))
			}
		};
		self.last_up = up;
		up
	}
}

/// Body-frame "up" (world +z in the body frame) of q = (w, x, y, z) — the
/// Mahony v-vector, f32 (twin of controller_rollout.metal mahony_up).
pub fn up_from_quat(q: [f32; 4]) -> [f32; 3]
{
	let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
	[2.0 * (x * z - w * y), 2.0 * (w * x + y * z), w * w - x * x - y * y + z * z]
}

#[cfg(test)]
mod tests
{
	use super::*;

	/// At a roll φ the "up" vector must give roll_est = atan2(up_y, up_z) = φ —
	/// the accel-only formula's convention (accel = −g_body ∝ up at rest).
	#[test]
	fn up_vector_matches_the_accel_convention()
	{
		let phi = 0.3f32;
		let q = [(phi * 0.5).cos(), (phi * 0.5).sin(), 0.0, 0.0];
		let u = up_from_quat(q);
		assert!((u[1].atan2(u[2]) - phi).abs() < 1e-6);
	}

	#[test]
	#[should_panic(expected = "UNSEEDED")]
	fn live_step_without_seed_panics()
	{
		let mut e = AttEstimator::new(0.001, 2.0, 0.1).unwrap();
		e.up([0.0; 3], [0.0, 0.0, 9.81]);
	}

	/// Override is one-shot and is what last_up reports.
	#[test]
	fn override_is_one_shot()
	{
		let mut e = AttEstimator::new(0.001, 2.0, 0.1).unwrap();
		e.seed([1.0, 0.0, 0.0, 0.0]);
		e.set_override([0.1, 0.2, 0.9]);
		assert_eq!(e.up([0.0; 3], [0.0, 0.0, 9.81]), [0.1, 0.2, 0.9]);
		assert_eq!(e.last_up(), [0.1, 0.2, 0.9]);
		let live = e.up([0.0; 3], [0.0, 0.0, 9.81]);
		assert!((live[2] - 1.0).abs() < 1e-5, "level + level accel stays level");
	}
}
