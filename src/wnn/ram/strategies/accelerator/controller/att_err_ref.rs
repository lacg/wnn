// AttErrRef — what the scorers' per-step ATTITUDE ERROR is measured against
// (CTRL-17 G5, ABI 32, 31/10/2026).
//
// LEVEL (the default, every ABI ≤ 31 result) = the geodesic angle to the
// identity quaternion. That is right for attitude-only and stage-1 episodes,
// but WRONG for a displaced stage-2 start: flying back to the target REQUIRES
// tilting toward it, so a level reference charges the correct manoeuvre as
// attitude error and the err/stable/steady triple punishes position control.
//
// TILT = the geodesic angle to the reference attitude the classical position
// loop would command at this state: PositionLoop::tilt_ref(e, v) — the SAME
// derived law (omega, zeta, max_tilt, b_xy = g) the DAgger teacher's cascade
// uses — as roll/pitch with yaw target 0 (xyz Tait-Bryan, the teachers'
// convention). On target and at rest the reference IS level, so a converged
// episode is scored exactly as before.
//
// Every scorer that reports err (CPU rollout_one, the Metal twin, the classical
// rivals, the position-teacher scorer, the DAgger step_reward / checkpoint)
// reads the reference through `error`, so WNN and rivals stay comparable.

use crate::controller::AttitudeSim;
use crate::position_loop::PositionLoop;

#[derive(Clone, Copy, Debug)]
pub struct AttErrRef
{
	/// None ⇒ LEVEL (identity). Some ⇒ the position loop's tilt reference.
	pos: Option<PositionLoop>,
}

impl AttErrRef
{
	pub const LEVEL: AttErrRef = AttErrRef { pos: None };

	/// The TILT reference, derived from the plant exactly like the teacher's loop.
	pub fn tilt(gravity: f64, omega_n: f64, zeta: f64, max_tilt_rad: f64) -> Result<Self, String>
	{
		Ok(AttErrRef {
			pos: Some(PositionLoop::from_plant(gravity, omega_n, zeta, max_tilt_rad)?),
		})
	}

	/// Build from the host's kwargs: `on` false ⇒ LEVEL (nothing derived).
	pub fn from_flag(
		on: bool,
		gravity: f64,
		omega_n: f64,
		zeta: f64,
		max_tilt_rad: f64,
	) -> Result<Self, String>
	{
		if on
		{
			Self::tilt(gravity, omega_n, zeta, max_tilt_rad)
		}
		else
		{
			Ok(Self::LEVEL)
		}
	}

	pub fn is_tilt(&self) -> bool
	{
		self.pos.is_some()
	}

	/// The tilt reference is only defined for a horizontally-displaced episode
	/// (the position loop needs a position error). Refuse it elsewhere.
	pub fn require_horizontal(&self, horizontal: bool, site: &str) -> Result<(), String>
	{
		if self.is_tilt() && !horizontal
		{
			return Err(format!(
				"{site}: att_err_tilt_ref needs the stage-2 horizontal channel \
				 (translation with xy_offset > 0) — without a position error the \
				 reference is level by construction"
			));
		}
		Ok(())
	}

	/// (b_xy, omega_n, zeta, max_tilt) as f32 for the Metal kernel; None = LEVEL.
	pub fn gpu_shape(&self) -> Option<[f32; 4]>
	{
		self.pos.map(|pl| {
			let (b, w, z, m) = pl.shape();
			[b as f32, w as f32, z as f32, m as f32]
		})
	}

	/// The reference quaternion (w, x, y, z) at the sim's CURRENT state, or None
	/// for LEVEL. Target is the origin, so e = −position.
	pub fn ref_quat(&self, sim: &AttitudeSim) -> Option<[f32; 4]>
	{
		let pl = self.pos.as_ref()?;
		let [x, y] = sim.position_xy_rs();
		let [vx, vy] = sim.velocity_xy_rs();
		let (roll, pitch) = pl.tilt_ref(-(x as f64), vx as f64, -(y as f64), vy as f64);
		Some(rp_to_quat(roll, pitch))
	}

	/// The per-step attitude error (rad) at the sim's current state. LEVEL is
	/// the exact legacy call `sim.attitude_error(None)`.
	#[inline]
	pub fn error(&self, sim: &AttitudeSim) -> f32
	{
		sim.attitude_error(self.ref_quat(sim))
	}
}

/// xyz Tait-Bryan (roll, pitch, yaw = 0) → quaternion (w, x, y, z). The yaw = 0
/// specialization of dagger_train::euler_to_quat_xyz; the Metal twin
/// (controller_rollout.metal tilt_ref_quat) computes the same four products.
fn rp_to_quat(roll: f64, pitch: f64) -> [f32; 4]
{
	let (cr, sr) = ((roll * 0.5).cos(), (roll * 0.5).sin());
	let (cp, sp) = ((pitch * 0.5).cos(), (pitch * 0.5).sin());
	[(cr * cp) as f32, (sr * cp) as f32, (cr * sp) as f32, (-sr * sp) as f32]
}

#[cfg(test)]
mod tests
{
	use super::*;

	fn sim_at(x: f32, y: f32) -> AttitudeSim
	{
		let mut s = AttitudeSim::new(0.001, 0.0707, 1.0, 0.0057, [1.66e-5, 1.66e-5, 2.93e-5], 9.81);
		s.set_translation_core(0.035).unwrap();
		s.set_horizontal_state(x, y, 0.0, 0.0);
		s
	}

	/// LEVEL is the legacy call, bit-for-bit; TILT at the origin at rest is level.
	#[test]
	fn level_is_legacy_and_tilt_is_level_on_target()
	{
		let s = sim_at(0.4, -0.3);
		assert_eq!(AttErrRef::LEVEL.error(&s).to_bits(), s.attitude_error(None).to_bits());
		let t = AttErrRef::tilt(9.81, 1.0, 1.0, 0.5236).unwrap();
		let o = sim_at(0.0, 0.0);
		assert!((t.error(&o) - o.attitude_error(None)).abs() < 1e-6);
	}

	/// Displaced at level: the tilt reference says "tilt toward the target", so
	/// a level vehicle carries a NON-zero error equal to the commanded tilt.
	#[test]
	fn displaced_level_vehicle_carries_the_commanded_tilt()
	{
		let t = AttErrRef::tilt(9.81, 1.0, 1.0, 0.5236).unwrap();
		let s = sim_at(0.5, 0.0); // target at −x ⇒ pitch −ω²·0.5/g
		let want = (1.0f64 * 0.5 / 9.81) as f32;
		assert!((t.error(&s) - want).abs() < 1e-4, "got {} want {}", t.error(&s), want);
		assert!(t.require_horizontal(false, "t").is_err());
		assert!(AttErrRef::LEVEL.require_horizontal(false, "t").is_ok());
	}
}
