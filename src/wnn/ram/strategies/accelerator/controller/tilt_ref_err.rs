//! L1 (CTRL-48 block 10, ABI 51, `--obs-tilt-ref-err`): the student's roll / pitch
//! error measured against the position loop's TILT REFERENCE, bias-corrected
//! (docs/ctrl48_block10_levers_spec.md §L1).
//!
//! WHAT IT CHANGES. Only the VALUE of the two existing inputs `roll_err`, `pitch_err`
//! (and so of their integrals). With the option on
//!     roll_err  = (target_r + roll_ref  + abp_r) − roll_est
//!     pitch_err = (target_p + pitch_ref + abp_p) − pitch_est
//! where (roll_ref, pitch_ref) is the position law `PositionLoop::tilt_ref` on the
//! horizontal observation the student already reads ([e_x, e_y, v_x, v_y], world
//! frame) and (abp_r, abp_p) is this step's Mahony-bias prediction. No input is
//! added, removed or moved.
//!
//! WHY f32. The student's signals are f32 and the Metal kernels evaluate the SAME
//! expressions in the same order (controller_rollout.metal `tilt_ref_rp`), so the
//! live rollout, the replay kernels and this type agree. `PositionLoop::tilt_ref`
//! (f64) is the definition; `matches_the_position_loop` pins the two together.

use crate::position_loop::PositionLoop;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct TiltRefErr
{
	/// The position loop's shape: b_xy = g, bandwidth, damping, tilt clamp (rad).
	b_xy: f32,
	omega_n: f32,
	zeta: f32,
	max_tilt: f32,
}

impl TiltRefErr
{
	/// From the position loop's plant and shape — validated by PositionLoop itself,
	/// so the feature can never carry a law the teacher's loop would refuse.
	pub(crate) fn new(gravity: f32, omega_n: f32, zeta: f32, max_tilt_rad: f32) -> Result<Self, String>
	{
		let pl = PositionLoop::from_plant(gravity as f64, omega_n as f64, zeta as f64, max_tilt_rad as f64)?;
		let (b, w, z, m) = pl.shape();
		Ok(TiltRefErr { b_xy: b as f32, omega_n: w as f32, zeta: z as f32, max_tilt: m as f32 })
	}

	/// [b_xy, omega_n, zeta, max_tilt] — the Metal uniforms and the Python getter.
	pub(crate) fn shape(&self) -> [f32; 4]
	{
		[self.b_xy, self.omega_n, self.zeta, self.max_tilt]
	}

	/// (roll_ref, pitch_ref) in rad from the student's horizontal observation
	/// [e_x, e_y, v_x, v_y]. f32 twin of PositionLoop::tilt_ref and of the kernel's
	/// tilt_ref_rp (same expressions, same order).
	#[inline]
	pub(crate) fn tilt_ref(&self, horiz: [f32; 4]) -> (f32, f32)
	{
		let w2 = self.omega_n * self.omega_n;
		let c2 = 2.0 * self.zeta * self.omega_n;
		let pitch = ((w2 * horiz[0] - c2 * horiz[2]) / self.b_xy).clamp(-self.max_tilt, self.max_tilt);
		let roll = (-(w2 * horiz[1] - c2 * horiz[3]) / self.b_xy).clamp(-self.max_tilt, self.max_tilt);
		(roll, pitch)
	}

	/// The (roll, pitch) targets the error is measured against this step:
	/// (target + tilt reference) + predicted bias.
	#[inline]
	pub(crate) fn effective_target(&self, target: [f32; 3], horiz: [f32; 4], abp: [f32; 2]) -> [f32; 2]
	{
		let (roll_ref, pitch_ref) = self.tilt_ref(horiz);
		[(target[0] + roll_ref) + abp[0], (target[1] + pitch_ref) + abp[1]]
	}
}

#[cfg(test)]
mod tests
{
	use super::*;

	/// The f32 law is PositionLoop::tilt_ref (the definition), clamp included.
	#[test]
	fn matches_the_position_loop()
	{
		let t = TiltRefErr::new(9.81, 1.0, 1.0, 0.5236).unwrap();
		let pl = PositionLoop::from_plant(9.81, 1.0, 1.0, 0.5236f32 as f64).unwrap();
		for h in [[0.3f32, -0.2, 0.05, 0.4], [0.0, 0.0, 0.0, 0.0], [-1.1, 0.7, -0.3, 0.2], [40.0, -40.0, 0.0, 0.0]]
		{
			let (r, p) = t.tilt_ref(h);
			let (rd, pd) = pl.tilt_ref(h[0] as f64, h[2] as f64, h[1] as f64, h[3] as f64);
			assert!((r as f64 - rd).abs() < 1e-6 && (p as f64 - pd).abs() < 1e-6, "{h:?}: ({r}, {p}) vs ({rd}, {pd})");
		}
		let (r, p) = t.tilt_ref([40.0, -40.0, 0.0, 0.0]);
		assert_eq!((r, p), (0.5236, 0.5236), "a large error saturates at the tilt clamp");
	}

	#[test]
	fn refuses_a_degenerate_loop()
	{
		assert!(TiltRefErr::new(0.0, 1.0, 1.0, 0.5).is_err());
		assert!(TiltRefErr::new(9.81, 0.0, 1.0, 0.5).is_err());
		assert!(TiltRefErr::new(9.81, 1.0, 1.0, 2.0).is_err());
	}
}
