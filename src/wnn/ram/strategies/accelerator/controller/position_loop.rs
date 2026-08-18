// Horizontal position loop — the outermost stage of the scope C stage 2
// teacher cascade (docs/scope_c_stage2_chunk_b_teacher.md).
//
// WHAT IT IS: position error → a TILT REFERENCE handed to the inner attitude
// teacher. Together with AltitudePd (which hands down the collective) this makes
// the classical teacher a full-state controller without replacing any of the
// five attitude rivals — each stays a valid inner loop, so stage 2 keeps the
// whole comparison table instead of collapsing it to one monolithic LQR.
//
// IT MUST BE DISCLOSED AS A CASCADE wherever it is reported. That is the same
// requirement AltitudePd carries, and it is honest precisely because the
// classical rivals ARE cascades — the WNN is the monolithic one, which is the
// entire point of the comparison.
//
// THE GAINS ARE DERIVED, NOT GUESSED, by the same construction AltitudePd uses.
// Linearize horizontal motion about hover: the supporting thrust is T ≈ mg, so
// tilting by θ tips that thrust sideways and
//     ẍ ≈ (T/m)·sin θ_pitch ≈ g·θ_pitch
//     ÿ ≈ −(T/m)·sin φ_roll ≈ −g·φ_roll
// The control effectiveness of TILT on horizontal acceleration is therefore just
//     b_xy = g          (m/s² per radian)
// with no mass and no thrust coefficient in it — they cancel, because the thrust
// being tipped is exactly the thrust holding the vehicle up. Choosing the loop
// by its SHAPE rather than by raw gains,
//     a_des = ωn²·e_p − 2ζωn·v
// and inverting b_xy gives the tilt reference directly.
//
// LOOP SEPARATION: attitude is the fast loop, altitude sits a decade below it
// (ωn = 2.0), and position must be slower still or the two outer loops argue.
// Default ωn = 1.0, ζ = 1.0 (critically damped).
//
// THE TILT CLAMP IS NOT OPTIONAL. The small-angle inversion above degrades as
// tilt grows, and an unclamped large position error would command a flip — the
// vehicle would tip past horizontal, lose its lift, and fall while accelerating
// the wrong way. Clamping is what makes large errors CONVERGE, and every real
// autopilot does it.

/// Outer-loop horizontal position controller producing a tilt reference.
#[derive(Clone, Copy, Debug)]
pub struct PositionLoop
{
	/// Tilt→horizontal-acceleration gain b_xy = g (m/s² per radian).
	b_xy: f64,
	/// Closed-loop bandwidth (rad/s) and damping ratio.
	omega_n: f64,
	zeta: f64,
	/// Bound on |roll_ref| and |pitch_ref| (radians).
	max_tilt: f64,
}

impl PositionLoop
{
	/// Derive from the plant. Only gravity enters — see the b_xy derivation above.
	pub fn from_plant(gravity: f64, omega_n: f64, zeta: f64, max_tilt: f64) -> Result<Self, String>
	{
		if !(gravity > 0.0) || !gravity.is_finite()
		{
			return Err(format!(
				"PositionLoop: gravity must be positive-finite, got {gravity}"
			));
		}
		if !(omega_n > 0.0) || !(zeta > 0.0)
		{
			return Err(format!(
				"PositionLoop: omega_n/zeta must be positive, got {omega_n}/{zeta}"
			));
		}
		if !(max_tilt > 0.0) || max_tilt >= std::f64::consts::FRAC_PI_2
		{
			return Err(format!(
				"PositionLoop: max_tilt must be in (0, π/2) rad — at or past horizontal the \
				 vehicle has no lift left to tip, got {max_tilt}"
			));
		}
		Ok(Self {
			b_xy: gravity,
			omega_n,
			zeta,
			max_tilt,
		})
	}

	/// Desired horizontal acceleration for one axis, from its error and velocity.
	#[inline]
	fn accel_des(&self, err: f64, vel: f64) -> f64
	{
		self.omega_n * self.omega_n * err - 2.0 * self.zeta * self.omega_n * vel
	}

	/// (roll_ref, pitch_ref) in radians for this step.
	///
	/// `err_x`/`err_y` are target − position (positive ⇒ move that way) and
	/// `vx`/`vy` are the world-frame velocities. The sign asymmetry is the
	/// geometry, not a bug: +pitch tips the thrust toward +x, while +roll tips
	/// it toward −y, so the y channel carries the minus.
	#[inline]
	pub fn tilt_ref(&self, err_x: f64, vx: f64, err_y: f64, vy: f64) -> (f64, f64)
	{
		let pitch = (self.accel_des(err_x, vx) / self.b_xy).clamp(-self.max_tilt, self.max_tilt);
		let roll = (-self.accel_des(err_y, vy) / self.b_xy).clamp(-self.max_tilt, self.max_tilt);
		(roll, pitch)
	}
}

#[cfg(test)]
mod tests
{
	use super::*;

	/// Defaults from the design record: ωn = 1.0 (a decade under the attitude
	/// loop, half the altitude loop), critically damped, 30° tilt limit.
	fn default_loop() -> PositionLoop
	{
		PositionLoop::from_plant(9.81, 1.0, 1.0, 30.0_f64.to_radians()).unwrap()
	}

	/// On target and stationary ⇒ level. Hover must be the fixed point, or the
	/// teacher drifts away from a position it has already reached.
	#[test]
	fn on_target_commands_level()
	{
		let (roll, pitch) = default_loop().tilt_ref(0.0, 0.0, 0.0, 0.0);
		assert_eq!((roll, pitch), (0.0, 0.0));
	}

	/// Signs, per axis. Getting one wrong makes the loop positive feedback and
	/// the vehicle accelerates away from the setpoint.
	#[test]
	fn signs_move_toward_the_target()
	{
		let p = default_loop();
		// Target ahead (+x) ⇒ pitch forward (+); behind ⇒ pitch back.
		assert!(
			p.tilt_ref(1.0, 0.0, 0.0, 0.0).1 > 0.0,
			"+x error must pitch +"
		);
		assert!(
			p.tilt_ref(-1.0, 0.0, 0.0, 0.0).1 < 0.0,
			"−x error must pitch −"
		);
		// Target to +y ⇒ roll NEGATIVE (+roll tips thrust toward −y).
		assert!(
			p.tilt_ref(0.0, 0.0, 1.0, 0.0).0 < 0.0,
			"+y error must roll −"
		);
		assert!(
			p.tilt_ref(0.0, 0.0, -1.0, 0.0).0 > 0.0,
			"−y error must roll +"
		);
		// Velocity damps: moving +x with no error must pitch back to arrest it.
		assert!(
			p.tilt_ref(0.0, 1.0, 0.0, 0.0).1 < 0.0,
			"+x velocity must be damped"
		);
		assert!(
			p.tilt_ref(0.0, 0.0, 0.0, 1.0).0 > 0.0,
			"+y velocity must be damped"
		);
	}

	/// THE DERIVATION, not just the sign: the commanded tilt must produce the
	/// acceleration the control law asked for, through the plant's own g·sinθ.
	/// This is the same relation `stage2_horizontal_inert_without_tilt_and_
	/// authoritative_with_it` pins on the SIM side — if they ever disagree, the
	/// teacher is inverting a plant the simulator does not have.
	#[test]
	fn tilt_realizes_the_requested_acceleration()
	{
		let g = 9.81_f64;
		// Wide clamp so this test measures the law, not the limiter.
		let p = PositionLoop::from_plant(g, 1.0, 1.0, 1.4).unwrap();
		let (err_x, vx) = (0.30, -0.05);
		let want_ax = 1.0 * 1.0 * err_x - 2.0 * 1.0 * 1.0 * vx;
		let (_, pitch) = p.tilt_ref(err_x, vx, 0.0, 0.0);
		// The sim accelerates at g·sin(pitch) (small-angle: ≈ g·pitch).
		let got_ax = g * pitch.sin();
		assert!(
			(got_ax - want_ax).abs() < 0.02 * want_ax.abs(),
			"commanded tilt must realize the requested accel: wanted {want_ax:.4} m/s², \
			 got {got_ax:.4} through g·sin({pitch:.4})"
		);
	}

	/// A large error must saturate at max_tilt, NOT command a flip. Unclamped,
	/// a 50 m error would ask for ~2.5 rad of tilt — past horizontal, where the
	/// vehicle has no lift and the small-angle inversion is meaningless.
	#[test]
	fn large_error_saturates_instead_of_flipping()
	{
		let p = default_loop();
		let limit = 30.0_f64.to_radians();
		let (roll, pitch) = p.tilt_ref(50.0, 0.0, -50.0, 0.0);
		assert!(
			(pitch - limit).abs() < 1e-12,
			"pitch must saturate at +30°, got {pitch}"
		);
		assert!(
			(roll - limit).abs() < 1e-12,
			"roll must saturate at +30°, got {roll}"
		);
	}

	/// A max_tilt at or past horizontal is a construction error, not something to
	/// silently accept: there is no lift left to tip.
	#[test]
	fn refuses_a_degenerate_tilt_limit()
	{
		assert!(PositionLoop::from_plant(9.81, 1.0, 1.0, std::f64::consts::FRAC_PI_2).is_err());
		assert!(PositionLoop::from_plant(9.81, 1.0, 1.0, 0.0).is_err());
		assert!(PositionLoop::from_plant(0.0, 1.0, 1.0, 0.5).is_err());
		assert!(PositionLoop::from_plant(9.81, 0.0, 1.0, 0.5).is_err());
	}
}
