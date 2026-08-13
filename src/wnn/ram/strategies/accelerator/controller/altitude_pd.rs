// Altitude PD — the collective channel bolted on top of an attitude teacher
// (scope C stage 1, item 4; docs/scope_c_full_controller_spec.md).
//
// WHY THIS IS HONEST: this makes the classical teacher an explicit CASCADE —
// an outer altitude loop handing a collective to an inner attitude loop —
// which is exactly what the classical rivals (and every real autopilot) are.
// It must be DISCLOSED as such wherever the teacher is reported, never sold as
// a monolithic controller.
//
// THE GAINS ARE DERIVED, NOT GUESSED. Linearize the stage-1 vertical dynamics
// about hover:
//     v̇z = (ΣTᵢ·cosθ)/m − g,     Tᵢ = k_thrust·pwmᵢ²
// At hover cosθ ≈ 1 and every motor sits at pwm_h = √(mg / 4k). A uniform
// collective nudge δ on all four motors gives
//     ∂v̇z/∂δ = 4 · 2·k_thrust·pwm_h / m  =  8·k_thrust·pwm_h / m  ≜ b_z
// so commanding a desired vertical acceleration costs δ = az_des / b_z.
// Choosing the closed loop by its SHAPE rather than by raw gains,
//     az_des = ωn²·alt_err − 2ζωn·vz
// leaves exactly two interpretable knobs: ωn (bandwidth, rad/s) and ζ
// (damping). Defaults ωn = 2.0, ζ = 1.0 (critically damped) — an altitude loop
// an order of magnitude slower than the attitude loop, which is the standard
// cascade separation and keeps the two loops from arguing.
//
// The correction is applied UNIFORMLY to all four motors, so every motor
// DIFFERENCE — and therefore the body torque the '+' mixer produces — is
// unchanged: the attitude teacher's behaviour is untouched by construction
// (pinned by uniform_correction_preserves_torque_differences).

/// Outer-loop altitude PD producing a collective PWM correction.
#[derive(Clone, Copy, Debug)]
pub struct AltitudePd {
	/// Collective→vertical-acceleration gain b_z (m/s² per unit PWM).
	b_z: f64,
	/// Closed-loop bandwidth (rad/s) and damping ratio.
	omega_n: f64,
	zeta: f64,
	/// Bound on |δ| so the outer loop can never eat the inner loop's authority.
	max_delta: f64,
}

impl AltitudePd {
	/// Derive from the plant. `hover_pwm` is √(mg/4k) — pass the sim's own
	/// value so the two never drift apart.
	pub fn from_plant(mass: f64, gravity: f64, k_thrust: f64, omega_n: f64, zeta: f64,
	                  max_delta: f64) -> Result<Self, String> {
		if !(mass > 0.0) || !(k_thrust > 0.0) || !(gravity > 0.0) {
			return Err(format!(
				"AltitudePd: mass/gravity/k_thrust must be positive, got {mass}/{gravity}/{k_thrust}"));
		}
		let hover_pwm = (mass * gravity / (4.0 * k_thrust)).sqrt();
		let b_z = 8.0 * k_thrust * hover_pwm / mass;
		if !(b_z.is_finite() && b_z > 0.0) {
			return Err(format!("AltitudePd: derived b_z is not positive-finite ({b_z})"));
		}
		Ok(Self { b_z, omega_n, zeta, max_delta })
	}

	/// Per-motor collective correction δ for this step.
	/// `alt_err` = target − z (positive ⇒ climb), `vz` = +up.
	#[inline]
	pub fn delta(&self, alt_err: f64, vz: f64) -> f64 {
		let az_des = self.omega_n * self.omega_n * alt_err - 2.0 * self.zeta * self.omega_n * vz;
		(az_des / self.b_z).clamp(-self.max_delta, self.max_delta)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// cf21_brushless numbers: mass 0.0393 kg, k_thrust 0.2 N/pwm², g 9.81.
	fn cf21() -> AltitudePd {
		AltitudePd::from_plant(0.0393, 9.81, 0.2, 2.0, 1.0, 0.25).unwrap()
	}

	/// Zero error, zero velocity ⇒ no correction (hover is the fixed point).
	#[test]
	fn at_hover_the_correction_is_zero() {
		assert_eq!(cf21().delta(0.0, 0.0), 0.0);
	}

	/// Signs must be right or the loop is positive feedback: below target
	/// (alt_err > 0) ⇒ push UP; climbing (vz > 0) ⇒ damp DOWN.
	#[test]
	fn signs_push_toward_the_target() {
		let pd = cf21();
		assert!(pd.delta(0.1, 0.0) > 0.0, "below target must command more thrust");
		assert!(pd.delta(-0.1, 0.0) < 0.0, "above target must command less thrust");
		assert!(pd.delta(0.0, 0.5) < 0.0, "climbing must be damped");
		assert!(pd.delta(0.0, -0.5) > 0.0, "sinking must be arrested");
	}

	/// The DERIVATION, not just the sign: commanding δ must produce the vertical
	/// acceleration the PD law asked for, through the plant's own b_z.
	#[test]
	fn delta_realizes_the_requested_acceleration() {
		let (mass, g, k) = (0.0393f64, 9.81f64, 0.2f64);
		let pd = AltitudePd::from_plant(mass, g, k, 2.0, 1.0, 10.0).unwrap();  // no clamp
		let (alt_err, vz) = (0.05, -0.1);
		let want_az = 2.0 * 2.0 * alt_err - 2.0 * 1.0 * 2.0 * vz;
		let d = pd.delta(alt_err, vz);
		// Apply δ to the real quadratic thrust model and read back the accel.
		let hover = (mass * g / (4.0 * k)).sqrt();
		let pwm = hover + d;
		let got_az = 4.0 * k * pwm * pwm / mass - g;
		assert!((got_az - want_az).abs() < 0.05 * want_az.abs().max(1e-3),
			"δ did not realize the requested accel: want {want_az:.4} m/s², got {got_az:.4}");
	}

	/// The clamp binds symmetrically — the outer loop can never eat the inner
	/// loop's control authority.
	#[test]
	fn delta_is_clamped_both_ways() {
		let pd = cf21();
		assert_eq!(pd.delta(1000.0, 0.0), 0.25);
		assert_eq!(pd.delta(-1000.0, 0.0), -0.25);
	}

	/// A UNIFORM correction leaves every motor difference — hence the body
	/// torque of the '+' mixer — exactly unchanged. This is what lets the
	/// collective ride on top of an attitude teacher without perturbing it.
	#[test]
	fn uniform_correction_preserves_torque_differences() {
		let base = [0.62f64, 0.71, 0.68, 0.66];
		let d = cf21().delta(0.08, -0.2);
		let out: Vec<f64> = base.iter().map(|p| p + d).collect();
		for (i, j) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
			assert!(((out[i] - out[j]) - (base[i] - base[j])).abs() < 1e-12,
				"collective changed the {i}-{j} motor difference — it would twist the airframe");
		}
	}

	/// A degenerate plant is refused rather than silently producing NaN gains.
	#[test]
	fn refuses_a_degenerate_plant() {
		assert!(AltitudePd::from_plant(0.0, 9.81, 0.2, 2.0, 1.0, 0.25).is_err());
		assert!(AltitudePd::from_plant(0.04, 9.81, 0.0, 2.0, 1.0, 0.25).is_err());
	}
}
