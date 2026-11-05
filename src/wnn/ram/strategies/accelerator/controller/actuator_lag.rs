//! ABI 34 (05/11/2026): the teachers' estimate of the LAGGED actuator state.
//!
//! Under `--motor-lag-s T` the plant filters every motor command through
//! Molchanov eq. (7), u' ← u' + (4dt/T)(u − u') (AttitudeSim::apply_motor_lag;
//! τ = T/4). A teacher that plans as if u reached the rotors instantly fights
//! its own actuator, and the mpcof observer reads the lag as a disturbance. The
//! lag-aware teachers (optimal.rs, ids 1-4) carry this estimate as the extra
//! state of their augmented design model and the mpcof observer subtracts
//! b·(this) instead of b·u_commanded.
//!
//! It is reconstructed exactly the way the plant builds it: filter the pwm the
//! teacher is TOLD was applied (observe(), the commanded convention — the same
//! value the plant's filter consumes) through the same coefficient. Per axis,
//! after the '+'-mixer inverse: the filter is linear, so filtering the four
//! motors then un-mixing equals un-mixing then filtering.
//!
//! The DAgger observe protocol calls observe() once at t = 0 with a hover
//! placeholder BEFORE anything has flown (every host: dagger_train,
//! position_score, calib_sampler, record_ops). That first call only PRIMES the
//! estimate. The second call carries the first real command and SEEDS it — the
//! plant seeds its filter at the first command too (no spin-up transient), so
//! from then on the two filters hold the same value.

/// The plant's lag coefficient for settling time `settling_s` at step `dt` —
/// the SAME f32 expression as AttitudeSim::apply_motor_lag (T clamped to ≥ 4dt,
/// Molchanov's admissibility condition), widened to f64 for the teacher math.
pub(crate) fn motor_lag_alpha(settling_s: f32, dt: f32) -> f64
{
	let t_eff = settling_s.max(4.0 * dt);
	(4.0 * dt / t_eff) as f64
}

/// '+'-mixer inverse (the teachers' mix_to_motors_f64):
/// u_roll = (m3 − m1)/2, u_pitch = (m2 − m0)/2, u_yaw = ((m0 + m2) − (m1 + m3))/4.
pub(crate) fn unmix_plus(m: [f64; 4]) -> [f64; 3]
{
	[
		(m[3] - m[1]) * 0.5,
		(m[2] - m[0]) * 0.5,
		((m[0] + m[2]) - (m[1] + m[3])) * 0.25,
	]
}

/// Per-axis estimate of the plant's filtered (lagged) control, in the teachers'
/// per-axis u units.
#[derive(Clone)]
pub(crate) struct ActuatorLag
{
	alpha: f64,
	ua: [f64; 3],
	primed: bool,
	seeded: bool,
}

impl ActuatorLag
{
	pub(crate) fn new(settling_s: f32, dt: f32) -> Self
	{
		ActuatorLag {
			alpha: motor_lag_alpha(settling_s, dt),
			ua: [0.0; 3],
			primed: false,
			seeded: false,
		}
	}

	/// Episode boundary (Teacher::reset) — the plant re-seeds its filter too.
	pub(crate) fn reset(&mut self)
	{
		self.ua = [0.0; 3];
		self.primed = false;
		self.seeded = false;
	}

	/// The pwm that flew last step (commanded convention). First call after
	/// reset = the t=0 placeholder: prime only. Second = seed. Then filter.
	pub(crate) fn observe(&mut self, applied_pwm: [f64; 4])
	{
		if !self.primed
		{
			self.primed = true;
			return;
		}
		let u = unmix_plus(applied_pwm);
		if !self.seeded
		{
			self.ua = u;
			self.seeded = true;
			return;
		}
		for axis in 0..3
		{
			self.ua[axis] += self.alpha * (u[axis] - self.ua[axis]);
		}
	}

	/// The estimate after the last observe() (0 before anything has flown).
	pub(crate) fn state(&self) -> [f64; 3]
	{
		self.ua
	}

	/// False until the first real command has been observed — i.e. on the
	/// episode's first decision, whose command the PLANT will seed its filter
	/// with (AttitudeSim::apply_motor_lag): that one command acts unlagged.
	pub(crate) fn seeded(&self) -> bool
	{
		self.seeded
	}
}

#[cfg(test)]
mod tests
{
	use super::*;
	use crate::controller::AttitudeSim;

	/// The estimate tracks the PLANT's own filter exactly: drive a lagged sim with
	/// a varying command, feed the estimator the observe() protocol, and the
	/// estimate must equal the un-mixed filter output the sim applied.
	#[test]
	fn estimate_matches_the_plant_filter()
	{
		let (tt, dt) = (0.15f32, 0.001f32);
		let mut sim = AttitudeSim::new(dt, 0.075, 2.4, 0.05, [0.0023, 0.0023, 0.0046], 9.81);
		sim.set_motor_lag(tt);
		let mut est = ActuatorLag::new(tt, dt);
		est.observe([0.5; 4]); // t=0 placeholder
		// Twin of the plant's filter in f32, the reference.
		let alpha = 4.0 * dt / tt.max(4.0 * dt);
		let mut filt = [0.0f32; 4];
		for t in 0..300usize
		{
			let w = (t as f32) * 0.03;
			let cmd = [0.5 + 0.1 * w.sin(), 0.5 - 0.05 * w.cos(), 0.55, 0.45 + 0.02 * (2.0 * w).sin()];
			sim.step(cmd);
			for m in 0..4
			{
				filt[m] = if t == 0 { cmd[m] } else { filt[m] + alpha * (cmd[m] - filt[m]) };
			}
			est.observe(cmd.map(|v| v as f64));
			let want = unmix_plus(filt.map(|v| v as f64));
			for axis in 0..3
			{
				assert!(
					(est.state()[axis] - want[axis]).abs() < 1e-5,
					"t={t} axis {axis}: estimate {} vs plant {}",
					est.state()[axis],
					want[axis]
				);
			}
		}
	}

	#[test]
	fn alpha_is_the_plant_coefficient()
	{
		assert_eq!(motor_lag_alpha(0.15, 0.001), (4.0f32 * 0.001 / 0.15) as f64);
		// Below 4dt the plant clamps T to 4dt ⇒ alpha = 1 (no lag).
		assert_eq!(motor_lag_alpha(0.001, 0.001), 1.0);
	}
}
