//! CTRL-86 S1 + S2 (ABI 65): the plant's ACTUATOR as one value — the motor lag's
//! up and down 2% settling times and the armed-idle pwm floor.
//!
//! S1, ASYMMETRIC LAG. A real ESC + rotor spins up faster than it spins down (the
//! gz MulticopterMotorModel the PX4 x500 SITL flies: timeConstantUp 0.0125 s,
//! timeConstantDown 0.025 s). Per motor per step the plant picks the coefficient by
//! the sign of the command step:
//!     T = (cmd > filt) ? T_up : T_down,   filt ← filt + 4dt/max(T, 4dt)·(cmd − filt)
//! — Molchanov eq. (7) (AttitudeSim::apply_motor_lag) with T chosen per motor. The
//! gz filter acts on rotor VELOCITY and ours on pwm ∝ ω, so the two are the same
//! filter. `down_s = 0` means SYMMETRIC (down := up), which evaluates the exact
//! ABI-64 expression on every step: bit-identical (ctrl86_flag_off_pins).
//!
//! S2, IDLE FLOOR. An armed ESC never commands a rotor below its idle speed
//! (4001_gz_x500: SIM_GZ_EC_MIN 150 of SIM_GZ_EC_MAX 1000 rad/s ⇒ pwm 0.15). The
//! plant clamps every motor command to [pwm_min, 1] where it consumes it, BEFORE
//! the lag. `pwm_min = 0` is the ABI-64 clamp [0, 1], the same f32 operation.
//!
//! WHY A VALUE. The lag's twins (the teachers' ActuatorLag, the cascade d̂, the
//! student's actuator twin, the Metal kernel) must reproduce the plant's filter
//! exactly; up/down switching breaks the linearity the per-axis twins relied on
//! ("filter-then-unmix = unmix-then-filter"), so every twin needs the SAME three
//! numbers the plant has. One value, read from the plant, is how they get them.
//!
//! Controller-only physics: nothing here is shared with the IDS worker, so it lives
//! in the controller crate, not ram_core.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActuatorSpec
{
	/// Spin-UP 2% settling time T_up (s); 0 ⇒ lag off.
	pub lag_up_s: f32,
	/// Spin-DOWN 2% settling time T_down (s); 0 ⇒ symmetric (T_down = T_up).
	pub lag_down_s: f32,
	/// Armed-idle floor on every motor command; 0 ⇒ the legacy clamp [0, 1].
	pub pwm_min: f32,
}

impl ActuatorSpec
{
	pub const OFF: ActuatorSpec = ActuatorSpec {
		lag_up_s: 0.0,
		lag_down_s: 0.0,
		pwm_min: 0.0,
	};

	/// The ABI-64 actuator: one symmetric settling time, no floor.
	pub fn symmetric(lag_s: f32) -> Self
	{
		ActuatorSpec {
			lag_up_s: lag_s,
			..Self::OFF
		}
	}

	/// Validated constructor. `lag_down_s > 0` needs `lag_up_s > 0` (a down time on a
	/// lag-free plant would be silently inert); the floor must sit in [0, 1).
	pub fn new(lag_up_s: f32, lag_down_s: f32, pwm_min: f32) -> Result<Self, String>
	{
		for (name, v) in [("motor_lag_s", lag_up_s), ("motor_lag_down_s", lag_down_s)]
		{
			if !(v.is_finite() && v >= 0.0)
			{
				return Err(format!("{name} = {v} must be a finite settling time >= 0 s"));
			}
		}
		if lag_down_s > 0.0 && !(lag_up_s > 0.0)
		{
			return Err(format!(
				"motor_lag_down_s = {lag_down_s} needs motor_lag_s > 0: on a lag-free plant the \
				 spin-down time would be silently inert"
			));
		}
		if !(pwm_min.is_finite() && (0.0..1.0).contains(&pwm_min))
		{
			return Err(format!("pwm_min = {pwm_min} must be an idle floor in [0, 1)"));
		}
		Ok(ActuatorSpec { lag_up_s, lag_down_s, pwm_min })
	}

	pub fn lag_on(&self) -> bool
	{
		self.lag_up_s > 0.0
	}

	/// T_down as the plant uses it (T_up when symmetric).
	pub fn down_s(&self) -> f32
	{
		if self.lag_down_s > 0.0 { self.lag_down_s } else { self.lag_up_s }
	}

	pub fn asymmetric(&self) -> bool
	{
		self.lag_on() && self.down_s() != self.lag_up_s
	}

	/// True when a twin cannot use the ABI-64 per-axis / symmetric path: the lag
	/// switches on direction, or a floor clamps the command.
	pub fn extended(&self) -> bool
	{
		self.asymmetric() || self.pwm_min > 0.0
	}

	/// The ONE settling time the linear teachers design at (they need a single τ):
	/// the mean of up and down (Luiz, CTRL-86 amendment 1); T_up when symmetric.
	pub fn design_lag_s(&self) -> f32
	{
		if self.asymmetric() { 0.5 * (self.lag_up_s + self.down_s()) } else { self.lag_up_s }
	}

	/// (α_up, α_down), the plant's f32 coefficients 4dt/max(T, 4dt).
	pub fn alphas(&self, dt: f32) -> (f32, f32)
	{
		(lag_alpha(self.lag_up_s, dt), lag_alpha(self.down_s(), dt))
	}

	/// The plant's clamp of a motor command: [pwm_min, 1] (tests: the plant itself
	/// clamps in AttitudeSim::actuate).
	#[cfg(test)]
	pub fn clamp_cmd(&self, cmd: f32) -> f32
	{
		cmd.clamp(self.pwm_min, 1.0)
	}
}

/// The plant's lag coefficient for settling time `t` at step `dt` (the ABI-64 f32
/// expression; T is clamped to Molchanov's admissibility bound 4dt).
#[inline]
pub fn lag_alpha(t: f32, dt: f32) -> f32
{
	4.0 * dt / t.max(4.0 * dt)
}

/// One filter step with the coefficient picked by the command's direction. With
/// a_up == a_down this is the ABI-64 update bit for bit.
#[inline]
pub fn lag_step(filt: f32, cmd: f32, a_up: f32, a_down: f32) -> f32
{
	let a = if cmd > filt { a_up } else { a_down };
	filt + a * (cmd - filt)
}

/// f64 twin of `lag_step` for the teachers' estimates.
#[inline]
pub fn lag_step_f64(filt: f64, cmd: f64, a_up: f64, a_down: f64) -> f64
{
	let a = if cmd > filt { a_up } else { a_down };
	filt + a * (cmd - filt)
}

#[cfg(test)]
mod tests
{
	use super::*;

	#[test]
	fn validates()
	{
		assert!(ActuatorSpec::new(-0.1, 0.0, 0.0).is_err());
		assert!(ActuatorSpec::new(0.05, f32::NAN, 0.0).is_err());
		assert!(ActuatorSpec::new(0.0, 0.1, 0.0).is_err(), "down without up");
		assert!(ActuatorSpec::new(0.05, 0.1, 1.0).is_err());
		assert!(ActuatorSpec::new(0.05, 0.1, -0.01).is_err());
		let x = ActuatorSpec::new(0.05, 0.10, 0.15).unwrap();
		assert!(x.asymmetric() && x.extended());
		assert_eq!(x.design_lag_s(), 0.075);
	}

	#[test]
	fn symmetric_spellings_are_the_legacy_actuator()
	{
		for s in [ActuatorSpec::symmetric(0.2), ActuatorSpec::new(0.2, 0.0, 0.0).unwrap(), ActuatorSpec::new(0.2, 0.2, 0.0).unwrap()]
		{
			assert!(!s.asymmetric() && !s.extended());
			assert_eq!(s.design_lag_s(), 0.2);
			let (u, d) = s.alphas(0.001);
			assert_eq!(u.to_bits(), d.to_bits());
			assert_eq!(u.to_bits(), (4.0f32 * 0.001 / 0.2f32.max(0.004)).to_bits());
		}
		assert!(!ActuatorSpec::OFF.lag_on() && !ActuatorSpec::OFF.extended());
		assert_eq!(ActuatorSpec::OFF.clamp_cmd(-0.3), 0.0);
		assert_eq!(ActuatorSpec::OFF.clamp_cmd(1.3), 1.0);
	}

	/// The plan's hand-computed step (CTRL-86 S1 tests): from 0.5, cmd 1.0 for 10 steps
	/// at dt 1 ms with T_up 0.05 gives 0.5 + 0.5·(1 − (1 − 0.08)^10); then cmd 0.0 uses
	/// the down coefficient 0.04 (T_down 0.10).
	#[test]
	fn hand_computed_up_then_down()
	{
		let s = ActuatorSpec::new(0.05, 0.10, 0.0).unwrap();
		let (au, ad) = s.alphas(0.001);
		assert!((au - 0.08).abs() < 1e-7 && (ad - 0.04).abs() < 1e-7, "{au} {ad}");
		let mut f = 0.5f32;
		for _ in 0..10
		{
			f = lag_step(f, 1.0, au, ad);
		}
		let want = 0.5 + 0.5 * (1.0 - (1.0f64 - 0.08).powi(10));
		assert!((f as f64 - want).abs() < 1e-6, "up: {f} vs {want}");
		let g = lag_step(f, 0.0, au, ad);
		assert!(((g as f64) - (f as f64) * (1.0 - 0.04)).abs() < 1e-6, "down step {g}");
	}
}
