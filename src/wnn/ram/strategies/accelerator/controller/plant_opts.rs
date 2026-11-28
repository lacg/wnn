// PlantOpts — the ABI-32 "real-life" plant options as ONE value (31/10/2026).
//
// Luiz's option A for stage 2: under translation a free-flying IMU reads
// SPECIFIC FORCE (thrust + drag over mass, body frame), not the ideal
// inclinometer −g_body, and a real quad feels a linear rotor-drag force on its
// horizontal body velocity. Both live on AttitudeSim (set_imu_specific_force /
// set_rotor_drag); this struct is how every host that BUILDS a sim receives
// them — the trainer (via RewardGatedConfigPacked → AirframeRs), both batch
// scorers (via Stage1Cfg), the classical baselines, the position scorer, the
// threshold-calibration sampler and the MEMORY recorder. One value, one
// `apply`, so a host cannot arm one option and forget the other (the stage-1
// trainer-gap pattern).
//
// OFF (the default everywhere in Rust) never calls a setter, so every
// ABI ≤ 31 rollout reproduces bit-identically. The real-life defaults live in
// the Python CLI layer (phased_ga), not here.
//
// Both options are only meaningful WITH translation (the specific force is the
// translational thrust; drag acts on the translational velocity). A host that
// receives them without translation refuses loudly (`require_translation`).

use crate::controller::AttitudeSim;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlantOpts
{
	/// Accelerometer reads specific force while translation is on.
	pub imu_specific_force: bool,
	/// Linear rotor-drag coefficient k_d (1/s) on the horizontal body velocity.
	pub rotor_drag: f32,
	/// CTRL-86 S1 (ABI 65): the motor lag's SPIN-DOWN settling time (s); 0 ⇒
	/// symmetric. The spin-up time stays the host's `motor_lag_s`.
	pub motor_lag_down_s: f32,
	/// CTRL-86 S2 (ABI 65): the armed-idle floor of every motor command; 0 ⇒ none.
	pub pwm_min: f32,
}

impl PlantOpts
{
	pub const OFF: PlantOpts = PlantOpts {
		imu_specific_force: false,
		rotor_drag: 0.0,
		motor_lag_down_s: 0.0,
		pwm_min: 0.0,
	};

	/// Validated constructor: k_d must be finite and >= 0.
	pub fn new(imu_specific_force: bool, rotor_drag: f32) -> Result<Self, String>
	{
		if !rotor_drag.is_finite() || rotor_drag < 0.0
		{
			return Err(format!(
				"rotor_drag must be finite and >= 0 (1/s), got {rotor_drag}"
			));
		}
		Ok(PlantOpts {
			imu_specific_force,
			rotor_drag,
			..Self::OFF
		})
	}

	/// CTRL-86 S1/S2: the same options with the actuator extras (validated against
	/// the host's spin-up time `motor_lag_s` by ActuatorSpec::new). (0, 0) is `self`.
	pub fn with_actuator(self, motor_lag_s: f32, motor_lag_down_s: f32, pwm_min: f32) -> Result<Self, String>
	{
		crate::actuator_spec::ActuatorSpec::new(motor_lag_s, motor_lag_down_s, pwm_min)?;
		Ok(PlantOpts {
			motor_lag_down_s,
			pwm_min,
			..self
		})
	}

	/// The plant's actuator for spin-up time `motor_lag_s`.
	pub fn actuator(&self, motor_lag_s: f32) -> crate::actuator_spec::ActuatorSpec
	{
		crate::actuator_spec::ActuatorSpec {
			lag_up_s: motor_lag_s,
			lag_down_s: self.motor_lag_down_s,
			pwm_min: self.pwm_min,
		}
	}

	pub fn is_off(&self) -> bool
	{
		*self == Self::OFF
	}

	/// Refuse the options on a host that is not flying translation — the IMU and
	/// drag would be silently inert there (the sim gates both on translation). The
	/// CTRL-86 actuator extras are part of the same real-life plant: the attitude-only
	/// scorers carry no PlantOpts at all, so they are refused here too rather than
	/// dropped on those paths.
	pub fn require_translation(&self, translation: bool, site: &str) -> Result<(), String>
	{
		if !self.is_off() && !translation
		{
			return Err(format!(
				"{site}: imu_specific_force / rotor_drag / motor_lag_down_s / pwm_min need \
				 translation=True — they belong to the real-life plant, which flies translation"
			));
		}
		Ok(())
	}

	/// Arm the options on a sim. OFF calls nothing (bit-identical legacy sim).
	/// The settings persist across `reset`, like the motor lag.
	pub fn apply(&self, sim: &mut AttitudeSim)
	{
		if self.imu_specific_force
		{
			sim.set_imu_specific_force(true);
		}
		if self.rotor_drag != 0.0
		{
			sim.set_rotor_drag_core(self.rotor_drag);
		}
		if self.motor_lag_down_s != 0.0 || self.pwm_min != 0.0
		{
			sim.set_actuator_extras_core(self.motor_lag_down_s, self.pwm_min);
		}
	}
}

#[cfg(test)]
mod tests
{
	use super::*;

	#[test]
	fn validates_and_requires_translation()
	{
		assert!(PlantOpts::new(false, -0.1).is_err());
		assert!(PlantOpts::new(false, f32::NAN).is_err());
		let on = PlantOpts::new(true, 0.3).unwrap();
		assert!(on.require_translation(false, "t").is_err());
		assert!(on.require_translation(true, "t").is_ok());
		assert!(PlantOpts::OFF.require_translation(false, "t").is_ok());
	}

	#[test]
	fn actuator_extras_validate_and_need_translation()
	{
		let p = PlantOpts::new(true, 0.12).unwrap();
		assert!(p.with_actuator(0.0, 0.1, 0.0).is_err(), "down without up");
		assert!(p.with_actuator(0.05, 0.1, 1.0).is_err(), "floor >= 1");
		let x = PlantOpts::OFF.with_actuator(0.05, 0.10, 0.15).unwrap();
		assert!(x.require_translation(false, "t").is_err());
		assert_eq!(x.actuator(0.05), crate::actuator_spec::ActuatorSpec::new(0.05, 0.10, 0.15).unwrap());
		assert_eq!(p.with_actuator(0.2, 0.0, 0.0).unwrap(), p, "(0, 0) is the ABI-64 plant");
	}
}
