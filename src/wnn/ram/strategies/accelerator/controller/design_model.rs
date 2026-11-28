//! CTRL-83 D1 (ABI 66): the controllers' DESIGN-TIME knowledge of the actuator, separate
//! from the plant they fly.
//!
//! Until ABI 65 one number, `motor_lag_s`, armed the plant AND every model of it: MPCOF /
//! LQI / LQR / MPC were re-derived on it (Teacher::from_id_lagged, CtbrMpcOf), the MPCOF
//! observer's actuator twin tracked it, and the WNN's d̂ / obs_pwm twin was re-armed with
//! it (set_observer_actuator_core(sim.actuator())). A stress cell that changed the lag
//! therefore re-designed the model-based rivals for the stressed aircraft, while the WNN's
//! cells and PID's gains stayed fixed — the rivals were TOLD the perturbation.
//!
//! A `DesignModel` with `motor_lag_s = Some(T0)` holds every model at the design value T0
//! while the plant flies its own lag: the rivals' designs and twins are built on T0, the
//! WNN's actuator twin is armed with T0. `None` (MATCHED) is the ABI-65 behaviour — every
//! model is the plant — and so is `Some(T)` with T equal to the plant's lag (the model is
//! then built from the same number), which is what keeps the in-range cell bit-identical.
//!
//! Only the lag needs this: no rival re-derives from motor wear, sensor noise, rotor drag
//! or the mass draw (the altitude PD and the hover are built on the NOMINAL mass, the
//! Mahony gains are fixed, nothing reads the wear vector or k_d) — audited for CTRL-83.
//!
//! Scope: a SYMMETRIC, floor-free plant actuator (the ABI-64 actuator every cf21 lineage
//! flies). The CTRL-86 extended actuator (asymmetric lag / idle floor) has no single
//! design T a model could be held at, so a design override on it is refused. Controller-
//! only physics: nothing here is shared with the IDS worker (not ram_core).

use crate::actuator_spec::ActuatorSpec;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DesignModel
{
	/// The motor-lag settling time T (s) every controller's model is built on; None ⇒ the
	/// plant's own (matched, ABI 65).
	pub motor_lag_s: Option<f32>,
}

impl DesignModel
{
	pub const MATCHED: DesignModel = DesignModel { motor_lag_s: None };

	/// Validated against the plant's actuator: finite T >= 0, and a symmetric floor-free
	/// plant when an override is given.
	pub fn new(motor_lag_s: Option<f32>, plant: ActuatorSpec) -> Result<Self, String>
	{
		let Some(t) = motor_lag_s
		else
		{
			return Ok(Self::MATCHED);
		};
		if !(t.is_finite() && t >= 0.0)
		{
			return Err(format!("model_motor_lag_s = {t} must be a finite settling time >= 0 s"));
		}
		if plant.extended()
		{
			return Err(
				"model_motor_lag_s: the plant actuator is extended (motor_lag_down_s / pwm_min) — \
				 there is no single design lag to hold the models at"
					.into(),
			);
		}
		Ok(DesignModel { motor_lag_s: Some(t) })
	}

	/// The actuator every controller's model is built on, for this plant actuator.
	pub fn actuator(&self, plant: ActuatorSpec) -> ActuatorSpec
	{
		match self.motor_lag_s
		{
			None => plant,
			Some(t) => ActuatorSpec::symmetric(t),
		}
	}

	/// The design lag for a plant spin-up time (the plant's own when matched).
	pub fn lag_s(&self, plant_lag_s: f32) -> f32
	{
		self.motor_lag_s.unwrap_or(plant_lag_s)
	}
}

#[cfg(test)]
mod tests
{
	use super::*;

	#[test]
	fn matched_is_the_plant_and_override_is_symmetric()
	{
		let plant = ActuatorSpec::symmetric(0.30);
		assert_eq!(DesignModel::MATCHED.actuator(plant), plant);
		let d = DesignModel::new(Some(0.20), plant).unwrap();
		assert_eq!(d.actuator(plant), ActuatorSpec::symmetric(0.20));
		assert_eq!(d.lag_s(0.30), 0.20);
		// The in-range cell: the override at the plant's own value IS the plant.
		assert_eq!(DesignModel::new(Some(0.30), plant).unwrap().actuator(plant), plant);
	}

	#[test]
	fn refuses_bad_values_and_extended_plants()
	{
		let plant = ActuatorSpec::symmetric(0.20);
		assert!(DesignModel::new(Some(f32::NAN), plant).is_err());
		assert!(DesignModel::new(Some(-0.1), plant).is_err());
		let ext = ActuatorSpec::new(0.05, 0.10, 0.0).unwrap();
		assert!(DesignModel::new(Some(0.2), ext).is_err());
		assert_eq!(DesignModel::new(None, ext).unwrap(), DesignModel::MATCHED);
	}
}
