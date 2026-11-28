//! CTRL-83 (ABI 66): the out-of-distribution stress options as ONE value carried by
//! Stage1Cfg — the design-model override (D1, design_model.rs) and the mid-episode
//! payload step (D3, payload_step.rs).
//!
//! One value, built once from the scorer kwargs by `from_kwargs`, so the WNN scorers (CPU
//! and Metal) and the classical rivals receive the SAME stress and cannot drift (the
//! stage-1 trainer-gap pattern). OFF ⇒ every rollout is the ABI-65 rollout, bit-identical
//! (pinned by ctrl83_flag_off_pins). Scorer-only: the trainer never reads it — a stress
//! panel scores published controllers, it does not train them.

use crate::actuator_spec::ActuatorSpec;
use crate::design_model::DesignModel;
use crate::payload_step::PayloadStep;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StressOpts
{
	pub design: DesignModel,
	pub payload: PayloadStep,
}

impl StressOpts
{
	pub const OFF: StressOpts = StressOpts {
		design: DesignModel::MATCHED,
		payload: PayloadStep::OFF,
	};

	/// The scorer kwargs → the validated stress for a plant actuator and episode length.
	pub fn from_kwargs(kw: StressKwargs, plant: ActuatorSpec, dt: f32, steps: usize) -> Result<Self, String>
	{
		Ok(StressOpts {
			design: DesignModel::new(kw.model_motor_lag_s, plant)?,
			payload: PayloadStep::from_kwargs(kw.payload_step_s, kw.payload_mass_scale, dt, steps)?,
		})
	}

	/// Both stresses ride the translating plant (the payload is a mass event; the design
	/// override is defined against the real-life actuator): refused without translation,
	/// where there is no Stage1Cfg to carry them.
	pub fn require_translation(&self, translation: bool, site: &str) -> Result<(), String>
	{
		if *self != Self::OFF && !translation
		{
			return Err(format!("{site}: model_motor_lag_s / payload_step_s need translation=True"));
		}
		Ok(())
	}
}

/// The three scorer kwargs as they arrive from Python (unvalidated) — the one spelling
/// every host (WNN CPU / Metal scorers, the classical rivals) forwards.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StressKwargs
{
	pub model_motor_lag_s: Option<f32>,
	pub payload_step_s: Option<f32>,
	pub payload_mass_scale: f32,
}

impl StressKwargs
{
	#[cfg_attr(not(test), allow(dead_code))] // the Rust callers that pass OFF are tests
	pub const OFF: StressKwargs = StressKwargs {
		model_motor_lag_s: None,
		payload_step_s: None,
		payload_mass_scale: 1.0,
	};
}
