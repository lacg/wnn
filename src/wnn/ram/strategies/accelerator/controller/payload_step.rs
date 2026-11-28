//! CTRL-83 D3 (ABI 66): a MID-EPISODE PAYLOAD STEP — a payload added (scale > 1) or
//! dropped (scale < 1) at one physical step of every episode.
//!
//! From step `step` on, the plant's mass is the episode's draw × `mass_scale`. The event
//! is a PLANT event: no controller is told (the WNN's collective anchor, the rivals'
//! hover and altitude-PD gains all keep what they were built with at episode start), so
//! every controller must notice it the way a real flyer does — it starts sinking.
//!
//! INERTIA IS NOT CHANGED. The payload is modelled as a point mass at the centre of mass:
//! its rotational inertia about the CoM is zero, so only the translational dynamics
//! (thrust / mass) see it. A payload hung off-centre would add inertia AND a CoM offset
//! (a constant torque); that is a different stress, not modelled here.
//!
//! OFF (step = u32::MAX) never fires, so every existing rollout is bit-identical. The
//! same value reaches the CPU scorer, the Metal kernel (Params.payload_step_idx /
//! payload_mass_scale) and the classical rivals, which is what makes the event identical
//! for every controller. Controller-only physics: nothing here is shared with the IDS
//! worker, so it lives in the controller crate, not ram_core.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PayloadStep
{
	/// The physical step (dt units) at whose START the mass changes; u32::MAX ⇒ off.
	pub step: u32,
	/// New mass = the episode's draw × this.
	pub mass_scale: f32,
}

impl PayloadStep
{
	pub const OFF: PayloadStep = PayloadStep {
		step: u32::MAX,
		mass_scale: 1.0,
	};

	/// The host kwargs (payload_step_s, payload_mass_scale) → the event. `None` ⇒ OFF and
	/// the scale must stay 1 (a scale with no time would be silently inert). The event must
	/// fall inside the episode, and the new mass must be positive and finite.
	pub fn from_kwargs(t_step_s: Option<f32>, mass_scale: f32, dt: f32, steps: usize) -> Result<Self, String>
	{
		let Some(t) = t_step_s
		else
		{
			if mass_scale != 1.0
			{
				return Err(format!("payload_mass_scale = {mass_scale} needs payload_step_s (the time of the step)"));
			}
			return Ok(Self::OFF);
		};
		if !(t.is_finite() && t >= 0.0)
		{
			return Err(format!("payload_step_s = {t} must be a finite time >= 0 s"));
		}
		if !(mass_scale.is_finite() && mass_scale > 0.0)
		{
			return Err(format!("payload_mass_scale = {mass_scale} must be finite and > 0"));
		}
		let step = (t / dt).round();
		if !(step < steps as f32)
		{
			return Err(format!(
				"payload_step_s = {t} s is step {step}, beyond the {steps}-step episode: the event would never fire"
			));
		}
		Ok(PayloadStep { step: step as u32, mass_scale })
	}

	pub fn is_on(&self) -> bool
	{
		self.step != u32::MAX
	}

	/// The plant mass to set AT step `t` (exactly once, at the event), else None.
	/// `base` is the episode's mass draw; the product is the kernel's f32 operation.
	#[inline]
	pub fn mass_at(&self, t: usize, base: f32) -> Option<f32>
	{
		(self.is_on() && t == self.step as usize).then(|| base * self.mass_scale)
	}
}

#[cfg(test)]
mod tests
{
	use super::*;

	#[test]
	fn off_never_fires_and_validates()
	{
		assert_eq!(PayloadStep::from_kwargs(None, 1.0, 0.001, 2000).unwrap(), PayloadStep::OFF);
		assert!(PayloadStep::from_kwargs(None, 1.25, 0.001, 2000).is_err(), "scale without a time");
		assert!(PayloadStep::from_kwargs(Some(2.0), 1.25, 0.001, 2000).is_err(), "step 2000 of a 2000-step episode");
		assert!(PayloadStep::from_kwargs(Some(1.0), 0.0, 0.001, 2000).is_err());
		assert!(PayloadStep::from_kwargs(Some(-0.1), 1.25, 0.001, 2000).is_err());
		for t in 0..3000
		{
			assert_eq!(PayloadStep::OFF.mass_at(t, 0.04), None);
		}
	}

	#[test]
	fn fires_once_at_the_rounded_step()
	{
		let p = PayloadStep::from_kwargs(Some(1.0), 1.25, 0.001, 2000).unwrap();
		assert_eq!(p.step, 1000);
		let fired: Vec<usize> = (0..2000).filter(|&t| p.mass_at(t, 0.04).is_some()).collect();
		assert_eq!(fired, vec![1000]);
		assert_eq!(p.mass_at(1000, 0.04), Some(0.04f32 * 1.25f32));
	}
}
