//! CTRL-80 (ABI 64, `--obs-yaw-lead`, Luiz 26/11/2026): the yaw LEAD input.
//!
//!     yaw_lead = yaw_err − τ · gyro_z
//!
//! `yaw_err` is the value the proportional yaw input carries this step (target −
//! anchored heading, so d(yaw_err)/dt = −gyro_z) and `gyro_z` is the measured body yaw
//! rate — both ALREADY student inputs. The sum is where yaw_err will be in τ seconds
//! at the present rate: the one number the (yaw_err code × gyro_z code) cell cannot
//! give a RAM neuron, because inside a cell every time-to-go shares one address
//! (docs/ctrl80_yaw_lead_prereg.md). τ is a fixed constant (default 0.10 s): no
//! observer, no plant model, no teacher internals.
//!
//! LAYOUT. One feature directly after the yaw-anchor pair (yaw_err, yaw_err_i) —
//! `yaw_channel_end`. Its thermometer rows therefore sit in the MIDDLE of the ladder;
//! `take_rows` lifts them out before new_core sizes the ladder and
//! WnnController::enable_yaw_lead puts them back.
//!
//! ONE formula: `YawLead::value` (CPU, every path through compute_features) and its
//! strict-FP Metal twin `yaw_lead_value` (controller_rollout.metal, FC_YAW_LEAD(6)).

/// Number of base sensor features ahead of the derived blocks (controller.rs NUM_FEATURES).
const BASE_FEATURES: usize = 9;

/// The lead input (τ seconds). Copy: the Metal hosts read τ off the controller.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct YawLead
{
	tau: f32,
}

impl YawLead
{
	/// τ must be finite and in (0, 1] s (0 would duplicate yaw_err).
	pub(crate) fn new(tau: f32) -> Result<Self, String>
	{
		if !(tau.is_finite() && tau > 0.0 && tau <= 1.0)
		{
			return Err(format!("obs_yaw_lead: tau must be finite and in (0, 1] s (got {tau})"));
		}
		Ok(Self { tau })
	}

	pub(crate) fn tau(&self) -> f32
	{
		self.tau
	}

	/// yaw_err − τ·gyro_z as two IEEE f32 ops (product, then difference) — the Metal
	/// twin runs the same two ops in a strict-FP scope (no fused mul-add).
	pub(crate) fn value(&self, yaw_err: f32, gyro_z: f32) -> f32
	{
		let d = self.tau * gyro_z;
		yaw_err - d
	}
}

/// The obs toggles that sit AHEAD of the lead slot in compute_features' canonical order.
#[derive(Clone, Copy, Debug)]
pub(crate) struct YawChannelLayout
{
	pub(crate) num_motors: usize,
	pub(crate) tilt_p: bool,
	pub(crate) tilt_i: bool,
	pub(crate) peraxis_p: bool,
	pub(crate) peraxis_i: bool,
	pub(crate) peraxis_yaw: bool,
	pub(crate) pwm: bool,
	pub(crate) yaw_err: bool,
	pub(crate) yaw_err_i: bool,
}

impl YawChannelLayout
{
	/// The feature index directly after the yaw-anchor pair = the lead input's slot.
	pub(crate) fn yaw_channel_end(&self) -> usize
	{
		let pa = if self.peraxis_yaw { 3 } else { 2 };
		BASE_FEATURES
			+ self.tilt_p as usize
			+ self.tilt_i as usize
			+ (self.peraxis_p as usize + self.peraxis_i as usize) * pa
			+ self.pwm as usize * self.num_motors
			+ self.yaw_err as usize
			+ self.yaw_err_i as usize
	}
}

/// Lift the lead input's `bpf` thermometer rows out of the full ladder (they sit at
/// feature index `slot`), leaving the ladder new_core expects.
pub(crate) fn take_rows(thresholds: &mut Vec<f32>, slot: usize, bpf: usize) -> Result<Vec<f32>, String>
{
	let (a, b) = (slot * bpf, (slot + 1) * bpf);
	if thresholds.len() < b
	{
		return Err(format!(
			"obs_yaw_lead: thresholds length {} is shorter than the lead input's rows [{a}, {b})",
			thresholds.len()
		));
	}
	Ok(thresholds.drain(a..b).collect())
}

#[cfg(test)]
mod tests
{
	use super::*;

	#[test]
	fn tau_range()
	{
		for bad in [0.0f32, -0.1, 1.5, f32::NAN, f32::INFINITY]
		{
			assert!(YawLead::new(bad).is_err(), "tau {bad} accepted");
		}
		assert_eq!(YawLead::new(0.1).unwrap().tau(), 0.1);
	}

	/// The lead predicts yaw_err τ ahead: d(yaw_err)/dt = −gyro_z.
	#[test]
	fn value_is_the_definition_and_leads()
	{
		let l = YawLead::new(0.1).unwrap();
		assert_eq!(l.value(0.03, 0.2).to_bits(), (0.03f32 - 0.1f32 * 0.2f32).to_bits());
		// closing at 0.2 rad/s on a +0.03 rad error: 0.1 s ahead the error is +0.01.
		assert!((l.value(0.03, 0.2) - 0.01).abs() < 1e-6);
		// opening: the predicted error grows.
		assert!(l.value(0.03, -0.2) > 0.03);
	}

	#[test]
	fn slot_and_rows()
	{
		let lay = YawChannelLayout { num_motors: 4, tilt_p: false, tilt_i: false, peraxis_p: true, peraxis_i: true, peraxis_yaw: false, pwm: false, yaw_err: true, yaw_err_i: true };
		assert_eq!(lay.yaw_channel_end(), 9 + 2 + 2 + 2);
		let all = YawChannelLayout { tilt_p: true, tilt_i: true, peraxis_yaw: true, pwm: true, ..lay };
		assert_eq!(all.yaw_channel_end(), 9 + 2 + 6 + 4 + 2);
		let mut th: Vec<f32> = (0..20).map(|i| i as f32).collect();
		assert_eq!(take_rows(&mut th, 2, 4).unwrap(), vec![8.0, 9.0, 10.0, 11.0]);
		assert_eq!(th.len(), 16);
		assert_eq!(th[8], 12.0);
		assert!(take_rows(&mut th, 4, 4).is_err(), "short ladder accepted");
	}
}
