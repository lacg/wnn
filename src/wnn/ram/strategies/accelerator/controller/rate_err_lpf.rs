//! ABI 62 (CTRL-69, Luiz 22/11/2026): the cascade student's MODEL-FREE disturbance
//! input — the firmware rate loop's own tracking error, low-passed.
//!
//! WHAT IT IS. Per axis e = rate_sp − gyro, exactly the error the FLOWN rate PID's
//! P-term used at its last update (pid_firmware.rs PidCh::last_err; pid.c
//! PidObject.error on the vehicle). The integrator I of that loop is ∫ki·e, so e is
//! what the rate loop has NOT yet absorbed, measured on board, with no plant model —
//! the model-free replacement for the MPCOF observer's unabsorbed residual (CTRL-48
//! a3). docs/ctrl69_rate_err_precheck.md: on roll/pitch e carries 0.79-1.04 of the
//! residual's label information in every arm (rate-I ~0); on yaw it does not (rate-I
//! does), hence the per-axis subsets of both blocks.
//!
//! THE FILTER. One pole, time constant τ (default 0.05 s = "E50", the precheck's best
//! variant), ticked once per PHYSICAL step (compute_features' cadence; the error itself
//! is zero-order-held between the 500 Hz rate updates):
//!     y ← y + α·(x − y),   α = 1 − exp(−dt/τ),   x = e / range_axis
//! The SAME f32 expression is the Metal twin (controller_rollout.metal rl_lpf_step) and
//! the on-board C (mcu/wnn_cascade.c); α and 1/range are computed ONCE here (f64 → f32)
//! and handed to both, never re-derived.
//!
//! THE SCALE. x = e / range[1+a]: the error as a fraction of the setpoint authority the
//! student itself commands on that axis (the CTBR bank's ±range, rad/s). It is
//! dimensionless and O(1) like rate-I (I/i_limit) and the residual (authority units).
//! The thermometer ladder is a quantile fit on the flown distribution, so the scale
//! does not move a single cut; it only keeps the logged/feature values readable and
//! comparable across axes.

/// Axis subset bitmask: bit 0 roll, bit 1 pitch, bit 2 yaw. ALL = the ABI-61 layout.
pub(crate) const AXES_ALL: u8 = 0b111;

/// Features an axis subset emits.
#[inline]
pub(crate) fn axes_count(mask: u8) -> usize
{
	(mask & AXES_ALL).count_ones() as usize
}

/// Whether axis `a` is in the subset.
#[inline]
pub(crate) fn axis_in(mask: u8, a: usize) -> bool
{
	(mask >> a) & 1 == 1
}

/// The one-pole step — ONE formula for the live path (the Metal and C twins copy it).
#[inline]
pub(crate) fn lpf_step(y: f32, x: f32, alpha: f32) -> f32
{
	y + alpha * (x - y)
}

/// The filter's constants alone — what a population must share (the GPU scorer's
/// uniform contract); the state y differs per controller and is not part of it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RateErrConsts
{
	pub alpha: f32,
	pub inv_scale: [f32; 3],
	pub mask: u8,
}

/// The low-pass state and its constants (per axis).
#[derive(Clone, Copy, Debug)]
pub(crate) struct RateErrLpf
{
	/// α = 1 − exp(−dt/τ) (f64 → f32 once).
	pub alpha: f32,
	/// 1 / range[1+a] (f64 → f32 once).
	pub inv_scale: [f32; 3],
	/// Emitted axes (all three are filtered and recorded; the mask picks the features).
	pub mask: u8,
	y: [f32; 3],
}

impl RateErrLpf
{
	pub(crate) fn new(tau_s: f64, dt: f32, rate_range: [f32; 3], mask: u8) -> Self
	{
		RateErrLpf {
			alpha: (1.0 - (-(dt as f64) / tau_s).exp()) as f32,
			inv_scale: std::array::from_fn(|a| (1.0 / rate_range[a] as f64) as f32),
			mask,
			y: [0.0; 3],
		}
	}

	pub(crate) fn consts(&self) -> RateErrConsts
	{
		RateErrConsts { alpha: self.alpha, inv_scale: self.inv_scale, mask: self.mask }
	}

	/// Episode start: the filter forgets (the rate loop's error is 0 after its reset).
	pub(crate) fn reset(&mut self)
	{
		self.y = [0.0; 3];
	}

	/// One physical step on the raw errors (rad/s); returns the filtered, scaled values.
	pub(crate) fn step(&mut self, err: [f32; 3]) -> [f32; 3]
	{
		for a in 0..3
		{
			self.y[a] = lpf_step(self.y[a], err[a] * self.inv_scale[a], self.alpha);
		}
		self.y
	}
}

#[cfg(test)]
mod tests
{
	use super::*;

	/// α is the exact discretisation of the continuous pole: a step input reaches
	/// 1 − 1/e after τ, and the scale is 1/range.
	#[test]
	fn step_response_is_one_pole_tau()
	{
		let (tau, dt) = (0.05f64, 0.001f32);
		let mut f = RateErrLpf::new(tau, dt, [4.0, 4.0, 2.0], AXES_ALL);
		let mut y = [0.0f32; 3];
		for _ in 0..50
		{
			y = f.step([4.0, -4.0, 2.0]);
		}
		for a in 0..3
		{
			let want = if a == 1 { -(1.0 - (-1.0f64).exp()) } else { 1.0 - (-1.0f64).exp() };
			assert!((y[a] as f64 - want).abs() < 1e-4, "axis {a}: {} vs {want}", y[a]);
		}
		f.reset();
		assert_eq!(f.step([0.0; 3]), [0.0; 3]);
		assert_eq!(axes_count(0b011), 2);
		assert!(axis_in(0b100, 2) && !axis_in(0b100, 0));
	}
}
