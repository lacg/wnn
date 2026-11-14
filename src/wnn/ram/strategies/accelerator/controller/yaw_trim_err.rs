//! L2y (CTRL-48 block 10, ABI 51, `--obs-yaw-trim-err`): the student's yaw error
//! referenced to the TRIM the yaw rate loop must ride on (docs/ctrl48_block10_levers_spec.md
//! §L2y). The yaw analogue of L1 (tilt_ref_err.rs).
//!
//! PHYSICS (BODY z = WORLD yaw at the tilts flown; θ = yaw − target = −yaw_err, rad).
//! The MPCOF>rate teacher plans the yaw-rate setpoint linearly on the shifted state
//! (ctbr_axis_mpc.rs): sp = K_θ·θ + K_ω·ω + K_ua·(ua − ua_ss) + K_I·(I − I_ss) + ….
//! Yaw has little authority (the rate integrator absorbs a yaw disturbance with
//! τ ≈ 7 s), so through a 2 s episode the integrator stays short of its rest value
//! I_ss = −d̂/(c·ki), and the plan carries a TRIM term K_I·(I − I_ss) next to the
//! proportional one. The student already reads that unabsorbed part as
//! cresid_z = (I − I_ss)/L (cascade_obs.rs a3, L = the yaw rate loop's i_limit), and
//! its yaw error as yaw_err. In the hold the label is, to first order,
//!     sp ≈ −K_θ·yaw_err + K_I·L·cresid_z = −K_θ·(yaw_err − κ·cresid_z),  κ = K_I·L / K_θ.
//! Two separately quantised inputs cannot form that sum; this option forms it in
//! floating point before the ladder, as L1 does for roll / pitch.
//!
//! WHAT IT CHANGES. Only the VALUE of the existing yaw-error inputs (and so of their
//! integrals): every yaw error is measured against the trim-shifted target
//!     target_yaw_eff = target_yaw − κ·cresid_z,   yaw_err = target_yaw_eff − yaw_heading.
//! No input is added, removed or moved.
//!
//! κ IS DERIVED, NOT FITTED: the teacher's own unconstrained gain row
//! (CtbrAxisMpc::gain_row) on the yaw axis of the student's own rate loop and airframe
//! (CtbrSpec: firmware rate gains, residual plant), at the student's decision hold
//! n_o, rate-loop phase 0, lag-free. cf21_brushless at n_o = 10: K_θ = −9.586 rad/s per
//! rad, K_I = −0.1372, L = 2.909 ⇒ κ = 0.04165 rad per unit of cresid_z. The motor lag
//! moves κ by < 0.2 % over 0-0.3 s (`kappa_barely_depends_on_the_lag`), so the
//! lag-free plan is used and the option needs nothing a host sets later.

use crate::ctbr_axis_mpc::{CtbrAxisMpc, IN, TH};
use crate::ctbr_mpcof::{axis_plant, CTBR_MPC_HORIZON, CTBR_MPC_RATE_MAX, CTBR_MPC_WEIGHTS};

/// The yaw axis.
const YAW: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct YawTrimErr
{
	/// rad of yaw per unit of cresid_z.
	kappa: f32,
}

impl YawTrimErr
{
	/// From the cascade student's own contract: the inner loop's rate gains and the
	/// residual block's airframe, the physics step and the decision hold.
	pub(crate) fn from_spec(spec: &crate::ctbr::CtbrSpec, dt: f32, hold: usize) -> Result<Self, String>
	{
		let p = spec
			.obs_dhat_residual_plant
			.ok_or("obs_yaw_trim_err needs the cascade residual block (--obs-cascade-dhat-residual): cresid_z is the trim it reads")?;
		let c = crate::cascade_dhat::torque_gains(p[0] as f32, p[1] as f32, [p[2] as f32, p[3] as f32, p[4] as f32])[YAW];
		let rate = spec.rate_stage();
		let g = rate.axis_gains(YAW);
		if !(g.ki > 0.0)
		{
			return Err(format!("obs_yaw_trim_err: the yaw rate loop has no integrator (ki = {}) — there is no unabsorbed trim to reference", g.ki));
		}
		let l = if g.i_limit > 0.0 { g.i_limit } else { 1.0 };
		let row = CtbrAxisMpc::gain_row(&axis_plant(&rate, c, dt, 1.0, YAW), hold.max(1), 0, CTBR_MPC_WEIGHTS, CTBR_MPC_HORIZON, CTBR_MPC_RATE_MAX[YAW]);
		let (k_theta, k_i) = (row[TH], row[IN]);
		if !(k_theta.is_finite() && k_theta < 0.0 && k_i.is_finite())
		{
			return Err(format!("obs_yaw_trim_err: degenerate yaw law (K_theta = {k_theta}, K_I = {k_i})"));
		}
		Ok(YawTrimErr { kappa: (k_i * l / k_theta) as f32 })
	}

	/// κ (rad per unit of cresid_z) — the Metal uniform and the Python getter.
	pub(crate) fn kappa(&self) -> f32
	{
		self.kappa
	}

	/// The yaw target the error is measured against this step, from this step's
	/// cresid_z. f32 twin of the kernels' `P.target2 - P.yte_kappa * cresid_z`.
	#[inline]
	pub(crate) fn effective_target(&self, target_yaw: f32, cresid_z: f32) -> f32
	{
		target_yaw - self.kappa * cresid_z
	}
}

#[cfg(test)]
mod tests
{
	use super::*;

	fn cf21_spec() -> crate::ctbr::CtbrSpec
	{
		crate::ctrl38_tests::obs_spec3(false, false, true)
	}

	/// κ is the teacher's own gain ratio on the yaw axis, and the teacher's plan from a
	/// state holding only a yaw error and an unabsorbed integrator IS −K_θ·(yaw_err −
	/// κ·cresid_z): the definition reproduces the law it is derived from.
	#[test]
	fn kappa_is_the_teachers_yaw_law()
	{
		let spec = cf21_spec();
		let dt = crate::abi38_pins::DT;
		let y = YawTrimErr::from_spec(&spec, dt, 10).unwrap();
		let rate = spec.rate_stage();
		let p = spec.obs_dhat_residual_plant.unwrap();
		let c = crate::cascade_dhat::torque_gains(p[0] as f32, p[1] as f32, [p[2] as f32, p[3] as f32, p[4] as f32])[YAW];
		let plant = axis_plant(&rate, c, dt, 1.0, YAW);
		let row = CtbrAxisMpc::gain_row(&plant, 10, 0, CTBR_MPC_WEIGHTS, CTBR_MPC_HORIZON, CTBR_MPC_RATE_MAX[YAW]);
		let l = rate.axis_gains(YAW).i_limit;
		eprintln!("[l2y] K_theta {:.4} K_I {:.5} L {l:.4} kappa {:.5}", row[TH], row[IN], y.kappa());
		let mut mpc = CtbrAxisMpc::build(&plant, 10, 0, CTBR_MPC_WEIGHTS, CTBR_MPC_HORIZON, CTBR_MPC_RATE_MAX[YAW]);
		for (yaw_err, cresid) in [(0.004f64, 0.0f64), (0.0, 0.3), (-0.003, -0.5), (0.006, 0.2)]
		{
			let mut x = [0.0f64; crate::ctbr_axis_mpc::NS];
			x[TH] = -yaw_err;
			x[IN] = cresid * l;
			let sp = mpc.solve(&x);
			let want = -row[TH] * (yaw_err - y.kappa() as f64 * cresid);
			assert!((sp - want).abs() < 1e-5 * want.abs().max(1e-3), "({yaw_err}, {cresid}): plan {sp} vs -K_theta*(yaw_err - kappa*cresid) {want}");
		}
		assert!(y.kappa() > 0.03 && y.kappa() < 0.06, "cf21 yaw: kappa {} outside the derived 0.04", y.kappa());
	}

	/// The lag moves κ by < 0.2 % (so the lag-free plan the option uses is the law the
	/// lagged teacher flies); the decision hold matters (it sets the plan's sample time).
	#[test]
	fn kappa_barely_depends_on_the_lag()
	{
		let spec = cf21_spec();
		let (dt, rate) = (crate::abi38_pins::DT, spec.rate_stage());
		let p = spec.obs_dhat_residual_plant.unwrap();
		let c = crate::cascade_dhat::torque_gains(p[0] as f32, p[1] as f32, [p[2] as f32, p[3] as f32, p[4] as f32])[YAW];
		let l = rate.axis_gains(YAW).i_limit;
		let free = YawTrimErr::from_spec(&spec, dt, 10).unwrap().kappa() as f64;
		for lag in [0.05f32, 0.2, 0.3]
		{
			let alpha = crate::actuator_lag::motor_lag_alpha(lag, dt);
			let row = CtbrAxisMpc::gain_row(&axis_plant(&rate, c, dt, alpha, YAW), 10, 0, CTBR_MPC_WEIGHTS, CTBR_MPC_HORIZON, CTBR_MPC_RATE_MAX[YAW]);
			let k = row[IN] * l / row[TH];
			assert!(((k - free) / free).abs() < 2e-3, "lag {lag}: kappa {k} vs lag-free {free}");
		}
		let k1 = YawTrimErr::from_spec(&spec, dt, 1).unwrap().kappa();
		assert!((k1 as f64 - free).abs() > 1e-4, "the hold must reach the plan");
	}

	#[test]
	fn refuses_without_the_residual_or_a_yaw_integrator()
	{
		let mut s = cf21_spec();
		s.obs_dhat_residual_plant = None;
		assert!(YawTrimErr::from_spec(&s, 0.001, 10).is_err());
		let mut s = cf21_spec();
		s.rate_gains[2 * 4 + 1] = 0.0; // yaw rate ki
		assert!(YawTrimErr::from_spec(&s, 0.001, 10).is_err());
	}
}
