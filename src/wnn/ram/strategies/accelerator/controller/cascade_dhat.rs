//! CTRL-38 (ABI 41): the cascade MPCOF teacher's OFFSET-FREE OBSERVER as a value of
//! its own, so the teacher (ctbr_mpcof.rs) and the cascade STUDENT's `--obs-cascade-dhat`
//! feature block (cascade_obs.rs) run ONE definition — the student's d̂ is the teacher's
//! d̂ law on the same inputs, never a re-derivation.
//!
//! THE LAW (extracted verbatim from ABI-40 CtbrMpcOf::observe/track_lag/lagged_force —
//! ctrl38_pins pins the teacher bit-identical across the extraction):
//!   ua  = the rotors' differential thrust from the MEASURED actuator: the pwm the
//!         plant was told to apply last tick, clamped to [0, 1], filtered through the
//!         plant's own first-order lag (ActuatorLag's prime/seed protocol: the first
//!         call after reset only primes, the second seeds the filter AT that command),
//!         then t = kt·pwm² per motor and the '+' mixer's axis forces
//!         [t3 − t1, t2 − t0, (t0 − t1 + t2 − t3)/4];
//!   d̂  ← clamp(d̂ + l·(ω'_measured − c·ua − d̂), ±c·out_limit),
//!         ω' the finite difference of consecutive gyro reads, c = [arm/Ix, arm/Iy,
//!         4·k_drag/Iz] (N → rad/s²), l = CTBR_MPC_L_GAIN per plant tick.
//! d̂ is therefore the angular acceleration the rotors' NOMINAL thrust does not
//! explain (motor asymmetry, torque-scale jitter, rotor drag, gusts), rad/s², bounded
//! by the most the rate loop's output limit can cancel.
//!
//! THE PROTOCOL every host keeps (DAgger, scorer, calibration, recorder): call
//! `observe(gyro_t, applied_{t−1})` ONCE per physical tick — the gyro just read and
//! the command the plant ACTUALLY received last tick (post rate loop, post trim /
//! exploration / expert override; the lag is modelled here, never read from the
//! plant). The 12/08/2026 DOB bug class (pre-trim pwm, frozen replay accumulator)
//! is excluded by construction: the input is the flown command, and a training
//! replay never re-runs this filter — it re-applies the value the rollout produced.

use crate::ctbr_mpcof::CTBR_MPC_L_GAIN;

/// Rotor force (N) → body angular acceleration (rad/s²) per axis: [arm/Ix, arm/Iy,
/// 4·k_drag/Iz] — the '+' mixer's roll/pitch arm and the yaw drag torque.
pub(crate) fn torque_gains(arm_length: f32, k_drag: f32, inertia: [f32; 3]) -> [f64; 3]
{
	let (arm, kd) = (arm_length as f64, k_drag as f64);
	[arm / inertia[0] as f64, arm / inertia[1] as f64, 4.0 * kd / inertia[2] as f64]
}

#[derive(Clone, Debug)]
pub(crate) struct CascadeDhat
{
	c: [f64; 3],
	k_thrust: f64,
	dt: f64,
	/// The plant's lag coefficient 4dt/max(T, 4dt) (actuator_lag::motor_lag_alpha).
	alpha: f64,
	clamp: [f64; 3],
	lag_pwm: [f64; 4],
	primed: bool,
	seeded: bool,
	dhat: [f64; 3],
	last_gyro: Option<[f64; 3]>,
}

impl CascadeDhat
{
	/// `c` = torque_gains(..), `out_limit_n` = the rate loop's output saturation (N),
	/// `motor_lag_s` = the plant's 2% settling time (0 ⇒ the command itself).
	pub(crate) fn new(c: [f64; 3], k_thrust: f64, dt: f32, motor_lag_s: f32, out_limit_n: f64) -> Self
	{
		CascadeDhat {
			c,
			k_thrust,
			dt: dt as f64,
			alpha: crate::actuator_lag::motor_lag_alpha(motor_lag_s, dt),
			clamp: [c[0] * out_limit_n, c[1] * out_limit_n, c[2] * out_limit_n],
			lag_pwm: [0.0; 4],
			primed: false,
			seeded: false,
			dhat: [0.0; 3],
			last_gyro: None,
		}
	}

	/// Episode start: the estimate, the actuator filter and the gyro memory forget.
	pub(crate) fn reset(&mut self)
	{
		self.lag_pwm = [0.0; 4];
		self.primed = false;
		self.seeded = false;
		self.dhat = [0.0; 3];
		self.last_gyro = None;
	}

	/// Re-arm the actuator filter for a plant with settling time `motor_lag_s` (the
	/// student's hosts call this per plant; the teacher is built with it).
	pub(crate) fn set_motor_lag(&mut self, motor_lag_s: f32, dt: f32)
	{
		self.alpha = crate::actuator_lag::motor_lag_alpha(motor_lag_s, dt);
	}

	/// The (gyro just read, pwm applied last tick) pair; the first call after reset
	/// is the t = 0 placeholder (prime only).
	pub(crate) fn observe(&mut self, gyro: [f32; 3], applied_pwm: [f64; 4])
	{
		let g = [gyro[0] as f64, gyro[1] as f64, gyro[2] as f64];
		if let (Some(ua), Some(prev)) = (self.track_lag(applied_pwm), self.last_gyro)
		{
			for a in 0..3
			{
				let rate_dot = (g[a] - prev[a]) / self.dt;
				let residual = rate_dot - self.c[a] * ua[a] - self.dhat[a];
				let lim = self.clamp[a];
				self.dhat[a] = (self.dhat[a] + CTBR_MPC_L_GAIN * residual).clamp(-lim, lim);
			}
		}
		self.last_gyro = Some(g);
	}

	/// Advance the measured-actuator filter; the lagged axis forces once seeded.
	fn track_lag(&mut self, applied: [f64; 4]) -> Option<[f64; 3]>
	{
		if !self.primed
		{
			self.primed = true;
			return None;
		}
		let cmd = applied.map(|v| v.clamp(0.0, 1.0));
		if !self.seeded
		{
			self.lag_pwm = cmd;
			self.seeded = true;
		}
		else
		{
			for m in 0..4
			{
				self.lag_pwm[m] += self.alpha * (cmd[m] - self.lag_pwm[m]);
			}
		}
		Some(self.lagged_force())
	}

	/// The rotors' differential thrust in the rate stage's axis-force coordinates.
	fn lagged_force(&self) -> [f64; 3]
	{
		let t = self.lag_pwm.map(|p| self.k_thrust * p * p);
		[t[3] - t[1], t[2] - t[0], (t[0] - t[1] + t[2] - t[3]) * 0.25]
	}

	/// The measured lagged axis force, None until the filter has seen a command.
	pub(crate) fn measured_force(&self) -> Option<[f64; 3]>
	{
		if self.seeded { Some(self.lagged_force()) } else { None }
	}

	/// d̂ (rad/s²).
	pub(crate) fn dhat(&self) -> [f64; 3]
	{
		self.dhat
	}

	/// d̂'s clamp: the most torque the rate loop's output limit can cancel (c·out_limit).
	pub(crate) fn clamp(&self) -> [f64; 3]
	{
		self.clamp
	}

	/// The torque gains c (rad/s² per N) — the GPU twin's constants.
	pub(crate) fn torque_gain(&self) -> [f64; 3]
	{
		self.c
	}

	pub(crate) fn k_thrust(&self) -> f64
	{
		self.k_thrust
	}
}
