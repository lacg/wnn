//! CTRL-38 (ABI 41, Luiz 07/11/2026): the cascade student's two default-OFF input
//! blocks — what the MPCOF>rate teacher plans with and the student could not see.
//! The 06/11 imitation probe (logs/controller/ctbr_probe/dlng_s1_s1.out) measured the
//! teacher's hidden disturbance state (ua / rate integrator I / d̂) lifting imitation
//! R² sharply; the 07/11 35-airframe sweep put every WNN failure on a SOFT steady
//! offset of a strongly asymmetric airframe. Two independent arms (A/B):
//!
//!   a1 `obs_rate_i` — the inner firmware rate PID's own integrator I per axis, the
//!      state the real vehicle holds onboard. Feature = I / i_limit (the firmware's
//!      integral clamp), so it lives in [−1, 1] exactly: −1/+1 is the integrator
//!      pinned at its limit. At trim (ω = sp = 0) Fh = ki·I = −d/c, so the feature
//!      reads −d/(c·ki·i_limit): the disturbance as a fraction of the integrator's
//!      whole authority (docs/ctrl_cascade_design.md ~L228). i_limit = 0 (unclamped
//!      firmware) ⇒ I itself.
//!   a2 `obs_dhat_plant` — the teacher's offset-free observer (cascade_dhat.rs) run
//!      inside the student on the APPLIED motor command and the gyro. Feature = d̂ in
//!      rad/s², NOT normalised: it is the quantity the teacher plans on, already
//!      bounded by c·out_limit, and the thermometer ladder is a quantile fit on the
//!      flown distribution, so a scale factor would buy nothing.
//!
//! Canonical order: LAST in compute_features (after the horizontal channel), rate-I
//! before d̂, so every pre-CTRL-38 layout is byte-for-byte unchanged when both are off.
//!
//! TRAIN = DEPLOY. Both values are computed by compute_features from live state —
//! the controller's own inner loop, and the observer fed `pwm_applied` (what the plant
//! received: step()'s return, or the host's observe_applied). A training REPLAY has
//! neither the flown loop nor the flown commands, so the rollout RECORDS the six
//! values per step (TrajectoryRs.cascade_obs → ReplayObs.cascade) and the replay
//! re-applies them (the obs_pwm / att_up pattern); replay_obs_guard refuses a replay
//! without the stream.

use crate::cascade_dhat::CascadeDhat;

#[derive(Clone, Debug)]
pub(crate) struct CascadeObs
{
	rate_i: bool,
	dhat: Option<CascadeDhat>,
	/// [rate I/i_limit ×3, d̂ ×3] the LAST compute_features produced (zeros for an
	/// off block) — what a rollout records for the replay.
	last: [f32; 6],
	/// One-shot replay override (apply_replay_obs); None on every deploy path.
	replay: Option<[f32; 6]>,
}

impl CascadeObs
{
	pub(crate) fn new(rate_i: bool, dhat: Option<CascadeDhat>) -> Self
	{
		CascadeObs { rate_i, dhat, last: [0.0; 6], replay: None }
	}

	/// Features this block appends (3 per enabled half).
	pub(crate) fn count(&self) -> usize
	{
		3 * (self.rate_i as usize) + 3 * (self.dhat.is_some() as usize)
	}

	pub(crate) fn rate_i_on(&self) -> bool
	{
		self.rate_i
	}

	pub(crate) fn dhat_on(&self) -> bool
	{
		self.dhat.is_some()
	}

	pub(crate) fn observer(&self) -> Option<&CascadeDhat>
	{
		self.dhat.as_ref()
	}

	/// Episode start.
	pub(crate) fn reset(&mut self)
	{
		if let Some(d) = self.dhat.as_mut()
		{
			d.reset();
		}
		self.last = [0.0; 6];
		self.replay = None;
	}

	pub(crate) fn set_motor_lag(&mut self, motor_lag_s: f32, dt: f32)
	{
		if let Some(d) = self.dhat.as_mut()
		{
			d.set_motor_lag(motor_lag_s, dt);
		}
	}

	pub(crate) fn set_replay(&mut self, v: [f32; 6])
	{
		self.replay = Some(v);
	}

	pub(crate) fn last(&self) -> [f32; 6]
	{
		self.last
	}

	/// compute_features' tail: tick the observer (once per physical step — holds
	/// included) and append the enabled halves. A replay override replaces the live
	/// value and skips the observer tick (the replay has no flown commands).
	pub(crate) fn append(&mut self, feats: &mut Vec<f32>, rate_i_norm: [f32; 3], gyro: [f32; 3], applied: &[f32])
	{
		let v = match self.replay.take()
		{
			Some(o) => o,
			None => self.live(rate_i_norm, gyro, applied),
		};
		self.last = v;
		if self.rate_i
		{
			feats.extend_from_slice(&v[0..3]);
		}
		if self.dhat.is_some()
		{
			feats.extend_from_slice(&v[3..6]);
		}
	}

	fn live(&mut self, rate_i_norm: [f32; 3], gyro: [f32; 3], applied: &[f32]) -> [f32; 6]
	{
		let mut v = [0.0f32; 6];
		if self.rate_i
		{
			v[..3].copy_from_slice(&rate_i_norm);
		}
		if let Some(d) = self.dhat.as_mut()
		{
			d.observe(gyro, [applied[0] as f64, applied[1] as f64, applied[2] as f64, applied[3] as f64]);
			let e = d.dhat();
			v[3..].copy_from_slice(&[e[0] as f32, e[1] as f32, e[2] as f32]);
		}
		v
	}
}
