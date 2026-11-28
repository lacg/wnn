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
//!   a3 `obs_dhat_residual_plant` (CTRL-48, ABI 45, `--obs-cascade-dhat-residual`) —
//!      the UNABSORBED disturbance the teacher plans on, per axis
//!          r = (d̂ + c·ki·I) / (c·ki·L)  =  d̂·inv_a + w·(I/L),   L = i_limit (1 if 0)
//!      (≡ I − I_ss(d̂) in authority units; ctbr_axis_mpc.rs `equilibrium`:
//!      I_ss = −d/(c·ki)). It decays to 0 as the rate loop's integrator absorbs d, so a
//!      held vehicle reads ~0 on roll/pitch (absorbed in 1–2 s) while yaw (authority
//!      3.65 rad/s², τ ≈ 7 s) keeps its deficit. ki = 0 (no integrator): r = d̂ / the
//!      observer clamp c·out_limit, w = 0. Computed from the SAME six recorded values
//!      as a1/a2 (rate-I and d̂ are produced whenever a3 is on, emitted or not), so the
//!      replay contract and the record format are unchanged.
//!
//!   a4 `obs_rate_err_tau` (CTRL-69, ABI 62, `--obs-rate-err`) — the MODEL-FREE input:
//!      the flown rate loop's tracking error e = rate_sp − gyro per axis, one-pole
//!      low-passed (rate_err_lpf.rs). Needs no observer and no plant constants.
//!      Recorded as its OWN replay stream (ReplayObs.rate_err, three values per step),
//!      so the six-value record above is unchanged.
//!
//!   ABI 62 also gives a1 an axis subset (`--obs-rate-i-axes`, CtbrSpec.obs_rate_i_axes;
//!   all three = the ABI-61 block exactly) and a4 its own (`--obs-rate-err-axes`).
//!
//! Canonical order: LAST in compute_features (after the horizontal channel), rate-I,
//! then the rate error, then d̂, then the residual (so the residual stays the block's
//! last slot — cresid_z, which the yaw-trim levers read by position), so every
//! pre-CTRL-38 layout is byte-for-byte unchanged when all are off.
//!
//! TRAIN = DEPLOY. Both values are computed by compute_features from live state —
//! the controller's own inner loop, and the observer fed `pwm_applied` (what the plant
//! received: step()'s return, or the host's observe_applied). A training REPLAY has
//! neither the flown loop nor the flown commands, so the rollout RECORDS the six
//! values per step (TrajectoryRs.cascade_obs → ReplayObs.cascade) and the replay
//! re-applies them (the obs_pwm / att_up pattern); replay_obs_guard refuses a replay
//! without the stream.

use crate::cascade_dhat::CascadeDhat;
use crate::rate_err_lpf::{axes_count, axis_in, RateErrLpf};

/// CTRL-48 a3: the residual's per-axis scale (see the module doc).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CasResidual
{
	pub inv_a: [f32; 3],
	pub w: [f32; 3],
}

impl CasResidual
{
	/// From the torque gains c, the rate loop's (ki, i_limit) per axis and the
	/// observer clamp c·out_limit.
	pub(crate) fn new(c: [f64; 3], ki_lim: [(f64, f64); 3], clamp: [f64; 3]) -> Self
	{
		let mut r = CasResidual { inv_a: [0.0; 3], w: [0.0; 3] };
		for a in 0..3
		{
			let (ki, lim) = ki_lim[a];
			let l = if lim > 0.0 { lim } else { 1.0 };
			let (inv, w) = if ki != 0.0 { (1.0 / (c[a] * ki * l), 1.0) } else { (1.0 / clamp[a], 0.0) };
			r.inv_a[a] = inv as f32;
			r.w[a] = w as f32;
		}
		r
	}

	/// The residual from the recorded six [rate I/L ×3, d̂ ×3] — ONE formula for the
	/// live path and the replay (the Metal twins use the same f32 expression).
	#[inline]
	pub(crate) fn of(&self, v: &[f32; 6]) -> [f32; 3]
	{
		std::array::from_fn(|a| v[3 + a] * self.inv_a[a] + self.w[a] * v[a])
	}
}

#[derive(Clone, Debug)]
pub(crate) struct CascadeObs
{
	/// a1 emitted axes (rate_err_lpf AXES bitmask; 0 = off).
	rate_i: u8,
	/// a4: the rate-error low-pass (Some ⇒ on; its mask = the emitted axes).
	rerr: Option<RateErrLpf>,
	/// a2 emitted (the observer below may also run for a3 alone).
	dhat_emit: bool,
	dhat: Option<CascadeDhat>,
	resid: Option<CasResidual>,
	/// [rate I/i_limit ×3, d̂ ×3] the LAST compute_features produced (zeros for an
	/// off block) — what a rollout records for the replay.
	last: [f32; 6],
	/// a4's three filtered values the LAST compute_features produced (zeros when off).
	last_rerr: [f32; 3],
	/// One-shot replay overrides (apply_replay_obs); None on every deploy path.
	replay: Option<[f32; 6]>,
	replay_rerr: Option<[f32; 3]>,
}

impl CascadeObs
{
	/// `rate_i` = a1's emitted axes (0 = off), `rerr` = a4 (Some ⇒ on), `dhat` = the
	/// observer (Some when a2 or a3 is on), `dhat_emit` = a2 on, `resid` = a3's scale
	/// (Some ⇒ a3 on; requires the observer).
	pub(crate) fn new(rate_i: u8, rerr: Option<RateErrLpf>, dhat: Option<CascadeDhat>, dhat_emit: bool, resid: Option<CasResidual>) -> Self
	{
		debug_assert!(dhat.is_some() || !(dhat_emit || resid.is_some()));
		CascadeObs { rate_i, rerr, dhat_emit, dhat, resid, last: [0.0; 6], last_rerr: [0.0; 3], replay: None, replay_rerr: None }
	}

	/// Features this block appends (one per emitted rate-I / rate-error axis, 3 per
	/// observer half).
	pub(crate) fn count(&self) -> usize
	{
		axes_count(self.rate_i)
			+ self.rerr.map_or(0, |f| axes_count(f.mask))
			+ 3 * (self.dhat_emit as usize)
			+ 3 * (self.resid.is_some() as usize)
	}

	/// a1 EMITTED on any axis (the GPU's cas_rate_i).
	pub(crate) fn rate_i_on(&self) -> bool
	{
		self.rate_i != 0
	}

	/// ABI 62: a1's emitted axes (the GPU's cas_rate_i_mask; AXES_ALL = ABI 61).
	pub(crate) fn rate_i_mask(&self) -> u8
	{
		self.rate_i
	}

	/// ABI 62 a4: the rate-error filter's constants (the GPU's cas_rerr_*).
	pub(crate) fn rate_err(&self) -> Option<RateErrLpf>
	{
		self.rerr
	}

	/// a2 EMITTED (the GPU's cas_dhat).
	pub(crate) fn dhat_on(&self) -> bool
	{
		self.dhat_emit
	}

	/// a3 (the GPU's cas_resid + its scale).
	pub(crate) fn residual(&self) -> Option<CasResidual>
	{
		self.resid
	}

	/// The rate-I VALUE is needed (a1 or a3).
	fn needs_rate_i(&self) -> bool
	{
		self.rate_i != 0 || self.resid.is_some()
	}

	/// The block reads the FLOWN rate loop (rate-I for a1/a3, its error for a4): a
	/// teacher-flown rollout must then hand the student the loop that is actually
	/// flying (ctbr_adopt_flown_rate_stage).
	pub(crate) fn reads_flown_rate(&self) -> bool
	{
		self.needs_rate_i() || self.rerr.is_some()
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
		if let Some(f) = self.rerr.as_mut()
		{
			f.reset();
		}
		self.last = [0.0; 6];
		self.last_rerr = [0.0; 3];
		self.replay = None;
		self.replay_rerr = None;
	}

	/// CTRL-86: the plant actuator (a symmetric, floor-free spec is the ABI-64
	/// set_motor_lag).
	pub(crate) fn set_actuator(&mut self, spec: crate::actuator_spec::ActuatorSpec, dt: f32)
	{
		if let Some(d) = self.dhat.as_mut()
		{
			d.set_actuator(spec, dt);
		}
	}

	pub(crate) fn set_replay(&mut self, v: [f32; 6])
	{
		self.replay = Some(v);
	}

	/// ABI 62: a4's recorded three (ReplayObs.rate_err).
	pub(crate) fn set_replay_rate_err(&mut self, v: [f32; 3])
	{
		self.replay_rerr = Some(v);
	}

	pub(crate) fn last(&self) -> [f32; 6]
	{
		self.last
	}

	/// ABI 62: a4's values of the last tick (what a rollout records).
	pub(crate) fn last_rate_err(&self) -> [f32; 3]
	{
		self.last_rerr
	}

	/// compute_features' cascade block, TICK half: this step's six values — the replay
	/// override (no observer tick: the replay has no flown commands), else the live
	/// rate-I and one observer step (once per physical step, holds included) — cached
	/// as `last`; and a4's three the same way (replay override, else one filter step on
	/// the flown loop's raw error `rate_err`, rad/s). Normally ticked at the tail; L2y
	/// of block 10 ticks it before the attitude block and emits at the tail.
	pub(crate) fn tick(&mut self, rate_i_norm: [f32; 3], rate_err: [f32; 3], gyro: [f32; 3], applied: &[f32]) -> [f32; 6]
	{
		let v = match self.replay.take()
		{
			Some(o) => o,
			None => self.live(rate_i_norm, gyro, applied),
		};
		self.last = v;
		self.last_rerr = match self.replay_rerr.take()
		{
			Some(o) => o,
			None => self.rerr.as_mut().map_or([0.0; 3], |f| f.step(rate_err)),
		};
		v
	}

	/// compute_features' tail, emit half: the enabled halves, canonical order (rate-I,
	/// rate error, d̂, residual).
	pub(crate) fn emit(&self, feats: &mut Vec<f32>, v: &[f32; 6])
	{
		Self::emit_axes(feats, self.rate_i, &v[0..3]);
		if let Some(f) = self.rerr
		{
			Self::emit_axes(feats, f.mask, &self.last_rerr);
		}
		if self.dhat_emit
		{
			feats.extend_from_slice(&v[3..6]);
		}
		if let Some(r) = self.resid
		{
			feats.extend_from_slice(&r.of(v));
		}
	}

	/// The axes of `mask`, in axis order (AXES_ALL = the whole triple).
	fn emit_axes(feats: &mut Vec<f32>, mask: u8, v: &[f32])
	{
		for (a, x) in v.iter().enumerate()
		{
			if axis_in(mask, a)
			{
				feats.push(*x);
			}
		}
	}

	fn live(&mut self, rate_i_norm: [f32; 3], gyro: [f32; 3], applied: &[f32]) -> [f32; 6]
	{
		let mut v = [0.0f32; 6];
		if self.needs_rate_i()
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
