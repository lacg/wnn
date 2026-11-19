//! OPT-IN probe on the classical-rival scorer (dagger_train::score_classical_baseline_probed),
//! for the two evaluation-only steps of docs/ctrl_indi_teacher_spec.md §10 (14/11/2026):
//!
//!   * stage −1 — the MPCOF cascade's offset-free observer at another gain
//!     (`observer_gain`; CascadeDhat::set_observer_gain on the TEACHER's instance only);
//!   * F1 — the est-fed MPCOF cascade with the Mahony specific-force bias compensated
//!     (`bias_comp`): the attitude reference handed to the teacher is shifted by the
//!     prediction of att_bias_pred::AttBiasPredictor, the student's `--obs-att-bias-pred`
//!     law, unchanged.
//!
//! NOTHING HERE IS ON A SHIPPED PATH. The pyfunction `score_classical_baseline` passes
//! `None`; with `None`, or with a probe whose options are all off, the scorer executes
//! the same arithmetic in the same order (pinned by rival_probe_tests).
//!
//! WHAT THE COMPENSATED TEACHER READS (no oracle leak into an est-fed row):
//!   * the Mahony estimate quaternion it is already fed;
//!   * the world-frame horizontal velocity — the same sim signal the student reads as
//!     `--obs-vel-xy` and this scorer's own position loop already reads;
//!   * the Mahony gains, dt and nominal g.
//! `compensated_target` takes exactly (target, v_world, q_fed). The TRUE quaternion is
//! read only by `observe_feed` and `record`, which measure and never feed back.
//!
//! FRAMES. [abp_r, abp_p] is the predicted (estimate − truth) of the Euler roll and
//! pitch (rotations about BODY x / y; att_bias_pred.rs). The teacher's attitude error is
//! Euler(q_fed) − target per axis (optimal::state_error), so target + abp makes that
//! error (estimate − abp) − reference: the bias-corrected angle against the same tilt
//! reference. Yaw is untouched. `record` reports TRUE Euler roll/pitch minus the tilt
//! reference's, and the absolute TRUE heading — the block-6 decomposition's frames
//! (docs/ctrl_steady_gap_block6.md §0).
//!
//! PER-AXIS SPLIT (ABI 53, CTRL-48 yaw floor, 16/11/2026). `axes` additionally splits the
//! scorer's own error EXACTLY: v = Log(q_ref⁻¹ ⊗ q_true), the rotation vector of the true
//! attitude relative to the scorer's reference (G5 tilt reference, yaw 0; identity under
//! LEVEL), components in the BODY frame (x roll, y pitch, z yaw). |v| is the scorer's
//! geodesic angle (2·acos|q·q_ref|) up to f32-vs-f64 rounding, so the per-axis means are
//! a decomposition of the printed err / steady, not a different metric. Same episode
//! accounting as the scorer: whole-episode means over every episode, hold-window means
//! over the episodes that reached t >= 0.8·steps. This is the metric of
//! logs/controller/ctrl48/l2yp_investigation/axis_trace.py (the WNN replay), so teacher
//! and student rows decompose identically. Reached from Python only through
//! `score_classical_baseline(axis_out=list)`; axis_out=None never builds a probe.

use crate::att_bias_pred::AttBiasPredictor;
use crate::att_err_ref::AttErrRef;
use crate::controller::{quat_to_euler_f64, wrap_angle_f64, AttitudeSim};
use crate::optimal::Teacher;

/// The cascade student's OUTPUT LAW (controller.rs decode_ctbr under `--ctbr-delta-law`):
/// per bank, setpoint ← clamp(leak·previous + δ, ±range), δ on the γ-shaped alphabet of
/// `delta_max`·range with `levels` cells per bank (BINARY antagonist decode: steps of
/// 1/levels in the raw decode around the neutral 0.5). Used by the probe to ask what the
/// LABEL LAW flies when its output must pass through that alphabet
/// (docs/ctrl_gap_after_block9.md §B.2). The encode/decode are the student's own
/// functions (controller::delta_to_decoded / decoded_to_delta) — one definition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct StudentAlphabet
{
	pub(crate) leak: f32,
	pub(crate) delta_max: f32,
	pub(crate) range: [f32; 4],
	pub(crate) gamma: [f32; 4],
	pub(crate) levels: u32,
}

impl StudentAlphabet
{
	const NEUTRAL: f32 = 0.5;

	/// The setpoint the student's law reaches from `prev` when asked for `want`.
	fn step(&self, m: usize, prev: f32, want: f32) -> f32
	{
		let (r, dm) = (self.range[m], self.delta_max * self.range[m]);
		let base = self.leak * prev;
		let raw = crate::controller::delta_to_decoded(want - base, dm, Self::NEUTRAL, self.gamma[m]);
		let l = self.levels as f32;
		let q = (raw * l).round() / l;
		(base + crate::controller::decoded_to_delta(q, dm, Self::NEUTRAL, self.gamma[m])).clamp(-r, r)
	}
}

/// CTRL-59 STAGE 0 (docs/ctrl59_yaw_pterm_spec.md §3): the float yaw law with NO table — the
/// yaw-rate setpoint the label law hands the inner loop is replaced, on decision ticks, by
///     sp_r = clamp(cT·cresid_z + K_P·e_yaw, ±range_r),
/// the two feed-forward terms of `--yaw-trim-ff` + `--yaw-p-ff` and nothing else (the bank's
/// contribution is zero). Roll, pitch and collective stay the label law's. Every input is one
/// the student reads: cresid_z from the flown rate loop's integrator and the teacher's d̂
/// observer through the student's own formula (CasResidual::of on rate_integral_norm_of),
/// e_yaw = target yaw − the Euler yaw of the FED quaternion (the Mahony estimate on an est-fed
/// row — the student's `--obs-yaw-from-att-est` heading). Gains from
/// YawTrimErr::float_law_gains: the same row and definitions as the student's options.
#[derive(Clone, Copy, Debug)]
pub(crate) struct YawFloatLaw
{
	ct: f32,
	kp: f32,
	resid: crate::cascade_obs::CasResidual,
	range: f32,
}

/// What the probe measured, averaged over episodes (radians, metres).
// Built only by the cargo tests (the pyfunction passes no probe).
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct RivalProbeReport
{
	/// Hold-window mean |TRUE error|: [roll, pitch] vs the tilt reference, [yaw] absolute.
	pub(crate) hold_abs: [f64; 3],
	/// Hold-window mean |Euler(q_fed) − Euler(q_true)|, roll / pitch: the fed attitude's bias.
	pub(crate) fed_bias_abs: [f64; 2],
	/// The same after subtracting the prediction (equal to fed_bias_abs when bias_comp is off).
	pub(crate) comp_resid_abs: [f64; 2],
	/// Episodes that contributed a hold window.
	pub(crate) episodes: usize,
	/// Mean 3-D position error (m) in five consecutive fifths of the episode.
	pub(crate) pos_fifths: [f64; 5],
	/// 3-D position error (m) at the first and the last recorded step.
	pub(crate) pos_first_last: [f64; 2],
	/// Whole-episode mean |dx|, |dy|, |dz| (m) and the hold-window (last fifth) means.
	pub(crate) abs_xyz: [f64; 3],
	pub(crate) hold_abs_xyz: [f64; 3],
	/// Hold-window mean horizontal speed (m/s).
	pub(crate) hold_speed: f64,
	/// ABI 53 per-axis split (rad), see the module doc "PER-AXIS SPLIT".
	pub(crate) axes: AxisSplit,
	/// CTRL-59 stage 0: share of decision ticks with the float yaw setpoint AT the clamp
	/// (0 without the float law).
	pub(crate) yaw_clamp_share: f64,
}

pub(crate) use crate::axis_split::{rotvec_rel, AxisSplit};
use crate::axis_split::{AxisAcc, HoldPolicy};

#[derive(Clone, Debug, Default)]
pub(crate) struct RivalProbe
{
	observer_gain: Option<f64>,
	bias_comp: bool,
	abp: Option<AttBiasPredictor>,
	last_pred: [f64; 2],
	/// Euler(q_fed) − Euler(q_true), roll / pitch, at the tick the teacher was fed.
	fed_bias: [f64; 2],
	ep: [f64; 7],
	ep_cnt: usize,
	sum: [f64; 7],
	episodes: usize,
	alphabet: Option<StudentAlphabet>,
	held: [f32; 4],
	/// Position accumulators: [fifth 0..5 sums, |dx|, |dy|, |dz|, hold |dx|, |dy|, |dz|, hold speed].
	pos_ep: [f64; 12],
	pos_cnt: [usize; 5],
	pos_first: Option<f64>,
	pos_last: f64,
	pos_sum: [f64; 14],
	pos_eps: usize,
	/// ABI 53 per-axis split (axis_split::AxisAcc, hold policy ReachedOnly — this scorer's).
	axes: AxisAcc,
	/// CTRL-59 stage 0: the requested scale, the armed law, its held setpoint, and
	/// (decision ticks, decision ticks at the clamp).
	yaw_float_scale: Option<f32>,
	yaw_float: Option<YawFloatLaw>,
	yaw_held: f32,
	yaw_ticks: [usize; 2],
}

// Built only by the cargo tests (the pyfunction passes no probe).
#[cfg_attr(not(test), allow(dead_code))]
impl RivalProbe
{
	/// Every option off: the scorer's shipped behaviour, plus the per-axis report.
	pub(crate) fn measure_only() -> Self
	{
		RivalProbe::default()
	}

	pub(crate) fn with_observer_gain(mut self, l: f64) -> Self
	{
		self.observer_gain = Some(l);
		self
	}

	pub(crate) fn with_bias_comp(mut self) -> Self
	{
		self.bias_comp = true;
		self
	}

	/// Fly the label through the student's output alphabet (see StudentAlphabet).
	pub(crate) fn with_student_alphabet(mut self, a: StudentAlphabet) -> Self
	{
		self.alphabet = Some(a);
		self
	}

	/// CTRL-59 stage 0: fly the float yaw law (see YawFloatLaw) at `scale`·(−K_θ).
	pub(crate) fn with_yaw_float_law(mut self, scale: f32) -> Self
	{
		self.yaw_float_scale = Some(scale);
		self
	}

	/// Once, after the cascade's inner rate loop is built: derive the float yaw law for THIS
	/// rate loop and airframe at the scorer's decision hold. No-op unless requested.
	/// `c` = cascade_dhat::torque_gains of the airframe, `clamp` = the d̂ observer's.
	pub(crate) fn arm_yaw_float_law(&mut self, rate: &crate::pid_firmware::RateStageFw, c: [f64; 3], clamp: [f64; 3], dt: f32, hold: usize) -> Result<(), String>
	{
		let Some(scale) = self.yaw_float_scale
		else
		{
			return Ok(());
		};
		let (ct, kp) = crate::yaw_trim_err::YawTrimErr::float_law_gains(rate, c[2], dt, hold, scale)?;
		let ki_lim = std::array::from_fn(|i| (rate.axis_gains(i).ki, rate.axis_gains(i).i_limit));
		self.yaw_float = Some(YawFloatLaw {
			ct,
			kp,
			resid: crate::cascade_obs::CasResidual::new(c, ki_lim, clamp),
			range: crate::ctbr_mpcof::CTBR_MPC_RATE_MAX[2] as f32,
		});
		Ok(())
	}

	/// CTRL-59 stage 0: (cT, K_P) of the armed float yaw law.
	pub(crate) fn yaw_float_gains(&self) -> Option<(f32, f32)>
	{
		self.yaw_float.map(|l| (l.ct, l.kp))
	}

	/// The CTBR label with its yaw-rate setpoint (slot 3) replaced by the float yaw law's,
	/// refreshed on decision ticks and held between (the student's cadence). Identity unless
	/// the law is armed. `rate` = the flown rate loop BEFORE this tick, `dhat` = the teacher's
	/// observer after this tick's observe, `q_fed` = the attitude the teacher is fed.
	pub(crate) fn yaw_float_label(&mut self, t: usize, hold: usize, label: [f64; 4], rate: &crate::pid_firmware::RateStageFw, dhat: [f32; 3], q_fed: [f32; 4], target_yaw: f32) -> [f64; 4]
	{
		let Some(law) = self.yaw_float
		else
		{
			return label;
		};
		if t % hold.max(1) == 0
		{
			let ri = crate::ctbr::rate_integral_norm_of(rate);
			let cresid_z = law.resid.of(&[ri[0], ri[1], ri[2], dhat[0], dhat[1], dhat[2]])[2];
			let e_yaw = target_yaw - quat_to_euler_f64(q_fed).2 as f32;
			// The student's association order (yaw_ff_value): trim, then + p; bank = 0.
			let ff = law.ct * cresid_z + law.kp * e_yaw;
			self.yaw_held = ff.clamp(-law.range, law.range);
			self.yaw_ticks[0] += 1;
			self.yaw_ticks[1] += (self.yaw_held.abs() >= law.range) as usize;
		}
		[label[0], label[1], label[2], self.yaw_held as f64]
	}

	/// The CTBR label the inner loop is handed. Identity unless an alphabet is set; then,
	/// on decision ticks (`t % hold == 0`, CtbrOverRate's own refresh), the label is
	/// re-expressed as what the student's delta law can reach from the setpoint it holds.
	pub(crate) fn flown_label(&mut self, t: usize, hold: usize, label: [f64; 4]) -> [f64; 4]
	{
		let Some(a) = self.alphabet
		else
		{
			return label;
		};
		if t % hold.max(1) == 0
		{
			for m in 0..4
			{
				self.held[m] = a.step(m, self.held[m], label[m] as f32);
			}
		}
		self.held.map(|v| v as f64)
	}

	/// MEASUREMENT ONLY: the position error after the plant stepped (target = origin at
	/// `target_alt`), split into fifths of the episode.
	pub(crate) fn record_position(&mut self, t: usize, steps: usize, sim: &AttitudeSim, target_alt: f32)
	{
		let [x, y] = sim.position_xy_rs();
		let [vx, vy] = sim.velocity_xy_rs();
		let dz = (target_alt - sim.altitude_rs()) as f64;
		let (x, y) = (x as f64, y as f64);
		let e = (x * x + y * y + dz * dz).sqrt();
		let fifth = (5 * t / steps.max(1)).min(4);
		self.pos_ep[fifth] += e;
		self.pos_cnt[fifth] += 1;
		self.pos_ep[5] += x.abs();
		self.pos_ep[6] += y.abs();
		self.pos_ep[7] += dz.abs();
		if fifth == 4
		{
			self.pos_ep[8] += x.abs();
			self.pos_ep[9] += y.abs();
			self.pos_ep[10] += dz.abs();
			self.pos_ep[11] += ((vx as f64).powi(2) + (vy as f64).powi(2)).sqrt();
		}
		self.pos_first.get_or_insert(e);
		self.pos_last = e;
	}

	/// Once, after the teacher is built. `est` = the Mahony (kp, ki) when the row is
	/// estimator-fed. Bias compensation on an oracle-fed row is refused: there is no
	/// estimator bias to compensate, and the row would be mislabelled.
	pub(crate) fn arm(&mut self, teacher: &mut Teacher, est: Option<(f64, f64)>, dt: f32) -> Result<(), String>
	{
		if let Some(l) = self.observer_gain
		{
			teacher.set_ctbr_observer_gain(l)?;
		}
		if self.bias_comp
		{
			let (kp, ki) = est.ok_or("RivalProbe: bias_comp needs an estimator-fed row (use_estimator)")?;
			self.abp = Some(AttBiasPredictor::new(kp as f32, ki as f32, dt));
		}
		Ok(())
	}

	pub(crate) fn begin_episode(&mut self)
	{
		if let Some(p) = self.abp.as_mut()
		{
			p.reset();
		}
		self.last_pred = [0.0; 2];
		self.fed_bias = [0.0; 2];
		self.ep = [0.0; 7];
		self.ep_cnt = 0;
		self.held = [0.0; 4];
		self.yaw_held = 0.0;
		self.pos_ep = [0.0; 12];
		self.pos_cnt = [0; 5];
		self.pos_first = None;
		self.pos_last = 0.0;
		self.axes.begin_episode();
	}

	/// The attitude reference the teacher is handed this tick. Advances the predictor
	/// once per physical tick (the student's compute_features cadence). Identity when
	/// bias_comp is off.
	pub(crate) fn compensated_target(&mut self, target: [f32; 3], v_world: [f32; 2], q_fed: [f32; 4]) -> [f32; 3]
	{
		let Some(p) = self.abp.as_mut()
		else
		{
			return target;
		};
		let b = p.step(v_world, q_fed);
		self.last_pred = [b[0] as f64, b[1] as f64];
		[target[0] + b[0], target[1] + b[1], target[2]]
	}

	/// MEASUREMENT ONLY: the quaternion the teacher is fed this tick next to the truth
	/// at the same instant. The truth is stored as a bias to report and is never handed
	/// to `compensated_target` or to the teacher.
	pub(crate) fn observe_feed(&mut self, q_fed: [f32; 4], q_true: [f32; 4])
	{
		let (rf, pf, yf) = quat_to_euler_f64(q_fed);
		let (r, p, y) = quat_to_euler_f64(q_true);
		self.fed_bias = [wrap_angle_f64(rf - r), wrap_angle_f64(pf - p)];
		self.axes.set_fed_yaw(wrap_angle_f64(yf - y));
	}

	/// The per-axis split (module doc "PER-AXIS SPLIT"), every recorded step.
	fn record_axes(&mut self, in_hold: bool, sim: &AttitudeSim, att_ref: &AttErrRef)
	{
		self.axes.record(in_hold, att_ref.ref_quat(sim), sim.quaternion());
	}

	/// After the plant stepped: accumulate the hold-window decomposition. Measurement only.
	pub(crate) fn record(&mut self, in_hold: bool, sim: &AttitudeSim, att_ref: &AttErrRef)
	{
		self.record_axes(in_hold, sim, att_ref);
		if !in_hold
		{
			return;
		}
		let (r, p, y) = quat_to_euler_f64(sim.quaternion());
		let (rr, pr, _) = att_ref.ref_quat(sim).map_or((0.0, 0.0, 0.0), quat_to_euler_f64);
		let bias = self.fed_bias;
		let v = [
			wrap_angle_f64(r - rr).abs(),
			wrap_angle_f64(p - pr).abs(),
			wrap_angle_f64(y).abs(),
			bias[0].abs(),
			bias[1].abs(),
			(bias[0] - self.last_pred[0]).abs(),
			(bias[1] - self.last_pred[1]).abs(),
		];
		for i in 0..7
		{
			self.ep[i] += v[i];
		}
		self.ep_cnt += 1;
	}

	pub(crate) fn end_episode(&mut self)
	{
		// Per-axis split: the rival scorer's accounting (axis_split::HoldPolicy::ReachedOnly).
		self.axes.end_episode(HoldPolicy::ReachedOnly);
		// Position: only episodes that reached the last fifth (a diverged one has no hold).
		if let (Some(first), true) = (self.pos_first, self.pos_cnt[4] > 0)
		{
			let all: usize = self.pos_cnt.iter().sum();
			for i in 0..5
			{
				self.pos_sum[i] += self.pos_ep[i] / self.pos_cnt[i] as f64;
			}
			for i in 5..8
			{
				self.pos_sum[i] += self.pos_ep[i] / all as f64;
			}
			for i in 8..12
			{
				self.pos_sum[i] += self.pos_ep[i] / self.pos_cnt[4] as f64;
			}
			self.pos_sum[12] += first;
			self.pos_sum[13] += self.pos_last;
			self.pos_eps += 1;
		}
		if self.ep_cnt == 0
		{
			return;
		}
		for i in 0..7
		{
			self.sum[i] += self.ep[i] / self.ep_cnt as f64;
		}
		self.episodes += 1;
	}

	fn pos_mean<const N: usize>(&self, from: usize) -> [f64; N]
	{
		let n = self.pos_eps.max(1) as f64;
		std::array::from_fn(|i| self.pos_sum[from + i] / n)
	}

	pub(crate) fn report(&self) -> RivalProbeReport
	{
		let n = self.episodes.max(1) as f64;
		let m = self.sum.map(|v| v / n);
		RivalProbeReport {
			hold_abs: [m[0], m[1], m[2]],
			fed_bias_abs: [m[3], m[4]],
			comp_resid_abs: [m[5], m[6]],
			episodes: self.episodes,
			pos_fifths: self.pos_mean::<5>(0),
			pos_first_last: self.pos_mean::<2>(12),
			abs_xyz: self.pos_mean::<3>(5),
			hold_abs_xyz: self.pos_mean::<3>(8),
			hold_speed: self.pos_mean::<1>(11)[0],
			axes: self.axis_split(),
			yaw_clamp_share: self.yaw_ticks[1] as f64 / self.yaw_ticks[0].max(1) as f64,
		}
	}

	fn axis_split(&self) -> AxisSplit
	{
		self.axes.split(HoldPolicy::ReachedOnly)
	}
}
