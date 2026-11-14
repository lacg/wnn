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
}

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
		self.pos_ep = [0.0; 12];
		self.pos_cnt = [0; 5];
		self.pos_first = None;
		self.pos_last = 0.0;
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
		let (rf, pf, _) = quat_to_euler_f64(q_fed);
		let (r, p, _) = quat_to_euler_f64(q_true);
		self.fed_bias = [wrap_angle_f64(rf - r), wrap_angle_f64(pf - p)];
	}

	/// After the plant stepped: accumulate the hold-window decomposition. Measurement only.
	pub(crate) fn record(&mut self, in_hold: bool, sim: &AttitudeSim, att_ref: &AttErrRef)
	{
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
		}
	}
}
