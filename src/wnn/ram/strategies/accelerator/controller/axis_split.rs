//! Per-axis split of a scorer's attitude error (ABI 53 rival scorer; ABI 56 WNN scorer).
//!
//! v = Log(q_ref⁻¹ ⊗ q_true): the rotation vector of the true attitude relative to the
//! scorer's own reference (G5 tilt reference, yaw 0; identity under LEVEL), components in
//! the BODY frame (x roll, y pitch, z yaw). |v| is the scorer's geodesic angle
//! (2·acos|q·q_ref|) up to f32-vs-f64 rounding, so the per-axis means DECOMPOSE the printed
//! err / steady — they are not a different metric.
//!
//! MEASUREMENT ONLY: reads the sim after it stepped; never feeds back. The two scorers
//! account the hold window differently and `HoldPolicy` mirrors each exactly:
//!   * ReachedOnly (score_classical_baseline): steady averages the episodes that reached
//!     t >= 0.8·steps;
//!   * FallbackWhole (score_controllers_cpu / rollout_one): an episode that never reached
//!     the window contributes its whole-episode mean (the kernel's else-branch), averaged
//!     over every episode.
//! err is the whole-episode mean averaged over every episode (a 0-step episode adds 0) in
//! both. `fed_yaw` is the heading drift of whatever the controller steers on: the rival's
//! fed quaternion (0 oracle-fed) or the student's anchored heading (NaN when it has none).

/// The split, radians (flat order = `AxisSplit::flat`, Python classical_baseline.AXIS_FIELDS).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct AxisSplit
{
	/// Whole-episode mean |v_x|, |v_y|, |v_z| (roll, pitch, yaw) — the decomposition of err.
	pub(crate) err_abs: [f64; 3],
	/// Whole-episode mean |v| (≈ the scorer's err).
	pub(crate) err_norm: f64,
	/// Hold-window mean |v_x|, |v_y|, |v_z| — the decomposition of steady.
	pub(crate) hold_abs: [f64; 3],
	/// Hold-window mean |v| (≈ the scorer's steady).
	pub(crate) hold_norm: f64,
	/// Hold-window mean SIGNED v_x, v_y, v_z (a constant offset survives; noise cancels).
	pub(crate) hold_signed: [f64; 3],
	/// Hold-window mean |yaw(steered-on) − yaw(true)| (see the module doc).
	pub(crate) hold_fed_yaw_abs: f64,
	/// Episodes in the whole-episode mean / episodes that reached the hold window.
	pub(crate) episodes_all: usize,
	pub(crate) episodes_hold: usize,
}

impl AxisSplit
{
	/// The 14 values `axis_out=` appends, in AXIS_FIELDS order (keep the two in step).
	pub(crate) fn flat(&self) -> [f64; 14]
	{
		[
			self.err_abs[0], self.err_abs[1], self.err_abs[2], self.err_norm,
			self.hold_abs[0], self.hold_abs[1], self.hold_abs[2], self.hold_norm,
			self.hold_signed[0], self.hold_signed[1], self.hold_signed[2],
			self.hold_fed_yaw_abs, self.episodes_all as f64, self.episodes_hold as f64,
		]
	}
}

/// Which hold-window accounting the owning scorer uses (module doc).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum HoldPolicy
{
	ReachedOnly,
	FallbackWhole,
}

/// Rotation vector (rad, body frame) of q relative to q_ref: Log(q_ref⁻¹ ⊗ q), shortest
/// rotation (w ≥ 0). f64 throughout; measurement only.
pub(crate) fn rotvec_rel(q_ref: [f32; 4], q: [f32; 4]) -> [f64; 3]
{
	let r = q_ref.map(|v| v as f64);
	let b = q.map(|v| v as f64);
	let (w1, x1, y1, z1) = (r[0], -r[1], -r[2], -r[3]);
	let (w2, x2, y2, z2) = (b[0], b[1], b[2], b[3]);
	let mut d = [
		w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
		w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
		w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
		w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
	];
	if d[0] < 0.0
	{
		d = d.map(|v| -v);
	}
	let n = (d[1] * d[1] + d[2] * d[2] + d[3] * d[3]).sqrt();
	if n < 1e-12
	{
		return [0.0; 3];
	}
	let a = 2.0 * n.atan2(d[0]) / n;
	[a * d[1], a * d[2], a * d[3]]
}

const ID: [f32; 4] = [1.0, 0.0, 0.0, 0.0];

/// The accumulator. Per-episode slots: [|v| x3, |v|, signed x3, |fed yaw|] for the whole
/// episode (0..8) and the hold window (8..16).
#[derive(Clone, Debug, Default)]
pub(crate) struct AxisAcc
{
	fed_yaw: f64,
	ep: [f64; 16],
	cnt: [usize; 2],
	sum_all: [f64; 4],
	sum_hold: [f64; 8],
	eps_hold: usize,
	begun: usize,
}

impl AxisAcc
{
	pub(crate) fn begin_episode(&mut self)
	{
		self.fed_yaw = 0.0;
		self.ep = [0.0; 16];
		self.cnt = [0; 2];
		self.begun += 1;
	}

	/// The heading drift (rad, wrapped) of what the controller steers on, this step.
	pub(crate) fn set_fed_yaw(&mut self, drift: f64)
	{
		self.fed_yaw = drift;
	}

	/// One step, after the plant stepped. `q_ref` None ⇒ LEVEL (identity).
	pub(crate) fn record(&mut self, in_hold: bool, q_ref: Option<[f32; 4]>, q: [f32; 4])
	{
		let v = rotvec_rel(q_ref.unwrap_or(ID), q);
		let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
		let vals = [v[0].abs(), v[1].abs(), v[2].abs(), norm, v[0], v[1], v[2], self.fed_yaw.abs()];
		for (i, x) in vals.iter().enumerate()
		{
			self.ep[i] += x;
		}
		self.cnt[0] += 1;
		if in_hold
		{
			for (i, x) in vals.iter().enumerate()
			{
				self.ep[8 + i] += x;
			}
			self.cnt[1] += 1;
		}
	}

	/// Close the episode under `policy` (module doc).
	pub(crate) fn end_episode(&mut self, policy: HoldPolicy)
	{
		let c0 = self.cnt[0].max(1) as f64;
		for i in 0..4
		{
			self.sum_all[i] += self.ep[i] / c0;
		}
		// hold slots in the ABI-53 order: [|v| x3, |v|, signed x3, |fed yaw|]
		if self.cnt[1] > 0
		{
			let c1 = self.cnt[1] as f64;
			for i in 0..8
			{
				self.sum_hold[i] += self.ep[8 + i] / c1;
			}
			self.eps_hold += 1;
		}
		else if policy == HoldPolicy::FallbackWhole && self.cnt[0] > 0
		{
			for i in 0..8
			{
				self.sum_hold[i] += self.ep[i] / c0;
			}
		}
	}

	pub(crate) fn split(&self, policy: HoldPolicy) -> AxisSplit
	{
		let na = self.begun.max(1) as f64;
		let nh = match policy
		{
			HoldPolicy::ReachedOnly => self.eps_hold.max(1) as f64,
			HoldPolicy::FallbackWhole => self.begun.max(1) as f64,
		};
		let a = self.sum_all.map(|v| v / na);
		let h = self.sum_hold.map(|v| v / nh);
		AxisSplit {
			err_abs: [a[0], a[1], a[2]],
			err_norm: a[3],
			hold_abs: [h[0], h[1], h[2]],
			hold_norm: h[3],
			hold_signed: [h[4], h[5], h[6]],
			hold_fed_yaw_abs: h[7],
			episodes_all: self.begun,
			episodes_hold: self.eps_hold,
		}
	}
}
