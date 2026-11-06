//! Optimal-control teachers for DAGGER, ported to Rust (hand-rolled, NO deps).
//!
//! Mirrors `src/wnn/control/optimal.py` (LQRController + MPCController) so the
//! accelerated phased-GA DAGGER loop can imitate an optimal-control teacher, not
//! just PID. Everything — plant calibration, the Riccati solve, and (for MPC) the
//! per-step QP — runs in Rust with zero external crates:
//!   * plant b-gains: `controller::calibrate_control_gains_rs` (steps AttitudeSim);
//!   * Riccati: the discrete recursion P ← Q + AᵀPA − AᵀPB(R+BᵀPB)⁻¹BᵀPA iterated
//!     to a fixed point — the only inverse is 3×3 (no eigensolver, no LAPACK). It
//!     yields BOTH the discrete-LQR gain and the MPC terminal cost-to-go;
//!   * MPC QP: the horizon problem is condensed to 3N control vars with box bounds
//!     and solved EXACTLY (box_qp.rs: primal-dual active set, primal active-set
//!     fallback, Cholesky on the free block), warm-started, with an unconstrained
//!     fast path (when the ±authority box does not bind the solution is the LQR
//!     feedback). Until 01/11/2026 this was projected FISTA capped at 200
//!     iterations, which exited at the cap on 93-95% of constrained solves.
//!
//! The teachers expose the same `step_rs(q, gyro, target) -> [f64;4]` interface as
//! `AttitudePidRs`, so they drop into `rollout_and_label_rs` unchanged. Being
//! label generators (not the deployed artifact — the WNN is), a little iteration is
//! fine here.

use crate::controller::{
	calibrate_control_gains_rs, clamp_f64, mix_to_motors_f64, quat_to_euler_f64, wrap_angle_f64,
	AttitudePidRs,
};
use crate::actuator_lag::{motor_lag_alpha, ActuatorLag};
use crate::box_qp::BoxQpSolver;
use pyo3::prelude::*;

// ===========================================================================
// Minimal dense f64 matrix (row-major). Only what the teachers need.
// ===========================================================================

#[derive(Clone)]
pub(crate) struct Mat
{
	r: usize,
	c: usize,
	d: Vec<f64>,
}

impl Mat
{
	fn zeros(r: usize, c: usize) -> Self
	{
		Mat {
			r,
			c,
			d: vec![0.0; r * c],
		}
	}
	fn identity(n: usize) -> Self
	{
		let mut m = Mat::zeros(n, n);
		for i in 0..n
		{
			m.set(i, i, 1.0);
		}
		m
	}
	#[inline]
	fn get(&self, i: usize, j: usize) -> f64
	{
		self.d[i * self.c + j]
	}
	#[inline]
	fn set(&mut self, i: usize, j: usize, v: f64)
	{
		self.d[i * self.c + j] = v;
	}
	fn transpose(&self) -> Mat
	{
		let mut t = Mat::zeros(self.c, self.r);
		for i in 0..self.r
		{
			for j in 0..self.c
			{
				t.set(j, i, self.get(i, j));
			}
		}
		t
	}
	fn matmul(&self, o: &Mat) -> Mat
	{
		assert_eq!(self.c, o.r, "matmul shape");
		let mut out = Mat::zeros(self.r, o.c);
		for i in 0..self.r
		{
			for k in 0..self.c
			{
				let a = self.get(i, k);
				if a == 0.0
				{
					continue;
				}
				for j in 0..o.c
				{
					let v = out.get(i, j) + a * o.get(k, j);
					out.set(i, j, v);
				}
			}
		}
		out
	}
	fn add(&self, o: &Mat) -> Mat
	{
		let mut out = self.clone();
		for x in 0..self.d.len()
		{
			out.d[x] += o.d[x];
		}
		out
	}
	fn sub(&self, o: &Mat) -> Mat
	{
		let mut out = self.clone();
		for x in 0..self.d.len()
		{
			out.d[x] -= o.d[x];
		}
		out
	}
	fn scale(&self, s: f64) -> Mat
	{
		let mut out = self.clone();
		for x in out.d.iter_mut()
		{
			*x *= s;
		}
		out
	}
	fn max_abs_diff(&self, o: &Mat) -> f64
	{
		let mut m = 0.0f64;
		for x in 0..self.d.len()
		{
			let dd = (self.d[x] - o.d[x]).abs();
			if dd > m
			{
				m = dd;
			}
		}
		m
	}
	/// Matrix × column vector (len = self.c) → column vector (len = self.r).
	fn mul_vec(&self, v: &[f64]) -> Vec<f64>
	{
		let mut out = vec![0.0; self.r];
		self.mul_vec_into(v, &mut out);
		out
	}
	/// mul_vec into a caller buffer (len = self.r) — same arithmetic order.
	fn mul_vec_into(&self, v: &[f64], out: &mut [f64])
	{
		assert_eq!(self.c, v.len(), "mul_vec shape");
		assert_eq!(self.r, out.len(), "mul_vec out shape");
		for i in 0..self.r
		{
			let mut acc = 0.0;
			for j in 0..self.c
			{
				acc += self.get(i, j) * v[j];
			}
			out[i] = acc;
		}
	}
}

/// General square-matrix inverse via Gauss-Jordan with partial pivoting.
/// Returns None if singular. Used for the MPC condensed Hessian (3N×3N).
fn inverse(a: &Mat) -> Option<Mat>
{
	assert_eq!(a.r, a.c, "inverse: square");
	let n = a.r;
	let mut m = a.clone();
	let mut inv = Mat::identity(n);
	for col in 0..n
	{
		// pivot = largest |value| in this column at/below the diagonal.
		let mut piv = col;
		let mut best = m.get(col, col).abs();
		for r in (col + 1)..n
		{
			let v = m.get(r, col).abs();
			if v > best
			{
				best = v;
				piv = r;
			}
		}
		if best < 1e-14
		{
			return None;
		}
		if piv != col
		{
			for j in 0..n
			{
				let (a1, a2) = (m.get(col, j), m.get(piv, j));
				m.set(col, j, a2);
				m.set(piv, j, a1);
				let (b1, b2) = (inv.get(col, j), inv.get(piv, j));
				inv.set(col, j, b2);
				inv.set(piv, j, b1);
			}
		}
		let d = m.get(col, col);
		for j in 0..n
		{
			m.set(col, j, m.get(col, j) / d);
			inv.set(col, j, inv.get(col, j) / d);
		}
		for r in 0..n
		{
			if r == col
			{
				continue;
			}
			let f = m.get(r, col);
			if f == 0.0
			{
				continue;
			}
			for j in 0..n
			{
				m.set(r, j, m.get(r, j) - f * m.get(col, j));
				inv.set(r, j, inv.get(r, j) - f * inv.get(col, j));
			}
		}
	}
	Some(inv)
}

/// Discrete algebraic Riccati equation via the backward recursion
///   P ← Q + AᵀPA − AᵀPB (R + BᵀPB)⁻¹ BᵀPA
/// iterated from P=Q to a fixed point. (R+BᵀPB) is m×m (m=3 controls) so its
/// inverse is tiny. Mirrors scipy.linalg.solve_discrete_are for our plant.
fn dare(a: &Mat, b: &Mat, q: &Mat, r: &Mat) -> Mat
{
	let at = a.transpose();
	let bt = b.transpose();
	let mut p = q.clone();
	for _ in 0..10_000
	{
		let atp = at.matmul(&p); // AᵀP
		let atpa = atp.matmul(a); // AᵀPA
		let atpb = atp.matmul(b); // AᵀPB
		let btpb = bt.matmul(&p).matmul(b); // BᵀPB
		let mid = r.add(&btpb); // R + BᵀPB  (m×m)
		let mid_inv = match inverse(&mid)
		{
			Some(x) => x,
			None => break,
		};
		let btpa = bt.matmul(&p).matmul(a); // BᵀPA
		let corr = atpb.matmul(&mid_inv).matmul(&btpa); // AᵀPB (R+BᵀPB)⁻¹ BᵀPA
		let p_new = q.add(&atpa).sub(&corr);
		let delta = p_new.max_abs_diff(&p);
		p = p_new;
		if delta < 1e-12
		{
			break;
		}
	}
	p
}

// ===========================================================================
// Attitude linear model: 6-state double-integrator (mirrors optimal.py).
//   x = [roll,pitch,yaw, p,q,r],  ẋ = A x + B u,  A[0,3]=A[1,4]=A[2,5]=1,
//   B[3,0]=b_roll, B[4,1]=b_pitch, B[5,2]=b_yaw.
// Returns CONTINUOUS (A, B). Callers discretize with dt.
// ===========================================================================

fn attitude_linear_model(b: [f64; 3]) -> (Mat, Mat)
{
	let mut a = Mat::zeros(6, 6);
	a.set(0, 3, 1.0);
	a.set(1, 4, 1.0);
	a.set(2, 5, 1.0);
	let mut bm = Mat::zeros(6, 3);
	bm.set(3, 0, b[0]);
	bm.set(4, 1, b[1]);
	bm.set(5, 2, b[2]);
	(a, bm)
}

// ===========================================================================
// ABI 34 (05/11/2026): LAG-AWARE design model. Under --motor-lag-s T the plant
// filters every command, ua' = (1−α)·ua + α·u with α = 4dt/T (= dt/τ, τ = T/4;
// AttitudeSim::apply_motor_lag), and the torque of the step is b·ua' — the
// FILTERED value. Discretized exactly as the plant steps (the attitude part is
// the lag-free model's I + A·dt):
//   angle' = angle + dt·rate
//   rate'  = rate  + dt·b·((1−α)·ua + α·u)
//   ua'    = (1−α)·ua + α·u
// x = [ (∫e,) angle, rate, ua ] per axis block of 3 — 9 states, 12 with the LQI
// integral (∫e' = ∫e + dt·angle). α = 1 ⇒ the ua columns vanish and the model is
// the lag-free one (pinned by lag_alpha_one_collapses_to_lag_free).
// ===========================================================================

pub(crate) fn attitude_lag_model(b: [f64; 3], dt: f64, alpha: f64, integral: bool) -> (Mat, Mat)
{
	let off = if integral { 3 } else { 0 };
	let ns = off + 9;
	let mut a = Mat::identity(ns);
	let mut bm = Mat::zeros(ns, 3);
	for i in 0..3
	{
		if integral
		{
			a.set(i, off + i, dt);
		}
		a.set(off + i, off + 3 + i, dt);
		a.set(off + 3 + i, off + 6 + i, dt * b[i] * (1.0 - alpha));
		a.set(off + 6 + i, off + 6 + i, 1.0 - alpha);
		bm.set(off + 3 + i, i, dt * b[i] * alpha);
		bm.set(off + 6 + i, i, alpha);
	}
	(a, bm)
}

/// ABI 35 (05/11/2026): the share of the effort weight priced on the COMMAND
/// under motor lag. RULE (fixed in advance, never tuned on results): the lag-free
/// teachers price the effort the rotors DELIVER, r·u²; under lag the delivered
/// effort is the filtered actuator ua, so the lag-aware cost keeps the SAME total
/// effort budget r and splits it —
///   (1 − ε)·r on the delivered actuator ua  +  ε·r on the command u,  ε = 0.1.
/// Because the split sums to r, the cost → q·x² + r·u² as τ → 0 (ua → u): the
/// design converges onto the lag-free one (care_lag_design_converges_to_closed_form,
/// lag_alpha_one_collapses_to_lag_free — exact at α = 1, where pricing u_k now and
/// ua_{k+1} = u_k next step sums to r·u_k²). ε is "small" by construction (one
/// tenth of the budget, an order of magnitude below the delivered share): it keeps
/// R positive definite (the Riccati problems need it) without letting the command
/// cost dominate. ABI 34 priced the COMMAND only (0 on ua, r on u), i.e. the
/// teachers minimised what they asked for, not what the rotors produced.
pub(crate) const LAG_CMD_SHARE: f64 = 0.1;

/// The lag-aware state cost: q_int (when integral) / q_att / q_rate as the
/// lag-free teachers weight them, and (1 − LAG_CMD_SHARE)·r_ctrl on the actuator
/// state — the effort the rotors actually deliver (see LAG_CMD_SHARE).
pub(crate) fn lag_state_cost(q_att: f64, q_rate: f64, q_int: Option<f64>, r_ctrl: f64) -> Mat
{
	let mut d: Vec<f64> = Vec::with_capacity(12);
	if let Some(qi) = q_int
	{
		d.extend_from_slice(&[qi; 3]);
	}
	d.extend_from_slice(&[q_att; 3]);
	d.extend_from_slice(&[q_rate; 3]);
	d.extend_from_slice(&[(1.0 - LAG_CMD_SHARE) * r_ctrl; 3]);
	let mut m = Mat::zeros(d.len(), d.len());
	for (i, v) in d.iter().enumerate()
	{
		m.set(i, i, *v);
	}
	m
}

/// The lag-aware command cost: LAG_CMD_SHARE·r_ctrl per axis (see LAG_CMD_SHARE).
pub(crate) fn lag_cmd_cost(r_ctrl: f64) -> Mat
{
	diag3([LAG_CMD_SHARE * r_ctrl; 3])
}

/// DARE by the structured doubling algorithm (Chu, Fan, Lin 2005):
///   G₀ = B R⁻¹ Bᵀ, H₀ = Q, A₀ = A;  W = I + G H
///   A ← A W⁻¹ A,  G ← G + A W⁻¹ G Aᵀ,  H ← H + Aᵀ H W⁻¹ A
/// H → the stabilizing P QUADRATICALLY. The plain recursion `dare` converges at
/// the closed loop's slowest pole squared per sweep — at dt = 1 ms the LQI
/// integral pole needs > 10⁴ sweeps (its cap) — so the lag-aware designs use
/// this. `dare` stays as is for the lag-free MPC terminal cost (T = 0 is
/// bit-identical to ABI 33).
pub(crate) fn dare_doubling(a: &Mat, b: &Mat, q: &Mat, r: &Mat) -> Mat
{
	let n = a.r;
	let eye = Mat::identity(n);
	let r_inv = inverse(r).expect("dare_doubling: R singular");
	let mut ak = a.clone();
	let mut gk = b.matmul(&r_inv).matmul(&b.transpose());
	let mut hk = q.clone();
	for _ in 0..64
	{
		let w_inv = inverse(&eye.add(&gk.matmul(&hk))).expect("dare_doubling: I+GH singular");
		let aw = ak.matmul(&w_inv);
		let a_next = aw.matmul(&ak);
		let g_next = gk.add(&aw.matmul(&gk).matmul(&ak.transpose()));
		let h_next = hk.add(&ak.transpose().matmul(&hk).matmul(&w_inv).matmul(&ak));
		let scale = h_next.d.iter().fold(1.0f64, |m, v| m.max(v.abs()));
		let delta = h_next.max_abs_diff(&hk);
		ak = a_next;
		gk = g_next;
		hk = h_next;
		if delta <= 1e-13 * scale
		{
			break;
		}
	}
	hk
}

/// Discrete LQR gain K = (R + BᵀPB)⁻¹ BᵀPA for the control law u = −K x.
pub(crate) fn dlqr_gain(a: &Mat, b: &Mat, q: &Mat, r: &Mat) -> Mat
{
	let p = dare_doubling(a, b, q, r);
	let bt = b.transpose();
	let mid = r.add(&bt.matmul(&p).matmul(b));
	inverse(&mid).expect("dlqr_gain: R+BᵀPB singular").matmul(&bt.matmul(&p).matmul(a))
}

/// ABI 34: the CONTINUOUS actuator-augmented model for the CARE-designed
/// teachers (LQR, LQI — their lag-free selves are continuous-CARE closed forms,
/// so the lag-aware ones stay in that family and converge onto them as τ → 0):
///   angle' = rate,  rate' = b·ua,  ua' = (u − ua)/τ   (+ ∫e' = angle for LQI)
/// i.e. A gains the −I/τ actuator block and B = [0; 0; I/τ].
pub(crate) fn attitude_lag_model_continuous(b: [f64; 3], tau: f64, integral: bool) -> (Mat, Mat)
{
	let off = if integral { 3 } else { 0 };
	let ns = off + 9;
	let mut a = Mat::zeros(ns, ns);
	let mut bm = Mat::zeros(ns, 3);
	for i in 0..3
	{
		if integral
		{
			a.set(i, off + i, 1.0);
		}
		a.set(off + i, off + 3 + i, 1.0);
		a.set(off + 3 + i, off + 6 + i, b[i]);
		a.set(off + 6 + i, off + 6 + i, -1.0 / tau);
		bm.set(off + 6 + i, i, 1.0 / tau);
	}
	(a, bm)
}

/// The actuator time constant τ = T/4 for the plant's (clamped) settling time.
pub(crate) fn motor_lag_tau(settling_s: f32, dt: f32) -> f64
{
	(settling_s.max(4.0 * dt) as f64) / 4.0
}

/// CARE  AᵀP + PA − PBR⁻¹BᵀP + Q = 0  by Newton–Kleinman (Kleinman 1968):
/// K₀ from the discrete DARE of a fine Euler discretization (h = 1e-5, Q·h, R·h
/// — a stabilizing approximation of the continuous gain), then
///   (A − BKₖ)ᵀPₖ + Pₖ(A − BKₖ) = −(Q + KₖᵀRKₖ),  Kₖ₊₁ = R⁻¹BᵀPₖ
/// to a fixed point (quadratic convergence). The Lyapunov solve is the n²×n²
/// Kronecker system — n ≤ 12 here, so ≤ 144 unknowns, no eigensolver needed.
pub(crate) fn care(a: &Mat, b: &Mat, q: &Mat, r: &Mat) -> Mat
{
	let n = a.r;
	let h = 1e-5;
	let ah = Mat::identity(n).add(&a.scale(h));
	let r_inv = inverse(r).expect("care: R singular");
	let mut k = dlqr_gain(&ah, &b.scale(h), &q.scale(h), &r.scale(h));
	let mut p = Mat::zeros(n, n);
	for _ in 0..50
	{
		let acl = a.sub(&b.matmul(&k));
		let m = q.add(&k.transpose().matmul(r).matmul(&k));
		p = lyapunov(&acl, &m);
		let k_next = r_inv.matmul(&b.transpose()).matmul(&p);
		let scale = k_next.d.iter().fold(1.0f64, |mx, v| mx.max(v.abs()));
		let delta = k_next.max_abs_diff(&k);
		k = k_next;
		if delta <= 1e-13 * scale
		{
			break;
		}
	}
	p
}

/// Solve AᵀP + PA = −M (A Hurwitz) via the Kronecker form on vec(P), row-major.
fn lyapunov(a: &Mat, m: &Mat) -> Mat
{
	let n = a.r;
	let nn = n * n;
	let mut l = Mat::zeros(nn, nn);
	for i in 0..n
	{
		for j in 0..n
		{
			let row = i * n + j;
			for k in 0..n
			{
				// (AᵀP)ᵢⱼ = Σₖ Aₖᵢ Pₖⱼ ;  (PA)ᵢⱼ = Σₖ Pᵢₖ Aₖⱼ
				l.set(row, k * n + j, l.get(row, k * n + j) + a.get(k, i));
				l.set(row, i * n + k, l.get(row, i * n + k) + a.get(k, j));
			}
		}
	}
	let rhs: Vec<f64> = m.d.iter().map(|v| -v).collect();
	let x = inverse(&l).expect("lyapunov: singular (closed loop not Hurwitz?)").mul_vec(&rhs);
	let mut p = Mat::zeros(n, n);
	p.d.copy_from_slice(&x);
	// Symmetrize away round-off.
	p.add(&p.transpose()).scale(0.5)
}

/// Continuous LQR gain K = R⁻¹BᵀP (u = −K x) from the CARE.
pub(crate) fn clqr_gain(a: &Mat, b: &Mat, q: &Mat, r: &Mat) -> Mat
{
	let p = care(a, b, q, r);
	inverse(r).expect("clqr_gain: R singular").matmul(&b.transpose()).matmul(&p)
}

/// ABI 34: a lag-aware state-feedback law — the augmented-model gain and the
/// actuator estimate it multiplies (LQR: x = [e, rate, ua]; LQI: [∫e, e, rate, ua]).
#[derive(Clone)]
pub(crate) struct LagFeedback
{
	pub(crate) k: Mat,
	pub(crate) act: ActuatorLag,
}

fn diag6(d: [f64; 6]) -> Mat
{
	let mut m = Mat::zeros(6, 6);
	for i in 0..6
	{
		m.set(i, i, d[i]);
	}
	m
}
fn diag3(d: [f64; 3]) -> Mat
{
	let mut m = Mat::zeros(3, 3);
	for i in 0..3
	{
		m.set(i, i, d[i]);
	}
	m
}

/// State error vector x = [roll-tr, pitch-tp, yaw-ty, p, q, r] (angles wrapped).
#[inline]
fn state_error(q: [f32; 4], gyro: [f32; 3], target_rpy: [f32; 3]) -> [f64; 6]
{
	let (roll, pitch, yaw) = quat_to_euler_f64(q);
	[
		wrap_angle_f64(roll - target_rpy[0] as f64),
		wrap_angle_f64(pitch - target_rpy[1] as f64),
		wrap_angle_f64(yaw - target_rpy[2] as f64),
		gyro[0] as f64,
		gyro[1] as f64,
		gyro[2] as f64,
	]
}

/// ABI 34: the episode's FIRST decision under motor lag. The plant seeds its
/// lag filter AT that command (AttitudeSim::apply_motor_lag: the vehicle is
/// already flying at it), so for this one step the actuator state IS the
/// command, ua = u — not the 0 the estimate holds before anything flew. The
/// self-consistent action is the law's fixed point under that identity: with
/// z the state without the actuator block (K = [K_z | K_ua]) and the actuator
/// entering as `act_sign`·v,
///   v = K_z·z + K_ua·(act_sign·v)   ⇒   (I − act_sign·K_ua)·v = K_z·z.
/// LQR/LQI apply u = −v (act_sign = −1); MPC applies u = v (act_sign = +1).
/// Planning step 0 with ua = 0 instead commands the lead a lagged actuator
/// needs, the plant applies it UNLAGGED and then holds it for ~τ: measured on
/// cf21 at T = 0.30, lag-aware LQR mean err 3.94° → 1.09° with this step
/// (abi34_tests::lag_aware_teachers_hold_the_lagged_plant).
fn first_step_fixed_point(k: &Mat, z: &[f64], act_sign: f64) -> Vec<f64>
{
	let nz = z.len();
	let mut lhs = Mat::identity(3);
	let mut rhs = vec![0.0f64; 3];
	for i in 0..3
	{
		for j in 0..nz
		{
			rhs[i] += k.get(i, j) * z[j];
		}
		for j in 0..3
		{
			lhs.set(i, j, lhs.get(i, j) - act_sign * k.get(i, nz + j));
		}
	}
	inverse(&lhs).expect("first-step fixed point: I − s·K_ua singular").mul_vec(&rhs)
}

/// ABI 34: append the actuator estimate to a state vector (augmented models).
#[inline]
fn with_actuator(x: &[f64], ua: [f64; 3]) -> Vec<f64>
{
	let mut z = Vec::with_capacity(x.len() + 3);
	z.extend_from_slice(x);
	z.extend_from_slice(&ua);
	z
}

// ===========================================================================
// LQR teacher — u = clamp(−Kx, ±authority), K from the discrete Riccati.
// ===========================================================================

/// Default LQR/MPC cost weights (match optimal.py defaults: q_att=12, q_rate=1,
/// r_ctrl=1). authority/hover default to the PID teacher's (0.4 / 0.5).
pub(crate) const Q_ATT: f64 = 12.0;
pub(crate) const Q_RATE: f64 = 1.0;
pub(crate) const R_CTRL: f64 = 1.0;

#[pyclass]
pub struct AttitudeLqrRs
{
	k: Mat, // 3×6 feedback gain
	hover: f64,
	authority: f64,
	/// ABI 34: Some ⇔ motor lag on — the 3×9 gain on [e, rate, ua] replaces `k`.
	lag: Option<LagFeedback>,
}

impl AttitudeLqrRs
{
	/// Build from explicit sim params (so the plant model matches the controlled
	/// sim) + cost weights + hover/authority.
	#[allow(clippy::too_many_arguments)]
	pub(crate) fn build(
		dt: f32,
		arm_length: f32,
		k_thrust: f32,
		k_drag: f32,
		inertia: [f32; 3],
		gravity: f32,
		hover: f64,
		authority: f64,
		q_att: f64,
		q_rate: f64,
		r_ctrl: f64,
	) -> Self
	{
		let b = calibrate_control_gains_rs(
			dt, arm_length, k_thrust, k_drag, inertia, gravity, hover, 0.05,
		);
		// The plant is DECOUPLED per axis (roll/pitch/yaw are independent double
		// integrators ẍ = b·u), so the continuous-CARE LQR has a CLOSED FORM per
		// axis — no Riccati iteration, and bit-faithful to scipy solve_continuous_are:
		//   k1 = √(q_att/r_ctrl)                      (angle gain, axis-independent)
		//   k2 = √((2·√(q_att·r_ctrl)/b_i + q_rate)/r_ctrl)   (rate gain, per-axis via b_i)
		// (Discrete DARE would give the discrete-time optimum — a slightly softer,
		// different controller; we want the same teacher Python uses.)
		let k1 = (q_att / r_ctrl).sqrt();
		let mut k = Mat::zeros(3, 6);
		for axis in 0..3
		{
			let k2 = ((2.0 * (q_att * r_ctrl).sqrt() / b[axis] + q_rate) / r_ctrl).sqrt();
			k.set(axis, axis, k1); // angle error → u
			k.set(axis, axis + 3, k2); // body rate → u
		}
		AttitudeLqrRs {
			k,
			hover,
			authority,
			lag: None,
		}
	}

	/// ABI 34: the LAG-AWARE LQR for a plant with motor-lag settling time
	/// `motor_lag_s` (> 0): continuous LQR (CARE) on the 9-state actuator-
	/// augmented model, same q_att/q_rate, r_ctrl split per LAG_CMD_SHARE — the
	/// lag-free teacher's own design family, so as τ → 0 it converges onto the
	/// closed form above (pinned by care_lag_design_converges_to_closed_form).
	#[allow(clippy::too_many_arguments)]
	pub(crate) fn build_lagged(
		dt: f32,
		arm_length: f32,
		k_thrust: f32,
		k_drag: f32,
		inertia: [f32; 3],
		gravity: f32,
		hover: f64,
		authority: f64,
		q_att: f64,
		q_rate: f64,
		r_ctrl: f64,
		motor_lag_s: f32,
	) -> Self
	{
		let mut t = Self::build(
			dt, arm_length, k_thrust, k_drag, inertia, gravity, hover, authority, q_att, q_rate, r_ctrl,
		);
		let b = calibrate_control_gains_rs(
			dt, arm_length, k_thrust, k_drag, inertia, gravity, hover, 0.05,
		);
		let (a, bm) = attitude_lag_model_continuous(b, motor_lag_tau(motor_lag_s, dt), false);
		t.lag = Some(LagFeedback {
			k: clqr_gain(&a, &bm, &lag_state_cost(q_att, q_rate, None, r_ctrl), &lag_cmd_cost(r_ctrl)),
			act: ActuatorLag::new(motor_lag_s, dt),
		});
		t
	}

	/// ABI 34: the pwm applied last step (commanded convention) → actuator
	/// estimate. No-op without lag (LQR is memoryless there).
	pub(crate) fn observe_applied(&mut self, applied_pwm: [f64; 4])
	{
		if let Some(l) = self.lag.as_mut()
		{
			l.act.observe(applied_pwm);
		}
	}

	/// Native f64 step (mirrors AttitudePidRs::step_rs signature).
	pub fn step_rs(&mut self, q: [f32; 4], gyro: [f32; 3], target_rpy: [f32; 3]) -> [f64; 4]
	{
		let x = state_error(q, gyro, target_rpy);
		let u = match self.lag.as_ref()
		{
			// ABI 34: K·[e, rate, ua] on the augmented model.
			Some(l) if l.act.seeded() => l.k.mul_vec(&with_actuator(&x, l.act.state())),
			Some(l) => first_step_fixed_point(&l.k, &x, -1.0),
			None => self.k.mul_vec(&x), // K x  (3,)
		};
		let a = self.authority;
		mix_to_motors_f64(
			self.hover,
			clamp_f64(-u[0], -a, a),
			clamp_f64(-u[1], -a, a),
			clamp_f64(-u[2], -a, a),
		)
	}
}

#[pymethods]
impl AttitudeLqrRs
{
	/// Python constructor for parity testing. Defaults match optimal.py +
	/// AttitudeSim::new defaults.
	#[new]
	#[pyo3(signature = (
		dt = 0.001, arm_length = 0.075, k_thrust = 2.4, k_drag = 0.05,
		inertia = [0.0023, 0.0023, 0.0046], gravity = 9.81,
		hover = 0.5, authority = 0.4, q_att = 12.0, q_rate = 1.0, r_ctrl = 1.0
	))]
	#[allow(clippy::too_many_arguments)]
	pub fn new(
		dt: f32,
		arm_length: f32,
		k_thrust: f32,
		k_drag: f32,
		inertia: [f32; 3],
		gravity: f32,
		hover: f64,
		authority: f64,
		q_att: f64,
		q_rate: f64,
		r_ctrl: f64,
	) -> Self
	{
		Self::build(
			dt, arm_length, k_thrust, k_drag, inertia, gravity, hover, authority, q_att, q_rate, r_ctrl,
		)
	}
	pub fn reset(&mut self)
	{
		// Memoryless without lag; with it, only the actuator estimate (ABI 34).
		if let Some(l) = self.lag.as_mut()
		{
			l.act.reset();
		}
	}
	/// One step → 4 motor PWMs (f32), for Python parity vs optimal.LQRController.
	fn step(&mut self, q: [f32; 4], gyro: [f32; 3], target_rpy: [f32; 3]) -> [f32; 4]
	{
		let p = self.step_rs(q, gyro, target_rpy);
		[p[0] as f32, p[1] as f32, p[2] as f32, p[3] as f32]
	}
	/// Flattened 3×6 gain, row-major (for parity vs optimal.LQRController.K).
	fn gain(&self) -> Vec<f64>
	{
		self.k.d.clone()
	}
}

// ===========================================================================
// LQI teacher — integral-augmented LQR: u = clamp(−Kx − Ki·∫e, ±authority).
//
// The stateful upgrade of AttitudeLqrRs (Phase-4 state-pressure work, 18/07/2026):
// LQR's optimal feedback PLUS the integral channel PID has — so under a constant
// torque bias (DisturbanceConfig D1) the teacher cancels the offset LQR/MPC
// cannot perceive, and its action becomes history-dependent (the property that
// generates DAGGER split-pressure conflicts for the student).
//
// Gains per axis by spectral factorization of the augmented plant [∫e, e, ė],
// ẍ = b·u (still closed-form-ish, NO Riccati matrices): the stable factor
// Δ(s) = s³ + c2·s² + c1·s + c0 of
//   Δ(s)Δ(−s)|_{s=jω} = ω⁶ + (b²/r)(q_rate·ω⁴ + q_att·ω² + q_int)
// satisfies  c0 = b·√(q_int/r)  exactly, and
//   c1 = √(b²·q_att/r + 2·c0·c2),   c2 = √(b²·q_rate/r + 2·c1)
// — a monotone 2-variable fixed point (converges in a handful of iterations).
// Gains: k_int = c0/b (= √(q_int/r), axis-independent), k_att = c1/b,
// k_rate = c2/b. At q_int = 0 this reduces EXACTLY to AttitudeLqrRs's k1/k2
// (the built-in parity check). Anti-windup clamp mirrors the PID teacher's
// i_clamp = 0.5, and integrals()/i_clamps() feed the Option-A --state-integral
// target just like PID's.
// ===========================================================================

/// Default integral weight for the LQI teacher (conservative; k_int = √(q_int/r)).
pub(crate) const Q_INT: f64 = 1.0;

#[pyclass]
pub struct AttitudeLqiRs
{
	k: Mat,             // 3×6 proportional+rate feedback (same layout as LQR)
	k_int: [f64; 3],    // integral gains per axis
	integral: [f64; 3], // ∫angle-error dt, anti-windup clamped
	i_clamp: f64,
	dt: f64,
	hover: f64,
	authority: f64,
	/// ABI 34: Some ⇔ motor lag on — the 3×12 gain on [∫e, e, rate, ua]
	/// replaces `k`/`k_int` (the integral state and its clamp are unchanged).
	lag: Option<LagFeedback>,
}

impl AttitudeLqiRs
{
	/// Build from explicit sim params (plant-matched like AttitudeLqrRs::build).
	#[allow(clippy::too_many_arguments)]
	pub(crate) fn build(
		dt: f32,
		arm_length: f32,
		k_thrust: f32,
		k_drag: f32,
		inertia: [f32; 3],
		gravity: f32,
		hover: f64,
		authority: f64,
		q_att: f64,
		q_rate: f64,
		r_ctrl: f64,
		q_int: f64,
	) -> Self
	{
		let b = calibrate_control_gains_rs(
			dt, arm_length, k_thrust, k_drag, inertia, gravity, hover, 0.05,
		);
		let mut k = Mat::zeros(3, 6);
		let mut k_int = [0.0; 3];
		for axis in 0..3
		{
			let bb = b[axis];
			let c0 = bb * (q_int / r_ctrl).sqrt();
			// Fixed point seeded at the q_int=0 (plain LQR) solution.
			let mut c1 = bb * (q_att / r_ctrl).sqrt();
			let mut c2 = (bb * bb * q_rate / r_ctrl + 2.0 * c1).sqrt();
			for _ in 0..64
			{
				let n1 = (bb * bb * q_att / r_ctrl + 2.0 * c0 * c2).sqrt();
				let n2 = (bb * bb * q_rate / r_ctrl + 2.0 * n1).sqrt();
				let done = (n1 - c1).abs() < 1e-12 && (n2 - c2).abs() < 1e-12;
				c1 = n1;
				c2 = n2;
				if done
				{
					break;
				}
			}
			k.set(axis, axis, c1 / bb); // angle error → u
			k.set(axis, axis + 3, c2 / bb); // body rate → u
			k_int[axis] = c0 / bb; // = √(q_int/r_ctrl)
		}
		AttitudeLqiRs {
			k,
			k_int,
			integral: [0.0; 3],
			i_clamp: 0.5,
			dt: dt as f64,
			hover,
			authority,
			lag: None,
		}
	}

	/// ABI 34: the LAG-AWARE LQI — continuous LQR (CARE) on the 12-state
	/// augmented model [∫e, e, rate, ua], same q_int/q_att/q_rate/r_ctrl, 0 on
	/// the actuator state; converges onto the spectral-factor gains as τ → 0.
	#[allow(clippy::too_many_arguments)]
	pub(crate) fn build_lagged(
		dt: f32,
		arm_length: f32,
		k_thrust: f32,
		k_drag: f32,
		inertia: [f32; 3],
		gravity: f32,
		hover: f64,
		authority: f64,
		q_att: f64,
		q_rate: f64,
		r_ctrl: f64,
		q_int: f64,
		motor_lag_s: f32,
	) -> Self
	{
		let mut t = Self::build(
			dt, arm_length, k_thrust, k_drag, inertia, gravity, hover, authority, q_att, q_rate, r_ctrl,
			q_int,
		);
		let b = calibrate_control_gains_rs(
			dt, arm_length, k_thrust, k_drag, inertia, gravity, hover, 0.05,
		);
		let (a, bm) = attitude_lag_model_continuous(b, motor_lag_tau(motor_lag_s, dt), true);
		t.lag = Some(LagFeedback {
			k: clqr_gain(&a, &bm, &lag_state_cost(q_att, q_rate, Some(q_int), r_ctrl), &lag_cmd_cost(r_ctrl)),
			act: ActuatorLag::new(motor_lag_s, dt),
		});
		t
	}

	/// ABI 34: the pwm applied last step → actuator estimate (no-op without lag).
	pub(crate) fn observe_applied(&mut self, applied_pwm: [f64; 4])
	{
		if let Some(l) = self.lag.as_mut()
		{
			l.act.observe(applied_pwm);
		}
	}

	/// Native f64 step (mirrors AttitudeLqrRs::step_rs, plus the integral state).
	pub fn step_rs(&mut self, q: [f32; 4], gyro: [f32; 3], target_rpy: [f32; 3]) -> [f64; 4]
	{
		let x = state_error(q, gyro, target_rpy);
		for axis in 0..3
		{
			self.integral[axis] = clamp_f64(
				self.integral[axis] + x[axis] * self.dt,
				-self.i_clamp,
				self.i_clamp,
			);
		}
		if let Some(l) = self.lag.as_ref()
		{
			// ABI 34: u = −K·[∫e, e, rate, ua] — the integral gain lives in K.
			let mut z = Vec::with_capacity(12);
			z.extend_from_slice(&self.integral);
			z.extend_from_slice(&x);
			let u = if l.act.seeded()
			{
				z.extend_from_slice(&l.act.state());
				l.k.mul_vec(&z)
			}
			else
			{
				first_step_fixed_point(&l.k, &z, -1.0)
			};
			let a = self.authority;
			return mix_to_motors_f64(
				self.hover,
				clamp_f64(-u[0], -a, a),
				clamp_f64(-u[1], -a, a),
				clamp_f64(-u[2], -a, a),
			);
		}
		let u = self.k.mul_vec(&x); // K x  (3,)
		let a = self.authority;
		mix_to_motors_f64(
			self.hover,
			clamp_f64(-(u[0] + self.k_int[0] * self.integral[0]), -a, a),
			clamp_f64(-(u[1] + self.k_int[1] * self.integral[1]), -a, a),
			clamp_f64(-(u[2] + self.k_int[2] * self.integral[2]), -a, a),
		)
	}

	/// Teacher integral (roll,pitch,yaw) — the Option-A --state-integral target.
	pub fn integrals(&self) -> [f32; 3]
	{
		[
			self.integral[0] as f32,
			self.integral[1] as f32,
			self.integral[2] as f32,
		]
	}
	/// Clamp magnitudes for normalizing the integral to [-1, 1].
	pub fn i_clamps(&self) -> [f32; 3]
	{
		[
			self.i_clamp as f32,
			self.i_clamp as f32,
			self.i_clamp as f32,
		]
	}
}

#[pymethods]
impl AttitudeLqiRs
{
	/// Python constructor for parity testing. Defaults match AttitudeLqrRs::new
	/// plus q_int (integral weight).
	#[new]
	#[pyo3(signature = (
		dt = 0.001, arm_length = 0.075, k_thrust = 2.4, k_drag = 0.05,
		inertia = [0.0023, 0.0023, 0.0046], gravity = 9.81,
		hover = 0.5, authority = 0.4, q_att = 12.0, q_rate = 1.0, r_ctrl = 1.0,
		q_int = 1.0
	))]
	#[allow(clippy::too_many_arguments)]
	pub fn new(
		dt: f32,
		arm_length: f32,
		k_thrust: f32,
		k_drag: f32,
		inertia: [f32; 3],
		gravity: f32,
		hover: f64,
		authority: f64,
		q_att: f64,
		q_rate: f64,
		r_ctrl: f64,
		q_int: f64,
	) -> Self
	{
		Self::build(
			dt, arm_length, k_thrust, k_drag, inertia, gravity, hover, authority, q_att, q_rate, r_ctrl,
			q_int,
		)
	}
	pub fn reset(&mut self)
	{
		self.integral = [0.0; 3];
		if let Some(l) = self.lag.as_mut()
		{
			l.act.reset();
		}
	}
	/// One step → 4 motor PWMs (f32), for Python parity testing.
	fn step(&mut self, q: [f32; 4], gyro: [f32; 3], target_rpy: [f32; 3]) -> [f32; 4]
	{
		let p = self.step_rs(q, gyro, target_rpy);
		[p[0] as f32, p[1] as f32, p[2] as f32, p[3] as f32]
	}
	/// Flattened 3×6 P+D gain, row-major (parity vs AttitudeLqrRs at q_int=0).
	fn gain(&self) -> Vec<f64>
	{
		self.k.d.clone()
	}
	/// Integral gains per axis (k_int = √(q_int/r_ctrl)).
	fn gain_int(&self) -> Vec<f64>
	{
		self.k_int.to_vec()
	}
}

// ===========================================================================
// Allocator-aware LQR teacher — overactuated Phase 2 (the residual baseline).
//
// The quad teachers end in the hard-coded '+' mixer (mix_to_motors). This
// teacher generalizes the SAME per-axis LQR to an N-rotor airframe by working
// in PHYSICAL torque units and handing the wrench to the classical damped
// pseudo-inverse allocator (overactuated.rs) built from the NOMINAL geometry:
//
//   x_err ──► u = clamp(−K·x, ±τ_max)  (N·m, per axis)
//         ──► wrench w = (τ; 0, 0, F_hover)
//         ──► pwm = geo.allocate(w)     (min-norm thrusts, √(T/k), [0,1])
//
// Gains use the same closed-form per-axis CARE as AttitudeLqrRs but with the
// torque plant b_i = 1/I_i (ẍ = τ/I) instead of the quad's calibrated
// pwm-space gain. τ_max defaults to the quad teacher's EQUIVALENT physical
// authority (authority 0.4 on the '+' mixer ≈ 4·L·k_thrust·hover·0.4 ≈
// 0.144 N·m with paper-#1 sim params) so octo/hex teacher aggressiveness is
// comparable to the quad baselines. F_hover defaults to the nominal
// geometry's collective thrust at hover PWM 0.5 (Σ kᵢ·0.25) — on a symmetric
// airframe the min-norm allocation then returns ≈0.5 per rotor at zero error.
//
// This is the DAGGER label generator for the WNN residual (Phase 2) — the
// deployed artifact stays the WNN; per-step 6×6 solves are fine here.
// ===========================================================================

/// The shared allocator-LQR BASELINE: gains + precomputed pinv rows in f32,
/// so the CPU batch scorer and the Metal kernel compute the SAME per-step
/// baseline PWM (matvec + sqrt — no per-step 6×6 solve). AllocLqrRs (the
/// DAGGER teacher) delegates here too: teacher ≡ eval baseline by
/// construction, the property the residual→0 sanity run (Phase 3) rests on.
///
/// GPU upload layout (buffer 28): [k1, k2x, k2y, k2z, tau_max, f_hover]
/// header then N×[m0..m5, k_thrust] rows — keep `to_gpu_blob` in lockstep
/// with `alloc_step` in controller_rollout.metal.
#[derive(Clone)]
pub struct AllocBaseline
{
	pub k1: f32,
	pub k2: [f32; 3],
	pub tau_max: f32,
	pub f_hover: f32,
	/// Per rotor: pinv row m0..m5 (+ nominal k_thrust) — T_i = mᵢ·w.
	pub rows: Vec<[f32; 7]>,
}

impl AllocBaseline
{
	/// Build from nominal geometry rows + LQR cost weights (same closed-form
	/// per-axis CARE as AttitudeLqrRs, torque plant b = 1/I).
	pub fn build(
		rows9: &[[f32; 9]],
		inertia: [f32; 3],
		q_att: f64,
		q_rate: f64,
		r_ctrl: f64,
		tau_max: f64,
		f_hover: Option<f64>,
		lambda: f32,
	) -> Result<Self, String>
	{
		let geo = crate::overactuated::RotorGeometry::from_rows(rows9)?;
		if tau_max <= 0.0
		{
			return Err(format!("tau_max must be > 0, got {tau_max}"));
		}
		let k1 = (q_att / r_ctrl).sqrt() as f32;
		let mut k2 = [0.0f32; 3];
		for axis in 0..3
		{
			let b = 1.0 / inertia[axis] as f64;
			k2[axis] = (((2.0 * (q_att * r_ctrl).sqrt() / b + q_rate) / r_ctrl).sqrt()) as f32;
		}
		// Hover collective: every rotor at PWM 0.5 on the nominal geometry.
		let f_hover =
			f_hover.unwrap_or_else(|| geo.rotors.iter().map(|r| (r.k_thrust * 0.25) as f64).sum()) as f32;
		let pinv = geo.allocation_pinv(lambda);
		let rows = pinv
			.iter()
			.zip(&geo.rotors)
			.map(|(m, r)| [m[0], m[1], m[2], m[3], m[4], m[5], r.k_thrust])
			.collect();
		Ok(AllocBaseline {
			k1,
			k2,
			tau_max: tau_max as f32,
			f_hover,
			rows,
		})
	}

	pub fn num_rotors(&self) -> usize
	{
		self.rows.len()
	}

	/// One baseline step, all-f32 — the bit-level template for the kernel's
	/// alloc_step: euler(q) → per-axis τ = clamp(k1·e − k2·rate, ±τ_max) →
	/// T = M·(τ; 0,0,F_hover) → pwm = √(max(T,0)/k) clamped [0,1].
	pub fn pwm(&self, q: [f32; 4], gyro: [f32; 3], target_rpy: [f32; 3], out: &mut [f32])
	{
		let (roll, pitch, yaw) = quat_to_euler_f64(q);
		let e = [
			wrap_angle_f64(target_rpy[0] as f64 - roll) as f32,
			wrap_angle_f64(target_rpy[1] as f64 - pitch) as f32,
			wrap_angle_f64(target_rpy[2] as f64 - yaw) as f32,
		];
		let t = self.tau_max;
		let wrench = [
			(self.k1 * e[0] - self.k2[0] * gyro[0]).clamp(-t, t),
			(self.k1 * e[1] - self.k2[1] * gyro[1]).clamp(-t, t),
			(self.k1 * e[2] - self.k2[2] * gyro[2]).clamp(-t, t),
			0.0,
			0.0,
			self.f_hover,
		];
		for (i, row) in self.rows.iter().enumerate()
		{
			let mut thrust = 0.0f32;
			for j in 0..6
			{
				thrust += row[j] * wrench[j];
			}
			out[i] = (thrust.max(0.0) / row[6]).sqrt().clamp(0.0, 1.0);
		}
	}

	/// Flat f32 blob for the GPU (buffer 28) — header + 7-float rotor rows.
	pub fn to_gpu_blob(&self) -> Vec<f32>
	{
		let mut v = Vec::with_capacity(6 + self.rows.len() * 7);
		v.extend_from_slice(&[
			self.k1,
			self.k2[0],
			self.k2[1],
			self.k2[2],
			self.tau_max,
			self.f_hover,
		]);
		for r in &self.rows
		{
			v.extend_from_slice(r);
		}
		v
	}
}

#[pyclass]
pub struct AllocLqrRs
{
	base: AllocBaseline, // NOMINAL geometry + gains (allocator side)
}

impl AllocLqrRs
{
	/// Plain-Rust constructor (house pattern: String errors; the pymethod wraps).
	pub(crate) fn build_core(
		rows: &[[f32; 9]],
		inertia: [f32; 3],
		q_att: f64,
		q_rate: f64,
		r_ctrl: f64,
		tau_max: f64,
		f_hover: Option<f64>,
		lambda: f32,
	) -> Result<Self, String>
	{
		Ok(AllocLqrRs {
			base: AllocBaseline::build(
				rows, inertia, q_att, q_rate, r_ctrl, tau_max, f_hover, lambda,
			)?,
		})
	}

	/// Native step: attitude state → N motor PWMs via LQR torque + allocation.
	pub fn step_alloc_rs(&mut self, q: [f32; 4], gyro: [f32; 3], target_rpy: [f32; 3]) -> Vec<f64>
	{
		let mut out = vec![0.0f32; self.base.num_rotors()];
		self.base.pwm(q, gyro, target_rpy, &mut out);
		out.into_iter().map(|p| p as f64).collect()
	}
}

#[pymethods]
impl AllocLqrRs
{
	/// Python constructor. `rows` follow the AttitudeSim.set_geometry contract
	/// [px,py,pz, ax,ay,az, spin, k_thrust, k_drag] and must be the NOMINAL
	/// geometry (the allocator's model — perturb only the SIM side).
	#[new]
	#[pyo3(signature = (
		rows,
		inertia = [0.0023, 0.0023, 0.0046],
		q_att = 12.0, q_rate = 1.0, r_ctrl = 1.0,
		tau_max = 0.144, f_hover = None, pinv_lambda = 1e-6
	))]
	#[allow(clippy::too_many_arguments)]
	pub fn new(
		rows: Vec<[f32; 9]>,
		inertia: [f32; 3],
		q_att: f64,
		q_rate: f64,
		r_ctrl: f64,
		tau_max: f64,
		f_hover: Option<f64>,
		pinv_lambda: f32,
	) -> PyResult<Self>
	{
		Self::build_core(
			&rows,
			inertia,
			q_att,
			q_rate,
			r_ctrl,
			tau_max,
			f_hover,
			pinv_lambda,
		)
		.map_err(pyo3::exceptions::PyValueError::new_err)
	}
	pub fn reset(&mut self) {} // memoryless
	/// One step → N motor PWMs (f32).
	fn step(&mut self, q: [f32; 4], gyro: [f32; 3], target_rpy: [f32; 3]) -> Vec<f32>
	{
		let mut out = vec![0.0f32; self.base.num_rotors()];
		self.base.pwm(q, gyro, target_rpy, &mut out);
		out
	}
	fn num_rotors(&self) -> usize
	{
		self.base.num_rotors()
	}
	/// Torque-space gains [k1, k2_roll, k2_pitch, k2_yaw].
	fn gain(&self) -> Vec<f64>
	{
		vec![
			self.base.k1 as f64,
			self.base.k2[0] as f64,
			self.base.k2[1] as f64,
			self.base.k2[2] as f64,
		]
	}
}

// ===========================================================================
// MPC teacher — condensed box-constrained QP solved EXACTLY per step (box_qp.rs).
// ===========================================================================

#[pyclass]
pub struct AttitudeMpcRs
{
	n: usize, // horizon
	hover: f64,
	authority: f64,
	/// condensed Hessian (3N×3N), cost = ½UᵀHU + gᵀU — the solver owns its own
	/// copy; this one is kept only for the test-side reference solves.
	#[cfg(test)]
	h: Mat,
	ustar_map: Mat, // −H⁻¹G (3N×6): unconstrained U* = ustar_map · x0
	g_map: Mat,     // G (3N×6): g = g_map · x0
	qp: BoxQpSolver, // exact solver for the constrained case (owns a copy of H)
	warm: Vec<f64>, // warm-start U from the previous step (shifted)
	u_buf: Vec<f64>, // scratch: unconstrained U* (3N)
	g_buf: Vec<f64>, // scratch: linear term g (3N)
	/// TEST-ONLY solver probe: when Some, every step records (x0, exit U,
	/// iterations, converged). Read-only — never feeds back into a solve.
	#[cfg(test)]
	probe: Option<Vec<MpcProbe>>,
	/// ABI 34: Some ⇔ motor lag on — the prediction model is the 9-state
	/// augmented one and x0 carries this actuator estimate. (MpcOf feeds it too.)
	act: Option<ActuatorLag>,
}

/// One recorded horizon solve (test-only probe).
#[cfg(test)]
#[derive(Clone)]
pub(crate) struct MpcProbe
{
	pub x: [f64; 6],
	pub u: Vec<f64>,
	pub constrained: bool,
	pub iters: usize,
	pub converged: bool,
}

impl AttitudeMpcRs
{
	#[allow(clippy::too_many_arguments)]
	pub(crate) fn build(
		dt: f32,
		arm_length: f32,
		k_thrust: f32,
		k_drag: f32,
		inertia: [f32; 3],
		gravity: f32,
		hover: f64,
		authority: f64,
		q_att: f64,
		q_rate: f64,
		r_ctrl: f64,
		horizon: usize,
		dt_mpc: f64,
	) -> Self
	{
		let b = calibrate_control_gains_rs(
			dt, arm_length, k_thrust, k_drag, inertia, gravity, hover, 0.05,
		);
		let (a_c, b_c) = attitude_linear_model(b);
		let ad = Mat::identity(6).add(&a_c.scale(dt_mpc));
		let bd = b_c.scale(dt_mpc);
		let qm = diag6([q_att, q_att, q_att, q_rate, q_rate, q_rate]);
		let rm = diag3([r_ctrl, r_ctrl, r_ctrl]);
		let qf = dare(&ad, &bd, &qm, &rm); // terminal cost-to-go (discrete LQR)
		Self::from_discrete(&ad, &bd, &qm, &rm, &qf, horizon, hover, authority)
	}

	/// ABI 34: the LAG-AWARE MPC — the same condensed horizon QP on the 9-state
	/// augmented model (attitude_lag_model at dt_mpc), terminal cost from the
	/// augmented DARE, effort split per LAG_CMD_SHARE (ABI 35). x0 = [e, rate, ua].
	#[allow(clippy::too_many_arguments)]
	pub(crate) fn build_lagged(
		dt: f32,
		arm_length: f32,
		k_thrust: f32,
		k_drag: f32,
		inertia: [f32; 3],
		gravity: f32,
		hover: f64,
		authority: f64,
		q_att: f64,
		q_rate: f64,
		r_ctrl: f64,
		horizon: usize,
		dt_mpc: f64,
		motor_lag_s: f32,
	) -> Self
	{
		let b = calibrate_control_gains_rs(
			dt, arm_length, k_thrust, k_drag, inertia, gravity, hover, 0.05,
		);
		// The prediction step is dt_mpc, so its lag coefficient is too (the
		// estimator below runs at the plant's dt).
		let t_eff = (motor_lag_s as f64).max(4.0 * dt_mpc);
		let (ad, bd) = attitude_lag_model(b, dt_mpc, 4.0 * dt_mpc / t_eff, false);
		let qm = lag_state_cost(q_att, q_rate, None, r_ctrl);
		let rm = lag_cmd_cost(r_ctrl);
		let qf = dare_doubling(&ad, &bd, &qm, &rm);
		let mut m = Self::from_discrete(&ad, &bd, &qm, &rm, &qf, horizon, hover, authority);
		m.act = Some(ActuatorLag::new(motor_lag_s, dt));
		m
	}

	/// The condensed horizon QP for a discrete model (ad, bd) with stage costs
	/// (qm, rm) and terminal cost qf — shared by the lag-free (6-state) and the
	/// lag-aware (9-state) builds. The state dimension is ad's.
	#[allow(clippy::too_many_arguments)]
	fn from_discrete(ad: &Mat, bd: &Mat, qm: &Mat, rm: &Mat, qf: &Mat, horizon: usize, hover: f64, authority: f64) -> Self
	{
		let n = horizon;
		let ns = ad.r;
		let nu = 3usize;
		let rows = ns * (n + 1);
		let cols = nu * n;
		// Prediction matrices: X = Sx·x0 + Su·U, X stacked over k=0..N.
		// Sx block-row k = Ad^k ; Su block (k,j)=Ad^{k-1-j}·Bd for j<k else 0.
		let mut ad_pow: Vec<Mat> = Vec::with_capacity(n + 1);
		ad_pow.push(Mat::identity(ns)); // Ad^0
		for k in 1..=n
		{
			ad_pow.push(ad_pow[k - 1].matmul(ad));
		}
		let mut sx = Mat::zeros(rows, ns);
		let mut su = Mat::zeros(rows, cols);
		for k in 0..=n
		{
			// Sx block
			for i in 0..ns
			{
				for j in 0..ns
				{
					sx.set(k * ns + i, j, ad_pow[k].get(i, j));
				}
			}
			// Su blocks
			for j in 0..k
			{
				let blk = ad_pow[k - 1 - j].matmul(bd); // ns×3
				for ii in 0..ns
				{
					for jj in 0..nu
					{
						su.set(k * ns + ii, j * nu + jj, blk.get(ii, jj));
					}
				}
			}
		}
		// Qbar = blkdiag(Q,...,Q [k=0..N-1], Qf [k=N]); Rbar = blkdiag(R,...,R).
		let mut qbar = Mat::zeros(rows, rows);
		for k in 0..n
		{
			for i in 0..ns
			{
				for j in 0..ns
				{
					qbar.set(k * ns + i, k * ns + j, qm.get(i, j));
				}
			}
		}
		for i in 0..ns
		{
			for j in 0..ns
			{
				qbar.set(n * ns + i, n * ns + j, qf.get(i, j));
			}
		}
		let mut rbar = Mat::zeros(cols, cols);
		for k in 0..n
		{
			for i in 0..nu
			{
				for j in 0..nu
				{
					rbar.set(k * nu + i, k * nu + j, rm.get(i, j));
				}
			}
		}
		// H = 2(SuᵀQbarSu + Rbar) ; g = G·x0, G = 2·SuᵀQbarSx.
		let sut = su.transpose();
		let sutq = sut.matmul(&qbar);
		let h = sutq.matmul(&su).add(&rbar).scale(2.0);
		let g_map = sutq.matmul(&sx).scale(2.0); // 3N×6
		let h_inv = inverse(&h).expect("MPC: H singular");
		let ustar_map = h_inv.matmul(&g_map).scale(-1.0); // −H⁻¹G
		let qp = BoxQpSolver::new(cols, h.d.clone(), authority);

		AttitudeMpcRs {
			n,
			hover,
			authority,
			#[cfg(test)]
			h,
			ustar_map,
			g_map,
			qp,
			warm: vec![0.0; cols],
			u_buf: vec![0.0; cols],
			g_buf: vec![0.0; cols],
			#[cfg(test)]
			probe: None,
			act: None,
		}
	}

	/// ABI 34: the pwm applied last step → actuator estimate (no-op without lag).
	pub(crate) fn observe_applied(&mut self, applied_pwm: [f64; 4])
	{
		if let Some(act) = self.act.as_mut()
		{
			act.observe(applied_pwm);
		}
	}

	pub fn step_rs(&mut self, q: [f32; 4], gyro: [f32; 3], target_rpy: [f32; 3]) -> [f64; 4]
	{
		let x = state_error(q, gyro, target_rpy);
		let z = self.act.as_ref().map(|act| (act.seeded(), with_actuator(&x, act.state())));
		let u = match z
		{
			Some((true, z)) => self.step_axes_rs(&z),
			Some((false, _)) => self.first_step_axes(&x),
			None => self.step_axes_rs(&x),
		};
		let a = self.authority;
		mix_to_motors_f64(
			self.hover,
			clamp_f64(u[0], -a, a),
			clamp_f64(u[1], -a, a),
			clamp_f64(u[2], -a, a),
		)
	}

	/// The horizon solve WITHOUT the final mix — returns the first per-axis
	/// control [u_roll, u_pitch, u_yaw] (pre-clamp). Extracted so the
	/// offset-free wrapper (AttitudeMpcOfRs) can inject its disturbance
	/// feedforward BEFORE mixing.
	pub(crate) fn step_axes_rs(&mut self, x: &[f64]) -> [f64; 3]
	{
		let a = self.authority;
		let cols = self.n * 3;
		// Unconstrained optimum U* = ustar_map · x0. If it satisfies the box for the
		// whole horizon it IS the constrained optimum (fast path == LQR feedback).
		self.ustar_map.mul_vec_into(x, &mut self.u_buf);
		let feasible = self.u_buf.iter().all(|&ui| ui.abs() <= a + 1e-9);
		if !feasible
		{
			// The box binds: the exact optimum of ½UᵀHU + gᵀU s.t. |U| ≤ a, its
			// active set seeded from the previous step's shifted solution.
			self.g_map.mul_vec_into(x, &mut self.g_buf);
			let u = self.qp.solve(&self.g_buf, &self.warm);
			self.u_buf.copy_from_slice(u);
		}
		#[cfg(test)]
		self.record_probe(x, !feasible);
		// Warm start next step: shift U by one control (receding horizon).
		self.warm[..cols - 3].copy_from_slice(&self.u_buf[3..]);
		self.warm[cols - 3..].copy_from_slice(&self.u_buf[cols - 3..]);
		[self.u_buf[0], self.u_buf[1], self.u_buf[2]]
	}

	/// ABI 34: the lag-aware MPC's first decision (see first_step_fixed_point):
	/// the fixed point of the unconstrained first-move map U*₀ = M·[x; ua] under
	/// ua = u, which step_rs then clamps to ±authority like any other move.
	pub(crate) fn first_step_axes(&self, x: &[f64]) -> [f64; 3]
	{
		let mut m0 = Mat::zeros(3, self.ustar_map.c);
		for i in 0..3
		{
			for j in 0..self.ustar_map.c
			{
				m0.set(i, j, self.ustar_map.get(i, j));
			}
		}
		let v = first_step_fixed_point(&m0, x, 1.0);
		[v[0], v[1], v[2]]
	}

	#[cfg(test)]
	fn record_probe(&mut self, x: &[f64], constrained: bool)
	{
		if let Some(p) = self.probe.as_mut()
		{
			let mut x6 = [0.0f64; 6];
			x6.copy_from_slice(&x[..6]);
			p.push(MpcProbe {
				x: x6,
				u: self.u_buf.clone(),
				constrained,
				iters: if constrained { self.qp.last_solves } else { 0 },
				converged: !(constrained && self.qp.last_fallback),
			});
		}
	}
}

#[pymethods]
impl AttitudeMpcRs
{
	#[new]
	#[pyo3(signature = (
		dt = 0.001, arm_length = 0.075, k_thrust = 2.4, k_drag = 0.05,
		inertia = [0.0023, 0.0023, 0.0046], gravity = 9.81,
		hover = 0.5, authority = 0.4, q_att = 12.0, q_rate = 1.0, r_ctrl = 1.0,
		horizon = 25, dt_mpc = 0.001
	))]
	#[allow(clippy::too_many_arguments)]
	pub fn new(
		dt: f32,
		arm_length: f32,
		k_thrust: f32,
		k_drag: f32,
		inertia: [f32; 3],
		gravity: f32,
		hover: f64,
		authority: f64,
		q_att: f64,
		q_rate: f64,
		r_ctrl: f64,
		horizon: usize,
		dt_mpc: f64,
	) -> Self
	{
		Self::build(
			dt, arm_length, k_thrust, k_drag, inertia, gravity, hover, authority, q_att, q_rate, r_ctrl,
			horizon, dt_mpc,
		)
	}
	pub fn reset(&mut self)
	{
		for w in self.warm.iter_mut()
		{
			*w = 0.0;
		}
		if let Some(act) = self.act.as_mut()
		{
			act.reset();
		}
	}
	fn step(&mut self, q: [f32; 4], gyro: [f32; 3], target_rpy: [f32; 3]) -> [f32; 4]
	{
		let p = self.step_rs(q, gyro, target_rpy);
		[p[0] as f32, p[1] as f32, p[2] as f32, p[3] as f32]
	}
}

// ===========================================================================
// Offset-free MPC teacher — MPC + input-disturbance observer (Phase-4, 18/07/2026).
//
// Vanilla MPC plans with the NOMINAL model, so a persistent unmodeled torque
// (DisturbanceConfig D1/D2) leaves a steady-state offset it can never remove.
// The classical fix is offset-free MPC: estimate the input-equivalent
// disturbance d̂ per axis from the model residual, and cancel it with a
// feedforward u_ff = −d̂/b on top of the nominal MPC solve:
//
//   observer:  d̂ ← d̂ + L·( (gyro − gyro_prev)/dt − b·u_applied − d̂ )
//   control :  u = clamp( u_mpc(x) − clamp(d̂/b, ±ff_clamp), ±authority )
//
// DAGGER subtlety (why observe() exists): the sim propagates under the
// STUDENT's action, not the teacher's — so the observer's u_applied must be
// the action ACTUALLY applied to the sim. The rollout loop feeds it via
// observe(gyro, applied_pwm) each step ('+'-mixer inverted back to per-axis
// u). Without observe() (teacher flying solo, e.g. baselines) it falls back
// to its own last command. The small L (default 0.05) low-pass filters the
// residual, so the noisy D4 gyro does not corrupt d̂ — a 1-pole estimator,
// which is all a constant/slow disturbance needs.
//
// Stateful (d̂ is accumulated history) → generates split-pressure conflicts
// like PID/LQI; integrals() exposes d̂ so --state-integral works with it too.
// ===========================================================================

#[pyclass]
pub struct AttitudeMpcOfRs
{
	mpc: AttitudeMpcRs,
	b: [f64; 3],    // per-axis calibrated control gain (rate-accel per u)
	dhat: [f64; 3], // input-equivalent disturbance estimate (rate-accel)
	last_gyro: Option<[f64; 3]>,
	last_u: [f64; 3], // per-axis u ACTUALLY applied last step (see observe)
	l_gain: f64,      // observer gain (1-pole low-pass on the residual)
	ff_clamp: f64,    // |u_ff| bound (keeps feedforward from saturating the mix)
	dt: f64,
	hover: f64,
	authority: f64,
}

impl AttitudeMpcOfRs
{
	#[allow(clippy::too_many_arguments)]
	pub(crate) fn build(
		dt: f32,
		arm_length: f32,
		k_thrust: f32,
		k_drag: f32,
		inertia: [f32; 3],
		gravity: f32,
		hover: f64,
		authority: f64,
		q_att: f64,
		q_rate: f64,
		r_ctrl: f64,
		horizon: usize,
		dt_mpc: f64,
		l_gain: f64,
		ff_clamp: f64,
	) -> Self
	{
		let b = calibrate_control_gains_rs(
			dt, arm_length, k_thrust, k_drag, inertia, gravity, hover, 0.05,
		);
		let mpc = AttitudeMpcRs::build(
			dt, arm_length, k_thrust, k_drag, inertia, gravity, hover, authority, q_att, q_rate, r_ctrl,
			horizon, dt_mpc,
		);
		Self::around(mpc, b, l_gain, ff_clamp, dt, hover, authority)
	}

	/// ABI 34: the LAG-AWARE offset-free MPC — the lag-aware MPC (9-state model,
	/// x0 carries the actuator estimate) plus the observer reading the FILTERED
	/// actuator: residual = rate_dot − b·ua − d̂. Subtracting b·u_commanded (the
	/// lag-free law) reads every commanded change the rotors have not yet
	/// followed as a disturbance, and the feedforward then fights the lag.
	#[allow(clippy::too_many_arguments)]
	pub(crate) fn build_lagged(
		dt: f32,
		arm_length: f32,
		k_thrust: f32,
		k_drag: f32,
		inertia: [f32; 3],
		gravity: f32,
		hover: f64,
		authority: f64,
		q_att: f64,
		q_rate: f64,
		r_ctrl: f64,
		horizon: usize,
		dt_mpc: f64,
		l_gain: f64,
		ff_clamp: f64,
		motor_lag_s: f32,
	) -> Self
	{
		let b = calibrate_control_gains_rs(
			dt, arm_length, k_thrust, k_drag, inertia, gravity, hover, 0.05,
		);
		let mpc = AttitudeMpcRs::build_lagged(
			dt, arm_length, k_thrust, k_drag, inertia, gravity, hover, authority, q_att, q_rate, r_ctrl,
			horizon, dt_mpc, motor_lag_s,
		);
		Self::around(mpc, b, l_gain, ff_clamp, dt, hover, authority)
	}

	#[allow(clippy::too_many_arguments)]
	fn around(mpc: AttitudeMpcRs, b: [f64; 3], l_gain: f64, ff_clamp: f64, dt: f32, hover: f64, authority: f64) -> Self
	{
		AttitudeMpcOfRs {
			mpc,
			b,
			dhat: [0.0; 3],
			last_gyro: None,
			last_u: [0.0; 3],
			l_gain,
			ff_clamp,
			dt: dt as f64,
			hover,
			authority,
		}
	}

	/// Feed the observer the gyro just read and the motor PWMs ACTUALLY applied
	/// to the sim last step (the student's, under DAGGER). Inverts the '+' mixer:
	/// u_roll=(m3−m1)/2, u_pitch=(m2−m0)/2, u_yaw=((m0+m2)−(m1+m3))/4.
	pub fn observe(&mut self, gyro: [f32; 3], applied_pwm: [f64; 4])
	{
		// ABI 34: under motor lag the torque came from the FILTERED actuator, so
		// that — not the command — is the model term's u. Filter first (the
		// plant's order: lag, then the step the gyro just measured).
		if let Some(act) = self.mpc.act.as_mut()
		{
			act.observe(applied_pwm);
			let ua = act.state();
			self.update_dhat(gyro, ua);
			return;
		}
		let m = applied_pwm;
		self.update_dhat(
			gyro,
			[
				(m[3] - m[1]) * 0.5,
				(m[2] - m[0]) * 0.5,
				((m[0] + m[2]) - (m[1] + m[3])) * 0.25,
			],
		);
	}

	fn update_dhat(&mut self, gyro: [f32; 3], u_applied: [f64; 3])
	{
		let g = [gyro[0] as f64, gyro[1] as f64, gyro[2] as f64];
		if let Some(prev) = self.last_gyro
		{
			for axis in 0..3
			{
				let rate_dot = (g[axis] - prev[axis]) / self.dt;
				let residual = rate_dot - self.b[axis] * u_applied[axis] - self.dhat[axis];
				self.dhat[axis] += self.l_gain * residual;
			}
		}
		self.last_gyro = Some(g);
		self.last_u = u_applied;
	}

	/// Native f64 step (mirrors the other teachers' step_rs signature).
	pub fn step_rs(&mut self, q: [f32; 4], gyro: [f32; 3], target_rpy: [f32; 3]) -> [f64; 4]
	{
		let x = state_error(q, gyro, target_rpy);
		if self.mpc.act.is_some()
		{
			return self.step_lagged(&x);
		}
		let u = self.mpc.step_axes_rs(&x);
		let a = self.authority;
		let mut u_cmd = [0.0f64; 3];
		for axis in 0..3
		{
			let u_ff = clamp_f64(
				self.dhat[axis] / self.b[axis],
				-self.ff_clamp,
				self.ff_clamp,
			);
			u_cmd[axis] = clamp_f64(u[axis] - u_ff, -a, a);
		}
		// last_u kept for telemetry; d̂ is updated ONLY via observe() (the DAGGER
		// loop feeds it the student's applied action). Solo flight without
		// observe() ⇒ d̂ stays 0 ⇒ mpcof degrades to plain MPC (safe no-op).
		self.last_u = u_cmd;
		mix_to_motors_f64(self.hover, u_cmd[0], u_cmd[1], u_cmd[2])
	}

	/// ABI 34 lag-aware step. The MPC plans in DISTURBANCE-SHIFTED coordinates:
	/// the rotors' net effect is b·ua + d = b·(ua + d/b) ≈ b·(ua + u_ff), so
	/// x0's actuator entry is ua + u_ff and the plan's u is the net command
	/// (u_cmd = u − u_ff, as the lag-free law). Feeding the raw ua instead would
	/// have the plan cancel the very feedforward that holds the offset at bay.
	fn step_lagged(&mut self, x: &[f64; 6]) -> [f64; 4]
	{
		let (seeded, ua) = self
			.mpc
			.act
			.as_ref()
			.map(|act| (act.seeded(), act.state()))
			.unwrap_or((false, [0.0; 3]));
		let mut u_ff = [0.0f64; 3];
		let mut z = [0.0f64; 9];
		z[..6].copy_from_slice(x);
		for axis in 0..3
		{
			u_ff[axis] = clamp_f64(self.dhat[axis] / self.b[axis], -self.ff_clamp, self.ff_clamp);
			z[6 + axis] = ua[axis] + u_ff[axis];
		}
		// First decision: the plant seeds the actuator at u_cmd = u − u_ff, so
		// the shifted actuator ua + u_ff equals the plan's own u — the same fixed
		// point as the plain MPC's.
		let u = if seeded { self.mpc.step_axes_rs(&z) } else { self.mpc.first_step_axes(x) };
		let a = self.authority;
		let mut u_cmd = [0.0f64; 3];
		for axis in 0..3
		{
			u_cmd[axis] = clamp_f64(u[axis] - u_ff[axis], -a, a);
		}
		self.last_u = u_cmd;
		mix_to_motors_f64(self.hover, u_cmd[0], u_cmd[1], u_cmd[2])
	}

	/// The observer state (per-axis d̂) — the Option-A --state-integral target
	/// analog (this IS the teacher's accumulated memory).
	pub fn integrals(&self) -> [f32; 3]
	{
		[
			self.dhat[0] as f32,
			self.dhat[1] as f32,
			self.dhat[2] as f32,
		]
	}
	/// Normalization clamps for the integral target: use the ff_clamp expressed
	/// in d̂ units (b·ff_clamp), the effective saturation of the estimator.
	pub fn i_clamps(&self) -> [f32; 3]
	{
		[
			(self.b[0] * self.ff_clamp) as f32,
			(self.b[1] * self.ff_clamp) as f32,
			(self.b[2] * self.ff_clamp) as f32,
		]
	}
}

#[pymethods]
impl AttitudeMpcOfRs
{
	/// Python constructor for parity testing. Defaults match AttitudeMpcRs::new
	/// plus the observer knobs (l_gain, ff_clamp).
	#[new]
	#[pyo3(signature = (
		dt = 0.001, arm_length = 0.075, k_thrust = 2.4, k_drag = 0.05,
		inertia = [0.0023, 0.0023, 0.0046], gravity = 9.81,
		hover = 0.5, authority = 0.4, q_att = 12.0, q_rate = 1.0, r_ctrl = 1.0,
		horizon = 25, dt_mpc = 0.001, l_gain = 0.05, ff_clamp = 0.2
	))]
	#[allow(clippy::too_many_arguments)]
	pub fn new(
		dt: f32,
		arm_length: f32,
		k_thrust: f32,
		k_drag: f32,
		inertia: [f32; 3],
		gravity: f32,
		hover: f64,
		authority: f64,
		q_att: f64,
		q_rate: f64,
		r_ctrl: f64,
		horizon: usize,
		dt_mpc: f64,
		l_gain: f64,
		ff_clamp: f64,
	) -> Self
	{
		Self::build(
			dt, arm_length, k_thrust, k_drag, inertia, gravity, hover, authority, q_att, q_rate, r_ctrl,
			horizon, dt_mpc, l_gain, ff_clamp,
		)
	}
	pub fn reset(&mut self)
	{
		self.dhat = [0.0; 3];
		self.last_gyro = None;
		self.last_u = [0.0; 3];
		self.mpc.reset();
	}
	/// One step → 4 motor PWMs (f32), for Python parity testing.
	fn step(&mut self, q: [f32; 4], gyro: [f32; 3], target_rpy: [f32; 3]) -> [f32; 4]
	{
		let p = self.step_rs(q, gyro, target_rpy);
		[p[0] as f32, p[1] as f32, p[2] as f32, p[3] as f32]
	}
	/// Python-side observe (for tests): gyro + the applied motor PWMs.
	fn observe_py(&mut self, gyro: [f32; 3], applied_pwm: [f64; 4])
	{
		self.observe(gyro, applied_pwm);
	}
	/// Current disturbance estimate d̂ (rate-accel units), for tests/telemetry.
	fn dhat(&self) -> Vec<f64>
	{
		self.dhat.to_vec()
	}
}

// ===========================================================================
// Teacher enum — the DAGGER loop's expert slot. Dispatches step/reset/integrals.
// ===========================================================================

pub enum Teacher
{
	Pid(AttitudePidRs),
	/// Firmware-sourced cascade (platform_defaults_cf21bl.h). Selected instead of `Pid`
	/// whenever the airframe supplies cascade gains — see `Teacher::pid_for_airframe`.
	PidFw(crate::pid_firmware::AttitudePidFirmwareRs),
	Lqr(AttitudeLqrRs),
	Mpc(AttitudeMpcRs),
	Lqi(AttitudeLqiRs),
	MpcOf(AttitudeMpcOfRs),
}

impl Teacher
{
	/// Construct the teacher selected by cfg (0=pid, 1=lqr, 2=mpc, 3=lqi,
	/// 4=mpcof), using the SAME sim params the DAGGER loop controls (so the
	/// LQR/MPC/LQI plant matches). hover/authority mirror the PID teacher's
	/// defaults (0.5 / 0.4).
	pub fn from_id(
		id: u8,
		dt: f32,
		arm_length: f32,
		k_thrust: f32,
		k_drag: f32,
		inertia: [f32; 3],
		gravity: f32,
	) -> Teacher
	{
		// 0.5 is the LEGACY neutral every attitude result was flown at. Changing
		// it here would re-base the whole attitude lineage, so the hover-aware
		// path is a separate constructor used only where gravity is simulated.
		Self::from_id_with_hover(id, dt, arm_length, k_thrust, k_drag, inertia, gravity, 0.5)
	}

	/// SCOPE C STAGE 2: build a teacher anchored at a GIVEN collective.
	///
	/// WHY THIS EXISTS. The attitude teachers were written for a sim with no
	/// gravity to fall against, so their neutral is 0.5 — an arbitrary midpoint.
	/// cf21_brushless actually hovers at √(mg/4k) = 0.6942. Once translation is
	/// on, a teacher sitting at 0.5 is 0.1942 of collective short, and the
	/// altitude PD (no integral term) can only supply that by sitting
	/// permanently below target: droop = (hover − 0.5)·b_z/ωn² = 1.372 m, which
	/// is exactly what the first full-state measurement produced, to four
	/// digits, identically for all five teachers.
	///
	/// This is the teacher-side twin of stage 1's collective_anchor
	/// (ed921bac) — the same 0.5-vs-0.6942 mismatch, found on the student side
	/// first. Anchoring at true hover leaves the PD to correct DEVIATIONS only,
	/// which is what a cascade's inner stage is supposed to receive.
	#[allow(clippy::too_many_arguments)]
	pub fn from_id_with_hover(
		id: u8,
		dt: f32,
		arm_length: f32,
		k_thrust: f32,
		k_drag: f32,
		inertia: [f32; 3],
		gravity: f32,
		hover: f64,
	) -> Teacher
	{
		let authority = 0.4;
		match id
		{
			1 => Teacher::Lqr(AttitudeLqrRs::build(
				dt, arm_length, k_thrust, k_drag, inertia, gravity, hover, authority, Q_ATT, Q_RATE, R_CTRL,
			)),
			2 => Teacher::Mpc(AttitudeMpcRs::build(
				dt, arm_length, k_thrust, k_drag, inertia, gravity, hover, authority, Q_ATT, Q_RATE,
				R_CTRL, 25, 0.001,
			)),
			3 => Teacher::Lqi(AttitudeLqiRs::build(
				dt, arm_length, k_thrust, k_drag, inertia, gravity, hover, authority, Q_ATT, Q_RATE,
				R_CTRL, Q_INT,
			)),
			4 => Teacher::MpcOf(AttitudeMpcOfRs::build(
				dt, arm_length, k_thrust, k_drag, inertia, gravity, hover, authority, Q_ATT, Q_RATE,
				R_CTRL, 25, 0.001, 0.05, 0.2,
			)),
			_ => Teacher::Pid(pid_teacher_at_hover(hover)),
		}
	}

	/// ABI 34 (05/11/2026): the teacher for a plant with motor-lag settling time
	/// `motor_lag_s` (Molchanov T; τ = T/4). The model-based teachers (ids 1-4)
	/// get the TRUE lag (Luiz, 05/11): LQR/LQI/MPC/MPCOF are designed on the
	/// actuator-augmented model and estimate the actuator by filtering the pwm
	/// they are told was applied. PID (id 0) is the firmware reference and stays
	/// exactly as built. `motor_lag_s <= 0` IS from_id_with_hover — the lag-free
	/// code path, not an augmented model evaluated at τ = 0 (T = 0 bit-identical
	/// to ABI 33, pinned by abi34_tests).
	#[allow(clippy::too_many_arguments)]
	pub fn from_id_lagged(
		id: u8,
		dt: f32,
		arm_length: f32,
		k_thrust: f32,
		k_drag: f32,
		inertia: [f32; 3],
		gravity: f32,
		hover: f64,
		motor_lag_s: f32,
	) -> Teacher
	{
		if !(motor_lag_s > 0.0)
		{
			return Self::from_id_with_hover(id, dt, arm_length, k_thrust, k_drag, inertia, gravity, hover);
		}
		let authority = 0.4;
		let t = motor_lag_s;
		match id
		{
			1 => Teacher::Lqr(AttitudeLqrRs::build_lagged(
				dt, arm_length, k_thrust, k_drag, inertia, gravity, hover, authority, Q_ATT, Q_RATE, R_CTRL, t,
			)),
			2 => Teacher::Mpc(AttitudeMpcRs::build_lagged(
				dt, arm_length, k_thrust, k_drag, inertia, gravity, hover, authority, Q_ATT, Q_RATE,
				R_CTRL, 25, 0.001, t,
			)),
			3 => Teacher::Lqi(AttitudeLqiRs::build_lagged(
				dt, arm_length, k_thrust, k_drag, inertia, gravity, hover, authority, Q_ATT, Q_RATE,
				R_CTRL, Q_INT, t,
			)),
			4 => Teacher::MpcOf(AttitudeMpcOfRs::build_lagged(
				dt, arm_length, k_thrust, k_drag, inertia, gravity, hover, authority, Q_ATT, Q_RATE,
				R_CTRL, 25, 0.001, 0.05, 0.2, t,
			)),
			_ => Self::from_id_with_hover(id, dt, arm_length, k_thrust, k_drag, inertia, gravity, hover),
		}
	}

	/// D0 (11/09/2026): the collective this teacher was BUILT at — the pwm every
	/// motor sits on at level / zero rates / zero attitude error, i.e. the
	/// teacher's own neutral. 0.5 for the legacy attitude teachers (from_id),
	/// the from_id_with_hover argument otherwise, and the firmware cascade's
	/// √(hover_n/k_thrust) for PidFw. The DAgger label is re-based on THIS value
	/// (dagger_train::teacher_label_f32), which is what lets a teacher at any
	/// hover label "deviation from your own anchor" instead of an absolute pwm.
	pub fn hover(&self) -> f64
	{
		match self
		{
			Teacher::Pid(p) => p.hover_throttle(),
			Teacher::PidFw(p) => p.hover_pwm(),
			Teacher::Lqr(l) => l.hover,
			Teacher::Mpc(m) => m.hover,
			Teacher::Lqi(l) => l.hover,
			Teacher::MpcOf(m) => m.hover,
		}
	}

	#[inline]
	pub fn step_rs(&mut self, q: [f32; 4], gyro: [f32; 3], target_rpy: [f32; 3]) -> [f64; 4]
	{
		match self
		{
			Teacher::Pid(p) => p.step_rs(q, gyro, target_rpy),
			Teacher::PidFw(p) => p.step_rs(
				[q[0] as f64, q[1] as f64, q[2] as f64, q[3] as f64],
				[gyro[0] as f64, gyro[1] as f64, gyro[2] as f64],
				[
					target_rpy[0] as f64,
					target_rpy[1] as f64,
					target_rpy[2] as f64,
				],
			),
			Teacher::Lqr(l) => l.step_rs(q, gyro, target_rpy),
			Teacher::Mpc(m) => m.step_rs(q, gyro, target_rpy),
			Teacher::Lqi(l) => l.step_rs(q, gyro, target_rpy),
			Teacher::MpcOf(m) => m.step_rs(q, gyro, target_rpy),
		}
	}
	/// SCOPE C STAGE 1: the attitude command with an outer-loop collective added
	/// uniformly on top — the DISCLOSED CASCADE (altitude PD → attitude teacher)
	/// that the classical rivals already are. A uniform add leaves every motor
	/// difference, hence the body torque, exactly as `step_rs` produced it, so
	/// the attitude behaviour is untouched by construction.
	///
	/// `alt_err` = target − z (positive ⇒ climb), `vz` = +up. The result is
	/// clamped to [0, 1] per motor, the same range step_rs promises.
	#[inline]
	pub fn step_with_collective(
		&mut self,
		q: [f32; 4],
		gyro: [f32; 3],
		target_rpy: [f32; 3],
		pd: &crate::altitude_pd::AltitudePd,
		alt_err: f64,
		vz: f64,
	) -> [f64; 4]
	{
		let base = self.step_rs(q, gyro, target_rpy);
		let d = pd.delta(alt_err, vz);
		[
			(base[0] + d).clamp(0.0, 1.0),
			(base[1] + d).clamp(0.0, 1.0),
			(base[2] + d).clamp(0.0, 1.0),
			(base[3] + d).clamp(0.0, 1.0),
		]
	}

	/// SCOPE C STAGE 2: the FULL-STATE command — position and altitude loops
	/// wrapped around the attitude teacher, i.e. the complete disclosed cascade
	/// (docs/scope_c_stage2_chunk_b_teacher.md).
	///
	/// This is deliberately a COMPOSITION of pieces that already exist rather
	/// than a new controller: the position loop turns horizontal error into a
	/// tilt reference, that reference REPLACES the commanded roll/pitch, and
	/// step_with_collective adds the altitude loop's collective on top. Every
	/// one of the six teachers therefore gains position control unchanged — none
	/// of them knows this happened, which is exactly why all five rivals survive
	/// into stage 2 instead of collapsing to one monolithic controller.
	///
	/// `yaw_ref` stays whatever the episode commands: a symmetric quad holds a
	/// point at any heading, so yaw is not a product of position.
	/// Errors are target − actual (positive ⇒ move that way); velocities are
	/// world-frame, vz +up.
	#[inline]
	#[allow(clippy::too_many_arguments)]
	pub fn step_full_state(
		&mut self,
		q: [f32; 4],
		gyro: [f32; 3],
		yaw_ref: f32,
		pos: &crate::position_loop::PositionLoop,
		pd: &crate::altitude_pd::AltitudePd,
		err_x: f64,
		vx: f64,
		err_y: f64,
		vy: f64,
		alt_err: f64,
		vz: f64,
	) -> [f64; 4]
	{
		let (roll_ref, pitch_ref) = pos.tilt_ref(err_x, vx, err_y, vy);
		let target_rpy = [roll_ref as f32, pitch_ref as f32, yaw_ref];
		self.step_with_collective(q, gyro, target_rpy, pd, alt_err, vz)
	}

	/// Feed the observer teachers the (gyro, actually-applied motor PWMs) pair
	/// each rollout step — under DAGGER the sim propagates on the STUDENT's
	/// action, which is what a model-residual observer must see. ABI 34: the
	/// lag-aware LQR/MPC/LQI also read it (their actuator estimate); every one
	/// of those is a no-op without motor lag. PID never observes.
	#[inline]
	pub fn observe(&mut self, gyro: [f32; 3], applied_pwm: [f64; 4])
	{
		match self
		{
			Teacher::MpcOf(m) => m.observe(gyro, applied_pwm),
			Teacher::Lqr(l) => l.observe_applied(applied_pwm),
			Teacher::Mpc(m) => m.observe_applied(applied_pwm),
			Teacher::Lqi(l) => l.observe_applied(applied_pwm),
			Teacher::Pid(_) | Teacher::PidFw(_) => {}
		}
	}
	#[inline]
	pub fn reset(&mut self)
	{
		match self
		{
			Teacher::Pid(p) => p.reset(),
			Teacher::PidFw(p) => p.reset(),
			Teacher::Lqr(l) => l.reset(),
			Teacher::Mpc(m) => m.reset(),
			Teacher::Lqi(l) => l.reset(),
			Teacher::MpcOf(m) => m.reset(),
		}
	}
	/// Teacher integral (roll,pitch,yaw). LQR/MPC are memoryless → zeros (the
	/// Option-A integral target applies to the STATEFUL teachers: PID, LQI, and
	/// MpcOf — whose d̂ observer state plays the integral's role).
	#[inline]
	pub fn integrals(&self) -> [f32; 3]
	{
		match self
		{
			Teacher::Pid(p) => p.integrals(),
			Teacher::PidFw(p) => p.integrals_f32(),
			Teacher::Lqi(l) => l.integrals(),
			Teacher::MpcOf(m) => m.integrals(),
			_ => [0.0, 0.0, 0.0],
		}
	}
	/// Clamp magnitudes for normalizing the integral. LQR/MPC: 1.0 (no scaling;
	/// integral is zero anyway).
	#[inline]
	pub fn i_clamps(&self) -> [f32; 3]
	{
		match self
		{
			Teacher::Pid(p) => p.i_clamps(),
			Teacher::PidFw(p) => p.i_clamps_f32(),
			Teacher::Lqi(l) => l.i_clamps(),
			Teacher::MpcOf(m) => m.i_clamps(),
			_ => [1.0, 1.0, 1.0],
		}
	}
}

/// PID teacher with the canonical defaults (mirrors dagger_train::pid_default).
// Kept: the canonical-defaults constructor of the teacher bank. Callers currently
// build via pid_teacher_at_hover directly; this is the documented entry point and
// the mirror of dagger_train::pid_default, so it stays.
#[allow(dead_code)]
fn pid_default_teacher() -> AttitudePidRs
{
	pid_teacher_at_hover(0.5)
}

/// The hand-tuned PID teacher with its collective anchor made explicit
/// (9th arg = hover_throttle). 0.5 reproduces the legacy teacher exactly.
fn pid_teacher_at_hover(hover: f64) -> AttitudePidRs
{
	AttitudePidRs::new(
		1.2, 0.05, 0.30, 0.5, 0.6, 0.02, 0.20, 0.5, hover, 0.4, 0.001,
	)
}

#[cfg(test)]
mod alloc_lqr_tests
{
	use super::*;
	use crate::controller::AttitudeSim;
	use crate::overactuated::RotorGeometry;

	const ARM: f32 = 0.075;
	const KT: f32 = 2.4;
	const KD: f32 = 0.05;
	const INERTIA: [f32; 3] = [0.0023, 0.0023, 0.0046];

	fn rows_from(geo: &RotorGeometry) -> Vec<[f32; 9]>
	{
		geo
			.rotors
			.iter()
			.map(|r| {
				[
					r.position[0],
					r.position[1],
					r.position[2],
					r.axis[0],
					r.axis[1],
					r.axis[2],
					r.spin,
					r.k_thrust,
					r.k_drag,
				]
			})
			.collect()
	}

	fn octo_rows() -> Vec<[f32; 9]>
	{
		rows_from(&RotorGeometry::octo_x(ARM, KT, KD))
	}

	fn teacher(rows: &[[f32; 9]]) -> AllocLqrRs
	{
		AllocLqrRs::build_core(rows, INERTIA, 12.0, 1.0, 1.0, 0.144, None, 1e-6).expect("teacher")
	}

	/// Roll out the teacher closed-loop on a sim carrying `sim_rows` (the TRUE
	/// vehicle) while the teacher allocates on `nom_rows` (the NOMINAL model).
	/// Returns final attitude error (rad).
	fn closed_loop_err(
		nom_rows: &[[f32; 9]],
		sim_rows: &[[f32; 9]],
		tilt_deg: f32,
		steps: usize,
	) -> f32
	{
		let mut t = teacher(nom_rows);
		let mut sim = AttitudeSim::new(0.001, ARM, KT, KD, INERTIA, 9.81);
		sim
			.set_geometry_core(sim_rows.to_vec())
			.expect("sim geometry");
		let half = tilt_deg.to_radians() * 0.5;
		sim.reset(
			Some([half.cos(), half.sin(), 0.0, 0.0]),
			Some([0.0, 0.0, 0.0]),
		);
		for _ in 0..steps
		{
			assert!(
				!sim.is_unstable(),
				"sim diverged under the alloc-LQR teacher"
			);
			let (gyro, _accel) = sim.read_imu();
			let pwm = t.step_alloc_rs(sim.quaternion(), gyro, [0.0, 0.0, 0.0]);
			let pwm32: Vec<f32> = pwm.iter().map(|&p| p as f32).collect();
			sim.step_n_core(&pwm32).expect("step_n");
		}
		sim.attitude_error(None)
	}

	/// Zero attitude error ⇒ the min-norm allocation of the pure-hover wrench
	/// on a symmetric octo is ≈0.5 per rotor (the f_hover default's contract).
	#[test]
	fn hover_allocation_at_zero_error()
	{
		let rows = octo_rows();
		let mut t = teacher(&rows);
		let pwm = t.step_alloc_rs([1.0, 0.0, 0.0, 0.0], [0.0; 3], [0.0; 3]);
		assert_eq!(pwm.len(), 8);
		for (i, p) in pwm.iter().enumerate()
		{
			assert!((p - 0.5).abs() < 0.02, "rotor {i}: pwm {p} not ≈ hover 0.5");
		}
	}

	/// Small-signal consistency: the torque the allocated PWMs actually produce
	/// on the NOMINAL geometry matches the clamped LQR demand.
	#[test]
	fn allocation_realizes_torque_demand()
	{
		let rows = octo_rows();
		let geo = RotorGeometry::from_rows(&rows).unwrap();
		let mut t = teacher(&rows);
		// 4° roll tilt, at rest: demand is unclamped and roll-dominant.
		let half = 4.0_f32.to_radians() * 0.5;
		let q = [half.cos(), half.sin(), 0.0, 0.0];
		// Demand per AllocBaseline::pwm: τ = clamp(k1·(target−angle) − k2·rate).
		let x = state_error(q, [0.0; 3], [0.0; 3]);
		let b = &t.base;
		let want = [
			clamp_f64(-(b.k1 as f64 * x[0] + b.k2[0] as f64 * x[3]), -0.144, 0.144) as f32,
			clamp_f64(-(b.k1 as f64 * x[1] + b.k2[1] as f64 * x[4]), -0.144, 0.144) as f32,
			clamp_f64(-(b.k1 as f64 * x[2] + b.k2[2] as f64 * x[5]), -0.144, 0.144) as f32,
		];
		let pwm: Vec<f32> = t
			.step_alloc_rs(q, [0.0; 3], [0.0; 3])
			.iter()
			.map(|&p| p as f32)
			.collect();
		let got = geo.body_torque(&pwm);
		for a in 0..3
		{
			assert!(
				(got[a] - want[a]).abs() < 0.144 * 0.05 + 2e-3,
				"axis {a}: realized torque {} vs demand {}",
				got[a],
				want[a]
			);
		}
	}

	/// The teacher stabilizes a NOMINAL octo from a 17° tilt.
	#[test]
	fn stabilizes_nominal_octo()
	{
		let rows = octo_rows();
		let err = closed_loop_err(&rows, &rows, 17.0, 1500);
		assert!(
			err < 2.0_f32.to_radians(),
			"final err {}° >= 2°",
			err.to_degrees()
		);
	}

	/// The teacher (nominal allocator) still stabilizes a PERTURBED true
	/// vehicle — tilt/position error + the geometry mismatch the WNN residual
	/// will learn. Bounded, not perfect: allow a residual offset.
	#[test]
	fn stabilizes_perturbed_octo_with_nominal_allocator()
	{
		let rows = octo_rows();
		let tilt: Vec<f32> = [1.5f32, -2.0, 1.0, -0.8, 1.8, -1.2, 0.6, -1.6]
			.iter()
			.map(|d| d.to_radians())
			.collect();
		let pos: Vec<[f32; 3]> = (0..8)
			.map(|i| [0.002 * (i as f32 - 3.5), -0.0015 * (i as f32 - 3.5), 0.001])
			.collect();
		let true_geo = RotorGeometry::from_rows(&rows)
			.unwrap()
			.perturbed(&tilt, &pos);
		let sim_rows = rows_from(&true_geo);
		let err = closed_loop_err(&rows, &sim_rows, 17.0, 1500);
		assert!(
			err < 5.0_f32.to_radians(),
			"perturbed-vehicle err {}° >= 5° (teacher must stay bounded)",
			err.to_degrees()
		);
	}

	/// Quad-plus as geometry: the allocator teacher also stabilizes the
	/// paper-#1 quad (rank-deficient Fx/Fy handled by the damped pinv).
	#[test]
	fn stabilizes_quad_plus_geometry()
	{
		let rows = rows_from(&RotorGeometry::quad_plus(ARM, KT, KD));
		let err = closed_loop_err(&rows, &rows, 17.0, 1500);
		assert!(
			err < 2.0_f32.to_radians(),
			"quad final err {}° >= 2°",
			err.to_degrees()
		);
	}
}

#[cfg(test)]
#[path = "mpc_solver_tests.rs"]
mod mpc_solver_tests;

/// ABI 34: the lag-aware design math (the augmented model, the doubling DARE).
#[cfg(test)]
mod lag_design_tests
{
	use super::*;

	fn b_cf21() -> [f64; 3]
	{
		calibrate_control_gains_rs(0.001, 0.0707, 0.2, 0.0057, [3.0e-5, 3.0e-5, 5.3e-5], 9.81, 0.6942, 0.05)
	}

	fn riccati_residual(a: &Mat, b: &Mat, q: &Mat, r: &Mat, p: &Mat) -> f64
	{
		let at = a.transpose();
		let bt = b.transpose();
		let mid = inverse(&r.add(&bt.matmul(p).matmul(b))).unwrap();
		let rhs = q
			.add(&at.matmul(p).matmul(a))
			.sub(&at.matmul(p).matmul(b).matmul(&mid).matmul(&bt.matmul(p).matmul(a)));
		let scale = p.d.iter().fold(1.0f64, |m, v| m.max(v.abs()));
		rhs.max_abs_diff(p) / scale
	}

	/// The doubling DARE solves the Riccati equation on every augmented model the
	/// teachers build (LQR/MPC 9-state, LQI 12-state) at both axis values.
	#[test]
	fn doubling_dare_solves_the_augmented_riccati()
	{
		for tt in [0.15f32, 0.30]
		{
			let alpha = motor_lag_alpha(tt, 0.001);
			for integral in [false, true]
			{
				let (a, b) = attitude_lag_model(b_cf21(), 0.001, alpha, integral);
				let q = lag_state_cost(Q_ATT, Q_RATE, if integral { Some(Q_INT) } else { None }, R_CTRL);
				let r = lag_cmd_cost(R_CTRL);
				let p = dare_doubling(&a, &b, &q, &r);
				let res = riccati_residual(&a, &b, &q, &r, &p);
				assert!(res < 1e-9, "T={tt} integral={integral}: relative Riccati residual {res:e}");
			}
		}
	}

	/// α = 1 (T clamped to 4dt: no lag) collapses the augmented design onto the
	/// lag-free one: the actuator columns of K vanish and the [e, rate] columns
	/// equal the 6-state discrete LQR on I + A·dt, B·dt.
	#[test]
	fn lag_alpha_one_collapses_to_lag_free()
	{
		let b = b_cf21();
		let r = diag3([R_CTRL; 3]);
		let (a9, b9) = attitude_lag_model(b, 0.001, 1.0, false);
		let k9 = dlqr_gain(&a9, &b9, &lag_state_cost(Q_ATT, Q_RATE, None, R_CTRL), &lag_cmd_cost(R_CTRL));
		let (a_c, b_c) = attitude_linear_model(b);
		let a6 = Mat::identity(6).add(&a_c.scale(0.001));
		let b6 = b_c.scale(0.001);
		let k6 = dlqr_gain(&a6, &b6, &diag6([Q_ATT, Q_ATT, Q_ATT, Q_RATE, Q_RATE, Q_RATE]), &r);
		for i in 0..3
		{
			for j in 0..6
			{
				let (x, y) = (k9.get(i, j), k6.get(i, j));
				assert!((x - y).abs() <= 1e-9 * y.abs().max(1.0), "K[{i},{j}] {x} vs {y}");
			}
			for j in 6..9
			{
				assert!(k9.get(i, j).abs() < 1e-12, "K[{i},{j}] actuator column {} ≠ 0 at α=1", k9.get(i, j));
			}
		}
	}

	/// The Newton–Kleinman CARE solves the continuous Riccati equation on the
	/// LQR (9-state) and LQI (12-state) augmented models at both axis values.
	#[test]
	fn care_solves_the_augmented_riccati()
	{
		for tt in [0.15f32, 0.30]
		{
			for integral in [false, true]
			{
				let (a, b) = attitude_lag_model_continuous(b_cf21(), motor_lag_tau(tt, 0.001), integral);
				let q = lag_state_cost(Q_ATT, Q_RATE, if integral { Some(Q_INT) } else { None }, R_CTRL);
				let r = lag_cmd_cost(R_CTRL);
				let p = care(&a, &b, &q, &r);
				let r_inv = inverse(&r).unwrap();
				let res = a
					.transpose()
					.matmul(&p)
					.add(&p.matmul(&a))
					.sub(&p.matmul(&b).matmul(&r_inv).matmul(&b.transpose()).matmul(&p))
					.add(&q);
				let scale = p.d.iter().fold(1.0f64, |m, v| m.max(v.abs()));
				let worst = res.d.iter().fold(0.0f64, |m, v| m.max(v.abs())) / scale;
				assert!(worst < 1e-8, "T={tt} integral={integral}: relative CARE residual {worst:e}");
			}
		}
	}

	/// τ → 0 converges onto the LAG-FREE teachers' closed forms (continuity with
	/// the T = 0 path): LQR's k1/k2 and LQI's k_att/k_rate/k_int.
	///
	/// ABI 35: with the effort split (LAG_CMD_SHARE) the comparison is on the
	/// DELIVERED law, not the raw K columns. The design feeds back the actuator
	/// state (u = −K_x·x − k_ua·ua), and as τ → 0 the actuator sits on its quasi-
	/// steady manifold ua = u, so the rotors deliver ua = −K_x·x / (1 + k_ua). That
	/// effective gain is what must equal the lag-free closed form; the raw K_x is
	/// sqrt(q/(ε·r)) — the command-only weight — and differs by design.
	#[test]
	fn care_lag_design_converges_to_closed_form()
	{
		let (dt, arm, kt, kd, inertia, g, hover) = (0.001f32, 0.0707f32, 0.2f32, 0.0057f32, [3.0e-5f32, 3.0e-5, 5.3e-5], 9.81f32, 0.6942);
		let lqr = AttitudeLqrRs::build(dt, arm, kt, kd, inertia, g, hover, 0.4, Q_ATT, Q_RATE, R_CTRL);
		let lqi = AttitudeLqiRs::build(dt, arm, kt, kd, inertia, g, hover, 0.4, Q_ATT, Q_RATE, R_CTRL, Q_INT);
		let tau = 1e-6;
		let r = lag_cmd_cost(R_CTRL);
		let (a, b) = attitude_lag_model_continuous(b_cf21(), tau, false);
		let k = clqr_gain(&a, &b, &lag_state_cost(Q_ATT, Q_RATE, None, R_CTRL), &r);
		let (ai, bi) = attitude_lag_model_continuous(b_cf21(), tau, true);
		let ki = clqr_gain(&ai, &bi, &lag_state_cost(Q_ATT, Q_RATE, Some(Q_INT), R_CTRL), &r);
		let close = |x: f64, y: f64, what: &str| {
			assert!((x - y).abs() <= 1e-2 * y.abs().max(1e-3), "{what}: {x} vs closed form {y}");
		};
		for axis in 0..3
		{
			let d = 1.0 + k.get(axis, 6 + axis);
			close(k.get(axis, axis) / d, lqr.k.get(axis, axis), "LQR angle");
			close(k.get(axis, 3 + axis) / d, lqr.k.get(axis, 3 + axis), "LQR rate");
			let di = 1.0 + ki.get(axis, 9 + axis);
			close(ki.get(axis, axis) / di, lqi.k_int[axis], "LQI int");
			close(ki.get(axis, 3 + axis) / di, lqi.k.get(axis, axis), "LQI angle");
			close(ki.get(axis, 6 + axis) / di, lqi.k.get(axis, 3 + axis), "LQI rate");
		}
	}

	/// ABI 35 rule pin: the lag-aware cost prices the DELIVERED actuator
	/// ((1 − ε)·r on ua) plus a small command term (ε·r on u), ε = 0.1, and the
	/// split sums to the lag-free r (what makes the τ → 0 limit the lag-free
	/// design). A change to ε or to the split must fail here, not drift silently.
	#[test]
	fn lag_effort_split_rule()
	{
		assert_eq!(LAG_CMD_SHARE, 0.1);
		for integral in [false, true]
		{
			let q = lag_state_cost(Q_ATT, Q_RATE, if integral { Some(Q_INT) } else { None }, 2.0);
			let off = if integral { 9 } else { 6 };
			let rc = lag_cmd_cost(2.0);
			for i in 0..3
			{
				let (q_ua, r_u) = (q.get(off + i, off + i), rc.get(i, i));
				assert_eq!(q_ua, 0.9 * 2.0);
				assert_eq!(r_u, 0.1 * 2.0);
				assert!((q_ua + r_u - 2.0).abs() < 1e-15);
			}
		}
	}

	/// The augmented closed loop is stable: x' = (A − B K) x decays	/// The augmented closed loop is stable: x' = (A − B K) x decays from a unit
	/// state on every axis (power iteration bound via a long free response).
	#[test]
	fn lag_aware_closed_loop_is_stable()
	{
		for tt in [0.15f32, 0.30]
		{
			for integral in [false, true]
			{
				let (a, b) = attitude_lag_model(b_cf21(), 0.001, motor_lag_alpha(tt, 0.001), integral);
				let q = lag_state_cost(Q_ATT, Q_RATE, if integral { Some(Q_INT) } else { None }, R_CTRL);
				let k = dlqr_gain(&a, &b, &q, &lag_cmd_cost(R_CTRL));
				let acl = a.sub(&b.matmul(&k));
				let mut x = vec![1.0f64; a.r];
				for _ in 0..200_000
				{
					x = acl.mul_vec(&x);
				}
				let norm = x.iter().fold(0.0f64, |m, v| m.max(v.abs()));
				assert!(norm < 1e-3, "T={tt} integral={integral}: closed loop not decaying (|x|={norm:e})");
			}
		}
	}
}

