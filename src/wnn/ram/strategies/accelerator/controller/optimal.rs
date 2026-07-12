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
//!     and solved by projected FISTA, warm-started, with an unconstrained fast path
//!     (when the ±authority box does not bind the solution is the LQR feedback).
//!
//! The teachers expose the same `step_rs(q, gyro, target) -> [f64;4]` interface as
//! `AttitudePidRs`, so they drop into `rollout_and_label_rs` unchanged. Being
//! label generators (not the deployed artifact — the WNN is), a little iteration is
//! fine here.

use crate::controller::{
	calibrate_control_gains_rs, clamp_f64, mix_to_motors_f64, quat_to_euler_f64, wrap_angle_f64,
	AttitudePidRs,
};
use pyo3::prelude::*;

// ===========================================================================
// Minimal dense f64 matrix (row-major). Only what the teachers need.
// ===========================================================================

#[derive(Clone)]
pub(crate) struct Mat {
	r: usize,
	c: usize,
	d: Vec<f64>,
}

impl Mat {
	fn zeros(r: usize, c: usize) -> Self {
		Mat { r, c, d: vec![0.0; r * c] }
	}
	fn identity(n: usize) -> Self {
		let mut m = Mat::zeros(n, n);
		for i in 0..n {
			m.set(i, i, 1.0);
		}
		m
	}
	#[inline]
	fn get(&self, i: usize, j: usize) -> f64 {
		self.d[i * self.c + j]
	}
	#[inline]
	fn set(&mut self, i: usize, j: usize, v: f64) {
		self.d[i * self.c + j] = v;
	}
	fn transpose(&self) -> Mat {
		let mut t = Mat::zeros(self.c, self.r);
		for i in 0..self.r {
			for j in 0..self.c {
				t.set(j, i, self.get(i, j));
			}
		}
		t
	}
	fn matmul(&self, o: &Mat) -> Mat {
		assert_eq!(self.c, o.r, "matmul shape");
		let mut out = Mat::zeros(self.r, o.c);
		for i in 0..self.r {
			for k in 0..self.c {
				let a = self.get(i, k);
				if a == 0.0 {
					continue;
				}
				for j in 0..o.c {
					let v = out.get(i, j) + a * o.get(k, j);
					out.set(i, j, v);
				}
			}
		}
		out
	}
	fn add(&self, o: &Mat) -> Mat {
		let mut out = self.clone();
		for x in 0..self.d.len() {
			out.d[x] += o.d[x];
		}
		out
	}
	fn sub(&self, o: &Mat) -> Mat {
		let mut out = self.clone();
		for x in 0..self.d.len() {
			out.d[x] -= o.d[x];
		}
		out
	}
	fn scale(&self, s: f64) -> Mat {
		let mut out = self.clone();
		for x in out.d.iter_mut() {
			*x *= s;
		}
		out
	}
	fn max_abs_diff(&self, o: &Mat) -> f64 {
		let mut m = 0.0f64;
		for x in 0..self.d.len() {
			let dd = (self.d[x] - o.d[x]).abs();
			if dd > m {
				m = dd;
			}
		}
		m
	}
	/// Matrix × column vector (len = self.c) → column vector (len = self.r).
	fn mul_vec(&self, v: &[f64]) -> Vec<f64> {
		assert_eq!(self.c, v.len(), "mul_vec shape");
		let mut out = vec![0.0; self.r];
		for i in 0..self.r {
			let mut acc = 0.0;
			for j in 0..self.c {
				acc += self.get(i, j) * v[j];
			}
			out[i] = acc;
		}
		out
	}
}

/// General square-matrix inverse via Gauss-Jordan with partial pivoting.
/// Returns None if singular. Used for the MPC condensed Hessian (3N×3N).
fn inverse(a: &Mat) -> Option<Mat> {
	assert_eq!(a.r, a.c, "inverse: square");
	let n = a.r;
	let mut m = a.clone();
	let mut inv = Mat::identity(n);
	for col in 0..n {
		// pivot = largest |value| in this column at/below the diagonal.
		let mut piv = col;
		let mut best = m.get(col, col).abs();
		for r in (col + 1)..n {
			let v = m.get(r, col).abs();
			if v > best {
				best = v;
				piv = r;
			}
		}
		if best < 1e-14 {
			return None;
		}
		if piv != col {
			for j in 0..n {
				let (a1, a2) = (m.get(col, j), m.get(piv, j));
				m.set(col, j, a2);
				m.set(piv, j, a1);
				let (b1, b2) = (inv.get(col, j), inv.get(piv, j));
				inv.set(col, j, b2);
				inv.set(piv, j, b1);
			}
		}
		let d = m.get(col, col);
		for j in 0..n {
			m.set(col, j, m.get(col, j) / d);
			inv.set(col, j, inv.get(col, j) / d);
		}
		for r in 0..n {
			if r == col {
				continue;
			}
			let f = m.get(r, col);
			if f == 0.0 {
				continue;
			}
			for j in 0..n {
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
fn dare(a: &Mat, b: &Mat, q: &Mat, r: &Mat) -> Mat {
	let at = a.transpose();
	let bt = b.transpose();
	let mut p = q.clone();
	for _ in 0..10_000 {
		let atp = at.matmul(&p); // AᵀP
		let atpa = atp.matmul(a); // AᵀPA
		let atpb = atp.matmul(b); // AᵀPB
		let btpb = bt.matmul(&p).matmul(b); // BᵀPB
		let mid = r.add(&btpb); // R + BᵀPB  (m×m)
		let mid_inv = match inverse(&mid) {
			Some(x) => x,
			None => break,
		};
		let btpa = bt.matmul(&p).matmul(a); // BᵀPA
		let corr = atpb.matmul(&mid_inv).matmul(&btpa); // AᵀPB (R+BᵀPB)⁻¹ BᵀPA
		let p_new = q.add(&atpa).sub(&corr);
		let delta = p_new.max_abs_diff(&p);
		p = p_new;
		if delta < 1e-12 {
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

fn attitude_linear_model(b: [f64; 3]) -> (Mat, Mat) {
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

fn diag6(d: [f64; 6]) -> Mat {
	let mut m = Mat::zeros(6, 6);
	for i in 0..6 {
		m.set(i, i, d[i]);
	}
	m
}
fn diag3(d: [f64; 3]) -> Mat {
	let mut m = Mat::zeros(3, 3);
	for i in 0..3 {
		m.set(i, i, d[i]);
	}
	m
}

/// State error vector x = [roll-tr, pitch-tp, yaw-ty, p, q, r] (angles wrapped).
#[inline]
fn state_error(q: [f32; 4], gyro: [f32; 3], target_rpy: [f32; 3]) -> [f64; 6] {
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

// ===========================================================================
// LQR teacher — u = clamp(−Kx, ±authority), K from the discrete Riccati.
// ===========================================================================

/// Default LQR/MPC cost weights (match optimal.py defaults: q_att=12, q_rate=1,
/// r_ctrl=1). authority/hover default to the PID teacher's (0.4 / 0.5).
pub(crate) const Q_ATT: f64 = 12.0;
pub(crate) const Q_RATE: f64 = 1.0;
pub(crate) const R_CTRL: f64 = 1.0;

#[pyclass]
pub struct AttitudeLqrRs {
	k: Mat, // 3×6 feedback gain
	hover: f64,
	authority: f64,
}

impl AttitudeLqrRs {
	/// Build from explicit sim params (so the plant model matches the controlled
	/// sim) + cost weights + hover/authority.
	#[allow(clippy::too_many_arguments)]
	pub(crate) fn build(
		dt: f32, arm_length: f32, k_thrust: f32, k_drag: f32, inertia: [f32; 3], gravity: f32,
		hover: f64, authority: f64, q_att: f64, q_rate: f64, r_ctrl: f64,
	) -> Self {
		let b = calibrate_control_gains_rs(dt, arm_length, k_thrust, k_drag, inertia, gravity, hover, 0.05);
		// The plant is DECOUPLED per axis (roll/pitch/yaw are independent double
		// integrators ẍ = b·u), so the continuous-CARE LQR has a CLOSED FORM per
		// axis — no Riccati iteration, and bit-faithful to scipy solve_continuous_are:
		//   k1 = √(q_att/r_ctrl)                      (angle gain, axis-independent)
		//   k2 = √((2·√(q_att·r_ctrl)/b_i + q_rate)/r_ctrl)   (rate gain, per-axis via b_i)
		// (Discrete DARE would give the discrete-time optimum — a slightly softer,
		// different controller; we want the same teacher Python uses.)
		let k1 = (q_att / r_ctrl).sqrt();
		let mut k = Mat::zeros(3, 6);
		for axis in 0..3 {
			let k2 = ((2.0 * (q_att * r_ctrl).sqrt() / b[axis] + q_rate) / r_ctrl).sqrt();
			k.set(axis, axis, k1); // angle error → u
			k.set(axis, axis + 3, k2); // body rate → u
		}
		AttitudeLqrRs { k, hover, authority }
	}

	/// Native f64 step (mirrors AttitudePidRs::step_rs signature).
	pub fn step_rs(&mut self, q: [f32; 4], gyro: [f32; 3], target_rpy: [f32; 3]) -> [f64; 4] {
		let x = state_error(q, gyro, target_rpy);
		let u = self.k.mul_vec(&x); // K x  (3,)
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
impl AttitudeLqrRs {
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
		dt: f32, arm_length: f32, k_thrust: f32, k_drag: f32, inertia: [f32; 3], gravity: f32,
		hover: f64, authority: f64, q_att: f64, q_rate: f64, r_ctrl: f64,
	) -> Self {
		Self::build(dt, arm_length, k_thrust, k_drag, inertia, gravity, hover, authority, q_att, q_rate, r_ctrl)
	}
	pub fn reset(&mut self) {} // memoryless
	/// One step → 4 motor PWMs (f32), for Python parity vs optimal.LQRController.
	fn step(&mut self, q: [f32; 4], gyro: [f32; 3], target_rpy: [f32; 3]) -> [f32; 4] {
		let p = self.step_rs(q, gyro, target_rpy);
		[p[0] as f32, p[1] as f32, p[2] as f32, p[3] as f32]
	}
	/// Flattened 3×6 gain, row-major (for parity vs optimal.LQRController.K).
	fn gain(&self) -> Vec<f64> {
		self.k.d.clone()
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
pub struct AllocBaseline {
	pub k1: f32,
	pub k2: [f32; 3],
	pub tau_max: f32,
	pub f_hover: f32,
	/// Per rotor: pinv row m0..m5 (+ nominal k_thrust) — T_i = mᵢ·w.
	pub rows: Vec<[f32; 7]>,
}

impl AllocBaseline {
	/// Build from nominal geometry rows + LQR cost weights (same closed-form
	/// per-axis CARE as AttitudeLqrRs, torque plant b = 1/I).
	pub fn build(
		rows9: &[[f32; 9]], inertia: [f32; 3],
		q_att: f64, q_rate: f64, r_ctrl: f64,
		tau_max: f64, f_hover: Option<f64>, lambda: f32,
	) -> Result<Self, String> {
		let geo = crate::overactuated::RotorGeometry::from_rows(rows9)?;
		if tau_max <= 0.0 {
			return Err(format!("tau_max must be > 0, got {tau_max}"));
		}
		let k1 = (q_att / r_ctrl).sqrt() as f32;
		let mut k2 = [0.0f32; 3];
		for axis in 0..3 {
			let b = 1.0 / inertia[axis] as f64;
			k2[axis] = (((2.0 * (q_att * r_ctrl).sqrt() / b + q_rate) / r_ctrl).sqrt()) as f32;
		}
		// Hover collective: every rotor at PWM 0.5 on the nominal geometry.
		let f_hover = f_hover.unwrap_or_else(|| {
			geo.rotors.iter().map(|r| (r.k_thrust * 0.25) as f64).sum()
		}) as f32;
		let pinv = geo.allocation_pinv(lambda);
		let rows = pinv.iter().zip(&geo.rotors)
			.map(|(m, r)| [m[0], m[1], m[2], m[3], m[4], m[5], r.k_thrust])
			.collect();
		Ok(AllocBaseline { k1, k2, tau_max: tau_max as f32, f_hover, rows })
	}

	pub fn num_rotors(&self) -> usize {
		self.rows.len()
	}

	/// One baseline step, all-f32 — the bit-level template for the kernel's
	/// alloc_step: euler(q) → per-axis τ = clamp(k1·e − k2·rate, ±τ_max) →
	/// T = M·(τ; 0,0,F_hover) → pwm = √(max(T,0)/k) clamped [0,1].
	pub fn pwm(&self, q: [f32; 4], gyro: [f32; 3], target_rpy: [f32; 3], out: &mut [f32]) {
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
		for (i, row) in self.rows.iter().enumerate() {
			let mut thrust = 0.0f32;
			for j in 0..6 {
				thrust += row[j] * wrench[j];
			}
			out[i] = (thrust.max(0.0) / row[6]).sqrt().clamp(0.0, 1.0);
		}
	}

	/// Flat f32 blob for the GPU (buffer 28) — header + 7-float rotor rows.
	pub fn to_gpu_blob(&self) -> Vec<f32> {
		let mut v = Vec::with_capacity(6 + self.rows.len() * 7);
		v.extend_from_slice(&[self.k1, self.k2[0], self.k2[1], self.k2[2], self.tau_max, self.f_hover]);
		for r in &self.rows {
			v.extend_from_slice(r);
		}
		v
	}
}

#[pyclass]
pub struct AllocLqrRs {
	base: AllocBaseline, // NOMINAL geometry + gains (allocator side)
}

impl AllocLqrRs {
	/// Plain-Rust constructor (house pattern: String errors; the pymethod wraps).
	pub(crate) fn build_core(
		rows: &[[f32; 9]], inertia: [f32; 3],
		q_att: f64, q_rate: f64, r_ctrl: f64,
		tau_max: f64, f_hover: Option<f64>, lambda: f32,
	) -> Result<Self, String> {
		Ok(AllocLqrRs {
			base: AllocBaseline::build(rows, inertia, q_att, q_rate, r_ctrl, tau_max, f_hover, lambda)?,
		})
	}

	/// Native step: attitude state → N motor PWMs via LQR torque + allocation.
	pub fn step_alloc_rs(&mut self, q: [f32; 4], gyro: [f32; 3], target_rpy: [f32; 3]) -> Vec<f64> {
		let mut out = vec![0.0f32; self.base.num_rotors()];
		self.base.pwm(q, gyro, target_rpy, &mut out);
		out.into_iter().map(|p| p as f64).collect()
	}
}

#[pymethods]
impl AllocLqrRs {
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
		rows: Vec<[f32; 9]>, inertia: [f32; 3],
		q_att: f64, q_rate: f64, r_ctrl: f64,
		tau_max: f64, f_hover: Option<f64>, pinv_lambda: f32,
	) -> PyResult<Self> {
		Self::build_core(&rows, inertia, q_att, q_rate, r_ctrl, tau_max, f_hover, pinv_lambda)
			.map_err(pyo3::exceptions::PyValueError::new_err)
	}
	pub fn reset(&mut self) {} // memoryless
	/// One step → N motor PWMs (f32).
	fn step(&mut self, q: [f32; 4], gyro: [f32; 3], target_rpy: [f32; 3]) -> Vec<f32> {
		let mut out = vec![0.0f32; self.base.num_rotors()];
		self.base.pwm(q, gyro, target_rpy, &mut out);
		out
	}
	fn num_rotors(&self) -> usize {
		self.base.num_rotors()
	}
	/// Torque-space gains [k1, k2_roll, k2_pitch, k2_yaw].
	fn gain(&self) -> Vec<f64> {
		vec![self.base.k1 as f64, self.base.k2[0] as f64, self.base.k2[1] as f64, self.base.k2[2] as f64]
	}
}

// ===========================================================================
// MPC teacher — condensed box-constrained QP solved per step by projected FISTA.
// ===========================================================================

#[pyclass]
pub struct AttitudeMpcRs {
	n: usize,           // horizon
	hover: f64,
	authority: f64,
	h: Mat,             // condensed Hessian (3N×3N), cost = ½UᵀHU + gᵀU
	ustar_map: Mat,     // −H⁻¹G (3N×6): unconstrained U* = ustar_map · x0
	g_map: Mat,         // G (3N×6): g = g_map · x0
	inv_l: f64,         // FISTA step = 1/L, L = spectral-radius upper bound of H
	warm: Vec<f64>,     // warm-start U from the previous step (shifted)
}

impl AttitudeMpcRs {
	#[allow(clippy::too_many_arguments)]
	pub(crate) fn build(
		dt: f32, arm_length: f32, k_thrust: f32, k_drag: f32, inertia: [f32; 3], gravity: f32,
		hover: f64, authority: f64, q_att: f64, q_rate: f64, r_ctrl: f64,
		horizon: usize, dt_mpc: f64,
	) -> Self {
		let b = calibrate_control_gains_rs(dt, arm_length, k_thrust, k_drag, inertia, gravity, hover, 0.05);
		let (a_c, b_c) = attitude_linear_model(b);
		let ad = Mat::identity(6).add(&a_c.scale(dt_mpc));
		let bd = b_c.scale(dt_mpc);
		let qm = diag6([q_att, q_att, q_att, q_rate, q_rate, q_rate]);
		let rm = diag3([r_ctrl, r_ctrl, r_ctrl]);
		let qf = dare(&ad, &bd, &qm, &rm); // terminal cost-to-go (discrete LQR)

		let n = horizon;
		let ns = 6usize;
		let nu = 3usize;
		let rows = ns * (n + 1);
		let cols = nu * n;
		// Prediction matrices: X = Sx·x0 + Su·U, X stacked over k=0..N.
		// Sx block-row k = Ad^k ; Su block (k,j)=Ad^{k-1-j}·Bd for j<k else 0.
		let mut ad_pow: Vec<Mat> = Vec::with_capacity(n + 1);
		ad_pow.push(Mat::identity(ns)); // Ad^0
		for k in 1..=n {
			ad_pow.push(ad_pow[k - 1].matmul(&ad));
		}
		let mut sx = Mat::zeros(rows, ns);
		let mut su = Mat::zeros(rows, cols);
		for k in 0..=n {
			// Sx block
			for i in 0..ns {
				for j in 0..ns {
					sx.set(k * ns + i, j, ad_pow[k].get(i, j));
				}
			}
			// Su blocks
			for j in 0..k {
				let blk = ad_pow[k - 1 - j].matmul(&bd); // 6×3
				for ii in 0..ns {
					for jj in 0..nu {
						su.set(k * ns + ii, j * nu + jj, blk.get(ii, jj));
					}
				}
			}
		}
		// Qbar = blkdiag(Q,...,Q [k=0..N-1], Qf [k=N]); Rbar = blkdiag(R,...,R).
		let mut qbar = Mat::zeros(rows, rows);
		for k in 0..n {
			for i in 0..ns {
				for j in 0..ns {
					qbar.set(k * ns + i, k * ns + j, qm.get(i, j));
				}
			}
		}
		for i in 0..ns {
			for j in 0..ns {
				qbar.set(n * ns + i, n * ns + j, qf.get(i, j));
			}
		}
		let mut rbar = Mat::zeros(cols, cols);
		for k in 0..n {
			for i in 0..nu {
				for j in 0..nu {
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

		// L = spectral-radius upper bound of H via a few power iterations.
		let mut v = vec![1.0f64; cols];
		let mut l = 1.0f64;
		for _ in 0..50 {
			let hv = h.mul_vec(&v);
			let norm = hv.iter().map(|x| x * x).sum::<f64>().sqrt();
			if norm < 1e-30 {
				break;
			}
			l = norm;
			for i in 0..cols {
				v[i] = hv[i] / norm;
			}
		}
		let inv_l = 1.0 / (l * 1.05); // 5% margin for safe descent

		AttitudeMpcRs {
			n,
			hover,
			authority,
			h,
			ustar_map,
			g_map,
			inv_l,
			warm: vec![0.0; cols],
		}
	}

	pub fn step_rs(&mut self, q: [f32; 4], gyro: [f32; 3], target_rpy: [f32; 3]) -> [f64; 4] {
		let x = state_error(q, gyro, target_rpy);
		let a = self.authority;
		let cols = self.n * 3;
		// Unconstrained optimum U* = ustar_map · x0. If it satisfies the box for the
		// whole horizon it IS the constrained optimum (fast path == LQR feedback).
		let u_unc = self.ustar_map.mul_vec(&x);
		let feasible = u_unc.iter().all(|&ui| ui.abs() <= a + 1e-9);
		let u = if feasible {
			u_unc
		} else {
			// Projected FISTA on ½UᵀHU + gᵀU s.t. |U| ≤ a. Warm-start from the
			// previous shifted solution clamped into the box.
			let g = self.g_map.mul_vec(&x);
			let mut uk: Vec<f64> = self.warm.iter().map(|&w| clamp_f64(w, -a, a)).collect();
			let mut yk = uk.clone();
			let mut t = 1.0f64;
			for _ in 0..200 {
				let hy = self.h.mul_vec(&yk); // H y
				let mut u_next = vec![0.0; cols];
				for i in 0..cols {
					let grad = hy[i] + g[i];
					u_next[i] = clamp_f64(yk[i] - self.inv_l * grad, -a, a);
				}
				let t_next = 0.5 * (1.0 + (1.0 + 4.0 * t * t).sqrt());
				let beta = (t - 1.0) / t_next;
				let mut max_step = 0.0f64;
				for i in 0..cols {
					let step = u_next[i] - uk[i];
					yk[i] = u_next[i] + beta * step;
					if step.abs() > max_step {
						max_step = step.abs();
					}
				}
				uk = u_next;
				t = t_next;
				if max_step < 1e-9 {
					break;
				}
			}
			uk
		};
		// Warm start next step: shift U by one control (receding horizon).
		for i in 0..(cols - 3) {
			self.warm[i] = u[i + 3];
		}
		for i in (cols - 3)..cols {
			self.warm[i] = u[i];
		}
		mix_to_motors_f64(
			self.hover,
			clamp_f64(u[0], -a, a),
			clamp_f64(u[1], -a, a),
			clamp_f64(u[2], -a, a),
		)
	}
}

#[pymethods]
impl AttitudeMpcRs {
	#[new]
	#[pyo3(signature = (
		dt = 0.001, arm_length = 0.075, k_thrust = 2.4, k_drag = 0.05,
		inertia = [0.0023, 0.0023, 0.0046], gravity = 9.81,
		hover = 0.5, authority = 0.4, q_att = 12.0, q_rate = 1.0, r_ctrl = 1.0,
		horizon = 25, dt_mpc = 0.001
	))]
	#[allow(clippy::too_many_arguments)]
	pub fn new(
		dt: f32, arm_length: f32, k_thrust: f32, k_drag: f32, inertia: [f32; 3], gravity: f32,
		hover: f64, authority: f64, q_att: f64, q_rate: f64, r_ctrl: f64,
		horizon: usize, dt_mpc: f64,
	) -> Self {
		Self::build(dt, arm_length, k_thrust, k_drag, inertia, gravity, hover, authority, q_att, q_rate, r_ctrl, horizon, dt_mpc)
	}
	pub fn reset(&mut self) {
		for w in self.warm.iter_mut() {
			*w = 0.0;
		}
	}
	fn step(&mut self, q: [f32; 4], gyro: [f32; 3], target_rpy: [f32; 3]) -> [f32; 4] {
		let p = self.step_rs(q, gyro, target_rpy);
		[p[0] as f32, p[1] as f32, p[2] as f32, p[3] as f32]
	}
}

// ===========================================================================
// Teacher enum — the DAGGER loop's expert slot. Dispatches step/reset/integrals.
// ===========================================================================

pub enum Teacher {
	Pid(AttitudePidRs),
	Lqr(AttitudeLqrRs),
	Mpc(AttitudeMpcRs),
}

impl Teacher {
	/// Construct the teacher selected by cfg (0=pid, 1=lqr, 2=mpc), using the
	/// SAME sim params the DAGGER loop controls (so the LQR/MPC plant matches).
	/// hover/authority mirror the PID teacher's defaults (0.5 / 0.4).
	pub fn from_id(
		id: u8, dt: f32, arm_length: f32, k_thrust: f32, k_drag: f32, inertia: [f32; 3], gravity: f32,
	) -> Teacher {
		let (hover, authority) = (0.5, 0.4);
		match id {
			1 => Teacher::Lqr(AttitudeLqrRs::build(
				dt, arm_length, k_thrust, k_drag, inertia, gravity, hover, authority, Q_ATT, Q_RATE, R_CTRL,
			)),
			2 => Teacher::Mpc(AttitudeMpcRs::build(
				dt, arm_length, k_thrust, k_drag, inertia, gravity, hover, authority, Q_ATT, Q_RATE, R_CTRL, 25, 0.001,
			)),
			_ => Teacher::Pid(pid_default_teacher()),
		}
	}

	#[inline]
	pub fn step_rs(&mut self, q: [f32; 4], gyro: [f32; 3], target_rpy: [f32; 3]) -> [f64; 4] {
		match self {
			Teacher::Pid(p) => p.step_rs(q, gyro, target_rpy),
			Teacher::Lqr(l) => l.step_rs(q, gyro, target_rpy),
			Teacher::Mpc(m) => m.step_rs(q, gyro, target_rpy),
		}
	}
	#[inline]
	pub fn reset(&mut self) {
		match self {
			Teacher::Pid(p) => p.reset(),
			Teacher::Lqr(l) => l.reset(),
			Teacher::Mpc(m) => m.reset(),
		}
	}
	/// Teacher integral (roll,pitch,yaw). LQR/MPC are memoryless → zeros (the
	/// Option-A integral target only applies to the PID teacher).
	#[inline]
	pub fn integrals(&self) -> [f32; 3] {
		match self {
			Teacher::Pid(p) => p.integrals(),
			_ => [0.0, 0.0, 0.0],
		}
	}
	/// Clamp magnitudes for normalizing the integral. LQR/MPC: 1.0 (no scaling;
	/// integral is zero anyway).
	#[inline]
	pub fn i_clamps(&self) -> [f32; 3] {
		match self {
			Teacher::Pid(p) => p.i_clamps(),
			_ => [1.0, 1.0, 1.0],
		}
	}
}

/// PID teacher with the canonical defaults (mirrors dagger_train::pid_default).
fn pid_default_teacher() -> AttitudePidRs {
	AttitudePidRs::new(1.2, 0.05, 0.30, 0.5, 0.6, 0.02, 0.20, 0.5, 0.5, 0.4, 0.001)
}

#[cfg(test)]
mod alloc_lqr_tests {
	use super::*;
	use crate::controller::AttitudeSim;
	use crate::overactuated::RotorGeometry;

	const ARM: f32 = 0.075;
	const KT: f32 = 2.4;
	const KD: f32 = 0.05;
	const INERTIA: [f32; 3] = [0.0023, 0.0023, 0.0046];

	fn rows_from(geo: &RotorGeometry) -> Vec<[f32; 9]> {
		geo.rotors.iter().map(|r| [
			r.position[0], r.position[1], r.position[2],
			r.axis[0], r.axis[1], r.axis[2],
			r.spin, r.k_thrust, r.k_drag,
		]).collect()
	}

	fn octo_rows() -> Vec<[f32; 9]> {
		rows_from(&RotorGeometry::octo_x(ARM, KT, KD))
	}

	fn teacher(rows: &[[f32; 9]]) -> AllocLqrRs {
		AllocLqrRs::build_core(rows, INERTIA, 12.0, 1.0, 1.0, 0.144, None, 1e-6)
			.expect("teacher")
	}

	/// Roll out the teacher closed-loop on a sim carrying `sim_rows` (the TRUE
	/// vehicle) while the teacher allocates on `nom_rows` (the NOMINAL model).
	/// Returns final attitude error (rad).
	fn closed_loop_err(nom_rows: &[[f32; 9]], sim_rows: &[[f32; 9]], tilt_deg: f32, steps: usize) -> f32 {
		let mut t = teacher(nom_rows);
		let mut sim = AttitudeSim::new(0.001, ARM, KT, KD, INERTIA, 9.81);
		sim.set_geometry_core(sim_rows.to_vec()).expect("sim geometry");
		let half = tilt_deg.to_radians() * 0.5;
		sim.reset(Some([half.cos(), half.sin(), 0.0, 0.0]), Some([0.0, 0.0, 0.0]));
		for _ in 0..steps {
			assert!(!sim.is_unstable(), "sim diverged under the alloc-LQR teacher");
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
	fn hover_allocation_at_zero_error() {
		let rows = octo_rows();
		let mut t = teacher(&rows);
		let pwm = t.step_alloc_rs([1.0, 0.0, 0.0, 0.0], [0.0; 3], [0.0; 3]);
		assert_eq!(pwm.len(), 8);
		for (i, p) in pwm.iter().enumerate() {
			assert!((p - 0.5).abs() < 0.02, "rotor {i}: pwm {p} not ≈ hover 0.5");
		}
	}

	/// Small-signal consistency: the torque the allocated PWMs actually produce
	/// on the NOMINAL geometry matches the clamped LQR demand.
	#[test]
	fn allocation_realizes_torque_demand() {
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
		let pwm: Vec<f32> = t.step_alloc_rs(q, [0.0; 3], [0.0; 3]).iter().map(|&p| p as f32).collect();
		let got = geo.body_torque(&pwm);
		for a in 0..3 {
			assert!((got[a] - want[a]).abs() < 0.144 * 0.05 + 2e-3,
				"axis {a}: realized torque {} vs demand {}", got[a], want[a]);
		}
	}

	/// The teacher stabilizes a NOMINAL octo from a 17° tilt.
	#[test]
	fn stabilizes_nominal_octo() {
		let rows = octo_rows();
		let err = closed_loop_err(&rows, &rows, 17.0, 1500);
		assert!(err < 2.0_f32.to_radians(), "final err {}° >= 2°", err.to_degrees());
	}

	/// The teacher (nominal allocator) still stabilizes a PERTURBED true
	/// vehicle — tilt/position error + the geometry mismatch the WNN residual
	/// will learn. Bounded, not perfect: allow a residual offset.
	#[test]
	fn stabilizes_perturbed_octo_with_nominal_allocator() {
		let rows = octo_rows();
		let tilt: Vec<f32> = [1.5f32, -2.0, 1.0, -0.8, 1.8, -1.2, 0.6, -1.6]
			.iter().map(|d| d.to_radians()).collect();
		let pos: Vec<[f32; 3]> = (0..8)
			.map(|i| [0.002 * (i as f32 - 3.5), -0.0015 * (i as f32 - 3.5), 0.001])
			.collect();
		let true_geo = RotorGeometry::from_rows(&rows).unwrap().perturbed(&tilt, &pos);
		let sim_rows = rows_from(&true_geo);
		let err = closed_loop_err(&rows, &sim_rows, 17.0, 1500);
		assert!(err < 5.0_f32.to_radians(),
			"perturbed-vehicle err {}° >= 5° (teacher must stay bounded)", err.to_degrees());
	}

	/// Quad-plus as geometry: the allocator teacher also stabilizes the
	/// paper-#1 quad (rank-deficient Fx/Fy handled by the damped pinv).
	#[test]
	fn stabilizes_quad_plus_geometry() {
		let rows = rows_from(&RotorGeometry::quad_plus(ARM, KT, KD));
		let err = closed_loop_err(&rows, &rows, 17.0, 1500);
		assert!(err < 2.0_f32.to_radians(), "quad final err {}° >= 2°", err.to_degrees());
	}
}
