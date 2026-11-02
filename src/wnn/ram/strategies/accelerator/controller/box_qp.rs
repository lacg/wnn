//! Exact solver for the MPC teacher's box-constrained QP (01/11/2026).
//!
//!     min ½uᵀHu + gᵀu   s.t.  −a ≤ u_i ≤ a,      H symmetric positive definite.
//!
//! WHY EXACT. The teacher used projected FISTA capped at 200 iterations. H is
//! 75×75 with cond(H) ≈ 499, so FISTA needs ~√κ·ln(1/ε) ≈ 450+ iterations to
//! reach its own 1e-9 step tolerance: on student-driven rollouts 93-95% of the
//! constrained solves exited AT the cap with a KKT residual up to 3.6e-2 — the
//! label was "wherever FISTA stopped", not the MPC law (mpc_solver_tests.rs).
//!
//! METHOD. H is SPD, so the optimum is unique and is fixed by its active set:
//! with the bound coordinates pinned at ±a, the free block solves
//! H_FF u_F = −(g_F + H_FB u_B) exactly (Cholesky). We search the active set by
//! the primal-dual active-set method (Hintermüller–Ito–Kunisch), warm-started
//! from the previous step's solution — it moves many coordinates per iteration
//! and typically settles in 1-3 solves. PDAS is not guaranteed to terminate for
//! a general SPD H, so after PDAS_MAX_ITERS it hands over to the classical
//! primal active-set method (Nocedal & Wright Alg. 16.3), which is finite for a
//! strictly convex QP. Either way the answer is returned only at a KKT point:
//! free coordinates inside the box with zero gradient, bound coordinates with a
//! non-negative multiplier — both to round-off tolerance.
//!
//! The answer is a deterministic function of (H, g, final active set), so it is
//! independent of the search path up to that set. Allocation-free: every buffer
//! is sized once at construction.

pub(crate) struct BoxQpSolver
{
	n: usize,
	h: Vec<f64>, // n×n row-major, SPD
	a: f64,
	u: Vec<f64>,      // candidate solution on the current active set
	x: Vec<f64>,      // feasible iterate (primal fallback only)
	grad: Vec<f64>,   // Hu + g
	state: Vec<i8>,   // 0 free, +1 at +a, −1 at −a
	next: Vec<i8>,    // PDAS proposal
	free: Vec<usize>, // free indices of the current set
	factor: Vec<f64>, // Cholesky L of H_FF (stride = #free)
	rhs: Vec<f64>,
	/// Linear solves the last call used (PDAS + fallback).
	pub(crate) last_solves: usize,
	/// Whether the last call needed the primal active-set fallback.
	pub(crate) last_fallback: bool,
}

const PDAS_MAX_ITERS: usize = 30;

impl BoxQpSolver
{
	pub(crate) fn new(n: usize, h: Vec<f64>, a: f64) -> Self
	{
		assert_eq!(h.len(), n * n, "BoxQpSolver: H must be n×n");
		BoxQpSolver {
			n,
			h,
			a,
			u: vec![0.0; n],
			x: vec![0.0; n],
			grad: vec![0.0; n],
			state: vec![0; n],
			next: vec![0; n],
			free: Vec::with_capacity(n),
			factor: vec![0.0; n * n],
			rhs: vec![0.0; n],
			last_solves: 0,
			last_fallback: false,
		}
	}

	/// The exact minimizer. `warm` (len n) seeds the active set: coordinates at
	/// or beyond ±a start bound. Returns the solution (borrowed until next call).
	pub(crate) fn solve(&mut self, g: &[f64], warm: &[f64]) -> &[f64]
	{
		debug_assert_eq!(g.len(), self.n);
		self.last_solves = 0;
		self.last_fallback = false;
		self.seed_active_set(warm);
		if !self.pdas(g)
		{
			self.last_fallback = true;
			self.primal_active_set(g);
		}
		&self.u
	}

	fn seed_active_set(&mut self, warm: &[f64])
	{
		let a = self.a;
		for (s, &w) in self.state.iter_mut().zip(warm)
		{
			*s = if w >= a { 1 } else if w <= -a { -1 } else { 0 };
		}
	}

	fn tol_u(&self) -> f64
	{
		1e-12 * self.a.max(1.0)
	}

	fn tol_g(&self, g: &[f64]) -> f64
	{
		1e-11 * (1.0 + g.iter().fold(0.0f64, |m, v| m.max(v.abs())))
	}

	/// Primal-dual active set. True when it settled on a KKT point (self.u).
	fn pdas(&mut self, g: &[f64]) -> bool
	{
		let (tol_u, tol_g) = (self.tol_u(), self.tol_g(g));
		for _ in 0..PDAS_MAX_ITERS
		{
			self.solve_on_set(g);
			self.gradient(g);
			self.propose_next(tol_u, tol_g);
			if self.next == self.state
			{
				self.clamp_free();
				return true;
			}
			std::mem::swap(&mut self.state, &mut self.next);
		}
		false
	}

	/// HIK update with c-free form: a free coordinate binds when it left the box;
	/// a bound coordinate frees when its multiplier turned negative.
	fn propose_next(&mut self, tol_u: f64, tol_g: f64)
	{
		let a = self.a;
		for i in 0..self.n
		{
			self.next[i] = match self.state[i]
			{
				0 if self.u[i] > a + tol_u => 1,
				0 if self.u[i] < -a - tol_u => -1,
				0 => 0,
				s if (s as f64) * self.grad[i] > tol_g => 0, // multiplier −s·grad < 0
				s => s,
			};
		}
	}

	/// Classical primal active-set (Nocedal & Wright Alg. 16.3) from the
	/// clamped PDAS iterate: feasible iterates, ratio-test steps, drop the most
	/// negative multiplier at each face optimum. Finite for strictly convex QPs.
	fn primal_active_set(&mut self, g: &[f64])
	{
		let (a, tol_g) = (self.a, self.tol_g(g));
		for i in 0..self.n
		{
			self.x[i] = self.u[i].clamp(-a, a);
			self.state[i] = if self.x[i] >= a { 1 } else if self.x[i] <= -a { -1 } else { 0 };
		}
		for _ in 0..(50 * self.n)
		{
			self.solve_on_set(g);
			if self.step_to_blocking()
			{
				continue;
			}
			self.u.copy_from_slice(&self.x);
			self.gradient(g);
			if !self.release_worst_multiplier(tol_g)
			{
				return;
			}
		}
		panic!("BoxQpSolver: primal active-set did not terminate in {} steps", 50 * self.n);
	}

	/// Move x toward the face optimum u. True when a bound blocked the step (x
	/// stops on it and the coordinate joins the active set); false when x
	/// reached u.
	fn step_to_blocking(&mut self) -> bool
	{
		let a = self.a;
		let (mut alpha, mut block) = (1.0f64, usize::MAX);
		for &i in &self.free
		{
			let p = self.u[i] - self.x[i];
			let t = if p > 0.0 && self.u[i] > a
			{
				(a - self.x[i]) / p
			}
			else if p < 0.0 && self.u[i] < -a
			{
				(-a - self.x[i]) / p
			}
			else
			{
				continue;
			};
			if t < alpha
			{
				(alpha, block) = (t, i);
			}
		}
		if block == usize::MAX
		{
			for &i in &self.free
			{
				self.x[i] = self.u[i];
			}
			return false;
		}
		for &i in &self.free
		{
			self.x[i] += alpha * (self.u[i] - self.x[i]);
		}
		let s: i8 = if self.u[block] > a { 1 } else { -1 };
		self.x[block] = s as f64 * a;
		self.state[block] = s;
		true
	}

	/// Free the bound coordinate with the most negative multiplier. False when
	/// every multiplier is non-negative (x is the KKT point).
	fn release_worst_multiplier(&mut self, tol_g: f64) -> bool
	{
		let (mut worst, mut idx) = (tol_g, usize::MAX);
		for i in 0..self.n
		{
			let viol = self.state[i] as f64 * self.grad[i];
			if self.state[i] != 0 && viol > worst
			{
				(worst, idx) = (viol, i);
			}
		}
		if idx == usize::MAX
		{
			return false;
		}
		self.state[idx] = 0;
		true
	}

	fn clamp_free(&mut self)
	{
		let a = self.a;
		for v in self.u.iter_mut()
		{
			*v = v.clamp(-a, a);
		}
	}

	fn gradient(&mut self, g: &[f64])
	{
		let n = self.n;
		for i in 0..n
		{
			let row = &self.h[i * n..(i + 1) * n];
			let mut acc = 0.0;
			for j in 0..n
			{
				acc += row[j] * self.u[j];
			}
			self.grad[i] = acc + g[i];
		}
	}

	/// u = the minimizer with the bound coordinates pinned at state·a: the free
	/// block solves H_FF u_F = −(g_F + H_FB u_B) by Cholesky.
	fn solve_on_set(&mut self, g: &[f64])
	{
		self.last_solves += 1;
		let (n, a) = (self.n, self.a);
		self.free.clear();
		for i in 0..n
		{
			if self.state[i] == 0
			{
				self.free.push(i);
			}
			else
			{
				self.u[i] = self.state[i] as f64 * a;
			}
		}
		let nf = self.free.len();
		if nf == 0
		{
			return;
		}
		self.assemble_reduced(g);
		self.cholesky(nf);
		self.chol_solve(nf);
		for r in 0..nf
		{
			self.u[self.free[r]] = self.rhs[r];
		}
	}

	fn assemble_reduced(&mut self, g: &[f64])
	{
		let (n, a, nf) = (self.n, self.a, self.free.len());
		for r in 0..nf
		{
			let i = self.free[r];
			let row = &self.h[i * n..(i + 1) * n];
			let mut acc = -g[i];
			for j in 0..n
			{
				if self.state[j] != 0
				{
					acc -= row[j] * (self.state[j] as f64 * a);
				}
			}
			self.rhs[r] = acc;
			for c in 0..nf
			{
				self.factor[r * nf + c] = row[self.free[c]];
			}
		}
	}

	/// In-place lower Cholesky of the nf×nf block in `factor`.
	fn cholesky(&mut self, nf: usize)
	{
		let f = &mut self.factor;
		for j in 0..nf
		{
			let mut d = f[j * nf + j];
			for k in 0..j
			{
				d -= f[j * nf + k] * f[j * nf + k];
			}
			assert!(d > 0.0, "BoxQpSolver: H_FF not positive definite (pivot {d})");
			let d = d.sqrt();
			f[j * nf + j] = d;
			for i in (j + 1)..nf
			{
				let mut s = f[i * nf + j];
				for k in 0..j
				{
					s -= f[i * nf + k] * f[j * nf + k];
				}
				f[i * nf + j] = s / d;
			}
		}
	}

	/// rhs ← (L Lᵀ)⁻¹ rhs.
	fn chol_solve(&mut self, nf: usize)
	{
		let (f, b) = (&self.factor, &mut self.rhs);
		for i in 0..nf
		{
			let mut s = b[i];
			for k in 0..i
			{
				s -= f[i * nf + k] * b[k];
			}
			b[i] = s / f[i * nf + i];
		}
		for i in (0..nf).rev()
		{
			let mut s = b[i];
			for k in (i + 1)..nf
			{
				s -= f[k * nf + i] * b[k];
			}
			b[i] = s / f[i * nf + i];
		}
	}
}
