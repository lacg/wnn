/* See mpcof_cascade.h. Every function names the Rust it mirrors (ram_controller
 * @ ABI 49: ctbr_mpcof.rs, ctbr_axis_mpc.rs, box_qp.rs, optimal.rs state_error);
 * keep the operation ORDER — the parity test asserts bit equality in double, and a
 * reassociated sum is a different function. Build with -ffp-contract=off. */
#include "mpcof_cascade.h"
#include "mpcof_model.h"

#include <math.h>
#include <string.h>

#ifdef MPCOF_F32
#define R_SQRT sqrtf
#define R_ATAN2 atan2f
#define R_ASIN asinf
#define R_FABS fabsf
#define R_PI 3.14159265358979323846f
/* box_qp.rs tolerances are f64 round-off scales (1e-12, 1e-11); in f32 they sit
 * below the epsilon and the active-set tests would chase round-off. f32 ONLY. */
#define TOL_U_REL 1e-6f
#define TOL_G_REL 1e-5f
#else
#define R_SQRT sqrt
#define R_ATAN2 atan2
#define R_ASIN asin
#define R_FABS fabs
#define R_PI 3.14159265358979323846
#define TOL_U_REL 1e-12
#define TOL_G_REL 1e-11
#endif

#define N MPCOF_N
/* Typed literal: in the f32 build a bare 2.0 would promote to (soft) double. */
#define R(x) ((mpc_real)(x))
#define NS MPCOF_NS

/* ctbr_axis_mpc.rs state indices: [theta, omega, ua, I, f1, f2, pm, Fh]. */
enum { S_TH = 0, S_W, S_UA, S_IN, S_F1, S_F2, S_PM, S_FH };

static mpc_real clamp_r(mpc_real x, mpc_real lo, mpc_real hi)
{
	if (x < lo) x = lo;
	if (x > hi) x = hi;
	return x;
}

/* ------------------------------------------- state_error (optimal.rs) ---- */

/* controller.rs wrap_angle_f64. */
static mpc_real wrap_angle(mpc_real a)
{
	while (a > R_PI) a -= R(2.0) * R_PI;
	while (a <= -R_PI) a += R(2.0) * R_PI;
	return a;
}

/* controller.rs quat_to_euler_f64 (f32 quaternion widened first). */
static void quat_to_euler(const float q[4], mpc_real e[3])
{
	mpc_real w = q[0], x = q[1], y = q[2], z = q[3];
	mpc_real sinr_cosp = R(2.0) * (w * x + y * z);
	mpc_real cosr_cosp = R(1.0) - R(2.0) * (x * x + y * y);
	e[0] = R_ATAN2(sinr_cosp, cosr_cosp);
	mpc_real sinp = R(2.0) * (w * y - z * x);
	e[1] = (sinp >= R(1.0)) ? R_PI / R(2.0) : (sinp <= R(-1.0)) ? -R_PI / R(2.0) : R_ASIN(sinp);
	mpc_real siny_cosp = R(2.0) * (w * z + x * y);
	mpc_real cosy_cosp = R(1.0) - R(2.0) * (y * y + z * z);
	e[2] = R_ATAN2(siny_cosp, cosy_cosp);
}

/* optimal.rs state_error: [wrapped Euler error x3, gyro x3]. */
static void state_error(const float q[4], const float gyro[3], const float target[3], mpc_real x[6])
{
	mpc_real e[3];
	quat_to_euler(q, e);
	for (int a = 0; a < 3; a++) x[a] = wrap_angle(e[a] - (mpc_real)target[a]);
	for (int a = 0; a < 3; a++) x[3 + a] = (mpc_real)gyro[a];
}

/* ------------------------------------------------- box QP (box_qp.rs) ---- */

static void qp_seed(mpcof_qp_t *q, const mpc_real *warm, mpc_real a)
{
	for (int i = 0; i < N; i++) q->state[i] = (warm[i] >= a) ? 1 : (warm[i] <= -a) ? -1 : 0;
}

static mpc_real qp_tol_g(const mpc_real *g)
{
	mpc_real m = R(0.0);
	for (int i = 0; i < N; i++) m = (R_FABS(g[i]) > m) ? R_FABS(g[i]) : m;
	return TOL_G_REL * (R(1.0) + m);
}

/* grad = H u + g. */
static void qp_gradient(mpcof_qp_t *q, const mpc_real *h, const mpc_real *g)
{
	for (int i = 0; i < N; i++)
	{
		const mpc_real *row = &h[i * N];
		mpc_real acc = R(0.0);
		for (int j = 0; j < N; j++) acc += row[j] * q->u[j];
		q->grad[i] = acc + g[i];
	}
}

/* Reduced system H_FF u_F = -(g_F + H_FB u_B) into factor / rhs. */
static void qp_assemble(mpcof_qp_t *q, const mpc_real *h, const mpc_real *g, mpc_real a)
{
	int nf = q->nfree;
	for (int r = 0; r < nf; r++)
	{
		int i = q->free_idx[r];
		const mpc_real *row = &h[i * N];
		mpc_real acc = -g[i];
		for (int j = 0; j < N; j++)
			if (q->state[j] != 0) acc -= row[j] * ((mpc_real)q->state[j] * a);
		q->rhs[r] = acc;
		for (int c = 0; c < nf; c++) q->factor[r * nf + c] = row[q->free_idx[c]];
	}
}

/* In-place lower Cholesky of the nf x nf block. 0 on a non-positive pivot. */
static int qp_cholesky(mpcof_qp_t *q, int nf)
{
	mpc_real *f = q->factor;
	for (int j = 0; j < nf; j++)
	{
		mpc_real d = f[j * nf + j];
		for (int k = 0; k < j; k++) d -= f[j * nf + k] * f[j * nf + k];
		if (!(d > R(0.0))) return 0;
		d = R_SQRT(d);
		f[j * nf + j] = d;
		for (int i = j + 1; i < nf; i++)
		{
			mpc_real s = f[i * nf + j];
			for (int k = 0; k < j; k++) s -= f[i * nf + k] * f[j * nf + k];
			f[i * nf + j] = s / d;
		}
	}
	return 1;
}

/* rhs <- (L L^T)^-1 rhs. */
static void qp_chol_solve(mpcof_qp_t *q, int nf)
{
	const mpc_real *f = q->factor;
	mpc_real *b = q->rhs;
	for (int i = 0; i < nf; i++)
	{
		mpc_real s = b[i];
		for (int k = 0; k < i; k++) s -= f[i * nf + k] * b[k];
		b[i] = s / f[i * nf + i];
	}
	for (int i = nf - 1; i >= 0; i--)
	{
		mpc_real s = b[i];
		for (int k = i + 1; k < nf; k++) s -= f[k * nf + i] * b[k];
		b[i] = s / f[i * nf + i];
	}
}

/* BoxQpSolver::solve_on_set: the minimiser with the bound coordinates pinned. */
static int qp_solve_on_set(mpcof_qp_t *q, const mpc_real *h, const mpc_real *g, mpc_real a)
{
	q->last_solves++;
	q->nfree = 0;
	for (int i = 0; i < N; i++)
	{
		if (q->state[i] == 0) q->free_idx[q->nfree++] = (uint8_t)i;
		else q->u[i] = (mpc_real)q->state[i] * a;
	}
	int nf = q->nfree;
	if (nf == 0) return 1;
	qp_assemble(q, h, g, a);
	if (!qp_cholesky(q, nf)) return 0;
	qp_chol_solve(q, nf);
	for (int r = 0; r < nf; r++) q->u[q->free_idx[r]] = q->rhs[r];
	return 1;
}

/* HIK update: a free coordinate binds when it left the box; a bound one frees
 * when its multiplier turned negative. 1 when the proposal equals the set. */
static int qp_propose_next(mpcof_qp_t *q, mpc_real a, mpc_real tol_u, mpc_real tol_g)
{
	int same = 1;
	for (int i = 0; i < N; i++)
	{
		int8_t s = q->state[i], n;
		if (s == 0) n = (q->u[i] > a + tol_u) ? 1 : (q->u[i] < -a - tol_u) ? -1 : 0;
		else n = ((mpc_real)s * q->grad[i] > tol_g) ? 0 : s;
		q->next[i] = n;
		same &= (n == s);
	}
	return same;
}

/* Primal-dual active set: 1 at a KKT point (q->u), 0 after PDAS_MAX_ITERS = 30. */
static int qp_pdas(mpcof_qp_t *q, const mpc_real *h, const mpc_real *g, mpc_real a, int *ok)
{
	mpc_real tol_u = TOL_U_REL * ((a > R(1.0)) ? a : R(1.0)), tol_g = qp_tol_g(g);
	for (int it = 0; it < 30; it++)
	{
		if (!qp_solve_on_set(q, h, g, a)) { *ok = 0; return 1; }
		qp_gradient(q, h, g);
		if (qp_propose_next(q, a, tol_u, tol_g))
		{
			for (int i = 0; i < N; i++) q->u[i] = clamp_r(q->u[i], -a, a);
			return 1;
		}
		memcpy(q->state, q->next, sizeof q->state);
	}
	return 0;
}

/* Move x toward the face optimum u; 1 when a bound blocked the step. */
static int qp_step_to_blocking(mpcof_qp_t *q, mpc_real a)
{
	mpc_real alpha = R(1.0);
	int block = -1;
	for (int r = 0; r < q->nfree; r++)
	{
		int i = q->free_idx[r];
		mpc_real p = q->u[i] - q->x[i], t;
		if (p > R(0.0) && q->u[i] > a) t = (a - q->x[i]) / p;
		else if (p < R(0.0) && q->u[i] < -a) t = (-a - q->x[i]) / p;
		else continue;
		if (t < alpha) { alpha = t; block = i; }
	}
	if (block < 0)
	{
		for (int r = 0; r < q->nfree; r++) q->x[q->free_idx[r]] = q->u[q->free_idx[r]];
		return 0;
	}
	for (int r = 0; r < q->nfree; r++)
	{
		int i = q->free_idx[r];
		q->x[i] += alpha * (q->u[i] - q->x[i]);
	}
	int8_t s = (q->u[block] > a) ? 1 : -1;
	q->x[block] = (mpc_real)s * a;
	q->state[block] = s;
	return 1;
}

/* Free the bound coordinate with the most negative multiplier; 0 when none. */
static int qp_release_worst(mpcof_qp_t *q, mpc_real tol_g)
{
	mpc_real worst = tol_g;
	int idx = -1;
	for (int i = 0; i < N; i++)
	{
		mpc_real viol = (mpc_real)q->state[i] * q->grad[i];
		if (q->state[i] != 0 && viol > worst) { worst = viol; idx = i; }
	}
	if (idx < 0) return 0;
	q->state[idx] = 0;
	return 1;
}

/* Classical primal active set (Nocedal & Wright 16.3) from the clamped PDAS
 * iterate. 0 if it hits the 50*N step cap (Rust panics there). */
static int qp_primal(mpcof_qp_t *q, const mpc_real *h, const mpc_real *g, mpc_real a)
{
	mpc_real tol_g = qp_tol_g(g);
	for (int i = 0; i < N; i++)
	{
		q->x[i] = clamp_r(q->u[i], -a, a);
		q->state[i] = (q->x[i] >= a) ? 1 : (q->x[i] <= -a) ? -1 : 0;
	}
	for (int it = 0; it < 50 * N; it++)
	{
		if (!qp_solve_on_set(q, h, g, a)) return 0;
		if (qp_step_to_blocking(q, a)) continue;
		memcpy(q->u, q->x, sizeof q->u);
		qp_gradient(q, h, g);
		if (!qp_release_worst(q, tol_g)) return 1;
	}
	return 0;
}

/* BoxQpSolver::solve: the exact minimiser in q->u; 0 on a failure Rust panics on. */
static int qp_solve(mpcof_qp_t *q, const mpc_real *h, const mpc_real *g, const mpc_real *warm, mpc_real a)
{
	int ok = 1;
	q->last_solves = 0;
	q->last_fallback = 0;
	qp_seed(q, warm, a);
	if (qp_pdas(q, h, g, a, &ok)) return ok;
	q->last_fallback = 1;
	return qp_primal(q, h, g, a);
}

/* ------------------------------------- per-axis plan (ctbr_axis_mpc.rs) -- */

/* Mat::mul_vec_into: out = M (N x NS) v. */
static void mat_vec(const mpc_real *m, const mpc_real v[NS], mpc_real out[N])
{
	for (int i = 0; i < N; i++)
	{
		mpc_real acc = R(0.0);
		for (int j = 0; j < NS; j++) acc += m[i * NS + j] * v[j];
		out[i] = acc;
	}
}

/* CtbrAxisMpc::solve: the plan's first setpoint from the shifted state x~0. */
static mpc_real axis_solve(mpcof_axis_t *ax, int axis, int phase, const mpc_real x[NS], mpcof_stats_t *st)
{
	mpc_real a = mpcof_rate_max[axis];
	mat_vec(mpcof_U[phase][axis], x, ax->u_buf);
	int inside = 1;
	for (int i = 0; i < N; i++) inside &= (R_FABS(ax->u_buf[i]) <= a + R(1e-9));
	if (!inside)
	{
		mat_vec(mpcof_G[phase][axis], x, ax->g_buf);
		st->failed |= !qp_solve(&ax->qp, mpcof_H[phase][axis], ax->g_buf, ax->warm, a);
		memcpy(ax->u_buf, ax->qp.u, sizeof ax->u_buf);
		st->qp_axes++;
		st->solves += ax->qp.last_solves;
		st->fallbacks += (uint32_t)ax->qp.last_fallback;
	}
	memcpy(ax->warm, &ax->u_buf[1], (N - 1) * sizeof(mpc_real));
	ax->warm[N - 1] = ax->u_buf[N - 1];
	return ax->u_buf[0];
}

/* --------------------------------------------- the decision (ctbr_mpcof.rs) */

/* AxisPlant::equilibrium(d-hat): the rest state's ua/Fh/I offsets and sp_ss. */
static mpc_real equilibrium(int axis, mpc_real d, mpc_real xss[NS])
{
	mpc_real f = -d / mpcof_c[axis];
	for (int i = 0; i < NS; i++) xss[i] = R(0.0);
	xss[S_UA] = f;
	xss[S_FH] = f;
	if (mpcof_ki[axis] != R(0.0))
	{
		xss[S_IN] = f / mpcof_ki[axis];
		return R(0.0);
	}
	return f / mpcof_kp[axis];
}

/* The plan's start state for one axis (CtbrMpcOf::decide's s0). */
static void start_state(const mpcof_t *m, int a, const mpc_real x[6], const cdhat_real *ua, mpc_real s0[NS])
{
	const wnn_pidch_t *p = &m->shadow.ax[a];
	mpc_real held = (mpc_real)m->shadow.held[a];
	s0[S_TH] = x[a];
	s0[S_W] = x[3 + a];
	s0[S_UA] = ua ? (mpc_real)ua[a] : held;
	s0[S_IN] = (mpc_real)p->integ;
	s0[S_F1] = (mpc_real)p->filt.d1;
	s0[S_F2] = (mpc_real)p->filt.d2;
	s0[S_PM] = p->first ? x[3 + a] : (mpc_real)p->prev_measured;
	s0[S_FH] = held;
}

/* CtbrMpcOf::decide. */
static void decide(mpcof_t *m, const float q[4], const float gyro[3], const float target[3])
{
	mpc_real x[6], s0[NS], xss[NS], xt[NS];
	cdhat_real ua[3];
	state_error(q, gyro, target, x);
	int phase = (int)(m->shadow.tick % m->shadow.decimation);
	int have_ua = cdhat_measured_force(&m->obs, ua);
	memset(&m->last, 0, sizeof m->last);
	for (int a = 0; a < 3; a++)
	{
		start_state(m, a, x, have_ua ? ua : 0, s0);
		mpc_real sp_ss = equilibrium(a, (mpc_real)m->obs.dhat[a], xss);
		for (int i = 0; i < NS; i++) xt[i] = s0[i] - xss[i];
		mpc_real b = mpcof_rate_max[a];
		m->held[a] = clamp_r(axis_solve(&m->ax[a], a, phase, xt, &m->last) + sp_ss, -b, b);
	}
}

/* ------------------------------------------------------------- public ---- */

void mpcof_init(mpcof_t *m)
{
	memset(m, 0, sizeof *m);
	cdhat_init(&m->obs, mpcof_obs_c, mpcof_obs_clamp, MPCOF_OBS_KT, MPCOF_OBS_DT, MPCOF_OBS_ALPHA,
	           MPCOF_OBS_L);
	wnn_ratefw_init_coeffs(&m->shadow, mpcof_rate_gains, MPCOF_RATE_OUT_LIMIT, MPCOF_RATE_DT,
	                       MPCOF_DECIMATION, MPCOF_RATE_HAS_LPF ? mpcof_rate_lpf : 0);
	mpcof_reset(m);
}

void mpcof_reset(mpcof_t *m)
{
	wnn_ratefw_reset(&m->shadow);
	cdhat_reset(&m->obs);
	m->tick = 0;
	for (int a = 0; a < 3; a++)
	{
		m->held[a] = R(0.0);
		for (int i = 0; i < N; i++) m->ax[a].warm[i] = R(0.0);
	}
}

void mpcof_observe(mpcof_t *m, const float gyro[3], const float applied_prev[4])
{
	cdhat_observe(&m->obs, gyro, applied_prev);
}

int mpcof_decide_if_due(mpcof_t *m, const float q[4], const float gyro[3], const float target[3])
{
	int due = (m->tick % MPCOF_N_O) == 0;
	if (due) decide(m, q, gyro, target);
	m->tick++;
	return due;
}

void mpcof_shadow_step(mpcof_t *m, const float gyro[3])
{
	double sp[3] = {(double)m->held[0], (double)m->held[1], (double)m->held[2]};
	double g[3] = {(double)gyro[0], (double)gyro[1], (double)gyro[2]};
	wnn_ratefw_update(&m->shadow, sp, g);
}

int mpcof_label(mpcof_t *m, const float q[4], const float gyro[3], const float target[3], mpc_real sp[3])
{
	int due = mpcof_decide_if_due(m, q, gyro, target);
	mpcof_shadow_step(m, gyro);
	for (int a = 0; a < 3; a++) sp[a] = m->held[a];
	return due;
}
