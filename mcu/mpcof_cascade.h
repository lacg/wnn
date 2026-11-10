/* On-board MPCOF cascade OUTER loop — the C twin of ram_controller ctbr_mpcof.rs
 * (CtbrMpcOf, ABI 39-49): the offset-free MPC whose prediction model is the closed
 * inner loop (firmware rate PID + motor lag + rigid body) and whose decision is the
 * body-rate setpoint [p, q, r] it hands that loop. CTRL-51 (10/11/2026): ported to
 * measure what the teacher costs on the same Cortex-M4 bench as the WNN cascade.
 *
 * ONLINE ONLY. The condensed horizon-20 QP (H, g_map, ustar_map per axis and per
 * rate-loop phase) and its DARE terminal cost are computed OFFLINE by the Rust
 * teacher's own functions and exported as tables (mcu/mpcof_model.h, written by
 * mcu/ctrl51/export_mpcof.py) — what a real build would do. Online, per physical
 * tick: the observer (cascade_dhat.c), the shadow rate stage (wnn_ratefw.c), and on
 * every n_o-th tick the state build + per-axis plan: U* = ustar_map * x~, accepted
 * when the whole plan is inside the box, else the exact box QP (box_qp.rs: PDAS
 * warm-started from the shifted previous plan, primal active-set fallback).
 *
 * FIRMWARE CONTRACT, every physical tick (1 kHz), in this order:
 *   mpcof_observe(m, gyro, applied_prev)   gyro just read, pwm applied LAST tick
 *   mpcof_label(m, q, gyro, target, sp)    sp = held body-rate setpoint (rad/s)
 * q = attitude (w,x,y,z) from the vehicle's estimator, target = RPY reference (rad).
 * The collective is not planned here (the altitude loop's, as in Rust).
 *
 * PRECISION. double = the Rust reference, bit-exact on the host with
 * -ffp-contract=off (mcu/ctrl51/parity_mpcof.py). On an M4F (single-precision FPU)
 * every double op is a libgcc soft-float call. -DMPCOF_F32 runs everything in float
 * (hardware FPU): cheaper, NOT bit-identical — its error is measured, not assumed. */
#ifndef MPCOF_CASCADE_H
#define MPCOF_CASCADE_H

#include <stdint.h>

#include "cascade_dhat.h"
#include "wnn_ratefw.h"

#define MPCOF_MODEL_MACROS_ONLY
#include "mpcof_model.h"
#undef MPCOF_MODEL_MACROS_ONLY

#ifdef MPCOF_F32
typedef float mpc_real;
#else
typedef double mpc_real;
#endif

#define MPCOF_N MPCOF_HORIZON

/* One axis's box QP workspace (box_qp.rs BoxQpSolver, allocation-free). */
typedef struct
{
	mpc_real u[MPCOF_N], x[MPCOF_N], grad[MPCOF_N], rhs[MPCOF_N];
	mpc_real factor[MPCOF_N * MPCOF_N];
	int8_t state[MPCOF_N], next[MPCOF_N];
	uint8_t free_idx[MPCOF_N];
	int nfree;
	uint32_t last_solves;  /* linear (Cholesky) solves of the last call */
	int last_fallback;     /* the primal active-set fallback ran */
} mpcof_qp_t;

/* One axis's planner state (ctbr_axis_mpc.rs CtbrAxisMpc): warm start + buffers. */
typedef struct
{
	mpc_real warm[MPCOF_N], u_buf[MPCOF_N], g_buf[MPCOF_N];
	mpcof_qp_t qp;
} mpcof_axis_t;

/* Per-decision telemetry (benchmark/parity only; the control law never reads it). */
typedef struct
{
	uint32_t qp_axes;      /* axes whose unconstrained plan left the box (QP ran) */
	uint32_t solves;       /* Cholesky solves, all axes */
	uint32_t fallbacks;    /* primal active-set fallbacks, all axes */
	int failed;            /* the fallback hit its step cap (Rust panics) */
} mpcof_stats_t;

typedef struct
{
	cdhat_t obs;
	wnn_ratefw_t shadow;
	mpcof_axis_t ax[3];
	uint32_t tick;
	mpc_real held[3];
	mpcof_stats_t last;
} mpcof_t;

/* Once at boot (tables -> observer + shadow constants). */
void mpcof_init(mpcof_t *m);
/* Every flight: CtbrMpcOf::reset (shadow, observer, tick, held, warm starts). */
void mpcof_reset(mpcof_t *m);
/* CtbrMpcOf::observe(gyro_t, applied_{t-1}). */
void mpcof_observe(mpcof_t *m, const float gyro[3], const float applied_prev[4]);
/* CtbrMpcOf::label: a fresh plan on decision ticks (tick % n_o == 0), the held one
 * otherwise; then the shadow rate stage steps. Returns 1 on a decision tick. */
int mpcof_label(mpcof_t *m, const float q[4], const float gyro[3], const float target[3], mpc_real sp[3]);

/* The two halves of mpcof_label, for the benchmark's cost split. */
int mpcof_decide_if_due(mpcof_t *m, const float q[4], const float gyro[3], const float target[3]);
void mpcof_shadow_step(mpcof_t *m, const float gyro[3]);

#endif /* MPCOF_CASCADE_H */
