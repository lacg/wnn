/* CTRL-51 (10/11/2026): on-board cost of the WNN cascade vs the MPCOF cascade
 * teacher, same Cortex-M4 bench (QEMU mps2-an386), REAL trajectories.
 *
 * Every physical tick is replayed from a recorded episode file read over ARM
 * semihosting (outside the measured regions), and each controller call is
 * bracketed by MARK_BEGIN / MARK_END: the insnmark.c plugin then reports the
 * instructions of EVERY call, so the cost per decision has a mean AND a worst case.
 *
 *   -DBENCH_MPCOF  mpcof_cascade.c on mcu/ctrl51 traj_*.bin (the Rust teacher's own
 *                  episodes): per tick three regions [observe][decide-if-due]
 *                  [shadow rate stage]; the held setpoint is checked against the
 *                  Rust label every tick (bit equality, f64 build).
 *   -DBENCH_WNN    wnn_cascade.c on record_wnn_flights.py's stream: per tick three
 *                  regions [wnn_tick][the d-hat observer alone][the Mahony attitude
 *                  estimator alone] — the two on TWIN instances fed the same inputs,
 *                  so each costs what it costs INSIDE wnn_tick (same code, same state
 *                  trajectory). wnn_cascade.c is included into this TU (as the parity
 *                  shim does) to reach its static estimator; wnn_tick is called
 *                  through a volatile pointer so it is never inlined here;
 *                  the firmware rate loop (wnn_ratefw.c) closes the loop outside the
 *                  regions, and every motor command is checked against Rust's.
 * Both begin with one EMPTY region (MARK_BEGIN; MARK_END) = the marker overhead.
 * argv-free: the trajectory path is -DTRAJ_PATH="...". Output: semihosting text. */
#include "bench.h"

#include <string.h>

#define MARK_BEGIN() __asm__ volatile("mov r11, r11" ::: "memory")
#define MARK_END() __asm__ volatile("mov r10, r10" ::: "memory")

/* ------------------------------------------------ semihosting file I/O --- */

static int sh_call(int op, void *args)
{
	register int r0 __asm__("r0") = op;
	register void *r1 __asm__("r1") = args;
	__asm__ volatile("bkpt 0xAB" : "+r"(r0) : "r"(r1) : "memory");
	return r0;
}

static int sh_open(const char *path)
{
	uint32_t a[3] = {(uint32_t)path, 1u /* "rb" */, (uint32_t)strlen(path)};
	return sh_call(0x01, a);
}

/* 1 when exactly n bytes were read. */
static int sh_read(int fd, void *buf, uint32_t n)
{
	uint32_t a[3] = {(uint32_t)fd, (uint32_t)buf, n};
	return sh_call(0x06, a) == 0;
}

static void report_u32(const char *k, uint32_t v) { report(k, v); }

/* ------------------------------------------------------------- MPCOF ---- */
#ifdef BENCH_MPCOF
#include "mpcof_cascade.h"

typedef struct
{
	float gyro[3], applied[4], q[4], tgt[3];
	double label[3], hidden[12];
} rec_t;

static mpcof_t g_m;
static uint32_t n_ticks, n_dec, n_mismatch, n_qp_dec, n_solves, max_solves, n_fallback, n_failed;
static double max_err;   /* max |C - Rust| held setpoint, rad/s */

static void count_stats(int due)
{
	if (!due) return;
	n_dec++;
	n_qp_dec += g_m.last.qp_axes > 0;
	n_solves += g_m.last.solves;
	max_solves = g_m.last.solves > max_solves ? g_m.last.solves : max_solves;
	n_fallback += g_m.last.fallbacks;
	n_failed += (uint32_t)g_m.last.failed;
}

static void tick(const rec_t *r)
{
	MARK_BEGIN();
	mpcof_observe(&g_m, r->gyro, r->applied);
	MARK_END();
	MARK_BEGIN();
	int due = mpcof_decide_if_due(&g_m, r->q, r->gyro, r->tgt);
	MARK_END();
	MARK_BEGIN();
	mpcof_shadow_step(&g_m, r->gyro);
	MARK_END();
	count_stats(due);
	for (int a = 0; a < 3; a++)
	{
		double sp = (double)g_m.held[a], d = sp - r->label[a];
		d = d < 0 ? -d : d;
		max_err = d > max_err ? d : max_err;
		n_mismatch += memcmp(&sp, &r->label[a], 8) != 0;
	}
	n_ticks++;
}

static void run(int fd)
{
	uint32_t n_eps = 0, steps = 0;
	static rec_t r;
	mpcof_init(&g_m);
	sh_read(fd, &n_eps, 4);
	for (uint32_t e = 0; e < n_eps && sh_read(fd, &steps, 4); e++)
	{
		mpcof_reset(&g_m);
		for (uint32_t t = 0; t < steps && sh_read(fd, &r, sizeof r); t++) tick(&r);
	}
	report_u32("TICKS", n_ticks);
	report_u32("DECISIONS", n_dec);
	report_u32("SETPOINT_MISMATCH_AXES", n_mismatch);
	/* in units of 1e-15 rad/s, saturating */
	report_u32("SETPOINT_MAX_ABS_ERR_FEMTO", max_err * 1e15 > 4e9 ? 4000000000u : (uint32_t)(max_err * 1e15));
	report_u32("QP_DECISIONS", n_qp_dec);
	report_u32("QP_SOLVES", n_solves);
	report_u32("QP_MAX_SOLVES_PER_DECISION", max_solves);
	report_u32("QP_FALLBACKS", n_fallback);
	report_u32("QP_FAILED", n_failed);
}
#endif

/* --------------------------------------------------------------- WNN ---- */
#ifdef BENCH_WNN
#include "wnn_cascade.c"
#include "wnn_ratefw.h"

typedef struct
{
	float gyro[3], accel[3], target[3], vert[3], horiz[4], pwm[4];
} wrec_t;

static wnn_ctrl_t g_c;
static cdhat_t g_twin;
static wnn_mahony_t g_est_twin;
static int (*volatile g_tick_fn)(wnn_ctrl_t *, const wnn_tick_in_t *, float *) = wnn_tick;

__attribute__((noinline)) static void est_twin(const float gyro[3], const float accel[3])
{
	float q[4];
	mahony_update(&g_est_twin, gyro, accel, q);
	__asm__ volatile("" : : "r"(q) : "memory");
}
static wnn_ratefw_t g_rate;
static float g_applied[4];
static uint32_t n_ticks, n_dec, n_mismatch;
static const double rate_gains[12] = WNN_RATE_GAINS_INIT;

static void episode_reset(const float q0[4], float yaw0, float anchor)
{
	wnn_reset(&g_c, yaw0, q0);
	g_twin = g_c.dhat;
	g_est_twin = g_c.est;
	wnn_ratefw_reset(&g_rate);
	wnn_ratefw_set_anchor(&g_rate, anchor);
	for (int m = 0; m < 4; m++) g_applied[m] = 0.5f;
}

static void fill_in(wnn_tick_in_t *in, const wrec_t *r)
{
	memcpy(in->gyro, r->gyro, sizeof in->gyro);
	memcpy(in->accel, r->accel, sizeof in->accel);
	memcpy(in->target, r->target, sizeof in->target);
	memcpy(in->vert, r->vert, sizeof in->vert);
	memcpy(in->horiz, r->horiz, sizeof in->horiz);
	wnn_ratefw_integ(&g_rate, in->rate_integ);
	memcpy(in->applied_prev, g_applied, sizeof g_applied);
}

static void tick(const wrec_t *r)
{
	wnn_tick_in_t in;
	float sp[4], pwm[4];
	fill_in(&in, r);
	MARK_BEGIN();
	n_dec += (uint32_t)g_tick_fn(&g_c, &in, sp);
	MARK_END();
	MARK_BEGIN();
	cdhat_observe(&g_twin, in.gyro, in.applied_prev);
	MARK_END();
	MARK_BEGIN();
	est_twin(in.gyro, in.accel);
	MARK_END();
	wnn_ratefw_fly(&g_rate, sp, r->gyro, pwm);
	n_mismatch += memcmp(pwm, r->pwm, sizeof pwm) != 0;
	memcpy(g_applied, pwm, sizeof pwm);
	n_ticks++;
}

static void run(int fd)
{
	uint32_t n_eps = 0, steps = 0;
	float hdr[6];
	static wrec_t r;
	wnn_init(&g_c);
	wnn_ratefw_init(&g_rate, rate_gains, WNN_RATE_OUT_LIMIT_N, WNN_RATE_HOVER_N, WNN_K_THRUST,
	                WNN_MAIN_LOOP_HZ, WNN_ATTITUDE_HZ, WNN_RATE_LPF_HZ);
	sh_read(fd, &n_eps, 4);
	for (uint32_t e = 0; e < n_eps && sh_read(fd, &steps, 4) && sh_read(fd, hdr, sizeof hdr); e++)
	{
		episode_reset(hdr, hdr[4], hdr[5]);
		for (uint32_t t = 0; t < steps && sh_read(fd, &r, sizeof r); t++) tick(&r);
	}
	report_u32("TICKS", n_ticks);
	report_u32("DECISIONS", n_dec);
	report_u32("PWM_MISMATCH_TICKS", n_mismatch);
}
#endif

/* Static state each controller keeps between ticks (bytes of .bss). */
static void report_state_size(void)
{
#ifdef BENCH_MPCOF
	report_u32("STATE_BYTES", (uint32_t)sizeof(mpcof_t));
#else
	report_u32("STATE_BYTES", (uint32_t)(sizeof(wnn_ctrl_t)));
#endif
}

/* Stack high-water mark: paint 32 KB below the current SP, scan after the run.
 * Reported = bytes below the painter's SP that were touched — the controller's
 * deepest call chain plus the replay loop's own small frames (an upper bound). */
#define PAINT_WORDS 8192u
#define PAINT 0xA5A5A5A5u
static uint32_t *g_paint_lo, *g_paint_hi;

__attribute__((noinline)) static void stack_paint(void)
{
	uint32_t sp;
	__asm__ volatile("mov %0, sp" : "=r"(sp));
	g_paint_hi = (uint32_t *)(sp - 64u);
	g_paint_lo = g_paint_hi - PAINT_WORDS;
	for (uint32_t *q = g_paint_lo; q < g_paint_hi; q++) *q = PAINT;
}

static uint32_t stack_peak(void)
{
	uint32_t *q = g_paint_lo;
	while (q < g_paint_hi && *q == PAINT) q++;
	return (uint32_t)((uint8_t *)g_paint_hi - (uint8_t *)q) + 64u;
}

/* Kept out of main: once tick() inlines, a function's prologue may save VFP
 * registers, and touching the FPU before fpu_enable() faults (no handler: lockup). */
__attribute__((noinline)) static void bench(void)
{
	report_state_size();
	MARK_BEGIN();
	MARK_END();
	int fd = sh_open(TRAJ_PATH);
	if (fd < 0) { put_str("OPEN_FAILED\n"); sh_exit(); }
	stack_paint();
	run(fd);
	report_u32("STACK_PEAK_BYTES", stack_peak());
}

int main(void)
{
	fpu_enable();
	bench();
	sh_exit();
	return 0;
}
