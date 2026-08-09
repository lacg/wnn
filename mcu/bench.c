/* Cortex-M4 control-step cost: WNN vs PID vs MLP.
 *
 * METHOD. QEMU's mps2-an386 does not implement DWT_CYCCNT, so we count RETIRED
 * INSTRUCTIONS with a TCG plugin instead. To isolate the workload from startup
 * and semihosting, every binary is run twice — ITERS=0 and ITERS=N — and the
 * per-step cost is (I_N - I_0) / N. That subtraction cancels all fixed overhead
 * exactly, so the residual is the control step and nothing else.
 *
 * These are INSTRUCTIONS, not cycles. On real Cortex-M4 silicon loads cost ~2
 * cycles, taken branches ~2-3, and flash wait states add more, so hardware
 * cycles run ABOVE this count. Treat the numbers as a lower bound and a
 * like-for-like ratio, not as a silicon timing claim.
 *
 * Build one of: -DBENCH_WNN | -DBENCH_PID | -DBENCH_MLP
 */
#include "bench.h"

#ifndef ITERS
#define ITERS 0
#endif

/* Deterministic input generator, shared by all three so they see identical work.
 * Each of the 15 features is an 8-bit THERMOMETER code (monotone run of ones),
 * which is what the encoder actually produces — a uniformly random bit pattern
 * would give the binary search unrealistic branch behaviour. */
static uint32_t rng_s = 0x12345678u;
static inline uint32_t xr(void) {
	rng_s ^= rng_s << 13; rng_s ^= rng_s >> 17; rng_s ^= rng_s << 5;
	return rng_s;
}
#define NFEAT 15
#define FBITS 8
static uint8_t inbits[NFEAT * FBITS];
static uint8_t inlevel[NFEAT];
static float   infloat[NFEAT];

static void make_input(void) {
	for (int f = 0; f < NFEAT; f++) {
		uint32_t lvl = xr() % (FBITS + 1u);        /* 0..8 ones, monotone */
		inlevel[f] = (uint8_t)lvl;
		infloat[f] = (float)lvl * 0.125f - 0.5f;
		for (uint32_t b = 0; b < FBITS; b++)
			inbits[f * FBITS + b] = (b < lvl) ? 1u : 0u;
	}
}

volatile uint32_t sink;   /* keeps the optimiser from deleting the workload */

/* ---------------------------------------------------------------- WNN ---- */
#if defined(BENCH_WNN) || defined(BENCH_WNN_ADDR) || defined(BENCH_WNN_FAST)
#include "wnn_model.h"

/* Membership in a sorted ON-set: the deployed representation. `end` is kept
 * separately because the loop clobbers hi, and the final bounds check must be
 * against the neuron's real upper bound, not the collapsed one. */
static inline int key_present(uint32_t addr, uint32_t lo, uint32_t end) {
	uint32_t hi = end;
	while (lo < hi) {
		uint32_t m = (lo + hi) >> 1;
		if (wnn_keys[m] < addr) lo = m + 1; else hi = m;
	}
	return (lo < end) && (wnn_keys[lo] == addr);
}

#ifdef BENCH_WNN_FAST
/* LEVEL-INDEXED GATHER, done properly.
 *
 * Each feature is an 8-bit thermometer, so its whole contribution to a neuron's
 * address is fixed by its LEVEL (0..8), not by eight independent bits. Precompute
 * per (neuron, feature, level) the OR-mask that feature contributes.
 *
 * Two details decide whether this actually wins, and the naive version lost
 * without them (26.7k instructions, WORSE than the 18.6k bit-gather):
 *   - stride 16, not 9: a [n][f][L] array needs two runtime multiplies per access
 *     on an M4. A power-of-two stride turns both into shifts. Costs 61 KB instead
 *     of 35 KB — space for speed.
 *   - hoist the index: inlevel[f] is the SAME for all 64 neurons, so f*16+level is
 *     computed once per step (15 ops) rather than 960 times. The per-neuron inner
 *     loop then degenerates to an indexed load and an OR.
 */
#define LSTRIDE 16
static uint32_t lvltab[WNN_NEURONS][NFEAT * LSTRIDE];
static uint16_t lidx[NFEAT];

static void build_lvltab(void) {
	for (int n = 0; n < WNN_NEURONS; n++)
		for (int f = 0; f < NFEAT; f++)
			for (int L = 0; L <= FBITS; L++) {
				uint32_t m = 0;
				for (int b = 0; b < WNN_BITS; b++) {
					int c = wnn_conn[n * WNN_BITS + b];
					if (c / FBITS != f) continue;
					if (L > (c % FBITS)) m |= 1u << (WNN_BITS - 1 - b);
				}
				lvltab[n][f * LSTRIDE + L] = m;
			}
}
static uint32_t wnn_step_fast(void) {
	uint8_t fired[WNN_NEURONS];
	for (int f = 0; f < NFEAT; f++)                 /* hoisted: once per step */
		lidx[f] = (uint16_t)(f * LSTRIDE + inlevel[f]);
	for (int n = 0; n < WNN_NEURONS; n++) {
		const uint32_t *t = lvltab[n];
		uint32_t addr = 0;
		for (int f = 0; f < NFEAT; f++) addr |= t[lidx[f]];
		fired[n] = (uint8_t)key_present(addr, wnn_off[n], wnn_off[n + 1]);
	}
	uint32_t acc = 0;
	for (int m = 0; m < WNN_MOTORS; m++) {
		uint32_t lvl = 0;
		for (int l = 0; l < WNN_LEVELS; l++) lvl += fired[m * WNN_LEVELS + l];
		acc += lvl << (8 * m);
	}
	return acc;
}
#endif

static uint32_t wnn_step(void) {
	uint8_t fired[WNN_NEURONS];
	for (int n = 0; n < WNN_NEURONS; n++) {
		const uint8_t *c = &wnn_conn[n * WNN_BITS];
		uint32_t addr = 0;
		for (int b = 0; b < WNN_BITS; b++)      /* MSB-first, project convention */
			addr = (addr << 1) | inbits[c[b]];
#ifdef BENCH_WNN_ADDR
		fired[n] = (uint8_t)(addr & 1u);          /* skip the search */
#else
		fired[n] = (uint8_t)key_present(addr, wnn_off[n], wnn_off[n + 1]);
#endif
	}
	uint32_t acc = 0;
	for (int m = 0; m < WNN_MOTORS; m++) {      /* thermometer count -> level */
		uint32_t lvl = 0;
		for (int l = 0; l < WNN_LEVELS; l++) lvl += fired[m * WNN_LEVELS + l];
		acc += lvl << (8 * m);
	}
	return acc;
}
#ifdef BENCH_WNN_FAST
#define STEP() wnn_step_fast()
#define NAME "WNN-fast(level-tab)"
#else
#define STEP() wnn_step()
#endif
#ifdef BENCH_WNN_ADDR
#define NAME "WNN-addr-only"
#else
#define NAME "WNN"
#endif
#endif

/* ---------------------------------------------------------------- PID ---- */
#ifdef BENCH_PID
/* Bitcraze firmware shape: cascade of 3 attitude PIDs -> 3 rate PIDs, with a
 * 2-pole low-pass on each rate derivative (src/wnn/control/pid_firmware.py).
 * Gains are placeholders — cost depends on the STRUCTURE, not the constants. */
typedef struct { float kp, ki, kd, integ, prev, i_lim; } pid_t;
typedef struct { float a1, a2, b0, b1, b2, d1, d2; } lpf2_t;

static pid_t att[3], rate[3];
static lpf2_t lpf[3];

static inline float lpf2(lpf2_t *f, float x) {
	float d0 = x - f->d1 * f->a1 - f->d2 * f->a2;
	float y  = d0 * f->b0 + f->d1 * f->b1 + f->d2 * f->b2;
	f->d2 = f->d1; f->d1 = d0;
	return y;
}
static inline float pid_up(pid_t *p, lpf2_t *f, float err, float dt) {
	p->integ += err * dt;
	if (p->integ >  p->i_lim) p->integ =  p->i_lim;
	if (p->integ < -p->i_lim) p->integ = -p->i_lim;
	float d = (err - p->prev) / dt;
	if (f) d = lpf2(f, d);
	p->prev = err;
	return p->kp * err + p->ki * p->integ + p->kd * d;
}
static uint32_t pid_step(void) {
	const float dt = 0.001f;
	float rate_sp[3], u[3];
	for (int i = 0; i < 3; i++)                      /* outer: attitude */
		rate_sp[i] = pid_up(&att[i], 0, infloat[i] - infloat[i + 3], dt);
	for (int i = 0; i < 3; i++)                      /* inner: body rate */
		u[i] = pid_up(&rate[i], &lpf[i], rate_sp[i] - infloat[i + 6], dt);
	float thr = infloat[9];
	float m0 = thr - u[0] + u[1] + u[2], m1 = thr - u[0] - u[1] - u[2];
	float m2 = thr + u[0] - u[1] + u[2], m3 = thr + u[0] + u[1] - u[2];
	return (uint32_t)(m0 + m1 + m2 + m3);
}
#define STEP() pid_step()
#define NAME "PID"
#endif

/* ---------------------------------------------------------------- MLP ---- */
#ifdef BENCH_MLP
/* REPRESENTATIVE MLP, 15-64-64-4 tanh — NOT the deleted run_mlp_ga.py baseline
 * (that script no longer exists, so its exact shape is unrecoverable). Cost is
 * dominated by MACs, so the per-MAC figure derived from this generalises to any
 * layer sizing; the absolute row must be read as "an MLP of this size".        */
#define H1 64
#define H2 64
#define NOUT 4
static float w1[NFEAT * H1], b1[H1], w2[H1 * H2], b2[H2], w3[H2 * NOUT], b3[NOUT];
static float h1[H1], h2[H2], o[NOUT];

static inline float act(float x) {          /* cheap tanh surrogate; a real tanh
                                             * would only make the MLP look worse */
	return x < -1.0f ? -1.0f : (x > 1.0f ? 1.0f : x);
}
static uint32_t mlp_step(void) {
	for (int j = 0; j < H1; j++) {
		float s = b1[j];
		for (int i = 0; i < NFEAT; i++) s += w1[i * H1 + j] * infloat[i];
		h1[j] = act(s);
	}
	for (int j = 0; j < H2; j++) {
		float s = b2[j];
		for (int i = 0; i < H1; i++) s += w2[i * H2 + j] * h1[i];
		h2[j] = act(s);
	}
	for (int j = 0; j < NOUT; j++) {
		float s = b3[j];
		for (int i = 0; i < H2; i++) s += w3[i * NOUT + j] * h2[i];
		o[j] = act(s);
	}
	return (uint32_t)(o[0] + o[1] + o[2] + o[3]);
}
static void mlp_init(void) {
	for (unsigned i = 0; i < sizeof(w1) / 4; i++) w1[i] = (float)(xr() & 255u) * 0.001f;
	for (unsigned i = 0; i < sizeof(w2) / 4; i++) w2[i] = (float)(xr() & 255u) * 0.001f;
	for (unsigned i = 0; i < sizeof(w3) / 4; i++) w3[i] = (float)(xr() & 255u) * 0.001f;
}
#define STEP() mlp_step()
#define NAME "MLP-15-64-64-4"
#endif

#ifdef BENCH_NONE
/* Input generation only: subtracting this row removes make_input() from all
 * three measurements, leaving the control step alone. */
#define STEP() 0u
#define NAME "NONE(input-only)"
#endif

int main(void) {
	fpu_enable();
#ifdef BENCH_WNN_FAST
	build_lvltab();
#endif
#ifdef BENCH_PID
	for (int i = 0; i < 3; i++) {
		att[i].kp = 6.0f;  att[i].ki = 3.0f;  att[i].kd = 0.0f;  att[i].i_lim = 20.0f;
		rate[i].kp = 250.0f; rate[i].ki = 500.0f; rate[i].kd = 2.5f; rate[i].i_lim = 33.3f;
		lpf[i].a1 = -1.6f; lpf[i].a2 = 0.67f;
		lpf[i].b0 = 0.02f; lpf[i].b1 = 0.04f; lpf[i].b2 = 0.02f;
	}
#endif
#ifdef BENCH_MLP
	mlp_init();
#endif
	uint32_t acc = 0;
	for (int it = 0; it < ITERS; it++) {
		make_input();
		acc += STEP();
	}
	sink = acc;
	put_str(NAME); put_str("\titers\t"); put_u32((uint32_t)ITERS);
	put_str("\tsink\t"); put_u32(acc); put_str("\n");
	sh_exit();
	return 0;
}
