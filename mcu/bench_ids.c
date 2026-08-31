/* Cortex-M4 cost of ONE IDS classification, measured the same way the controller
 * step was (mcu/bench.c): build at ITERS=0 and ITERS=N, count retired instructions
 * with the TCG plugin, per-decision = (I_N - I_0)/N, then subtract the NONE row so
 * input synthesis cancels exactly.
 *
 * WHY A SEPARATE BENCHMARK EXISTS, and it is not just a bigger model.
 *
 * The controller's 820 instructions/step (mcu/bench_inc.c) is bought almost entirely
 * by INCREMENTAL ADDRESSING, and that optimisation is only available because a 1 kHz
 * attitude loop has strong TEMPORAL COHERENCE: a thermometer level moves a step or two
 * per millisecond, so one input bit flips and the running address is patched with ~16
 * XORs instead of re-gathered. A neuron whose address did not change cannot change
 * output, so it is never even looked up.
 *
 * CONSECUTIVE NETWORK FLOWS ARE UNRELATED. There is no coherence to exploit, so an
 * IDS decision pays the FULL gather and a lookup for EVERY neuron -- the BASE path,
 * 20,245 instructions in the controller's own table, not 820. Reusing bench_inc.c's
 * input model here would fabricate a coherence that the workload does not have and
 * would understate the cost by more than an order of magnitude. make_input() below
 * therefore draws a FRESH INDEPENDENT flow every iteration, on purpose.
 *
 * That is the same lesson the controller harness learned in the opposite direction:
 * benchmark inputs must model the workload's temporal statistics, or the dominant
 * term is invisible. Here the honest model is "no coherence at all".
 *
 * Build one of: -DBENCH_NONE | -DBENCH_IDS_ADDR | -DBENCH_IDS
 */
#include "bench.h"
#include "ids_model.h"

#ifndef ITERS
#define ITERS 0
#endif

static uint32_t rng_s = 0x12345678u;
static inline uint32_t xr(void) {
	rng_s ^= rng_s << 13; rng_s ^= rng_s >> 17; rng_s ^= rng_s << 5;
	return rng_s;
}

static uint8_t inbits[IDS_INPUT_BITS];

/* One flow, thermometer-encoded: each feature contributes a monotone run of ones.
 * Independent of the previous flow -- see the header comment. */
static void make_input(void) {
	for (int f = 0; f < IDS_FEATURES; f++) {
		uint32_t lvl = xr() % (IDS_LEVELS + 1u);
		uint8_t *p = &inbits[f * IDS_LEVELS];
		for (uint32_t b = 0; b < IDS_LEVELS; b++) p[b] = (b < lvl) ? 1u : 0u;
	}
}

volatile uint32_t sink;

#if IDS_WIDE
/* bits > 64: the tuple no longer fits one word, so ram_core names it with a
 * splitmix64 hash of the two halves rather than OR-folding slot i onto i+64
 * (the 29/08 address fix). Two gathers plus the mix, then the same u64 search. */
static inline uint64_t splitmix64(uint64_t z) {
	z += 0x9E3779B97F4A7C15ull;
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
	return z ^ (z >> 31);
}
#endif

/* Membership in a sorted ON-set. Runs to convergence with NO early exit, so a hit
 * and a miss cost the same -- the bounded-WCET property, and what makes synthetic
 * keys legitimate for a cost measurement. */
static inline int key_present(IDS_KEY_T addr, uint32_t lo, uint32_t end) {
	uint32_t hi = end;
	while (lo < hi) {
		uint32_t m = (lo + hi) >> 1;
		if (ids_keys[m] < addr) lo = m + 1; else hi = m;
	}
	return (lo < end) && (ids_keys[lo] == addr);
}

/* One classification: gather every neuron's address, look each up, count the
 * neurons that fired. The response count is what the threshold modes consume. */
static uint32_t classify(int do_search) {
	uint32_t fired = 0;
	for (int n = 0; n < IDS_NEURONS; n++) {
		const uint16_t *c = &ids_conn[n * IDS_BITS];
#if IDS_WIDE
		uint64_t w0 = 0, w1 = 0;
		for (int b = 0; b < 64; b++)        w0 = (w0 << 1) | inbits[c[b]];
		for (int b = 64; b < IDS_BITS; b++) w1 = (w1 << 1) | inbits[c[b]];
		IDS_KEY_T addr = splitmix64(w0 ^ splitmix64(w1));
#else
		IDS_KEY_T addr = 0;
		for (int b = 0; b < IDS_BITS; b++) addr = (addr << 1) | inbits[c[b]];
#endif
		if (do_search) fired += (uint32_t)key_present(addr, ids_off[n], ids_off[n + 1]);
		else           fired += (uint32_t)addr;      /* keep the gather alive */
	}
	return fired;
}

int main(void) {
	fpu_enable();
	uint32_t acc = 0;
	for (int i = 0; i < ITERS; i++) {
		make_input();
#if defined(BENCH_IDS)
		acc += classify(1);
#elif defined(BENCH_IDS_ADDR)
		acc += classify(0);
#else
		acc += inbits[0];                            /* BENCH_NONE: input only */
#endif
	}
	sink = acc;
	report("DONE", sink);
	sh_exit();
	return 0;
}
