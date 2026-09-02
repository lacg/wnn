/* Control-step cost with the two optimisations a 1 kHz loop actually permits.
 *
 * WHY THE FIRST BENCHMARK COULD NOT SEE THESE. bench.c draws fresh RANDOM feature
 * levels every step. A real attitude loop at 1 kHz has strong temporal coherence:
 * levels move by at most a step or two per millisecond. Random inputs destroy that
 * before it can be exploited, so the naive full re-gather looked like the floor.
 * It is not.
 *
 *   OPT 1  INCREMENTAL ADDRESSING. Keep a running 30-bit address per neuron. A
 *          feature level moving L -> L+1 flips EXACTLY one input bit (index L of
 *          that feature's thermometer). An inverted index maps each of the 120
 *          input bits to the (neuron, address-bit-mask) pairs that read it — about
 *          1920/120 ~ 16 neurons per bit — so one flipped bit costs ~16 XORs
 *          instead of re-gathering 1,920 bits.
 *
 *   OPT 2  O(1) LOOKUP, ONLY WHERE DIRTY. Binary search costs ~13 iterations per
 *          neuron; an open-addressed hash over the ON-set costs ~1 probe. And a
 *          neuron whose address did not change cannot change output, so it is not
 *          looked up at all.
 *
 * The realistic input model is a bounded random walk on each feature level, which
 * is what sampling a smooth attitude signal into a thermometer produces. STEP_JITTER
 * controls how many features move per step so the cost can be reported as a function
 * of input activity rather than a single number.
 *
 * Build: -DBENCH_BASE (full gather + binsearch) | -DBENCH_INC (opt1+opt2)
 *
 * TWO KNOBS ADDED 02/09/2026, both OFF by default so the banked 820 instr/step
 * number reproduces byte-identically:
 *
 *   -DPROBE_STATS            Count, per control step, how many neurons go DIRTY
 *                            and how many key-array reads the lookups perform.
 *                            Counting costs instructions, so it MUST NOT be on
 *                            when measuring instr/step — the two builds answer
 *                            different questions and are never the same run.
 *
 *   -DINC_LOOKUP_BINSEARCH   Use the sorted-array binary search instead of the
 *                            open-addressed hash. This is the DEPLOYABLE path
 *                            for a large model: the hash needs 8 bytes/slot at
 *                            >=2x the key count, which at n=256 (894,552 keys)
 *                            is a 16 MB table — off the H743 and off this
 *                            board's 4 MB. Binary search reads the key array in
 *                            place, so the model can live in external
 *                            memory-mapped QSPI NOR and only the probes cost.
 *
 * WHY PROBES ARE THE NUMBER THAT MATTERS FOR EXTERNAL MEMORY. Instructions are
 * ~free at 480 MHz (1 ms = 480,000 cycles). What is not free is a random read
 * over QSPI, which pays a command+address+dummy phase before any data moves.
 * The cost of putting the model off-chip is therefore (probes/step) x (random
 * read latency), and probes/step is what this build measures.
 */
#include "bench.h"
/* Model header is selectable so a second model can be benchmarked without
 * overwriting the first. A quoted include resolves next to THIS file before any
 * -I path, so -I alone silently keeps mcu/wnn_model.h — pass
 * -DWNN_MODEL_HEADER='"wnn_model_b32n256.h"' to switch. */
#ifndef WNN_MODEL_HEADER
#define WNN_MODEL_HEADER "wnn_model.h"
#endif
#include WNN_MODEL_HEADER

#ifndef ITERS
#define ITERS 0
#endif
#ifndef STEP_JITTER
#define STEP_JITTER 2      /* features whose level moves per control step */
#endif

/* Feature geometry. Taken from the model header when it carries it (exporter
 * emits WNN_FEATURES / WNN_BITS_PER_FEATURE since 02/09/2026); the 15x8 literals
 * are the pre-existing default so an older header benchmarks exactly as before. */
#ifdef WNN_FEATURES
#define NFEAT WNN_FEATURES
#else
#define NFEAT 15
#endif
#ifdef WNN_BITS_PER_FEATURE
#define FBITS WNN_BITS_PER_FEATURE
#else
#define FBITS 8
#endif

/* Neuron index width. 0..255 fits a byte, so n=256 needs no change and the
 * default build stays byte-identical; anything wider promotes automatically
 * rather than wrapping silently. */
#if WNN_NEURONS <= 256
typedef uint8_t nidx_t;
#else
typedef uint16_t nidx_t;
#endif

static uint32_t rng_s = 0x12345678u;
static inline uint32_t xr(void) {
	rng_s ^= rng_s << 13; rng_s ^= rng_s >> 17; rng_s ^= rng_s << 5;
	return rng_s;
}

/* ---- probe accounting (compiled out unless -DPROBE_STATS) ---------------- */
#ifdef PROBE_STATS
#define PB_BINS 32
static uint32_t pb_steps, pb_dirty_tot, pb_dirty_max;
static uint32_t pb_mem_tot, pb_mem_max;            /* key-array reads */
static uint32_t pb_hist[PB_BINS + 1];              /* dirty-per-step histogram */
static uint32_t pb_mem_step;                       /* accumulator, current step */
#define PROBE_MEM() (pb_mem_step++)
static void probe_step_end(uint32_t nd) {
	pb_steps++;
	pb_dirty_tot += nd;
	if (nd > pb_dirty_max) pb_dirty_max = nd;
	pb_hist[nd < PB_BINS ? nd : PB_BINS]++;
	pb_mem_tot += pb_mem_step;
	if (pb_mem_step > pb_mem_max) pb_mem_max = pb_mem_step;
	pb_mem_step = 0;
}
/* NO DIVISION HERE. -nostdlib drops libgcc, so a 64-bit divide would fail to
 * link on __aeabi_uldivmod (the same trap the README records for soft-float).
 * Raw totals are emitted and the runner divides — it also keeps the counters
 * exact rather than pre-rounded. */
#else
#define PROBE_MEM() ((void)0)
#endif

static uint8_t inbits[NFEAT * FBITS];
static uint8_t level[NFEAT], prev_level[NFEAT];
volatile uint32_t sink;

static void input_init(void) {
	for (int f = 0; f < NFEAT; f++) {
		level[f] = (uint8_t)(xr() % (FBITS + 1u));
		prev_level[f] = level[f];
		for (int b = 0; b < FBITS; b++) inbits[f * FBITS + b] = (b < level[f]);
	}
}
/* Bounded random walk: STEP_JITTER features move by +-1, clamped to 0..8. */
static void input_step(void) {
	for (int k = 0; k < STEP_JITTER; k++) {
		int f = (int)(xr() % NFEAT);
		int d = (xr() & 1u) ? 1 : -1;
		int nl = (int)level[f] + d;
		if (nl < 0) nl = 0;
		if (nl > FBITS) nl = FBITS;
		level[f] = (uint8_t)nl;
	}
	for (int f = 0; f < NFEAT; f++)
		for (int b = 0; b < FBITS; b++) inbits[f * FBITS + b] = (b < level[f]);
}

/* ---- sorted-ON-set membership (the b=30 deployed representation) ---------- */
static inline int key_present(uint32_t addr, uint32_t lo, uint32_t end) {
	uint32_t hi = end;
	while (lo < hi) {
		uint32_t m = (lo + hi) >> 1;
		PROBE_MEM();                     /* one key-array read */
		if (wnn_keys[m] < addr) lo = m + 1; else hi = m;
	}
	if (lo < end) { PROBE_MEM(); return wnn_keys[lo] == addr; }
	return 0;
}

/* ---- OPT 2: open-addressed hash over (neuron, addr) ----------------------- */
#define HBITS 18
#define HSIZE (1u << HBITS)
static uint64_t htab[HSIZE];              /* 0 = empty; key = (n<<32)|addr */
static inline uint32_t hmix(uint64_t k) {
	k *= 0x9E3779B97F4A7C15ull;
	return (uint32_t)(k >> (64 - HBITS));
}
static void hash_build(void) {
#ifdef INC_LOOKUP_BINSEARCH
	/* Not the lookup in use — and at n=256 the table would be 16 MB, larger than
	 * this board's whole RAM. Building it would fail before a probe is counted. */
	return;
#else
	for (uint32_t i = 0; i < HSIZE; i++) htab[i] = 0;
	for (int n = 0; n < WNN_NEURONS; n++)
		for (uint32_t i = wnn_off[n]; i < wnn_off[n + 1]; i++) {
			uint64_t key = ((uint64_t)(n + 1) << 32) | wnn_keys[i];
			uint32_t h = hmix(key);
			while (htab[h]) h = (h + 1u) & (HSIZE - 1u);
			htab[h] = key;
		}
#endif
}
static inline int hash_present(int n, uint32_t addr) {
	uint64_t key = ((uint64_t)(n + 1) << 32) | addr;
	uint32_t h = hmix(key);
	for (;;) {
		uint64_t v = htab[h];
		PROBE_MEM();                     /* one table read */
		if (v == key) return 1;
		if (!v) return 0;
		h = (h + 1u) & (HSIZE - 1u);
	}
}

/* Which membership test the INC path uses. The hash is the default so the
 * banked 820 instr/step reproduces; binary search is the deployable path for a
 * model too large to hash in SRAM. */
#ifdef INC_LOOKUP_BINSEARCH
#define INC_PRESENT(n, a) key_present((a), wnn_off[(n)], wnn_off[(n) + 1])
#else
#define INC_PRESENT(n, a) hash_present((n), (a))
#endif

/* ---- OPT 1: inverted index, input bit -> (neuron, address-bit mask) ------- */
static uint16_t inv_off[NFEAT * FBITS + 1];
static nidx_t   inv_neuron[WNN_NEURONS * WNN_BITS];
static uint32_t inv_mask[WNN_NEURONS * WNN_BITS];
static uint32_t addr_of[WNN_NEURONS];
static uint8_t  fired[WNN_NEURONS];

static void inv_build(void) {
	uint16_t cnt[NFEAT * FBITS + 1];
	for (int i = 0; i <= NFEAT * FBITS; i++) cnt[i] = 0;
	for (int n = 0; n < WNN_NEURONS; n++)
		for (int b = 0; b < WNN_BITS; b++) cnt[wnn_conn[n * WNN_BITS + b] + 1]++;
	inv_off[0] = 0;
	for (int i = 0; i < NFEAT * FBITS; i++) inv_off[i + 1] = inv_off[i] + cnt[i + 1];
	uint16_t cur[NFEAT * FBITS];
	for (int i = 0; i < NFEAT * FBITS; i++) cur[i] = inv_off[i];
	for (int n = 0; n < WNN_NEURONS; n++)
		for (int b = 0; b < WNN_BITS; b++) {
			int c = wnn_conn[n * WNN_BITS + b];
			uint16_t s = cur[c]++;
			inv_neuron[s] = (nidx_t)n;
			inv_mask[s] = 1u << (WNN_BITS - 1 - b);
		}
}
/* Full rebuild — used once to prime the running addresses. */
static void addr_prime(void) {
	for (int n = 0; n < WNN_NEURONS; n++) {
		uint32_t a = 0;
		for (int b = 0; b < WNN_BITS; b++)
			a = (a << 1) | inbits[wnn_conn[n * WNN_BITS + b]];
		addr_of[n] = a;
		fired[n] = (uint8_t)INC_PRESENT(n, a);
	}
}

static uint32_t decode(void) {
	uint32_t acc = 0;
	for (int m = 0; m < WNN_MOTORS; m++) {
		uint32_t lvl = 0;
		for (int l = 0; l < WNN_LEVELS; l++) lvl += fired[m * WNN_LEVELS + l];
		acc += lvl << (8 * m);
	}
	return acc;
}

/* ---- the two steps under test -------------------------------------------- */
#ifdef BENCH_NONE
/* Input synthesis only. The INC path never reads inbits[] (it works off levels),
 * so rebuilding 120 thermometer bits each step is pure harness cost and must be
 * subtracted or it dominates the optimised number. */
static uint32_t step(void) { return 0u; }
#define NAME "NONE input-only"
#endif

#ifdef BENCH_BASE
static uint32_t step(void) {
	for (int n = 0; n < WNN_NEURONS; n++) {
		const uint8_t *c = &wnn_conn[n * WNN_BITS];
		uint32_t a = 0;
		for (int b = 0; b < WNN_BITS; b++) a = (a << 1) | inbits[c[b]];
		fired[n] = (uint8_t)key_present(a, wnn_off[n], wnn_off[n + 1]);
	}
	return decode();
}
#define NAME "BASE full-gather+binsearch"
#endif

#ifdef BENCH_INC
static int32_t mlev[WNN_MOTORS];
static uint8_t dirty[WNN_NEURONS];
static nidx_t  dlist[WNN_NEURONS];
static uint32_t step(void) {
	uint32_t nd = 0;
	for (int f = 0; f < NFEAT; f++) {
		int L0 = prev_level[f], L1 = level[f];
		if (L0 == L1) continue;
		int lo = L0 < L1 ? L0 : L1, hi = L0 < L1 ? L1 : L0;
		for (int L = lo; L < hi; L++) {          /* one bit per level crossed */
			int bit = f * FBITS + L;
			for (uint16_t s = inv_off[bit]; s < inv_off[bit + 1]; s++) {
				nidx_t n = inv_neuron[s];
				addr_of[n] ^= inv_mask[s];
				if (!dirty[n]) { dirty[n] = 1; dlist[nd++] = n; }
			}
		}
		prev_level[f] = (uint8_t)L1;
	}
	for (uint32_t i = 0; i < nd; i++) {          /* re-lookup ONLY the dirty */
		nidx_t n = dlist[i];
		dirty[n] = 0;
		uint8_t f = (uint8_t)INC_PRESENT(n, addr_of[n]);
		/* Incremental decode: a motor's thermometer count can only change when
		 * one of its neurons flips, so track the delta instead of re-summing 64. */
		if (f != fired[n]) {
			mlev[n / WNN_LEVELS] += f ? 1 : -1;
			fired[n] = f;
		}
	}
#ifdef PROBE_STATS
	probe_step_end(nd);
#endif
	return (uint32_t)(mlev[0] + (mlev[1] << 8) + (mlev[2] << 16) + (mlev[3] << 24));
}
#define NAME "INC incremental+hash"
#endif

int main(void) {
	fpu_enable();
	input_init();
	hash_build();
	inv_build();
	addr_prime();
#ifdef BENCH_INC
	for (int n = 0; n < WNN_NEURONS; n++) if (fired[n]) mlev[n / WNN_LEVELS]++;
#endif
	uint32_t acc = 0;
	for (int it = 0; it < ITERS; it++) {
		input_step();
		acc += step();
	}
	sink = acc;
#ifdef BENCH_INC
	/* EQUIVALENCE CHECK. An optimisation that computes something else is worth
	 * nothing, and synthetic inputs make every lookup miss, so `sink` cannot
	 * distinguish the two paths. Verify directly at the end of the run:
	 *   OPT 1 — every running address must equal a full re-gather.
	 *   OPT 2 — hash membership must agree with binary search on every neuron.
	 *
	 * The OPT 2 half is SKIPPED under -DINC_LOOKUP_BINSEARCH: that build never
	 * populates the hash (it would be 16 MB at n=256), so comparing against it
	 * would report every hit as a mismatch — a false alarm, not a finding. The
	 * binsearch build IS the reference, so there is nothing to cross-check. */
	{
		uint32_t bad_addr = 0, bad_look = 0;
		for (int n = 0; n < WNN_NEURONS; n++) {
			uint32_t a = 0;
			for (int b = 0; b < WNN_BITS; b++)
				a = (a << 1) | inbits[wnn_conn[n * WNN_BITS + b]];
			if (a != addr_of[n]) bad_addr++;
#ifndef INC_LOOKUP_BINSEARCH
			if (hash_present(n, a) != key_present(a, wnn_off[n], wnn_off[n + 1]))
				bad_look++;
#endif
		}
		put_str("CHECK\taddr_mismatch\t"); put_u32(bad_addr);
		put_str("\tlookup_mismatch\t"); put_u32(bad_look); put_str("\n");
	}
#endif
#ifdef PROBE_STATS
	/* One tab-separated block, raw counts plus x100 means (no FPU). The consumer
	 * is run_bench_probes.sh; the fields are deliberately self-describing so a
	 * stray copy of the output is still readable. */
	put_str("PROBES\tneurons\t"); put_u32(WNN_NEURONS);
	put_str("\tbits\t"); put_u32(WNN_BITS);
	put_str("\tkeys\t"); put_u32(WNN_NUM_KEYS);
	put_str("\tjitter\t"); put_u32(STEP_JITTER);
	put_str("\tsteps\t"); put_u32(pb_steps);
	put_str("\tdirty_tot\t"); put_u32(pb_dirty_tot);
	put_str("\tdirty_max\t"); put_u32(pb_dirty_max);
	put_str("\tmemreads_tot\t"); put_u32(pb_mem_tot);
	put_str("\tmemreads_max\t"); put_u32(pb_mem_max);
	put_str("\n");
	put_str("PBHIST");
	for (int i = 0; i <= PB_BINS; i++) { put_str("\t"); put_u32(pb_hist[i]); }
	put_str("\n");
#endif
	put_str(NAME); put_str("\tjitter\t"); put_u32(STEP_JITTER);
	put_str("\titers\t"); put_u32((uint32_t)ITERS);
	put_str("\tsink\t"); put_u32(acc); put_str("\n");
	sh_exit();
	return 0;
}
