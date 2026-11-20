/* Lookup cost on REAL addresses: uint32 binary search vs pef64 (wnn_pef.h).
 *
 * bench_inc.c feeds synthetic thermometer walks, which miss almost everywhere
 * and never land beside a stored key. This bench replays a winner's recorded VAL
 * addresses (mcu/pef_replay_queries.py -> a query header: Q_TICKS decision ticks
 * x WNN_NEURONS addresses) and brackets regions for the insnmark.c plugin, so
 * every region's instruction count is reported — a worst case, not only a mean.
 *
 * Modes (one per build):
 *   -DREPLAY_LOOKUP  one region per lookup         -> per-lookup distribution
 *   -DREPLAY_FULL    one region per decision tick, every neuron looked up
 *   -DREPLAY_INC     one region per decision tick, only the neurons whose address
 *                    changed since the previous decision (incremental addressing;
 *                    the change test is inside the region)
 * The first region of every run is EMPTY (marker overhead, subtract it).
 * The address gather is NOT in any region: this prices the key store alone.
 *
 * Model: -DWNN_MODEL_HEADER='"...h"' (uint32 or pef64 header, same winner);
 * queries: -DQ_HEADER='"...h"'. Runner: mcu/run_bench_pef.sh. */
#include "bench.h"
#include WNN_MODEL_HEADER
#include Q_HEADER

#define MARK_BEGIN() __asm__ volatile("mov r11, r11" ::: "memory")
#define MARK_END() __asm__ volatile("mov r10, r10" ::: "memory")

#ifdef WNN_KEY_FORMAT_PEF64
#include "wnn_pef.h"
static const wnn_pef_t wnn_pef_model = WNN_PEF_MODEL;
#define NEURON_HAS(n, a) wnn_pef_contains(&wnn_pef_model, (uint32_t)(n), (a))
#define FORMAT "pef64"
#else
/* wnn_cascade.c key_present, verbatim. */
static int key_present(uint32_t addr, uint32_t lo, uint32_t end)
{
	uint32_t hi = end;
	while (lo < hi)
	{
		uint32_t m = (lo + hi) >> 1;
		if (wnn_keys[m] < addr) lo = m + 1; else hi = m;
	}
	return (lo < end) && (wnn_keys[lo] == addr);
}
#define NEURON_HAS(n, a) key_present((a), wnn_off[(n)], wnn_off[(n) + 1])
#define FORMAT "uint32"
#endif

volatile uint32_t sink;   /* keeps the optimiser from deleting the workload */
static uint8_t fired[WNN_NEURONS];

static uint32_t lookup_regions(void)
{
	uint32_t hits = 0;
	for (uint32_t t = 0; t < Q_TICKS; t++)
		for (uint32_t n = 0; n < WNN_NEURONS; n++)
		{
			uint32_t a = q_addr[t * WNN_NEURONS + n];
			int r;
			MARK_BEGIN();
			r = NEURON_HAS(n, a);
			MARK_END();
			hits += (uint32_t)r;
		}
	return hits;
}

static uint32_t full_regions(void)
{
	uint32_t hits = 0;
	for (uint32_t t = 0; t < Q_TICKS; t++)
	{
		const uint32_t *a = &q_addr[t * WNN_NEURONS];
		MARK_BEGIN();
		for (uint32_t n = 0; n < WNN_NEURONS; n++) fired[n] = (uint8_t)NEURON_HAS(n, a[n]);
		MARK_END();
		for (uint32_t n = 0; n < WNN_NEURONS; n++) hits += fired[n];
	}
	return hits;
}

static uint32_t inc_regions(void)
{
	uint32_t hits = 0;
	for (uint32_t n = 0; n < WNN_NEURONS; n++) fired[n] = (uint8_t)NEURON_HAS(n, q_addr[n]);
	for (uint32_t t = 1; t < Q_TICKS; t++)
	{
		const uint32_t *a = &q_addr[t * WNN_NEURONS], *p = a - WNN_NEURONS;
		MARK_BEGIN();
		for (uint32_t n = 0; n < WNN_NEURONS; n++)
			if (a[n] != p[n]) fired[n] = (uint8_t)NEURON_HAS(n, a[n]);
		MARK_END();
		for (uint32_t n = 0; n < WNN_NEURONS; n++) hits += fired[n];
	}
	return hits;
}

int main(void)
{
	uint32_t hits;
	fpu_enable();
	MARK_BEGIN();
	MARK_END();
#if defined(REPLAY_LOOKUP)
	hits = lookup_regions();
#elif defined(REPLAY_FULL)
	hits = full_regions();
#elif defined(REPLAY_INC)
	hits = inc_regions();
#else
#error "pick -DREPLAY_LOOKUP, -DREPLAY_FULL or -DREPLAY_INC"
#endif
	sink = hits;
	put_str("REPLAY\t" FORMAT "\tticks\t"); put_u32(Q_TICKS);
	put_str("\tneurons\t"); put_u32(WNN_NEURONS);
	put_str("\thits\t"); put_u32(hits); put_str("\n");
	sh_exit();
	return 0;
}
