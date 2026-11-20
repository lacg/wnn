/* One lookup translation unit of the pef64 host exactness check
 * (test_pef64_exactness.py). Compiled TWICE, against the same winner's uint32
 * header and its pef64 header, each exposing LOOKUP_FN(n, addr) and NEURONS_FN();
 * the model tables are `static`, so the two copies never collide at link time.
 *
 * The uint32 branch is the MCU's binary search verbatim (wnn_cascade.c
 * key_present); the pef64 branch is mcu/wnn_pef.h, called as the MCU calls it. */
#include <stdint.h>
#include WNN_MODEL_HEADER

int LOOKUP_FN(uint32_t n, uint32_t addr);
uint32_t NEURONS_FN(void);

#ifdef WNN_KEY_FORMAT_PEF64
#include "../wnn_pef.h"
static const wnn_pef_t model = WNN_PEF_MODEL;

int LOOKUP_FN(uint32_t n, uint32_t addr)
{
	return wnn_pef_contains(&model, n, addr);
}
#else
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

int LOOKUP_FN(uint32_t n, uint32_t addr)
{
	return key_present(addr, wnn_off[n], wnn_off[n + 1]);
}
#endif

uint32_t NEURONS_FN(void)
{
	return WNN_NEURONS;
}
