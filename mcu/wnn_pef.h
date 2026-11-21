/* Partitioned Elias-Fano (PEF, B=64) TRUE-key store: the C decoder.
 *
 * Same question as the uint32 sorted array: "is address A in neuron n's TRUE
 * set?" — answered EXACTLY (lossless; mcu/parity/test_pef64_exactness.py checks
 * every stored key, every VAL-trace address and 1M random addresses per winner
 * against the uint32 binary search). Built by scripts/pef_encoder.py, emitted by
 * `scripts/export_controller_c.py --key-format pef64`; layout in scripts/pef_tables.py.
 *
 * Lookup: binary search on the neuron's block first-keys (the block index), then
 * Elias-Fano within ONE block of <= 64 keys: find the (x >> l)-th zero of the
 * unary high part, from whichever end is nearer, a 32-bit word at a time
 * (popcount by a 256-byte LUT — the Cortex-M7 has no popcount instruction;
 * CLZ/RBIT give the trailing-zero count); measure that bucket's run of ones,
 * then binary-search its ascending l-bit lows.
 *
 * C99, header-only, no malloc, no state beyond const tables. A model header with
 * pef64 tables defines WNN_KEY_FORMAT_PEF64; the call sites switch on it.
 *
 * WNN_PEF_PROBE(): optional hook, one call per table read (bench_inc.c points it
 * at PROBE_MEM to count reads for an external-memory model). Default: nothing.
 */
#ifndef WNN_PEF_H
#define WNN_PEF_H

#include <stdint.h>

#ifndef WNN_PEF_PROBE
#define WNN_PEF_PROBE() ((void)0)
#endif

#define WNN_PEF_META_L_BITS 6u

typedef struct
{
	uint32_t block;          /* keys per block (64) */
	const uint32_t *off;     /* [N+1] key-count prefix (the uint32 format's wnn_off) */
	const uint32_t *blk;     /* [N+1] block prefix per neuron */
	const uint32_t *first;   /* [NB] first key of each block */
	const uint32_t *meta;    /* [NB+1] (bit offset << 6) | l */
	const uint32_t *data;    /* EF bitstream, LSB-first, + 1 pad word */
} wnn_pef_t;

static const uint8_t wnn_pef_pop8[256] = {
#define WNN_POP2(n) n, n + 1, n + 1, n + 2
#define WNN_POP4(n) WNN_POP2(n), WNN_POP2(n + 1), WNN_POP2(n + 1), WNN_POP2(n + 2)
#define WNN_POP6(n) WNN_POP4(n), WNN_POP4(n + 1), WNN_POP4(n + 1), WNN_POP4(n + 2)
	WNN_POP6(0), WNN_POP6(1), WNN_POP6(1), WNN_POP6(2)
#undef WNN_POP6
#undef WNN_POP4
#undef WNN_POP2
};

/* `width` (0..32) bits starting at bit `pos`. Two word reads at most. */
static inline uint32_t wnn_pef_bits(const uint32_t *d, uint32_t pos, uint32_t width)
{
	uint32_t w = pos >> 5, s = pos & 31u;
	uint32_t v;
	if (width == 0u) return 0u;
	WNN_PEF_PROBE();
	v = d[w] >> s;
	if (s + width > 32u) { WNN_PEF_PROBE(); v |= d[w + 1] << (32u - s); }
	return (width == 32u) ? v : (v & ((1u << width) - 1u));
}

static inline uint32_t wnn_pef_pop32(uint32_t v)
{
	return (uint32_t)wnn_pef_pop8[v & 255u] + wnn_pef_pop8[(v >> 8) & 255u]
	     + wnn_pef_pop8[(v >> 16) & 255u] + wnn_pef_pop8[v >> 24];
}

/* Last block in [b0, b1) whose first key <= addr; b0 - 1 when none. */
static inline uint32_t wnn_pef_find_block(const wnn_pef_t *p, uint32_t b0, uint32_t b1, uint32_t addr)
{
	uint32_t lo = b0, hi = b1;
	while (lo < hi)
	{
		uint32_t m = (lo + hi) >> 1;
		WNN_PEF_PROBE();
		if (p->first[m] <= addr) lo = m + 1; else hi = m;
	}
	return lo - 1u;
}

/* Count of trailing zero bits of x (x != 0). ARMv7-M has CLZ+RBIT, no popcount. */
#if defined(__GNUC__) || defined(__clang__)
#define WNN_PEF_CTZ(x) ((uint32_t)__builtin_ctz(x))
#else
static inline uint32_t wnn_pef_ctz_c99(uint32_t x)
{
	uint32_t n = 0;
	while (!(x & 1u)) { x >>= 1; n++; }
	return n;
}
#define WNN_PEF_CTZ(x) wnn_pef_ctz_c99(x)
#endif

/* Low `width` bits set (width 1..32). */
static inline uint32_t wnn_pef_mask(uint32_t width)
{
	return (width >= 32u) ? 0xFFFFFFFFu : ((1u << width) - 1u);
}

/* Bit index of the k-th (0-based) zero of byte v (it has more than k zeros). */
static inline uint32_t wnn_pef_select0_8(uint32_t v, uint32_t k)
{
	uint32_t z = ~v & 0xFFu;
	while (k--) z &= z - 1u;
	return WNN_PEF_CTZ(z);
}

/* Bit index of the k-th (0-based) zero of word w (it has more than k zeros):
 * byte by byte with the LUT, then bit by bit inside the byte. */
static inline uint32_t wnn_pef_select0_32(uint32_t w, uint32_t k)
{
	uint32_t s = 0, z;
	for (;;)
	{
		z = 8u - wnn_pef_pop8[(w >> s) & 255u];
		if (z > k) break;
		k -= z; s += 8u;
	}
	return s + wnn_pef_select0_8((w >> s) & 255u, k);
}

/* Position just past the zero that has r zeros after it, scanning BACKWARD
 * from the end (the high part's first window pads with ONES above `take`). */
static inline uint32_t wnn_pef_skip_zeros_back(const uint32_t *d, uint32_t hi, uint32_t len, uint32_t r)
{
	uint32_t e = len, w, z, take;
	for (;;)
	{
		take = (e < 32u) ? e : 32u;
		w = wnn_pef_bits(d, hi + e - take, take) | ~wnn_pef_mask(take);
		z = 32u - wnn_pef_pop32(w);
		if (z > r) return e - take + wnn_pef_select0_32(w, z - 1u - r) + 1u;
		r -= z; e -= take;
	}
}

/* Position just past the h-th zero of the high part (0 when h == 0);
 * UINT32_MAX when it has fewer than h zeros. Word by word (popcount LUT); a
 * partial last word pads with ONES so bits past `len` never count as zeros. */
static inline uint32_t wnn_pef_skip_zeros(const uint32_t *d, uint32_t hi, uint32_t len, uint32_t h)
{
	uint32_t p = 0, w, z, take;
	if (h == 0u) return 0u;
	for (;;)
	{
		take = (len - p < 32u) ? len - p : 32u;
		if (take == 0u) return UINT32_MAX;
		w = wnn_pef_bits(d, hi + p, take) | ~wnn_pef_mask(take);
		z = 32u - wnn_pef_pop32(w);
		if (z >= h) return p + wnn_pef_select0_32(w, h - 1u) + 1u;
		h -= z; p += take;
	}
}

/* Length of the run of ones starting at high-part position p (= the bucket size). */
static inline uint32_t wnn_pef_run_ones(const uint32_t *d, uint32_t hi, uint32_t len, uint32_t p)
{
	uint32_t c = 0, take, zeros;
	while (p < len)
	{
		take = (len - p < 32u) ? len - p : 32u;
		zeros = ~wnn_pef_bits(d, hi + p, take) & wnn_pef_mask(take);
		if (zeros) return c + WNN_PEF_CTZ(zeros);
		c += take; p += take;
	}
	return c;
}

/* Is x_low among the c ascending l-bit lows starting at element i? (lower bound) */
static inline int wnn_pef_find_low(const uint32_t *d, uint32_t lo, uint32_t l, uint32_t i,
                                   uint32_t c, uint32_t x_low)
{
	uint32_t a = i, b = i + c;
	while (a < b)
	{
		uint32_t m = (a + b) >> 1;
		if (wnn_pef_bits(d, lo + m * l, l) < x_low) a = m + 1u; else b = m;
	}
	return (a < i + c) && (wnn_pef_bits(d, lo + a * l, l) == x_low);
}

/* Keys in block b of neuron n (B except the neuron's last block). */
static inline uint32_t wnn_pef_block_keys(const wnn_pef_t *p, uint32_t n, uint32_t b)
{
	if (b + 1u < p->blk[n + 1]) return p->block;
	return (p->off[n + 1] - p->off[n]) - (b - p->blk[n]) * p->block;
}

/* Is x = addr - first - 1 among block b's EF values? */
static inline int wnn_pef_in_block(const wnn_pef_t *p, uint32_t n, uint32_t b, uint32_t x)
{
	uint32_t m = wnn_pef_block_keys(p, n, b) - 1u;
	uint32_t meta = p->meta[b], l = meta & 63u, lo = meta >> WNN_PEF_META_L_BITS;
	uint32_t hi = lo + m * l;
	uint32_t len = (p->meta[b + 1] >> WNN_PEF_META_L_BITS) - hi;
	uint32_t h = (l < 32u) ? (x >> l) : 0u;
	uint32_t zeros = len - m, pos;
	if (h > zeros) return 0;
	pos = (2u * h > zeros) ? wnn_pef_skip_zeros_back(p->data, hi, len, zeros - h)
	                       : wnn_pef_skip_zeros(p->data, hi, len, h);
	if (pos == UINT32_MAX) return 0;
	return wnn_pef_find_low(p->data, lo, l, pos - h, wnn_pef_run_ones(p->data, hi, len, pos),
	                        (l < 32u) ? (x & ((1u << l) - 1u)) : x);
}

/* THE lookup: is addr in neuron n's TRUE set? */
static inline int wnn_pef_contains(const wnn_pef_t *p, uint32_t n, uint32_t addr)
{
	uint32_t b0 = p->blk[n], b1 = p->blk[n + 1], b;
	if (b0 == b1) return 0;
	b = wnn_pef_find_block(p, b0, b1, addr);
	if (b + 1u == b0) return 0;      /* addr below the neuron's smallest key (b0 >= 0: wraps safely) */
	WNN_PEF_PROBE();
	if (p->first[b] == addr) return 1;
	return wnn_pef_in_block(p, n, b, addr - p->first[b] - 1u);
}

/* The model header's tables as a decoder handle (needs WNN_KEY_FORMAT_PEF64). */
#define WNN_PEF_MODEL { WNN_PEF_BLOCK, wnn_off, wnn_pef_blk, wnn_pef_first, wnn_pef_meta, wnn_pef_data }

#endif /* WNN_PEF_H */
