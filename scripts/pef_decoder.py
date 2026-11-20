"""PefDecoder: pure-Python reference lookup over PefTables — a step-for-step
mirror of mcu/wnn_pef.h (same block search, same word/byte popcount-LUT select,
same bucket binary search), so the Python exactness test exercises the
algorithm the MCU runs."""

from bisect import bisect_right

from pef_tables import META_L_BITS, PefTables

POP8 = [bin(i).count("1") for i in range(256)]


def ctz(x: int) -> int:
	return (x & -x).bit_length() - 1


MASK32 = 0xFFFFFFFF


def select0_32(w: int, k: int) -> int:
	"""Bit index of the k-th (0-based) zero of word w: bytes by LUT, then bits."""
	s = 0
	while 8 - POP8[(w >> s) & 255] <= k:
		k -= 8 - POP8[(w >> s) & 255]
		s += 8
	return s + select0_8((w >> s) & 255, k)


def select0_8(v: int, k: int) -> int:
	"""Bit index of the k-th (0-based) zero of byte v."""
	z = ~v & 0xFF
	for _ in range(k):
		z &= z - 1
	return ctz(z)


class PefDecoder:
	def __init__(self, t: PefTables):
		self.t = t

	def read_bits(self, pos: int, width: int) -> int:
		if width == 0:
			return 0
		d, w, s = self.t.data, pos >> 5, pos & 31
		v = d[w] >> s
		if s + width > 32:
			v |= d[w + 1] << (32 - s)
		return v & ((1 << width) - 1)

	def contains(self, n: int, addr: int) -> bool:
		t = self.t
		b0, b1 = t.blk[n], t.blk[n + 1]
		b = bisect_right(t.first, addr, b0, b1) - 1   # last block with first <= addr
		if b < b0:
			return False
		if t.first[b] == addr:
			return True
		return self.in_block(n, b, addr - t.first[b] - 1)

	def block_keys(self, n: int, b: int) -> int:
		t = self.t
		if b + 1 < t.blk[n + 1]:
			return t.block
		return (t.off[n + 1] - t.off[n]) - (b - t.blk[n]) * t.block

	def in_block(self, n: int, b: int, x: int) -> bool:
		t = self.t
		m = self.block_keys(n, b) - 1
		l, lo = t.meta[b] & 63, t.meta[b] >> META_L_BITS
		hi = lo + m * l
		hi_len = (t.meta[b + 1] >> META_L_BITS) - hi
		h, zeros = x >> l, hi_len - m
		if h > zeros:
			return False
		p = self.skip_zeros_back(hi, hi_len, zeros - h) if 2 * h > zeros else self.skip_zeros(hi, hi_len, h)
		if p < 0:
			return False
		return self.find_low(lo, l, p - h, self.run_ones(hi, hi_len, p), x & ((1 << l) - 1))

	def skip_zeros(self, hi: int, hi_len: int, h: int) -> int:
		"""Position just past the h-th zero of the high part (0 if h == 0), -1 if
		it has fewer than h zeros. Word by word; a partial last word pads with
		ONES so bits past hi_len never count."""
		if h == 0:
			return 0
		p = 0
		while True:
			take = min(32, hi_len - p)
			if take == 0:
				return -1
			w = self.read_bits(hi + p, take) | (MASK32 ^ ((1 << take) - 1))
			z = 32 - self.pop32(w)
			if z >= h:
				return p + select0_32(w, h - 1) + 1
			h, p = h - z, p + take

	def skip_zeros_back(self, hi: int, hi_len: int, r: int) -> int:
		"""Position just past the zero with r zeros after it, scanning backward."""
		e = hi_len
		while True:
			take = min(32, e)
			w = self.read_bits(hi + e - take, take) | (MASK32 ^ ((1 << take) - 1))
			z = 32 - self.pop32(w)
			if z > r:
				return e - take + select0_32(w, z - 1 - r) + 1
			r, e = r - z, e - take

	@staticmethod
	def pop32(v: int) -> int:
		return POP8[v & 255] + POP8[(v >> 8) & 255] + POP8[(v >> 16) & 255] + POP8[v >> 24]

	def run_ones(self, hi: int, hi_len: int, p: int) -> int:
		"""Length of the run of ones at high-part position p (the bucket size)."""
		c = 0
		while p < hi_len:
			take = min(32, hi_len - p)
			zeros = ~self.read_bits(hi + p, take) & ((1 << take) - 1)
			if zeros:
				return c + ctz(zeros)
			c, p = c + take, p + take
		return c

	def find_low(self, lo: int, l: int, i: int, c: int, x_low: int) -> bool:
		"""Is x_low among the c ascending l-bit lows from element i? (lower bound)"""
		a, b = i, i + c
		while a < b:
			m = (a + b) >> 1
			if self.read_bits(lo + m * l, l) < x_low:
				a = m + 1
			else:
				b = m
		return a < i + c and self.read_bits(lo + a * l, l) == x_low
