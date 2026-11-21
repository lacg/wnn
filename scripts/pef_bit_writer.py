"""PefBitWriter: an LSB-first bitstream packed into uint32 words (the layout
mcu/wnn_pef.h's reader expects: bit p lives in word p >> 5 at bit p & 31)."""


class PefBitWriter:
	def __init__(self):
		self.words = []
		self.cur = 0
		self.fill = 0

	def position(self) -> int:
		return 32 * len(self.words) + self.fill

	def write(self, value: int, width: int):
		"""Append the low `width` bits of value (width <= 32)."""
		if width == 0:
			return
		value &= (1 << width) - 1
		self.cur |= value << self.fill
		self.fill += width
		if self.fill >= 32:
			self.words.append(self.cur & 0xFFFFFFFF)
			self.cur >>= 32
			self.fill -= 32

	def set_bits(self, positions, length: int):
		"""Append `length` bits, 1 at each (relative, ascending) position."""
		start, prev = self.position(), 0
		for p in positions:
			self.zeros(p - prev)
			self.write(1, 1)
			prev = p + 1
		self.zeros(length - prev)
		assert self.position() == start + length

	def zeros(self, n: int):
		while n > 0:
			k = min(32, n)
			self.write(0, k)
			n -= k

	def finish(self) -> list:
		"""The words, the partial word flushed, plus one zero pad word so a
		two-word read at the last bit never runs off the array."""
		out = list(self.words)
		if self.fill:
			out.append(self.cur & 0xFFFFFFFF)
		return out + [0]
