"""PefEncoder: per-neuron sorted TRUE keys -> PefTables (partitioned Elias-Fano).

Lossless: the decoder (scripts/pef_decoder.py, mcu/wnn_pef.h) answers "is A in
neuron n's TRUE set?" identically to binary search on the uint32 array.

Block size B=64 was chosen by the 20/11/2026 compression probe
(logs/compression_probe/sizes.txt): 0.39-0.41x of the uint32 store, worst-case
lookup ~ today's binary search; Rice was 3x slower, FOR 2x slower.
"""

from pef_bit_writer import PefBitWriter
from pef_tables import META_L_BITS, MAX_BIT_OFFSET, PefTables

DEFAULT_BLOCK = 64


def low_width(u: int, m: int) -> int:
	"""Elias-Fano low-part width floor(log2(u/m)) (0 when u < 2m)."""
	if m == 0 or u < m:
		return 0
	return (u // m).bit_length() - 1


class PefEncoder:
	def __init__(self, block: int = DEFAULT_BLOCK):
		if block < 2:
			raise ValueError(f"block size {block} < 2")
		self.block = block

	def encode(self, keys: list, off: list) -> PefTables:
		"""keys/off exactly as true_onset_genome returns them (concatenated, per-neuron sorted)."""
		t = PefTables(block=self.block, off=list(off), blk=[0])
		w = PefBitWriter()
		for n in range(len(off) - 1):
			self.encode_neuron(keys[off[n]:off[n + 1]], t, w)
			t.blk.append(len(t.first))
		t.meta.append(self.meta_word(w.position(), 0))
		t.data = w.finish()
		return t

	def encode_neuron(self, k: list, t: PefTables, w: PefBitWriter):
		if any(b <= a for a, b in zip(k, k[1:])):
			raise ValueError("neuron keys must be strictly increasing")
		for s in range(0, len(k), self.block):
			self.encode_block(k[s:s + self.block], t, w)

	def encode_block(self, blk: list, t: PefTables, w: PefBitWriter):
		v = [x - blk[0] - 1 for x in blk[1:]]
		m = len(v)
		l = low_width(v[-1], m) if m else 0
		t.first.append(blk[0])
		t.meta.append(self.meta_word(w.position(), l))
		for x in v:
			w.write(x, l)
		hi_len = m + ((v[-1] >> l) if m else 0)
		w.set_bits([(x >> l) + i for i, x in enumerate(v)], hi_len)

	@staticmethod
	def meta_word(bit_offset: int, l: int) -> int:
		if bit_offset >= MAX_BIT_OFFSET:
			raise ValueError(f"EF bitstream {bit_offset} bits >= {MAX_BIT_OFFSET} (meta packs 26 bits)")
		return (bit_offset << META_L_BITS) | l
