"""PefTables: the flat arrays of a partitioned Elias-Fano (PEF) key store, exactly
as the MCU header ships them (mcu/wnn_pef.h decodes them).

Layout (all uint32, one global copy for all neurons):
  off[N+1]     key count prefix per neuron (the uint32 format's wnn_off, unchanged)
  blk[N+1]     block prefix per neuron: neuron n owns blocks [blk[n], blk[n+1])
  first[NB]    the first (smallest) key of each block
  meta[NB+1]   (bit offset of the block's EF data << 6) | l ; meta[NB] = end sentinel
  data[W+1]    the EF bitstream, LSB-first in 32-bit words, + one zero pad word

A block of c keys k0 < k1 < ... stores m = c-1 values v_i = k_i - k0 - 1 as
Elias-Fano: m low parts of l bits each, then the high parts in unary
(bit (v_i >> l) + i set), m + (v_max >> l) bits. Block c = B except a neuron's
last block. Built by PefEncoder (scripts/pef_encoder.py).
"""

from dataclasses import dataclass, field

# meta packs l (<= 31) into the low 6 bits; the bit offset gets the other 26.
META_L_BITS = 6
MAX_BIT_OFFSET = 1 << (32 - META_L_BITS)


@dataclass
class PefTables:
	block: int
	off: list = field(default_factory=list)
	blk: list = field(default_factory=list)
	first: list = field(default_factory=list)
	meta: list = field(default_factory=list)
	data: list = field(default_factory=list)

	def num_blocks(self) -> int:
		return len(self.first)

	def table_bytes(self) -> dict:
		"""Bytes of every shipped array (conn excluded: same in both formats)."""
		return dict(off=4 * len(self.off), blk=4 * len(self.blk), first=4 * len(self.first),
		            meta=4 * len(self.meta), data=4 * len(self.data))

	def nbytes(self) -> int:
		return sum(self.table_bytes().values())
