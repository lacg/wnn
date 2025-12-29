from __future__ import annotations

from typing import Optional

from torch import tensor
from torch import Tensor
from torch import uint8

from dataclasses import dataclass
from random import Random

@dataclass(frozen=True)
class KVSpec:
	k_bits: int = 3
	v_bits: int = 2
	query_value: int = 0  # v=00 means query

	@property
	def window_bits(self) -> int:
		return self.k_bits + self.v_bits

	@property
	def k_size(self) -> int:
		return 1 << self.k_bits

	@property
	def v_size(self) -> int:
		return 1 << self.v_bits

	@staticmethod
	def int_to_bits(x: int, n_bits: int) -> list[int]:
		return [(x >> (n_bits - 1 - i)) & 1 for i in range(n_bits)]

	@staticmethod
	def bits_to_int(bits: list[int]) -> int:
		x = 0
		for b in bits:
			x = (x << 1) | (1 if b else 0)
		return x

	def decode_value_bits(self, out_bits: Tensor) -> int:
		# out_bits expected [1, v_bits] or [v_bits]
		if out_bits.ndim == 2:
			out_bits = out_bits[0]
		bits = [int(b.item()) for b in out_bits.to(uint8)]
		return KVSpec.bits_to_int(bits)

	def encode_window(self, key: int, value: int) -> Tensor:
		kb = KVSpec.int_to_bits(key, self.k_bits)
		vb = KVSpec.int_to_bits(value, self.v_bits)
		bits = kb + vb
		return tensor(bits, dtype=uint8).unsqueeze(0)  # [1, window_bits]

	def oracle_last_write_value(self, windows: list[tuple[int, int]]) -> int:
		"""
		windows: list of (key, value) ints, last window must be query: value==query_value
		returns expected value for that query key (most recent write)
		"""
		qk, qv = windows[-1]
		assert qv == self.query_value, "last window must be query"
		mem: dict[int, int] = {}
		for k, v in windows[:-1]:
			if v != self.query_value:  # writes only
				mem[k] = v
		# If key never written, define fallback = 0 (or choose random / special)
		return mem.get(qk, 0)

	def generate_episode(self, n_writes: int = 6, allow_overwrite: bool = True, require_query_key_seen: bool = True, rng: Optional[Random] = None) -> tuple[list[Tensor], Tensor, list[tuple[int,int]]]:
		"""
		Returns:
			windows_bits: list of [1, window_bits]
			target_bits:  [1, v_bits] (uint8)
			raw:          list of (k,v) ints for debugging
		"""
		rng = rng or Random()

		raw: list[tuple[int,int]] = []

		# writes
		used_keys = []
		for _ in range(n_writes):
			if allow_overwrite and used_keys and rng.random() < 0.35:
				k = rng.choice(used_keys)  # overwrite some key
			else:
				k = rng.randrange(self.k_size)
				used_keys.append(k)

			# avoid query token for write
			v = rng.randrange(self.v_size)
			if v == self.query_value:
				v = 1 if self.v_size > 1 else 0
			raw.append((k, v))

		# query
		if require_query_key_seen and used_keys:
			qk = rng.choice(used_keys)
		else:
			qk = rng.randrange(self.k_size)
		raw.append((qk, self.query_value))

		expected_v = self.oracle_last_write_value(raw)
		windows = [self.encode_window(k, v) for (k, v) in raw]
		target = tensor(KVSpec.int_to_bits(expected_v, self.v_bits), dtype=uint8).unsqueeze(0)  # [1, v_bits]
		return windows, target, raw


