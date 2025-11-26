from torch.nn import Module
from torch import arange, bool as tbool, int64, LongTensor, int64, tensor, Tensor, zeros

# -------------------------
# RAMNeuron: basic 1-bit RAM
# -------------------------
class RAMNeuron(Module):

	"""
	Single weightless RAM neuron.
	- n_bits: number of input address bits (k)
	- memory_size: 2**k if use_hashing=False, else hash_size
	- memory stored as int64 but logically 0/1
	- supports forward (read) returns torch.bool and train_write (direct table writes)
	"""

	def __init__(self, n_bits: int, use_hashing: bool = False, hash_size: int = 1024):
		super().__init__()
		assert n_bits >= 1, "n_bits must be >= 1"
		self.n_bits = n_bits
		self.use_hashing = use_hashing
		if use_hashing:
			assert hash_size >= 2, "hash_size must be >= 2"
			self.memory_size = int(hash_size)
		else:
			# be careful: 2**n_bits can be huge
			self.memory_size = int(2 ** n_bits)

		# storage: int64 for alignment. Entries will be 0/1 only.
		self.register_buffer("_memory", zeros(self.memory_size, dtype=int64))

		# precompute bit weights (as int64) for address calculation
		# shape (n_bits,)
		self.register_buffer("_bit_weights", (2 ** arange(self.n_bits, dtype=int64)))

	def _address_from_bits(self, bits: Tensor) -> LongTensor:
		"""
		bits: Tensor[batch, n_bits] with values 0/1 (uint8 or float)
		returns: LongTensor[batch] addresses in [0, memory_size)
		"""
		# ensure int64 for arithmetic
		b = bits.to(int64)
		# dot with precomputed powers-of-two
		# result is int64 scalar per batch; then mod memory_size if hashing
		idx_64 = (b * self._bit_weights).sum(dim=-1)  # int64
		if self.use_hashing:
			# reduce to available memory via modulo (simple but effective)
			idx = (idx_64 % tensor(self.memory_size, dtype=int64, device=idx_64.device)).long()
		else:
			# idx_64 might exceed torch.long range if n_bits large; but for typical n_bits (<=48) it's ok
			idx = idx_64.long()
		return idx

	def forward(self, bits: Tensor) -> Tensor:
		"""
		bits: [batch, n_bits] with 0/1
		returns: [batch] int64 values {0,1}
		"""
		idx = self._address_from_bits(bits)
		mem = self._memory[idx]  # int64
		# logically 0/1, return as bool
		return (mem & 1).to(tbool)

	def train_write(self, bits: Tensor, target_bits: Tensor):
		"""
		Direct write to memory table (supervised / teacher-forcing style).
		bits: [batch, n_bits]
		target_bits: [batch] 0/1 (uint8/64 or convertible)
		"""
		idx = self._address_from_bits(bits)
		tb = target_bits.to(int64) & 1
		# write elementwise
		self._memory[idx] = tb