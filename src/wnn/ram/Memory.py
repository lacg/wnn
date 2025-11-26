from typing import Optional

from torch import arange
from torch import bool as tbool
from torch import device
from torch import full_like
from torch import int64
from torch import long
from torch import manual_seed
from torch import randint
from torch import Tensor
from torch import uint8
from torch import where
from torch import zeros
from torch.nn import Module

from enum import Enum

class MemoryVal(Enum):
	EMPTY = 0
	VAL0  = 1
	VAL1  = 2

class Memory(Module):
	"""
	Memory for the Layer RAM neurons with 2-bit memory values stored as uint8.
	Memory encoding:
		EMPTY = 0
		VAL0  = 1
		VAL1  = 2

	Connections are fixed (not trainable here).
	"""


	def __init__(
		self,
		total_input_bits: int,
		num_neurons: int,
		n_bits_per_neuron: int,
		connections: Optional[Tensor] = None,
		use_hashing: bool = False,
		hash_size: int = 1024,
		rng: Optional[int] = None,
	):
		super().__init__()

		self.total_input_bits = int(total_input_bits)
		self.num_neurons = int(num_neurons)
		self.n_bits_per_neuron = int(n_bits_per_neuron)
		self.use_hashing = bool(use_hashing)

		self.memory_size = hash_size if use_hashing else (2 ** n_bits_per_neuron)

		# memory: uint8, values in {EMPTY, VAL0, VAL1}
		self.register_buffer(
			"memory",
			zeros(self.num_neurons, self.memory_size, dtype=uint8)
		)

		# binary address place values: [1, 2, 4, ..., 2^(k-1)]
		self.register_buffer(
			"binary_addresses",
			(2 ** arange(self.n_bits_per_neuron, dtype=int64)).unsqueeze(0).repeat(self.num_neurons, 1)
		)

		if connections is None:
			self.register_buffer("connections", self._randomize_connections(rng))
		else:
			conn = connections.clone().long()
			assert conn.shape == (self.num_neurons, self.n_bits_per_neuron)
			assert conn.min().item() >= 0 and conn.max().item() < self.total_input_bits
			self.register_buffer("connections", conn)


	@property
	def device(self) -> device:
		self.memory.device


	def __getitem__(self, input_bits: Tensor) -> int:
		if input_bits.dtype != tbool:
			input_bits = input_bits.to(tbool)

		return self.memory[self._get_index(input_bits), self.get_addresses(input_bits)]
	

	def __setitem__(self, input_bits: Tensor, value: int) -> None:
		self.memory[self._get_index(input_bits), self.get_addresses(input_bits)] = value


	def _encode(self, bits: Tensor) -> int:
		if bits.dtype != tbool:
			bits = bits.to(tbool)

		# encode bits
		return where(
			bits,
			full_like(bits, MemoryVal.VAL1.value, dtype=uint8),
			full_like(bits, MemoryVal.VAL0.value, dtype=uint8),
		)


	def _get_index(self, input_bits: Tensor) -> int:
		"""
		Get index in int64 based on the input bits in binary array Tensor format.
		input_bits: [batch_size, total_input_bits], dtype bool or 0/1
		returns: index in int64
		"""
		# read memory values
		return arange(self.num_neurons, device=input_bits.device).unsqueeze(0).expand(input_bits.shape[0], -1)


	def _norm(self, tensor: Tensor) -> int:
		"""
		Used by EDRA: directly overwrite specific memory cells.
		bits = 0/1
		"""
		return tensor.to(self.device).long()


	def _randomize_connections(self, rng: Optional[int]) -> int:
		if rng is not None:
			manual_seed(rng)
		return randint(
			0,
			self.total_input_bits,
			(self.num_neurons, self.n_bits_per_neuron),
			dtype=long
		)


	def flip(self, neuron_idx: int, output: Tensor) -> None:
		"""
		Randomly flip ONE address in the given layer’s memory
		that corresponds to the given neuron index.
		Used by A1.2-R influence propagation.
		"""
		# compute addresses
		addr = int(self.get_addresses(output)[0, neuron_idx].item())

		# flip the memory cell
		self.memory[neuron_idx, addr] = MemoryVal.VAL0.value if (int(self.memory[neuron_idx, addr].item()) == MemoryVal.VAL1.value) else MemoryVal.VAL1.value


	def get_addresses(self, bits: Tensor) -> int:
		"""
		Get address in int64 based on the input bits in binary array Tensor format.
		bits: [batch_size, total_bits], dtype bool or 0/1
		returns: address in int64
		"""
		# gather inputs for each neuron → [batch_size, N, k]
		gathered = bits[:, self.connections]

		# compute addresses → [batch_size, N]
		addresses = (gathered.to(int64) * self.binary_addresses.unsqueeze(0)).sum(-1)
		return (addresses % self.memory_size).long() if self.use_hashing else addresses.long()


	def select_connection(self, neuron: Tensor) -> int:
		# RANDOM INFLUENCE:
		# Return one of the connections randomically.
		n_bits = self.connections.shape[1]
		rand_pos = randint(0, n_bits, (1,)).item()
		return self.connections[neuron, rand_pos]


	def set_memory(self, neuron_indice: int, address: int, bits: int) -> None:
		"""
		Used by EDRA: directly overwrite specific memory cells.
		bits = 0/1
		"""
		self.memory[neuron_indice, address] = bits

