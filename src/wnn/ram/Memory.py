from enum import Enum
from typing import Optional

from torch import arange
from torch import argmax
from torch import bool as tbool
from torch import cat
from torch import empty
from torch import full
from torch import full_like
from torch import int64
from torch import long
from torch import manual_seed
from torch import randint
from torch import randperm
from torch import Tensor
from torch import uint8
from torch import where
from torch import stack
from torch.nn import Module


class MemoryVal(Enum):
	"""
	Ternary memory values stored as uint8:
	- FALSE = 0
	- TRUE  = 1
	- EMPTY = 2   (safe to overwrite; means "untrained")
	"""
	FALSE = 0
	TRUE = 1
	EMPTY = 2


class Memory(Module):
	"""
	Low-level RAM memory for a layer of RAM neurons.

	- total_input_bits: how many input bits each pattern has
	- num_neurons: number of RAM neurons
	- n_bits_per_neuron: how many input bits each neuron connects to
	- connections: [num_neurons, n_bits_per_neuron] indices into the input
	- memory: [num_neurons, memory_size] uint8 with semantics:

		FALSE = 0
		TRUE  = 1
		EMPTY = 2
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
		self.memory_size = int(hash_size if use_hashing else (1 << n_bits_per_neuron))

		# 1) memory: [num_neurons, memory_size], initialized to EMPTY
		self.register_buffer(
			"memory",
			full((self.num_neurons, self.memory_size), MemoryVal.EMPTY.value, dtype=uint8),
		)

		# 2) binary address weights for each neuron: [num_neurons, n_bits_per_neuron]
		addr_base = arange(self.n_bits_per_neuron, dtype=int64)
		self.register_buffer(
			"binary_addresses",
			(2 ** addr_base).unsqueeze(0).repeat(self.num_neurons, 1),
		)

		# 3) connections: [num_neurons, n_bits_per_neuron]
		if connections is None:
			self.register_buffer("connections", self._randomize_connections(rng))
		else:
			conn = connections.clone().long()
			assert conn.shape == (self.num_neurons, self.n_bits_per_neuron)
			assert conn.min().item() >= 0 and conn.max().item() < self.total_input_bits
			self.register_buffer("connections", conn)

	def _randomize_connections(self, rng: Optional[int]) -> Tensor:
		"""
		Random connectivity:
		- each neuron has n_bits_per_neuron distinct inputs when possible
		- if n_bits_per_neuron > total_input_bits, we reuse inputs
		"""
		if self.num_neurons == 0:
			# Return empty tensor with correct shape and type
			return empty(0, self.n_bits_per_neuron, dtype=long, device=self.memory.device)

		if rng is not None:
			manual_seed(rng)

		layer_connections = []
		for _ in range(self.num_neurons):
			# Random permutation of all available bits
			unique = randperm(self.total_input_bits, device=self.memory.device)

			if self.n_bits_per_neuron <= self.total_input_bits:
				# Enough bits available → take first k (all unique)
				neuron_connections = unique[: self.n_bits_per_neuron]
			else:
				# Not enough bits → need to recycle
				full_cycles, remaining = divmod(self.n_bits_per_neuron, self.total_input_bits)

				# Add repeated full cycles of the unique set (reshuffled each time)
				partial_neuron_connections = [randperm(self.total_input_bits, device=self.memory.device) for _ in range(full_cycles)]

				# Add partial cycle for leftover bits
				if remaining > 0:
					partial_neuron_connections.append(randperm(self.total_input_bits, device=self.memory.device)[:remaining])

				neuron_connections = cat(partial_neuron_connections, dim=0)

			layer_connections.append(neuron_connections)

		return stack(layer_connections, dim=0).long()

	def get_address(self, input_bits: Tensor) -> Tensor:
		"""
		Compute integer addresses [batch_size, num_neurons] for given input bits.
		input_bits: [B, total_input_bits], bool or {0,1}
		"""
		if input_bits.dtype != tbool:
			input_bits = input_bits.to(tbool)

		# gather inputs for each neuron → [B, N, k]
		gathered = input_bits[:, self.connections]

		# compute addresses → [B, N]
		address = (gathered.to(int64) * self.binary_addresses.unsqueeze(0)).sum(-1)
		if self.use_hashing:
			address = address % self.memory_size
		return address.long()

	def forward(self, input_bits: Tensor) -> Tensor:
		"""
		Forward lookup: returns boolean outputs [B, num_neurons].
		TRUE cells (1) → True
		FALSE and EMPTY → False
		"""
		address = self.get_address(input_bits)  # [B, N]
		batch_size = address.shape[0]

		neuron_index = arange(self.num_neurons, dtype=long, device=address.device).unsqueeze(0).expand(batch_size, -1)
		mem_vals = self.memory[neuron_index, address]  # [B, N]
		return mem_vals == MemoryVal.TRUE.value

	def train_write(self, input_bits: Tensor, target_bits: Tensor) -> None:
		"""
		Direct write: set memory for each (sample, neuron) to target bit (False/True).
		target_bits must be bool or {0,1}.
		"""
		if input_bits.dtype != tbool:
			input_bits = input_bits.to(tbool)
		if target_bits.dtype != tbool:
			target_bits = target_bits.to(tbool)

		address = self.get_address(input_bits)  # [B, N]
		batch_size = address.shape[0]

		neuron_index = arange(self.num_neurons, dtype=long, device=address.device).unsqueeze(0).expand(batch_size, -1)

		encoded = where(
			target_bits,
			full_like(target_bits, MemoryVal.TRUE.value, dtype=uint8),
			full_like(target_bits, MemoryVal.FALSE.value, dtype=uint8),
		)
		self.memory[neuron_index, address] = encoded

	def select_connection(self, neuron_index: int, use_high_impact: bool = True) -> int:
		"""
		Select one contributing input-bit index for this neuron.
		If use_high_impact=True, chooses the connection with the largest binary weight.
		Otherwise chooses a random connection.
		Returns: integer index into the layer's input bit vector.
		"""
		num_bits = self.n_bits_per_neuron
		if use_high_impact:
			impacts = self.binary_addresses[neuron_index]  # [k]
			pos = int(argmax(impacts).item())
		else:
			dev = self.memory.device
			pos = int(randint(0, num_bits, (1,), device=dev).item())
		return int(self.connections[neuron_index, pos].item())

	def set_memory(self, neuron_index: int, address: int, bit: bool) -> None:
		"""
		Used by EDRA: directly overwrite specific memory cell (for one neuron & address).
		bit = False/True
		"""
		self.memory[neuron_index, address] = MemoryVal.TRUE.value if bit else MemoryVal.FALSE.value