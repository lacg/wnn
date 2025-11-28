from typing import Optional

from torch import arange
from torch import argmax
from torch import bool as tbool
from torch import cat
from torch import device
from torch import empty
from torch import full_like
from torch import int64
from torch import long
from torch import manual_seed
from torch import randint
from torch import randperm
from torch import stack
from torch import Tensor
from torch import uint8
from torch import where
from torch import zeros
from torch.nn import Module

from enum import Enum

class MemoryVal(Enum):
	FALSE			= 0
	TRUE			= 1
	EMPTY			= 2

class Memory(Module):
	"""
	Low-level RAM memory for a layer of RAM neurons.

	- total_input_bits: how many input bits each pattern has
	- num_neurons: number of RAM neurons
	- n_bits_per_neuron: how many input bits each neuron connects to
	- connections: [num_neurons, n_bits_per_neuron] indices into the input
	- memory: [num_neurons, memory_size] uint8 with 2-bit semantics:
		FALSE	= 0
		TRUE	= 1
		EMPTY	= 2
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

		# memory table: uint8 values in {EMPTY, FALSE, TRUE}
		self.register_buffer(
			"memory",
			zeros(self.num_neurons, self.memory_size, dtype=uint8),
		)

		# binary address weights for each neuron: [num_neurons, n_bits_per_neuron]
		addr_base = arange(self.n_bits_per_neuron, dtype=int64)
		self.register_buffer(
			"binary_addresses",
			(2 ** addr_base).unsqueeze(0).repeat(self.num_neurons, 1),
		)

		# connections: [num_neurons, n_bits_per_neuron]
		if connections is None:
			self.register_buffer("connections", self._randomize_connections(rng))
		else:
			conn = connections.clone().long()
			assert conn.shape == (self.num_neurons, self.n_bits_per_neuron)
			assert conn.min().item() >= 0 and conn.max().item() < self.total_input_bits
			self.register_buffer("connections", conn)


	def _randomize_connections(self, rng: Optional[int]) -> Tensor:
		if self.num_neurons == 0:
			# Return empty tensor with correct shape and type
			return empty(0, self.n_bits_per_neuron, dtype=long, device=self.memory.device)

		if rng is not None:
				manual_seed(rng)

		connections = []
		for _ in range(self.num_neurons):
			# Random permutation of all available bits
			unique = randperm(self.total_input_bits, device=self.memory.device)

			if self.n_bits_per_neuron <= self.total_input_bits:
				# Enough bits available → take first k (all unique)
				conn = unique[: self.n_bits_per_neuron]
			else:
				# Not enough bits → need to recycle
				full_cycles, remaining = divmod(self.n_bits_per_neuron, self.total_input_bits)

				# Add repeated full cycles of the unique set (reshuffled each time)
				parts = [randperm(self.total_input_bits, device=self.memory.device) for _ in range(full_cycles)]

				# Add partial cycle for leftover bits
				if remaining > 0:
					parts.append(randperm(self.total_input_bits, device=self.memory.device)[:remaining])

				conn = cat(parts, dim=0)

			connections.append(conn)

		return stack(connections, dim=0).long()

	def _set_memory(self, neuron_index: int, address: int, bits: Tensor) -> None:
		encoded = where(
			bits,
			full_like(bits, MemoryVal.TRUE.value, dtype=uint8),
			full_like(bits, MemoryVal.FALSE.value, dtype=uint8),
		)

		self.memory[neuron_index, address] = encoded


	def forward(self, input_bits: Tensor) -> Tensor:
		"""Forward lookup: returns boolean outputs [B, num_neurons]."""
		if input_bits.dtype != tbool:
			input_bits = input_bits.to(tbool)

		batch_size = input_bits.shape[0]
		addresses = self.get_addresses(input_bits)

		neuron_index = arange(self.num_neurons, device=input_bits.device).unsqueeze(0).expand(batch_size, -1)
		mem_vals = self.memory[neuron_index, addresses]  # [B, N]
		return mem_vals == MemoryVal.TRUE.value


	def get_addresses(self, input_bits: Tensor) -> Tensor:
		"""Compute integer addresses [batch_size, num_neurons] for given input bits."""
		if input_bits.dtype != tbool:
			input_bits = input_bits.to(tbool)
		gathered = input_bits[:, self.connections]
		addresses = (gathered.to(int64) * self.binary_addresses.unsqueeze(0)).sum(-1)
		if self.use_hashing:
			addresses = addresses % self.memory_size
		return addresses.long()


	def train_write(self, input_bits: Tensor, target_bits: Tensor) -> None:
		"""Direct write: set memory for each (sample, neuron) to target bit (0/1)."""
		if input_bits.dtype != tbool:
			input_bits = input_bits.to(tbool)
		if target_bits.dtype != tbool:
			target_bits = target_bits.to(tbool)

		batch_size = input_bits.shape[0]
		assert target_bits.shape[0] == batch_size
		assert target_bits.shape[1] == self.num_neurons

		addresses = self.get_addresses(input_bits)  # [batch_size, neuron_size]
		neuron_index = arange(self.num_neurons, device=input_bits.device).unsqueeze(0).expand(batch_size, -1)
		self._set_memory(neuron_index, addresses, target_bits)


	def force_set_memory(self, neuron_indexs: Tensor, addresses: Tensor, bits: Tensor) -> None:
		"""Used by EDRA: directly overwrite specific memory cells. bits = 0/1 or bool."""
		if bits.dtype != tbool:
			bits = bits.to(tbool)

		device = self.memory.device
		neuron_index = neuron_indexs.to(device).long()
		addresses = addresses.to(device).long()
		bits = bits.to(device)
		self._set_memory(neuron_index, addresses, bits)


	def flip(self, neuron_index: int, input_bits: Tensor) -> None:
		"""Flip one memory cell for given neuron, based on input_bits (batch size 1)."""
		addresses = self.get_addresses(input_bits)  # [batch_size, neuron_size]
		addr = int(addresses[0, neuron_index].item())

		current_val = int(self.memory[neuron_index, addr].item())
		self.memory[neuron_index, addr] = MemoryVal.FALSE.value if current_val == MemoryVal.TRUE.value else MemoryVal.TRUE.value


	def select_connection(self, neuron_index: int, use_high_impact: bool = True) -> int:
		"""
		Select one contributing input bit index for this neuron.

		- If use_high_impact is True (A1.3), pick the connection whose
		  address weight (binary_addresses) is largest for that neuron.
		- If use_high_impact is False, pick a random connection (A1.2-R).
		"""
		# number of bits this neuron is connected to
		num_bits = self.connections.shape[1]

		if num_bits == 0:
			# degenerate case, shouldn't really happen
			return 0

		if use_high_impact:
			# A1.3: choose the position with the largest binary weight
			# impacts: [num_bits] for this neuron
			impacts = self.binary_addresses[neuron_index]  # [num_bits]
			# index of max impact
			pos = int(argmax(impacts).item())
		else:
			# A1.2-R: random connection
			device = self.memory.device
			pos = int(randint(0, num_bits, (1,), device=device).item())
		return int(self.connections[neuron_index, pos].item())

	def set_memory(self, neuron_index: int, address: int, bit: bool):
		self.memory[neuron_index, address] = MemoryVal.TRUE.value if bit else MemoryVal.FALSE.value
