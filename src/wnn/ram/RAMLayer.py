from typing import Optional

from torch import Tensor
from torch.nn import Module

from wnn.ram.Memory import Memory


class RAMLayer(Module):
	"""
	Layer of RAM neurons backed by a shared Memory object.
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
		self.memory = Memory(
			total_input_bits=total_input_bits,
			num_neurons=num_neurons,
			n_bits_per_neuron=n_bits_per_neuron,
			connections=connections,
			use_hashing=use_hashing,
			hash_size=hash_size,
			rng=rng,
		)

	def __repr__(self):
		return (
			f"RAMLayer"
			f"("
			f"{self.memory.__repr__()}"
			f")"
		)

	def __str__(self):
		lines = []
		lines.append("=== RAMLayer ===")
		lines.append(str(self.memory))
		return "\n".join(lines)

	@property
	def num_neurons(self) -> int:
		return self.memory.num_neurons

	def forward(self, input_bits: Tensor) -> Tensor:
		"""
		Forward pass for RAM layer.
		input_bits: [batch_size, total_input_bits], dtype bool or 0/1
		returns: [batch_size, num_neurons] boolean outputs
		"""
		return self.memory(input_bits)

	def train_write(self, input_bits: Tensor, target_bits: Tensor) -> None:
		"""
		Write to memory directly.
		target_bits must be bool or {0,1}.
		"""
		self.memory.train_write(input_bits, target_bits)

	def get_addresses(self, input_bits: Tensor) -> Tensor:
		"""
		Expose Memory.get_address for EDRA use.
		"""
		return self.memory.get_addresses(input_bits)

	def get_address_for_neuron(self, neuron_index: int, input_bits: Tensor) -> int:
		return self.memory.get_address_for_neuron(neuron_index, input_bits)

	def get_memories_for_bits(self, input_bits: Tensor) -> Tensor:
		return self.memory.get_memories_for_bits(input_bits)

	def get_memory(self, neuron_index: int, address: int) -> int:
		"""
		Used by EDRA: directly get specific memory cell.
		returns a MemoryVal.{x}
		"""
		return self.memory.get_memory(neuron_index, address)

	def get_memory_row(self, neuron_index: int) -> Tensor:
		return self.memory.get_memory_row(neuron_index)

	def select_connection(self, neuron_index: int, use_high_impact: bool = True) -> int:
		"""
		Select a contributing input-bit index for this neuron.
		"""
		return self.memory.select_connection(neuron_index, use_high_impact)

	def set_memory(self, neuron_index: int, address: int, bit: bool, allow_override: bool = False) -> None:
		"""
		Used by EDRA: directly overwrite specific memory cell.
		bit = False/True
		"""
		self.memory.set_memory(neuron_index, address, bit)

	def set_memory_batch(self, neuron_indices: Tensor, addresses: Tensor, bits: Tensor, allow_override: bool = True) -> None:
		"""
		Vectorized memory set: writes many (neuron, address) cells in one call.

		neuron_indices: [K] int64
		addresses:      [K] int64
		bits:           [K] bool or {0,1}
		"""
		self.memory.set_memory_batch(neuron_indices, addresses, bits, allow_override=allow_override)
