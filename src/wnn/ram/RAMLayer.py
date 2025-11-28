from typing import Optional

from torch import Tensor
from torch.nn import Module

from wnn.ram.Memory import Memory
from wnn.ram.Memory import MemoryVal


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

	def get_address(self, input_bits: Tensor) -> Tensor:
		"""
		Expose Memory.get_address for EDRA use.
		"""
		return self.memory.get_address(input_bits)

	def select_connection(self, neuron_index: int, use_high_impact: bool = True) -> int:
		"""
		Select a contributing input-bit index for this neuron.
		"""
		return self.memory.select_connection(neuron_index, use_high_impact)

	def set_memory(self, neuron_index: int, address: int, bit: bool) -> None:
		"""
		Used by EDRA: directly overwrite specific memory cell.
		bit = False/True
		"""
		self.memory.set_memory(neuron_index, address, bit)