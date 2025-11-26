from typing import Optional

from torch import Tensor
from torch.nn import Module


from wnn.ram.Memory import Memory
from wnn.ram.Memory import MemoryVal


class RAMLayer(Module):
	"""
	Layer of RAM neurons with 2-bit memory values stored as uint8.
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

		self._memory = Memory(total_input_bits, num_neurons, n_bits_per_neuron, connections, use_hashing, hash_size, rng)


	@property
	def memory(self) -> Memory:
		return self._memory


	@memory.setter
	def memory(self, new_memory) -> None:
		if not isinstance(new_memory, Memory):
			raise TypeError("Must assign an instance of Memory.")
		self._memory = new_memory


	def flip_memory(self, neuron_idx: Tensor, output: Tensor) -> None:
		"""
		Randomly flip ONE address in the given layer’s memory
		that corresponds to the given neuron index.
		Used by A1.2-R influence propagation.
		"""
		self.memory.flip(neuron_idx, output)


	def forward(self, input_bits: Tensor) -> bool:
		"""
		Forward pass for RAM layer.
		input_bits: [batch_size, total_input_bits], dtype bool or 0/1
		returns: [batch_size, num_neurons] boolean outputs
		"""
		# actual bit output: True if MemoryVal.VAL1.value, otherwise False
		return self.memory[input_bits] == MemoryVal.VAL1.value


	def get_addresses(self, input_bits: Tensor) -> int:
		"""
		Get address in int64 based on the input bits in binary array Tensor format.
		input_bits: [batch_size, total_input_bits], dtype bool or 0/1
		returns: address in int64
		"""
		# gather inputs for each neuron → [batch_size, N, k]
		return self.memory.get_addresses(input_bits)


	def train_write(self, input_bits: Tensor, target_bits: Tensor):
		"""
		Write to memory directly (output layer only uses this after stability).
		target_bits must be bool or {0,1}.
		"""
		self.memory[input_bits] = target_bits


	def select_connection(self, neuron: Tensor) -> int:
		# RANDOM INFLUENCE:
		# randomly flip 1 contributing hidden bit in output_layer_input
		return self.memory.select_connection(neuron)


	def set_memory(self, neuron_indice: int, address: int, bits: int):
		"""
		Used by EDRA: directly overwrite specific memory cells.
		bits = 0/1
		"""
		self.memory.set_memory(neuron_indice, address, bits)

