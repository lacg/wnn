from wnn.ram.Memory import Memory
from wnn.ram.RAMEnums import MemoryVal

from typing import Optional

from torch import arange
from torch import device
from torch import int64
from torch import randint
from torch import Tensor
from torch.nn import Module



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

	def choose_address(self, neuron_index: int, desired_output: bool, current_address: int, dev: device) -> int:
		"""
		Generic helper: pick a memory address in `layer` for neuron_index that is
		either EMPTY or already stores the desired_output label.

		Hidden consistency is handled by EDRA writes afterwards.
		"""
		label_val = MemoryVal.TRUE if desired_output else MemoryVal.FALSE
		if self.num_neurons == 0 or self.get_memory(neuron_index, current_address) in [MemoryVal.EMPTY, label_val]:
			return current_address

		row = self.get_memory_row(neuron_index)  # [memory_size]

		label_mask = (row == MemoryVal.EMPTY) | (row == label_val)
		if not label_mask.any():
			return current_address

		candidate_indices = label_mask.nonzero(as_tuple=False).view(-1).to(device=dev, dtype=int64)
		idx = randint(0, candidate_indices.numel(), (1,), device=dev).item()
		return int(candidate_indices[idx].item())

	def forward(self, input_bits: Tensor) -> Tensor:
		"""
		Forward pass for RAM layer.
		input_bits: [batch_size, total_input_bits], dtype bool or 0/1
		returns: [batch_size, num_neurons] boolean outputs
		"""
		return self.memory(input_bits)

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

	def train_write(self, input_bits: Tensor, target_bits: Tensor) -> None:
		"""
		Write to memory directly.
		target_bits must be bool or {0,1}.
		"""
		self.memory.train_write(input_bits, target_bits)

	def resolve(self, addresses: Tensor, hidden_vals: Tensor) -> Tensor:
		"""
		Check EDRA feasibility for a set of candidate memory addresses.

		addresses:   [M] int64 candidate addresses in [0, memory_size)
		hidden_vals: [k] int64 memory values of each hidden neuron feeding this neuron
		             (in the same order as self.memory.connections[neuron_index])
		             values in {MemoryVal.FALSE, MemoryVal.TRUE, MemoryVal.EMPTY}

		Returns:
			ok_mask: [M] bool – True if for that address, for ALL hidden neurons:
			         cell is either EMPTY or already equal to the desired bit.
		"""
		if addresses.numel() == 0 or self.memory.n_bits_per_neuron == 0:
			return addresses == addresses  # all True / trivial

		# Decode each address into k bits (LSB-first)
		shifts = arange(self.memory.n_bits_per_neuron, device=addresses.device).unsqueeze(0)  # [1, self.memory.n_bits_per_neuron]
		desired_bits = ((addresses.unsqueeze(1) >> shifts) & 1).to(int64)											# [M, self.memory.n_bits_per_neuron]

		# Broadcast hidden values to [M, self.memory.n_bits_per_neuron]
		hidden_vals = hidden_vals.to(int64)
		hidden_matrix = hidden_vals.unsqueeze(0).expand(addresses.shape[0], self.memory.n_bits_per_neuron)

		# Feasible if each cell is EMPTY or matches desired bit
		ok = (hidden_matrix == MemoryVal.EMPTY) | (hidden_matrix == desired_bits)
		return ok.all(dim=1)

	def select_memory_address(self, neuron_index: int, desired_bit: bool, hidden_vals: Tensor) -> Optional[int]:
		"""
		Generic EDRA address selection for THIS layer & neuron.

		neuron_index: which neuron in this layer
		desired_bit:  bool, desired output for this neuron
		hidden_vals:  [k] int64 memory values for each hidden neuron
		              connected to this neuron, as seen at this timestep.

		Returns:
			address (int) in [0, memory_size), or None if no feasible address found.
		"""
		if self.num_neurons == 0:
			return None

		row = self.get_memory_row(neuron_index)   # [memory_size]

		label_val = MemoryVal.TRUE if desired_bit else MemoryVal.FALSE

		# Addresses that are label-compatible (EMPTY or already that label)
		label_mask = (row == MemoryVal.EMPTY) | (row == label_val)
		if not label_mask.any():
			return None

		candidates = label_mask.nonzero(as_tuple=False).view(-1).to(device=row.device, dtype=int64)

		# Check hidden feasibility
		ok_mask = self.resolve(candidates, hidden_vals)
		if not ok_mask.any():
			return None

		valid = candidates[ok_mask]
		if valid.numel() == 0:
			return None

		# Randomly pick one among valid addresses
		idx = randint(0, valid.numel(), (1,), device=row.device).item()
		return int(valid[idx].item())
