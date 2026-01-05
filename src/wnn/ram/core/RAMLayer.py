from wnn.ram.core.Memory import Memory
from wnn.ram.core.base import RAMComponent
from wnn.ram.core import MemoryVal

from typing import Optional

from torch import arange
from torch import device
from torch import int64
from torch import randint
from torch import Tensor


class RAMLayer(RAMComponent):
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

	def commit(self, input_bits: Tensor, target_bits: Tensor, allow_override: bool = False) -> bool:
		"""
		Finalize semantics:
		This input MUST map to this output.
		Direct write: set memory for each (sample, neuron) to target bit (False/True).
		target_bits must be bool or {0,1}.
		Finalize a correct mapping after convergence.

		Return:					True if any change happened, False otherwise.
		"""
		return self.memory.commit(input_bits, target_bits, allow_override)

	def explore(self, neuron_index: int, address: int, bit: bool, allow_override: bool = False) -> None:
		"""
		Explore a hypothesis:
		Write ONLY if the cell is EMPTY or already compatible.

		This does NOT finalize semantics, unless when allow_override = True.
		Materialize a hypothesis discovered by solve_constraints.
		"""
		self.memory.explore(neuron_index, address, bit, allow_override)

	def explore_batch(self, neuron_indices: Tensor, addresses: Tensor, bits: Tensor, allow_override: bool = True) -> None:
		"""
		Vectorized explore operation.

		neuron_indices: [K] int64
		addresses:      [K] int64
		bits:           [K] bool or {0,1}
		"""
		self.memory.explore_batch(neuron_indices, addresses, bits, allow_override=allow_override)

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

	def get_memory(self, neuron_index: int, address: int) -> int:
		"""
		Used by EDRA: directly get specific memory cell.
		returns a MemoryVal.{x}
		"""
		return self.memory.get_memory(neuron_index, address)

	def get_memory_row(self, neuron_index: int) -> Tensor:
		return self.memory.get_memory_row(neuron_index)

	def solve(self, input_bits: Tensor, target_bits: Tensor, n_immutable_bits: Optional[int] = None) -> Optional[Tensor]:
		"""
		Find an alternative input_bits vector that would produce target_bits.

		No memory is modified.
		Returns:
			- Tensor[new_input_bits] or
			- None if no admissible solution exists
		"""
		desired_input_bits = self.solve_constraints(input_bits, target_bits, False, n_immutable_bits)
		if desired_input_bits is None:
			return self.solve_constraints(input_bits, target_bits, True, n_immutable_bits)
		return desired_input_bits

	def solve_constraints(self, input_bits: Tensor, target_bits: Tensor, allow_override: bool = False, n_immutable_bits: Optional[int] = None) -> Optional[Tensor]:
		"""
		Find an alternative input_bits vector that would produce target_bits.

		No memory is modified.
		Returns:
			- Tensor[new_input_bits] or
			- None if no admissible solution exists
		"""
		return self.memory.solve_constraints(input_bits, target_bits, allow_override, n_immutable_bits)

	# Serialization support
	def get_config(self) -> dict:
		"""Get configuration dict for model recreation."""
		return {
			'total_input_bits': self.memory.total_input_bits,
			'num_neurons': self.memory.num_neurons,
			'n_bits_per_neuron': self.memory.n_bits_per_neuron,
			'use_hashing': self.memory.use_hashing,
			'hash_size': self.memory.memory_size if self.memory.use_hashing else 1024,
		}

	@classmethod
	def from_config(cls, config: dict) -> "RAMLayer":
		"""Create a RAMLayer from a configuration dict."""
		return cls(**config)

	def save(self, path: str) -> None:
		"""Save model to file."""
		from wnn.ram.core.serialization import save_model
		save_model(self, path)

	@classmethod
	def load(cls, path: str, device: str = 'cpu') -> "RAMLayer":
		"""Load model from file."""
		from wnn.ram.core.serialization import load_model
		return load_model(path, model_class=cls, device=device)
