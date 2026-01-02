"""
Sparse Memory for RAM Neurons

Memory-efficient storage for large address spaces where most cells remain empty.
Uses dictionary-based storage instead of dense packed arrays.

Key differences from dense Memory:
- Storage: dict[(neuron_idx, address)] -> value  (only non-EMPTY cells)
- Memory: O(written_cells) instead of O(neurons × 2^bits_per_neuron)
- Lookup: O(1) hash lookup instead of O(1) array indexing (slightly slower)
- Best for: Large address spaces (>16 bits) with sparse writes

Trade-offs:
- Pro: Memory scales with actual usage, not potential address space
- Pro: No memory allocation for unused addresses
- Con: Slightly slower lookups due to hash table overhead
- Con: No vectorized operations on dense memory regions

Usage:
	# Choose sparse when address space is large and sparse
	memory = SparseMemory(
		total_input_bits=24,  # 16M addresses - would be huge with dense!
		num_neurons=64,
		n_bits_per_neuron=24,
	)

	# Same interface as dense Memory
	memory.commit(input_bits, target_bits)
	output = memory.forward(input_bits)
"""

from typing import Optional
from collections import defaultdict

from torch import arange
from torch import bool as tbool
from torch import device
from torch import empty
from torch import float64
from torch import full
from torch import int8
from torch import int64
from torch import isfinite
from torch import long
from torch import manual_seed
from torch import ones_like
from torch import randperm
from torch import tensor
from torch import Tensor
from torch import stack
from torch import uint8
from torch import where
from torch import zeros
from torch.nn import Module

from wnn.ram.enums import CostCalculatorType, MemoryVal
from wnn.ram.factories.cost import CostCalculatorFactory


class SparseMemory(Module):
	"""
	Sparse RAM memory using dictionary-based storage.

	Only stores non-EMPTY cells, making it memory-efficient for large
	address spaces with sparse writes.

	Interface is compatible with dense Memory class.
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
		cost_calculator_type: CostCalculatorType = CostCalculatorType.STOCHASTIC,
		epochs: int = 1000,
	) -> None:
		super().__init__()

		assert total_input_bits >= 0
		assert num_neurons >= 0
		assert n_bits_per_neuron >= 0

		self.total_input_bits = int(total_input_bits)
		self.num_neurons = int(num_neurons)
		self.n_bits_per_neuron = int(n_bits_per_neuron)
		self.use_hashing = bool(use_hashing)
		self.memory_size = int(hash_size if use_hashing else (1 << n_bits_per_neuron))

		self.cost_calculator = CostCalculatorFactory.create(
			cost_calculator_type, epochs, num_neurons
		)

		# Sparse storage: dict[(neuron_idx, address)] -> MemoryVal
		# Only stores non-EMPTY values
		self._storage: dict[tuple[int, int], int] = {}

		# Track statistics
		self._write_count = 0

		# Binary address computation
		if self.n_bits_per_neuron > 0:
			base = (2 ** arange(self.n_bits_per_neuron - 1, -1, -1, dtype=int64)).unsqueeze(0)
			self.register_buffer("binary_addresses", base.repeat(self.num_neurons, 1))
		else:
			self.register_buffer("binary_addresses", empty(self.num_neurons, 0, dtype=int64))

		# Connections
		if connections is None:
			self.register_buffer("connections", self._randomize_connections(rng))
		else:
			conn = connections.clone().long()
			assert conn.shape == (self.num_neurons, self.n_bits_per_neuron)
			assert conn.min().item() >= 0 and conn.max().item() < self.total_input_bits
			self.register_buffer("connections", conn)

		# Address bits lookup (for constraint solving)
		self.register_buffer(
			"addresses_bits",
			self._decode_addresses_bits(self.n_bits_per_neuron, arange(self.memory_size)).to(int8)
		)

	def __repr__(self):
		density = self._write_count / max(1, self.num_neurons * self.memory_size)
		return (
			f"SparseMemory("
			f"neurons={self.num_neurons}, "
			f"bits_per_neuron={self.n_bits_per_neuron}, "
			f"memory_size={self.memory_size}, "
			f"stored={self._write_count}, "
			f"density={density:.4%})"
		)

	def __str__(self):
		lines = []
		lines.append(f"\n=== SparseMemory ===")
		lines.append(f"total_input_bits = {self.total_input_bits}")
		lines.append(f"num_neurons      = {self.num_neurons}")
		lines.append(f"bits_per_neuron  = {self.n_bits_per_neuron}")
		lines.append(f"memory_size      = {self.memory_size}")
		lines.append(f"stored_cells     = {self._write_count}")
		density = self._write_count / max(1, self.num_neurons * self.memory_size)
		lines.append(f"density          = {density:.4%}")

		lines.append("\nConnections:")
		for i in range(min(self.num_neurons, 5)):  # Limit output
			row = self.connections[i].tolist()
			lines.append(f"\tneuron {i}: {row}")
		if self.num_neurons > 5:
			lines.append(f"\t... ({self.num_neurons - 5} more neurons)")

		lines.append("\nStored cells (sample):")
		count = 0
		for (neuron_idx, addr), val in sorted(self._storage.items())[:20]:
			val_name = {0: "FALSE", 1: "TRUE", 2: "EMPTY"}.get(val, str(val))
			lines.append(f"\tneuron {neuron_idx}, addr {addr}: {val_name}")
			count += 1
		if len(self._storage) > 20:
			lines.append(f"\t... ({len(self._storage) - 20} more cells)")

		lines.append("")
		return "\n".join(lines)

	def _randomize_connections(self, rng: Optional[int]) -> Tensor:
		"""Generate random connections with guaranteed coverage."""
		if self.num_neurons == 0 or self.n_bits_per_neuron == 0 or self.total_input_bits == 0:
			return empty(self.num_neurons, self.n_bits_per_neuron, dtype=long)

		# Full connectivity: use canonical connections
		if self.n_bits_per_neuron == self.total_input_bits:
			canonical = arange(self.total_input_bits, dtype=long)
			return canonical.unsqueeze(0).expand(self.num_neurons, -1).clone()

		if rng is not None:
			manual_seed(rng)

		# Random connections with coverage guarantee
		total_slots = self.num_neurons * self.n_bits_per_neuron
		connections = []

		for _ in range(self.num_neurons):
			if self.n_bits_per_neuron >= self.total_input_bits:
				# Use all bits, possibly repeated
				conn = list(range(self.total_input_bits))
				while len(conn) < self.n_bits_per_neuron:
					conn.extend(range(self.total_input_bits))
				conn = conn[:self.n_bits_per_neuron]
			else:
				# Random subset
				perm = randperm(self.total_input_bits)
				conn = perm[:self.n_bits_per_neuron].tolist()
			connections.append(tensor(conn, dtype=long))

		return stack(connections, dim=0)

	@staticmethod
	def _decode_addresses_bits(n_bits: int, addresses: Tensor) -> Tensor:
		"""Decode integer addresses to bit patterns."""
		shifts = arange(n_bits).unsqueeze(0)
		address = addresses.to(int64).unsqueeze(1)
		return ((address >> (n_bits - 1 - shifts)) & 1).to(tbool)

	# =========================================================================
	# Core read/write operations
	# =========================================================================

	def _read_cell(self, neuron_idx: int, address: int) -> int:
		"""Read a single cell, returning EMPTY if not stored."""
		return self._storage.get((neuron_idx, address), MemoryVal.EMPTY)

	def _write_cell(self, neuron_idx: int, address: int, value: int) -> bool:
		"""Write a single cell. Returns True if storage changed."""
		key = (neuron_idx, address)

		if value == MemoryVal.EMPTY:
			# Remove from storage if exists
			if key in self._storage:
				del self._storage[key]
				self._write_count -= 1
				return True
			return False

		old_value = self._storage.get(key)
		if old_value != value:
			if old_value is None:
				self._write_count += 1
			self._storage[key] = value
			return True
		return False

	def get_memory(self, neuron_index: int, address: int) -> int:
		"""Get memory value at specific location."""
		return self._read_cell(neuron_index, address)

	def get_memory_row(self, neuron_index: int) -> Tensor:
		"""Get all memory values for a neuron (for constraint solving)."""
		row = full((self.memory_size,), MemoryVal.EMPTY, dtype=int64)
		for addr in range(self.memory_size):
			val = self._storage.get((neuron_index, addr))
			if val is not None:
				row[addr] = val
		return row

	# =========================================================================
	# Address computation
	# =========================================================================

	def get_addresses(self, input_bits: Tensor) -> Tensor:
		"""Compute integer addresses for given input bits."""
		if self.num_neurons == 0:
			return empty(input_bits.shape[0], 0, dtype=int64, device=input_bits.device)

		if input_bits.ndim == 1:
			input_bits = input_bits.unsqueeze(0)

		assert input_bits.shape[1] == self.total_input_bits

		if input_bits.dtype == tbool:
			input_bits64 = input_bits.to(int64)
		else:
			input_bits64 = (input_bits != 0).to(int64)

		gathered = input_bits64[:, self.connections]
		addresses = (gathered.to(int64) * self.binary_addresses.unsqueeze(0)).sum(-1)

		if self.use_hashing:
			addresses = addresses % self.memory_size

		return addresses.long()

	def get_memories_for_bits(self, input_bits: Tensor) -> Tensor:
		"""Get memory values for input patterns."""
		if self.num_neurons == 0:
			return empty(input_bits.shape[0], 0, dtype=int64, device=input_bits.device)

		addresses = self.get_addresses(input_bits)
		batch_size = addresses.shape[0]

		result = full((batch_size, self.num_neurons), MemoryVal.EMPTY, dtype=int64)

		for b in range(batch_size):
			for n in range(self.num_neurons):
				addr = int(addresses[b, n].item())
				result[b, n] = self._read_cell(n, addr)

		return result

	# =========================================================================
	# Forward pass
	# =========================================================================

	def forward(self, input_bits: Tensor) -> Tensor:
		"""Forward lookup: TRUE cells → True, others → False."""
		return self.get_memories_for_bits(input_bits) == MemoryVal.TRUE

	# =========================================================================
	# Training operations
	# =========================================================================

	def commit(
		self,
		input_bits: Tensor,
		target_bits: Tensor,
		allow_override: bool = False,
	) -> bool:
		"""
		Commit a mapping (finalize semantics).

		Writes even if cell is occupied.
		"""
		input_bits = input_bits.to(uint8)
		if input_bits.ndim == 1:
			input_bits = input_bits.unsqueeze(0)

		target_bits = target_bits.to(tbool)
		if target_bits.ndim == 1:
			target_bits = target_bits.unsqueeze(0)

		addresses = self.get_addresses(input_bits)[0]
		changed = False

		for n in range(self.num_neurons):
			addr = int(addresses[n].item())
			value = MemoryVal.TRUE if target_bits[0, n].item() else MemoryVal.FALSE

			current = self._read_cell(n, addr)
			if current == MemoryVal.EMPTY or allow_override:
				if self._write_cell(n, addr, value):
					changed = True

		return changed

	def explore(
		self,
		neuron_index: int,
		address: int,
		bit: bool,
		allow_override: bool = False,
	) -> None:
		"""Explore a hypothesis (provisional write)."""
		if self.num_neurons == 0:
			return

		current = self._read_cell(neuron_index, address)
		desired = MemoryVal.TRUE if bit else MemoryVal.FALSE

		if current == MemoryVal.EMPTY or (allow_override and desired != current):
			self._write_cell(neuron_index, address, desired)

	def explore_batch(
		self,
		neuron_indices: Tensor,
		addresses: Tensor,
		bits: Tensor,
		allow_override: bool = False,
	) -> bool:
		"""Vectorized explore operation."""
		if self.num_neurons == 0:
			return False

		if bits.dtype != tbool:
			bits = bits.to(tbool)

		changed = False
		for i in range(len(neuron_indices)):
			n = int(neuron_indices[i].item())
			addr = int(addresses[i].item())
			bit = bits[i].item()

			current = self._read_cell(n, addr)
			desired = MemoryVal.TRUE if bit else MemoryVal.FALSE

			if current == MemoryVal.EMPTY or (allow_override and current != desired):
				if self._write_cell(n, addr, desired):
					changed = True

		return changed

	def flip(self, neuron_index: int, input_bits: Tensor) -> None:
		"""Flip one memory cell."""
		if self.num_neurons == 0:
			return

		addresses = self.get_addresses(input_bits)
		addr = int(addresses[0, neuron_index].item())

		current = self._read_cell(neuron_index, addr)
		new_val = MemoryVal.FALSE if current == MemoryVal.TRUE else MemoryVal.TRUE
		self._write_cell(neuron_index, addr, new_val)

	# =========================================================================
	# Constraint solving (simplified version)
	# =========================================================================

	def solve_constraints(
		self,
		input_bits: Tensor,
		target_bits: Tensor,
		allow_override: bool = False,
		n_immutable_bits: int = 0,
		topk_per_neuron: int = 4,
	) -> Optional[Tensor]:
		"""
		Constraint solver for finding input bits that produce target outputs.

		Simplified version that delegates to per-neuron search.
		"""
		if input_bits.ndim == 2:
			assert input_bits.shape[0] == 1
			input_bits = input_bits[0]

		assert input_bits.ndim == 1 and input_bits.numel() == self.total_input_bits
		assert target_bits.ndim == 2 and target_bits.shape[0] == 1

		input_bits = input_bits.to(tbool)
		target_bits = target_bits.to(tbool)

		# Use the partial connectivity solver approach
		return self._solve_partial_connectivity(
			input_bits, target_bits, allow_override, n_immutable_bits, topk_per_neuron
		)

	def _solve_partial_connectivity(
		self,
		input_bits: Tensor,
		target_bits: Tensor,
		allow_override: bool,
		n_immutable_bits: int,
		topk_per_neuron: int,
	) -> Optional[Tensor]:
		"""Per-neuron beam search for constraint solving."""
		MAX_BEAM_WIDTH = 64
		CONFLICT_COST, EMPTY_COST, HAMMING_COST = 20.0, 10.0, 1.0
		beam_width = min(MAX_BEAM_WIDTH, max(8 * self.num_neurons, 16))
		k_top = min(topk_per_neuron, self.memory_size)
		device_val = input_bits.device

		# Precompute per-neuron costs
		memory_rows = stack([self.get_memory_row(n) for n in range(self.num_neurons)], dim=0)
		desired_bits = target_bits[0].to(tbool)
		desired_memories = where(desired_bits, MemoryVal.TRUE, MemoryVal.FALSE).unsqueeze(1)

		conflict = (memory_rows != MemoryVal.EMPTY) & (memory_rows != desired_memories)
		empty_cells = (memory_rows == MemoryVal.EMPTY)
		valid = ~conflict if not allow_override else ones_like(conflict, dtype=tbool)

		current_addr_bits = input_bits[self.connections].to(int8)
		hamming = (self.addresses_bits.unsqueeze(0) != current_addr_bits.unsqueeze(1)).sum(dim=2).to(float64)

		per_neuron_cost = (
			CONFLICT_COST * conflict.to(float64) +
			EMPTY_COST * empty_cells.to(float64) +
			HAMMING_COST * hamming
		)
		per_neuron_cost = per_neuron_cost.masked_fill(~valid, float("inf"))

		# Initialize beam
		candidate_bits = full((1, self.total_input_bits), -1, dtype=int8, device=device_val)
		candidate_cost = zeros(1, dtype=float64, device=device_val)

		if n_immutable_bits > 0:
			candidate_bits[0, :n_immutable_bits] = input_bits[:n_immutable_bits].to(int8)

		# Process each neuron
		for neuron_idx in range(self.num_neurons):
			neuron_conn = self.connections[neuron_idx]
			neuron_costs = per_neuron_cost[neuron_idx]

			topk_costs, topk_addrs = neuron_costs.topk(k_top, largest=False)
			finite_mask = isfinite(topk_costs)
			if not finite_mask.any():
				return None

			n_candidates = candidate_bits.shape[0]
			n_addrs = finite_mask.sum().item()

			expanded_bits = candidate_bits.unsqueeze(1).expand(-1, n_addrs, -1).reshape(-1, self.total_input_bits).clone()
			expanded_cost = candidate_cost.unsqueeze(1).expand(-1, n_addrs).reshape(-1).clone()

			valid_addrs = topk_addrs[finite_mask]
			valid_addr_costs = topk_costs[finite_mask]
			addr_bits = self.addresses_bits[valid_addrs]

			addr_bits_tiled = addr_bits.unsqueeze(0).expand(n_candidates, -1, -1).reshape(-1, self.n_bits_per_neuron)
			addr_costs_tiled = valid_addr_costs.unsqueeze(0).expand(n_candidates, -1).reshape(-1)

			existing = expanded_bits[:, neuron_conn]

			# Check immutable constraints
			if n_immutable_bits > 0:
				immutable_in_conn = neuron_conn < n_immutable_bits
				if immutable_in_conn.any():
					immutable_existing = existing[:, immutable_in_conn]
					immutable_required = addr_bits_tiled[:, immutable_in_conn]
					immutable_conflict = (immutable_existing != immutable_required).any(dim=1)
					expanded_cost = expanded_cost.masked_fill(immutable_conflict, float("inf"))

			# Check already-constrained conflicts
			constrained_mask = existing != -1
			constraint_conflict = constrained_mask & (existing != addr_bits_tiled)
			has_conflict = constraint_conflict.any(dim=1)
			expanded_cost = expanded_cost.masked_fill(has_conflict, float("inf"))

			expanded_cost = expanded_cost + addr_costs_tiled

			# Merge
			merged = where(existing == -1, addr_bits_tiled, existing)
			expanded_bits[:, neuron_conn] = merged

			# Prune
			if expanded_bits.shape[0] > beam_width:
				best_costs, best_indices = expanded_cost.topk(beam_width, largest=False)
				candidate_bits = expanded_bits[best_indices]
				candidate_cost = best_costs
			else:
				candidate_bits = expanded_bits
				candidate_cost = expanded_cost

			if not isfinite(candidate_cost).any():
				return None

		if not isfinite(candidate_cost).any():
			return None

		best_idx = self.cost_calculator.calculate_index(candidate_cost)
		best = candidate_bits[best_idx]
		best = where(best == -1, input_bits.to(int8), best)
		return best.to(tbool)

	# =========================================================================
	# Utility methods
	# =========================================================================

	def clear(self) -> None:
		"""Clear all stored values."""
		self._storage.clear()
		self._write_count = 0

	def density(self) -> float:
		"""Return storage density (stored / possible)."""
		total_possible = self.num_neurons * self.memory_size
		if total_possible == 0:
			return 0.0
		return self._write_count / total_possible

	def stored_count(self) -> int:
		"""Return number of stored (non-EMPTY) cells."""
		return self._write_count

	# =========================================================================
	# Serialization
	# =========================================================================

	def get_config(self) -> dict:
		"""Get configuration dict for model recreation."""
		return {
			'total_input_bits': self.total_input_bits,
			'num_neurons': self.num_neurons,
			'n_bits_per_neuron': self.n_bits_per_neuron,
			'use_hashing': self.use_hashing,
			'hash_size': self.memory_size if self.use_hashing else 1024,
		}

	def get_state(self) -> dict:
		"""Get state dict including sparse storage."""
		return {
			'storage': dict(self._storage),
			'write_count': self._write_count,
			'connections': self.connections.clone(),
		}

	def set_state(self, state: dict) -> None:
		"""Restore state from dict."""
		self._storage = dict(state['storage'])
		self._write_count = state['write_count']
		if 'connections' in state:
			self.connections.copy_(state['connections'])

	@classmethod
	def from_config(cls, config: dict) -> "SparseMemory":
		"""Create a SparseMemory from a configuration dict."""
		return cls(**config)

	def save(self, path: str) -> None:
		"""Save model to file."""
		from wnn.ram.core.serialization import save_model
		save_model(self, path)

	@classmethod
	def load(cls, path: str, device: str = 'cpu') -> "SparseMemory":
		"""Load model from file."""
		from wnn.ram.core.serialization import load_model
		return load_model(path, model_class=cls, device=device)

	# =========================================================================
	# Device helpers
	# =========================================================================

	@property
	def device(self) -> device:
		"""
		Get the device this memory is on.

		Note: SparseMemory uses Python dict storage which is CPU-only.
		The tensor buffers (connections, addresses_bits) can be on GPU
		but the main storage remains on CPU.
		"""
		return self.connections.device

	def cuda(self, device_id: int | None = None) -> "SparseMemory":
		"""
		Move tensor buffers to GPU.

		Warning: SparseMemory uses Python dict storage which remains on CPU.
		Only tensor buffers (connections, addresses_bits) are moved to GPU.
		For full GPU support, use dense Memory instead.
		"""
		import warnings
		warnings.warn(
			"SparseMemory uses dict storage which remains on CPU. "
			"Only tensor buffers are moved to GPU. Use dense Memory for full GPU support.",
			UserWarning,
		)
		if device_id is not None:
			return self.to(device(f"cuda:{device_id}"))
		return self.to(device("cuda"))

	def cpu(self) -> "SparseMemory":
		"""Move tensor buffers to CPU."""
		return self.to(device("cpu"))
