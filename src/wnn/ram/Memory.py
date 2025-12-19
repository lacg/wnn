from wnn.ram.decoders import TransformerDecoder
from wnn.ram.cost import CostCalculator
from wnn.ram.cost import CostCalculatorFactory
from wnn.ram.cost import CostCalculatorType
from wnn.ram.RAMEnums import MemoryVal

from typing import Optional

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

	WORD_SIZE = 62

	def __init__(self, total_input_bits: int, num_neurons: int, n_bits_per_neuron: int, connections: Optional[Tensor] = None, use_hashing: bool = False, hash_size: int = 1024, rng: Optional[int] = None, cost_calculator_type: CostCalculatorType = CostCalculatorType.STOCHASTIC, epochs: int = 1000) -> None:
		super().__init__()

		assert total_input_bits >= 0
		assert num_neurons >= 0
		assert n_bits_per_neuron >= 0

		self.total_input_bits = int(total_input_bits)
		self.num_neurons = int(num_neurons)
		self.n_bits_per_neuron = int(n_bits_per_neuron)
		self.use_hashing = bool(use_hashing)
		self.memory_size = int(hash_size if use_hashing else (1 << n_bits_per_neuron))

		# bit-packing constants
		self.bits_per_cell	= 2
		self.cells_per_word	= self.WORD_SIZE // self.bits_per_cell		# 31 cells per int64, as 1 bit is for sign and 63 // 2 is 31, so WORD_SIZE = 62 and bits per cell is 2 thus cells are 62 // 2 = 31.
		self.words_per_neuron = (self.memory_size + self.cells_per_word - 1) // self.cells_per_word

		self.cost_calculator = CostCalculatorFactory.create(cost_calculator_type, epochs, num_neurons)

		# main storage: [num_neurons, words_per_neuron]
		# initialized to all EMPTY, which we interpret as FALSE (0).
		# EMPTY is explicitly written as MemoryVal.EMPTY (2).
		EMPTY_WORD = 0
		for i in range(self.cells_per_word):
			EMPTY_WORD |= (MemoryVal.EMPTY << (i * self.bits_per_cell))

		self.register_buffer("memory_words", full((self.num_neurons, self.words_per_neuron), EMPTY_WORD, dtype=int64))

		if self.n_bits_per_neuron > 0:
			base = (2 ** arange(self.n_bits_per_neuron - 1, -1, -1, dtype=int64)).unsqueeze(0)		# MSB-first for sanity... 2 => 0010 (MSB), not 0100 (LSB)
			self.register_buffer("binary_addresses", base.repeat(self.num_neurons, 1),)
		else:
			self.register_buffer("binary_addresses", empty(self.num_neurons, 0, dtype=int64),)

		# 3) connections: [num_neurons, n_bits_per_neuron]
		if connections is None:
			self.register_buffer("connections", self._randomize_connections(rng))
		else:
			conn = connections.clone().long()
			assert conn.shape == (self.num_neurons, self.n_bits_per_neuron)
			assert conn.min().item() >= 0 and conn.max().item() < self.total_input_bits
			self.register_buffer("connections", conn)

		# [self.memory_size, self.n_bits_per_neuron]		
		self.register_buffer("addresses_bits", Memory.decode_addresses_bits(self.n_bits_per_neuron, arange(self.memory_size), device("cpu")).to(int8))

	def __repr__(self):
		return (
			f"Memory"
			f"("
			f"neurons={self.num_neurons}, "
			f"bits_per_neuron={self.n_bits_per_neuron}, "
			f"memory_size={self.memory_size}"
			f")"
		)

	def __str__(self):
		lines = []
		lines.append(f"\n=== Memory ===")
		lines.append(f"total_input_bits = {self.total_input_bits}")
		lines.append(f"num_neurons      = {self.num_neurons}")
		lines.append(f"bits_per_neuron  = {self.n_bits_per_neuron}")
		lines.append(f"memory_size      = {self.memory_size}")

		lines.append("\nConnections:")
		for i in range(self.num_neurons):
			row = self.connections[i].tolist()
			lines.append(f"\tneuron {i}: {row}")

		lines.append("\nMemory Matrix:")
		for neuron_index in range(self.num_neurons):
			decoded = " ".join([str(self._read_cells_int(neuron_index, addr)) for addr in range(self.memory_size)])
			if len(decoded) > 256:
				decoded = decoded[:256] + f"... (size {len(decoded)})"
			lines.append(f"\tneuron {neuron_index}: {decoded}")

		lines.append("")  # final newline
		return "\n".join(lines)

	def _randomize_connections(self, rng: Optional[int]) -> Tensor:
		"""
    Guaranteed-coverage connection initializer.

    Goals:
      - Every upstream index in [0, total_input_bits) is used at least once
        across all neurons (global coverage).
      - Each neuron receives `n_bits_per_neuron` *unique* connections.
      - After guaranteeing coverage, we continue sampling from a shuffled pool
        to fill the remainder uniformly.

    Result:
      Tensor[num_neurons, n_bits_per_neuron] of dtype long.
		"""
		def refill_pool():
				nonlocal pool, pool_index
				pool = randperm(self.total_input_bits, device=self.memory_words.device)
				pool_index = 0

		if self.num_neurons == 0 or self.n_bits_per_neuron == 0 or self.total_input_bits == 0:
			# Return empty tensor with correct shape and type
			return empty(self.num_neurons, self.n_bits_per_neuron, dtype=long, device=device("cpu"))

		# ---------
		# STEP 1: Guaranteed coverage bucket
		# ---------
		# First pass: assign each of the T bits exactly once.
		# If we have more neuron slots than T, great — T assignments are done here.
		# If num_neurons * n_bits_per_neuron < total_input_bits, the architecture is underspecified (user error).
		total_slots = self.num_neurons * self.n_bits_per_neuron
		# if total_slots < self.total_input_bits:
		# 		raise RuntimeError(f"Cannot guarantee coverage: num_neurons * n_bits_per_neuron = {total_slots} < total_input_bits = {self.total_input_bits}")

		if rng is not None:
			manual_seed(rng)

		memory_connections = [[None] * self.n_bits_per_neuron for _ in range(self.num_neurons)]

		# --- GLOBAL SHUFFLED BAG OF INPUT BITS ---
		pool = None
		pool_index = filled = 0
		refill_pool()

		for i, conn in enumerate(pool):
			row, col = divmod(i, self.n_bits_per_neuron)
			if filled < total_slots:
				memory_connections[row][col] = int(conn.item())
				filled += 1
			else:
				break

		filled, (row, col) = self.total_input_bits, divmod(self.total_input_bits, self.n_bits_per_neuron)
		while filled < total_slots:
			# If pool is empty, reshuffle
			if pool_index >= self.total_input_bits:
				refill_pool()

			connection = int(pool[pool_index].item())
			pool_index += 1
			if connection not in memory_connections[row]:
				memory_connections[row][col] = connection
				filled += 1
				row, col = divmod(filled, self.n_bits_per_neuron)

		return stack([tensor(neuron_connections, dtype=long, device=self.memory_words.device) for neuron_connections in memory_connections], dim=0)

	@staticmethod
	def decode_address_bits(n_bits: int, address, dev: device) -> Tensor:
		"""
		Decode an integer address into [n_bits] bool bits (MSB-first).
		"""
		shifts = arange(n_bits, device=dev)
		address_tensor = tensor(address, device=dev, dtype=int64) if not isinstance(address, Tensor) else address.to(device=dev, dtype=int64)
		return ((address_tensor >> (n_bits - 1 - shifts)) & 1).to(tbool) # MSB-first for sanity... 2 => 0010 (MSB), not 0100 (LSB)

	@staticmethod
	def decode_addresses_bits(n_bits: int, addresses: Tensor, dev: device) -> Tensor:
		"""
		Decode an integer address into [n_bits] bool bits (MSB-first).
		"""
		shifts = arange(n_bits, device=dev).unsqueeze(0)							# [1, self.n_bits_per_neuron]
		address = addresses.to(int64).unsqueeze(1)										# [self.memory_size, 1]
		return ((address >> (n_bits - 1 - shifts)) & 1).to(tbool)			# [self.memory_size, self.n_bits_per_neuron]
																																	# MSB-first for sanity... 2 => 0010 (MSB), not 0100 (LSB)
																															
	# ------------------------------------------------------------------
	# Bit-packed cell helpers
	# ------------------------------------------------------------------

	def _cell_coords(self, address: Tensor) -> tuple[Tensor, Tensor]:
		"""
		Given addresses (int64), return:
			word_index:  address // cells_per_word
			bit_shift: (address % cells_per_word) * bits_per_cell
		"""
		word_index	= address // self.cells_per_word
		cell_index	= address % self.cells_per_word
		bit_shift	= cell_index * self.bits_per_cell
		return word_index, bit_shift

	def _cell_coords_int(self, address: int) -> tuple[int, int]:
		"""
		Given addresses (int64), return:
			word_index:  address // cells_per_word
			bit_shift: (address % cells_per_word) * bits_per_cell
		"""
		word_index, cell_index	= divmod(address, self.cells_per_word)
		bit_shift	= cell_index * self.bits_per_cell
		return word_index, bit_shift

	def _explore_batch_raw(self, neuron_indices: Tensor, addresses: Tensor, encoded_vals: Tensor) -> None:
		"""
		Vectorized write of K **already-encoded** values in {0,1,2}.
		encoded_vals MUST be int64 and already contain the MemoryVal numbers.
		"""
		word_index, bit_shift = self._cell_coords(addresses)

		# Gather the words
		words = self.memory_words[neuron_indices, word_index]

		# Build masks
		mask        = (3 << bit_shift)     # which cell to update
		clear_mask  = ~mask
		val_shifted = (encoded_vals & 0b11) << bit_shift

		# Perform packed write
		new_words = (words & clear_mask) | val_shifted
		self.memory_words[neuron_indices, word_index] = new_words

	# In Memory class (near other helpers like get_memory, get_memories_for_bits)

	def _get_memory_batch_raw(self, neuron_indices: Tensor, addresses: Tensor) -> Tensor:
		"""
		Vectorized raw read of K memory cells.

		neuron_indices: [K]
		addresses:      [K]
		Returns:        [K] int64 in {0,1,2}
		"""
		# Determine word and bit-shift for each address
		word_index, bit_shift = self._cell_coords(addresses)

		# Fetch words
		words = self.memory_words[neuron_indices, word_index]

		# Extract the 2-bit cell
		return (words >> bit_shift) & 0b11	  											# NOTE: bit-packed memory cells are positional, NOT MSB/LSB semantic

	def _read_cells(self, neuron_indices: Tensor, addresses: Tensor) -> int:
		"""
		Read bit-packed cells for given neurons & addresses.

		neuron_indices: [flat] (int64) indices in [0, num_neurons)
		addresses:      [flat] (int64) addresses in [0, memory_size)

		Returns:
			values: [flat] int64 in {0,1,2}  (FALSE, TRUE, EMPTY)
		"""
		word_index, bit_shift = self._cell_coords(addresses)
		words = self.memory_words[neuron_indices, word_index]
		return (words >> bit_shift) & 0b11		 											# NOTE: bit-packed memory cells are positional, NOT MSB/LSB semantic

	def _read_cells_int(self, neuron_index: int, address: int) -> int:
		"""
		Read bit-packed cells for given neurons & addresses.

		neuron_indices: [flat] (int64) indices in [0, num_neurons)
		addresses:      [flat] (int64) addresses in [0, memory_size)

		Returns:
			values: [flat] int64 in {0,1,2}  (FALSE, TRUE, EMPTY)
		"""
		word_index, bit_shift = self._cell_coords_int(address)
		words = self.memory_words[neuron_index, word_index]
		values = (words >> bit_shift) & 0b11										# NOTE: bit-packed memory cells are positional, NOT MSB/LSB semantic
		return int(values.item())

	def _write_cells_int(self, neuron_index: int, address: int, value: int) -> None:
		"""
		Write bit-packed cells for given neurons & addresses.

		values: int64 in {0,1,2} (FALSE, TRUE, EMPTY)
		"""
		word_index, bit_shift = self._cell_coords_int(address)

		# Gather the words we will modify
		words = self.memory_words[neuron_index, word_index]

		# For each element:
		#	clear mask: ~(0b11 << shift)
		#	write: (words & clear_mask) | (value << shift)
		mask = (3 << bit_shift)
		clear_mask = (~mask) & ((1 << 64) - 1)   # ensure 64-bit width
		value_shifted	= (value & 0b11) << bit_shift

		new_words = (words & clear_mask) | value_shifted
		self.memory_words[neuron_index, word_index] = new_words

	def commit(self, input_bits: Tensor, target_bits: Tensor, allow_override: bool = False) -> bool:
		"""
		Finalize semantics:
		This input MUST map to this output.
		Direct write: set memory for each (sample, neuron) to target bit (False/True).
		target_bits must be bool or {0,1}.

		COMMIT:
		- Writes are final
		- Overwrites any previous value
		- Freezes DFA transitions

		Return:					True if any change happened, False otherwise.
		"""
		input_bits = input_bits.to(uint8)
		if input_bits.ndim == 1:
			input_bits = input_bits.unsqueeze(0)

		target_bits = target_bits.to(tbool)
		if target_bits.ndim == 1:
				target_bits = target_bits.unsqueeze(0)

		addresses = self.get_addresses(input_bits)[0]   # [N]
		neuron_index = arange(self.num_neurons, device=input_bits.device, dtype=long)

		# EMPTY-first or override-aware write
		return self.explore_batch(neuron_index, addresses, target_bits[0], allow_override)

	def explore(self, neuron_index: int, address: int, bit: bool, allow_override: bool = False) -> None:
		"""
		Explore a hypothesis:
		Write ONLY if the cell is EMPTY or already compatible.

		This does NOT finalize semantics, unless when allow_override = True.

		EXPLORE:
		- Writes are provisional
		- Only EMPTY or compatible cells may be written
		- Used during EDRA and constraint solving
		"""
		if self.num_neurons > 0:
			current_memory = self.get_memory(neuron_index, address)
			desired = MemoryVal.TRUE if bit else MemoryVal.FALSE
			if current_memory == MemoryVal.EMPTY or (allow_override and desired != current_memory):
				self._write_cells_int(neuron_index, address, desired)

	def explore_batch(self, neuron_indices: Tensor, addresses: Tensor, bits: Tensor, allow_override: bool = False) -> bool:
		"""
		Vectorized explore operation.

		neuron_indices: [K]
		addresses:      [K]
		bits:           [K] bool or {0,1}

		Return:					True if any change happened, False otherwise.
		"""
		if self.num_neurons == 0:
			return False

		if bits.dtype != tbool:
			bits = bits.to(tbool)

		device = self.memory_words.device  # or self.memory.device depending on your impl

		neuron_indices = neuron_indices.to(device=device, dtype=long)
		addresses      = addresses.to(device=device, dtype=long)

		# Encode bits → {FALSE, TRUE}
		# (EMPTY is only used when a cell was uninitialized; we don't "write EMPTY")
		encoded = full(bits.shape, MemoryVal.FALSE, dtype=int64, device=device)
		encoded = encoded.masked_fill(bits, MemoryVal.TRUE)

		current = self._get_memory_batch_raw(neuron_indices, addresses)
		mask_write = (current != encoded) if allow_override else (current == MemoryVal.EMPTY)

		if bool(mask_write.any()):
			# Finally write all at once.
			self._explore_batch_raw(neuron_indices[mask_write], addresses[mask_write], encoded[mask_write])
			return True
		return False

	def flip(self, neuron_index: int, input_bits: Tensor) -> None:
		"""
		Flip one memory cell for given neuron, based on input_bits (batch size 1).

		- FALSE -> TRUE
		- TRUE  -> FALSE
		- EMPTY -> TRUE (arbitrary but consistent choice)
		"""
		if self.num_neurons == 0:
			return

		addresses = self.get_addresses(input_bits)  # [B, N]
		addr = int(addresses[0, neuron_index].item())
		new_val = MemoryVal.FALSE if self._read_cells_int(neuron_index, addr) == MemoryVal.TRUE else MemoryVal.TRUE

		self._write_cells_int(neuron_index, addr, new_val)

	def forward(self, input_bits: Tensor) -> Tensor:
		"""
		Forward lookup: returns boolean outputs [B, num_neurons].
		TRUE cells (1) → True
		FALSE and EMPTY → False
		"""
		return self.get_memories_for_bits(input_bits) == MemoryVal.TRUE

	def get_addresses(self, input_bits: Tensor) -> Tensor:
		"""
		Compute integer addresses [batch_size, num_neurons] for given input bits.
		input_bits: [B, total_input_bits], bool or {0,1}
		"""
		if self.num_neurons == 0:
			return empty(input_bits.shape[0], 0, dtype=int64, device=input_bits.device)

		# Normalize to [B, total_input_bits]
		if input_bits.ndim == 1:
			input_bits = input_bits.unsqueeze(0)
		assert input_bits.ndim == 2, f"input_bits must be 1D or 2D, got {input_bits.shape}"
		assert input_bits.shape[1] == self.total_input_bits, (f"input_bits has {input_bits.shape[1]} bits but memory expects {self.total_input_bits}")

		if input_bits.dtype == tbool:
			input_bits64 = input_bits.to(int64)
		else:
			input_bits64 = (input_bits != 0).to(int64)

		# input_bits64[:, connections] -> [B, N, k]
		gathered = input_bits64[:, self.connections]  # advanced indexing

		# compute addresses → [B, N]
		addresses = (gathered.to(int64) * self.binary_addresses.unsqueeze(0)).sum(-1)
		if self.use_hashing:
			addresses = addresses % self.memory_size
		return addresses.long()

	def get_memories_for_bits(self, input_bits: Tensor) -> Tensor:
		"""
		Used by EDRA: directly get specific memory cell.
		returns ints
		"""
		if self.num_neurons == 0:
			return empty(input_bits.shape[0], 0, dtype=tbool, device=input_bits.device)

		addresses = self.get_addresses(input_bits)  # [B, N]
		batch_size = addresses.shape[0]

		neuron_indices = arange(self.num_neurons, device=input_bits.device).unsqueeze(0).expand(batch_size, -1).reshape(-1)
		addr_flat = addresses.reshape(-1)

		return self._read_cells(neuron_indices, addr_flat).reshape(batch_size, self.num_neurons)

	def get_memory(self, neuron_index: int, address: int) -> int:
		"""
		Used by EDRA: directly get specific memory cell.
		returns a MemoryVal.{x}
		"""
		return self._read_cells_int(neuron_index, address)

	def get_memory_row(self, neuron_index: int) -> Tensor:
		"""
		Vectorized read of all memory cells for a single neuron.
		Returns: Tensor[memory_size] with values in {MemoryVal.FALSE, MemoryVal.TRUE, MemoryVal.EMPTY}
		"""
		device = self.memory_words.device

		# addresses: 0..memory_size-1
		addresses = arange(self.memory_size, dtype=int64, device=device)

		# same neuron index for all addresses
		neuron_indices = full((self.memory_size,), int(neuron_index), dtype=int64, device=device,)

		# use bit-packed vectorized reader
		return self._read_cells(neuron_indices, addresses)

	def solve_constraints(self, input_bits: Tensor, target_bits: Tensor, allow_override: bool = False, n_immutable_bits: int = 0, topk_per_candidate: int = 4) -> Optional[Tensor]:
		"""
    Vectorized global constraint solver.
		Find an alternative upstream input bits vector that would produce target_bits.
		No memory is modified.
		Returns:
			- Tensor[new_input_bits] or
			- None if no admissible solution exists
		"""
    # ---- Normalize shapes ----
		if input_bits.ndim == 2:
				assert input_bits.shape[0] == 1
				input_bits = input_bits[0]
		assert input_bits.ndim == 1 and input_bits.numel() == self.total_input_bits
		assert target_bits.ndim == 2 and target_bits.shape[0] == 1
		input_bits = input_bits.to(tbool)
		target_bits = target_bits.to(tbool)

    # ---- Constants ----
		MAX_BEAM_WIDTH, CONFLICT_COST, EMPTY_COST, HAMMING_COST = 32, 20, 10, 1
		mutable_size = 2 ** (self.total_input_bits - n_immutable_bits)
		beam_width = min(MAX_BEAM_WIDTH, mutable_size, max(4 * self.num_neurons, 1))
		k_top = min(topk_per_candidate, self.memory_size)

		# ---- Address bits [self.memory_size, self.n_bits_per_neuron] ----
		memory_rows = stack([self.get_memory_row(neuron_index) for neuron_index in range(self.num_neurons)], dim=0)		# [self.num_neurons, self.memory_size]
		desired_bits = target_bits[0].to(tbool)																		# [self.num_neurons]
		desired_memories = where(desired_bits, MemoryVal.TRUE, MemoryVal.FALSE).unsqueeze(1)				# [self.num_neurons, 1]

		# --- Conflict / empty masks ---
		conflict = (memory_rows != MemoryVal.EMPTY) & (memory_rows != desired_memories)
		empty    = (memory_rows == MemoryVal.EMPTY)

		valid_memories = ones_like(conflict, dtype=tbool) if allow_override else ~conflict

    # Candidate upstream assignments:
    # -1 = unknown, 0/1 = fixed
		candidate_bits = full((beam_width, self.total_input_bits), -1, dtype=int8, device=input_bits.device)
		candidate_cost = zeros(beam_width, device=input_bits.device, dtype=float64)

		# Seed: immutable bits fixed from input_bits
		if n_immutable_bits > 0:
			candidate_bits[:, :n_immutable_bits] = input_bits[:n_immutable_bits].to(int8)

		# Keep only first candidate active initially
		candidate_cost[1:] = float("inf")

		# -------------------------------------------------------
		# Address compatibility with candidate bits
		# -------------------------------------------------------

		# candidate_bits: [beam_width, total_input_bits]
		# connections:    [self.num_neurons, self.n_bits_per_neuron]

		# Project candidate bits → [beam_width, self.num_neurons, self.n_bits_per_neuron]
		candidate_proj = candidate_bits[:, self.connections]   # [beam_width, self.num_neurons, self.n_bits_per_neuron]
		fixed_mask     = candidate_proj != -1             # [beam_width, self.num_neurons, self.n_bits_per_neuron]

		# Address bits: [self.memory_size, self.n_bits_per_neuron] → [1, 1, self.memory_size, self.n_bits_per_neuron]
		addresses_bits = self.addresses_bits.unsqueeze(0).unsqueeze(0)

		# Expand to [beam_width, self.num_neurons, self.memory_size, self.n_bits_per_neuron]
		candidate_proj = candidate_proj.unsqueeze(2)
		fixed_mask     = fixed_mask.unsqueeze(2)

		mismatch = fixed_mask & (addresses_bits != candidate_proj)
		compatible = ~mismatch.any(dim=3)                 # [beam_width, self.num_neurons, self.memory_size]

		# -------------------------------------------------------
		# Hamming cost (relative to current input)
		# -------------------------------------------------------

		current = input_bits[self.connections].to(int8)        # [self.num_neurons, self.n_bits_per_neuron]
		hamming = (self.addresses_bits.unsqueeze(0) != current.unsqueeze(1)).sum(dim=2)  # [self.num_neurons, self.memory_size]

		# -------------------------------------------------------
		# Per-neuron per-address cost
		# -------------------------------------------------------

		per_neuron_cost = (CONFLICT_COST * conflict.to(float64) + EMPTY_COST * empty.to(float64) + HAMMING_COST * hamming.to(float64))	# [self.num_neurons, self.memory_size]

		# -------------------------------------------------------
		# Combine all neurons → total cost
		# -------------------------------------------------------

		# Expand to [beam_width, self.num_neurons, self.memory_size]
		cost = per_neuron_cost.unsqueeze(0).expand(beam_width, -1, -1)

		cost = cost.masked_fill(~valid_memories.unsqueeze(0), float("inf"))
		cost = cost.masked_fill(~compatible, float("inf"))

		# Sum neuron costs → [beam_width, self.memory_size]
		cost = cost.sum(dim=1)

		# Add candidate cost
		cost = cost + candidate_cost.unsqueeze(1)

		# -------------------------------------------------------
		# Beam expansion
		# -------------------------------------------------------

		topk_cost, topk_index = cost.topk(k=k_top, dim=1, largest=False)

		new_bits = candidate_bits.unsqueeze(1).expand(-1, k_top, -1).reshape(-1, self.total_input_bits).clone()
		new_cost = topk_cost.reshape(-1)

		# chosen addresses per expanded candidate
		chosen_addresses_index = topk_index.reshape(-1)                 # [beam_width * k_top]

		# bits of the chosen address
		chosen_address_bits = self.addresses_bits[chosen_addresses_index]  # [beam_width * k_top, n_bits_per_neuron]

		# Merge chosen address bits
		for neuron_index in range(self.num_neurons):
			neuron_connection = self.connections[neuron_index]		# [n_bits_per_neuron]

			existing = new_bits[:, neuron_connection]							# [beam_width*k_top, n_bits_per_neuron]

			conflict_merge = (existing != -1) & (existing != chosen_address_bits)
			new_cost = new_cost.masked_fill(conflict_merge.any(dim=1), float("inf"))

			merged = where(existing != -1, existing, chosen_address_bits)
			new_bits[:, neuron_connection] = merged
		# -------------------------------------------------------
		# Prune beam
		# -------------------------------------------------------

		best_cost, best_index = new_cost.topk(k=beam_width, largest=False)
		candidate_bits = new_bits[best_index]
		candidate_cost = best_cost

		# ---- Select best candidate ----
		if not isfinite(candidate_cost).any():
			return None

		best = candidate_bits[self.cost_calculator.calculate_index(candidate_cost)]

		# Fill unconstrained bits from original input
		best = where(best == -1, input_bits.to(int8), best)
		return best.to(tbool)

