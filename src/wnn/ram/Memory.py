from wnn.ram.RAMEnums import MemoryVal

from typing import Optional

from torch import arange
from torch import argmax
from torch import bool as tbool
from torch import cat
from torch import device
from torch import empty
from torch import full
from torch import int64
from torch import long
from torch import manual_seed
from torch import randint
from torch import randperm
from torch import tensor
from torch import Tensor
from torch import stack
from torch import where
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

	def __init__(self, total_input_bits: int, num_neurons: int, n_bits_per_neuron: int, connections: Optional[Tensor] = None, use_hashing: bool = False, hash_size: int = 1024, rng: Optional[int] = None) -> None:
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
		self.cells_per_word	= self.WORD_SIZE // self.bits_per_cell		# 32 cells per int64
		self.words_per_neuron = (self.memory_size + self.cells_per_word - 1) // self.cells_per_word

		# main storage: [num_neurons, words_per_neuron]
		# initialized to all EMPTY, which we interpret as FALSE (0).
		# EMPTY is explicitly written as MemoryVal.EMPTY (2).
		EMPTY_WORD = 0
		for i in range(self.cells_per_word):
			EMPTY_WORD |= (MemoryVal.EMPTY << (i * self.bits_per_cell))

		self.register_buffer("memory_words", full((self.num_neurons, self.words_per_neuron), EMPTY_WORD, dtype=int64))

		if self.n_bits_per_neuron > 0:
			base = (2 ** arange(self.n_bits_per_neuron, dtype=int64)).unsqueeze(0)
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

	# ------------------------------------------------------------------
	# Bit-packed cell helpers
	# ------------------------------------------------------------------

	def _cell_coords(self, address: Tensor) -> tuple[Tensor, Tensor]:
		"""
		Given addresses (int64), return:
			word_idx:  address // cells_per_word
			bit_shift: (address % cells_per_word) * bits_per_cell
		"""
		word_idx	= address // self.cells_per_word
		cell_idx	= address % self.cells_per_word
		bit_shift	= cell_idx * self.bits_per_cell
		return word_idx, bit_shift

	def _cell_coords_int(self, address: int) -> tuple[int, int]:
		"""
		Given addresses (int64), return:
			word_idx:  address // cells_per_word
			bit_shift: (address % cells_per_word) * bits_per_cell
		"""
		word_idx, cell_idx	= divmod(address, self.cells_per_word)
		bit_shift	= cell_idx * self.bits_per_cell
		return word_idx, bit_shift

	def _read_cells(self, neuron_indices: Tensor, addresses: Tensor) -> int:
		"""
		Read bit-packed cells for given neurons & addresses.

		neuron_indices: [flat] (int64) indices in [0, num_neurons)
		addresses:      [flat] (int64) addresses in [0, memory_size)

		Returns:
			values: [flat] int64 in {0,1,2}  (FALSE, TRUE, EMPTY)
		"""
		word_idx, bit_shift = self._cell_coords(addresses)
		words = self.memory_words[neuron_indices, word_idx]
		return (words >> bit_shift) & 0b11

	def _read_cells_int(self, neuron_index: int, address: int) -> int:
		"""
		Read bit-packed cells for given neurons & addresses.

		neuron_indices: [flat] (int64) indices in [0, num_neurons)
		addresses:      [flat] (int64) addresses in [0, memory_size)

		Returns:
			values: [flat] int64 in {0,1,2}  (FALSE, TRUE, EMPTY)
		"""
		word_idx, bit_shift = self._cell_coords_int(address)
		words = self.memory_words[neuron_index, word_idx]
		values = (words >> bit_shift) & 0b11
		return int(values.item())

	def _write_cells(self, neuron_indices: Tensor, addresses: Tensor, value: Tensor) -> None:
		"""
		Write bit-packed cells for given neurons & addresses.

		values: int64 in {0,1,2} (FALSE, TRUE, EMPTY)
		"""
		word_idx, bit_shift = self._cell_coords(addresses)

		# Gather the words we will modify
		words = self.memory_words[neuron_indices, word_idx]

		# For each element:
		#	clear mask: ~(0b11 << shift)
		#	write: (words & clear_mask) | (value << shift)
		mask = (3 << bit_shift).to(int64)
		clear_mask = (~mask) & ((1 << 64) - 1)   # ensure 64-bit width
		value_shifted	= (value & 0b11) << bit_shift

		new_words = (words & clear_mask) | value_shifted
		self.memory_words[neuron_indices, word_idx] = new_words

	def _write_cells_int(self, neuron_index: int, address: int, value: int) -> None:
		"""
		Write bit-packed cells for given neurons & addresses.

		values: int64 in {0,1,2} (FALSE, TRUE, EMPTY)
		"""
		word_idx, bit_shift = self._cell_coords_int(address)

		# Gather the words we will modify
		words = self.memory_words[neuron_index, word_idx]

		# For each element:
		#	clear mask: ~(0b11 << shift)
		#	write: (words & clear_mask) | (value << shift)
		mask = (3 << bit_shift)
		clear_mask = (~mask) & ((1 << 64) - 1)   # ensure 64-bit width
		value_shifted	= (value & 0b11) << bit_shift

		new_words = (words & clear_mask) | value_shifted
		self.memory_words[neuron_index, word_idx] = new_words

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

	def get_addresses(self, input_bits: Tensor) -> Tensor:
		"""
		Compute integer addresses [batch_size, num_neurons] for given input bits.
		input_bits: [B, total_input_bits], bool or {0,1}
		"""
		if self.num_neurons == 0:
			return empty(input_bits.shape[0], 0, dtype=int64, device=input_bits.device)

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

	def get_address_for_neuron(self, neuron_index: int, input_bits: Tensor) -> int:
		input_bits64 = input_bits.to(int64) if input_bits.dtype == tbool else (input_bits != 0).to(int64)

		weights = self.binary_addresses[neuron_index].unsqueeze(0)  # [1, k]
		addresses = (input_bits64 * weights).sum(-1)
		if self.use_hashing:
			addresses = addresses % self.memory_size
		return int(addresses[0].item())

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

	# In Memory class (near other helpers like get_memory, get_memories_for_bits)

	def _get_memory_batch_raw(self, neuron_indices: Tensor, addresses: Tensor) -> Tensor:
		"""
		Vectorized raw read of K memory cells.

		neuron_indices: [K]
		addresses:      [K]
		Returns:        [K] int64 in {0,1,2}
		"""
		# Determine word and bit-shift for each address
		word_idx, bit_shift = self._cell_coords(addresses)

		# Fetch words
		words = self.memory_words[neuron_indices, word_idx]

		# Extract the 2-bit cell
		return (words >> bit_shift) & 0b11

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

	def get_memory_row_raw(self, neuron_index: int) -> Tensor:
		"""
		Vectorized read of ALL memory cells for a single neuron.
		Returns: Tensor[memory_size] int64 values in {0,1,2}
		"""
		device = self.memory_words.device

		# addresses: 0...memory_size-1
		addresses = arange(self.memory_size, dtype=int64, device=device)
		neuron_vec = full((self.memory_size,), neuron_index, dtype=int64, device=device)

		return self._get_memory_batch_raw(neuron_vec, addresses)

	def forward(self, input_bits: Tensor) -> Tensor:
		"""
		Forward lookup: returns boolean outputs [B, num_neurons].
		TRUE cells (1) → True
		FALSE and EMPTY → False
		"""
		return self.get_memories_for_bits(input_bits) == MemoryVal.TRUE

	def select_connection(self, neuron_index: int, use_high_impact: bool = True) -> int:
		"""
		Select one contributing input-bit index for this neuron.
		If use_high_impact=True, chooses the connection with the largest binary weight.
		Otherwise chooses a random connection.
		Returns: integer index into the layer's input bit vector.
		"""
		if self.n_bits_per_neuron == 0:
			return 0

		if use_high_impact:
			impacts = self.binary_addresses[neuron_index]  # [k]
			pos = int(argmax(impacts).item())
		else:
			pos = int(randint(0, self.n_bits_per_neuron, (1,), device=self.memory_words.device).item())

		return int(self.connections[neuron_index, pos].item())

	def set_memory(self, neuron_index: int, address: int, bit: bool, allow_override: bool = False) -> None:
		"""
		Used by EDRA: directly overwrite specific memory cell (for one neuron & address).
		bit = False/True
		"""
		if self.num_neurons > 0:
			current_memory = self.get_memory(neuron_index, address)
			desired = MemoryVal.TRUE if bit else MemoryVal.FALSE
			if current_memory == MemoryVal.EMPTY or (allow_override and desired != current_memory):
				self._write_cells_int(neuron_index, address, desired)

	def set_memory_batch(self, neuron_indices: Tensor, addresses: Tensor, bits: Tensor, allow_override: bool = False) -> None:
		"""
		Vectorized version of set_memory for K cells.
		neuron_indices: [K]
		addresses:      [K]
		bits:           [K] bool or {0,1]
		"""
		if self.num_neurons == 0:
			return

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
			self._set_memory_batch_raw(neuron_indices[mask_write], addresses[mask_write], encoded[mask_write])

	def _set_memory_batch_raw(self, neuron_indices: Tensor, addresses: Tensor, encoded_vals: Tensor) -> None:
		"""
		Vectorized write of K **already-encoded** values in {0,1,2}.
		encoded_vals MUST be int64 and already contain the MemoryVal numbers.
		"""
		word_idx, bit_shift = self._cell_coords(addresses)

		# Gather the words
		words = self.memory_words[neuron_indices, word_idx]

		# Build masks
		mask        = (3 << bit_shift)     # which cell to update
		clear_mask  = ~mask
		val_shifted = (encoded_vals & 0b11) << bit_shift

		# Perform packed write
		new_words = (words & clear_mask) | val_shifted
		self.memory_words[neuron_indices, word_idx] = new_words

	def set_memory_row_raw(self, neuron_index: int, values: Tensor) -> None:
		"""
		Vectorized write of ALL memory cells for a neuron.
		values must be shape [memory_size] and contain {0,1,2}
		"""
		device = self.memory_words.device

		addresses = arange(self.memory_size, dtype=int64, device=device)
		neuron_vec = full((self.memory_size,), neuron_index, dtype=int64, device=device)

		self._set_memory_batch_raw(neuron_vec, addresses, values)

	def train_write(self, input_bits: Tensor, target_bits: Tensor) -> None:
		"""
		Direct write: set memory for each (sample, neuron) to target bit (False/True).
		target_bits must be bool or {0,1}.
		"""
		norm = lambda bits: bits.to(int64) if bits.dtype == tbool else (bits != 0).to(int64)
		input_bits64 = norm(input_bits)
		target_bits64 = norm(target_bits)

		addresses = self.get_addresses(input_bits64)  # [B, N]
		batch_size = addresses.shape[0]
		cell_vals = where(target_bits64 == 1, MemoryVal.TRUE, MemoryVal.FALSE)
		neuron_indices = arange(self.num_neurons, device=input_bits.device).unsqueeze(0).expand(batch_size, -1).reshape(-1)

		self._write_cells(neuron_indices, addresses.reshape(-1), cell_vals.reshape(-1))

