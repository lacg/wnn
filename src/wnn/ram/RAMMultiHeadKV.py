"""
RAM Multi-Head Key-Value Memory

Combines ensemble multi-head architecture with explicit KV operations:
- Multiple independent KV memory heads
- Key-based routing to heads
- Explicit Write(key, value) and Read(query) operations
- Query detection via zero-value convention

Architecture:
  Input: [key_bits | value_bits]

  If value_bits != 0 (WRITE):
    head_idx = route(key_bits)
    heads[head_idx].store(key_bits, value_bits)

  If value_bits == 0 (READ/QUERY):
    head_idx = route(key_bits)
    output = heads[head_idx].retrieve(key_bits)
"""

from wnn.ram.RAMLayer import RAMLayer
from wnn.ram.encoders_decoders import OutputMode
from wnn.ram.encoders_decoders import TransformerDecoder
from wnn.ram.encoders_decoders import TransformerDecoderFactory

from typing import Optional

from torch import cat
from torch import uint8
from torch import zeros
from torch import Tensor
from torch.nn import Module
from torch.nn import ModuleList


class RAMMultiHeadKV(Module):
	"""
	Multi-head Key-Value memory with explicit read/write operations.

	Each head is an independent KV store:
	- Key bits route to a specific head
	- Each head has memory (RAM layer) for storing key→value mappings
	- Zero-value input signals a query/read operation

	This is transformer-style attention with:
	- Hard routing (discrete head selection)
	- Explicit memory (RAM-based storage)
	- No soft attention weights
	"""

	def __init__(
		self,
		num_heads: int,
		k_bits: int,  # Key bits for routing AND addressing
		v_bits: int,  # Value bits to store/retrieve
		neurons_per_head: int = 8,  # Memory capacity per head
		n_bits_per_neuron: int | None = None,
		output_mode: OutputMode = OutputMode.RAW,
		use_hashing: bool = False,
		hash_size: int = 1024,
		rng: int | None = None,
	):
		"""
		Args:
			num_heads: Number of KV memory heads
			k_bits: Bits for key (used for routing AND memory addressing)
			v_bits: Bits for value storage/retrieval
			neurons_per_head: RAM neurons per head (memory capacity)
			n_bits_per_neuron: Connections per neuron (auto if None)
			output_mode: How to decode output (TOKEN, RAW, etc.)
			use_hashing: Use hash-based addressing
			hash_size: Hash table size
			rng: Random seed
		"""
		super().__init__()

		self.num_heads = num_heads
		self.k_bits = k_bits
		self.v_bits = v_bits
		self.neurons_per_head = neurons_per_head
		self.input_bits = k_bits + v_bits  # Full window size
		self.output_mode = output_mode

		# Auto-calculate connectivity
		if n_bits_per_neuron is None:
			# Memory neurons see key bits for addressing
			n_bits_per_neuron = min(k_bits + neurons_per_head, 12)

		# Validate routing
		max_addressable = 2 ** k_bits
		if max_addressable < num_heads:
			print(f"[MultiHeadKV] Warning: k_bits={k_bits} addresses {max_addressable} < {num_heads} heads")

		# Create decoder for output
		self.decoder: TransformerDecoder = TransformerDecoderFactory.create(output_mode, v_bits)

		# Each head has:
		# 1. State layer: stores key→internal_state mappings
		# 2. Output layer: maps internal_state→value
		self.state_layers = ModuleList([
			RAMLayer(
				total_input_bits=k_bits + neurons_per_head,  # [key, prev_state]
				num_neurons=neurons_per_head,
				n_bits_per_neuron=n_bits_per_neuron,
				use_hashing=use_hashing,
				hash_size=hash_size,
				rng=rng + i * 2 if rng is not None else None,
			)
			for i in range(num_heads)
		])

		self.output_layers = ModuleList([
			RAMLayer(
				total_input_bits=neurons_per_head,  # [state]
				num_neurons=v_bits,
				n_bits_per_neuron=min(neurons_per_head, v_bits),
				use_hashing=use_hashing,
				hash_size=hash_size,
				rng=rng + i * 2 + 1 if rng is not None else None,
			)
			for i in range(num_heads)
		])

		# Per-head recurrent state
		self.head_states: list[Tensor | None] = [None] * num_heads

		print(f"[MultiHeadKV] Created: {num_heads} heads, k={k_bits}, v={v_bits}, neurons/head={neurons_per_head}")

	def _reset_states(self, device=None) -> None:
		"""Reset all head states to zeros."""
		if device is None:
			device = self.state_layers[0].memory.memory_words.device
		for i in range(self.num_heads):
			self.head_states[i] = zeros(1, self.neurons_per_head, dtype=uint8, device=device)

	def _extract_key(self, window_bits: Tensor) -> tuple[Tensor, int]:
		"""
		Extract key bits and compute head index.

		Args:
			window_bits: [1, k_bits + v_bits] or [k_bits + v_bits]

		Returns:
			key_bits: [k_bits] tensor
			head_idx: int index of routed head
		"""
		if window_bits.ndim == 2:
			window_bits = window_bits.squeeze(0)

		key_bits = window_bits[:self.k_bits]

		# Convert to head index
		key_value = 0
		for bit in key_bits:
			key_value = (key_value << 1) | int(bit)

		head_idx = key_value % self.num_heads
		return key_bits, head_idx

	def _extract_value(self, window_bits: Tensor) -> Tensor:
		"""Extract value bits from window."""
		if window_bits.ndim == 2:
			window_bits = window_bits.squeeze(0)
		return window_bits[self.k_bits:]

	def is_query(self, window_bits: Tensor) -> bool:
		"""Check if this is a query (read) operation. Query = all value bits are zero."""
		value_bits = self._extract_value(window_bits)
		return bool((value_bits == 0).all())

	def _forward_head(self, head_idx: int, key_bits: Tensor, is_write: bool) -> Tensor:
		"""
		Forward pass through a single head.

		Args:
			head_idx: Which head to use
			key_bits: Key for addressing [k_bits]
			is_write: Whether to update state (write) or just read (query)

		Returns:
			Output value bits [v_bits]
		"""
		# Ensure key_bits is 2D
		if key_bits.ndim == 1:
			key_bits = key_bits.unsqueeze(0)

		# Get current state for this head
		if self.head_states[head_idx] is None:
			self._reset_states(device=key_bits.device)

		state = self.head_states[head_idx]

		# State layer: [key, prev_state] → new_state
		state_input = cat([key_bits, state], dim=1)
		new_state = self.state_layers[head_idx](state_input)

		# Update state only for writes
		if is_write:
			self.head_states[head_idx] = new_state.detach().clone()

		# Output layer: state → value
		output = self.output_layers[head_idx](new_state)

		return output.squeeze(0)

	def forward(self, input_bits: Tensor) -> str:
		"""
		Process a sequence of KV operations.

		For each window:
		- If value != 0: WRITE (store value at key)
		- If value == 0: READ (retrieve value for key)

		Args:
			input_bits: [total_bits] concatenated windows

		Returns:
			Decoded output from last query
		"""
		input_bits = self._normalize_bits(input_bits)
		self._reset_states(device=input_bits.device)

		windows = self._make_windows(input_bits)
		last_output = None

		for window in windows:
			key_bits, head_idx = self._extract_key(window)
			is_write = not self.is_query(window)

			output = self._forward_head(head_idx, key_bits, is_write)

			if not is_write:  # Query - remember output
				last_output = output

		if last_output is None:
			# No queries, return from last window anyway
			last_output = output

		return self.decoder.decode(last_output.unsqueeze(0))

	def _normalize_bits(self, bits: Tensor) -> Tensor:
		"""Ensure dtype uint8."""
		if bits.dtype != uint8:
			bits = bits.to(uint8)
		return bits

	def _make_windows(self, input_bits: Tensor) -> list[Tensor]:
		"""Split input into windows of input_bits size."""
		if input_bits.ndim == 2:
			seq = input_bits.squeeze(0)
		else:
			seq = input_bits

		windows = []
		for i in range(0, seq.numel(), self.input_bits):
			chunk = seq[i:i + self.input_bits]
			if chunk.numel() < self.input_bits:
				pad = zeros(self.input_bits - chunk.numel(), dtype=chunk.dtype, device=chunk.device)
				chunk = cat([chunk, pad], dim=0)
			windows.append(chunk.unsqueeze(0))

		return windows

	def train(self, windows: list[Tensor], targets: str | list[str] | None = None) -> None:
		"""
		Train the KV memory on a sequence of operations.

		For WRITE operations: Learn to store the value
		For QUERY operations: Learn to retrieve the correct target

		Args:
			windows: List of [1, k_bits + v_bits] tensors
			targets: Target values for QUERY operations (one per query)
		"""
		if len(windows) == 0:
			return

		# Normalize
		windows = [self._normalize_bits(w) for w in windows]

		# Convert targets
		if targets is None:
			target_list = []
		elif isinstance(targets, str):
			target_list = list(targets)
		else:
			target_list = list(targets)

		# Encode targets
		encoded_targets = [self.decoder.encode(t) for t in target_list] if target_list else []

		# Reset states
		self._reset_states(device=windows[0].device)

		# Collect contexts for all windows
		contexts = []
		query_idx = 0

		for t, window in enumerate(windows):
			key_bits, head_idx = self._extract_key(window)
			value_bits = self._extract_value(window)
			is_write = not self.is_query(window)

			# Ensure key_bits is 2D
			if key_bits.ndim == 1:
				key_bits_2d = key_bits.unsqueeze(0)
			else:
				key_bits_2d = key_bits

			# Get current state
			if self.head_states[head_idx] is None:
				self._reset_states(device=window.device)

			state = self.head_states[head_idx]

			# State layer forward
			state_input = cat([key_bits_2d, state], dim=1)
			new_state = self.state_layers[head_idx](state_input)

			# Output layer forward
			output = self.output_layers[head_idx](new_state)

			# Determine target
			if is_write:
				# For writes, target is the value being written
				target = value_bits.unsqueeze(0) if value_bits.ndim == 1 else value_bits
				# Update state for writes
				self.head_states[head_idx] = new_state.detach().clone()
			else:
				# For queries, target comes from target_list
				if query_idx < len(encoded_targets):
					target = encoded_targets[query_idx]
					if target.ndim == 1:
						target = target.unsqueeze(0)
					query_idx += 1
				else:
					target = output  # No target, use current output

			contexts.append({
				"window": window,
				"key_bits": key_bits_2d,
				"head_idx": head_idx,
				"is_write": is_write,
				"state_input": state_input,
				"state_output": new_state,
				"output": output,
				"target": target,
			})

		# Find first error
		first_error_t = None
		for t, ctx in enumerate(contexts):
			if not bool((ctx["output"] == ctx["target"]).all()):
				first_error_t = t
				break

		if first_error_t is None:
			# All correct - materialize
			self._materialize_contexts(contexts)
			return

		# Fix first error with EDRA
		self._fix_error(contexts, first_error_t, windows)

	def _materialize_contexts(self, contexts: list[dict]) -> None:
		"""Commit existing correct mappings."""
		for ctx in contexts:
			head_idx = ctx["head_idx"]
			self.state_layers[head_idx].commit(ctx["state_input"], ctx["state_output"])
			self.output_layers[head_idx].commit(ctx["state_output"], ctx["target"])

	def _fix_error(self, contexts: list[dict], error_t: int, windows: list[Tensor]) -> None:
		"""EDRA error correction for KV memory."""
		ctx = contexts[error_t]
		head_idx = ctx["head_idx"]
		target = ctx["target"]

		# Ensure target is 2D
		if target.ndim == 1:
			target = target.unsqueeze(0)

		# Solve output layer: what state would produce correct output?
		desired_state = self.output_layers[head_idx].solve(
			ctx["state_output"].squeeze(0),
			target,
			n_immutable_bits=0
		)

		if desired_state is None:
			return

		# Commit output layer
		desired_state_2d = desired_state.unsqueeze(0)
		self.output_layers[head_idx].commit(desired_state_2d, target)

		# Backprop through this head's state layer
		# For KV memory, we primarily care about the current timestep
		# (unlike sequence models where we backprop through time)

		# Ensure desired_state is 2D
		if desired_state.ndim == 1:
			desired_state_2d = desired_state.unsqueeze(0)

		# Solve state layer
		desired_input = self.state_layers[head_idx].solve(
			ctx["state_input"].squeeze(0),
			desired_state_2d,
			n_immutable_bits=self.k_bits  # Key bits are immutable
		)

		if desired_input is None:
			return

		# Commit state layer
		desired_input_2d = desired_input.unsqueeze(0)
		self.state_layers[head_idx].commit(desired_input_2d, desired_state_2d)

	def write(self, key: str, value: str) -> None:
		"""
		Explicit write operation.

		Args:
			key: Key character (will be encoded to k_bits)
			value: Value character (will be encoded to v_bits)
		"""
		key_bits = self.decoder.encode(key)
		value_bits = self.decoder.encode(value)

		# Flatten if 2D
		if key_bits.ndim == 2:
			key_bits = key_bits.squeeze(0)
		if value_bits.ndim == 2:
			value_bits = value_bits.squeeze(0)

		# Pad/truncate to match expected sizes
		if key_bits.numel() < self.k_bits:
			key_bits = cat([key_bits, zeros(self.k_bits - key_bits.numel(), dtype=uint8)])
		key_bits = key_bits[:self.k_bits]

		if value_bits.numel() < self.v_bits:
			value_bits = cat([value_bits, zeros(self.v_bits - value_bits.numel(), dtype=uint8)])
		value_bits = value_bits[:self.v_bits]

		window = cat([key_bits, value_bits]).unsqueeze(0)
		self.train([window], None)

	def read(self, key: str) -> str:
		"""
		Explicit read operation.

		Args:
			key: Key character to query

		Returns:
			Retrieved value character
		"""
		key_bits = self.decoder.encode(key)

		# Flatten if 2D
		if key_bits.ndim == 2:
			key_bits = key_bits.squeeze(0)

		# Pad/truncate
		if key_bits.numel() < self.k_bits:
			key_bits = cat([key_bits, zeros(self.k_bits - key_bits.numel(), dtype=uint8)])
		key_bits = key_bits[:self.k_bits]

		# Query = key + zeros
		value_bits = zeros(self.v_bits, dtype=uint8)
		window = cat([key_bits, value_bits])

		key_bits_only, head_idx = self._extract_key(window)
		output = self._forward_head(head_idx, key_bits_only, is_write=False)

		return self.decoder.decode(output.unsqueeze(0))

	def __repr__(self):
		return (
			f"RAMMultiHeadKV("
			f"heads={self.num_heads}, "
			f"k={self.k_bits}, "
			f"v={self.v_bits}, "
			f"neurons/head={self.neurons_per_head})"
		)
