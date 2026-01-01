from wnn.ram.core import RAMLayer
from wnn.ram.encoders_decoders import OutputMode
from wnn.ram.encoders_decoders import TransformerDecoder
from wnn.ram.encoders_decoders import TransformerDecoderFactory
from wnn.ram.cost.CostCalculatorRAM import CostCalculatorRAM

from typing import Optional

from torch import cat
from torch import uint8
from torch import zeros
from torch import Tensor
from torch.nn import Module
from torch.nn import ModuleList


class RAMMultiHeadShared(Module):
	"""
	Multi-head sequence model with SHARED state layer.

	Unlike RAMMultiHeadSequence (ensemble of independent RAMSequences),
	this architecture has:
	- ONE shared state layer that evolves over time
	- MULTIPLE output heads that interpret the same state
	- Routing selects which output head to use

	Benefits for long-term memory:
	- Unified temporal context (no fragmented state across heads)
	- All output heads see the full sequence history
	- Specialization happens at output interpretation, not state evolution

	Architecture:
	  input[t] + state[t-1] → state_layer → state[t]
                                              ↓
                              ┌───────────────┼───────────────┐
                              ↓               ↓               ↓
                         output_head[0]  output_head[1]  output_head[2]
                              ↓               ↓               ↓
                           pred[0]         pred[1]         pred[2]
                                              ↓
                                     router selects one
	"""

	def __init__(
		self,
		num_heads: int,
		input_bits: int,
		n_state_neurons: int = 10,  # Larger for long-term memory
		n_output_neurons: int = 5,
		n_bits_per_state_neuron: int | None = None,
		n_bits_per_output_neuron: int | None = None,
		output_mode: OutputMode = OutputMode.RAW,
		use_hashing: bool = False,
		hash_size: int = 1024,
		k_bits: int = 0,  # Key bits for routing (0 = use learned router)
		key_position: str = "last",
		use_learned_router: bool = False,
		rng: int | None = None,
	):
		"""
		Args:
			num_heads: Number of output heads
			input_bits: Input size per token
			n_state_neurons: State layer size (shared across all heads)
			n_output_neurons: Output neurons per head
			n_bits_per_state_neuron: Connections per state neuron (auto if None)
			n_bits_per_output_neuron: Connections per output neuron (auto if None)
			output_mode: Output decoder mode
			use_hashing: Whether to use hash-based addressing
			hash_size: Hash table size
			k_bits: Key bits for hard routing (0 = disabled)
			key_position: Where to extract key bits ("first" or "last")
			use_learned_router: Use RAM-based learned routing
			rng: Random seed
		"""
		super().__init__()

		self.num_heads = num_heads
		self.input_bits = input_bits
		self.n_state_neurons = n_state_neurons
		self.k_bits = k_bits
		self.key_position = key_position
		self.use_kv_routing = k_bits > 0
		self.use_learned_router = use_learned_router
		self.output_mode = output_mode

		# Auto-calculate connectivity
		if n_bits_per_state_neuron is None:
			n_bits_per_state_neuron = input_bits + n_state_neurons  # Full connectivity
		if n_bits_per_output_neuron is None:
			n_bits_per_output_neuron = min(n_state_neurons, 8)

		# Validate routing
		if self.use_kv_routing:
			max_addressable = 2 ** k_bits
			if max_addressable < num_heads:
				print(f"[MultiHeadShared] Warning: k_bits={k_bits} addresses {max_addressable} < {num_heads} heads")
			if k_bits > input_bits:
				raise ValueError(f"k_bits ({k_bits}) cannot exceed input_bits ({input_bits})")

		# Create decoder
		self.decoder: TransformerDecoder = TransformerDecoderFactory.create(output_mode, n_output_neurons)

		# SHARED state layer - sees [input, previous_state]
		self.state_layer = RAMLayer(
			total_input_bits=input_bits + n_state_neurons,
			num_neurons=n_state_neurons,
			n_bits_per_neuron=n_bits_per_state_neuron,
			use_hashing=use_hashing,
			hash_size=hash_size,
			rng=rng,
		)

		# Multiple output heads - each sees [current_state]
		self.output_heads = ModuleList([
			RAMLayer(
				total_input_bits=n_state_neurons,
				num_neurons=n_output_neurons,
				n_bits_per_neuron=n_bits_per_output_neuron,
				use_hashing=use_hashing,
				hash_size=hash_size,
				rng=rng + i + 1 if rng is not None else None,
			)
			for i in range(num_heads)
		])

		# Learned router (optional)
		self.router: CostCalculatorRAM | None = None
		if use_learned_router:
			self.router = CostCalculatorRAM(
				input_bits=input_bits,
				num_options=num_heads,
				n_bits_per_neuron=min(input_bits, 8),
				use_hashing=use_hashing,
				hash_size=hash_size,
				rng=rng,
			)

		# Recurrent state
		self.state_bits: Optional[Tensor] = None

		print(f"[MultiHeadShared] Created: {n_state_neurons} shared state neurons, {num_heads} output heads")

	def _reset_state(self, batch_size: int = 1, device=None) -> None:
		"""Reset recurrent state to zeros."""
		if device is None:
			device = self.state_layer.memory.memory_words.device
		self.state_bits = zeros(batch_size, self.n_state_neurons, dtype=uint8, device=device)

	def _extract_key(self, input_bits: Tensor) -> int:
		"""Extract key bits and convert to head index."""
		if input_bits.ndim == 2:
			input_bits = input_bits.squeeze(0)

		if self.key_position == "first":
			key_bits = input_bits[:self.k_bits]
		else:
			key_bits = input_bits[-self.k_bits:]

		key_value = 0
		for bit in key_bits:
			key_value = (key_value << 1) | int(bit)

		return key_value % self.num_heads

	def _get_routed_head(self, input_bits: Tensor) -> int:
		"""Get head index based on routing mode."""
		if input_bits.ndim == 2:
			input_bits = input_bits.squeeze(0)

		if self.use_learned_router and self.router is not None:
			return int(self.router.calculate_index(input_bits))
		elif self.use_kv_routing:
			return self._extract_key(input_bits)
		else:
			return 0  # Default to head 0 if no routing

	def _normalize_bits(self, bits: Tensor) -> Tensor:
		"""Ensure shape [1, n] and dtype uint8."""
		if bits.ndim == 1:
			bits = bits.unsqueeze(0)
		if bits.dtype != uint8:
			bits = bits.to(uint8)
		return bits

	def _forward_step(self, window_bits: Tensor, update_state: bool = True) -> tuple[Tensor, Tensor, list[Tensor]]:
		"""
		Single forward step through shared state and all output heads.

		Args:
			window_bits: Input tensor [1, input_bits]
			update_state: Whether to update recurrent state

		Returns:
			state_input: [1, input_bits + n_state_neurons]
			state_output: [1, n_state_neurons]
			head_outputs: List of [1, n_output_neurons] for each head
		"""
		batch_size = window_bits.shape[0]

		if self.state_bits is None or self.state_bits.shape[0] != batch_size:
			self._reset_state(batch_size, device=window_bits.device)

		# State layer: [input, prev_state] → new_state
		state_input = cat([window_bits, self.state_bits], dim=1)
		state_output = self.state_layer(state_input)

		# All output heads interpret the same state
		head_outputs = [head(state_output) for head in self.output_heads]

		if update_state:
			self.state_bits = state_output.detach().clone()

		return state_input, state_output, head_outputs

	def forward(self, input_bits: Tensor) -> str:
		"""
		Forward pass: evolve shared state, route to output head.

		Args:
			input_bits: Input tensor [total_bits] or [1, total_bits]

		Returns:
			Predicted character
		"""
		input_bits = self._normalize_bits(input_bits)
		self._reset_state(1, device=input_bits.device)

		# Make windows
		windows = self._make_windows(input_bits)

		# Process all windows through shared state
		for window in windows:
			state_input, state_output, head_outputs = self._forward_step(window, update_state=True)

		# Get last window for routing decision
		last_window = windows[-1].squeeze(0)

		# Route to select output head
		if self.use_kv_routing or self.use_learned_router:
			head_idx = self._get_routed_head(last_window)
			selected_output = head_outputs[head_idx]
		else:
			# No routing - use head 0 (or could vote)
			selected_output = head_outputs[0]

		return self.decoder.decode(selected_output)

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

	def train(self, windows: list[Tensor], targets: str | list[str]) -> None:
		"""
		Train the shared-state multi-head model.

		All timesteps go through the SAME shared state layer.
		Each output head is trained only when it's routed to.

		Args:
			windows: List of input tensors [1, input_bits]
			targets: Target characters per timestep
		"""
		n_steps = len(windows)
		if n_steps == 0:
			return

		# Normalize
		windows = [self._normalize_bits(w) for w in windows]
		if isinstance(targets, str):
			target_list = list(targets)
		else:
			target_list = list(targets)

		if len(target_list) != n_steps:
			raise ValueError(f"Expected {n_steps} targets, got {len(target_list)}")

		# Encode targets
		encoded_targets = [self.decoder.encode(t) for t in target_list]

		# Reset state and collect contexts
		self._reset_state(1, device=windows[0].device)

		contexts = []
		for t, window in enumerate(windows):
			state_input, state_output, head_outputs = self._forward_step(window, update_state=True)

			# Determine which head is routed for this timestep
			window_bits = window.squeeze(0)
			head_idx = self._get_routed_head(window_bits) if (self.use_kv_routing or self.use_learned_router) else 0

			contexts.append({
				"window_bits": window,
				"state_input": state_input,
				"state_output": state_output,
				"head_outputs": head_outputs,
				"routed_head": head_idx,
				"target": encoded_targets[t],
			})

		# Find first error (check only routed head's output)
		first_error_t = None
		for t in range(n_steps):
			ctx = contexts[t]
			routed_output = ctx["head_outputs"][ctx["routed_head"]]
			if not bool((routed_output == ctx["target"]).all()):
				first_error_t = t
				break

		# If all correct, materialize and return
		if first_error_t is None:
			self._materialize_contexts(contexts)
			return

		# EDRA: fix the first error
		self._fix_error(contexts, first_error_t, windows)

	def _materialize_contexts(self, contexts: list[dict]) -> None:
		"""Commit existing mappings to solidify EMPTY cells."""
		for ctx in contexts:
			# State layer
			self.state_layer.commit(ctx["state_input"], ctx["state_output"])
			# Routed output head
			head_idx = ctx["routed_head"]
			target = ctx["target"]
			if target.ndim == 1:
				target = target.unsqueeze(0)
			self.output_heads[head_idx].commit(ctx["state_output"], target)

	def _fix_error(self, contexts: list[dict], error_t: int, windows: list[Tensor]) -> None:
		"""
		EDRA error correction for shared state architecture.

		1. Solve output head constraints at error timestep
		2. Backprop desired state through time via state layer
		"""
		ctx = contexts[error_t]
		head_idx = ctx["routed_head"]
		target_bits = ctx["target"]

		# Ensure target_bits is 2D for solve
		if target_bits.ndim == 1:
			target_bits_2d = target_bits.unsqueeze(0)
		else:
			target_bits_2d = target_bits

		# Solve output head: what state would produce correct output?
		desired_state = self.output_heads[head_idx].solve(
			ctx["state_output"].squeeze(0),
			target_bits_2d,
			n_immutable_bits=0
		)

		if desired_state is None:
			return  # No solution

		# Commit output head mapping
		desired_state_2d = desired_state.unsqueeze(0)
		self.output_heads[head_idx].commit(desired_state_2d, target_bits_2d)

		# Backprop through state layer
		for t in range(error_t, -1, -1):
			ctx = contexts[t]

			# Ensure desired_state is 2D for solve
			if desired_state.ndim == 1:
				desired_state_2d = desired_state.unsqueeze(0)
			else:
				desired_state_2d = desired_state

			# Solve state layer: what input would produce desired_state?
			desired_input = self.state_layer.solve(
				ctx["state_input"].squeeze(0),
				desired_state_2d,
				n_immutable_bits=self.input_bits  # Input bits are immutable
			)

			if desired_input is None:
				return

			# Commit state layer
			desired_input_2d = desired_input.unsqueeze(0)
			self.state_layer.commit(desired_input_2d, desired_state_2d)

			# Extract desired previous state for next iteration
			desired_state = desired_input[self.input_bits:]

			# Recompute contexts if changed
			if t > 0:
				self._reset_state(1, device=windows[0].device)
				for i in range(t):
					_, state_output, head_outputs = self._forward_step(windows[i], update_state=True)
				contexts[t-1]["state_output"] = state_output

	def __repr__(self):
		if self.use_learned_router:
			routing = "learned_router"
		elif self.use_kv_routing:
			routing = f"k_bits={self.k_bits}"
		else:
			routing = "none"
		return (
			f"RAMMultiHeadShared("
			f"heads={self.num_heads}, "
			f"state={self.n_state_neurons}, "
			f"routing={routing})"
		)
