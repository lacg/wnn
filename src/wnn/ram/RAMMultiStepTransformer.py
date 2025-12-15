from wnn.ram.RAMEnums import MemoryVal
from wnn.ram.RAMLayer import RAMLayer
from wnn.ram.decoders import OutputMode
from wnn.ram.decoders import TransformerDecoder
from wnn.ram.decoders import TransformerDecoderFactory


from typing import Optional

from torch import arange
from torch import bool as tbool
from torch import cat
from torch import device
from torch import int64
from torch import long
from torch import randint
from torch import uint8
from torch import tensor
from torch import zeros
from torch import Tensor
from torch.nn import Module


class RAMMultiStepTransformer(Module):
	"""
	Three-layer RAM-based "transformer":
	 - input_layer: sees raw input bits
	 - state_layer: sees [input_layer_output, previous_state_bits]
	 - output_layer: sees [input_layer_output, current_state_bits]

	All layers are trained via EDRA-style EMPTY-first steering.
	"""

	def __init__(
		self,
		input_bits: int,
		n_input_neurons: int,
		n_state_neurons: int,
		n_output_neurons: int,
		n_bits_per_input_neuron: int,
		n_bits_per_state_neuron: int,
		n_bits_per_output_neuron: int,
		use_hashing: bool = False,
		hash_size: int = 1024,
		rng: Optional[int] = None,
		max_iters: int = 4,
		output_mode: OutputMode = OutputMode.HAMMING,
	):
		super().__init__()

		self.input_bits = int(input_bits)
		self.max_iters = max_iters
		self.output_mode = output_mode
		self.decoder: TransformerDecoder = TransformerDecoderFactory.create(output_mode, n_output_neurons)

		# -------------------------
		# Layers
		# -------------------------
		# Input layer: raw window bits -> input_neuron_outputs
		self.input_layer = RAMLayer(
			total_input_bits=self.input_bits,
			num_neurons=n_input_neurons,
			n_bits_per_neuron=n_bits_per_input_neuron,
			use_hashing=use_hashing,
			hash_size=hash_size,
			rng=None if rng is None else rng,
		)

		# State layer: sees [input_layer_output, previous_state_bits]
		self.state_layer = RAMLayer(
			total_input_bits=n_input_neurons + n_state_neurons,
			num_neurons=n_state_neurons,
			n_bits_per_neuron=n_bits_per_state_neuron,
			use_hashing=use_hashing,
			hash_size=hash_size,
			rng=None if rng is None else (rng + 1),
		)

		# Output layer: sees [input_layer_output, current_state_bits]
		self.output_layer = RAMLayer(
			total_input_bits=n_input_neurons + n_state_neurons,
			num_neurons=n_output_neurons,
			n_bits_per_neuron=n_bits_per_output_neuron,
			use_hashing=use_hashing,
			hash_size=hash_size,
			rng=None if rng is None else (rng + 2),
		)

		# recurrent state bits (updated after stable examples)
		self.state_bits: Optional[Tensor] = None

	def __repr__(self):
		return (
			f"RAMTransformer"
			f"("
			f"InputLayer"
			f"("
			f"{self.input_layer.__repr__()}"
			f")"
			f"StateLayer"
			f"("
			f"{self.state_layer.__repr__()}"
			f")"
			f"OutputLayer"
			f"("
			f"{self.output_layer.__repr__()}"
			f")"
			f")"
		)

	def __str__(self):
		lines = []
		lines.append("\n==============================")
		lines.append(" RAMTransformer Full Dump")
		lines.append("==============================")

		lines.append("\n--- Input Layer ---")
		lines.append(str(self.input_layer))

		lines.append("\n--- State Layer ---")
		if self.state_layer.num_neurons > 0:
			lines.append(str(self.state_layer))
		else:
			lines.append(" (no state neurons)")

		lines.append("\n--- Output Layer ---")
		lines.append(str(self.output_layer))

		lines.append("")  # newline
		return "\n".join(lines)

	# ------------------------------------------------------------------
	# Utility helpers
	# ------------------------------------------------------------------

	def _calculate_addresses(self, context: dict) -> None:
		input_address = self.input_layer.get_addresses(context["window_bits"])[0]
		state_address = self.state_layer.get_addresses((context["state_layer_input"]))[0] if context["state_layer_input"].shape[1] > 0 else None
		return input_address, state_address

	# ------------------------------------------------------------------
	# Single-step EDRA (train_one)
	# ------------------------------------------------------------------
	def _continue_train_one_iteration(self, input_bits: Tensor, target_bits: Tensor) -> bool:
		"""
		One EDRA iteration for a single-step example (no BPTT), used by train_one.

		Returns:
			False if training for this sample is done (all outputs stable),
			True  if another iteration is needed.
		"""
		# Forward pass (no commit)
		(state_layer_input, input_layer_output, state_layer_output, output_layer_input, output_layer_output) = self._get_outputs(input_bits, update_state=False)

		# Track which neurons are stable
		stable_mask = (output_layer_output == target_bits)  # [1, self.output_layer.num_neurons]

		if bool(stable_mask.all()):
			# ALL outputs correct → now final commit
			self._train_write(input_bits, target_bits, state_layer_input, input_layer_output, state_layer_output, output_layer_input)
			# Leave the train_one iteration
			return False

		# -------------------------------------------------
		# Precompute hidden-layer addresses ONCE per sample
		# -------------------------------------------------
		input_addrs = self.input_layer.get_addresses(input_bits)[0]  # [N_in]
		state_addrs = (self.state_layer.get_addresses(state_layer_input)[0]) if self.state_layer.num_neurons > 0 else None	# [N_state]

		self._update_neurons(stable_mask, input_bits, target_bits, state_layer_input, output_layer_input, input_addrs, state_addrs)

		return True

	def _decode_address_bits(self, n_bits: int, address: int, dev: device) -> Tensor:
		"""
		Decode an integer address into [n_bits] bool bits (LSB-first).
		"""
		shifts = arange(n_bits, device=dev)
		addr_tensor = tensor(address, device=dev, dtype=int64)
		return ((addr_tensor >> shifts) & 1).to(tbool)

	# ------------------------------------------------------------------
	# EDRA for hidden (input+state) layers backward in time
	# ------------------------------------------------------------------
	def _edra_update_hidden_step(self, context_t: dict, window_addrs_t: Tensor, prev_state_addrs: Optional[Tensor], state_target_next: Tensor) -> Optional[Tensor]:
		"""
		EDRA update on the Input and State layers (the hidden stack) at timestep t,
		using as target the desired state bits at timestep t+1 (state_target_next).

		context_t:            context dict at timestep t
		window_addrs_t:    [N_in] addresses of input neurons at timestep t
		prev_state_addrs: [N_state] addresses of state neurons at timestep t-1,
		                  or None at t=0
		state_target_next:[1, N_state] bool — desired state output at timestep t+1

		Returns:
			state_target_prev: [1, N_state] bool or None, describing the desired
			previous-state bits at timestep t (to propagate further back).
		"""
		if self.state_layer.num_neurons == 0:
			return None

		if state_target_next.ndim == 1:
			state_target_next = state_target_next.unsqueeze(0)

		state_layer_input = context_t["state_layer_input"]   # [1, N_in + N_state]
		state_layer_output = context_t["state_layer_output"] # [1, N_state]

		# Addresses of state neurons at *this* timestep (needed when state acts as hidden)
		state_addrs_t = self.state_layer.get_addresses(state_layer_input)[0]  # [N_state]
		state_addrs_start = self.input_layer.num_neurons
		state_addrs_end = state_addrs_start + self.state_layer.num_neurons

		# Initialize previous-state target as the current "previous-state" bits for this step.
		# They are the second half of state_layer_input: [input_out_t, prev_state_bits_t]
		if state_layer_input.shape[1] >= state_addrs_end:
			prev_state_bits_t = state_layer_input[:, state_addrs_start : state_addrs_end]
			state_target_prev = prev_state_bits_t.detach().clone().to(tbool)
		else:
			state_target_prev = None

		for neuron_index in range(self.state_layer.num_neurons):
			current_bit = bool(state_layer_output[0, neuron_index].item())
			desired_bit = bool(state_target_next[0, neuron_index].item())
			if current_bit == desired_bit:
				continue

			current_address = int(state_addrs_t[neuron_index].item())

			# IMPORTANT: this must be state_layer, not output_layer
			target_address = self.state_layer.choose_address(neuron_index, desired_bit, current_address, state_layer_input.device)

			desired_bits = self._decode_address_bits(self.state_layer.memory.n_bits_per_neuron, target_address, state_layer_input.device)

			hidden_indices = self.state_layer.memory.connections[neuron_index].to(device=state_layer_input.device, dtype=long)
			mask_window = hidden_indices < self.input_layer.num_neurons
			mask_prev = ~mask_window

			# INPUT side (timestep t)
			if mask_window.any():
				window_index = hidden_indices[mask_window]	# indices into input_layer
				required_bits = desired_bits[mask_window]
				window_address = window_addrs_t[window_index]	# addresses for those input neurons
				self.input_layer.set_memory_batch(window_index, window_address, required_bits, allow_override=False)

			# PREVIOUS-STATE side (t-1)
			if mask_prev.any() and prev_state_addrs is not None:
				state_index_prev = hidden_indices[mask_prev] - self.input_layer.num_neurons
				required_bits = desired_bits[mask_prev]
				state_address_prev = prev_state_addrs[state_index_prev]			# addresses for those input neurons
				self.state_layer.set_memory_batch(state_index_prev, state_address_prev, required_bits, allow_override=False)

				if state_target_prev is not None:
					# these bits are what we want the state layer to output at this step
					state_target_prev[0, state_index_prev] = required_bits

			self.state_layer.set_memory(neuron_index, target_address, desired_bit, allow_override=False)

		return state_target_prev

	# ------------------------------------------------------------------
	# EDRA for final output step (sequence case)
	# ------------------------------------------------------------------
	def _edra_update_output_step(self, context: dict, target_bits: Tensor, input_addrs: Tensor, state_addrs: Optional[Tensor]) -> Optional[Tensor]:
		"""
		Single-step EDRA on the OUTPUT layer for the last timestep.

		context: dict for last step from get_context_ts(...)
		target_bits: [1, N_out] bool
		input_addrs: [N_in] addresses for input_layer at this step
		state_addrs: [N_state] addresses for state_layer at this step (or None)

		Returns:
			state_target: [1, N_state] bool or None
				Desired state bits at this final step (to be enforced at T-1
			 by EDRA on the state layer in earlier timesteps).
		"""
		if target_bits.ndim == 1:
			target_bits = target_bits.unsqueeze(0)

		state_layer_output = context["state_layer_output"]
		output_layer_input = context["output_layer_input"]
		output_layer_output = context["output_layer_output"]

		# Precompute output addresses at this step
		output_addresses = self.output_layer.get_addresses(output_layer_input)[0]  # [N_out]

		# Initialize state target as "whatever the state already outputs" at this step
		state_target = state_layer_output.detach().clone().to(tbool) if self.state_layer.num_neurons > 0 else None

		for neuron_index in range(self.output_layer.num_neurons):
			current_bit = bool(output_layer_output[0, neuron_index].item())
			desired_bit = bool(target_bits[0, neuron_index].item())
			if current_bit == desired_bit:
				continue

			current_address = int(output_addresses[neuron_index].item())

			# Step 1: choose label-compatible address in output layer
			target_address = self.output_layer.choose_address(neuron_index, desired_bit, current_address, output_layer_input.device)

			# Step 2: decode address → bits per connection
			desired_bits = self._decode_address_bits(self.output_layer.memory.n_bits_per_neuron, target_address, output_layer_input.device)

			hidden_indices = self.output_layer.memory.connections[neuron_index].to(device=output_layer_input.device, dtype=long)
			mask_input = hidden_indices < self.input_layer.num_neurons
			mask_state = ~mask_input

			# ----- INPUT HIDDEN UPDATES -----
			if mask_input.any():
				input_index = hidden_indices[mask_input]	# indices into input_layer
				required_bits = desired_bits[mask_input]
				input_address = input_addrs[input_index]	# addresses for those input neurons
				self.input_layer.set_memory_batch(input_index, input_address, required_bits, allow_override=False)

			# ----- STATE UPDATES -----
			if (mask_state.any() and state_addrs is not None and self.state_layer.num_neurons > 0):
				state_index = hidden_indices[mask_state] - self.input_layer.num_neurons
				required_bits = desired_bits[mask_state]
				state_address = state_addrs[state_index]			# addresses for those input neurons
				self.state_layer.set_memory_batch(state_index, state_address, required_bits, allow_override=False)

				if state_target is not None:
					# these bits are what we want the state layer to output at this step
					state_target[0, state_index] = required_bits

			# finally write output memory
			self.output_layer.set_memory(neuron_index, target_address, desired_bit, allow_override=False)

		return state_target

	def _get_outputs(self, window_bits: Tensor, update_state: bool = False) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
		"""
		Run full forward pass:
			state_layer_input = [raw window bits, previous state bits]
			output_layer_input = [input_layer_output, state_layer_output]

		Returns:
			(
				state_layer_input,   # [B, N_in + N_state] or [B, 0]
				input_layer_output,  # [B, N_in]
				state_layer_output,  # [B, N_state] or [B, 0]
				output_layer_input,  # [B, N_in + N_state]
				output_layer_output, # [B, N_out]
			)
		"""
		batch_size = window_bits.shape[0]

		if self.state_bits is None or self.state_bits.shape[0] != batch_size:
			self.reset_state(batch_size, device=window_bits.device)

		# 1) Input layer: raw window bits -> hashed/abstracted features
		input_layer_output = self.input_layer(window_bits)  # [B, N_in]

		# 2) State layer: [input_layer_output, prev_state_bits]
		if self.state_layer.num_neurons > 0:
			state_layer_input = cat([input_layer_output, self.state_bits], dim=1)
			state_layer_output = self.state_layer(state_layer_input)  # [B, N_state]
		else:
			state_layer_input = zeros(batch_size, 0, dtype=uint8, device=window_bits.device)
			state_layer_output = state_layer_input

		# 3) Output layer: [input_layer_output, current_state_bits]
		output_layer_input = cat([input_layer_output, state_layer_output], dim=1)  # [B, N_in + N_state]
		output_layer_output = self.output_layer(output_layer_input)  # [B, N_out]

		if update_state and self.state_layer.num_neurons > 0:
			self.state_bits = state_layer_output.detach().clone()

		return state_layer_input, input_layer_output, state_layer_output, output_layer_input, output_layer_output

	def _materialize_step(self, context):
		# INPUT layer
		self.input_layer.train_write(context["window_bits"], context["input_layer_output"])
		# STATE layer
		if self.state_layer.num_neurons > 0:
			self.state_layer.train_write(context["state_layer_input"], context["state_layer_output"])
		# OUTPUT layer
		self.output_layer.train_write(context["output_layer_input"], context["output_layer_output"])

	def _normalize_bits(self, bits: Tensor) -> Tensor:
		"""
		Ensure shape [1, n] and dtype uint8.
		"""
		if bits.ndim == 1:
			bits = bits.unsqueeze(0)
		if bits.dtype != uint8:
			bits = bits.to(uint8)
		return bits


	def _train_write(self, input_bits: Tensor, target_bits: Tensor, state_layer_input: Tensor, input_layer_output: Tensor, state_layer_output: Tensor, output_layer_input: Tensor) -> None:
		"""
		Commit writes to memory when all outputs are correct.
		"""
		# ALL outputs correct → now final commit
		# commit input memory
		self.input_layer.train_write(input_bits, input_layer_output)
		# commit state memory
		if self.state_layer.num_neurons > 0:
			self.state_layer.train_write(state_layer_input, state_layer_output)
			self.state_bits = state_layer_output.detach().clone()
		# commit output memory
		self.output_layer.train_write(output_layer_input, target_bits)

	def _update_neurons(self, stable_mask: Tensor, target_bits: Tensor, output_layer_input: Tensor, input_addrs: Tensor, state_addrs: Optional[Tensor]) -> None:
		"""
		Vectorized EDRA update for all unstable output neurons.

		- For each output neuron j where stable_mask[0, j] is False:
			1) Find a feasible target address (label-compatible + hidden-feasible)
			2) Decode the address into k bits
			3) For each connected hidden neuron:
				- if it's an input neuron, write required bit into input memory
				- if it's a state neuron, write required bit into state memory
				* compute its address (from precomputed input_addrs/state_addrs)
				* write required bit via set_memory_batch
			4) Finally write the output memory cell for neuron j
		"""
		output_addresses = self.output_layer.get_addresses(output_layer_input)[0]  # [1, self.output_layer.num_neurons]

		for neuron_index in range(self.output_layer.num_neurons):
			if bool(stable_mask[0, neuron_index]):
				continue  # skip neurons that already match

			desired_output_bit = bool(target_bits[0, neuron_index].item())
			current_address = int(output_addresses[neuron_index].item())

			# Step 1: choose a good output address
			target_output_address = self.output_layer.choose_address(neuron_index, desired_output_bit, current_address, output_layer_input.device)

			# Step 2: decode address -> bits per connection
			desired_bits = self._decode_address_bits(self.output_layer.memory.n_bits_per_neuron, target_output_address, output_layer_input.device)  # [n_bits]
	
			# Step 3: correct hidden neurons
			# Connected hidden neurons (indices into [input_neurons + state_neurons])
			hidden_indices = self.output_layer.memory.connections[neuron_index].to(output_layer_input.device, dtype=long)

			mask_input = hidden_indices < self.input_layer.num_neurons
			mask_state = ~mask_input

			# INPUT SIDE
			if mask_input.any():
				input_index = hidden_indices[mask_input]	# indices into input_layer
				required_bits = desired_bits[mask_input]
				input_address = input_addrs[input_index]	# addresses for those input neurons
				self.input_layer.set_memory_batch(input_index, input_address, required_bits, allow_override=False)

			# STATE SIDE
			if mask_state.any() and state_addrs is not None and self.state_layer.num_neurons > 0:
				state_index = hidden_indices[mask_state] - self.input_layer.num_neurons
				required_bits = desired_bits[mask_state]
				state_address = state_addrs[state_index]			# addresses for those input neurons
				self.state_layer.set_memory_batch(state_index, state_address, required_bits, allow_override=False)

			# Step 4: after hidden has been corrected, write output memory
			self.output_layer.set_memory(neuron_index, target_output_address, desired_output_bit, allow_override=False)


	# ------------------------------------------------------------------
	# Public API
	# ------------------------------------------------------------------
	def forward(self, input_bits: Tensor) -> Tensor:
		"""
		Inference-only forward: returns output bits [B, N_out].
		"""
		self.reset_state(input_bits.shape[0], input_bits.device)
		*_, output_layer_output = self._get_outputs(self._normalize_bits(input_bits), update_state=True)
		return self.decoder.decode(output_layer_output)

	def get_context_t(self, windows: list[Tensor], steps: int) -> dict:
		"""
		Run a full multi-step episode over a list of input windows,
		*without* modifying memory, *with* state evolution,
		and return per-step contexts for EDRA-BPTT.

		Return is the last response[step_t]:
			{
				"window_bits":        [1, input_bits],
				"state_layer_input":  [1, N_in + N_state] or [1, 0] if no state layer,
				"input_layer_output": [1, N_in],
				"state_layer_output": [1, N_state] or [1, 0],
				"output_layer_input": [1, N_in + N_state],
				"output_layer_output":[1, N_out],
			}
		"""
		if len(windows) == 0:
			raise ValueError("get_context_ts: windows list is empty")

		device = windows[0].device

		# Reset recurrent state for this episode
		self.reset_state(batch_size=1, device=device)

		for i in range(steps):
			raw_window = windows[i]
			window_bits = self._normalize_bits(raw_window)

			(state_layer_input, input_layer_output, state_layer_output, output_layer_input, output_layer_output) = self._get_outputs(window_bits, update_state=True)

		return 				{
					"window_bits":					window_bits,
					"state_layer_input":		state_layer_input,
					"input_layer_output":		input_layer_output,
					"state_layer_output":		state_layer_output,
					"output_layer_input":		output_layer_input,
					"output_layer_output":	output_layer_output,
				}

	# input_bits: [n] bits (0/1 or bool)
	def make_windows(self, input_bits: Tensor) -> list[Tensor]:
		"""
		Build episode windows from an input bit vector.

		Accepts:
			- [n]          (1D bit vector)
			- [1, n]       (single-sample batch)

		Returns:
			list of [1, input_bits] tensors
		"""
		# Flatten to a single 1D sequence of bits
		if input_bits.ndim == 2:
			if input_bits.shape[0] != 1:
				raise ValueError(f"make_windows expects shape [n] or [1, n], got {input_bits.shape}")
			seq = input_bits.squeeze(0)					# [n]
		elif input_bits.ndim == 1:
			seq = input_bits									# [n]
		else:
			# Fallback: just flatten everything
			seq = input_bits.reshape(-1)

		windows: list[Tensor] = []
		window_size = self.input_bits

		for i in range(0, seq.numel(), window_size):
			chunk = seq[i:i + window_size]			# [<= window_size]
			if chunk.numel() < window_size:
				# pad with 0s (input FALSE) for missing bits
				pad = zeros(window_size - chunk.numel(), dtype=chunk.dtype, device=chunk.device)
				chunk = cat([chunk, pad], dim=0)	# [window_size]
			windows.append(chunk.unsqueeze(0))	# [1, window_size]

		return windows

	def reset_state(self, batch_size: int, device) -> None:
		"""
		Initialize recurrent state to all False.
		"""
		self.state_bits = zeros(batch_size, self.state_layer.num_neurons, dtype=uint8, device=device) if self.state_layer.num_neurons > 0 else zeros(batch_size, 0, dtype=uint8, device=device)

	def train_one(self, input_bits: Tensor, target_bits: Tensor) -> None:
		"""
		Multi-output EDRA:
			- For each output neuron neuron_index independently:
				* If stable, skip
				* Else steer hidden/input memory so that output neuron neuron_index
				  reaches a good memory address storing the correct label.
			- Only after ALL outputs are stable do we commit writes.
		"""

		# --- ensure batch dimension ---
		input_bits = self._normalize_bits(input_bits)
		target_bits = self.decoder.encode(target_bits)

		if input_bits.shape[0] != 1:
			raise ValueError("train_one supports only batch_size=1")

		# Init state if needed
		if self.state_bits is None or self.state_bits.shape[0] != 1:
			self.reset_state(1, device=input_bits.device)

		i = 0
		while i < self.max_iters and self._continue_train_one_iteration(input_bits, target_bits):
			i += 1

	def train_sequence_edra_bptt(self, windows: list[Tensor], target_bits: Tensor) -> None:
		"""
		Full EDRA-through-time (EDRA-BPTT):

		1) Run the whole sequence (no writes) and record per-step contexts.
		2) If final output is incorrect:
		   - Run EDRA on the output layer at the last step, producing a state
		     target at T-1.
		   - Then, for t = T-2 down to 0, run EDRA on the state layer using the
		     propagated state targets, updating input+state memories backwards
		     in time.

		This reduces exactly to single-step EDRA when:
			- len(windows) == 1, and
			- self.state_layer.num_neurons == 0.
		"""
		target_bits = self.decoder.encode(target_bits)

		# Run sequence and collect contexts
		n_steps = len(windows)
		context = self.get_context_t(windows, n_steps)

		# Degenerate single-step / no-state case → use classic EDRA
		if n_steps == 1 and self.state_layer.num_neurons == 0:
			window_bits = self._normalize_bits(windows[0])
			self.train_one(window_bits, target_bits)
			return

		# Final output at n_steps-1
		final_output = context["output_layer_output"]
		if bool((final_output == target_bits).all()):
			# Reinforce EMPTY memories to answer FALSE
			for i in range(n_steps, 0, -1):
				self._materialize_step(self.get_context_t(windows, i))
			# Now, NN will converge correctly, nothing else to do.
			return

		if self.state_layer.num_neurons > 0:
			input_address, state_address = self._calculate_addresses(context)
			state_target_next = self._edra_update_output_step(context, target_bits, input_address, state_address)

			# Backward through time: EDRA on state layer
			# We go from t = T-1 down to t = 1
			t = n_steps
			while t > 0 and state_target_next is not None:

				# Previous-state addresses for timestep t are the state addresses
				# at timestep t-1 (when those states were produced).
				context = self.get_context_t(windows, t)
				input_address, state_address = self._calculate_addresses(context)
				state_target_prev = self._edra_update_hidden_step(context, input_address, state_address, state_target_next)
				t, state_target_next = t - 1, state_target_prev
