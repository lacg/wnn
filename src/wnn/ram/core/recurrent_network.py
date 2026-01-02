from wnn.ram.core import RAMLayer
from wnn.ram.encoders_decoders import OutputMode
from wnn.ram.encoders_decoders import TransformerDecoder
from wnn.ram.encoders_decoders import TransformerDecoderFactory
from wnn.ram.enums import StateMode


from typing import Optional

from torch import cat
from torch import tensor
from torch import uint8
from torch import zeros
from torch import Tensor
from torch.nn import Module


class RAMRecurrentNetwork(Module):
	"""
	Two-layer recurrent RAM-based network:
	 - state_layer: sees [input, previous_state_bits]
	 - output_layer: sees [current_state_bits]

	All layers are trained via EDRA-style EMPTY-first steering.
	"""

	def __init__(
		self,
		input_bits: int,
		n_state_neurons: int,
		n_output_neurons: int,
		n_bits_per_state_neuron: int,
		n_bits_per_output_neuron: int,
		use_hashing: bool = False,
		hash_size: int = 1024,
		rng: Optional[int] = None,
		max_iters: int = 4,
		output_mode: OutputMode = OutputMode.HAMMING,
	):
		super().__init__()

		# Store config for serialization
		self.input_bits = int(input_bits)
		self.n_state_neurons = n_state_neurons
		self.n_output_neurons = n_output_neurons
		self.n_bits_per_state_neuron = n_bits_per_state_neuron
		self.n_bits_per_output_neuron = n_bits_per_output_neuron
		self.use_hashing = use_hashing
		self.hash_size = hash_size
		self.rng = rng
		self.max_iters = max_iters
		self.output_mode = output_mode
		self.decoder: TransformerDecoder = TransformerDecoderFactory.create(output_mode, n_output_neurons)

		# -------------------------
		# Layers
		# -------------------------
		# State layer: sees [input_layer_output, previous_state_bits]
		self.state_layer = self._create_state_layer(
			total_input_bits=input_bits + n_state_neurons,
			num_neurons=n_state_neurons,
			n_bits_per_neuron=n_bits_per_state_neuron,
			use_hashing=use_hashing,
			hash_size=hash_size,
			rng=None if rng is None else rng,
		)

		# Output layer: sees [input_layer_output, current_state_bits]
		self.output_layer = self._create_output_layer(
			num_neurons=n_output_neurons,
			n_bits_per_neuron=n_bits_per_output_neuron,
			use_hashing=use_hashing,
			hash_size=hash_size,
			rng=None if rng is None else (rng + 1),
		)

		# recurrent state bits (updated after stable examples)
		self.state_bits: Optional[Tensor] = None

	def __repr__(self):
		return (
			f"RAMTransformer"
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

		lines.append("\n--- State Layer ---")
		lines.append(str(self.state_layer))

		lines.append("\n--- Output Layer ---")
		lines.append(str(self.output_layer))

		lines.append("")  # newline
		return "\n".join(lines)

	def _create_state_layer(self,
		total_input_bits: int,
		num_neurons: int,
		n_bits_per_neuron: int,
		use_hashing: bool = False,
		hash_size: int = 1024,
		rng: Optional[int] = None,
	) -> RAMLayer:
		"""
		Factory method for creating the state layer.

		Override in subclasses to customize connection patterns.
		"""
		return RAMLayer(
			total_input_bits=total_input_bits,
			num_neurons=num_neurons,
			n_bits_per_neuron=n_bits_per_neuron,
			use_hashing=use_hashing,
			hash_size=hash_size,
			rng=rng,
		)

	def _create_output_layer(self,
		num_neurons: int,
		n_bits_per_neuron: int,
		use_hashing: bool = False,
		hash_size: int = 1024,
		rng: Optional[int] = None,
	) -> RAMLayer:
		return RAMLayer(
			total_input_bits=self.state_layer.num_neurons,
			num_neurons=num_neurons,
			n_bits_per_neuron=n_bits_per_neuron,
			use_hashing=use_hashing,
			hash_size=hash_size,
			rng=rng,
		)


	# ------------------------------------------------------------------
	# Utility helpers
	# ------------------------------------------------------------------

	def _calculate_state_output(self, window_bits: Tensor, desired_state_output: Tensor) -> tuple[Optional[slice], Tensor]:
		return None, desired_state_output

	def _return_state_output(self, window_bits: Tensor, desired_previous_state_input: Tensor, head: slice) -> Tensor:
		return desired_previous_state_input[:, window_bits.shape[1]:]

	def _commit(self, window_bits: Tensor, current_state_input: Tensor, desired_state_output: Tensor) -> tuple[bool, Optional[Tensor]]:
		"""
		Commit desired state output at the CURRENT address.

		IMPORTANT: We write directly at current_state_input's address rather than
		solving for a different input. This ensures:
		1. Single-step tasks (like parity) get correct training at the address
		   that will be read during inference.
		2. For multi-step tasks, BPTT still works because we return the desired
		   state for backpropagation to earlier timesteps.

		Return:
			changed:        True if any change happened, False otherwise.
			previous bits:  Tensor (the desired state for BPTT to earlier timesteps)
		"""
		if desired_state_output is None:
			return False, None
		assert current_state_input.shape[0] == self.input_bits + self.state_layer.num_neurons

		head, calculated_state_output = self._calculate_state_output(window_bits, desired_state_output)

		# Direct write at CURRENT address - don't solve for a different input.
		# This fixes the address mismatch bug for single-step tasks.
		current_state_input_2d = current_state_input.unsqueeze(0)  # [1, N_in+N_state]

		# Use explore_batch for direct write with allow_override
		from torch import arange, long
		addresses = self.state_layer.get_addresses(current_state_input_2d)[0]  # [N_neurons]
		neuron_indices = arange(self.state_layer.num_neurons, dtype=long, device=addresses.device)

		# Get target bits (handle both 1D and 2D)
		target_bits = calculated_state_output[0] if calculated_state_output.ndim == 2 else calculated_state_output

		changed = self.state_layer.memory.explore_batch(
			neuron_indices, addresses, target_bits, allow_override=True
		)

		# Return desired state for BPTT to earlier timesteps
		return (changed, self._return_state_output(window_bits, current_state_input_2d, head))

	def _get_contexts(self, windows: list[Tensor], steps: int) -> dict:
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
		self._reset_state(batch_size=1, device=device)

		contexts = []
		for i in range(steps):
			window_bits = windows[i]
			(state_layer_input, state_layer_output, output_layer_input, output_layer_output) = self._get_outputs(window_bits, update_state=True)
			contexts.append({
					"window_bits":					window_bits,
					"state_layer_input":		state_layer_input,
					"state_layer_output":		state_layer_output,
					"output_layer_input":		output_layer_input,
					"output_layer_output":	output_layer_output,
				})
		return contexts

	def _get_outputs(self, window_bits: Tensor, update_state: bool = False) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
		"""
		Run full forward pass:
			state_layer_input = [raw window bits, previous state bits]
			output_layer_input = [input_layer_output, state_layer_output]

		Returns:
			(
				state_layer_input,   # [B, N_in + N_state] or [B, 0]
				state_layer_output,  # [B, N_state] or [B, 0]
				output_layer_input,  # [B, N_in + N_state]
				output_layer_output, # [B, N_out]
			)
		"""
		batch_size = window_bits.shape[0]

		if self.state_bits is None or self.state_bits.shape[0] != batch_size:
			self._reset_state(batch_size, device=window_bits.device)

		# 1) State layer: [input_layer_output, prev_state_bits]
		state_layer_input = cat([window_bits, self.state_bits], dim=1)
		state_layer_output = self.state_layer(state_layer_input)  # [B, N_state]

		# 3) Output layer: [input_layer_output, current_state_bits]
		output_layer_output = self.output_layer(state_layer_output)  # [B, N_out]

		if self._is_update_state(update_state, window_bits):
			self.state_bits = state_layer_output.detach().clone()

		return state_layer_input, state_layer_output, state_layer_output, output_layer_output

	def _is_update_output(self, window_bits: Tensor) -> bool:
		return True

	def _is_update_state(self, update_state: bool, window_bits: Tensor) -> bool:
		return update_state

	def _materialize_step(self, context: dict) -> bool:
		"""
		Return:
			changed:				True if state has changed, False otherwise.
		"""
		# STATE layer
		state_changed = self.state_layer.commit(context["state_layer_input"], context["state_layer_output"], self._is_update_state(True, context["window_bits"]))
		# OUTPUT layer - the context won't become stale if the output has changed.
		if self._is_update_output(context["window_bits"]):
			self.output_layer.commit(context["output_layer_input"], context["output_layer_output"], True)
		return state_changed

	def _normalize_bits(self, bits: Tensor) -> Tensor:
		"""
		Ensure shape [1, n] and dtype uint8.
		"""
		if bits.ndim == 1:
			bits = bits.unsqueeze(0)
		if bits.dtype != uint8:
			bits = bits.to(uint8)
		return bits

	def _reset_state(self, batch_size: int, device) -> None:
		"""
		Initialize recurrent state to all False.
		"""
		self.state_bits = zeros(batch_size, self.state_layer.num_neurons, dtype=uint8, device=device) if self.state_layer.num_neurons > 0 else zeros(batch_size, 0, dtype=uint8, device=device)

	def _solve_output(self, context: dict, target_bits: Tensor) -> Tensor:
		"""
		Solve for hidden_T bits that produce target_bits via output_layer.

		IMPORTANT: We use target_bits as the desired state instead of solving from
		current state. This ensures the output layer learns a generalizable mapping
		(state → output) rather than memorizing at the current state.

		For identity output layers (common case), this means:
		  - output_layer[target] = target
		  - Desired state = target

		This fixes the "solver bias" bug where minimal-change preference caused
		the solver to keep the current (incorrect) state and memorize at that address.
		"""
		# Desired state = target (for identity/simple output mappings)
		# This ensures generalization: all inputs needing output=X get state=X
		desired_state_output_bits_t = target_bits.clone()
		if desired_state_output_bits_t.ndim == 2:
			desired_state_output_bits_t = desired_state_output_bits_t[0]

		# Commit output mapping: state=target → output=target
		self.output_layer.commit(desired_state_output_bits_t.unsqueeze(0), target_bits)
		return desired_state_output_bits_t.unsqueeze(0)

	# ------------------------------------------------------------------
	# Public API
	# ------------------------------------------------------------------
	def forward(self, input_bits: Tensor) -> Tensor:
		"""
	Inference: run over all windows in sequence and return decoded output of last step.
		"""
		self._reset_state(input_bits.shape[0], input_bits.device)
		input_bits = self._normalize_bits(input_bits)													# [1, n_total_bits] or [B, n_total_bits]
		assert input_bits.shape[0] == 1, "This forward() currently assumes batch_size=1"

		windows, output_layer_output = self.make_windows(input_bits), None		# list of [1, self.input_bits]
		for window_bits in windows:
			*_, output_layer_output = self._get_outputs(window_bits, update_state=True)

		return self.decoder.decode(output_layer_output)

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

	def train(self, windows: list[Tensor], target_bits: Tensor) -> None:
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
		# Run sequence and collect contexts
		n_steps = len(windows)

		if n_steps == 0:
			return

		windows = [self._normalize_bits(window) for window in windows]

		target_bits = self.decoder.encode(target_bits)
		contexts = self._get_contexts(windows, n_steps)
		context = contexts[-1]

		# Final output at n_steps-1
		final_output = context["output_layer_output"]
	# If already correct, you can optionally "materialize" EMPTY paths here (you already did that)
		if bool((final_output == target_bits).all()):
			# Reinforce EMPTY memories to answer FALSE
			for i in range(n_steps-1, -1, -1):
				if self._materialize_step(contexts[i]) and i > 0:
					contexts = self._get_contexts(windows, i)
			# Now, NN will converge correctly, nothing else to do.
			return

	# 1) Solve OUTPUT constraints at T-1:
	#    find desired hidden bits (input_out(T) + state_out(T)) that can yield target output.
		desired_state_output_bits_t = self._solve_output(context, target_bits)
		if desired_state_output_bits_t is None:
			return	# no solution even with override (should be rare and a limit of the architecture).

		# 2) Backprop through time over the STATE layer:
		#    For t = T-1 ... 0:
		#      Solve: state_layer_input(t) -> desired_state_out(t)
		#      Commit state layer at t
		#
		# Note: contexts[t]["state_layer_input"] = concat(input_in(t), state_out(t-1)) from baseline run
		for t in range(n_steps-1, -1, -1):
			context = contexts[t]
			changed, desired_state_output_bits_t = self._commit(context["window_bits"], context["state_layer_input"][0], desired_state_output_bits_t)
			if desired_state_output_bits_t is None:
				return
			if changed and t > 0:
				contexts = self._get_contexts(windows, t)

	# ------------------------------------------------------------------
	# XOR Pre-training for Parity
	# ------------------------------------------------------------------

	def pretrain_xor(self, neuron_idx: int = 0) -> bool:
		"""
		Pre-train a state neuron to compute XOR of input and previous state.

		For parity computation, the state layer needs to learn:
			new_state[i] = prev_state[i] XOR input[i]

		This method directly writes the XOR truth table to the specified
		state neuron's RAM, guaranteeing 100% generalization.

		Args:
			neuron_idx: Which state neuron to train (default: 0)

		Returns:
			True if successful, False if architecture doesn't support XOR

		Requirements:
			- input_bits >= 1
			- n_state_neurons >= neuron_idx + 1
			- State neuron must see both input bit and its previous state bit

		Example:
			>>> model = RAMRecurrentNetwork(input_bits=1, n_state_neurons=1, ...)
			>>> model.pretrain_xor()  # Now computes running XOR
			>>> model.pretrain_identity_output()  # Output = state
		"""
		if neuron_idx >= self.state_layer.num_neurons:
			return False
		if self.input_bits < 1:
			return False

		# XOR truth table: (prev_state, input) -> new_state
		# We need to find which bits the neuron sees and write XOR patterns
		xor_patterns = [
			# (prev_state, input) -> output
			([0, 0], 0),  # 0 XOR 0 = 0
			([0, 1], 1),  # 0 XOR 1 = 1
			([1, 0], 1),  # 1 XOR 0 = 1
			([1, 1], 0),  # 1 XOR 1 = 0
		]

		# The state layer input is [input_bits..., prev_state_bits...]
		# For XOR, we need input bit 0 and prev_state bit neuron_idx
		# Total input size: input_bits + n_state_neurons

		total_input = self.input_bits + self.state_layer.num_neurons
		input_bit_idx = 0  # First input bit
		state_bit_idx = self.input_bits + neuron_idx  # Corresponding state bit

		# Generate all possible inputs and write XOR for this neuron
		for prev_state, input_bit in [(0, 0), (0, 1), (1, 0), (1, 1)]:
			# Create input tensor with all combinations of other bits = 0
			inp = zeros(1, total_input, dtype=uint8)
			inp[0, input_bit_idx] = input_bit
			inp[0, state_bit_idx] = prev_state

			# Expected output: XOR
			expected = prev_state ^ input_bit
			out = zeros(1, self.state_layer.num_neurons, dtype=uint8)
			out[0, neuron_idx] = expected

			# Commit to memory
			self.state_layer.commit(inp, out)

		return True

	def pretrain_identity_output(self) -> bool:
		"""
		Pre-train output layer to be identity (output = state).

		For parity, after XOR state transition, the output should simply
		copy the state bit. This writes the identity mapping.

		Returns:
			True if successful
		"""
		if self.output_layer.num_neurons < 1:
			return False

		# Identity: for each state bit pattern, output the same pattern
		n_states = self.state_layer.num_neurons
		n_outputs = self.output_layer.num_neurons

		# Only train mappings where we copy state to output
		for state_val in range(min(2, 2 ** n_states)):
			inp = zeros(1, n_states, dtype=uint8)
			out = zeros(1, n_outputs, dtype=uint8)

			# Set state bits
			for b in range(n_states):
				inp[0, b] = (state_val >> b) & 1

			# Copy to output (as many bits as we can)
			for b in range(min(n_states, n_outputs)):
				out[0, b] = inp[0, b]

			self.output_layer.commit(inp, out)

		return True

	def pretrain_for_parity(self) -> bool:
		"""
		Pre-train network for parity computation.

		Configures state layer to compute XOR and output layer to be identity.
		After this, the network computes parity of input bit sequence with
		100% generalization to any sequence length.

		Returns:
			True if successful

		Example:
			>>> model = RAMRecurrentNetwork(
			...     input_bits=1,
			...     n_state_neurons=1,
			...     n_output_neurons=1,
			...     n_bits_per_state_neuron=2,  # Must see input + prev_state
			...     n_bits_per_output_neuron=1,
			... )
			>>> model.pretrain_for_parity()
			>>> # Now computes parity of any bit sequence
			>>> parity = model.forward(tensor([1, 0, 1, 1, 0]))  # = 1
		"""
		xor_ok = self.pretrain_xor(neuron_idx=0)
		identity_ok = self.pretrain_identity_output()
		return xor_ok and identity_ok

	def pretrain_state_function(self, mode: StateMode, neuron_idx: int = 0) -> bool:
		"""
		Pre-train state neuron with a specific boolean function.

		Args:
			mode: The state transition function to use
			neuron_idx: Which state neuron to train

		Returns:
			True if successful
		"""
		if neuron_idx >= self.state_layer.num_neurons:
			return False

		total_input = self.input_bits + self.state_layer.num_neurons
		input_bit_idx = 0
		state_bit_idx = self.input_bits + neuron_idx

		# Define function based on mode
		def compute_output(prev_state: int, input_bit: int) -> int:
			match mode:
				case StateMode.XOR:
					return prev_state ^ input_bit
				case StateMode.IDENTITY:
					return input_bit
				case StateMode.OR:
					return prev_state | input_bit
				case _:
					return 0

		# Write all patterns
		for prev_state in [0, 1]:
			for input_bit in [0, 1]:
				inp = zeros(1, total_input, dtype=uint8)
				inp[0, input_bit_idx] = input_bit
				inp[0, state_bit_idx] = prev_state

				out = zeros(1, self.state_layer.num_neurons, dtype=uint8)
				out[0, neuron_idx] = compute_output(prev_state, input_bit)

				self.state_layer.commit(inp, out)

		return True

	# -------------------------
	# Serialization
	# -------------------------

	def get_config(self) -> dict:
		"""Get configuration for serialization."""
		return {
			'input_bits': self.input_bits,
			'n_state_neurons': self.n_state_neurons,
			'n_output_neurons': self.n_output_neurons,
			'n_bits_per_state_neuron': self.n_bits_per_state_neuron,
			'n_bits_per_output_neuron': self.n_bits_per_output_neuron,
			'use_hashing': self.use_hashing,
			'hash_size': self.hash_size,
			'rng': self.rng,
			'max_iters': self.max_iters,
			'output_mode': self.output_mode.value if hasattr(self.output_mode, 'value') else self.output_mode,
		}

	@classmethod
	def from_config(cls, config: dict) -> "RAMRecurrentNetwork":
		"""Create model from configuration."""
		return cls(**config)

	def save(self, path: str) -> None:
		"""Save model to file."""
		from wnn.ram.core.serialization import save_model
		save_model(self, path)
