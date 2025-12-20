from wnn.ram import RAMLayer
from wnn.ram.decoders import OutputMode
from wnn.ram.decoders import TransformerDecoder
from wnn.ram.decoders import TransformerDecoderFactory


from typing import Optional

from torch import cat
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

		self.input_bits = int(input_bits)
		self.max_iters = max_iters
		self.output_mode = output_mode
		self.decoder: TransformerDecoder = TransformerDecoderFactory.create(output_mode, n_output_neurons)

		# -------------------------
		# Layers
		# -------------------------
		# State layer: sees [input_layer_output, previous_state_bits]
		self.state_layer = RAMLayer(
			total_input_bits=input_bits + n_state_neurons,
			num_neurons=n_state_neurons,
			n_bits_per_neuron=n_bits_per_state_neuron,
			use_hashing=use_hashing,
			hash_size=hash_size,
			rng=None if rng is None else rng,
		)

		# Output layer: sees [input_layer_output, current_state_bits]
		self.output_layer = RAMLayer(
			total_input_bits=n_state_neurons,
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

	# ------------------------------------------------------------------
	# Utility helpers
	# ------------------------------------------------------------------

	def _commit(self, window_bits: Tensor, current_state_input: Tensor, desired_state_output: Tensor) -> tuple[bool, Optional[Tensor]]:
		"""
		Return:
			changed:				True if any change happened, False otherwise.
			previous bits:	Tensor
		"""
		if desired_state_output is None:
			return False, None
		assert current_state_input.shape[0] == self.input_bits + self.state_layer.num_neurons
		desired_previous_state_input = self.state_layer.solve(current_state_input, desired_state_output, int(window_bits.shape[1]))
		if desired_previous_state_input is None:
			return False, None
		desired_previous_state_input = desired_previous_state_input.unsqueeze(0)	# [1, N_in+N_state]
		changed = self.state_layer.commit(desired_previous_state_input, desired_state_output)
		return (changed, desired_previous_state_input[:, window_bits.shape[1]:])

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
			(state_layer_input, state_layer_output, output_layer_output) = self._get_outputs(window_bits, update_state=True)
			contexts.append({
					"window_bits":					window_bits,
					"state_layer_input":		state_layer_input,
					"state_layer_output":		state_layer_output,
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
				input_layer_output,  # [B, N_in]
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

		if update_state and self.state_layer.num_neurons > 0:
			self.state_bits = state_layer_output.detach().clone()

		return state_layer_input, state_layer_output, output_layer_output

	def _materialize_step(self, context) -> bool:
		"""
		Return:
			changed:				True if state has changed, False otherwise.
		"""
		# STATE layer
		state_changed = self.state_layer.commit(context["state_layer_input"], context["state_layer_output"], True)
		# OUTPUT layer - the context won't become stale if the output has changed.
		self.output_layer.commit(context["state_layer_output"], context["output_layer_output"], True)
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
    # If already correct, you can optionally “materialize” EMPTY paths here (you already did that)
		if bool((final_output == target_bits).all()):
			# Reinforce EMPTY memories to answer FALSE
			for i in range(n_steps-1, -1, -1):
				if self._materialize_step(contexts[i]) and i > 0:
					contexts = self._get_contexts(windows, i)
			# Now, NN will converge correctly, nothing else to do.
			return

    # 1) Solve OUTPUT constraints at T-1:
    #    find desired hidden bits (input_out(T) + state_out(T)) that can yield target output.
		desired_state_output_bits_t = self.output_layer.solve(context["state_layer_output"][0], target_bits, 0)
		if desired_state_output_bits_t is None:
			return	# no solution even with override (should be rare and a limit of the architecture).

    # Commit output mapping: hidden_T -> target_bits
		desired_state_output_bits_t = desired_state_output_bits_t.unsqueeze(0)
		self.output_layer.commit(desired_state_output_bits_t, target_bits)


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
