from wnn.ram.RAMRecurrentNetwork import RAMRecurrentNetwork
from wnn.ram.encoders_decoders import OutputMode

from torch import Tensor
from typing import Optional

class RAMSequence(RAMRecurrentNetwork):
	"""
	Sequence-to-Sequence extension of RAMRecurrentNetwork.

	Enables training with one target per timestep (multi-target mode),
	suitable for tasks like next-character prediction where each step
	should produce an output.

	Inherits the base architecture from RAMRecurrentNetwork but overrides
	train() to support supervising all timesteps, not just the final one.
	"""

	def __init__(
		self,
		input_bits: int,
		n_state_neurons: int,
		n_output_neurons: int,
		n_bits_per_state_neuron: int,
		n_bits_per_output_neuron: int,
		output_mode: OutputMode = OutputMode.RAW,
		use_hashing: bool = False,
		hash_size: int = 1024,
		rng: int | None = None,
	):
		"""
		Same parameters as RAMRecurrentNetwork.
		See RAMRecurrentNetwork.__init__ for parameter details.
		"""
		super().__init__(
			input_bits=input_bits,
			n_state_neurons=n_state_neurons,
			n_output_neurons=n_output_neurons,
			n_bits_per_state_neuron=n_bits_per_state_neuron,
			n_bits_per_output_neuron=n_bits_per_output_neuron,
			output_mode=output_mode,
			use_hashing=use_hashing,
			hash_size=hash_size,
			rng=rng,
		)

	def train(self, windows: list[Tensor], targets: str | list[str]) -> None:
		"""
		Multi-target EDRA-BPTT for sequence-to-sequence learning.

		Unlike RAMRecurrentNetwork.train() which supervises only the final output,
		this method supervises EACH timestep with its own target.

		Args:
			windows: List of input tensors, one per timestep [1, input_bits]
			targets: Either a string (one char per timestep) or list of strings

		Algorithm:
			1) Run forward pass through all timesteps, collecting contexts
			2) Check each output against its corresponding target
			3) Find the FIRST error (earliest timestep with wrong output)
			4) Run EDRA-BPTT from that error back to timestep 0
			5) Subsequent training epochs will fix later errors

		Example:
			For "next character" task with input="ABCD", targets="BCDE":
			- Timestep 0: input="A", expect output="B"
			- Timestep 1: input="B", expect output="C"
			- Timestep 2: input="C", expect output="D"
			- Timestep 3: input="D", expect output="E"
		"""
		n_steps = len(windows)

		if n_steps == 0:
			return

		# Normalize windows
		windows = [self._normalize_bits(window) for window in windows]

		# Convert targets to list
		if isinstance(targets, str):
			target_list = list(targets)
		else:
			target_list = targets

		if len(target_list) != n_steps:
			raise ValueError(f"Expected {n_steps} targets, got {len(target_list)}")

		# Encode all targets
		encoded_targets = [self.decoder.encode(t) for t in target_list]

		# Run forward pass and collect contexts
		contexts = self._get_contexts(windows, n_steps)

		# Check all outputs and find FIRST error
		first_error_t = None
		for t in range(n_steps):
			output = contexts[t]["output_layer_output"]
			target = encoded_targets[t]
			if not bool((output == target).all()):
				first_error_t = t
				break

		# If all outputs correct, materialize EMPTY paths and return
		if first_error_t is None:
			for i in range(n_steps-1, -1, -1):
				if self._materialize_step(contexts[i]) and i > 0:
					contexts = self._get_contexts(windows, i)
			return

		# Run EDRA-BPTT from the first error
		error_t = first_error_t
		target_bits = encoded_targets[error_t]
		context = contexts[error_t]

		# Solve output layer at error timestep
		desired_state_output_bits = self._solve_output(context, target_bits)
		if desired_state_output_bits is None:
			return  # No solution found

		# Backprop through time from error_t down to timestep 0
		for t in range(error_t, -1, -1):
			context = contexts[t]
			changed, desired_state_output_bits = self._commit(
				context["window_bits"],
				context["state_layer_input"][0],
				desired_state_output_bits
			)
			if desired_state_output_bits is None:
				return
			if changed and t > 0:
				# Memory changed, need to recompute contexts
				contexts = self._get_contexts(windows, t)

	def train_masked(
		self,
		windows: list[Tensor],
		targets: str | list[str],
		timestep_mask: Optional[list[bool]] = None
	) -> None:
		"""
		Masked multi-target EDRA-BPTT for selective sequence learning.

		Like train(), but only considers timesteps where mask[t] is True.
		State evolution happens for ALL timesteps (maintaining temporal coherence),
		but EDRA error correction only triggers for masked timesteps.

		Args:
			windows: List of input tensors, one per timestep [1, input_bits]
			targets: Either a string (one char per timestep) or list of strings
			timestep_mask: Boolean mask indicating which timesteps to train on.
			               If None, trains on all timesteps (same as train()).

		Use case: KV routing where only certain timesteps route to this head.
		"""
		n_steps = len(windows)

		if n_steps == 0:
			return

		# Default mask: all True
		if timestep_mask is None:
			timestep_mask = [True] * n_steps

		if len(timestep_mask) != n_steps:
			raise ValueError(f"Mask length {len(timestep_mask)} != n_steps {n_steps}")

		# Normalize windows
		windows = [self._normalize_bits(window) for window in windows]

		# Convert targets to list
		if isinstance(targets, str):
			target_list = list(targets)
		else:
			target_list = targets

		if len(target_list) != n_steps:
			raise ValueError(f"Expected {n_steps} targets, got {len(target_list)}")

		# Encode all targets
		encoded_targets = [self.decoder.encode(t) for t in target_list]

		# Run forward pass and collect contexts (ALL timesteps for state evolution)
		contexts = self._get_contexts(windows, n_steps)

		# Check MASKED outputs and find FIRST error among them
		first_error_t = None
		for t in range(n_steps):
			if not timestep_mask[t]:
				continue  # Skip non-masked timesteps
			output = contexts[t]["output_layer_output"]
			target = encoded_targets[t]
			if not bool((output == target).all()):
				first_error_t = t
				break

		# If all masked outputs correct, materialize EMPTY paths and return
		if first_error_t is None:
			for i in range(n_steps-1, -1, -1):
				if timestep_mask[i]:  # Only materialize masked timesteps
					if self._materialize_step(contexts[i]) and i > 0:
						contexts = self._get_contexts(windows, i)
			return

		# Run EDRA-BPTT from the first masked error
		error_t = first_error_t
		target_bits = encoded_targets[error_t]
		context = contexts[error_t]

		# Solve output layer at error timestep
		desired_state_output_bits = self._solve_output(context, target_bits)
		if desired_state_output_bits is None:
			return  # No solution found

		# Backprop through time from error_t down to timestep 0
		# Note: We backprop through ALL timesteps (not just masked ones)
		# because state evolution requires the full chain
		for t in range(error_t, -1, -1):
			context = contexts[t]
			changed, desired_state_output_bits = self._commit(
				context["window_bits"],
				context["state_layer_input"][0],
				desired_state_output_bits
			)
			if desired_state_output_bits is None:
				return
			if changed and t > 0:
				# Memory changed, need to recompute contexts
				contexts = self._get_contexts(windows, t)
