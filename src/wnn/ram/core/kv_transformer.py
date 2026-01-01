
from wnn.ram.architecture import KVSpec
from wnn.ram.enums import MemoryVal
from wnn.ram.core import RAMLayer
from wnn.ram.core.recurrent_network import RAMRecurrentNetwork
from wnn.ram.encoders_decoders import OutputMode

from typing import Optional

from torch import cat
from torch import full
from torch import Tensor

class RAMTransformer(RAMRecurrentNetwork):
	"""
	RAM multi-head KV memory with hard key routing.
	"""

	def __init__(
		self,
		spec: KVSpec,
		neurons_per_head: int,
		n_bits_per_state_neuron: int,
		n_bits_per_output_neuron: int,
		use_hashing: bool = False,
		hash_size: int = 1024,
		rng: Optional[int] = None,
		max_iters: int = 4,
		output_mode: OutputMode = OutputMode.RAW,
	):
		self.spec = spec
		self.num_heads = 1 << spec.k_bits
		self.neurons_per_head = neurons_per_head

		total_state_neurons = self.num_heads * neurons_per_head

		super().__init__(
			input_bits=spec.window_bits,
			n_state_neurons=total_state_neurons,
			n_output_neurons=spec.v_bits,
			n_bits_per_state_neuron=n_bits_per_state_neuron,
			n_bits_per_output_neuron=n_bits_per_output_neuron,
			use_hashing=use_hashing,
			hash_size=hash_size,
			rng=rng,
			max_iters=max_iters,
			output_mode=output_mode,
		)
		self.head_dim = self.state_layer.num_neurons // self.num_heads

		assert self.state_layer.num_neurons == self.num_heads * self.head_dim

		# Precompute head slices [num_heads, head_dim]
		self._head_slices = [slice(h * self.head_dim, (h + 1) * self.head_dim) for h in range(self.num_heads)]


	def _create_output_layer(self,
		num_neurons: int,
		n_bits_per_neuron: int,
		use_hashing: bool = False,
		hash_size: int = 1024,
		rng: Optional[int] = None,
	) -> RAMLayer:
		return RAMLayer(
			total_input_bits=self.spec.k_bits + self.neurons_per_head,
			num_neurons=num_neurons,
			n_bits_per_neuron=n_bits_per_neuron,
			use_hashing=use_hashing,
			hash_size=hash_size,
			rng=rng,
		)
	# ------------------------------------------------------------
	# Head utilities
	# ------------------------------------------------------------

	def _decode_key(self, window_bits: Tensor) -> int:
		key_bits = window_bits[:, :self.spec.k_bits]
		key = 0
		for b in key_bits[0]:
			key = (key << 1) | int(b)
		return key

	def _head_slice(self, key: int) -> slice:
		return self._head_slices[key]

	def _is_update_output(self, window_bits: Tensor) -> bool:
		return self.is_query(window_bits)

	def _is_update_state(self, update_state: bool, window_bits: Tensor) -> bool:
		return super()._is_update_state(update_state, window_bits) and not self.is_query(window_bits)

	def is_query(self, window_bits: Tensor) -> bool:
		value_bits = window_bits[:, -self.spec.v_bits:]
		return bool((value_bits == 0).all())


	# ------------------------------------------------------------
	# Overrides
	# ------------------------------------------------------------

	def _calculate_state_output(self, window_bits: Tensor, desired_state_output: Tensor) -> tuple[Optional[slice], Tensor]:
		head = self._head_slice(self._decode_key(window_bits))
		# desired_state_output should represent the HEAD target only (head_dim bits)
		# Normalize to 1D [head_dim]
		if desired_state_output.ndim == 2:
			head_target_1d = desired_state_output[0]          # [head_dim]
		else:
			head_target_1d = desired_state_output              # [head_dim]

		# Build full state target as 1D then expand to [1, N_state]
		masked_state_target_1d = full(
			(self.state_layer.num_neurons,),
			fill_value=MemoryVal.EMPTY,
			dtype=head_target_1d.dtype,
			device=head_target_1d.device,
		)
		masked_state_target_1d[head] = head_target_1d

		return head, masked_state_target_1d.unsqueeze(0)       # ✅ [1, N_state]


	def _return_state_output(self, window_bits: Tensor, desired_previous_state_input: Tensor, head: slice) -> Tensor:
		# desired_previous_state_input is [1, N_in + N_state]
		prev_state = desired_previous_state_input[:, window_bits.shape[1]:]  # [1, N_state]
		return prev_state[:, head]                                           # ✅ [1, head_dim]

	def _get_outputs(self, window_bits: Tensor, update_state: bool = False):
		batch_size = window_bits.shape[0]

		if self.state_bits is None or self.state_bits.shape[0] != batch_size:
			self._reset_state(batch_size, device=window_bits.device)

		head = self._head_slice(self._decode_key(window_bits))

		# ---- STATE LAYER (full width, but we only commit head)
		state_layer_input = cat([window_bits, self.state_bits], dim=1)
		full_state_output = self.state_layer(state_layer_input)

		# ---- UPDATE STATE (writes only, head only)
		if update_state and not self.is_query(window_bits):
			self.state_bits[:, head] = full_state_output[:, head]

		# ---- OUTPUT (queries only, head only)
		key_bits = window_bits[:, :self.spec.k_bits]
		output_layer_input = cat([key_bits, full_state_output[:, head]], dim=1)
		output_layer_output = self.output_layer(output_layer_input)

		return state_layer_input, full_state_output, output_layer_input, output_layer_output

	def _solve_output(self, context: dict, target_bits: Tensor) -> Tensor:
		# keep key bits immutable during solve
		desired_output_in_bits_t = self.output_layer.solve(
				context["output_layer_input"][0],									# 1D [k_bits + n_state]
				target_bits,
				self.spec.k_bits)
		if desired_output_in_bits_t is None:
				return None

		desired_output_in_bits_t = desired_output_in_bits_t.unsqueeze(0)
		self.output_layer.commit(desired_output_in_bits_t, target_bits)

		# For BPTT back into state layer, strip key bits off:
		return desired_output_in_bits_t[:, self.spec.k_bits:]		# 1D [n_state]
