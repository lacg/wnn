
from wnn.ram.architecture import KVSpec
from wnn.ram.enums import MemoryVal
from wnn.ram.core import RAMLayer
from wnn.ram.core.recurrent_network import RAMRecurrentNetwork
from wnn.ram.encoders_decoders import OutputMode

from typing import Optional

from torch import arange
from torch import cat
from torch import full
from torch import long
from torch import randperm
from torch import Tensor
from torch import zeros

class RAMKVMemory(RAMRecurrentNetwork):
	"""
	RAM multi-head KV memory with hard key routing.

	Uses recurrent state partitioned into heads, where key bits
	determine which head to read/write. Value bits all zero = query.

	Architecture:
	  [k_bits | v_bits] → state_layer → head_state → output_layer → output

	Different from RAMTransformer (transformers/transformer.py) which is
	a stacked block architecture with attention + FFN.
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

		# KEY-ONLY ADDRESSING: State layer sees [key_bits, state], not [window_bits, state]
		# This ensures writes and queries hash to the same address (both use key only)
		super().__init__(
			input_bits=spec.k_bits,  # Key-only for state layer addressing
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

	# ------------------------------------------------------------
	# Structured connections: ensure neurons see window bits
	# ------------------------------------------------------------

	def _create_structured_connections(
		self,
		total_input_bits: int,
		num_neurons: int,
		n_bits_per_neuron: int,
		priority_bits: int,
		rng: Optional[int] = None,
	) -> Tensor:
		"""
		Create connections that guarantee all neurons observe priority_bits first.

		For KV memory, priority_bits = window_bits (key + value).
		This ensures neurons can learn the key-value mapping rather than
		randomly observing mostly state bits.

		Args:
			total_input_bits: Total input dimension
			num_neurons: Number of neurons in layer
			n_bits_per_neuron: How many bits each neuron observes
			priority_bits: Number of leading bits that ALL neurons must see
			rng: Random seed

		Returns:
			Tensor[num_neurons, n_bits_per_neuron] connection indices
		"""
		if rng is not None:
			from torch import manual_seed
			manual_seed(rng)

		connections = []
		remaining_bits = total_input_bits - priority_bits

		for _ in range(num_neurons):
			# Start with all priority bits (window bits: key + value)
			neuron_conn = list(range(priority_bits))

			# Fill remaining slots with random state bits
			extra_needed = n_bits_per_neuron - priority_bits
			if extra_needed > 0 and remaining_bits > 0:
				# Random selection from non-priority bits
				perm = randperm(remaining_bits)[:extra_needed]
				extra_bits = (perm + priority_bits).tolist()
				neuron_conn.extend(extra_bits)

			# If we still need more (rare), repeat priority bits
			while len(neuron_conn) < n_bits_per_neuron:
				neuron_conn.append(neuron_conn[len(neuron_conn) % priority_bits])

			connections.append(neuron_conn[:n_bits_per_neuron])

		from torch import tensor
		return tensor(connections, dtype=long)

	def _create_state_layer(
		self,
		total_input_bits: int,
		num_neurons: int,
		n_bits_per_neuron: int,
		use_hashing: bool,
		hash_size: int,
		rng: Optional[int],
	) -> RAMLayer:
		"""
		Override to use structured connections for state layer.

		KEY-ONLY ADDRESSING: State layer input is [key_bits, state_bits].
		All neurons observe key bits first, ensuring writes and queries
		(which differ only in value bits, not present here) hash to same address.
		"""
		# Priority bits = k_bits only (not window_bits)
		# Value is NOT part of addressing - it flows through EDRA training
		priority_bits = self.spec.k_bits

		# Create structured connections
		connections = self._create_structured_connections(
			total_input_bits=total_input_bits,
			num_neurons=num_neurons,
			n_bits_per_neuron=n_bits_per_neuron,
			priority_bits=priority_bits,
			rng=rng,
		)

		return RAMLayer(
			total_input_bits=total_input_bits,
			num_neurons=num_neurons,
			n_bits_per_neuron=n_bits_per_neuron,
			connections=connections,
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
		"""
		Override to use structured connections for output layer.

		Output layer input: [k_bits, head_state_bits]
		With direct value storage, head_state starts with value_bits.
		All neurons must observe BOTH key_bits AND value_bits for correct decoding.
		"""
		total_input_bits = self.spec.k_bits + self.neurons_per_head

		# Priority bits = k_bits + v_bits (key AND value bits)
		# Value bits are stored at head_state[0:v_bits], so they're at
		# indices [k_bits : k_bits + v_bits] in output_layer_input
		priority_bits = self.spec.k_bits + self.spec.v_bits

		# Create structured connections
		connections = self._create_structured_connections(
			total_input_bits=total_input_bits,
			num_neurons=num_neurons,
			n_bits_per_neuron=n_bits_per_neuron,
			priority_bits=priority_bits,
			rng=rng,
		)

		return RAMLayer(
			total_input_bits=total_input_bits,
			num_neurons=num_neurons,
			n_bits_per_neuron=n_bits_per_neuron,
			connections=connections,
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

	def _commit(self, window_bits: Tensor, current_state_input: Tensor, desired_state_output: Tensor):
		"""
		Override to directly commit desired output at CURRENT address.

		CRITICAL: In recurrent networks, we can't change the input (it's determined
		by sequence history). We can only change what's STORED at the current address.

		Unlike base _commit which uses solve() to find a different input, we:
		1. Compute the address from current_state_input (the actual input)
		2. Directly write desired_state_output at that address
		"""
		if desired_state_output is None:
			return False, None

		head = self._head_slice(self._decode_key(window_bits))

		# Extract head target (desired_state_output is already just head_dim bits)
		if desired_state_output.ndim == 2:
			head_target = desired_state_output[0]  # [head_dim]
		else:
			head_target = desired_state_output  # [head_dim]

		# Get addresses from CURRENT input (not a hypothetical solved input)
		current_input_2d = current_state_input.unsqueeze(0)
		addresses = self.state_layer.get_addresses(current_input_2d)[0]  # [N_neurons]

		from torch import arange, long
		head_indices = arange(head.start, head.stop, device=head_target.device, dtype=long)
		head_addresses = addresses[head]
		head_bits = head_target

		# Directly commit the desired output at current address
		changed = self.state_layer.memory.explore_batch(
			head_indices,
			head_addresses,
			head_bits,
			allow_override=True  # Allow override to fix conflicts
		)

		# Return current head state for BPTT (unchanged, since we wrote to memory)
		current_state = current_state_input[self.spec.k_bits:]  # state portion
		return changed, current_state[head].unsqueeze(0)  # [1, head_dim]

	def _calculate_state_output(self, window_bits: Tensor, desired_state_output: Tensor) -> tuple[Optional[slice], Tensor]:
		"""Legacy method - no longer used after _commit override."""
		head = self._head_slice(self._decode_key(window_bits))
		if desired_state_output.ndim == 2:
			head_target_1d = desired_state_output[0]
		else:
			head_target_1d = desired_state_output

		masked_state_target_1d = full(
			(self.state_layer.num_neurons,),
			fill_value=MemoryVal.EMPTY,
			dtype=head_target_1d.dtype,
			device=head_target_1d.device,
		)
		masked_state_target_1d[head] = head_target_1d

		return head, masked_state_target_1d.unsqueeze(0)


	def _return_state_output(self, window_bits: Tensor, desired_previous_state_input: Tensor, head: slice) -> Tensor:
		# desired_previous_state_input is [1, k_bits + N_state]
		# State layer uses key-only addressing, so skip k_bits (not window_bits)
		prev_state = desired_previous_state_input[:, self.spec.k_bits:]  # [1, N_state]
		return prev_state[:, head]                                        # [1, head_dim]

	def _get_outputs(self, window_bits: Tensor, update_state: bool = False):
		batch_size = window_bits.shape[0]

		if self.state_bits is None or self.state_bits.shape[0] != batch_size:
			self._reset_state(batch_size, device=window_bits.device)

		head = self._head_slice(self._decode_key(window_bits))
		key_bits = window_bits[:, :self.spec.k_bits]
		value_bits = window_bits[:, self.spec.k_bits:]

		# ---- DIRECT VALUE STORAGE (bypass state layer for KV memory)
		# For writes: directly store value bits in head state
		# For queries: read from head state and decode via output layer
		if update_state and not self.is_query(window_bits):
			# WRITE: Store value encoding directly in head state
			# Encode value in first v_bits neurons, zeros elsewhere
			self.state_bits[:, head] = 0
			head_size = head.stop - head.start
			v_bits_to_copy = min(self.spec.v_bits, head_size)
			self.state_bits[:, head.start:head.start + v_bits_to_copy] = value_bits[:, :v_bits_to_copy]

		# ---- STATE LAYER (for context tracking, not value storage)
		state_layer_input = cat([key_bits, self.state_bits], dim=1)
		full_state_output = self.state_layer(state_layer_input)

		# ---- OUTPUT LAYER: [key, head_state] → value
		output_layer_input = cat([key_bits, self.state_bits[:, head]], dim=1)
		output_layer_output = self.output_layer(output_layer_input)

		return state_layer_input, full_state_output, output_layer_input, output_layer_output

	def train(self, windows: list, target_bits) -> None:
		"""
		Training for direct value storage KV memory.

		With direct value storage, we only need to train the OUTPUT LAYER:
		- Writes directly store value bits in head state (no learning needed)
		- Output layer learns: [key, head_state] → value

		Training flow:
		1. Run forward pass (writes update state directly)
		2. Train output layer on the query step's input/output mapping
		"""
		n_steps = len(windows)
		if n_steps == 0:
			return

		windows = [self._normalize_bits(w) for w in windows]
		target_bits = self.decoder.encode(target_bits)

		# Run forward pass - writes will directly store values in state
		self._reset_state(batch_size=1, device=windows[0].device)
		for t, window in enumerate(windows):
			is_last = (t == n_steps - 1)
			# For all but last: update_state=True (writes store values)
			# For last (query): update_state=False
			_, _, output_layer_input, _ = self._get_outputs(
				window, update_state=not is_last
			)

		# Train output layer: current input → target
		# output_layer_input from the query step contains [key, head_state]
		self.output_layer.commit(output_layer_input, target_bits)

	def _solve_output(self, context: dict, target_bits) -> "Tensor":
		"""
		Override to force distinctive state patterns for each value.

		Instead of letting solve() pick any solution (which defaults to zeros),
		we explicitly construct a state pattern that encodes the target value.
		This ensures different values produce different head states.
		"""
		from torch import zeros, uint8

		# Extract key bits from context
		key_bits = context["output_layer_input"][0, :self.spec.k_bits]

		# Create a distinctive head state pattern based on target value
		# Strategy: encode target_bits directly in the head state
		# Use enough neurons to distinguish all possible values
		head_state = zeros(self.neurons_per_head, dtype=uint8, device=target_bits.device)

		# Encode target value in the first v_bits neurons of the head
		# This creates distinctive patterns: value=1→[0,1,0,...], value=2→[1,0,0,...], etc.
		target_flat = target_bits.flatten() if target_bits.ndim > 1 else target_bits
		for i in range(min(len(target_flat), len(head_state))):
			head_state[i] = target_flat[i]

		# Build the complete output layer input: [key_bits, head_state]
		desired_output_in_bits_t = cat([key_bits, head_state]).unsqueeze(0)

		# Commit output mapping: this input -> target_bits
		self.output_layer.commit(desired_output_in_bits_t, target_bits)

		# Return desired head state for BPTT into state layer
		return desired_output_in_bits_t[:, self.spec.k_bits:]  # [1, head_dim]

	# -------------------------
	# Serialization
	# -------------------------

	def get_config(self) -> dict:
		"""Get configuration for serialization."""
		return {
			'spec': self.spec.to_dict() if hasattr(self.spec, 'to_dict') else {
				'k_bits': self.spec.k_bits,
				'v_bits': self.spec.v_bits,
				'query_value': self.spec.query_value,
			},
			'neurons_per_head': self.neurons_per_head,
			'n_bits_per_state_neuron': self.n_bits_per_state_neuron,
			'n_bits_per_output_neuron': self.n_bits_per_output_neuron,
			'use_hashing': self.use_hashing,
			'hash_size': self.hash_size,
			'rng': self.rng,
			'max_iters': self.max_iters,
			'output_mode': self.output_mode.value if hasattr(self.output_mode, 'value') else self.output_mode,
		}

	@classmethod
	def from_config(cls, config: dict) -> "RAMKVMemory":
		"""Create model from configuration."""
		spec_dict = config.pop('spec')
		spec = KVSpec(**spec_dict)
		return cls(spec=spec, **config)

	def save(self, path: str) -> None:
		"""Save model to file."""
		from wnn.ram.core.serialization import save_model
		save_model(self, path)
