from typing import Optional

from torch import bool as tbool
from torch import cat
from torch import device
from torch import int64
from torch import nonzero
from torch import randint
from torch import zeros
from torch import Tensor
from torch.nn import Module

from wnn.ram.RAMLayer import RAMLayer
from wnn.ram.Memory import MemoryVal

class RAMTransformer(Module):
	"""
	RAM-based recurrent cell with 3 RAMLayers:
		- input_layer:  connected to raw input bits
		- state_layer:  connected to [input_layer output, previous state]
		- output_layer: connected to [input_layer output, current state]

	At each forward step:
		input_layer_output	= input_layer(input)
		state_layer_output  = state_layer(cat(input_layer_output, prev_state))
		output        			= output_layer(cat(input_layer_output, state_layer_output))
		state(t+1) 					= state_layer_output
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
	):
		super().__init__()

		self.input_bits = int(input_bits)
		self.n_input_neurons = int(n_input_neurons)
		self.n_state_neurons = int(n_state_neurons)
		self.n_output_neurons = int(n_output_neurons)

		# --- RAM layers ---

		# Input layer: sees raw input bits only
		self.input_layer = RAMLayer(
			total_input_bits=self.input_bits,
			num_neurons=self.n_input_neurons,
			n_bits_per_neuron=n_bits_per_input_neuron,
			use_hashing=use_hashing,
			hash_size=hash_size,
			rng=None if rng is None else rng,
		)

		# State layer: sees [input_layer_output, previous_state]
		self.state_layer = RAMLayer(
			total_input_bits=self.n_input_neurons + self.n_state_neurons,
			num_neurons=self.n_state_neurons,
			n_bits_per_neuron=n_bits_per_state_neuron,
			use_hashing=use_hashing,
			hash_size=hash_size,
			rng=None if rng is None else (rng + 1),
		)

		# Output layer: sees [input_layer_output, current_state]
		self.output_layer = RAMLayer(
			total_input_bits=self.n_input_neurons + self.n_state_neurons,
			num_neurons=self.n_output_neurons,
			n_bits_per_neuron=n_bits_per_output_neuron,
			use_hashing=use_hashing,
			hash_size=hash_size,
			rng=None if rng is None else (rng + 2),
		)

		# recurrent state: [batch_size, n_state_neurons] (initialized lazily)
		self.state_bits: Optional[Tensor] = None


	def reset_state(self, batch_size: int, device: Optional[device] = None) -> None:
		"""
		Reset recurrent state to all zeros for a given batch size.
		Must be called before processing a new sequence / batch.
		"""
		if device is None:
			device = next(self.parameters()).device

		self.state_bits = zeros(
			batch_size,
			self.n_state_neurons,
			dtype=tbool,
			device=device,
		)


	def forward(self, input: Tensor, update_state: bool = True) -> Tensor:
		"""
		Forward one time step.

		input: [batch_size, input_bits]  (bool or {0,1})
		update_state: whether to update internal state_bits with the new state.

		returns:
			output: [batch_size, n_output_neurons]
		"""
		if input.dtype != tbool:
			input = input.to(tbool)

		batch_size = input.shape[0]
		device = input.device

		# lazy state init if needed
		if self.state_bits is None or self.state_bits.shape[0] != batch_size:
			self.reset_state(batch_size, device=device)

		# --- input layer ---
		input_layer_output = self.input_layer(input)		# [batch_size, n_input_neurons], bool

		# --- state layer ---
		state_input = cat([input_layer_output, self.state_bits], dim=1)		# [batch_size, n_input_neurons + n_state_neurons]
		state_layer_output = self.state_layer(state_input)						# [batch_size, n_state_neurons]

		# --- output layer ---
		output_input = cat([input_layer_output, state_layer_output], dim=1)			# [batch_size, n_input_neurons + n_state_neurons]
		output = self.output_layer(output_input)							# [batch_size, n_output_neurons]

		# update recurrent state
		if update_state:
			self.state_bits = state_layer_output.detach().clone()

		return output


	def train_one(self, input_bits: Tensor, target_bits: Tensor, max_iters: int = 8) -> None:
		"""
		EDRA A1.2-R for the full 3-layer RAMTransformer:
		 - trains input layer
		 - trains state layer
		 - trains output layer
		 - random flip influence
		 - state transitions handled
		"""
		def get_index(neuron_idx: Tensor, addresses: Tensor, bits: Tensor, layer: RAMLayer) -> int:
			neuron = neuron_idx.item()
			address = addresses[0, neuron].item()

			# desired bit for this neuron
			desired = bits[0, neuron].item()

			# write correct value
			layer.set_memory(neuron, address, (MemoryVal.VAL1.value if desired else MemoryVal.VAL0.value))

			# RANDOM INFLUENCE:
			# randomly flip 1 contributing hidden bit in output_layer_input
			return layer.select_connection(neuron)


		# -----------------------------------------
		# Fix each conflicting neurons
		# -----------------------------------------
		def fix_conflicting(layer: RAMLayer, input: Tensor, conflicts: Tensor, target: Tensor, change_state: bool = True) -> None:
			# compute addresses for output layer
			# address dtype = int64
			addresses = layer.get_addresses(input)
			conflicting_neurons = nonzero(conflicts[0], as_tuple=False)
			for neuron_idx in conflicting_neurons:
				src_bit_index = get_index(neuron_idx, addresses, target, layer)

				# find which layer the contributing bit belongs to
				if src_bit_index < self.n_input_neurons:
					# influence input layer
					self.input_layer.flip_memory(src_bit_index, input_layer_output)
				elif change_state:
					# influence state layer
					self.state_layer.flip_memory(src_bit_index - self.n_input_neurons, state_layer_output)


		norm_bool = lambda bits: bits if bits.dtype == tbool else bits.to(tbool)
		input_bits = norm_bool(input_bits)
		target_bits = norm_bool(target_bits)

		batch_size = input_bits.shape[0]
		device = input_bits.device

		# lazy state init
		if self.state_bits is None or self.state_bits.shape[0] != batch_size:
			self.reset_state(batch_size, device)

		# iterative stabilization
		for _ in range(max_iters):
			# -----------------------------------------
			# 1. Forward pass (no state update)
			# -----------------------------------------
			input_layer_output = self.input_layer(input_bits)
			state_input = cat([input_layer_output, self.state_bits], dim=1)
			state_layer_output = self.state_layer(state_input)
			output_layer_input = cat([input_layer_output, state_layer_output], dim=1)
			output_layer_output = self.output_layer(output_layer_input)

			# -----------------------------------------
			# 2. Output conflicts
			# -----------------------------------------
			output_conflicts = (output_layer_output != target_bits)  # [batch_size, n_output_neurons]

			if not output_conflicts.any():
				# No conflicts — stable
				self.state_bits = state_layer_output.detach().clone()
				return

			# -----------------------------------------
			# 3. Fix each conflicting output neuron
			# -----------------------------------------
			# compute addresses for output layer
			# address dtype = int64
			fix_conflicting(self.output_layer, output_layer_input, output_conflicts, target_bits)

			# -----------------------------------------
			# 4. State-layer stabilization
			# -----------------------------------------
			# recompute state_layer_output after output fixes
			input_layer_output = self.input_layer(input_bits)
			state_input = cat([input_layer_output, self.state_bits], dim=1)
			state_layer_output2 = self.state_layer(state_input)

			# if state changed, propagate training
			state_conflicts = (state_layer_output2 != state_layer_output)
			if state_conflicts.any():
				fix_conflicting(self.state_layer, state_input, state_conflicts, state_layer_output2, False)

			# update state for next iteration
			self.state_bits = state_layer_output2.detach().clone()
	
