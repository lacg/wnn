from typing import Optional

from torch import bool as tbool
from torch import cat
from torch import device
from torch import nonzero
from torch import rand
from torch import randint
from torch import zeros
from torch import Tensor
from torch.nn import Module

from wnn.ram.RAMLayer import RAMLayer

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
		use_high_impact: bool = True
	):
		super().__init__()

		self.input_bits = int(input_bits)
		self.use_high_impact = use_high_impact

		# --- Flipping probability ---

		self.flip_count = 0
		self.min_flip_prob = 0.01	# never stop flipping completely
		self.flip_decay = 1000.0	# controls how fast flips fade

		# --- RAM layers ---

		# Input layer: sees raw input bits only
		self.input_layer = RAMLayer(
			total_input_bits=self.input_bits,
			num_neurons=n_input_neurons,
			n_bits_per_neuron=n_bits_per_input_neuron,
			use_hashing=use_hashing,
			hash_size=hash_size,
			rng=None if rng is None else rng,
		)

		# State layer: sees [input_layer_output, previous_state]
		self.state_layer = RAMLayer(
			total_input_bits=n_input_neurons + n_state_neurons,
			num_neurons=n_state_neurons,
			n_bits_per_neuron=n_bits_per_state_neuron,
			use_hashing=use_hashing,
			hash_size=hash_size,
			rng=None if rng is None else (rng + 1),
		)

		# Output layer: sees [input_layer_output, current_state]
		self.output_layer = RAMLayer(
			total_input_bits=n_input_neurons + n_state_neurons,
			num_neurons=n_output_neurons,
			n_bits_per_neuron=n_bits_per_output_neuron,
			use_hashing=use_hashing,
			hash_size=hash_size,
			rng=None if rng is None else (rng + 2),
		)

		# recurrent state: [batch_size, n_state_neurons] (initialized lazily)
		self.state_bits: Optional[Tensor] = None


	def _flip_probability(self) -> float:
		return max(self.min_flip_prob, 1.0 / (1.0 + (self.flip_count / self.flip_decay)))


	def _flip_memory(self, layer: RAMLayer, bit_index: int, input_bits: Tensor) -> None:
		if rand(1).item() < self._flip_probability():
			self.flip_count += 1
			layer.flip_memory(bit_index, input_bits)


	def _get_outputs(self, input_bits: Tensor, batch_size: int, input_device, update_state: bool = False) -> tuple[Tensor, Tensor, Tensor, Tensor]:
		input_layer_output = self.input_layer(input_bits)

		if self.state_layer.num_neurons > 0:
			state_layer_input = cat([input_layer_output, self.state_bits], dim=1)
			state_layer_output = self.state_layer(state_layer_input)
			output_layer_input = cat([input_layer_output, state_layer_output], dim=1)
		else:
			state_layer_input = zeros(batch_size, 0, dtype=tbool, device=input_device)
			state_layer_output = state_layer_input  # empty
			output_layer_input = input_layer_output

		output_layer_output = self.output_layer(output_layer_input)

		if update_state and self.state_layer.num_neurons > 0:
			self.state_bits = state_layer_output.detach().clone()

		return (
			input_layer_output,
			state_layer_input,
			state_layer_output,
			output_layer_input,
			output_layer_output,
		)


	def reset_state(self, batch_size: int, device: Optional[device] = None) -> None:
		"""
		Reset recurrent state to all zeros for a given batch size.
		Must be called before processing a new sequence / batch.
		"""
		if device is None:
			device = next(self.parameters(), zeros(1)).device

		self.state_bits = zeros(
			batch_size,
			self.state_layer.num_neurons,
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

		*_, output = self._get_outputs(input, batch_size, device, update_state)

		return output


	def train_one(self, input_bits: Tensor, target_bits: Tensor, max_iters: int = 32) -> None:
		"""
		EDRA A1.2-R / A1.3 over the full 3-layer RAMTransformer for a single sample.

		- Only flips memory when there are output conflicts.
		- Recomputes outputs after each flip (iterative stabilization).
		- When conflict-free, commits the mapping into the OUTPUT layer memory.
		- Influence is propagated back into input/state layers via random / high-impact
		  bit selection (depending on self.use_high_impact).
		"""
		# -----------------------------------------
		# Fix each conflicting neurons
		# -----------------------------------------
		def fix_conflicting(layer: RAMLayer, conflicts: Tensor, input_device: device, change_state: bool = True) -> int:
			# compute addresses for output layer
			# address dtype = int64
			conflicting_neuron_indices = nonzero(conflicts[0], as_tuple=False)
			if conflicting_neuron_indices.shape[0] > 0:
				index = randint(0, conflicting_neuron_indices.shape[0], (1,), device=input_device).item()
				neuron = int(conflicting_neuron_indices[index].item())

				# RANDOM INFLUENCE:
				# randomly flip 1 contributing hidden bit in output_layer_input
				src_bit_index = layer.select_connection(neuron, self.use_high_impact)

				# find which layer the contributing bit belongs to
				if src_bit_index < self.input_layer.num_neurons:
					# influence input layer
					self._flip_memory(self.input_layer, src_bit_index, input_bits)
				elif change_state and self.state_layer.num_neurons > 0:
					# influence state layer
					self._flip_memory(self.state_layer, src_bit_index - self.input_layer.num_neurons, state_layer_input)
				return neuron
			return -1

		norm_bool = lambda bits: bits if bits.dtype == tbool else bits.to(tbool)
		input_bits = norm_bool(input_bits)
		target_bits = norm_bool(target_bits)

		"""EDRA A1.2-R over output + influence into input/state layers for a single batch sample."""
		batch_size = input_bits.shape[0]
		assert batch_size == 1, "train_one currently supports batch_size=1 only"
		dev = input_bits.device

		# lazy state init
		if self.state_bits is None or self.state_bits.shape[0] != batch_size:
			self.reset_state(batch_size, dev)

		# iterative stabilization
		for _ in range(max_iters):

			# -----------------------------------------
			# 1. Forward pass (no state update)
			# -----------------------------------------
			(input_layer_output, state_layer_input, state_layer_output, output_layer_input, output_layer_output) = self._get_outputs(input_bits, batch_size, dev)

			# -----------------------------------------
			# 2. Output conflicts
			# -----------------------------------------
			output_conflicts = (output_layer_output != target_bits)  # [batch_size, n_output_neurons]

			# 3) If there are NO conflicts: commit + optionally update state and stop
			if not output_conflicts.any():
				# commit mapping into OUTPUT layer memory:
				#   addresses determined by output_layer_input,
				#   payload is target_bits (labels)
				self.output_layer.train_write(output_layer_input, target_bits)
				self.input_layer.train_write(input_bits, input_layer_output)
				# update recurrent state for future time steps
				if self.state_layer.num_neurons > 0:
					self.state_layer.train_write(state_layer_input, state_layer_output)
					self.state_bits = state_layer_output.detach().clone()
				return  # stabilized for this sample

			# 4) There ARE conflicts: pick ONE conflicting output neuron
			# compute addresses for output layer
			# address dtype = int64
			address = self.output_layer.get_addresses(output_layer_input)
			neuron = fix_conflicting(self.output_layer, output_conflicts, dev)
			if neuron < 0:
				# should not happen because we checked any(), but be defensive
				continue


			# desired bit for this neuron
			desired = bool(target_bits[0, neuron].item())

			(_, state_layer_input, state_layer_output, output_layer_input, output_layer_output) = self._get_outputs(input_bits, batch_size, dev)
			address = self.output_layer.get_addresses(output_layer_input)
			# write correct value
			self.output_layer.set_memory(neuron, address, desired)

			# -----------------------------------------
			# 4. State-layer stabilization
			# -----------------------------------------
			# recompute state_layer_output after output fixes
			# input_layer_output = self.input_layer(input_bits)
			# state_input = cat([input_layer_output, self.state_bits], dim=1)
			# state_layer_output2 = self.state_layer(state_input)

			# # if state changed, propagate training
			# state_conflicts = (state_layer_output2 != state_layer_output)
			# if state_conflicts.any():
			# 	address = self.state_layer.get_addresses(state_input)
			# 	neuron = fix_conflicting(self.state_layer, state_conflicts, dev, False)

			# 	# desired bit for this neuron
			# 	desired = bool(state_layer_output2[0, neuron].item())

			# 	(input_layer_output, state_layer_input, state_layer_output, output_layer_input, output_layer_output) = self._get_outputs(state_input, batch_size, dev)
			# 	address = self.state_layer.get_addresses(output_layer_input)
			# 	# write correct value
			# 	self.state_layer.set_memory(neuron, address, desired)

			# update state for next iteration
			# self.state_bits = state_layer_output2.detach().clone()
	
