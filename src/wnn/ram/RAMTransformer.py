from random import choice

from typing import Optional

from torch import cat
from torch import nonzero
from torch import randint
from torch import uint8
from torch import zeros
from torch import Tensor
from torch.nn import Module

from wnn.ram.RAMLayer import RAMLayer
from wnn.ram.Memory import MemoryVal


class RAMTransformer(Module):
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
	):
		super().__init__()

		self.input_bits = int(input_bits)

		# -------------------------
		# Layers
		# -------------------------
		self.input_layer = RAMLayer(
			total_input_bits=self.input_bits,
			num_neurons=n_input_neurons,
			n_bits_per_neuron=n_bits_per_input_neuron,
			use_hashing=use_hashing,
			hash_size=hash_size,
			rng=None if rng is None else rng,
		)

		self.state_layer = RAMLayer(
			total_input_bits=n_input_neurons + n_state_neurons,
			num_neurons=n_state_neurons,
			n_bits_per_neuron=n_bits_per_state_neuron,
			use_hashing=use_hashing,
			hash_size=hash_size,
			rng=None if rng is None else (rng + 1),
		)

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

		# maps hidden_neuron → list of (output_neuron, bit_position)
		self.inverse_output_connections = self._build_inverse_connections()
		self.output_desired_bits = self._build_output_desired_bits()

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

	def _build_inverse_connections(self):
		conn = self.output_layer.memory.connections  # shape [N_out, bits_per_neuron]
		inv_conn = []
		
		for out_neuron in range(conn.shape[0]):
			inv_conn.append([int(conn[out_neuron, bit_pos].item()) for bit_pos in range(conn.shape[1])])
		
		return inv_conn

	def _build_output_desired_bits(self):
		n_bits = self.output_layer.memory.n_bits_per_neuron
		return [self._get_desired_bits(n_bits, address) for address in range(2 ** (self.input_layer.num_neurons + self.state_layer.num_neurons))]


	def _can_hidden_output(self, address: int, input_bits: Tensor, state_layer_input: Tensor) -> bool:
		input_memories = self.input_layer.get_memories_for_bits(input_bits)
		state_memories = self.state_layer.get_memories_for_bits(state_layer_input) if self.state_layer.num_neurons > 0 else []
		for hidden_neuron_index, desired_hidden_output in enumerate(self._get_desired_bits(self.output_layer.memory.n_bits_per_neuron, address)):
			if hidden_neuron_index >= self.input_layer.num_neurons + self.state_layer.num_neurons:
				return False
			else:
				memory_value = int(input_memories[0, hidden_neuron_index].item()) if hidden_neuron_index < self.input_layer.num_neurons else int(state_memories[0, hidden_neuron_index - self.input_layer.num_neurons].item())
			if not ((memory_value == MemoryVal.EMPTY.value) or
				  (memory_value == MemoryVal.TRUE.value and desired_hidden_output) or
					(memory_value == MemoryVal.FALSE.value and (not desired_hidden_output))):
				return False
		return True

	def _choose_conflicting_neuron(self, conflicts: Tensor, device) -> Optional[int]:
		"""
		Pick one conflicting output neuron index at random, or None if no conflicts.
		conflicts: [B, N_out] bool
		"""
		idxs = nonzero(conflicts[0], as_tuple=False).view(-1)
		if idxs.numel() == 0:
			return None
		return int(idxs[randint(0, idxs.numel(), (1,), device=device)].item())

	def _continue_train_one_iteration(self, input_bits: Tensor, target_bits: Tensor) -> bool:
		# Forward pass (no commit)
		(state_layer_input, input_layer_output, state_layer_output, output_layer_input, output_layer_output) = self._get_outputs(input_bits, update_state=False)

		# Track which neurons are stable
		stable_mask = (output_layer_output == target_bits)  # [1, self.output_layer.num_neurons]

		if bool(stable_mask.all()):
			# ALL outputs correct → now final commit
			self._train_write(input_bits, target_bits, state_layer_input, input_layer_output, state_layer_output, output_layer_input)
			# Leave the train_one iteration
			return False

		self._update_neurons(stable_mask, input_bits, target_bits, state_layer_input, output_layer_input)

		return True

	def _get_desired_bits(self, bits_per_neuron: int, address: int) -> list[bool]:
		desired_bits = []
		for _ in range(bits_per_neuron):
			desired_bits.append(bool(address & 1))
			address >>= 1
		return desired_bits[::-1]

	def _get_outputs(self, input_bits: Tensor, update_state: bool = False) -> tuple[Tensor, Tensor, Tensor, Tensor]:
		"""
		Run full forward pass:
			returns (input_layer_output, state_layer_output, output_layer_input, output_layer_output)
		"""
		batch_size = input_bits.shape[0]
		device = input_bits.device

		if self.state_bits is None or self.state_bits.shape[0] != batch_size:
			self.reset_state(batch_size, device=device)

		input_layer_output = self.input_layer(input_bits)  # [B, N_in]

		if self.state_layer.num_neurons > 0:
			state_layer_input = cat([input_layer_output, self.state_bits], dim=1)
			state_layer_output = self.state_layer(state_layer_input)  # [B, N_state]
		else:
			state_layer_input = zeros(batch_size, 0, dtype=uint8, device=device)
			state_layer_output = state_layer_input

		output_layer_input = cat([input_layer_output, state_layer_output], dim=1)  # [B, N_in + N_state]
		output_layer_output = self.output_layer(output_layer_input)  # [B, N_out]

		if update_state and self.state_layer.num_neurons > 0:
			self.state_bits = state_layer_output.detach().clone()

		return state_layer_input, input_layer_output, state_layer_output, output_layer_input, output_layer_output

	def _get_target_addresses(self, neuron_index: int, desired_output: bool, current_address: int, input_bits: Tensor, state_layer_input: Tensor) -> int:
		good_addresses = []
		for memory_address in range(self.output_layer.memory.memory_size):
			memory_value = self.output_layer.get_memory(neuron_index, memory_address)
			if (((memory_value == MemoryVal.EMPTY.value) or
				   (memory_value == MemoryVal.TRUE.value and desired_output) or
					 (memory_value == MemoryVal.FALSE.value and (not desired_output))) and
				  (self._can_hidden_output(memory_address, input_bits, state_layer_input))):
				good_addresses.append(memory_address)

		# Fallback if no good addresses exist
		return current_address if len(good_addresses) == 0 else choice(good_addresses)

	def _normalize_bits(self, bits: Tensor) -> Tensor:
		if bits.ndim == 1:
			bits = bits.unsqueeze(0)
		if bits.dtype != uint8:
			bits = bits.to(uint8)
		return bits

	def _train_write(self, input_bits: Tensor, target_bits: Tensor, state_layer_input: Tensor, input_layer_output: Tensor, state_layer_output: Tensor, output_layer_input: Tensor) -> None:
		# ALL outputs correct → now final commit
		# commit input memory
		self.input_layer.train_write(input_bits, input_layer_output)
		# commit state memory
		if self.state_layer.num_neurons > 0:
			self.state_layer.train_write(state_layer_input, state_layer_output)
			self.state_bits = state_layer_output.detach().clone()
		# commit output memory
		self.output_layer.train_write(output_layer_input, target_bits)

	def _update_hidden_layer(self, neuron_index: int, bit: bool, input_bits: Tensor, state_layer_input: Tensor) -> None:
		# Determine which layer owns that src bit
		if neuron_index < self.input_layer.num_neurons:
			prev_layer = self.input_layer
			prev_neuron = neuron_index
			prev_input = input_bits
		else:
			if self.state_layer.num_neurons == 0:
				return
			prev_layer = self.state_layer
			prev_neuron = neuron_index - self.input_layer.num_neurons
			prev_input = cat([state_layer_input, self.state_bits], dim=1)

		# connections of this neuron
		conn = prev_layer.memory.connections[prev_neuron]  # [k]

		# desired pattern for THIS neuron to produce bit
		desired_local_bits = prev_input[:, conn]  # shape [1, k]
		target_output_address = prev_layer.get_address_for_neuron(prev_neuron, desired_local_bits)

		# write memory
		# what does this neuron currently output for this sample?
		curr_hidden_bits = prev_layer(prev_input)      # [1, N_prev]
		curr_bit = bool(curr_hidden_bits[0, prev_neuron].item())

		if curr_bit != bit:
			prev_layer.set_memory(prev_neuron, target_output_address, bit)
		# else: leave it alone to preserve previously learned patterns

	def _update_neurons(self, stable_mask: Tensor, input_bits: Tensor, target_bits: Tensor, state_layer_input: Tensor, output_layer_input: Tensor) -> None:
		# Some neurons unstable → perform EDRA for each conflicting neuron
		output_addresses = self.output_layer.get_addresses(output_layer_input)  # [1, self.output_layer.num_neurons]

		for neuron_index in range(self.output_layer.num_neurons):
			if bool(stable_mask[0, neuron_index]):
				continue  # skip neurons that already match

			desired_output_bit = bool(target_bits[0, neuron_index].item())
			current_address = int(output_addresses[0, neuron_index].item())

			# -------------------------------------------------------
			# Step 1: Find a target address where desired label fits
			# -------------------------------------------------------
			target_output_address = self._get_target_addresses(neuron_index, desired_output_bit, current_address, input_bits, state_layer_input)

			# -------------------------------------------------------
			# Step 2: Decode output address → required hidden bits
			# -------------------------------------------------------
			desired_bits = self.output_desired_bits[target_output_address]

			# -------------------------------------------------------
			# Step 3: Update HIDDEN layer BEFORE output layer
			# steering hidden memory toward the desired codeword
			# -------------------------------------------------------
			for pos in range(self.output_layer.memory.n_bits_per_neuron):

					hidden_neuron_index = self.inverse_output_connections[neuron_index][pos]
					required_bit        = desired_bits[hidden_neuron_index]

					# Force this hidden neuron to output required_bit
					# This function updates hidden memory (input layer)
					# WITHOUT modifying output address prematurely.
					self._update_hidden_layer(
							hidden_neuron_index,
							required_bit,
							input_bits,
							state_layer_input
					)

			# -------------------------------------------------------
			# Step 4: Only AFTER hidden is corrected, write output mem
			# -------------------------------------------------------
			self.output_layer.set_memory(
					neuron_index,
					target_output_address,
					desired_output_bit,
					allow_override=True
			)

	def forward(self, input_bits: Tensor) -> Tensor:
		"""
		Inference-only forward: returns output bits [B, N_out].
		"""
		*_, output_layer_output = self._get_outputs(self._normalize_bits(input_bits), update_state=True)
		return output_layer_output

	def reset_state(self, batch_size: int, device) -> None:
		"""
		Initialize recurrent state to all False.
		"""
		if self.state_layer.num_neurons > 0:
			self.state_bits = zeros(
				batch_size,
				self.state_layer.num_neurons,
				dtype=uint8,
				device=device,
			)
		else:
			self.state_bits = zeros(batch_size, 0, dtype=uint8, device=device)

	def train_one(self, input_bits: Tensor, target_bits: Tensor, max_iters: int = 16) -> None:
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
		target_bits = self._normalize_bits(target_bits)

		if input_bits.shape[0] != 1:
			raise ValueError("train_one supports only batch_size=1")

		# Init state if needed
		if self.state_bits is None or self.state_bits.shape[0] != 1:
			self.reset_state(1, device=input_bits.device)

		i = 0

		while i < max_iters and self._continue_train_one_iteration(input_bits, target_bits):
			i += 1

