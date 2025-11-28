from typing import Optional

from torch import bool as tbool
from torch import cat
from torch import nonzero
from torch import randint
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
		if self.n_state_neurons > 0:
			lines.append(str(self.state_layer))
		else:
			lines.append(" (no state neurons)")

		lines.append("\n--- Output Layer ---")
		lines.append(str(self.output_layer))

		lines.append("")  # newline
		return "\n".join(lines)

	def reset_state(self, batch_size: int, device) -> None:
		"""
		Initialize recurrent state to all False.
		"""
		if self.state_layer.num_neurons > 0:
			self.state_bits = zeros(
				batch_size,
				self.state_layer.num_neurons,
				dtype=tbool,
				device=device,
			)
		else:
			self.state_bits = zeros(batch_size, 0, dtype=tbool, device=device)

	def _get_outputs(
		self,
		input_bits: Tensor,
		update_state: bool = False,
	) -> tuple[Tensor, Tensor, Tensor, Tensor]:
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
			state_layer_input = zeros(batch_size, 0, dtype=tbool, device=device)
			state_layer_output = state_layer_input

		output_layer_input = cat([input_layer_output, state_layer_output], dim=1)  # [B, N_in + N_state]
		output_layer_output = self.output_layer(output_layer_input)  # [B, N_out]

		if update_state and self.state_layer.num_neurons > 0:
			self.state_bits = state_layer_output.detach().clone()

		return input_layer_output, state_layer_output, output_layer_input, output_layer_output

	def forward(self, input_bits: Tensor) -> Tensor:
		"""
		Inference-only forward: returns output bits [B, N_out].
		"""
		if input_bits.ndim == 1:
			input_bits = input_bits.unsqueeze(0)
		if input_bits.dtype != tbool:
			input_bits = input_bits.to(tbool)

		*_, output_layer_output = self._get_outputs(input_bits, update_state=True)
		return output_layer_output

	def _choose_conflicting_neuron(self, conflicts: Tensor, device) -> Optional[int]:
		"""
		Pick one conflicting output neuron index at random, or None if no conflicts.
		conflicts: [B, N_out] bool
		"""
		idxs = nonzero(conflicts[0], as_tuple=False).view(-1)
		if idxs.numel() == 0:
			return None
		return int(idxs[randint(0, idxs.numel(), (1,), device=device)].item())

	def train_one(self, input_bits: Tensor, target_bits: Tensor, max_iters: int = 16) -> None:
		"""
		EDRA Option A (EMPTY-first) for a single training sample.

		Algorithm:
		 - repeatedly run forward
		 - if outputs match target → commit mapping to all three layers and stop
		 - otherwise:
		 		* pick one conflicting output neuron
		 		* choose one contributing hidden bit
		 		* try to update the corresponding input/state memory cell
		 		* prefer EMPTY cells; only overwrite if no EMPTY exists
		"""
		# ensure batch dimension
		if input_bits.ndim == 1:
			input_bits = input_bits.unsqueeze(0)
		if target_bits.ndim == 1:
			target_bits = target_bits.unsqueeze(0)

		if input_bits.dtype != tbool:
			input_bits = input_bits.to(tbool)
		if target_bits.dtype != tbool:
			target_bits = target_bits.to(tbool)

		batch_size = input_bits.shape[0]
		device = input_bits.device

		if batch_size != 1:
			raise ValueError("train_one currently supports batch_size=1 only for EDRA debugging.")

		# lazily initialize state
		if self.state_bits is None or self.state_bits.shape[0] != batch_size:
			self.reset_state(batch_size, device=device)

		for _ in range(max_iters):
			# 1) full forward without updating state yet
			(
				input_layer_output,
				state_layer_output,
				output_layer_input,
				output_layer_output,
			) = self._get_outputs(input_bits, update_state=False)

			# 2) output conflicts
			output_conflicts = (output_layer_output != target_bits)  # [1, N_out]

			# 2a) no conflicts → commit + update state and stop
			if not output_conflicts.any():
				# commit mappings
				self.input_layer.train_write(input_bits, input_layer_output)

				if self.state_layer.num_neurons > 0:
					state_layer_input = cat([input_layer_output, self.state_bits], dim=1)
					self.state_layer.train_write(state_layer_input, state_layer_output)
					self.state_bits = state_layer_output.detach().clone()

				self.output_layer.train_write(output_layer_input, target_bits)
				return

			# 3) conflicts present → choose one output neuron to fix
			neuron_j = self._choose_conflicting_neuron(output_conflicts, device)
			if neuron_j is None:
				return  # should not happen, but safe

			desired_out = bool(target_bits[0, neuron_j].item())

			# 4) choose a contributing bit index from output_layer connections
			conn_row = self.output_layer.memory.connections[neuron_j]  # [k_out]
			used_any = False

			for pos in range(conn_row.numel()):
				src_bit_index = int(conn_row[pos].item())

				# map src_bit_index into previous layer neuron index
				if src_bit_index < self.input_layer.num_neurons:
					prev_layer = self.input_layer
					prev_neuron = src_bit_index
					prev_input_bits = input_bits  # input_layer addresses from raw input bits
				else:
					if self.state_layer.num_neurons == 0:
						continue
					prev_layer = self.state_layer
					prev_neuron = src_bit_index - self.input_layer.num_neurons
					prev_input_bits = cat([input_layer_output, self.state_bits], dim=1)

				# 4a) compute address for this previous neuron
				prev_addresses = prev_layer.get_address(prev_input_bits)  # [1, N_prev]
				addr = int(prev_addresses[0, prev_neuron].item())

				# 4b) read current memory value
				curr_val = int(prev_layer.memory.memory[prev_neuron, addr].item())

				# 4c) prefer EMPTY cells: write desired_out only into EMPTY
				if curr_val == MemoryVal.EMPTY.value:
					prev_layer.set_memory(prev_neuron, addr, desired_out)
					used_any = True
					break

			# 5) fallback: if no EMPTY cell was found among contributors, overwrite the first one
			if not used_any:
				src_bit_index = int(conn_row[0].item())
				if src_bit_index < self.input_layer.num_neurons:
					prev_layer = self.input_layer
					prev_neuron = src_bit_index
					prev_input_bits = input_bits
				else:
					if self.state_layer.num_neurons == 0:
						continue
					prev_layer = self.state_layer
					prev_neuron = src_bit_index - self.input_layer.num_neurons
					prev_input_bits = cat([input_layer_output, self.state_bits], dim=1)

				prev_addresses = prev_layer.get_address(prev_input_bits)
				addr = int(prev_addresses[0, prev_neuron].item())
				prev_layer.set_memory(prev_neuron, addr, desired_out)

		# if we exit the loop without stabilizing, we simply do not commit this sample
		# this makes EDRA training conservative rather than destructive