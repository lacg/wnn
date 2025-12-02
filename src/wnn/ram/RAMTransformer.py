from typing import Optional

from torch import arange
from torch import bool as tbool
from torch import cat
from torch import empty
from torch import int64
from torch import long
from torch import nonzero
from torch import randint
from torch import uint8
from torch import zeros
from torch import tensor
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
		max_iters: int = 4,
	):
		super().__init__()

		self.input_bits = int(input_bits)
		self.max_iters = max_iters

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

	def _can_hidden_output(self, addresses: Tensor, input_bits: Tensor, state_layer_input: Tensor) -> Tensor:
		"""
		Fully vectorized feasibility test WITHOUT recomputing hidden addresses.

		Parameters:
			addresses:      [M] candidate output-memory addresses
			hidden_indices: [k] for *this* output neuron (connections row)
			input_addrs:    [N_in] precomputed addresses for input layer
			state_addrs:    [N_state] precomputed addresses for state layer, or None

		Returns:
			ok_mask: [M] bool — True if all required hidden neurons can realize
			the hidden codeword for that output address.
		"""
		M = addresses.shape[0]

		# ----------------------------------------------------
		# 1. Decode desired output bits from addresses
		#    desired_bits[i][p] = bit p of address i
		# ----------------------------------------------------
		shifts = arange(self.output_layer.memory.n_bits_per_neuron, device=input_bits.device).unsqueeze(0)  # [1, self.output_layer.memory.n_bits_per_neuron]
		desired_bits = ((addresses.unsqueeze(1) >> shifts) & 1).to(int64)  # [M, self.output_layer.memory.n_bits_per_neuron]

		# ----------------------------------------------------
		# 2. Retrieve the K contributing neurons (input or state)
		# ----------------------------------------------------
		hidden_indices = self.output_layer.memory.connections[0].to(input_bits.device, dtype=int64)          # [self.output_layer.memory.n_bits_per_neuron]

		# ---------------------------------------------------------
		# 3) Compute memory addresses for *all* hidden neurons
		#    input-layer first, then state-layer if present
		# ---------------------------------------------------------
		addr_input = self.input_layer.get_addresses(input_bits)[0]		# [self.input_layer.num_neurons]
		addr_state = self.state_layer.get_addresses(state_layer_input)[0]	if self.state_layer.num_neurons > 0 else None # [self.state_layer.num_neurons]

		# ---------------------------------------------------------
		# 4) Get memory values for all hidden neurons (vectorized)
		# ---------------------------------------------------------
		hidden_addrs = empty((self.output_layer.memory.n_bits_per_neuron,), dtype=int64, device=input_bits.device)
		# vectorized computation of hidden addresses
		mask_input = hidden_indices < self.input_layer.num_neurons
		if mask_input.any():
			hidden_addrs[mask_input] = addr_input[hidden_indices[mask_input] ]

		mask_state = ~mask_input
		if mask_state.any() and self.state_layer.num_neurons > 0:
			hidden_addrs[mask_state] = addr_state[hidden_indices[mask_state] - self.input_layer.num_neurons]

		# ---------------------------------------------------------
		# 5. Read hidden memory values (input + state layers)
		# ---------------------------------------------------------
		hidden_vals = empty((self.output_layer.memory.n_bits_per_neuron,), dtype=int64, device=input_bits.device)

		# INPUT memory reads
		if mask_input.any():
			hi = hidden_indices[mask_input]
			ha = hidden_addrs[mask_input]
			hidden_vals[mask_input] = self.input_layer.memory._get_memory_batch_raw(hi, ha)

		# STATE memory reads
		if mask_state.any() and self.state_layer.num_neurons > 0:
			hs = hidden_indices[mask_state] - self.input_layer.num_neurons
			ha = hidden_addrs[mask_state]
			hidden_vals[mask_state] = self.state_layer.memory._get_memory_batch_raw(hs, ha)

		# If NO state neurons exist, mask_state cols remain un-initialized.
		# But in that case mask_state is all False.

		# ---------------------------------------------------------
		# 6. Feasibility check: mem ∈ {EMPTY, desired_bit}
		# ---------------------------------------------------------
		hidden_matrix = hidden_vals.unsqueeze(0).expand(M, self.output_layer.memory.n_bits_per_neuron)  # [M, self.output_layer.memory.n_bits_per_neuron]
		ok = (hidden_matrix == MemoryVal.EMPTY) | (hidden_matrix == desired_bits)
		return ok.all(dim=1)

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

		# -------------------------------------------------
		# Precompute hidden-layer addresses ONCE per sample
		# -------------------------------------------------
		input_addrs = self.input_layer.get_addresses(input_bits)[0]  # [N_in]
		if self.state_layer.num_neurons > 0:
            # [N_state]
			state_addrs = self.state_layer.get_addresses(state_layer_input)[0]
		else:
			state_addrs = None
		self._update_neurons(stable_mask, input_bits, target_bits, state_layer_input, output_layer_input, input_addrs, state_addrs)

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
		"""
		Find a good output address for this neuron, using vectorized label filtering.
		We only loop in Python over addresses that are EMPTY or already match desired_output.
		"""
		# 1) Read entire memory row for this output neuron in one go
		row = self.output_layer.get_memory_row(neuron_index)	# [memory_size], int64

		# 2) Build a mask of addresses that are "label-compatible"
		label_mask = (row == MemoryVal.EMPTY) | (row == desired_output)
		# If no label-compatible cells → fallback
		if not label_mask.any():
				return current_address

		# addresses that pass the label test
		candidate_indices = label_mask.nonzero(as_tuple=False).view(-1).to(input_bits.device, dtype=int64)

		# Vectorized check of hidden feasibility
		ok_mask = self._can_hidden_output(candidate_indices, input_bits, state_layer_input)
		if not ok_mask.any():
			return current_address

		valid_addresses = candidate_indices[ok_mask]
		if valid_addresses.numel() == 0:
			return current_address

		# Random choice among vectorized-valid ones
		index = randint(0, valid_addresses.numel(), (1,), device=input_bits.device).item()
		return int(valid_addresses[index].item())
	
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

	def _update_neurons(self, stable_mask: Tensor, input_bits: Tensor, target_bits: Tensor, state_layer_input: Tensor, output_layer_input: Tensor, input_addrs: Tensor, state_addrs: Optional[Tensor]) -> None:
		"""
		Vectorized EDRA update for all unstable output neurons.

		- For each output neuron j where stable_mask[0, j] is False:
			1) Find a feasible target address (label-compatible + hidden-feasible)
			2) Decode the address into k bits
			3) For each connected hidden neuron:
				* compute its address (from precomputed input_addrs/state_addrs)
				* write required bit via set_memory_batch
			4) Finally write the output memory cell for neuron j
		"""
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
			# Step 2: Decode address → self.output_layer.memory.n_bits_per_neuron desired bits (one per conn)
			# -------------------------------------------------------
			shifts = arange(self.output_layer.memory.n_bits_per_neuron, device=input_bits.device)
			addr_tensor = tensor(target_output_address, device=input_bits.device, dtype=int64)
			# desired_bits[p] = bit p of address (LSB at p=0)
			desired_bits = (((addr_tensor >> shifts) & 1) != 0)  # [self.output_layer.memory.n_bits_per_neuron] bool
			# Connections for this output neuron: which hidden neurons it sees
			hidden_indices = tensor(self.inverse_output_connections[neuron_index], device=input_bits.device, dtype=long)


			# -------------------------------------------------------
			# Step 3: Split hidden indices into input vs state
			# -------------------------------------------------------
			# required bits for each hidden neuron
			# required_bits = tensor([desired_bits[h] for h in hidden_indices], device=input_bits.device, dtype=tbool)

			# Split into input vs state
			mask_input = hidden_indices < self.input_layer.num_neurons
			mask_state = ~mask_input

			# ----- INPUT-LAYER SIDE -----
			if mask_input.any():
				inp_idx		= hidden_indices[mask_input]                     # real neuron indices
				# req_bits = required_bits[mask_input]                     # bools
				req_bits	= desired_bits[mask_input]              # [K_in] bool				
				inp_addrs	= input_addrs[inp_idx]                  # [K_in]
				self.input_layer.set_memory_batch(inp_idx, inp_addrs, req_bits, allow_override=True)

			# ----- STATE-LAYER SIDE -----
			if mask_state.any() and self.state_layer.num_neurons > 0:
				st_idx	 = hidden_indices[mask_state] - self.input_layer.num_neurons            # idx into state layer
				req_bits = desired_bits[mask_state]
				st_addrs = state_addrs[st_idx]                  # [K_in]
				self.state_layer.set_memory_batch(st_idx, st_addrs, req_bits, allow_override=True)

		
			# -------------------------------------------------------
			# Step 4: Only AFTER hidden is corrected, write output mem
			# -------------------------------------------------------
			self.output_layer.set_memory(neuron_index, target_output_address, desired_output_bit, allow_override=True)

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
		target_bits = self._normalize_bits(target_bits)

		if input_bits.shape[0] != 1:
			raise ValueError("train_one supports only batch_size=1")

		# Init state if needed
		if self.state_bits is None or self.state_bits.shape[0] != 1:
			self.reset_state(1, device=input_bits.device)

		i = 0

		while i < self.max_iters and self._continue_train_one_iteration(input_bits, target_bits):
			i += 1

