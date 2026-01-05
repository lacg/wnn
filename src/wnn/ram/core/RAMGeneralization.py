"""
Generalization Components for RAM Networks

RAM neurons are lookup tables - they only know mappings they've seen.
These components enable generalization by:

1. BitLevelMapper: Learn transformations at bit level
   - Each output bit depends on relevant input bits
   - Covers patterns, not individual examples

2. CompositionalMapper: Split into high/low bit groups
   - Reduces combinatorial explosion
   - Each component generalizes independently

3. GeneralizingProjection: Combines both strategies
   - Configurable decomposition
   - Automatic pattern coverage

Note: MapperFactory is located in wnn.ram.factories.mapper
"""

from wnn.ram.core.RAMLayer import RAMLayer
from wnn.ram.core import ContextMode, MapperStrategy, BitMapperMode as OutputMode

from torch import Tensor, zeros, uint8, cat, tensor
from torch.nn import Module, ModuleList


class BitLevelMapper(Module):
	"""
	Bit-level transformation mapper.

	Instead of learning whole-token mappings (A->B, B->C),
	learns patterns at the bit level.

	Two modes:
	1. "output" mode: Learn what each output bit should be
	2. "flip" mode: Learn whether each bit should flip (XOR with input)

	For "increment" operation, flip mode is better:
	  - Bit 0 always flips (learn: always output 1)
	  - Bit 1 flips if bit 0 is 1 (learn: output 1 when bit0=1)
	  - Bit 2 flips if bits 0,1 are both 1 (learn: output 1 when bits0,1=1)
	  - etc.

	This generalizes because the flip decision only depends on
	LOWER bits, not higher bits that might be unseen.
	"""

	def __init__(
		self,
		n_bits: int,
		context_mode: ContextMode | str = ContextMode.CUMULATIVE,
		output_mode: OutputMode | str = OutputMode.FLIP,
		local_window: int = 3,
		shift_offset: int = 1,
		rng: int | None = None,
	):
		"""
		Args:
			n_bits: Number of bits per token
			context_mode: How much context each bit sees (ContextMode enum)
				- CUMULATIVE: bit i sees bits 0..i-1 (only LOWER bits for flip)
				- FULL: each bit sees all bits
				- LOCAL: each bit sees nearby bits (sliding window)
				- BIDIRECTIONAL: bit i sees symmetric window before/after
				- CAUSAL: bit i sees bits 0..i (autoregressive)
				- SHIFTED: bit i sees bit (i+offset) mod n (for shift/rotate)
			output_mode: What to learn (OutputMode enum)
				- OUTPUT: Learn the output bit value directly
				- FLIP: Learn whether to flip (XOR) the input bit
			local_window: Window size for LOCAL/BIDIRECTIONAL modes
			shift_offset: Offset for SHIFTED mode (1 = shift-left, -1 = shift-right)
			rng: Random seed
		"""
		super().__init__()

		self.n_bits = n_bits
		# Convert string to enum if needed (backwards compatibility)
		if isinstance(context_mode, str):
			context_mode = ContextMode[context_mode.upper()]
		if isinstance(output_mode, str):
			output_mode = OutputMode[output_mode.upper()]
		self.context_mode = context_mode
		self.output_mode = output_mode
		self.local_window = local_window
		self.shift_offset = shift_offset

		# Create a mapper for each output bit
		self.bit_mappers = ModuleList()

		for bit_pos in range(n_bits):
			input_bits = self._compute_context_size(bit_pos)
			mapper = RAMLayer(
				total_input_bits=input_bits,
				num_neurons=1,  # Output: flip decision or output value
				n_bits_per_neuron=input_bits,
				rng=rng + bit_pos if rng else None,
			)
			self.bit_mappers.append(mapper)

	def _compute_context_size(self, bit_pos: int) -> int:
		"""Compute the context size for a given bit position."""
		match self.context_mode:
			case ContextMode.CUMULATIVE:
				if self.output_mode == OutputMode.FLIP:
					# For flip mode, bit i only needs to see bits 0..i-1
					# Bit 0 always flips, so it needs 0 context bits -> use 1
					return max(1, bit_pos)
				else:
					# For output mode, include current bit too
					return bit_pos + 1

			case ContextMode.FULL:
				return self.n_bits

			case ContextMode.LOCAL:
				# local_window=0 means "just the current bit" (minimum 1)
				return max(1, min(self.local_window, self.n_bits))

			case ContextMode.BIDIRECTIONAL:
				# Symmetric window around bit position
				# Includes bits before and after
				return min(self.local_window, self.n_bits)

			case ContextMode.CAUSAL:
				# See bits 0..bit_pos (includes self)
				return bit_pos + 1

			case ContextMode.SHIFTED:
				# Each output bit sees exactly one input bit (shifted position)
				# This enables perfect generalization for shift/rotate operations
				return 1

			case _:
				raise ValueError(f"Unknown context_mode: {self.context_mode}")

	def _get_context_bits(self, bits: Tensor, bit_pos: int) -> Tensor:
		"""Get the context bits for a given output bit position."""
		match self.context_mode:
			case ContextMode.CUMULATIVE:
				if self.output_mode == OutputMode.FLIP:
					# For flip mode: only look at LOWER bits (0..bit_pos-1)
					# These determine the carry, not the current bit value
					if bit_pos == 0:
						# Bit 0 always flips, return dummy context
						return zeros(1, dtype=uint8)
					# Get bits 0..bit_pos-1 in LSB-first order
					start = self.n_bits - bit_pos
					context = bits[start:].clone()
					return context.flip(0)  # LSB first
				else:
					# For output mode: include current bit too
					start = self.n_bits - bit_pos - 1
					context = bits[start:].clone()
					return context.flip(0)  # LSB first

			case ContextMode.FULL:
				return bits.clone()

			case ContextMode.LOCAL:
				# Window centered on bit_pos
				# local_window=0 means "just the current bit"
				if self.local_window == 0:
					# Return just the current bit
					idx = self.n_bits - bit_pos - 1
					return bits[idx:idx+1].clone()

				half = self.local_window // 2
				start = max(0, bit_pos - half)
				end = min(self.n_bits, bit_pos + half + 1)
				context = bits[self.n_bits - end:self.n_bits - start].clone()
				# Pad if needed
				if context.numel() < self.local_window:
					padded = zeros(self.local_window, dtype=uint8)
					padded[:context.numel()] = context
					return padded
				return context

			case ContextMode.BIDIRECTIONAL:
				# Symmetric window centered on bit_pos
				# Unlike LOCAL, this explicitly includes both before and after
				half = self.local_window // 2
				# Convert bit_pos to array index (bits are stored MSB first)
				idx = self.n_bits - bit_pos - 1
				start_idx = max(0, idx - half)
				end_idx = min(self.n_bits, idx + half + 1)
				context = bits[start_idx:end_idx].clone()
				# Pad symmetrically if at edges
				if context.numel() < self.local_window:
					padded = zeros(self.local_window, dtype=uint8)
					# Center the actual bits in the padded window
					offset = (self.local_window - context.numel()) // 2
					padded[offset:offset + context.numel()] = context
					return padded
				return context

			case ContextMode.CAUSAL:
				# See bits 0..bit_pos (autoregressive, includes self)
				# Return bits from position 0 to bit_pos in LSB-first order
				start = self.n_bits - bit_pos - 1
				context = bits[start:].clone()
				return context.flip(0)  # LSB first

			case ContextMode.SHIFTED:
				# SHIFTED mode works with ARRAY indices for position-based transforms
				# For shift-left: output_array[i] = input_array[(i+1) % n]
				# For shift-right: output_array[i] = input_array[(i-1) % n]
				#
				# We need to convert bit_pos (logical) to array index first
				array_idx = self.n_bits - bit_pos - 1  # Current output array position
				source_array_idx = (array_idx + self.shift_offset) % self.n_bits
				return bits[source_array_idx:source_array_idx + 1].clone()

			case _:
				raise ValueError(f"Unknown context_mode: {self.context_mode}")

	def forward(self, bits: Tensor) -> Tensor:
		"""
		Transform input bits to output bits.

		Args:
			bits: Input tensor of shape [n_bits] or [batch, n_bits]

		Returns:
			Output tensor of same shape
		"""
		if bits.ndim == 1:
			bits = bits.unsqueeze(0)
			squeeze_output = True
		else:
			squeeze_output = False

		batch_size = bits.shape[0]
		outputs = zeros(batch_size, self.n_bits, dtype=uint8)

		for b in range(batch_size):
			for bit_pos in range(self.n_bits):
				context = self._get_context_bits(bits[b], bit_pos)
				mapper_out = self.bit_mappers[bit_pos](context.unsqueeze(0)).squeeze()

				if self.output_mode == OutputMode.FLIP:
					# XOR the input bit with the flip decision
					input_bit = bits[b, self.n_bits - bit_pos - 1]
					outputs[b, self.n_bits - bit_pos - 1] = input_bit ^ mapper_out
				else:
					# Direct output
					outputs[b, self.n_bits - bit_pos - 1] = mapper_out

		if squeeze_output:
			return outputs.squeeze(0)
		return outputs

	def train_mapping(
		self,
		input_bits: Tensor,
		output_bits: Tensor,
	) -> int:
		"""
		Train the bit-level mapping on a single example.

		Args:
			input_bits: Input tensor [n_bits]
			output_bits: Target output tensor [n_bits]

		Returns:
			Number of bits that needed training
		"""
		input_bits = input_bits.squeeze()
		output_bits = output_bits.squeeze()

		trained = 0
		for bit_pos in range(self.n_bits):
			context = self._get_context_bits(input_bits, bit_pos)
			input_bit = input_bits[self.n_bits - bit_pos - 1]
			target_out = output_bits[self.n_bits - bit_pos - 1]

			if self.output_mode == OutputMode.FLIP:
				# Target is whether to flip: 1 if input != output, 0 if same
				target_flip = (input_bit != target_out).to(uint8)
				target_value = target_flip
			else:
				# Target is the output value directly
				target_value = target_out

			# Check current output
			current = self.bit_mappers[bit_pos](context.unsqueeze(0)).squeeze()

			if current != target_value:
				trained += 1
				target = tensor([[target_value]], dtype=uint8)
				self.bit_mappers[bit_pos].commit(context.unsqueeze(0), target)

		return trained

	def __repr__(self):
		return f"BitLevelMapper(bits={self.n_bits}, mode={self.context_mode})"


class RecurrentParityMapper(Module):
	"""
	Recurrent parity mapper for computing XOR of all bits.

	Uses a 1-bit state to compute parity incrementally:
	  state[0] = input[0]
	  state[i] = state[i-1] XOR input[i]
	  parity = state[n-1]

	Only needs to learn 4 patterns (XOR truth table):
	  (0,0) -> 0, (0,1) -> 1, (1,0) -> 1, (1,1) -> 0

	This achieves 100% generalization because all 4 patterns
	are learned regardless of training set size.

	For the parity task (output = input with last bit = parity):
	  - Bits 0..n-2: Identity (copy input to output)
	  - Bit n-1: Computed parity via recurrent XOR
	"""

	def __init__(self, n_bits: int, rng: int | None = None):
		super().__init__()
		self.n_bits = n_bits

		# XOR mapper: sees (state, input) -> new_state
		# 2 input bits, 1 output bit
		self.xor_mapper = RAMLayer(
			total_input_bits=2,
			num_neurons=1,
			n_bits_per_neuron=2,
			rng=rng,
		)

	def _compute_parity(self, bits: Tensor) -> Tensor:
		"""Compute parity using recurrent XOR."""
		bits = bits.squeeze()
		state = bits[0].clone()  # Start with first bit (MSB)

		for i in range(1, self.n_bits):
			# XOR state with next bit
			xor_input = cat([state.unsqueeze(0), bits[i:i+1]])
			state = self.xor_mapper(xor_input.unsqueeze(0)).squeeze()

		return state

	def forward(self, bits: Tensor) -> Tensor:
		"""
		Transform bits: output = input with last bit = parity.

		Args:
			bits: Input tensor [n_bits] (MSB first)

		Returns:
			Output tensor with parity in last bit
		"""
		if bits.ndim == 2:
			bits = bits.squeeze(0)

		# Copy all bits
		output = bits.clone()

		# Compute parity and set last bit
		parity = self._compute_parity(bits)
		output[-1] = parity

		return output

	def train_mapping(self, input_bits: Tensor, output_bits: Tensor) -> int:
		"""
		Train the XOR mapper on an example.

		Only trains the XOR operation - identity bits need no training.

		Returns:
			Number of XOR patterns that needed training
		"""
		input_bits = input_bits.squeeze()
		output_bits = output_bits.squeeze()

		trained = 0

		# Train XOR mapper by stepping through the recurrence
		state = input_bits[0].clone()

		for i in range(1, self.n_bits):
			next_bit = input_bits[i]
			xor_input = cat([state.unsqueeze(0), next_bit.unsqueeze(0)])

			# Expected new state = state XOR next_bit
			expected_state = (state ^ next_bit).to(uint8)

			# Check current output
			current = self.xor_mapper(xor_input.unsqueeze(0)).squeeze()

			if current != expected_state:
				trained += 1
				target = tensor([[expected_state]], dtype=uint8)
				self.xor_mapper.commit(xor_input.unsqueeze(0), target)

			# Update state for next step (use expected, not predicted)
			state = expected_state

		return trained

	def __repr__(self):
		return f"RecurrentParityMapper(bits={self.n_bits})"


class CompositionalMapper(Module):
	"""
	Compositional decomposition mapper.

	Splits input into groups (e.g., high/low bits) and processes
	each group with a smaller mapper. This reduces the pattern
	space exponentially.

	For n bits split into k groups of n/k bits each:
	  - Full mapper: 2^n patterns
	  - Compositional: k * 2^(n/k) patterns

	Example (8 bits):
	  - Full: 256 patterns
	  - 2 groups of 4: 2 * 16 = 32 patterns
	  - 4 groups of 2: 4 * 4 = 16 patterns
	"""

	def __init__(
		self,
		n_bits: int,
		n_groups: int = 2,
		cross_group_context: bool = True,
		rng: int | None = None,
	):
		"""
		Args:
			n_bits: Total bits per token
			n_groups: Number of groups to split into
			cross_group_context: If True, groups can see each other
				(needed for operations like increment with carry)
			rng: Random seed
		"""
		super().__init__()

		if n_bits % n_groups != 0:
			raise ValueError(f"n_bits ({n_bits}) must be divisible by n_groups ({n_groups})")

		self.n_bits = n_bits
		self.n_groups = n_groups
		self.bits_per_group = n_bits // n_groups
		self.cross_group_context = cross_group_context

		# Create mapper for each group
		self.group_mappers = ModuleList()

		for g in range(n_groups):
			if cross_group_context:
				# Each group sees its own bits + carry from lower groups
				# Carry is 1 bit per lower group
				input_bits = self.bits_per_group + g  # own bits + g carry bits
			else:
				input_bits = self.bits_per_group

			mapper = RAMLayer(
				total_input_bits=input_bits,
				num_neurons=self.bits_per_group,
				n_bits_per_neuron=input_bits,
				rng=rng + g * 100 if rng else None,
			)
			self.group_mappers.append(mapper)

		# Carry detectors (for cross-group context)
		if cross_group_context:
			self.carry_detectors = ModuleList()
			for g in range(n_groups - 1):  # No carry from last group
				detector = RAMLayer(
					total_input_bits=self.bits_per_group,
					num_neurons=1,
					n_bits_per_neuron=self.bits_per_group,
					rng=rng + n_groups * 100 + g if rng else None,
				)
				self.carry_detectors.append(detector)
		else:
			self.carry_detectors = None

	def _get_group_bits(self, bits: Tensor, group_idx: int) -> Tensor:
		"""Extract bits for a specific group."""
		start = group_idx * self.bits_per_group
		end = start + self.bits_per_group
		return bits[start:end].clone()

	def forward(self, bits: Tensor) -> Tensor:
		"""
		Transform input bits using compositional processing.

		Args:
			bits: Input tensor [n_bits] or [batch, n_bits]

		Returns:
			Output tensor of same shape
		"""
		if bits.ndim == 1:
			bits = bits.unsqueeze(0)
			squeeze_output = True
		else:
			squeeze_output = False

		batch_size = bits.shape[0]
		outputs = zeros(batch_size, self.n_bits, dtype=uint8)

		for b in range(batch_size):
			carries = []

			for g in range(self.n_groups):
				group_bits = self._get_group_bits(bits[b], g)

				if self.cross_group_context and g > 0:
					# Include carry bits from previous groups
					carry_bits = tensor(carries, dtype=uint8)
					mapper_input = cat([group_bits, carry_bits])
				else:
					mapper_input = group_bits

				# Process this group
				group_out = self.group_mappers[g](mapper_input.unsqueeze(0)).squeeze()

				# Store output
				start = g * self.bits_per_group
				end = start + self.bits_per_group
				outputs[b, start:end] = group_out

				# Compute carry for next group (if needed)
				if self.cross_group_context and g < self.n_groups - 1:
					carry = self.carry_detectors[g](group_bits.unsqueeze(0)).squeeze().item()
					carries.append(carry)

		if squeeze_output:
			return outputs.squeeze(0)
		return outputs

	def train_mapping(
		self,
		input_bits: Tensor,
		output_bits: Tensor,
		carry_targets: list[int] | None = None,
	) -> int:
		"""
		Train the compositional mapping.

		Args:
			input_bits: Input tensor [n_bits]
			output_bits: Target output [n_bits]
			carry_targets: Optional list of carry values between groups

		Returns:
			Number of groups that needed training
		"""
		input_bits = input_bits.squeeze()
		output_bits = output_bits.squeeze()

		trained = 0

		# Compute or use provided carries
		if carry_targets is None and self.cross_group_context:
			# Infer carries from the transformation
			# (This is task-specific; for increment, carry when group overflows)
			carry_targets = []
			for g in range(self.n_groups - 1):
				in_group = self._get_group_bits(input_bits, g)
				out_group = self._get_group_bits(output_bits, g)
				# Heuristic: carry if output < input (wrapped around)
				in_val = sum(int(b) << i for i, b in enumerate(reversed(in_group)))
				out_val = sum(int(b) << i for i, b in enumerate(reversed(out_group)))
				carry_targets.append(1 if out_val < in_val else 0)

		carries = []
		for g in range(self.n_groups):
			group_in = self._get_group_bits(input_bits, g)
			group_out = self._get_group_bits(output_bits, g)

			if self.cross_group_context and g > 0:
				carry_bits = tensor(carries, dtype=uint8)
				mapper_input = cat([group_in, carry_bits])
			else:
				mapper_input = group_in

			# Train mapper
			current = self.group_mappers[g](mapper_input.unsqueeze(0)).squeeze()
			if not (current == group_out).all():
				trained += 1
				self.group_mappers[g].commit(
					mapper_input.unsqueeze(0),
					group_out.unsqueeze(0)
				)

			# Train carry detector
			if self.cross_group_context and g < self.n_groups - 1:
				carry_target = carry_targets[g] if carry_targets else 0
				current_carry = self.carry_detectors[g](group_in.unsqueeze(0)).squeeze()
				if current_carry.item() != carry_target:
					target = tensor([[carry_target]], dtype=uint8)
					self.carry_detectors[g].commit(group_in.unsqueeze(0), target)
				carries.append(carry_target)

		return trained

	def __repr__(self):
		return (f"CompositionalMapper(bits={self.n_bits}, "
				f"groups={self.n_groups}, cross={self.cross_group_context})")


class HashMapper(Module):
	"""
	Hash-based generalization mapper.

	Reduces the pattern space by hashing input to a smaller key space.
	Trade-off: May cause collisions but generalizes to unseen inputs.

	For n-bit input hashed to h-bit key:
	  - Full mapper: 2^n patterns
	  - Hash mapper: 2^h patterns (with possible collisions)

	Works well when similar inputs should produce similar outputs.
	The hash function groups "similar" bit patterns together.
	"""

	def __init__(
		self,
		n_bits: int,
		hash_bits: int = 6,
		n_hash_functions: int = 3,
		rng: int | None = None,
	):
		"""
		Args:
			n_bits: Number of bits per token
			hash_bits: Bits for hash key (smaller = more generalization)
			n_hash_functions: Number of hash functions for voting
			rng: Random seed
		"""
		super().__init__()

		self.n_bits = n_bits
		self.hash_bits = hash_bits
		self.n_hash_functions = n_hash_functions

		# Create multiple hash-based mappers for voting
		self.hash_mappers = ModuleList()
		self.hash_masks = []

		for h in range(n_hash_functions):
			# Each hash function uses different random bit selection
			# Create a random mask of which bits to use for hashing
			import random
			if rng is not None:
				random.seed(rng + h * 1000)

			# Select hash_bits positions from n_bits
			positions = sorted(random.sample(range(n_bits), min(hash_bits, n_bits)))
			self.hash_masks.append(positions)

			mapper = RAMLayer(
				total_input_bits=len(positions),
				num_neurons=n_bits,  # Output full bits
				n_bits_per_neuron=len(positions),
				rng=rng + h * 100 if rng else None,
			)
			self.hash_mappers.append(mapper)

	def _hash_input(self, bits: Tensor, mask_idx: int) -> Tensor:
		"""Hash input bits using the specified mask."""
		positions = self.hash_masks[mask_idx]
		# Extract bits at the selected positions
		return tensor([bits[self.n_bits - 1 - p].item() for p in positions], dtype=uint8)

	def forward(self, bits: Tensor) -> Tensor:
		"""
		Transform input bits using hash-based voting.

		Args:
			bits: Input tensor [n_bits] or [batch, n_bits]

		Returns:
			Output tensor of same shape
		"""
		if bits.ndim == 1:
			bits = bits.unsqueeze(0)
			squeeze_output = True
		else:
			squeeze_output = False

		batch_size = bits.shape[0]
		outputs = zeros(batch_size, self.n_bits, dtype=uint8)

		for b in range(batch_size):
			# Collect votes from all hash functions
			votes = zeros(self.n_bits, dtype=float)

			for h in range(self.n_hash_functions):
				hash_key = self._hash_input(bits[b], h)
				result = self.hash_mappers[h](hash_key.unsqueeze(0)).squeeze()
				votes += result.float()

			# Majority vote
			outputs[b] = (votes > self.n_hash_functions / 2).to(uint8)

		if squeeze_output:
			return outputs.squeeze(0)
		return outputs

	def train_mapping(
		self,
		input_bits: Tensor,
		output_bits: Tensor,
	) -> int:
		"""
		Train the hash mapping on a single example.

		Args:
			input_bits: Input tensor [n_bits]
			output_bits: Target output tensor [n_bits]

		Returns:
			Number of hash functions that needed training
		"""
		input_bits = input_bits.squeeze()
		output_bits = output_bits.squeeze()

		trained = 0
		for h in range(self.n_hash_functions):
			hash_key = self._hash_input(input_bits, h)
			current = self.hash_mappers[h](hash_key.unsqueeze(0)).squeeze()

			if not (current == output_bits).all():
				trained += 1
				self.hash_mappers[h].commit(
					hash_key.unsqueeze(0),
					output_bits.unsqueeze(0)
				)

		return trained

	def __repr__(self):
		return (f"HashMapper(bits={self.n_bits}, "
				f"hash_bits={self.hash_bits}, n_hash={self.n_hash_functions})")


class ResidualMapper(Module):
	"""
	Residual generalization mapper.

	Learns corrections to identity transformation.
	Good for operations that make small changes to input.

	output = input XOR correction

	This is effective because:
	1. For identity (no change), correction = 0 (easy to learn)
	2. For small changes, only a few bits differ
	3. The correction is typically simpler than the full transformation

	Example: For successor operation (A→B), most bits stay the same.
	Learning the difference is easier than learning the full mapping.
	"""

	def __init__(
		self,
		n_bits: int,
		context_mode: ContextMode | str = ContextMode.CAUSAL,
		local_window: int = 3,
		rng: int | None = None,
	):
		"""
		Args:
			n_bits: Number of bits per token
			context_mode: How much context the correction mapper sees
			local_window: Window size for LOCAL/BIDIRECTIONAL modes
			rng: Random seed
		"""
		super().__init__()

		self.n_bits = n_bits
		if isinstance(context_mode, str):
			context_mode = ContextMode[context_mode.upper()]
		self.context_mode = context_mode

		# The correction mapper learns XOR differences
		# Using bit-level mapper with FLIP mode (learns what to change)
		self.correction = BitLevelMapper(
			n_bits=n_bits,
			context_mode=context_mode,
			output_mode=OutputMode.FLIP,  # Learn whether to flip each bit
			local_window=local_window,
			rng=rng,
		)

	def forward(self, bits: Tensor) -> Tensor:
		"""
		Apply residual transformation: output = input XOR correction.

		Args:
			bits: Input tensor [n_bits] or [batch, n_bits]

		Returns:
			Output tensor of same shape
		"""
		# The BitLevelMapper with FLIP mode already applies XOR
		return self.correction(bits)

	def train_mapping(
		self,
		input_bits: Tensor,
		output_bits: Tensor,
	) -> int:
		"""
		Train the residual correction.

		Args:
			input_bits: Input tensor [n_bits]
			output_bits: Target output tensor [n_bits]

		Returns:
			Number of bits that needed training
		"""
		# BitLevelMapper with FLIP mode learns to produce output directly
		# (it internally computes and learns the XOR)
		return self.correction.train_mapping(input_bits, output_bits)

	def __repr__(self):
		return f"ResidualMapper(bits={self.n_bits}, context={self.context_mode.name})"


class GeneralizingProjection(Module):
	"""
	A projection layer that uses generalization strategies.

	Combines:
	- BitLevelMapper for fine-grained bit transformations
	- CompositionalMapper for structured decomposition
	- Standard RAMLayer for direct lookups

	Use this instead of RAMLayer when generalization is needed.
	"""

	def __init__(
		self,
		input_bits: int,
		output_bits: int,
		strategy: MapperStrategy | str = MapperStrategy.HYBRID,
		n_groups: int = 2,
		rng: int | None = None,
	):
		"""
		Args:
			input_bits: Input dimension
			output_bits: Output dimension
			strategy: Generalization strategy (MapperStrategy enum)
				- DIRECT: Standard RAMLayer (no generalization)
				- BIT_LEVEL: Use BitLevelMapper
				- COMPOSITIONAL: Use CompositionalMapper
				- HYBRID: Combine both
			n_groups: Groups for compositional strategy
			rng: Random seed
		"""
		super().__init__()

		self.input_bits = input_bits
		self.output_bits = output_bits
		# Convert string to enum if needed (backwards compatibility)
		if isinstance(strategy, str):
			strategy = MapperStrategy[strategy.upper()]
		self.strategy = strategy

		# Validate input/output size for strategies that require it
		if strategy != MapperStrategy.DIRECT and input_bits != output_bits:
			raise ValueError(f"{strategy.name} strategy requires input_bits == output_bits")

		# Lazy import to avoid circular dependency
		from wnn.ram.factories import MapperFactory

		# Use match statement with factory for mapper creation
		match strategy:
			case MapperStrategy.DIRECT | MapperStrategy.BIT_LEVEL | MapperStrategy.COMPOSITIONAL:
				self.mapper = MapperFactory.create(
					strategy=strategy,
					n_bits=input_bits,
					n_groups=n_groups,
					rng=rng,
				)
				self.compositional = None
				self.bit_level = None

			case MapperStrategy.HYBRID:
				# Hybrid uses both compositional and bit-level
				self.compositional = MapperFactory.create(
					strategy=MapperStrategy.COMPOSITIONAL,
					n_bits=input_bits,
					n_groups=n_groups,
					rng=rng,
				)
				self.bit_level = MapperFactory.create(
					strategy=MapperStrategy.BIT_LEVEL,
					n_bits=input_bits,
					context_mode=ContextMode.LOCAL,
					output_mode=OutputMode.FLIP,
					local_window=3,
					rng=rng + 1000 if rng else None,
				)
				self.mapper = None

			case _:
				raise ValueError(f"Unknown strategy: {strategy}")

	def forward(self, bits: Tensor) -> Tensor:
		"""Apply the projection."""
		if self.strategy == MapperStrategy.HYBRID:
			# Combine compositional and bit-level
			comp_out = self.compositional(bits)
			bit_out = self.bit_level(bits)
			# XOR combine (one learns coarse, one learns fine corrections)
			return comp_out ^ bit_out
		else:
			return self.mapper(bits)

	def train_mapping(self, input_bits: Tensor, output_bits: Tensor) -> int:
		"""Train the projection on an example."""
		if self.strategy == MapperStrategy.DIRECT:
			input_bits = input_bits.squeeze()
			output_bits = output_bits.squeeze()
			current = self.mapper(input_bits.unsqueeze(0)).squeeze()
			if not (current == output_bits).all():
				self.mapper.commit(input_bits.unsqueeze(0), output_bits.unsqueeze(0))
				return 1
			return 0
		elif self.strategy == MapperStrategy.HYBRID:
			# Train compositional first, then bit-level on residual
			t1 = self.compositional.train_mapping(input_bits, output_bits)
			comp_out = self.compositional(input_bits)
			residual = output_bits.squeeze() ^ comp_out
			t2 = self.bit_level.train_mapping(input_bits, residual)
			return t1 + t2
		else:
			return self.mapper.train_mapping(input_bits, output_bits)

	def __repr__(self):
		return f"GeneralizingProjection(in={self.input_bits}, out={self.output_bits}, strategy={self.strategy.name})"
