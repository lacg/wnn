from wnn.ram.RAMSequence import RAMSequence
from wnn.ram.encoders_decoders import OutputMode
from wnn.ram.cost import CostCalculatorFactory
from wnn.ram.cost import CostCalculatorType
from wnn.ram.cost.CostCalculatorRAM import CostCalculatorRAM

from torch import Tensor
from torch import tensor
from torch import zeros
from torch import uint8
from torch.nn import Module
from torch.nn import ModuleList

class RAMMultiHeadSequence(Module):
	"""
	Multi-head sequence-to-sequence model using parallel RAMSequence heads.

	Similar to multi-head attention in Transformers, but with discrete RAM neurons:
	- Multiple heads process inputs in parallel
	- Each head can specialize in different patterns
	- Outputs are combined via CostCalculator (VOTE, RAM, etc.) or KV routing

	KV Routing Mode (when k_bits > 0):
	- Key bits extracted from input determine which head to use
	- All heads still evolve state (temporal coherence preserved)
	- Only the routed head's output is used
	- Only the routed head receives EDRA error signal during training

	Automatically calculates optimal state neurons per head based on
	vocabulary partitioning: vocab_size / num_heads.
	"""

	def __init__(
		self,
		num_heads: int,
		input_bits: int,
		vocab_size: int = 26,  # Vocabulary size for automatic calculation
		n_state_neurons_per_head: int | None = None,  # Auto-calculated if None
		n_output_neurons: int = 5,
		n_bits_per_state_neuron: int | None = None,  # Auto-calculated if None
		n_bits_per_output_neuron: int = 5,
		output_mode: OutputMode = OutputMode.RAW,
		use_hashing: bool = False,
		hash_size: int = 1024,
		cost_calculator_type: CostCalculatorType = CostCalculatorType.VOTE,
		capacity_margin: int = 1,  # Extra state neurons for memory headroom
		k_bits: int = 0,  # Key bits for hard routing (0 = use cost calculator)
		key_position: str = "first",  # "first" or "last" - where to extract key
		selective_training: bool = False,  # Only train routed head at each timestep
		use_learned_router: bool = False,  # Use RAM-based learned routing
		rng: int | None = None,
	):
		"""
		Args:
			num_heads: Number of parallel heads
			input_bits: Input size (e.g., 5 for A-Z tokens)
			vocab_size: Vocabulary size (default 26 for A-Z)
			n_state_neurons_per_head: State neurons per head (auto if None)
			n_output_neurons: Output neurons
			n_bits_per_state_neuron: Connections per state neuron (auto if None)
			n_bits_per_output_neuron: Connections per output neuron
			output_mode: Output decoder mode
			use_hashing: Whether to use hash-based addressing
			hash_size: Hash table size if using hashing
			cost_calculator_type: How to combine head outputs (VOTE, RAM, etc.)
			capacity_margin: Extra state neurons beyond minimum (default 1, doubles capacity)
			k_bits: Number of input bits to use as key for hard routing (0 = disabled)
			key_position: Where to extract key bits from input ("first" or "last")
			selective_training: If True, only train the routed head at each timestep.
			                    Requires k_bits > 0 or use_learned_router. Enables head specialization.
			use_learned_router: If True, use RAM to learn which head to route to.
			                    The router learns from which heads give correct answers.
			rng: Random seed
		"""
		super().__init__()

		self.num_heads = num_heads
		self.input_bits = input_bits
		self.vocab_size = vocab_size
		self.cost_calculator_type = cost_calculator_type
		self.k_bits = k_bits
		self.key_position = key_position
		self.use_kv_routing = k_bits > 0
		self.use_learned_router = use_learned_router
		self.selective_training = selective_training

		# Validate k_bits can address all heads
		if self.use_kv_routing:
			max_addressable = 2 ** k_bits
			if max_addressable < num_heads:
				print(f"[MultiHead] Warning: k_bits={k_bits} can only address {max_addressable} heads, but num_heads={num_heads}")
			if k_bits > input_bits:
				raise ValueError(f"k_bits ({k_bits}) cannot exceed input_bits ({input_bits})")

		# Validate selective_training requires routing
		if selective_training and not (self.use_kv_routing or self.use_learned_router):
			raise ValueError("selective_training=True requires k_bits > 0 or use_learned_router=True")

		# Create learned router if requested
		self.router: CostCalculatorRAM | None = None
		if use_learned_router:
			self.router = CostCalculatorRAM(
				input_bits=input_bits,
				num_options=num_heads,
				n_bits_per_neuron=min(input_bits, 8),
				use_hashing=use_hashing,
				hash_size=hash_size,
				rng=rng,
			)

		# Auto-calculate optimal state neurons per head
		# Each head handles vocab_size / num_heads characters
		# Add capacity_margin for memory headroom (each extra bit doubles capacity)
		# Floor at vocab_size bits to ensure sufficient capacity for any head
		if n_state_neurons_per_head is None:
			chars_per_head = (vocab_size + num_heads - 1) // num_heads  # ceiling division
			min_bits = (chars_per_head - 1).bit_length()  # minimum bits for partition
			vocab_bits = (vocab_size - 1).bit_length()  # minimum bits for full vocab
			calculated = min_bits + capacity_margin
			n_state_neurons_per_head = max(calculated, vocab_bits)
			print(f"[MultiHead] Auto: {chars_per_head} chars/head → {n_state_neurons_per_head} state neurons/head (calc={calculated}, vocab_floor={vocab_bits})")

		self.n_state_neurons_per_head = n_state_neurons_per_head

		# Auto-calculate bits per state neuron (full connectivity)
		if n_bits_per_state_neuron is None:
			n_bits_per_state_neuron = input_bits + n_state_neurons_per_head

		# Cap output neuron connections at available input bits (state neurons)
		# Output layer sees only state bits, so can't connect to more than that
		if n_bits_per_output_neuron > n_state_neurons_per_head:
			n_bits_per_output_neuron = n_state_neurons_per_head

		# Create cost calculator for combining head outputs
		match cost_calculator_type:
			case CostCalculatorType.RAM:
				# RAM calculator needs input context
				self.cost_calculator = CostCalculatorFactory.create(
					mode=cost_calculator_type,
					input_bits=input_bits,
					num_options=num_heads,
					n_bits_per_neuron=min(n_bits_per_state_neuron, 10),  # Reasonable default
					use_hashing=use_hashing,
					hash_size=hash_size,
					rng=rng,
				)
			case _:
				# VOTE, ARGMIN, etc. don't need parameters
				self.cost_calculator = CostCalculatorFactory.create(mode=cost_calculator_type)

		# Create multiple heads
		self.heads = ModuleList([
			RAMSequence(
				input_bits=input_bits,
				n_state_neurons=n_state_neurons_per_head,
				n_output_neurons=n_output_neurons,
				n_bits_per_state_neuron=n_bits_per_state_neuron,
				n_bits_per_output_neuron=n_bits_per_output_neuron,
				output_mode=output_mode,
				use_hashing=use_hashing,
				hash_size=hash_size,
				rng=rng + i if rng is not None else None,
			)
			for i in range(num_heads)
		])

		# Store decoder from first head (all heads share same decoder type)
		self.decoder = self.heads[0].decoder

	def _extract_key(self, input_bits: Tensor) -> int:
		"""
		Extract key bits from input and convert to head index.

		Args:
			input_bits: Input tensor [total_bits] or [1, total_bits]

		Returns:
			Head index (0 to num_heads-1)
		"""
		if input_bits.ndim == 2:
			input_bits = input_bits.squeeze(0)

		# Extract key bits based on position
		if self.key_position == "first":
			key_bits = input_bits[:self.k_bits]
		else:  # "last"
			key_bits = input_bits[-self.k_bits:]

		# Convert bits to integer
		key_value = 0
		for bit in key_bits:
			key_value = (key_value << 1) | int(bit)

		# Map to head index (modulo for safety)
		return key_value % self.num_heads

	def _get_routed_head(self, input_bits: Tensor) -> int:
		"""
		Get the head index to route to based on routing mode.

		Args:
			input_bits: Input tensor [total_bits] or [1, total_bits]

		Returns:
			Head index (0 to num_heads-1)
		"""
		if input_bits.ndim == 2:
			input_bits = input_bits.squeeze(0)

		if self.use_learned_router:
			# Use RAM router to select head
			return int(self.router.calculate_index(input_bits))
		elif self.use_kv_routing:
			# Use key-based routing
			return self._extract_key(input_bits)
		else:
			# No routing - shouldn't be called
			return 0

	def _train_router(self, windows: list[Tensor], targets: list[str]) -> None:
		"""
		Train the learned router based on which heads give correct answers.

		For each timestep:
		1. Get predictions from all heads (without updating state)
		2. Score heads: high score for correct, low for incorrect
		3. Train router to produce these scores

		Args:
			windows: List of input tensors [1, input_bits]
			targets: Target characters per timestep
		"""
		if self.router is None:
			return

		# Reset all head states for evaluation
		for head in self.heads:
			head._reset_state(batch_size=1, device=windows[0].device)

		for t, (window, target) in enumerate(zip(windows, targets)):
			window_bits = window.squeeze(0) if window.ndim == 2 else window

			# Get predictions from all heads at this timestep
			head_correct = []
			for head in self.heads:
				# Forward through head (updates its internal state)
				_, _, _, output = head._get_outputs(window, update_state=True)
				predicted = head.decoder.decode(output)
				head_correct.append(predicted == target)

			# Create target scores: 7 for correct, 0 for incorrect
			target_scores = zeros(self.num_heads, dtype=uint8)
			for head_idx, correct in enumerate(head_correct):
				target_scores[head_idx] = 7 if correct else 0

			# Train router to produce these scores
			self.router.train_scores(window_bits, target_scores)

	def _combine_outputs(self, head_outputs: list[str], input_bits: Tensor | None = None) -> str:
		"""
		Combine head outputs using CostCalculator.

		Args:
			head_outputs: List of character predictions from each head
			input_bits: Optional input context for RAM calculator

		Returns:
			Combined prediction
		"""
		if not head_outputs:
			return '?'

		# Build vocabulary of unique outputs
		unique_chars = list(set(head_outputs))
		if len(unique_chars) == 1:
			return unique_chars[0]

		# Count votes for each character
		votes = tensor([head_outputs.count(char) for char in unique_chars])

		# For RAM calculator, use input_bits as context
		if self.cost_calculator_type == CostCalculatorType.RAM and input_bits is not None:
			# RAM calculator uses input as context, not votes
			winner_idx = self.cost_calculator.calculate_index(input_bits.squeeze())
			# Map to actual head output
			return head_outputs[winner_idx % len(head_outputs)]
		else:
			# Use votes as cost
			winner_idx = self.cost_calculator.calculate_index(votes)
			return unique_chars[winner_idx]

	def train(self, windows: list[Tensor], targets: str | list[str]) -> None:
		"""
		Train heads on sequence data.

		Training Modes:
		1. Ensemble (selective_training=False, default):
		   - All heads trained on all timesteps
		   - Maintains consistent state evolution
		   - Routing only affects forward() output selection

		2. Selective (selective_training=True):
		   - Each head only trains on timesteps where it's routed
		   - All heads still see all windows (state evolution preserved)
		   - EDRA only triggers for routed timesteps
		   - Enables true head specialization

		3. Learned Routing (use_learned_router=True):
		   - Router is trained to select heads that give correct answers
		   - Can be combined with selective_training

		Args:
			windows: List of input tensors, one per timestep [1, input_bits]
			targets: Either a string (one char per timestep) or list of strings
		"""
		n_steps = len(windows)
		if n_steps == 0:
			return

		# Convert targets to list
		if isinstance(targets, str):
			target_list = list(targets)
		else:
			target_list = list(targets)

		if not self.selective_training and not self.use_learned_router:
			# Ensemble mode: train all heads on all data
			for head in self.heads:
				head.train(windows, targets)
			return

		# For selective training or learned routing, we need per-timestep analysis
		# Compute per-head masks and track which heads are correct
		head_masks: list[list[bool]] = [
			[False] * n_steps for _ in range(self.num_heads)
		]

		for t, window in enumerate(windows):
			window_bits = window.squeeze(0) if window.ndim == 2 else window
			head_idx = self._get_routed_head(window_bits)
			head_masks[head_idx][t] = True

		# Train heads
		if self.selective_training:
			# Selective mode: train each head only on its routed timesteps
			for head_idx, head in enumerate(self.heads):
				mask = head_masks[head_idx]
				if any(mask):
					head.train_masked(windows, targets, mask)
		else:
			# Ensemble mode with learned routing
			for head in self.heads:
				head.train(windows, targets)

		# Train the learned router if enabled
		if self.use_learned_router:
			self._train_router(windows, target_list)

	def forward(self, input_bits: Tensor) -> str:
		"""
		Forward pass through all heads and combine outputs.

		Routing modes:
		- KV routing (k_bits > 0): Key bits determine which head to use
		- Learned routing (use_learned_router): RAM router selects head
		- No routing: All heads vote via CostCalculator

		In all routing modes, all heads process input to maintain state evolution.

		Args:
			input_bits: Input tensor [batch_size, total_bits] or [total_bits]

		Returns:
			Predicted character string
		"""
		# Extract last window for routing decisions
		if input_bits.ndim == 1:
			last_window = input_bits[-self.input_bits:]
		else:
			last_window = input_bits[:, -self.input_bits:]

		if self.use_kv_routing or self.use_learned_router:
			# Routing mode: All heads process, router selects output
			head_idx = self._get_routed_head(last_window)

			# All heads must forward to maintain state evolution
			for i, head in enumerate(self.heads):
				output = head.forward(input_bits)
				if i == head_idx:
					selected_output = output

			return selected_output
		else:
			# Standard: Get all outputs and combine via voting
			head_outputs = [head.forward(input_bits) for head in self.heads]
			return self._combine_outputs(head_outputs, last_window)

	def __repr__(self):
		train_mode = "selective" if self.selective_training else "ensemble"
		if self.use_learned_router:
			routing_info = f"learned_router, train={train_mode}"
		elif self.use_kv_routing:
			routing_info = f"k_bits={self.k_bits}, train={train_mode}"
		else:
			routing_info = f"cost={self.cost_calculator_type.name}"
		return (
			f"RAMMultiHeadSequence("
			f"num_heads={self.num_heads}, "
			f"{routing_info}, "
			f"state_neurons/head={self.n_state_neurons_per_head})"
		)
