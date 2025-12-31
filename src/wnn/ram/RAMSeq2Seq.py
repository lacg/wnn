"""
RAM-based Sequence-to-Sequence Model

A transformer-like architecture using RAM attention layers.
Key differences from traditional transformers:
  - Discrete attention (binary) instead of continuous
  - No gradient-based training - uses EDRA
  - Learned aggregation instead of weighted sums

Architecture:
                    ┌─────────────────────────────────────┐
                    │         Token Embedding              │
  Input tokens ────▶│  (Optional: RAM-based projection)   │
                    └─────────────────────────────────────┘
                                     │
                    ┌────────────────▼────────────────────┐
                    │         Attention Layer 0           │
                    │  (multi-head, causal, learned)      │
                    └─────────────────────────────────────┘
                                     │
                                     │ + residual (XOR)
                                     │
                    ┌────────────────▼────────────────────┐
                    │         Attention Layer 1           │
                    └─────────────────────────────────────┘
                                     │
                                    ...
                                     │
                    ┌────────────────▼────────────────────┐
                    │         Output Projection           │
                    │  (RAM layer: hidden -> output)      │
                    └─────────────────────────────────────┘
                                     │
  Output tokens ◀───────────────────┘
"""

from wnn.ram.RAMLayer import RAMLayer
from wnn.ram.RAMAttention import RAMAttention
from wnn.ram.encoders_decoders import PositionMode

from torch import Tensor, zeros, uint8, cat
from torch.nn import Module, ModuleList


class RAMSeq2Seq(Module):
	"""
	RAM-based sequence-to-sequence model with stacked attention.

	This is the closest we can get to a Transformer with RAM neurons:
	- Stack of attention layers (depth)
	- Multi-head attention in each layer
	- Causal masking for autoregressive generation
	- Optional residual connections (XOR-based)
	"""

	def __init__(
		self,
		input_bits: int,
		hidden_bits: int | None = None,
		output_bits: int | None = None,
		num_layers: int = 2,
		num_heads: int = 4,
		position_mode: PositionMode = PositionMode.RELATIVE,
		max_seq_len: int = 32,
		use_residual: bool = True,
		rng: int | None = None,
	):
		"""
		Args:
			input_bits: Bits per input token
			hidden_bits: Hidden dimension (defaults to input_bits)
			output_bits: Output dimension (defaults to input_bits)
			num_layers: Number of stacked attention layers
			num_heads: Attention heads per layer
			position_mode: Position encoding mode (BINARY, RELATIVE, NONE)
			max_seq_len: Maximum sequence length
			use_residual: Whether to use residual connections (XOR)
			rng: Random seed
		"""
		super().__init__()

		self.input_bits = input_bits
		self.hidden_bits = hidden_bits or input_bits
		self.output_bits = output_bits or input_bits
		self.num_layers = num_layers
		self.num_heads = num_heads
		self.max_seq_len = max_seq_len
		self.use_residual = use_residual
		self.position_mode = position_mode

		# Input projection (if hidden != input)
		if self.hidden_bits != self.input_bits:
			self.input_proj = RAMLayer(
				total_input_bits=input_bits,
				num_neurons=self.hidden_bits,
				n_bits_per_neuron=min(input_bits, 12),
				rng=rng,
			)
		else:
			self.input_proj = None

		# Stacked attention layers
		self.attention_layers = ModuleList([
			RAMAttention(
				input_bits=self.hidden_bits,
				num_heads=num_heads,
				position_mode=position_mode,
				max_seq_len=max_seq_len,
				causal=True,
				rng=rng + i * 100 if rng else None,
			)
			for i in range(num_layers)
		])

		# Output projection (if output != hidden)
		if self.output_bits != self.hidden_bits:
			self.output_proj = RAMLayer(
				total_input_bits=self.hidden_bits,
				num_neurons=self.output_bits,
				n_bits_per_neuron=min(self.hidden_bits, 12),
				rng=rng + num_layers * 100 if rng else None,
			)
		else:
			self.output_proj = None

		print(f"[RAMSeq2Seq] layers={num_layers}, heads={num_heads}, "
			  f"dims={input_bits}->{self.hidden_bits}->{self.output_bits}, "
			  f"pos={position_mode.name}, residual={use_residual}")

	def forward(self, tokens: list[Tensor]) -> list[Tensor]:
		"""
		Process a sequence through stacked attention layers.

		Args:
			tokens: List of [input_bits] tensors

		Returns:
			outputs: List of [output_bits] tensors
		"""
		seq_len = len(tokens)
		if seq_len > self.max_seq_len:
			raise ValueError(f"Sequence length {seq_len} exceeds max {self.max_seq_len}")

		# Normalize inputs
		hidden = [t.squeeze() if t.ndim > 1 else t for t in tokens]

		# Input projection
		if self.input_proj is not None:
			hidden = [
				self.input_proj(h.unsqueeze(0)).squeeze()
				for h in hidden
			]

		# Pass through stacked attention layers
		for layer in self.attention_layers:
			layer_output = layer.forward(hidden)

			if self.use_residual:
				# XOR residual connection
				hidden = [h ^ out for h, out in zip(hidden, layer_output)]
			else:
				hidden = layer_output

		# Output projection
		if self.output_proj is not None:
			outputs = [
				self.output_proj(h.unsqueeze(0)).squeeze()
				for h in hidden
			]
		else:
			outputs = hidden

		return outputs

	def forward_with_intermediates(
		self,
		tokens: list[Tensor]
	) -> tuple[list[Tensor], list[list[Tensor]]]:
		"""
		Forward pass returning intermediate representations.

		Useful for debugging and visualization.

		Returns:
			outputs: Final outputs
			intermediates: List of hidden states after each layer
		"""
		seq_len = len(tokens)
		if seq_len > self.max_seq_len:
			raise ValueError(f"Sequence length {seq_len} exceeds max {self.max_seq_len}")

		intermediates = []
		hidden = [t.squeeze() if t.ndim > 1 else t for t in tokens]

		# Input projection
		if self.input_proj is not None:
			hidden = [
				self.input_proj(h.unsqueeze(0)).squeeze()
				for h in hidden
			]
		intermediates.append(list(hidden))

		# Pass through layers
		for layer in self.attention_layers:
			layer_output = layer.forward(hidden)

			if self.use_residual:
				hidden = [h ^ out for h, out in zip(hidden, layer_output)]
			else:
				hidden = layer_output

			intermediates.append(list(hidden))

		# Output projection
		if self.output_proj is not None:
			outputs = [
				self.output_proj(h.unsqueeze(0)).squeeze()
				for h in hidden
			]
		else:
			outputs = hidden

		return outputs, intermediates

	def generate(
		self,
		prompt: list[Tensor],
		max_new_tokens: int,
		decoder=None,
	) -> list[Tensor]:
		"""
		Autoregressive generation.

		Args:
			prompt: Initial tokens
			max_new_tokens: How many tokens to generate
			decoder: Optional decoder for token visualization

		Returns:
			full_sequence: Prompt + generated tokens
		"""
		sequence = list(prompt)

		for _ in range(max_new_tokens):
			if len(sequence) >= self.max_seq_len:
				break

			# Forward pass on full sequence
			outputs = self.forward(sequence)

			# Take last position's output as next token
			next_token = outputs[-1]
			sequence.append(next_token)

			if decoder:
				decoded = decoder.decode(next_token.unsqueeze(0))
				print(f"  Generated: '{decoded}'")

		return sequence

	def train_step(
		self,
		input_tokens: list[Tensor],
		target_tokens: list[Tensor],
	) -> dict:
		"""
		Train the model on a single input/target pair.

		Implements EDRA-style backpropagation through stacked layers:
		1. Forward pass to record all intermediate states
		2. Compute error at output
		3. Backpropagate error through layers (reverse order)
		4. Train each layer's components

		Args:
			input_tokens: Input sequence
			target_tokens: Target output sequence

		Returns:
			Dictionary with training statistics
		"""
		seq_len = len(input_tokens)
		input_tokens = [t.squeeze() if t.ndim > 1 else t for t in input_tokens]
		target_tokens = [t.squeeze() if t.ndim > 1 else t for t in target_tokens]

		# Forward pass with intermediates
		outputs, intermediates = self.forward_with_intermediates(input_tokens)

		# Count output errors
		output_errors = sum(
			1 for out, tgt in zip(outputs, target_tokens)
			if not (out == tgt).all()
		)

		if output_errors == 0:
			return {"output_errors": 0, "layer_updates": [0] * self.num_layers}

		# Train output projection if it exists
		if self.output_proj is not None:
			hidden = intermediates[-1]  # Last hidden state
			for i, (h, tgt) in enumerate(zip(hidden, target_tokens)):
				out = self.output_proj(h.unsqueeze(0)).squeeze()
				if not (out == tgt).all():
					self.output_proj.commit(h.unsqueeze(0), tgt.unsqueeze(0))

		# Compute target hidden states for each layer (backprop)
		# For now, use a simplified approach: train each layer to produce
		# outputs that eventually lead to correct final output

		layer_updates = []

		# Work backwards through layers
		current_targets = target_tokens if self.output_proj is None else None

		for layer_idx in range(self.num_layers - 1, -1, -1):
			layer = self.attention_layers[layer_idx]

			# Get input to this layer
			layer_input = intermediates[layer_idx]

			# Get output of this layer (before residual)
			layer_output = layer.forward(layer_input)

			# For the last layer, we know what we want
			# For earlier layers, we need to infer desired outputs
			if layer_idx == self.num_layers - 1:
				# Last layer: target is what would produce correct output
				if self.use_residual:
					# hidden[i] = layer_input[i] ^ layer_output[i]
					# We want hidden[i] to eventually produce target
					# For simplicity, train layer to produce correct combined output
					if self.output_proj is None:
						desired_hidden = target_tokens
						# layer_output = hidden ^ layer_input (to get desired)
						desired_layer_output = [
							d ^ inp for d, inp in zip(desired_hidden, layer_input)
						]
					else:
						# More complex: need to solve for desired hidden
						# For now, just train the layer on its current context
						desired_layer_output = layer_output
				else:
					desired_layer_output = target_tokens if self.output_proj is None else intermediates[layer_idx + 1]
			else:
				# Earlier layers: train to be consistent
				# This is simplified - full EDRA would solve constraints
				desired_layer_output = layer_output

			# Train this attention layer
			updates = layer.train_step(layer_input, desired_layer_output)
			layer_updates.append(updates)

		layer_updates.reverse()  # Put back in forward order

		return {
			"output_errors": output_errors,
			"layer_updates": layer_updates,
		}

	def train_epoch(
		self,
		dataset: list[tuple[list[Tensor], list[Tensor]]],
		verbose: bool = False,
	) -> dict:
		"""
		Train on a dataset for one epoch.

		Args:
			dataset: List of (input_tokens, target_tokens) pairs
			verbose: Print progress

		Returns:
			Epoch statistics
		"""
		total_errors = 0
		total_positions = 0

		for idx, (inputs, targets) in enumerate(dataset):
			stats = self.train_step(inputs, targets)
			total_errors += stats["output_errors"]
			total_positions += len(inputs)

			if verbose and (idx + 1) % 10 == 0:
				acc = 100 * (1 - total_errors / total_positions)
				print(f"  Sample {idx + 1}/{len(dataset)}: {acc:.1f}% accuracy")

		accuracy = 100 * (1 - total_errors / total_positions) if total_positions > 0 else 0

		return {
			"total_errors": total_errors,
			"total_positions": total_positions,
			"accuracy": accuracy,
		}

	def train(
		self,
		dataset: list[tuple[list[Tensor], list[Tensor]]],
		epochs: int = 10,
		verbose: bool = True,
	) -> list[dict]:
		"""
		Train the model for multiple epochs.

		Args:
			dataset: List of (input_tokens, target_tokens) pairs
			epochs: Number of training epochs
			verbose: Print progress

		Returns:
			List of epoch statistics
		"""
		history = []

		for epoch in range(epochs):
			stats = self.train_epoch(dataset, verbose=False)
			history.append(stats)

			if verbose:
				print(f"Epoch {epoch + 1}/{epochs}: "
					  f"{stats['total_errors']} errors, "
					  f"{stats['accuracy']:.1f}% accuracy")

		return history

	def __repr__(self):
		return (
			f"RAMSeq2Seq(layers={self.num_layers}, heads={self.num_heads}, "
			f"dims={self.input_bits}->{self.hidden_bits}->{self.output_bits}, "
			f"pos={self.position_mode.name})"
		)
