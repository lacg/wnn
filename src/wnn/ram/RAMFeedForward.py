"""
RAM-based Feed-Forward Network

In traditional transformers, the FFN is:
    Linear(d_model → d_ff) → Activation → Linear(d_ff → d_model)

For RAM networks:
    RAMLayer(input → hidden) → RAMLayer(hidden → output)

RAM lookups are inherently non-linear (discrete lookup tables),
so no explicit activation function is needed.

Architecture:
                    ┌─────────────────────────────────────┐
                    │         Expansion Layer             │
  Input bits ──────▶│  RAMLayer: input_bits → hidden_bits │
                    └─────────────────────────────────────┘
                                     │
                                     │ (non-linear RAM lookup)
                                     │
                    ┌────────────────▼────────────────────┐
                    │         Projection Layer            │
                    │  RAMLayer: hidden_bits → output_bits│
                    └─────────────────────────────────────┘
                                     │
                                     │ + residual (XOR)
                                     │
  Output bits ◀─────────────────────┘
"""

from wnn.ram.RAMLayer import RAMLayer
from wnn.ram.RAMGeneralization import (
	MapperStrategy,
	GeneralizingProjection,
)

from torch import Tensor, zeros, uint8
from torch.nn import Module
from enum import IntEnum


class FFNMode(IntEnum):
	"""Feed-forward network mode."""
	STANDARD = 0     # Two RAMLayer projections
	GENERALIZED = 1  # Uses GeneralizingProjection for better generalization
	GATED = 2        # Gated variant: output = gate * up_proj


class RAMFeedForward(Module):
	"""
	RAM-based feed-forward network.

	Provides the non-linear transformation between attention layers,
	similar to the FFN in traditional transformers.

	Key differences from traditional FFN:
	- RAM lookup is inherently non-linear (no activation needed)
	- Discrete (binary) representations
	- Can use generalization strategies for better transfer
	"""

	def __init__(
		self,
		input_bits: int,
		hidden_bits: int | None = None,
		output_bits: int | None = None,
		expansion_factor: int = 4,
		mode: FFNMode = FFNMode.STANDARD,
		use_residual: bool = True,
		generalization: MapperStrategy = MapperStrategy.DIRECT,
		rng: int | None = None,
	):
		"""
		Args:
			input_bits: Input dimension
			hidden_bits: Hidden dimension (defaults to input_bits * expansion_factor)
			output_bits: Output dimension (defaults to input_bits)
			expansion_factor: Multiplier for hidden dimension if hidden_bits not specified
			mode: FFN mode (STANDARD, GENERALIZED, GATED)
			use_residual: Whether to add residual connection (XOR)
			generalization: Strategy for output projection (if GENERALIZED mode)
			rng: Random seed
		"""
		super().__init__()

		self.input_bits = input_bits
		self.hidden_bits = hidden_bits or (input_bits * expansion_factor)
		self.output_bits = output_bits or input_bits
		self.mode = mode
		self.use_residual = use_residual
		self.generalization = generalization

		# Expansion layer: input → hidden
		self.up_proj = RAMLayer(
			total_input_bits=input_bits,
			num_neurons=self.hidden_bits,
			n_bits_per_neuron=min(input_bits, 12),
			rng=rng,
		)

		# For gated mode, we have a separate gate projection
		if mode == FFNMode.GATED:
			self.gate_proj = RAMLayer(
				total_input_bits=input_bits,
				num_neurons=self.hidden_bits,
				n_bits_per_neuron=min(input_bits, 12),
				rng=rng + 500 if rng else None,
			)
		else:
			self.gate_proj = None

		# Projection layer: hidden → output
		if mode == FFNMode.GENERALIZED and self.hidden_bits == self.output_bits:
			self.down_proj = GeneralizingProjection(
				input_bits=self.hidden_bits,
				output_bits=self.output_bits,
				strategy=generalization,
				rng=rng + 1000 if rng else None,
			)
		else:
			self.down_proj = RAMLayer(
				total_input_bits=self.hidden_bits,
				num_neurons=self.output_bits,
				n_bits_per_neuron=min(self.hidden_bits, 12),
				rng=rng + 1000 if rng else None,
			)

		print(f"[RAMFeedForward] {input_bits}→{self.hidden_bits}→{self.output_bits}, "
			  f"mode={mode.name}, residual={use_residual}")

	def forward(self, bits: Tensor) -> Tensor:
		"""
		Forward pass through the FFN.

		Args:
			bits: Input tensor [input_bits] or [batch, input_bits]

		Returns:
			Output tensor of shape [output_bits] or [batch, output_bits]
		"""
		# Handle both single and batch inputs
		if bits.ndim == 1:
			bits = bits.unsqueeze(0)
			squeeze_output = True
		else:
			squeeze_output = False

		batch_size = bits.shape[0]
		outputs = zeros(batch_size, self.output_bits, dtype=uint8)

		for b in range(batch_size):
			x = bits[b]

			# Expansion: input → hidden
			hidden = self.up_proj(x.unsqueeze(0)).squeeze()

			# Gated mode: apply gate
			if self.mode == FFNMode.GATED:
				gate = self.gate_proj(x.unsqueeze(0)).squeeze()
				hidden = hidden & gate  # AND gate

			# Projection: hidden → output
			if isinstance(self.down_proj, GeneralizingProjection):
				out = self.down_proj(hidden)
			else:
				out = self.down_proj(hidden.unsqueeze(0)).squeeze()

			# Residual connection
			if self.use_residual and self.input_bits == self.output_bits:
				out = out ^ x  # XOR residual

			outputs[b] = out

		if squeeze_output:
			return outputs.squeeze(0)
		return outputs

	def train_step(
		self,
		input_bits: Tensor,
		target_bits: Tensor,
	) -> dict:
		"""
		Train the FFN on a single input/target pair.

		Uses EDRA-style training:
		1. Forward pass to get current output
		2. If wrong, backpropagate error
		3. Train layers to produce correct intermediate states

		Args:
			input_bits: Input tensor [input_bits]
			target_bits: Target output [output_bits]

		Returns:
			Training statistics
		"""
		input_bits = input_bits.squeeze()
		target_bits = target_bits.squeeze()

		# Forward pass
		hidden = self.up_proj(input_bits.unsqueeze(0)).squeeze()

		if self.mode == FFNMode.GATED:
			gate = self.gate_proj(input_bits.unsqueeze(0)).squeeze()
			hidden = hidden & gate

		if isinstance(self.down_proj, GeneralizingProjection):
			output = self.down_proj(hidden)
		else:
			output = self.down_proj(hidden.unsqueeze(0)).squeeze()

		# Apply residual for comparison
		if self.use_residual and self.input_bits == self.output_bits:
			output_with_residual = output ^ input_bits
		else:
			output_with_residual = output

		# Check if correct
		if (output_with_residual == target_bits).all():
			return {"errors": 0, "up_trained": 0, "down_trained": 0}

		# Compute desired output (before residual)
		if self.use_residual and self.input_bits == self.output_bits:
			desired_output = target_bits ^ input_bits
		else:
			desired_output = target_bits

		# Train down projection
		down_trained = 0
		if isinstance(self.down_proj, GeneralizingProjection):
			down_trained = self.down_proj.train_mapping(hidden, desired_output)
		else:
			current = self.down_proj(hidden.unsqueeze(0)).squeeze()
			if not (current == desired_output).all():
				self.down_proj.commit(hidden.unsqueeze(0), desired_output.unsqueeze(0))
				down_trained = 1

		# For the up projection, we need to solve for what hidden state
		# would produce the correct output. This is complex for general
		# mappings, so for now we just ensure the current path is trained.
		# Full EDRA would solve constraints backwards.

		return {
			"errors": 1,
			"up_trained": 0,  # TODO: implement backward constraint solving
			"down_trained": down_trained,
		}

	def train_batch(
		self,
		inputs: list[Tensor],
		targets: list[Tensor],
		epochs: int = 10,
		verbose: bool = True,
	) -> list[dict]:
		"""
		Train on a batch of input/target pairs.

		Args:
			inputs: List of input tensors
			targets: List of target tensors
			epochs: Number of training epochs
			verbose: Print progress

		Returns:
			List of epoch statistics
		"""
		history = []

		for epoch in range(epochs):
			total_errors = 0
			total_down = 0

			for inp, tgt in zip(inputs, targets):
				stats = self.train_step(inp, tgt)
				total_errors += stats["errors"]
				total_down += stats["down_trained"]

			history.append({
				"epoch": epoch + 1,
				"errors": total_errors,
				"down_trained": total_down,
			})

			if verbose:
				acc = 100 * (1 - total_errors / len(inputs))
				print(f"  Epoch {epoch + 1}/{epochs}: {total_errors} errors ({acc:.1f}% acc)")

			if total_errors == 0:
				if verbose:
					print(f"  Converged at epoch {epoch + 1}!")
				break

		return history

	def __repr__(self):
		return (f"RAMFeedForward({self.input_bits}→{self.hidden_bits}→{self.output_bits}, "
				f"mode={self.mode.name})")
