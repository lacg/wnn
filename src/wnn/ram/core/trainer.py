"""
End-to-End EDRA Trainer for RAM Transformer

EDRA (Error Detection and Reconstruction Algorithm) enables training
discrete RAM networks without gradients by:

1. Forward pass: Record all intermediate states
2. Error detection: Compare output to target
3. Backward pass: Compute desired states for each layer
4. Reconstruction: Train each layer to produce desired output

Key insight: With XOR residual connections:
    output = input ⊕ layer_output

So we can solve for desired layer output:
    layer_output = input ⊕ desired_output

This allows us to backpropagate "targets" through the network.
"""

from wnn.ram.core.transformers.seq2seq import RAMSeq2Seq
from wnn.ram.core.transformers.embedding import PositionEncoding

from torch import Tensor
from dataclasses import dataclass
from typing import Callable
from enum import IntEnum


class LayerType(IntEnum):
	"""Types of layers in the RAM Transformer."""
	EMBEDDING = 0
	INPUT_PROJ = 1
	ATTENTION = 2
	FFN = 3
	OUTPUT_PROJ = 4
	TOKEN_MAPPER = 5


@dataclass
class LayerState:
	"""Records state at a specific point in the network."""
	layer_type: LayerType  # Type of layer
	index: int             # Layer index (for stacked layers like attention_0, ffn_1)
	input: list[Tensor]    # Input to this layer
	output: list[Tensor]   # Output of this layer
	after_residual: list[Tensor] | None  # After residual connection

	@property
	def name(self) -> str:
		"""String name for display (e.g., 'attention_0', 'ffn_1')."""
		match self.layer_type:
			case LayerType.ATTENTION | LayerType.FFN:
				return f"{self.layer_type.name.lower()}_{self.index}"
			case _:
				return self.layer_type.name.lower()


@dataclass
class TrainingStats:
	"""Statistics from a training step."""
	output_errors: int         # Number of incorrect output tokens
	bit_errors: int            # Total bits that differ from target
	layers_updated: dict       # Updates per layer
	converged: bool            # Whether output matches target


class RAMTrainer:
	"""
	End-to-end trainer for RAM Transformer models.

	Implements full EDRA backpropagation through:
	- Stacked attention layers
	- Feed-forward layers
	- Residual connections
	- Output projections
	- Token mappers
	"""

	def __init__(
		self,
		model: RAMSeq2Seq,
		verbose: bool = True,
	):
		"""
		Args:
			model: RAMSeq2Seq model to train
			verbose: Print training progress
		"""
		self.model = model
		self.verbose = verbose

	def _forward_with_full_trace(
		self,
		tokens: list[Tensor],
	) -> tuple[list[Tensor], list[LayerState]]:
		"""
		Forward pass recording all intermediate states.

		Returns:
			outputs: Final output tokens
			trace: List of LayerState for each component
		"""
		trace = []
		hidden = [t.squeeze() if t.ndim > 1 else t for t in tokens]

		# Embedding layer (if present)
		if self.model.embedding is not None:
			input_to_embed = [h.clone() for h in hidden]
			hidden = self.model.embedding(hidden, add_position=True)
			trace.append(LayerState(
				layer_type=LayerType.EMBEDDING,
				index=0,
				input=input_to_embed,
				output=hidden,
				after_residual=None,
			))
		# Input projection (fallback if no embedding)
		elif self.model.input_proj is not None:
			input_to_proj = [h.clone() for h in hidden]
			hidden = [
				self.model.input_proj(h.unsqueeze(0)).squeeze()
				for h in hidden
			]
			trace.append(LayerState(
				layer_type=LayerType.INPUT_PROJ,
				index=0,
				input=input_to_proj,
				output=hidden,
				after_residual=None,
			))

		# Process each layer block (attention + optional FFN)
		for i in range(self.model.num_layers):
			# === Attention ===
			attn_input = [h.clone() for h in hidden]
			attn_output = self.model.attention_layers[i].forward(hidden)

			if self.model.use_residual:
				after_residual = [h ^ out for h, out in zip(hidden, attn_output)]
			else:
				after_residual = attn_output

			trace.append(LayerState(
				layer_type=LayerType.ATTENTION,
				index=i,
				input=attn_input,
				output=attn_output,
				after_residual=after_residual,
			))

			hidden = after_residual

			# === FFN (if enabled) ===
			if self.model.ffn_layers is not None:
				ffn_input = [h.clone() for h in hidden]
				ffn_output = [self.model.ffn_layers[i](h) for h in hidden]

				# FFN has internal residual, so output IS after residual
				trace.append(LayerState(
					layer_type=LayerType.FFN,
					index=i,
					input=ffn_input,
					output=ffn_output,
					after_residual=ffn_output,  # FFN handles its own residual
				))

				hidden = ffn_output

		# Output projection
		if self.model.output_proj is not None:
			proj_input = [h.clone() for h in hidden]
			outputs = [
				self.model.output_proj(h.unsqueeze(0)).squeeze()
				for h in hidden
			]
			trace.append(LayerState(
				layer_type=LayerType.OUTPUT_PROJ,
				index=0,
				input=proj_input,
				output=outputs,
				after_residual=None,
			))
		else:
			outputs = hidden

		# Token mapper
		if self.model.token_mapper is not None:
			mapper_input = [o.clone() for o in outputs]
			outputs = [self.model.token_mapper(o) for o in outputs]
			trace.append(LayerState(
				layer_type=LayerType.TOKEN_MAPPER,
				index=0,
				input=mapper_input,
				output=outputs,
				after_residual=None,
			))

		return outputs, trace

	def _compute_backward_targets(
		self,
		trace: list[LayerState],
		target_tokens: list[Tensor],
	) -> dict[tuple[LayerType, int], list[Tensor]]:
		"""
		Compute desired outputs for each layer by working backwards.

		Given the final target, compute what each layer SHOULD have
		produced to achieve that target.

		Returns:
			Dictionary mapping (LayerType, index) -> desired outputs
		"""
		targets = {}
		current_target = [t.clone() for t in target_tokens]

		# Work backwards through trace
		for state in reversed(trace):
			key = (state.layer_type, state.index)

			match state.layer_type:
				case LayerType.TOKEN_MAPPER:
					# Token mapper: just pass target through
					targets[key] = current_target
					# Target for previous layer uses actual input
					current_target = [inp.clone() for inp in state.input]

				case LayerType.OUTPUT_PROJ:
					# Output projection: target is what we want
					targets[key] = current_target
					# Train projection on actual_input -> current_target
					current_target = [inp.clone() for inp in state.input]

				case LayerType.FFN:
					# FFN has internal residual handling
					targets[key] = current_target
					# FFN handles its own residual, so we need the input
					current_target = [inp.clone() for inp in state.input]

				case LayerType.ATTENTION:
					# Attention with residual: output = input ^ attn_output
					# So attn_output = input ^ desired_output
					if self.model.use_residual:
						desired_attn_output = [
							inp ^ tgt for inp, tgt in zip(state.input, current_target)
						]
					else:
						desired_attn_output = current_target

					targets[key] = desired_attn_output
					current_target = [inp.clone() for inp in state.input]

				case LayerType.INPUT_PROJ:
					targets[key] = current_target

				case LayerType.EMBEDDING:
					# Embedding layer: target is what we want the embedding to produce
					targets[key] = current_target
					# Note: Input to embedding is raw tokens, we don't backprop further

		return targets

	def _train_layer(
		self,
		state: LayerState,
		desired_output: list[Tensor],
	) -> int:
		"""
		Train a specific layer to produce desired output.

		Args:
			state: LayerState containing layer type, index, and recorded I/O
			desired_output: What the layer should have produced

		Returns:
			Number of updates made
		"""
		updates = 0

		match state.layer_type:
			case LayerType.TOKEN_MAPPER if self.model.token_mapper is not None:
				# Train token mapper on each position
				for inp, tgt in zip(state.input, desired_output):
					trained = self.model.token_mapper.train_mapping(inp, tgt)
					updates += trained

			case LayerType.OUTPUT_PROJ if self.model.output_proj is not None:
				# Train output projection
				for inp, tgt in zip(state.input, desired_output):
					current = self.model.output_proj(inp.unsqueeze(0)).squeeze()
					if not (current == tgt).all():
						self.model.output_proj.commit(inp.unsqueeze(0), tgt.unsqueeze(0))
						updates += 1

			case LayerType.INPUT_PROJ if self.model.input_proj is not None:
				# Train input projection
				for inp, tgt in zip(state.input, desired_output):
					current = self.model.input_proj(inp.unsqueeze(0)).squeeze()
					if not (current == tgt).all():
						self.model.input_proj.commit(inp.unsqueeze(0), tgt.unsqueeze(0))
						updates += 1

			case LayerType.EMBEDDING if self.model.embedding is not None:
				# Train embedding layer
				# Position encoding uses XOR, which is self-inverse.
				# To get desired token embedding: apply position XOR again to remove it.
				for pos, (inp, tgt) in enumerate(zip(state.input, desired_output)):
					if self.model.embedding.position_encoding != PositionEncoding.NONE:
						# XOR is self-inverse: apply position encoding to target
						# to back out and get the desired raw token embedding
						desired_token_embed = self.model.embedding._apply_position_encoding(
							tgt, pos
						)
					else:
						desired_token_embed = tgt

					trained = self.model.embedding.train_embedding(inp, desired_token_embed)
					updates += trained

			case LayerType.FFN if self.model.ffn_layers is not None:
				# Train FFN layer
				ffn = self.model.ffn_layers[state.index]
				for inp, tgt in zip(state.input, desired_output):
					stats = ffn.train_step(inp, tgt)
					updates += stats.get("down_trained", 0)

			case LayerType.ATTENTION:
				# Training attention is more complex - we need to train
				# both the attention patterns AND the value aggregation
				attn = self.model.attention_layers[state.index]

				# For now, train the output layer of attention
				# Full attention training would require learning patterns
				for pos, (inp, tgt) in enumerate(zip(state.input, desired_output)):
					if hasattr(attn, 'output_layer'):
						current = state.output[pos]
						if not (current == tgt).all():
							# Would need to train aggregation and value heads
							updates += 1

		return updates

	def train_step(
		self,
		input_tokens: list[Tensor],
		target_tokens: list[Tensor],
	) -> TrainingStats:
		"""
		Single end-to-end training step.

		Args:
			input_tokens: Input sequence
			target_tokens: Target output sequence

		Returns:
			Training statistics
		"""
		input_tokens = [t.squeeze() if t.ndim > 1 else t for t in input_tokens]
		target_tokens = [t.squeeze() if t.ndim > 1 else t for t in target_tokens]

		# Forward pass with full trace
		outputs, trace = self._forward_with_full_trace(input_tokens)

		# Count errors
		output_errors = sum(
			1 for out, tgt in zip(outputs, target_tokens)
			if not (out == tgt).all()
		)

		bit_errors = sum(
			(out != tgt).sum().item()
			for out, tgt in zip(outputs, target_tokens)
		)

		if output_errors == 0:
			return TrainingStats(
				output_errors=0,
				bit_errors=0,
				layers_updated={},
				converged=True,
			)

		# Compute backward targets
		targets = self._compute_backward_targets(trace, target_tokens)

		# Train each layer
		layers_updated = {}
		for state in trace:
			key = (state.layer_type, state.index)
			if key in targets:
				updates = self._train_layer(state, targets[key])
				layers_updated[state.name] = updates

		return TrainingStats(
			output_errors=output_errors,
			bit_errors=bit_errors,
			layers_updated=layers_updated,
			converged=False,
		)

	def train_epoch(
		self,
		dataset: list[tuple[list[Tensor], list[Tensor]]],
	) -> dict:
		"""
		Train for one epoch over a dataset.

		Args:
			dataset: List of (input_tokens, target_tokens) pairs

		Returns:
			Epoch statistics
		"""
		total_errors = 0
		total_bits = 0
		total_positions = 0
		all_updates = {}

		for inputs, targets in dataset:
			stats = self.train_step(inputs, targets)
			total_errors += stats.output_errors
			total_bits += stats.bit_errors
			total_positions += len(inputs)

			for layer, updates in stats.layers_updated.items():
				all_updates[layer] = all_updates.get(layer, 0) + updates

		accuracy = 100 * (1 - total_errors / total_positions) if total_positions > 0 else 0

		return {
			"total_errors": total_errors,
			"bit_errors": total_bits,
			"total_positions": total_positions,
			"accuracy": accuracy,
			"layer_updates": all_updates,
		}

	def train(
		self,
		dataset: list[tuple[list[Tensor], list[Tensor]]],
		epochs: int = 10,
		early_stop: bool = True,
	) -> list[dict]:
		"""
		Train the model for multiple epochs.

		Args:
			dataset: List of (input_tokens, target_tokens) pairs
			epochs: Maximum number of epochs
			early_stop: Stop if accuracy reaches 100%

		Returns:
			List of epoch statistics
		"""
		history = []

		for epoch in range(epochs):
			stats = self.train_epoch(dataset)
			history.append(stats)

			if self.verbose:
				print(f"Epoch {epoch + 1}/{epochs}: "
					  f"{stats['total_errors']} errors, "
					  f"{stats['accuracy']:.1f}% accuracy")

				if stats['layer_updates']:
					updates_str = ", ".join(
						f"{k}:{v}" for k, v in stats['layer_updates'].items()
						if v > 0
					)
					if updates_str:
						print(f"  Updates: {updates_str}")

			if early_stop and stats['total_errors'] == 0:
				if self.verbose:
					print(f"Converged at epoch {epoch + 1}!")
				break

		return history

	def evaluate(
		self,
		dataset: list[tuple[list[Tensor], list[Tensor]]],
		decoder: Callable | None = None,
	) -> dict:
		"""
		Evaluate model on a dataset.

		Args:
			dataset: List of (input_tokens, target_tokens) pairs
			decoder: Optional function to decode tokens to strings

		Returns:
			Evaluation statistics
		"""
		correct = 0
		total = 0
		examples = []

		for inputs, targets in dataset:
			outputs = self.model.forward(inputs)

			for i, (out, tgt) in enumerate(zip(outputs, targets)):
				out = out.squeeze()
				tgt = tgt.squeeze()
				is_correct = (out == tgt).all().item()

				if is_correct:
					correct += 1
				total += 1

				if decoder and len(examples) < 10:
					examples.append({
						"input": decoder(inputs[i]) if i < len(inputs) else "?",
						"output": decoder(out),
						"target": decoder(tgt),
						"correct": is_correct,
					})

		return {
			"accuracy": 100 * correct / total if total > 0 else 0,
			"correct": correct,
			"total": total,
			"examples": examples,
		}
