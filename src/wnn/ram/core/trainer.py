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

Training Modes:
- GREEDY: Train all layers in one pass (fast)
- ITERATIVE: Multiple passes until stable (more accurate)
- OUTPUT_FIRST: Prioritize training output layers first

Features:
- Per-layer error tracking
- Curriculum learning support
- Patience-based early stopping
- Training callbacks
"""

from wnn.ram.core.transformers.seq2seq import RAMSeq2Seq
from wnn.ram.enums import PositionEncoding, LayerType, TrainingMode, TrainingPhase

from torch import Tensor, zeros, float32, save as torch_save, load as torch_load
from dataclasses import dataclass, field, asdict
from typing import Callable, Protocol, Any
from random import shuffle
from pathlib import Path
import json


class TrainingCallback(Protocol):
	"""Protocol for training callbacks."""

	def on_epoch_end(self, epoch: int, stats: dict) -> bool:
		"""Called at end of each epoch. Return False to stop training."""
		...

	def on_improvement(self, epoch: int, old_acc: float, new_acc: float) -> None:
		"""Called when accuracy improves."""
		...


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
	layer_errors: dict = field(default_factory=dict)  # Per-layer error counts


@dataclass
class EpochStats:
	"""Detailed statistics from a training epoch."""
	epoch: int
	total_errors: int
	bit_errors: int
	total_positions: int
	accuracy: float
	layer_updates: dict
	layer_error_rates: dict  # Per-layer error contribution
	examples_mastered: int   # Number of examples with 100% accuracy
	hard_examples: list      # Indices of examples that failed


@dataclass
class Checkpoint:
	"""Training checkpoint for save/resume."""
	epoch: int                   # Current epoch
	best_accuracy: float         # Best accuracy seen so far
	epochs_without_improvement: int  # Epochs without improvement
	training_history: list       # History of epoch stats (as dicts)
	mastered_indices: list       # Indices of mastered examples
	training_mode: str           # Training mode name
	phase: str | None            # Current curriculum phase
	model_path: str              # Path to saved model weights

	def to_dict(self) -> dict:
		"""Convert to JSON-serializable dict."""
		return {
			"epoch": self.epoch,
			"best_accuracy": self.best_accuracy,
			"epochs_without_improvement": self.epochs_without_improvement,
			"training_history": self.training_history,
			"mastered_indices": list(self.mastered_indices),
			"training_mode": self.training_mode,
			"phase": self.phase,
			"model_path": self.model_path,
		}

	@classmethod
	def from_dict(cls, d: dict) -> "Checkpoint":
		"""Create from dict."""
		return cls(**d)


class RAMTrainer:
	"""
	End-to-end trainer for RAM Transformer models.

	Implements full EDRA backpropagation through:
	- Stacked attention layers
	- Feed-forward layers
	- Residual connections
	- Output projections
	- Token mappers

	Features:
	- Multiple training modes (GREEDY, ITERATIVE, OUTPUT_FIRST)
	- Patience-based early stopping
	- Curriculum learning support
	- Per-layer error tracking
	- Training callbacks
	"""

	def __init__(
		self,
		model: RAMSeq2Seq,
		mode: TrainingMode = TrainingMode.GREEDY,
		patience: int = 5,
		verbose: bool = True,
	):
		"""
		Args:
			model: RAMSeq2Seq model to train
			mode: Training mode (GREEDY, ITERATIVE, OUTPUT_FIRST)
			patience: Epochs without improvement before stopping
			verbose: Print training progress
		"""
		self.model = model
		self.mode = mode
		self.patience = patience
		self.verbose = verbose

		# Training state
		self.best_accuracy = 0.0
		self.epochs_without_improvement = 0
		self.training_history: list[EpochStats] = []
		self.callbacks: list[TrainingCallback] = []

	def add_callback(self, callback: TrainingCallback) -> None:
		"""Add a training callback."""
		self.callbacks.append(callback)

	def reset_state(self) -> None:
		"""Reset training state for a new training run."""
		self.best_accuracy = 0.0
		self.epochs_without_improvement = 0
		self.training_history = []

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
				# Train attention layer using the train_step interface
				attn = self.model.attention_layers[state.index]
				updates += self._train_attention_layer(
					attn, state.input, state.output, desired_output
				)

		return updates

	def _train_attention_layer(
		self,
		attn,
		inputs: list[Tensor],
		actual_outputs: list[Tensor],
		desired_outputs: list[Tensor],
	) -> int:
		"""
		Train an attention layer to produce desired outputs.

		Attention training is complex because it involves:
		1. Similarity computation (which positions to attend to)
		2. Value aggregation (how to combine attended values)
		3. Output projection (final transformation)

		We use the attention layer's train_step method if available,
		otherwise fall back to training the output layer directly.

		Args:
			attn: The attention layer to train
			inputs: Input tokens to the attention layer
			actual_outputs: What the layer actually produced
			desired_outputs: What we want it to produce

		Returns:
			Number of updates made
		"""
		updates = 0

		# Method 1: Use attention's built-in train_step if available
		# This trains the full attention mechanism properly
		if hasattr(attn, 'train_step'):
			try:
				updates = attn.train_step(inputs, desired_outputs)
				return updates
			except Exception:
				pass  # Fall through to other methods

		# Method 2: Train output layer if available
		if hasattr(attn, 'output_layer') and attn.output_layer is not None:
			for pos, (actual, desired) in enumerate(zip(actual_outputs, desired_outputs)):
				actual = actual.squeeze()
				desired = desired.squeeze()
				if not (actual == desired).all():
					# The output layer maps aggregated values to final output
					# We need to train it to produce the desired output
					# given what the aggregation actually produced
					if hasattr(attn.output_layer, 'commit'):
						# Get the pre-output value (what went into output layer)
						# This requires access to attention internals
						attn.output_layer.commit(
							actual.unsqueeze(0),
							desired.unsqueeze(0)
						)
						updates += 1

		# Method 3: Train value heads to produce correct outputs
		# This is a simplified approach that may help in some cases
		if hasattr(attn, 'value_heads') and updates == 0:
			for pos, (inp, desired) in enumerate(zip(inputs, desired_outputs)):
				# For each position, train value heads to produce
				# values that would aggregate to the desired output
				for head_idx, value_head in enumerate(attn.value_heads):
					if hasattr(value_head, 'commit'):
						# Simplified: train head to produce desired on input
						value_head.commit(
							inp.unsqueeze(0),
							desired.unsqueeze(0)
						)
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
		epoch_num: int = 0,
		shuffle_data: bool = True,
	) -> EpochStats:
		"""
		Train for one epoch over a dataset.

		Args:
			dataset: List of (input_tokens, target_tokens) pairs
			epoch_num: Current epoch number (for statistics)
			shuffle_data: Whether to shuffle the dataset

		Returns:
			Detailed epoch statistics
		"""
		# Optionally shuffle the dataset
		if shuffle_data:
			indices = list(range(len(dataset)))
			shuffle(indices)
			dataset = [dataset[i] for i in indices]
		else:
			indices = list(range(len(dataset)))

		total_errors = 0
		total_bits = 0
		total_positions = 0
		all_updates = {}
		layer_errors = {}
		examples_mastered = 0
		hard_examples = []

		for idx, (inputs, targets) in enumerate(dataset):
			# Use iterative mode if specified
			if self.mode == TrainingMode.ITERATIVE:
				stats = self._train_step_iterative(inputs, targets)
			else:
				stats = self.train_step(inputs, targets)

			total_errors += stats.output_errors
			total_bits += stats.bit_errors
			total_positions += len(inputs)

			# Track per-example success
			if stats.output_errors == 0:
				examples_mastered += 1
			else:
				hard_examples.append(indices[idx] if shuffle_data else idx)

			# Aggregate layer updates
			for layer, updates in stats.layers_updated.items():
				all_updates[layer] = all_updates.get(layer, 0) + updates

			# Aggregate layer errors
			for layer, errors in stats.layer_errors.items():
				layer_errors[layer] = layer_errors.get(layer, 0) + errors

		accuracy = 100 * (1 - total_errors / total_positions) if total_positions > 0 else 0

		# Compute per-layer error rates
		layer_error_rates = {
			layer: errors / total_positions if total_positions > 0 else 0
			for layer, errors in layer_errors.items()
		}

		return EpochStats(
			epoch=epoch_num,
			total_errors=total_errors,
			bit_errors=total_bits,
			total_positions=total_positions,
			accuracy=accuracy,
			layer_updates=all_updates,
			layer_error_rates=layer_error_rates,
			examples_mastered=examples_mastered,
			hard_examples=hard_examples[:10],  # Keep top 10 hard examples
		)

	def _train_step_iterative(
		self,
		input_tokens: list[Tensor],
		target_tokens: list[Tensor],
		max_iterations: int = 3,
	) -> TrainingStats:
		"""
		Train with multiple iterations until stable or max iterations reached.

		Args:
			input_tokens: Input sequence
			target_tokens: Target output sequence
			max_iterations: Maximum number of training iterations

		Returns:
			Combined training statistics
		"""
		total_stats = TrainingStats(
			output_errors=0,
			bit_errors=0,
			layers_updated={},
			converged=False,
			layer_errors={},
		)

		for iteration in range(max_iterations):
			stats = self.train_step(input_tokens, target_tokens)

			# Accumulate statistics
			total_stats.output_errors = stats.output_errors
			total_stats.bit_errors = stats.bit_errors
			for layer, updates in stats.layers_updated.items():
				total_stats.layers_updated[layer] = (
					total_stats.layers_updated.get(layer, 0) + updates
				)

			# Stop if converged
			if stats.converged:
				total_stats.converged = True
				break

			# Stop if no updates made (stuck)
			if sum(stats.layers_updated.values()) == 0:
				break

		return total_stats

	def train(
		self,
		dataset: list[tuple[list[Tensor], list[Tensor]]],
		epochs: int = 10,
		early_stop: bool = True,
		use_patience: bool = True,
		shuffle_data: bool = True,
	) -> list[EpochStats]:
		"""
		Train the model for multiple epochs.

		Args:
			dataset: List of (input_tokens, target_tokens) pairs
			epochs: Maximum number of epochs
			early_stop: Stop if accuracy reaches 100%
			use_patience: Stop if no improvement for `patience` epochs
			shuffle_data: Shuffle dataset each epoch

		Returns:
			List of epoch statistics
		"""
		self.reset_state()
		history = []

		for epoch in range(epochs):
			stats = self.train_epoch(dataset, epoch_num=epoch, shuffle_data=shuffle_data)
			history.append(stats)
			self.training_history.append(stats)

			# Check for improvement
			if stats.accuracy > self.best_accuracy:
				old_acc = self.best_accuracy
				self.best_accuracy = stats.accuracy
				self.epochs_without_improvement = 0

				# Notify callbacks
				for callback in self.callbacks:
					if hasattr(callback, 'on_improvement'):
						callback.on_improvement(epoch, old_acc, stats.accuracy)
			else:
				self.epochs_without_improvement += 1

			# Verbose output
			if self.verbose:
				print(f"Epoch {epoch + 1}/{epochs}: "
					  f"{stats.total_errors} errors, "
					  f"{stats.accuracy:.1f}% accuracy, "
					  f"{stats.examples_mastered}/{len(dataset)} examples mastered")

				if stats.layer_updates:
					updates_str = ", ".join(
						f"{k}:{v}" for k, v in stats.layer_updates.items()
						if v > 0
					)
					if updates_str:
						print(f"  Updates: {updates_str}")

			# Notify callbacks
			for callback in self.callbacks:
				if hasattr(callback, 'on_epoch_end'):
					should_continue = callback.on_epoch_end(epoch, stats.__dict__)
					if should_continue is False:
						if self.verbose:
							print("Training stopped by callback")
						return history

			# Early stopping conditions
			if early_stop and stats.total_errors == 0:
				if self.verbose:
					print(f"Converged at epoch {epoch + 1}!")
				break

			if use_patience and self.epochs_without_improvement >= self.patience:
				if self.verbose:
					print(f"Early stopping: no improvement for {self.patience} epochs")
				break

		return history

	def train_curriculum(
		self,
		dataset: list[tuple[list[Tensor], list[Tensor]]],
		epochs_per_phase: int = 5,
		phases: list[TrainingPhase] | None = None,
	) -> list[EpochStats]:
		"""
		Train with curriculum learning: easy examples first, then harder.

		Args:
			dataset: List of (input_tokens, target_tokens) pairs
			epochs_per_phase: Epochs to train per phase
			phases: List of training phases (default: WARMUP, MAIN, REFINEMENT)

		Returns:
			Combined training history
		"""
		if phases is None:
			phases = [TrainingPhase.WARMUP, TrainingPhase.MAIN, TrainingPhase.REFINEMENT]

		all_history = []
		mastered_indices = set()

		for phase in phases:
			if self.verbose:
				print(f"\n=== Phase: {phase.name} ===")

			# Select examples based on phase
			if phase == TrainingPhase.WARMUP:
				# Start with shortest sequences
				sorted_dataset = sorted(dataset, key=lambda x: len(x[0]))
				phase_dataset = sorted_dataset[:len(dataset) // 3]
			elif phase == TrainingPhase.MAIN:
				# Use all examples
				phase_dataset = dataset
			elif phase == TrainingPhase.REFINEMENT:
				# Focus on hard examples (not mastered)
				phase_dataset = [
					dataset[i] for i in range(len(dataset))
					if i not in mastered_indices
				]
				if not phase_dataset:
					if self.verbose:
						print("All examples mastered, skipping refinement")
					continue
			else:
				phase_dataset = dataset

			# Train on this phase
			history = self.train(
				phase_dataset,
				epochs=epochs_per_phase,
				early_stop=True,
				use_patience=True,
				shuffle_data=True,
			)
			all_history.extend(history)

			# Update mastered examples
			if history:
				last_stats = history[-1]
				hard_set = set(last_stats.hard_examples)
				for i in range(len(dataset)):
					if i not in hard_set:
						mastered_indices.add(i)

		return all_history

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

	def save_checkpoint(
		self,
		path: str,
		epoch: int,
		mastered_indices: set | None = None,
		phase: str | None = None,
	) -> str:
		"""
		Save training checkpoint to disk.

		Creates two files:
		- {path}.json: Training state (epoch, accuracy, history, etc.)
		- {path}.model.pt: Model weights

		Args:
			path: Base path for checkpoint files (without extension)
			epoch: Current epoch number
			mastered_indices: Set of mastered example indices
			phase: Current training phase (for curriculum learning)

		Returns:
			Path to the saved checkpoint JSON file
		"""
		path = Path(path)
		model_path = str(path.with_suffix('.model.pt'))
		checkpoint_path = str(path.with_suffix('.checkpoint.json'))

		# Save model weights
		self.model.save(model_path)

		# Convert training history to JSON-serializable format
		history_dicts = []
		for stats in self.training_history:
			if isinstance(stats, EpochStats):
				history_dicts.append({
					"epoch": stats.epoch,
					"total_errors": stats.total_errors,
					"bit_errors": stats.bit_errors,
					"total_positions": stats.total_positions,
					"accuracy": stats.accuracy,
					"layer_updates": stats.layer_updates,
					"layer_error_rates": stats.layer_error_rates,
					"examples_mastered": stats.examples_mastered,
					"hard_examples": stats.hard_examples,
				})
			else:
				history_dicts.append(stats)

		# Create checkpoint
		checkpoint = Checkpoint(
			epoch=epoch,
			best_accuracy=self.best_accuracy,
			epochs_without_improvement=self.epochs_without_improvement,
			training_history=history_dicts,
			mastered_indices=list(mastered_indices) if mastered_indices else [],
			training_mode=self.mode.name,
			phase=phase,
			model_path=model_path,
		)

		# Save checkpoint JSON
		with open(checkpoint_path, 'w') as f:
			json.dump(checkpoint.to_dict(), f, indent=2)

		if self.verbose:
			print(f"Saved checkpoint to {checkpoint_path}")

		return checkpoint_path

	def load_checkpoint(self, path: str) -> Checkpoint:
		"""
		Load training checkpoint from disk.

		Args:
			path: Path to checkpoint JSON file

		Returns:
			Loaded Checkpoint object
		"""
		path = Path(path)
		if not path.suffix:
			path = path.with_suffix('.checkpoint.json')

		with open(path, 'r') as f:
			data = json.load(f)

		checkpoint = Checkpoint.from_dict(data)

		# Restore trainer state
		self.best_accuracy = checkpoint.best_accuracy
		self.epochs_without_improvement = checkpoint.epochs_without_improvement

		# Restore training history as EpochStats objects
		self.training_history = []
		for stats_dict in checkpoint.training_history:
			self.training_history.append(EpochStats(**stats_dict))

		# Load model weights
		if Path(checkpoint.model_path).exists():
			self.model = type(self.model).load(checkpoint.model_path)

		if self.verbose:
			print(f"Loaded checkpoint from epoch {checkpoint.epoch}, "
				  f"best_accuracy={checkpoint.best_accuracy:.1f}%")

		return checkpoint

	def train_with_checkpoints(
		self,
		dataset: list[tuple[list[Tensor], list[Tensor]]],
		epochs: int = 10,
		checkpoint_every: int = 5,
		checkpoint_path: str = "training_checkpoint",
		resume_from: str | None = None,
		**kwargs,
	) -> list[EpochStats]:
		"""
		Train with periodic checkpointing.

		Args:
			dataset: List of (input_tokens, target_tokens) pairs
			epochs: Maximum number of epochs
			checkpoint_every: Save checkpoint every N epochs
			checkpoint_path: Base path for checkpoint files
			resume_from: Path to checkpoint to resume from
			**kwargs: Additional arguments passed to train()

		Returns:
			List of epoch statistics
		"""
		start_epoch = 0
		mastered_indices: set = set()

		# Resume from checkpoint if specified
		if resume_from is not None:
			checkpoint = self.load_checkpoint(resume_from)
			start_epoch = checkpoint.epoch + 1
			mastered_indices = set(checkpoint.mastered_indices)

			if self.verbose:
				print(f"Resuming from epoch {start_epoch}")

		all_history = []

		for epoch in range(start_epoch, epochs):
			# Train one epoch
			stats = self.train_epoch(dataset, epoch_num=epoch, shuffle_data=True)
			all_history.append(stats)
			self.training_history.append(stats)

			# Update mastered indices
			hard_set = set(stats.hard_examples)
			for i in range(len(dataset)):
				if i not in hard_set:
					mastered_indices.add(i)

			# Check for improvement
			if stats.accuracy > self.best_accuracy:
				old_acc = self.best_accuracy
				self.best_accuracy = stats.accuracy
				self.epochs_without_improvement = 0

				for callback in self.callbacks:
					if hasattr(callback, 'on_improvement'):
						callback.on_improvement(epoch, old_acc, stats.accuracy)
			else:
				self.epochs_without_improvement += 1

			# Verbose output
			if self.verbose:
				print(f"Epoch {epoch + 1}/{epochs}: "
					  f"{stats.total_errors} errors, "
					  f"{stats.accuracy:.1f}% accuracy, "
					  f"{stats.examples_mastered}/{len(dataset)} mastered")

			# Save checkpoint
			if (epoch + 1) % checkpoint_every == 0:
				self.save_checkpoint(
					path=checkpoint_path,
					epoch=epoch,
					mastered_indices=mastered_indices,
				)

			# Notify callbacks
			for callback in self.callbacks:
				if hasattr(callback, 'on_epoch_end'):
					should_continue = callback.on_epoch_end(epoch, stats.__dict__)
					if should_continue is False:
						if self.verbose:
							print("Training stopped by callback")
						return all_history

			# Early stopping
			if kwargs.get('early_stop', True) and stats.total_errors == 0:
				if self.verbose:
					print(f"Converged at epoch {epoch + 1}!")
				break

			if kwargs.get('use_patience', True) and self.epochs_without_improvement >= self.patience:
				if self.verbose:
					print(f"Early stopping: no improvement for {self.patience} epochs")
				break

		# Save final checkpoint
		self.save_checkpoint(
			path=checkpoint_path,
			epoch=epoch,
			mastered_indices=mastered_indices,
		)

		return all_history
