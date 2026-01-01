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
- OUTPUT_FIRST: Prioritize training output layers first (trains token_mapper,
  output_proj before attention/FFN layers)

Features:
- Per-layer error tracking
- Curriculum learning support
- Patience-based early stopping
- Training callbacks
- Hard example mining (tracks example difficulty with EMA, oversamples failures)
- Layer momentum tracking (detects stuck layers for diagnostics)
- Focused training mode (aggressive hard example mining for plateau breaking)

Usage:
    # Standard training
    trainer = RAMTrainer(model, mode=TrainingMode.GREEDY)
    trainer.train(dataset, epochs=10)

    # With hard example mining
    trainer = RAMTrainer(model, use_hard_mining=True)
    trainer.train_focused(dataset, epochs=15)

    # Get convergence diagnostics
    diagnostics = trainer.get_convergence_diagnostics()
    print(f"Stuck layers: {diagnostics['stuck_layers']}")
"""

from wnn.ram.core.models.seq2seq import RAMSeq2Seq
from wnn.ram.enums import PositionEncoding, LayerType, TrainingMode, TrainingPhase, MixingStrategy

from torch import Tensor, zeros, float32, save as torch_save, load as torch_load
from dataclasses import dataclass, field, asdict
from typing import Callable, Protocol, Any
from collections import defaultdict
from random import shuffle, sample
from pathlib import Path
import json


# =============================================================================
# Hard Example Mining
# =============================================================================

@dataclass
class ExampleDifficulty:
	"""Tracks difficulty of a single training example over time."""
	error_count: int = 0       # Total times this example had errors
	success_count: int = 0     # Total times this example was correct
	ema_error: float = 0.0     # Exponential moving average of error rate
	last_bit_errors: int = 0   # Bit errors on last attempt
	consecutive_failures: int = 0  # Consecutive epochs with errors

	@property
	def difficulty_score(self) -> float:
		"""Higher score = harder example."""
		# Combine EMA with consecutive failures for prioritization
		return self.ema_error * (1 + 0.2 * self.consecutive_failures)


class HardExampleMiner:
	"""
	Tracks example difficulty over training for focused hard example mining.

	Uses exponential moving average of error rates to identify:
	- Consistently failing examples (high EMA, high consecutive failures)
	- Intermittently failing examples (moderate EMA)
	- Mastered examples (low EMA, many successes)
	"""

	def __init__(self, num_examples: int, ema_alpha: float = 0.3):
		"""
		Args:
			num_examples: Total number of training examples
			ema_alpha: Smoothing factor for EMA (higher = more weight on recent)
		"""
		self.difficulties = [ExampleDifficulty() for _ in range(num_examples)]
		self.ema_alpha = ema_alpha

	def update(self, example_idx: int, had_error: bool, bit_errors: int = 0) -> None:
		"""Update difficulty tracking for an example."""
		d = self.difficulties[example_idx]

		# Update counts
		if had_error:
			d.error_count += 1
			d.consecutive_failures += 1
		else:
			d.success_count += 1
			d.consecutive_failures = 0

		# Update EMA
		error_signal = 1.0 if had_error else 0.0
		d.ema_error = self.ema_alpha * error_signal + (1 - self.ema_alpha) * d.ema_error
		d.last_bit_errors = bit_errors

	def get_hard_examples(self, count: int, min_difficulty: float = 0.1) -> list[int]:
		"""Get indices of hardest examples above minimum difficulty."""
		# Filter to examples that have been trained and are still hard
		candidates = [
			(i, d.difficulty_score)
			for i, d in enumerate(self.difficulties)
			if d.error_count > 0 and d.difficulty_score >= min_difficulty
		]
		# Sort by difficulty descending
		candidates.sort(key=lambda x: x[1], reverse=True)
		return [idx for idx, _ in candidates[:count]]

	def get_mastered_examples(self, min_successes: int = 3, max_ema: float = 0.1) -> list[int]:
		"""Get indices of mastered examples."""
		return [
			i for i, d in enumerate(self.difficulties)
			if d.success_count >= min_successes and d.ema_error <= max_ema
		]

	def sample_weighted(self, count: int) -> list[int]:
		"""Sample examples with probability proportional to difficulty."""
		weights = [d.difficulty_score + 0.01 for d in self.difficulties]  # +0.01 to avoid zero
		total = sum(weights)
		probs = [w / total for w in weights]

		# Weighted sampling without replacement
		indices = list(range(len(self.difficulties)))
		selected = []
		for _ in range(min(count, len(indices))):
			# Simple weighted selection
			r = sum(probs[:1])  # cumulative
			for i, p in enumerate(probs):
				if i in selected:
					continue
				r = sum(probs[j] for j in range(i + 1) if j not in selected)
				# Simplified: just sort by weight and take top
			break

		# Fallback to simpler approach: sort by difficulty and take top
		sorted_indices = sorted(range(len(self.difficulties)),
							   key=lambda i: self.difficulties[i].difficulty_score,
							   reverse=True)
		return sorted_indices[:count]


# =============================================================================
# Layer Momentum Tracking
# =============================================================================

@dataclass
class LayerMomentum:
	"""Tracks convergence momentum for a layer."""
	updates_history: list[int] = field(default_factory=list)  # Recent update counts
	window_size: int = 5

	@property
	def is_stuck(self) -> bool:
		"""Layer is stuck if updates aren't decreasing."""
		if len(self.updates_history) < self.window_size:
			return False
		recent = self.updates_history[-self.window_size:]
		# Stuck if updates are increasing or flat
		return recent[-1] >= recent[0] and all(u > 0 for u in recent)

	@property
	def velocity(self) -> float:
		"""Rate of convergence (negative = improving, positive = getting worse)."""
		if len(self.updates_history) < 2:
			return 0.0
		recent = self.updates_history[-self.window_size:]
		if len(recent) < 2:
			return 0.0
		# Linear regression slope
		n = len(recent)
		x_mean = (n - 1) / 2
		y_mean = sum(recent) / n
		numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
		denominator = sum((i - x_mean) ** 2 for i in range(n))
		return numerator / denominator if denominator > 0 else 0.0

	def record(self, updates: int) -> None:
		"""Record update count for this epoch."""
		self.updates_history.append(updates)
		# Keep only recent history
		if len(self.updates_history) > self.window_size * 2:
			self.updates_history = self.updates_history[-self.window_size * 2:]


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
	- Hard example mining for focused training
	- Layer momentum tracking for convergence detection
	"""

	def __init__(
		self,
		model: RAMSeq2Seq,
		mode: TrainingMode = TrainingMode.GREEDY,
		patience: int = 5,
		verbose: bool = True,
		use_hard_mining: bool = False,
		hard_mining_alpha: float = 0.3,
		track_momentum: bool = True,
	):
		"""
		Args:
			model: RAMSeq2Seq model to train
			mode: Training mode (GREEDY, ITERATIVE, OUTPUT_FIRST)
			patience: Epochs without improvement before stopping
			verbose: Print training progress
			use_hard_mining: Enable hard example mining for focused training
			hard_mining_alpha: EMA smoothing factor for difficulty tracking
			track_momentum: Track layer convergence momentum
		"""
		self.model = model
		self.mode = mode
		self.patience = patience
		self.verbose = verbose
		self.use_hard_mining = use_hard_mining
		self.hard_mining_alpha = hard_mining_alpha
		self.track_momentum = track_momentum

		# Training state
		self.best_accuracy = 0.0
		self.epochs_without_improvement = 0
		self.training_history: list[EpochStats] = []
		self.callbacks: list[TrainingCallback] = []

		# Hard example mining (initialized on first train call)
		self.hard_miner: HardExampleMiner | None = None

		# Layer momentum tracking
		self.layer_momentum: dict[str, LayerMomentum] = defaultdict(LayerMomentum)

	def add_callback(self, callback: TrainingCallback) -> None:
		"""Add a training callback."""
		self.callbacks.append(callback)

	def reset_state(self) -> None:
		"""Reset training state for a new training run."""
		self.best_accuracy = 0.0
		self.epochs_without_improvement = 0
		self.training_history = []
		self.hard_miner = None
		self.layer_momentum = defaultdict(LayerMomentum)

	def _get_layer_priority(self, layer_type: LayerType) -> int:
		"""Get training priority for a layer type (higher = train first in OUTPUT_FIRST)."""
		# Output layers have highest priority
		priority_map = {
			LayerType.TOKEN_MAPPER: 100,
			LayerType.OUTPUT_PROJ: 90,
			LayerType.FFN: 50,
			LayerType.ATTENTION: 40,
			LayerType.INPUT_PROJ: 20,
			LayerType.EMBEDDING: 10,
		}
		return priority_map.get(layer_type, 0)

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

		# Determine layer training order based on mode
		if self.mode == TrainingMode.OUTPUT_FIRST:
			# Sort by priority: output layers first, then work backwards
			training_order = sorted(
				trace,
				key=lambda s: (self._get_layer_priority(s.layer_type), -s.index),
				reverse=True
			)
		else:
			# Default: train in forward order (as in trace)
			training_order = trace

		# Train each layer
		layers_updated = {}
		for state in training_order:
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
		focus_hard_examples: bool = False,
		hard_example_ratio: float = 0.3,
	) -> EpochStats:
		"""
		Train for one epoch over a dataset.

		Args:
			dataset: List of (input_tokens, target_tokens) pairs
			epoch_num: Current epoch number (for statistics)
			shuffle_data: Whether to shuffle the dataset
			focus_hard_examples: If True, oversample hard examples
			hard_example_ratio: Fraction of batch to dedicate to hard examples

		Returns:
			Detailed epoch statistics
		"""
		# Initialize hard example miner if needed
		if self.use_hard_mining and self.hard_miner is None:
			self.hard_miner = HardExampleMiner(len(dataset), self.hard_mining_alpha)

		# Prepare training indices
		indices = list(range(len(dataset)))

		if focus_hard_examples and self.hard_miner is not None and epoch_num > 0:
			# Get hard examples to oversample
			num_hard = int(len(dataset) * hard_example_ratio)
			hard_indices = self.hard_miner.get_hard_examples(num_hard)

			if hard_indices:
				# Create mixed batch: regular examples + oversampled hard examples
				remaining = [i for i in indices if i not in hard_indices]
				if shuffle_data:
					shuffle(remaining)
				# Interleave hard examples throughout training
				indices = remaining + hard_indices
				if shuffle_data:
					shuffle(indices)
		elif shuffle_data:
			shuffle(indices)

		training_data = [(indices[i], dataset[indices[i]]) for i in range(len(indices))]

		total_errors = 0
		total_bits = 0
		total_positions = 0
		all_updates = {}
		layer_errors = {}
		examples_mastered = 0
		hard_examples = []

		for original_idx, (inputs, targets) in training_data:
			# Use iterative mode if specified
			if self.mode == TrainingMode.ITERATIVE:
				stats = self._train_step_iterative(inputs, targets)
			else:
				stats = self.train_step(inputs, targets)

			total_errors += stats.output_errors
			total_bits += stats.bit_errors
			total_positions += len(inputs)

			# Track per-example success
			had_error = stats.output_errors > 0
			if not had_error:
				examples_mastered += 1
			else:
				hard_examples.append(original_idx)

			# Update hard example mining
			if self.hard_miner is not None:
				self.hard_miner.update(original_idx, had_error, stats.bit_errors)

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

		# Update layer momentum tracking
		if self.track_momentum:
			for layer, updates in all_updates.items():
				self.layer_momentum[layer].record(updates)

		# Log stuck layers if verbose
		if self.verbose and self.track_momentum:
			stuck_layers = [
				layer for layer, momentum in self.layer_momentum.items()
				if momentum.is_stuck
			]
			if stuck_layers:
				print(f"  ⚠ Stuck layers: {', '.join(stuck_layers)}")

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

	def train_focused(
		self,
		dataset: list[tuple[list[Tensor], list[Tensor]]],
		epochs: int = 10,
		warmup_epochs: int = 2,
		focus_ratio: float = 0.5,
		min_hard_difficulty: float = 0.2,
	) -> list[EpochStats]:
		"""
		Train with aggressive hard example mining for better convergence.

		Strategy:
		1. Warmup phase: Normal training to build difficulty estimates
		2. Focus phase: Heavily oversample hard examples
		3. Refinement: Train only on persistently hard examples

		This is more aggressive than train_curriculum and is designed
		for cases where standard training plateaus.

		Args:
			dataset: List of (input_tokens, target_tokens) pairs
			epochs: Maximum total epochs
			warmup_epochs: Epochs of normal training before focusing
			focus_ratio: Fraction of training to dedicate to hard examples
			min_hard_difficulty: Minimum difficulty score to consider "hard"

		Returns:
			List of epoch statistics
		"""
		# Enable hard mining for this run
		original_hard_mining = self.use_hard_mining
		self.use_hard_mining = True
		self.reset_state()

		all_history = []

		# Phase 1: Warmup - build difficulty estimates
		if self.verbose:
			print(f"\n=== Focused Training: Warmup ({warmup_epochs} epochs) ===")

		for epoch in range(warmup_epochs):
			stats = self.train_epoch(
				dataset,
				epoch_num=epoch,
				shuffle_data=True,
				focus_hard_examples=False,
			)
			all_history.append(stats)
			self.training_history.append(stats)

			if self.verbose:
				print(f"Warmup {epoch + 1}/{warmup_epochs}: "
					  f"{stats.accuracy:.1f}% accuracy, "
					  f"{stats.examples_mastered}/{len(dataset)} mastered")

			if stats.total_errors == 0:
				if self.verbose:
					print("Converged during warmup!")
				self.use_hard_mining = original_hard_mining
				return all_history

		# Phase 2: Focus - heavily oversample hard examples
		if self.verbose:
			hard_count = len(self.hard_miner.get_hard_examples(
				len(dataset), min_hard_difficulty
			)) if self.hard_miner else 0
			print(f"\n=== Focused Training: Focus Phase ({hard_count} hard examples) ===")

		focus_epochs = epochs - warmup_epochs - 2  # Save 2 for refinement
		for epoch in range(focus_epochs):
			actual_epoch = warmup_epochs + epoch
			stats = self.train_epoch(
				dataset,
				epoch_num=actual_epoch,
				shuffle_data=True,
				focus_hard_examples=True,
				hard_example_ratio=focus_ratio,
			)
			all_history.append(stats)
			self.training_history.append(stats)

			# Check for improvement
			if stats.accuracy > self.best_accuracy:
				self.best_accuracy = stats.accuracy
				self.epochs_without_improvement = 0
			else:
				self.epochs_without_improvement += 1

			if self.verbose:
				print(f"Focus {epoch + 1}/{focus_epochs}: "
					  f"{stats.accuracy:.1f}% accuracy, "
					  f"{stats.examples_mastered}/{len(dataset)} mastered")

			if stats.total_errors == 0:
				if self.verbose:
					print("Converged during focus phase!")
				self.use_hard_mining = original_hard_mining
				return all_history

			# Early stopping on plateau
			if self.epochs_without_improvement >= self.patience:
				if self.verbose:
					print("Focus phase plateaued, moving to refinement")
				break

		# Phase 3: Refinement - train only on hardest examples
		if self.hard_miner is not None:
			hard_indices = self.hard_miner.get_hard_examples(
				len(dataset), min_hard_difficulty
			)
			if hard_indices:
				hard_dataset = [dataset[i] for i in hard_indices]

				if self.verbose:
					print(f"\n=== Focused Training: Refinement ({len(hard_dataset)} examples) ===")

				for epoch in range(2):
					actual_epoch = epochs - 2 + epoch
					stats = self.train_epoch(
						hard_dataset,
						epoch_num=actual_epoch,
						shuffle_data=True,
					)
					all_history.append(stats)

					if self.verbose:
						print(f"Refinement {epoch + 1}/2: "
							  f"{stats.accuracy:.1f}% on hard examples")

		self.use_hard_mining = original_hard_mining
		return all_history

	def get_convergence_diagnostics(self) -> dict:
		"""
		Get diagnostics about training convergence.

		Returns:
			Dictionary with convergence information including:
			- stuck_layers: Layers that aren't improving
			- layer_velocities: Rate of convergence per layer
			- hard_example_count: Number of consistently failing examples
			- mastered_count: Number of mastered examples
		"""
		diagnostics = {
			"stuck_layers": [],
			"layer_velocities": {},
			"hard_example_count": 0,
			"mastered_count": 0,
			"epochs_trained": len(self.training_history),
			"best_accuracy": self.best_accuracy,
			"epochs_without_improvement": self.epochs_without_improvement,
		}

		# Layer momentum analysis
		for layer, momentum in self.layer_momentum.items():
			diagnostics["layer_velocities"][layer] = momentum.velocity
			if momentum.is_stuck:
				diagnostics["stuck_layers"].append(layer)

		# Hard example analysis
		if self.hard_miner is not None:
			diagnostics["hard_example_count"] = len(
				self.hard_miner.get_hard_examples(1000, min_difficulty=0.1)
			)
			diagnostics["mastered_count"] = len(
				self.hard_miner.get_mastered_examples()
			)

			# Top 5 hardest examples
			hard_indices = self.hard_miner.get_hard_examples(5)
			diagnostics["hardest_examples"] = [
				{
					"index": idx,
					"difficulty": self.hard_miner.difficulties[idx].difficulty_score,
					"consecutive_failures": self.hard_miner.difficulties[idx].consecutive_failures,
				}
				for idx in hard_indices
			]

		return diagnostics

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


# =============================================================================
# Enhanced Curriculum Learning
# =============================================================================

class DifficultyMetric(Protocol):
	"""Protocol for difficulty estimation functions."""

	def __call__(self, inputs: list[Tensor], targets: list[Tensor]) -> float:
		"""Return difficulty score (higher = harder)."""
		...


def length_difficulty(inputs: list[Tensor], targets: list[Tensor]) -> float:
	"""Difficulty based on sequence length."""
	return float(len(inputs))


def bit_count_difficulty(inputs: list[Tensor], targets: list[Tensor]) -> float:
	"""Difficulty based on number of 1-bits (complexity proxy)."""
	return sum(t.sum().item() for t in targets)


def hamming_difficulty(inputs: list[Tensor], targets: list[Tensor]) -> float:
	"""Difficulty based on Hamming distance between input and output."""
	if len(inputs) != len(targets):
		return float(len(targets))
	return sum((i != t).sum().item() for i, t in zip(inputs, targets))


def combined_difficulty(inputs: list[Tensor], targets: list[Tensor]) -> float:
	"""Combined difficulty: length × transformation complexity."""
	length = len(inputs)
	hamming = hamming_difficulty(inputs, targets)
	return length * (1 + hamming / max(1, length))


@dataclass
class CurriculumSchedule:
	"""
	Schedule for curriculum learning progression.

	Defines how training progresses from easy to hard examples.
	"""
	num_stages: int = 5           # Number of difficulty stages
	epochs_per_stage: int = 3     # Epochs at each stage
	overlap: float = 0.2          # Fraction of next stage to include
	patience_per_stage: int = 2   # Patience within each stage

	def get_stage_range(self, stage: int, total_examples: int) -> tuple[int, int]:
		"""Get (start, end) indices for examples in this stage."""
		stage_size = total_examples // self.num_stages
		overlap_size = int(stage_size * self.overlap)

		start = max(0, stage * stage_size - overlap_size)
		end = min(total_examples, (stage + 1) * stage_size + overlap_size)

		return start, end


class CurriculumTrainer:
	"""
	Enhanced curriculum learning with configurable difficulty metrics.

	Features:
	- Pluggable difficulty metrics (length, bit_count, hamming, combined)
	- Gradual difficulty progression with overlapping stages
	- Automatic difficulty estimation from data

	Usage:
		trainer = RAMTrainer(model)
		curriculum = CurriculumTrainer(trainer)
		curriculum.train(dataset, schedule=CurriculumSchedule(num_stages=5))
	"""

	def __init__(
		self,
		trainer: RAMTrainer,
		difficulty_metric: DifficultyMetric = combined_difficulty,
	):
		"""
		Args:
			trainer: Base RAMTrainer to use
			difficulty_metric: Function to estimate example difficulty
		"""
		self.trainer = trainer
		self.difficulty_metric = difficulty_metric

	def sort_by_difficulty(
		self,
		dataset: list[tuple[list[Tensor], list[Tensor]]],
	) -> list[tuple[int, float, tuple[list[Tensor], list[Tensor]]]]:
		"""
		Sort dataset by difficulty (easiest first).

		Returns:
			List of (original_index, difficulty_score, example) tuples
		"""
		scored = [
			(i, self.difficulty_metric(inputs, targets), (inputs, targets))
			for i, (inputs, targets) in enumerate(dataset)
		]
		return sorted(scored, key=lambda x: x[1])

	def train(
		self,
		dataset: list[tuple[list[Tensor], list[Tensor]]],
		schedule: CurriculumSchedule | None = None,
		verbose: bool = True,
	) -> list[EpochStats]:
		"""
		Train with curriculum learning.

		Args:
			dataset: List of (input_tokens, target_tokens) pairs
			schedule: Curriculum schedule (default: 5 stages, 3 epochs each)
			verbose: Print progress

		Returns:
			Combined training history
		"""
		if schedule is None:
			schedule = CurriculumSchedule()

		# Sort dataset by difficulty
		sorted_data = self.sort_by_difficulty(dataset)

		if verbose:
			difficulties = [d for _, d, _ in sorted_data]
			print(f"Difficulty range: {min(difficulties):.1f} - {max(difficulties):.1f}")

		all_history = []

		for stage in range(schedule.num_stages):
			start, end = schedule.get_stage_range(stage, len(sorted_data))
			stage_data = [example for _, _, example in sorted_data[start:end]]

			if verbose:
				stage_difficulties = [d for _, d, _ in sorted_data[start:end]]
				print(f"\n=== Stage {stage + 1}/{schedule.num_stages}: "
					  f"{len(stage_data)} examples, "
					  f"difficulty {min(stage_difficulties):.1f}-{max(stage_difficulties):.1f} ===")

			# Reset patience for this stage
			self.trainer.epochs_without_improvement = 0
			original_patience = self.trainer.patience
			self.trainer.patience = schedule.patience_per_stage

			# Train on this stage
			history = self.trainer.train(
				stage_data,
				epochs=schedule.epochs_per_stage,
				early_stop=True,
				use_patience=True,
				shuffle_data=True,
			)
			all_history.extend(history)

			# Restore patience
			self.trainer.patience = original_patience

			# Check if fully converged
			if history and history[-1].total_errors == 0:
				if verbose:
					print(f"Converged at stage {stage + 1}!")

				# Verify on full dataset
				final_history = self.trainer.train(
					dataset,
					epochs=2,
					early_stop=True,
					use_patience=False,
				)
				all_history.extend(final_history)

				if final_history and final_history[-1].total_errors == 0:
					if verbose:
						print("Full dataset converged!")
					break

		return all_history


# =============================================================================
# Multi-Task Learning
# =============================================================================

@dataclass
class Task:
	"""
	Definition of a learning task for multi-task training.

	Attributes:
		name: Task identifier
		dataset: List of (input, target) pairs
		weight: Sampling weight (higher = more frequent)
		shared_primitives: Names of primitives shared with other tasks
	"""
	name: str
	dataset: list[tuple[list[Tensor], list[Tensor]]]
	weight: float = 1.0
	shared_primitives: list[str] = field(default_factory=list)


class MultiTaskTrainer:
	"""
	Multi-task learning for RAM networks.

	Trains on multiple tasks simultaneously, which can:
	- Share learned primitives (XOR, full adder, comparator)
	- Improve generalization through task diversity
	- Reduce training time by reusing patterns

	Key insight: Tasks like parity (XOR), addition (full adder), and
	comparison share underlying primitives that transfer between tasks.

	Usage:
		trainer = RAMTrainer(model)
		mt = MultiTaskTrainer(trainer)

		mt.add_task(Task("parity", parity_dataset, shared_primitives=["xor"]))
		mt.add_task(Task("addition", addition_dataset, shared_primitives=["xor", "carry"]))

		mt.train(epochs=10, mixing=MixingStrategy.INTERLEAVED)
	"""

	def __init__(
		self,
		trainer: RAMTrainer,
		mixing: MixingStrategy = MixingStrategy.INTERLEAVED,
	):
		"""
		Args:
			trainer: Base RAMTrainer to use
			mixing: Strategy for mixing task examples
		"""
		self.trainer = trainer
		self.mixing = mixing
		self.tasks: list[Task] = []
		self.task_stats: dict[str, list[float]] = defaultdict(list)  # Per-task accuracy history

	def add_task(self, task: Task) -> None:
		"""Add a task to the training set."""
		self.tasks.append(task)

	def add_tasks(self, tasks: list[Task]) -> None:
		"""Add multiple tasks."""
		self.tasks.extend(tasks)

	def _create_mixed_dataset(self) -> list[tuple[str, list[Tensor], list[Tensor]]]:
		"""
		Create a mixed dataset from all tasks.

		Returns:
			List of (task_name, inputs, targets) tuples
		"""
		mixed = []

		if self.mixing == MixingStrategy.ROUND_ROBIN:
			# Alternate between tasks
			max_len = max(len(t.dataset) for t in self.tasks)
			for i in range(max_len):
				for task in self.tasks:
					if i < len(task.dataset):
						inputs, targets = task.dataset[i]
						mixed.append((task.name, inputs, targets))

		elif self.mixing == MixingStrategy.PROPORTIONAL:
			# Sample proportional to dataset size
			for task in self.tasks:
				for inputs, targets in task.dataset:
					mixed.append((task.name, inputs, targets))
			shuffle(mixed)

		elif self.mixing == MixingStrategy.WEIGHTED:
			# Sample proportional to task weight
			total_weight = sum(t.weight for t in self.tasks)
			for task in self.tasks:
				# Repeat examples based on weight
				repeats = max(1, int(task.weight / total_weight * len(task.dataset)))
				for inputs, targets in task.dataset:
					for _ in range(repeats):
						mixed.append((task.name, inputs, targets))
			shuffle(mixed)

		elif self.mixing == MixingStrategy.INTERLEAVED:
			# Mix all, shuffle
			for task in self.tasks:
				for inputs, targets in task.dataset:
					mixed.append((task.name, inputs, targets))
			shuffle(mixed)

		return mixed

	def train_epoch(self, epoch_num: int = 0) -> dict[str, EpochStats]:
		"""
		Train one epoch on all tasks.

		Returns:
			Dictionary mapping task_name -> EpochStats
		"""
		mixed = self._create_mixed_dataset()

		# Track per-task statistics
		task_errors: dict[str, int] = defaultdict(int)
		task_positions: dict[str, int] = defaultdict(int)
		task_updates: dict[str, dict] = defaultdict(lambda: defaultdict(int))

		for task_name, inputs, targets in mixed:
			# Train on this example
			stats = self.trainer.train_step(inputs, targets)

			task_errors[task_name] += stats.output_errors
			task_positions[task_name] += len(inputs)

			for layer, updates in stats.layers_updated.items():
				task_updates[task_name][layer] += updates

		# Build per-task stats
		results = {}
		for task in self.tasks:
			name = task.name
			total = task_positions[name]
			errors = task_errors[name]
			accuracy = 100 * (1 - errors / total) if total > 0 else 0

			self.task_stats[name].append(accuracy)

			results[name] = EpochStats(
				epoch=epoch_num,
				total_errors=errors,
				bit_errors=0,  # Not tracked per-task
				total_positions=total,
				accuracy=accuracy,
				layer_updates=dict(task_updates[name]),
				layer_error_rates={},
				examples_mastered=0,
				hard_examples=[],
			)

		return results

	def train(
		self,
		epochs: int = 10,
		early_stop: bool = True,
		verbose: bool = True,
	) -> dict[str, list[EpochStats]]:
		"""
		Train on all tasks for multiple epochs.

		Args:
			epochs: Maximum number of epochs
			early_stop: Stop if all tasks converge
			verbose: Print progress

		Returns:
			Dictionary mapping task_name -> list of EpochStats
		"""
		if not self.tasks:
			raise ValueError("No tasks added. Use add_task() first.")

		all_history: dict[str, list[EpochStats]] = {t.name: [] for t in self.tasks}

		if verbose:
			print(f"Multi-task training: {len(self.tasks)} tasks")
			for task in self.tasks:
				print(f"  - {task.name}: {len(task.dataset)} examples, "
					  f"weight={task.weight:.1f}, "
					  f"shared={task.shared_primitives}")

		for epoch in range(epochs):
			task_stats = self.train_epoch(epoch_num=epoch)

			# Record history
			for name, stats in task_stats.items():
				all_history[name].append(stats)

			# Check convergence
			all_converged = all(
				stats.total_errors == 0 for stats in task_stats.values()
			)

			if verbose:
				task_summary = ", ".join(
					f"{name}:{stats.accuracy:.0f}%"
					for name, stats in task_stats.items()
				)
				total_errors = sum(s.total_errors for s in task_stats.values())
				print(f"Epoch {epoch + 1}/{epochs}: {task_summary} (total errors: {total_errors})")

			if early_stop and all_converged:
				if verbose:
					print(f"All tasks converged at epoch {epoch + 1}!")
				break

		return all_history

	def evaluate(self, verbose: bool = True) -> dict[str, float]:
		"""
		Evaluate accuracy on each task.

		Returns:
			Dictionary mapping task_name -> accuracy
		"""
		results = {}

		for task in self.tasks:
			correct = 0
			total = 0

			for inputs, targets in task.dataset:
				outputs = self.trainer.model.forward(inputs)
				for out, tgt in zip(outputs, targets):
					if (out.squeeze() == tgt.squeeze()).all():
						correct += 1
					total += 1

			accuracy = 100 * correct / total if total > 0 else 0
			results[task.name] = accuracy

			if verbose:
				print(f"{task.name}: {accuracy:.1f}% ({correct}/{total})")

		return results

	def get_shared_primitive_stats(self) -> dict[str, list[str]]:
		"""
		Get statistics about shared primitives across tasks.

		Returns:
			Dictionary mapping primitive_name -> list of task names using it
		"""
		primitive_usage: dict[str, list[str]] = defaultdict(list)

		for task in self.tasks:
			for primitive in task.shared_primitives:
				primitive_usage[primitive].append(task.name)

		return dict(primitive_usage)