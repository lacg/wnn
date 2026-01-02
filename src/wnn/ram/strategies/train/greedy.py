"""
Greedy Training Strategy

Trains all layers in a single backward pass per example.
Fast but may miss layer dependencies.

This is the default training strategy for RAM networks.
"""

from torch import Tensor

from wnn.ram.strategies.base import (
	TrainStrategyBase,
	StepStats,
	EpochStats,
)
from wnn.ram.strategies.config import GreedyTrainConfig, TrainConfig


class GreedyTrainStrategy(TrainStrategyBase):
	"""
	Greedy training: single backward pass through all layers.

	For each training example:
	1. Forward pass to compute outputs
	2. Backpropagate targets through all layers
	3. Train each layer on its computed target

	This is fast but may not capture layer dependencies well.
	For better convergence on complex tasks, use IterativeTrainStrategy.

	Usage:
		strategy = GreedyTrainStrategy()
		history = strategy.train(model, dataset)

		# Or with custom config
		strategy = GreedyTrainStrategy(GreedyTrainConfig(
			epochs=20,
			early_stop=True,
			verbose=True,
		))
	"""

	def __init__(self, config: GreedyTrainConfig | TrainConfig | None = None):
		if config is None:
			config = GreedyTrainConfig()
		elif isinstance(config, TrainConfig) and not isinstance(config, GreedyTrainConfig):
			# Upgrade base config to greedy config
			config = GreedyTrainConfig(
				epochs=config.epochs,
				early_stop=config.early_stop,
				shuffle=config.shuffle,
				verbose=config.verbose,
			)
		super().__init__(config)

	def train_step(
		self,
		model,  # RAMSeq2Seq or compatible
		inputs: list[Tensor],
		targets: list[Tensor],
	) -> StepStats:
		"""
		Train on a single example with greedy (single-pass) backpropagation.

		Args:
			model: The model to train (must have train_step method or compatible interface)
			inputs: Input token sequence
			targets: Target token sequence

		Returns:
			Training statistics for this step
		"""
		# Ensure tensors are properly squeezed
		inputs = [t.squeeze() if t.ndim > 1 else t for t in inputs]
		targets = [t.squeeze() if t.ndim > 1 else t for t in targets]

		# Check if model has native train_step
		if hasattr(model, 'train_step') and callable(model.train_step):
			# Use model's native training
			result = model.train_step(inputs, targets)

			# Convert to StepStats if needed
			if hasattr(result, 'output_errors'):
				return StepStats(
					output_errors=result.output_errors,
					bit_errors=getattr(result, 'bit_errors', 0),
					layers_updated=getattr(result, 'layers_updated', {}),
				)
			elif isinstance(result, dict):
				# Model returns dict (e.g., RAMSeq2Seq.train_step)
				layer_updates = result.get('layers_updated', result.get('layer_updates', {}))
				# Convert list to dict if needed
				if isinstance(layer_updates, list):
					layer_updates = {f"layer_{i}": u for i, u in enumerate(layer_updates) if u}
				return StepStats(
					output_errors=result.get('output_errors', 0),
					bit_errors=result.get('bit_errors', 0),
					layers_updated=layer_updates if isinstance(layer_updates, dict) else {},
				)
			return StepStats(output_errors=int(result))

		# Fallback: manual training using model's layers
		return self._train_step_manual(model, inputs, targets)

	def _train_step_manual(
		self,
		model,
		inputs: list[Tensor],
		targets: list[Tensor],
	) -> StepStats:
		"""
		Manual training when model doesn't have train_step.

		Uses forward pass and layer-by-layer training.
		"""
		# Forward pass
		outputs = model.forward(inputs)

		# Count errors
		output_errors = sum(
			1 for out, tgt in zip(outputs, targets)
			if not (out.squeeze() == tgt.squeeze()).all()
		)

		bit_errors = sum(
			(out.squeeze() != tgt.squeeze()).sum().item()
			for out, tgt in zip(outputs, targets)
		)

		if output_errors == 0:
			return StepStats(output_errors=0, bit_errors=0)

		# Train layers using available methods
		layers_updated = {}

		# Try to get trainable layers
		if hasattr(model, 'get_layers'):
			layers = model.get_layers()
			for name, layer in layers.items():
				if hasattr(layer, 'commit'):
					# Train this layer (simplified - real impl would backprop targets)
					# This is a placeholder - full implementation needs target computation
					layers_updated[name] = 0

		return StepStats(
			output_errors=output_errors,
			bit_errors=bit_errors,
			layers_updated=layers_updated,
		)

	def train_epoch(
		self,
		model,
		dataset: list[tuple[list[Tensor], list[Tensor]]],
		epoch_num: int,
	) -> EpochStats:
		"""
		Train for one epoch with greedy updates.

		Overrides base to add greedy-specific behavior if needed.
		"""
		# Use base implementation which handles shuffle and aggregation
		return super().train_epoch(model, dataset, epoch_num)
