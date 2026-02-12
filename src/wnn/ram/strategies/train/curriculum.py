"""
Curriculum Training Strategy

Trains on progressively harder examples for better generalization.
Starts with easy patterns and gradually introduces complexity.
"""

from torch import Tensor

from wnn.ram.strategies.base import TrainStrategyBase, StepStats, EpochStats
from wnn.ram.strategies.config import CurriculumTrainConfig, TrainConfig


class CurriculumTrainStrategy(TrainStrategyBase):
	"""
	Curriculum learning: progressive difficulty training.

	For each stage:
	1. Select examples appropriate for current difficulty
	2. Train until convergence or patience exhausted
	3. Move to next stage with harder examples

	Default difficulty: sequence length (shorter = easier).

	Usage:
		strategy = CurriculumTrainStrategy(CurriculumTrainConfig(
			num_stages=5,
			epochs_per_stage=3,
			overlap=0.2,
		))
		history = strategy.train(model, dataset)
	"""

	def __init__(self, config: CurriculumTrainConfig | TrainConfig | None = None):
		if config is None:
			config = CurriculumTrainConfig()
		elif isinstance(config, TrainConfig) and not isinstance(config, CurriculumTrainConfig):
			config = CurriculumTrainConfig(
				epochs=config.epochs,
				early_stop=config.early_stop,
				shuffle=config.shuffle,
				verbose=config.verbose,
			)
		super().__init__(config)

	@property
	def curriculum_config(self) -> CurriculumTrainConfig:
		"""Get config with curriculum-specific fields."""
		return self._config  # type: ignore

	def _compute_difficulty(
		self,
		inputs: list[Tensor],
		targets: list[Tensor],
	) -> float:
		"""
		Compute difficulty score for an example.

		Default: sequence length. Override via config.difficulty_metric.
		"""
		if self.curriculum_config.difficulty_metric is not None:
			return self.curriculum_config.difficulty_metric(inputs, targets)

		# Default: longer sequences are harder
		return float(len(inputs))

	def _sort_by_difficulty(
		self,
		dataset: list[tuple[list[Tensor], list[Tensor]]],
	) -> list[tuple[list[Tensor], list[Tensor], float]]:
		"""Sort dataset by difficulty, returning (inputs, targets, difficulty)."""
		scored = []
		for inputs, targets in dataset:
			diff = self._compute_difficulty(inputs, targets)
			scored.append((inputs, targets, diff))

		return sorted(scored, key=lambda x: x[2])

	def _get_stage_subset(
		self,
		sorted_data: list[tuple[list[Tensor], list[Tensor], float]],
		stage: int,
	) -> list[tuple[list[Tensor], list[Tensor]]]:
		"""
		Get examples for a curriculum stage.

		Early stages get easier examples, later stages get all examples.
		Overlap parameter controls how much easier examples carry forward.
		"""
		n_stages = self.curriculum_config.num_stages
		n_examples = len(sorted_data)

		if n_stages == 1:
			return [(inp, tgt) for inp, tgt, _ in sorted_data]

		# Compute boundaries
		stage_size = n_examples / n_stages
		overlap = self.curriculum_config.overlap

		# Start index: earlier stages start from 0
		start_idx = 0

		# End index: grows with each stage
		base_end = int((stage + 1) * stage_size)
		overlap_extend = int(stage * stage_size * overlap)
		end_idx = min(n_examples, base_end + overlap_extend)

		# For later stages, include some earlier examples (overlap)
		if stage > 0:
			overlap_start = max(0, int(stage * stage_size - overlap * stage_size))
			start_idx = overlap_start

		subset = sorted_data[start_idx:end_idx]
		return [(inp, tgt) for inp, tgt, _ in subset]

	def train_step(
		self,
		model,
		inputs: list[Tensor],
		targets: list[Tensor],
	) -> StepStats:
		"""
		Train on a single example.

		Delegates to model's train_step if available.
		"""
		inputs = [t.squeeze() if t.ndim > 1 else t for t in inputs]
		targets = [t.squeeze() if t.ndim > 1 else t for t in targets]

		if hasattr(model, 'train_step') and callable(model.train_step):
			result = model.train_step(inputs, targets)
			if hasattr(result, 'output_errors'):
				return StepStats(
					output_errors=result.output_errors,
					bit_errors=getattr(result, 'bit_errors', 0),
					layers_updated=getattr(result, 'layers_updated', {}),
				)
			elif isinstance(result, dict):
				layer_updates = result.get('layers_updated', result.get('layer_updates', {}))
				if isinstance(layer_updates, list):
					layer_updates = {f"layer_{i}": u for i, u in enumerate(layer_updates) if u}
				return StepStats(
					output_errors=result.get('output_errors', 0),
					bit_errors=result.get('bit_errors', 0),
					layers_updated=layer_updates if isinstance(layer_updates, dict) else {},
				)
			return StepStats(output_errors=int(result))

		# Fallback: forward pass only (no training)
		outputs = model.forward(inputs)
		errors = sum(
			1 for out, tgt in zip(outputs, targets)
			if not (out.squeeze() == tgt.squeeze()).all()
		)
		return StepStats(output_errors=errors)

	def train(
		self,
		model,
		dataset: list[tuple[list[Tensor], list[Tensor]]],
	) -> list[EpochStats]:
		"""
		Train with curriculum learning.

		Overrides base to implement stage-based training.
		"""
		self._history = []

		# Sort by difficulty
		sorted_data = self._sort_by_difficulty(dataset)

		if self.curriculum_config.verbose:
			difficulties = [d for _, _, d in sorted_data]
			print(f"Curriculum: {len(dataset)} examples, "
					f"difficulty range [{min(difficulties):.1f}, {max(difficulties):.1f}]")

		epoch_counter = 0
		for stage in range(self.curriculum_config.num_stages):
			# Get subset for this stage
			stage_data = self._get_stage_subset(sorted_data, stage)

			if self.curriculum_config.verbose:
				print(f"\n=== Stage {stage + 1}/{self.curriculum_config.num_stages} "
						f"({len(stage_data)} examples) ===")

			# Track patience for this stage
			best_errors = float('inf')
			patience_counter = 0

			for stage_epoch in range(self.curriculum_config.epochs_per_stage):
				stats = self.train_epoch(model, stage_data, epoch_counter)
				self._history.append(stats)
				epoch_counter += 1

				if self.curriculum_config.verbose:
					print(f"  Epoch {stage_epoch + 1}: "
							f"accuracy={stats.accuracy:.1f}%, "
							f"errors={stats.total_errors}")

				# Check for improvement
				if stats.total_errors < best_errors:
					best_errors = stats.total_errors
					patience_counter = 0
				else:
					patience_counter += 1

				# Early stop within stage
				if stats.total_errors == 0:
					break

				if patience_counter >= self.curriculum_config.patience_per_stage:
					if self.curriculum_config.verbose:
						print(f"  (patience exhausted at stage {stage + 1})")
					break

			# Early stop overall if perfect on full dataset
			if stage == self.curriculum_config.num_stages - 1 and best_errors == 0:
				if self.curriculum_config.verbose:
					print("Converged on full dataset!")
				break

		return self._history
