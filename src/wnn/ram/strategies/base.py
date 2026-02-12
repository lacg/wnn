"""
Strategy Base Protocols

Defines the interfaces for forward and training strategies.
Models can be composed with different strategies for flexibility.

Usage:
	model = RAMSeq2Seq(...)
	model.forward_strategy = AutoregressiveForward()
	model.train_strategy = CurriculumTrainStrategy()

	# Or pass at construction
	model = RAMSeq2Seq(
		...,
		forward_strategy=BeamSearchForward(BeamSearchConfig(beam_width=4)),
		train_strategy=ContrastiveTrainStrategy(ContrastiveTrainConfig()),
	)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from torch import Tensor

from wnn.ram.strategies.config import (
	ForwardConfig,
	TrainConfig,
)

if TYPE_CHECKING:
	from wnn.ram.core.base import RAMComponent
	from wnn.ram.core.RAMLayer import RAMLayer


# =============================================================================
# TRAINING STATISTICS
# =============================================================================

@dataclass
class StepStats:
	"""Statistics from a single training step."""
	output_errors: int = 0
	bit_errors: int = 0
	layers_updated: dict[str, int] = field(default_factory=dict)


@dataclass
class EpochStats:
	"""Statistics from a training epoch."""
	epoch: int = 0
	total_errors: int = 0
	bit_errors: int = 0
	total_positions: int = 0
	accuracy: float = 0.0
	layer_updates: dict[str, int] = field(default_factory=dict)
	examples_correct: int = 0
	examples_total: int = 0

	@property
	def example_accuracy(self) -> float:
		"""Accuracy at example level (all tokens correct)."""
		if self.examples_total == 0:
			return 0.0
		return 100 * self.examples_correct / self.examples_total


# =============================================================================
# FORWARD STRATEGY PROTOCOL
# =============================================================================

@runtime_checkable
class ForwardStrategy(Protocol):
	"""
	Protocol for forward pass strategies.

	A forward strategy defines HOW to run inference on a model:
	- Autoregressive: Token-by-token, using previous outputs
	- Parallel: All tokens at once (for non-causal models)
	- Beam search: Maintain multiple hypotheses
	- Sampling: Stochastic generation with temperature

	The strategy receives the model and can call its internal methods.
	"""

	@property
	def config(self) -> ForwardConfig:
		"""Get the strategy configuration."""
		...

	def forward(
		self,
		model: "RAMComponent",
		tokens: list[Tensor],
	) -> list[Tensor]:
		"""
		Run forward pass on the model.

		Args:
			model: The model to run inference on
			tokens: Input token sequence

		Returns:
			Output token sequence
		"""
		...


class ForwardStrategyBase(ABC):
	"""
	Abstract base class for forward strategies.

	Provides common functionality and enforces the protocol.
	"""

	def __init__(self, config: ForwardConfig | None = None):
		self._config = config or ForwardConfig()

	@property
	def config(self) -> ForwardConfig:
		return self._config

	@abstractmethod
	def forward(
		self,
		model: "RAMComponent",
		tokens: list[Tensor],
	) -> list[Tensor]:
		"""Run forward pass on the model."""
		...

	def __repr__(self) -> str:
		return f"{self.__class__.__name__}(config={self._config})"


# =============================================================================
# TRAIN STRATEGY PROTOCOL
# =============================================================================

@runtime_checkable
class TrainStrategy(Protocol):
	"""
	Protocol for training strategies.

	A training strategy defines HOW to train a model:
	- Greedy: Single backward pass through all layers
	- Iterative: Multiple passes until convergence
	- Curriculum: Progressive difficulty
	- Contrastive: Triplet-based discrimination learning

	The strategy receives the model and can call its internal methods.
	"""

	@property
	def config(self) -> TrainConfig:
		"""Get the strategy configuration."""
		...

	def train_step(
		self,
		model: "RAMComponent",
		inputs: list[Tensor],
		targets: list[Tensor],
	) -> StepStats:
		"""
		Train on a single example.

		Args:
			model: The model to train
			inputs: Input token sequence
			targets: Target token sequence

		Returns:
			Training statistics for this step
		"""
		...

	def train_epoch(
		self,
		model: "RAMComponent",
		dataset: list[tuple[list[Tensor], list[Tensor]]],
		epoch_num: int,
	) -> EpochStats:
		"""
		Train for one epoch on the dataset.

		Args:
			model: The model to train
			dataset: List of (inputs, targets) pairs
			epoch_num: Current epoch number

		Returns:
			Training statistics for this epoch
		"""
		...

	def train(
		self,
		model: "RAMComponent",
		dataset: list[tuple[list[Tensor], list[Tensor]]],
	) -> list[EpochStats]:
		"""
		Train the model on the dataset.

		Args:
			model: The model to train
			dataset: List of (inputs, targets) pairs

		Returns:
			Training history (list of epoch statistics)
		"""
		...


class TrainStrategyBase(ABC):
	"""
	Abstract base class for training strategies.

	Provides common functionality like:
	- Epoch iteration with early stopping
	- Shuffle handling
	- Verbose output
	- Statistics tracking
	"""

	def __init__(self, config: TrainConfig | None = None):
		self._config = config or TrainConfig()
		self._history: list[EpochStats] = []

	@property
	def config(self) -> TrainConfig:
		return self._config

	@property
	def history(self) -> list[EpochStats]:
		"""Get training history."""
		return self._history

	@abstractmethod
	def train_step(
		self,
		model: "RAMComponent",
		inputs: list[Tensor],
		targets: list[Tensor],
	) -> StepStats:
		"""Train on a single example."""
		...

	def train_epoch(
		self,
		model: "RAMComponent",
		dataset: list[tuple[list[Tensor], list[Tensor]]],
		epoch_num: int,
	) -> EpochStats:
		"""
		Train for one epoch on the dataset.

		Default implementation iterates over examples and aggregates stats.
		Override for custom epoch behavior (e.g., curriculum sampling).
		"""
		from random import shuffle as shuffle_list

		# Optionally shuffle
		if self._config.shuffle:
			dataset = list(dataset)  # Copy to avoid modifying original
			shuffle_list(dataset)

		# Track statistics
		total_errors = 0
		bit_errors = 0
		total_positions = 0
		layer_updates: dict[str, int] = {}
		examples_correct = 0

		for inputs, targets in dataset:
			# Train on this example
			stats = self.train_step(model, inputs, targets)

			# Aggregate
			total_errors += stats.output_errors
			bit_errors += stats.bit_errors
			total_positions += len(targets)

			for layer, count in stats.layers_updated.items():
				layer_updates[layer] = layer_updates.get(layer, 0) + count

			if stats.output_errors == 0:
				examples_correct += 1

		# Compute accuracy
		accuracy = 100 * (1 - total_errors / total_positions) if total_positions > 0 else 0

		return EpochStats(
			epoch=epoch_num,
			total_errors=total_errors,
			bit_errors=bit_errors,
			total_positions=total_positions,
			accuracy=accuracy,
			layer_updates=layer_updates,
			examples_correct=examples_correct,
			examples_total=len(dataset),
		)

	def train(
		self,
		model: "RAMComponent",
		dataset: list[tuple[list[Tensor], list[Tensor]]],
	) -> list[EpochStats]:
		"""
		Train the model for multiple epochs.

		Handles early stopping and verbose output.
		"""
		self._history = []

		for epoch in range(self._config.epochs):
			# Train epoch
			stats = self.train_epoch(model, dataset, epoch)
			self._history.append(stats)

			# Verbose output
			if self._config.verbose:
				print(f"Epoch {epoch + 1}/{self._config.epochs}: "
						f"accuracy={stats.accuracy:.1f}%, "
						f"errors={stats.total_errors}, "
						f"examples={stats.examples_correct}/{stats.examples_total}")

			# Early stopping
			if self._config.early_stop and stats.total_errors == 0:
				if self._config.verbose:
					print(f"Converged at epoch {epoch + 1}!")
				break

		return self._history

	def __repr__(self) -> str:
		return f"{self.__class__.__name__}(config={self._config})"


# =============================================================================
# MODEL INTERFACE FOR STRATEGIES
# =============================================================================

@runtime_checkable
class StrategyCompatible(Protocol):
	"""
	Protocol that models must implement to work with strategies.

	This defines the interface that strategies can rely on.
	"""

	def get_layers(self) -> dict[str, "RAMLayer"]:
		"""Get all trainable layers by name."""
		...

	def forward_layer(
		self,
		layer_name: str,
		inputs: Tensor,
	) -> Tensor:
		"""Run forward on a specific layer."""
		...

	def train_layer(
		self,
		layer_name: str,
		inputs: Tensor,
		targets: Tensor,
	) -> int:
		"""Train a specific layer, return number of updates."""
		...

	def compute_layer_targets(
		self,
		layer_name: str,
		desired_output: Tensor,
		actual_input: Tensor,
	) -> Tensor:
		"""Backpropagate to get desired input for a layer."""
		...
