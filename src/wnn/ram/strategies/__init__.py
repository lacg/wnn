"""
Strategy classes for RAM neural networks.

Contains strategy pattern implementations for:
- Training strategies (greedy, iterative, curriculum, contrastive)
- Forward strategies (autoregressive, parallel)
- Attention masking strategies

Usage:
	from wnn.ram.strategies import (
		GreedyTrainStrategy,
		CurriculumTrainStrategy,
		AutoregressiveForwardStrategy,
		CausalMask,
	)

	# Configure and use
	model = RAMSeq2Seq(...)
	train_strategy = CurriculumTrainStrategy(CurriculumTrainConfig(num_stages=5))
	train_strategy.train(model, dataset)
"""

# Training strategies
from wnn.ram.strategies.train import (
	GreedyTrainStrategy,
	IterativeTrainStrategy,
	CurriculumTrainStrategy,
	ContrastiveTrainStrategy,
)

# Forward strategies
from wnn.ram.strategies.forward import (
	AutoregressiveForwardStrategy,
	ParallelForwardStrategy,
)

# Base classes and protocols
from wnn.ram.strategies.base import (
	ForwardStrategy,
	ForwardStrategyBase,
	TrainStrategy,
	TrainStrategyBase,
	StepStats,
	EpochStats,
	StrategyCompatible,
)

# Factories
from wnn.ram.strategies.factory import (
	TrainStrategyFactory,
	ForwardStrategyFactory,
	TrainStrategyType,
	ForwardStrategyType,
)

# Configuration dataclasses
from wnn.ram.strategies.config import (
	ForwardConfig,
	AutoregressiveConfig,
	SamplingConfig,
	BeamSearchConfig,
	TrainConfig,
	GreedyTrainConfig,
	IterativeTrainConfig,
	CurriculumTrainConfig,
	ContrastiveTrainConfig,
	MultiTaskTrainConfig,
	ScheduledSamplingConfig,
	PatienceConfig,
	CheckpointConfig,
)

# Attention mask strategies
from wnn.ram.strategies.attention_mask import (
	MaskStrategy,
	AttentionMaskStrategy,
	CausalMask,
	BidirectionalMask,
	SlidingWindowMask,
	BlockMask,
	PrefixMask,
	StridedMask,
	DilatedMask,
	LocalGlobalMask,
	CustomMask,
	LearnedSparseMask,
	MaskStrategyFactory,
	combine_masks,
)

__all__ = [
	# Training strategies
	'GreedyTrainStrategy',
	'IterativeTrainStrategy',
	'CurriculumTrainStrategy',
	'ContrastiveTrainStrategy',
	# Forward strategies
	'AutoregressiveForwardStrategy',
	'ParallelForwardStrategy',
	# Factories
	'TrainStrategyFactory',
	'ForwardStrategyFactory',
	'TrainStrategyType',
	'ForwardStrategyType',
	# Base classes
	'ForwardStrategy',
	'ForwardStrategyBase',
	'TrainStrategy',
	'TrainStrategyBase',
	'StepStats',
	'EpochStats',
	'StrategyCompatible',
	# Configs
	'ForwardConfig',
	'AutoregressiveConfig',
	'SamplingConfig',
	'BeamSearchConfig',
	'TrainConfig',
	'GreedyTrainConfig',
	'IterativeTrainConfig',
	'CurriculumTrainConfig',
	'ContrastiveTrainConfig',
	'MultiTaskTrainConfig',
	'ScheduledSamplingConfig',
	'PatienceConfig',
	'CheckpointConfig',
	# Mask strategies
	'MaskStrategy',
	'AttentionMaskStrategy',
	'CausalMask',
	'BidirectionalMask',
	'SlidingWindowMask',
	'BlockMask',
	'PrefixMask',
	'StridedMask',
	'DilatedMask',
	'LocalGlobalMask',
	'CustomMask',
	'LearnedSparseMask',
	'MaskStrategyFactory',
	'combine_masks',
]
