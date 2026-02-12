"""
Training Strategies

Pluggable training strategies for RAM networks.

Usage:
	from wnn.ram.strategies.train import GreedyTrainStrategy, CurriculumTrainStrategy

	model = RAMSeq2Seq(...)
	strategy = CurriculumTrainStrategy(CurriculumTrainConfig(num_stages=5))
	history = strategy.train(model, dataset)
"""

from wnn.ram.strategies.train.greedy import GreedyTrainStrategy
from wnn.ram.strategies.train.iterative import IterativeTrainStrategy
from wnn.ram.strategies.train.curriculum import CurriculumTrainStrategy
from wnn.ram.strategies.train.contrastive import ContrastiveTrainStrategy

__all__ = [
	'GreedyTrainStrategy',
	'IterativeTrainStrategy',
	'CurriculumTrainStrategy',
	'ContrastiveTrainStrategy',
]
