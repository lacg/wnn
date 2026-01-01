"""
Trainer-related enumerations.
"""

from enum import IntEnum


class LayerType(IntEnum):
	"""Types of layers in the RAM Transformer."""
	EMBEDDING = 0
	INPUT_PROJ = 1
	ATTENTION = 2
	FFN = 3
	OUTPUT_PROJ = 4
	TOKEN_MAPPER = 5


class TrainingMode(IntEnum):
	"""
	Training strategies for EDRA.

	Different modes trade off between training speed and accuracy:
	- GREEDY: Train all layers in one pass (fast, may miss dependencies)
	- ITERATIVE: Multiple passes until stable (slower, better accuracy)
	- LAYERWISE: Train one layer at a time, freeze others (most controlled)
	- OUTPUT_FIRST: Train output layers first, then propagate backward
	"""
	GREEDY = 0        # Train all layers in single backward pass
	ITERATIVE = 1     # Multiple passes until convergence
	LAYERWISE = 2     # Train one layer at a time
	OUTPUT_FIRST = 3  # Prioritize output layers


class TrainingPhase(IntEnum):
	"""
	Training phases for curriculum learning.

	Supports progressive training from simple to complex:
	- WARMUP: Train on easiest examples
	- MAIN: Train on full dataset
	- REFINEMENT: Focus on hard examples
	"""
	WARMUP = 0
	MAIN = 1
	REFINEMENT = 2


class MixingStrategy(IntEnum):
	"""How to mix examples from multiple tasks."""
	ROUND_ROBIN = 0   # Alternate between tasks
	PROPORTIONAL = 1  # Sample proportional to dataset size
	WEIGHTED = 2      # Sample proportional to task weight
	INTERLEAVED = 3   # Mix all examples, shuffle
