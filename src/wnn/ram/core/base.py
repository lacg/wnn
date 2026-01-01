"""
RAM Base Classes

Provides common interfaces for RAM neural network components.

Two main families:
1. RAMComponent - Low-level Tensor→Tensor operations (Memory, Layer, Mappers)
2. RAMSequenceModel - Sequence-to-sequence models (Transformers, Attention)

Usage:
    class MyMapper(RAMComponent):
        def forward(self, bits: Tensor) -> Tensor:
            ...

    class MyTransformer(RAMSequenceModel):
        def forward(self, tokens: list[Tensor]) -> list[Tensor]:
            ...
"""

from abc import ABC, abstractmethod
from torch import Tensor
from torch.nn import Module


class RAMComponent(Module, ABC):
	"""
	Base class for low-level RAM components.

	These components process single Tensors:
	- Memory: bit storage and lookup
	- RAMLayer: neural layer wrapper
	- Mappers: generalization strategies (BitLevel, Compositional, etc.)
	- FFN: feedforward networks

	Subclasses must implement:
	- forward(bits: Tensor) -> Tensor
	"""

	@abstractmethod
	def forward(self, bits: Tensor) -> Tensor:
		"""
		Transform input bits to output bits.

		Args:
		    bits: Input tensor of shape [n_bits] or [batch, n_bits]

		Returns:
		    Output tensor of same shape as input (or defined output size)
		"""
		...


class RAMSequenceModel(Module, ABC):
	"""
	Base class for sequence-to-sequence RAM models.

	These models process sequences of tokens:
	- Attention mechanisms (SoftRAM, Position-only, etc.)
	- Transformer blocks and stacks
	- Encoder-decoder architectures

	Subclasses must implement:
	- forward(tokens: list[Tensor]) -> list[Tensor]

	Optional training interface:
	- train(windows: list[Tensor], targets) -> None
	"""

	@abstractmethod
	def forward(self, tokens: list[Tensor]) -> list[Tensor]:
		"""
		Process a sequence of tokens.

		Args:
		    tokens: List of token tensors, each of shape [n_bits]

		Returns:
		    List of output tokens, same length as input
		"""
		...

	def train_step(
		self,
		windows: list[Tensor],
		targets: Tensor | list[Tensor],
	) -> dict:
		"""
		Single training step using EDRA.

		Args:
		    windows: Input sequence as list of tensors
		    targets: Target output (Tensor or list of Tensors)

		Returns:
		    dict with training statistics (errors, updates, etc.)

		Note:
		    Default implementation raises NotImplementedError.
		    Subclasses should override if they support training.
		"""
		raise NotImplementedError(
			f"{self.__class__.__name__} does not implement train_step"
		)


class RAMTrainable(ABC):
	"""
	Mixin for models that support EDRA training.

	Use this with RAMSequenceModel for trainable sequence models:

	    class MyTrainableModel(RAMSequenceModel, RAMTrainable):
	        def forward(self, tokens): ...
	        def train_step(self, windows, targets): ...
	"""

	@abstractmethod
	def train_step(
		self,
		windows: list[Tensor],
		targets: Tensor | list[Tensor],
	) -> dict:
		"""Perform one EDRA training step."""
		...


__all__ = [
	'RAMComponent',
	'RAMSequenceModel',
	'RAMTrainable',
]
