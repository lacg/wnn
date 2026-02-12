from abc import ABC, abstractmethod

from torch import arange
from torch import bool as tbool
from torch import device
from torch import int64
from torch import tensor
from torch import Tensor

class TransformerDecoder(ABC):

	@abstractmethod
	def decode(self, output_bits: Tensor) -> Tensor:
		"""
		Decode output-layer bits into user-level prediction.
		"""
		pass

	@abstractmethod
	def encode(self, target: Tensor) -> Tensor:
		"""
		Encode user-level target into output-layer bit targets.
		"""
		pass

