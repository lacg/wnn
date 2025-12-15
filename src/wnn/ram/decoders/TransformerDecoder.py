from abc import ABC, abstractmethod
from torch import Tensor

class TransformerDecoder(ABC):

	@abstractmethod
	def encode(self, target: Tensor) -> Tensor:
		"""
		Encode user-level target into output-layer bit targets.
		"""
		pass

	@abstractmethod
	def decode(self, output_bits: Tensor) -> Tensor:
		"""
		Decode output-layer bits into user-level prediction.
		"""
		pass