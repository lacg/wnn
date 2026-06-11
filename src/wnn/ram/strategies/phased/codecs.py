"""Genome codecs for the unified phased-search checkpoint store.

A codec turns a genome into a JSON-able payload and back. Each strand
registers the codec for its genome type; the checkpoint envelope itself is
strand-agnostic.
"""

import base64
import pickle
from typing import Any, Protocol


class GenomeCodec(Protocol):
	"""Encode/decode one genome to/from a JSON-able value."""
	name: str

	def encode(self, genome: Any) -> Any: ...
	def decode(self, data: Any) -> Any: ...


class ClusterGenomeCodec:
	"""ClusterGenome (IDS/LM architecture search) — native JSON serialize."""
	name = "cluster_genome"

	def encode(self, genome: Any) -> Any:
		return genome.serialize()

	def decode(self, data: Any) -> Any:
		from wnn.ram.genome import ClusterGenome
		return ClusterGenome.deserialize(data)


class ControllerGenomeCodec:
	"""RecurrentArchGenome (drone controller) — native serialize/deserialize.

	No pickle anywhere: controller checkpoints are plain data, refactor-proof
	and shell-inspectable like every other strand.
	"""
	name = "controller_genome"

	def encode(self, genome: Any) -> Any:
		return genome.serialize()

	def decode(self, data: Any) -> Any:
		from wnn.control.recurrent_genome import RecurrentArchGenome
		return RecurrentArchGenome.deserialize(data)


class PickleBase64Codec:
	"""LAST-RESORT codec for genome types without native (de)serialization.

	Nothing in the codebase uses it for writing anymore (11/06/2026) — kept
	only so tests can exercise the codec protocol with arbitrary objects.
	"""
	name = "pickle_b64"

	def encode(self, genome: Any) -> Any:
		return base64.b64encode(pickle.dumps(genome)).decode("ascii")

	def decode(self, data: Any) -> Any:
		return pickle.loads(base64.b64decode(data.encode("ascii")))
