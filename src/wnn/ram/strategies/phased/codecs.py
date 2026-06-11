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


class PickleBase64Codec:
	"""Fallback codec for genome types without JSON (de)serialization.

	Used by the controller strand (RecurrentArchGenome has serialize() but no
	deserialize yet — TODO: add it and switch to a native codec). The pickled
	payload is base64-wrapped so the ENVELOPE stays a valid schema-2 json.gz;
	only the genome blobs are opaque.
	"""
	name = "pickle_b64"

	def encode(self, genome: Any) -> Any:
		return base64.b64encode(pickle.dumps(genome)).decode("ascii")

	def decode(self, data: Any) -> Any:
		return pickle.loads(base64.b64decode(data.encode("ascii")))
