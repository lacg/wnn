"""
WNN model configuration for HuggingFace integration.

WNNConfig stores the architecture specification of a WNN RAM language model:
- Vocabulary and context parameters
- Per-cluster neuron counts and bit widths
- Memory mode and sampling configuration
- Optional tier configuration for tiered architectures

This is a standard HuggingFace PretrainedConfig subclass, so it supports
save_pretrained/from_pretrained and auto-registration.
"""

from __future__ import annotations

from transformers import PretrainedConfig


class WNNConfig(PretrainedConfig):
	"""Configuration for a Weightless Neural Network language model.

	A WNN model is defined by its cluster architecture:
	- Each output token cluster has a specific number of neurons
	- Each neuron observes a specific number of input bits
	- During inference, neurons look up addresses in memory tables

	The architecture is discovered via GA/TS optimization, not hand-designed.
	"""

	model_type = "wnn-ram"

	def __init__(
		self,
		vocab_size: int = 50257,
		context_size: int = 4,
		architecture_type: str = "bitwise",
		bits_per_cluster: list[int] | None = None,
		neurons_per_cluster: list[int] | None = None,
		memory_mode: str = "QUAD_WEIGHTED",
		neuron_sample_rate: float = 0.25,
		tier_config: str | None = None,
		num_clusters: int | None = None,
		encoding_type: str = "binary",
		sparse_threshold: int | None = None,
		**kwargs,
	):
		"""
		Args:
			vocab_size: Token vocabulary size (default: GPT-2 50257)
			context_size: Number of context tokens for prediction
			architecture_type: "bitwise" (per-cluster heterogeneous) or
				"tiered" (grouped by frequency)
			bits_per_cluster: Bits per neuron for each cluster [num_clusters].
				For bitwise: one entry per cluster.
				For tiered: derived from tier_config.
			neurons_per_cluster: Neurons per cluster [num_clusters].
			memory_mode: RAM memory mode — "TERNARY", "QUAD_BINARY",
				or "QUAD_WEIGHTED"
			neuron_sample_rate: Fraction of neurons to sample during
				inference (1.0 = use all)
			tier_config: Tiered architecture string
				(e.g., "100,15,20;400,10,12;rest,5,8")
			num_clusters: Total number of output clusters
				(inferred from bits_per_cluster if not set)
			encoding_type: Token bit encoding — "binary" or "gray_code"
			sparse_threshold: Bits threshold for sparse memory storage
				(None = auto)
		"""
		self.vocab_size = vocab_size
		self.context_size = context_size
		self.architecture_type = architecture_type
		self.bits_per_cluster = bits_per_cluster or []
		self.neurons_per_cluster = neurons_per_cluster or []
		self.memory_mode = memory_mode
		self.neuron_sample_rate = neuron_sample_rate
		self.tier_config = tier_config
		self.num_clusters = num_clusters or len(self.bits_per_cluster)
		self.encoding_type = encoding_type
		self.sparse_threshold = sparse_threshold
		super().__init__(**kwargs)

	@classmethod
	def from_genome(
		cls,
		genome,
		vocab_size: int = 50257,
		context_size: int = 4,
		memory_mode: str = "QUAD_WEIGHTED",
		neuron_sample_rate: float = 0.25,
		**kwargs,
	) -> WNNConfig:
		"""Create a config from a ClusterGenome object.

		Args:
			genome: A ClusterGenome with bits_per_cluster and
				neurons_per_cluster attributes
			vocab_size: Token vocabulary size
			context_size: Context window size
			memory_mode: RAM memory mode
			neuron_sample_rate: Neuron sampling rate
		"""
		return cls(
			vocab_size=vocab_size,
			context_size=context_size,
			architecture_type="bitwise",
			bits_per_cluster=list(genome.bits_per_cluster),
			neurons_per_cluster=list(genome.neurons_per_cluster),
			memory_mode=memory_mode,
			neuron_sample_rate=neuron_sample_rate,
			num_clusters=len(genome.bits_per_cluster),
			**kwargs,
		)

	@classmethod
	def from_tier_config(
		cls,
		tier_config: str,
		vocab_size: int = 50257,
		context_size: int = 4,
		**kwargs,
	) -> WNNConfig:
		"""Create a config from a tier configuration string.

		Args:
			tier_config: e.g., "100,15,20;400,10,12;rest,5,8"
			vocab_size: Token vocabulary size
			context_size: Context window size
		"""
		bits = []
		neurons = []
		remaining = vocab_size

		for tier_str in tier_config.split(";"):
			parts = tier_str.strip().split(",")
			if len(parts) != 3:
				continue
			clusters_str, n_str, b_str = parts
			n, b = int(n_str), int(b_str)

			if clusters_str.strip().lower() == "rest":
				count = remaining
			else:
				count = int(clusters_str.strip())
			remaining -= count

			bits.extend([b] * count)
			neurons.extend([n] * count)

		return cls(
			vocab_size=vocab_size,
			context_size=context_size,
			architecture_type="tiered",
			bits_per_cluster=bits,
			neurons_per_cluster=neurons,
			tier_config=tier_config,
			num_clusters=len(bits),
			**kwargs,
		)

	@property
	def total_neurons(self) -> int:
		"""Total neurons across all clusters."""
		return sum(self.neurons_per_cluster)

	@property
	def total_memory_words(self) -> int:
		"""Total memory words (2^bits * neurons) across all clusters."""
		return sum(
			n * (2 ** b)
			for n, b in zip(self.neurons_per_cluster, self.bits_per_cluster)
		)
