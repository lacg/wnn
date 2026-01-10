"""
Reporting utilities for RAM model evaluation results.

Provides reusable table formatting for per-tier results display.
"""

from dataclasses import dataclass
from typing import Optional, Callable


@dataclass
class TierResultRow:
	"""A single row of tier results."""
	tier: int
	clusters: int
	neurons: int
	bits: int
	data_pct: float
	ppl: float
	accuracy: float


class TierResultsTable:
	"""
	Formats per-tier evaluation results into a consistent table format.

	Usage:
		# From model stats dict (has 'by_tier' key)
		table = TierResultsTable.from_stats("Validation", stats, overall_ppl, overall_acc)
		for line in table.format():
			print(line)

		# Or with custom logging function
		table.print(log_fn=my_logger)

		# From model directly (queries tier configs)
		table = TierResultsTable.from_model("Test", model, stats)
		table.print()
	"""

	WIDTH = 78

	def __init__(
		self,
		title: str,
		rows: list[TierResultRow],
		overall_ppl: float,
		overall_accuracy: float,
	):
		"""
		Initialize the table.

		Args:
			title: Table title (e.g., "Validation", "Test", "Initial Validation")
			rows: List of TierResultRow objects
			overall_ppl: Overall perplexity across all tiers
			overall_accuracy: Overall accuracy across all tiers
		"""
		self.title = title
		self.rows = rows
		self.overall_ppl = overall_ppl
		self.overall_accuracy = overall_accuracy

	@classmethod
	def from_stats(
		cls,
		title: str,
		stats: dict,
		overall_ppl: Optional[float] = None,
		overall_accuracy: Optional[float] = None,
	) -> "TierResultsTable":
		"""
		Create table from stats dict with 'by_tier' key.

		The stats dict should have the structure returned by RAMLM.evaluate():
		{
			'by_tier': [
				{
					'name': 'tier_0',
					'cluster_count': 100,
					'neurons_per_cluster': 20,
					'bits_per_neuron': 12,
					'data_pct': 46.5,
					'perplexity': 49789.7,
					'accuracy': 0.0001,
				},
				...
			],
			'perplexity': 46676.7,
			'accuracy': 0.0002,
		}

		Args:
			title: Table title
			stats: Stats dict with 'by_tier' list
			overall_ppl: Override overall perplexity (defaults to stats['perplexity'])
			overall_accuracy: Override overall accuracy (defaults to stats['accuracy'])

		Returns:
			TierResultsTable instance
		"""
		if 'by_tier' not in stats:
			raise ValueError("Stats dict must contain 'by_tier' key")

		rows = []
		for tier_stat in stats['by_tier']:
			# Extract tier number from name (e.g., 'tier_0' -> 0)
			tier_num = int(tier_stat['name'].replace('tier_', ''))
			rows.append(TierResultRow(
				tier=tier_num,
				clusters=tier_stat['cluster_count'],
				neurons=tier_stat['neurons_per_cluster'],
				bits=tier_stat['bits_per_neuron'],
				data_pct=tier_stat['data_pct'],
				ppl=tier_stat['perplexity'],
				accuracy=tier_stat['accuracy'],
			))

		return cls(
			title=title,
			rows=rows,
			overall_ppl=overall_ppl if overall_ppl is not None else stats.get('perplexity', 0),
			overall_accuracy=overall_accuracy if overall_accuracy is not None else stats.get('accuracy', 0),
		)

	@classmethod
	def from_model(
		cls,
		title: str,
		model,
		stats: dict,
	) -> "TierResultsTable":
		"""
		Create table from model and stats, querying model for tier configs.

		This is useful when stats doesn't have full tier config info.

		Args:
			title: Table title
			model: RAMLM model instance (must have layer.tier_configs)
			stats: Stats dict with 'by_tier' list (may have partial info)

		Returns:
			TierResultsTable instance
		"""
		if 'by_tier' not in stats:
			raise ValueError("Stats dict must contain 'by_tier' key")

		# Get tier configs from model
		tier_configs = model.layer.tier_configs

		rows = []
		for i, (tier_stat, tc) in enumerate(zip(stats['by_tier'], tier_configs)):
			rows.append(TierResultRow(
				tier=i,
				clusters=tc.cluster_count,
				neurons=tc.neurons_per_cluster,
				bits=tc.bits_per_neuron,
				data_pct=tier_stat.get('data_pct', 0),
				ppl=tier_stat.get('perplexity', 0),
				accuracy=tier_stat.get('accuracy', 0),
			))

		return cls(
			title=title,
			rows=rows,
			overall_ppl=stats.get('perplexity', 0),
			overall_accuracy=stats.get('accuracy', 0),
		)

	def format(self) -> list[str]:
		"""
		Format the table as a list of strings.

		Returns:
			List of formatted lines (without newlines)
		"""
		lines = []

		# Title and header
		lines.append(f"Per-Tier {self.title} Results:")
		lines.append("=" * self.WIDTH)
		lines.append(
			f"{'Tier':<8} {'Clusters':>10} {'Neurons':>8} {'Bits':>6} "
			f"{'Data %':>8} {'PPL':>12} {'Accuracy':>10}"
		)
		lines.append("-" * self.WIDTH)

		# Data rows
		total_clusters = 0
		for row in self.rows:
			total_clusters += row.clusters
			lines.append(
				f"{'Tier ' + str(row.tier):<8} {row.clusters:>10,} {row.neurons:>8} {row.bits:>6} "
				f"{row.data_pct:>7.1f}% {row.ppl:>12.1f} {row.accuracy:>9.2%}"
			)

		# Total row
		lines.append("-" * self.WIDTH)
		lines.append(
			f"{'TOTAL':<8} {total_clusters:>10,} {'':>8} {'':>6} "
			f"{'100.0':>7}% {self.overall_ppl:>12.1f} {self.overall_accuracy:>9.2%}"
		)

		return lines

	def print(self, log_fn: Optional[Callable[[str], None]] = None) -> None:
		"""
		Print the formatted table.

		Args:
			log_fn: Optional logging function (defaults to print)
		"""
		output = log_fn or print
		for line in self.format():
			output(line)

	def __str__(self) -> str:
		"""Return table as a single string with newlines."""
		return "\n".join(self.format())
