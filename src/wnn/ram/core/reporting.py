"""
Reporting utilities for RAM model evaluation results.

Provides reusable table formatting for:
- Per-tier results display (TierResultsTable)
- Optimization progress comparison (OptimizationResultsTable)
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, List
import math


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


# =============================================================================
# Optimization Results Table
# =============================================================================

@dataclass
class OptimizationStage:
	"""A single stage/phase in optimization progress."""
	name: str
	ce: float
	accuracy: Optional[float] = None
	note: Optional[str] = None  # e.g., "(baseline)", "(variance)"

	@property
	def ppl(self) -> float:
		"""Perplexity = exp(cross_entropy)."""
		return math.exp(self.ce)


class OptimizationResultsTable:
	"""
	Formats optimization progress results into a comparison table.

	Works for both:
	- Phase-based optimization (Initial → Phase 1a → 1b → 2)
	- Tier-based comparison (per-tier metrics before/after)

	Usage:
		# Phase-based comparison
		table = OptimizationResultsTable("Validation")
		table.add_stage("Initial", ce=10.55, accuracy=0.05)
		table.add_stage("After GA", ce=10.50, accuracy=0.055)
		table.add_stage("After TS", ce=10.48, accuracy=0.056)
		table.print(logger)

		# With notes
		table.add_stage("Phase 2 mean", ce=10.52, note="(variance)")
	"""

	WIDTH = 78

	def __init__(self, title: str = "Optimization Results"):
		"""
		Initialize the table.

		Args:
			title: Table title (e.g., "Validation", "Test", "Architecture Search")
		"""
		self.title = title
		self.stages: List[OptimizationStage] = []

	def add_stage(
		self,
		name: str,
		ce: float,
		accuracy: Optional[float] = None,
		note: Optional[str] = None,
	) -> "OptimizationResultsTable":
		"""
		Add a stage to the table.

		Args:
			name: Stage name (e.g., "Initial", "After Phase 1a")
			ce: Cross-entropy at this stage
			accuracy: Optional accuracy at this stage
			note: Optional note (e.g., "(baseline)", "(variance)")

		Returns:
			Self for method chaining
		"""
		self.stages.append(OptimizationStage(
			name=name,
			ce=ce,
			accuracy=accuracy,
			note=note,
		))
		return self

	@classmethod
	def from_phases(
		cls,
		title: str,
		phases: List[dict],
	) -> "OptimizationResultsTable":
		"""
		Create table from a list of phase dicts.

		Each dict should have: name, ce, and optionally accuracy, note.

		Args:
			title: Table title
			phases: List of phase dicts

		Returns:
			OptimizationResultsTable instance
		"""
		table = cls(title)
		for phase in phases:
			table.add_stage(
				name=phase['name'],
				ce=phase['ce'],
				accuracy=phase.get('accuracy'),
				note=phase.get('note'),
			)
		return table

	def format(self) -> List[str]:
		"""
		Format the table as a list of strings.

		Returns:
			List of formatted lines (without newlines)
		"""
		lines = []

		# Check if we have accuracy data
		has_accuracy = any(s.accuracy is not None for s in self.stages)

		# Title and header
		lines.append(f"{self.title} Results:")
		lines.append("=" * self.WIDTH)

		if has_accuracy:
			lines.append(
				f"{'Stage':<30} {'CE':>10} {'PPL':>12} {'Accuracy':>10} {'Improvement':>12}"
			)
		else:
			lines.append(
				f"{'Stage':<30} {'CE':>10} {'PPL':>12} {'Improvement':>12}"
			)
		lines.append("-" * self.WIDTH)

		# Data rows
		initial_ce = self.stages[0].ce if self.stages else None

		for i, stage in enumerate(self.stages):
			# Calculate improvement vs initial
			if i == 0 or initial_ce is None:
				improvement_str = "-"
			else:
				improvement_pct = (1 - stage.ce / initial_ce) * 100
				improvement_str = f"{improvement_pct:>+.2f}%"

			# Stage name with optional note
			name = stage.name
			if stage.note:
				name = f"{name} {stage.note}"

			# Format row
			if has_accuracy:
				acc_str = f"{stage.accuracy:>9.2%}" if stage.accuracy is not None else f"{'N/A':>10}"
				lines.append(
					f"{name:<30} {stage.ce:>10.4f} {stage.ppl:>12.1f} {acc_str} {improvement_str:>12}"
				)
			else:
				lines.append(
					f"{name:<30} {stage.ce:>10.4f} {stage.ppl:>12.1f} {improvement_str:>12}"
				)

		# Total row
		lines.append("-" * self.WIDTH)
		if self.stages and initial_ce is not None:
			final = self.stages[-1]
			total_improvement = (1 - final.ce / initial_ce) * 100
			if has_accuracy:
				lines.append(
					f"{'Total Improvement':<30} {'':>10} {'':>12} {'':>10} {total_improvement:>+11.2f}%"
				)
			else:
				lines.append(
					f"{'Total Improvement':<30} {'':>10} {'':>12} {total_improvement:>+11.2f}%"
				)
		lines.append("=" * self.WIDTH)

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
