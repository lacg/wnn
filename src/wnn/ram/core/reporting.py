"""
Reporting utilities for RAM model evaluation results.

Provides reusable table formatting for:
- Per-tier results display (TierResultsTable)
- Optimization progress comparison (OptimizationResultsTable)
"""

from dataclasses import dataclass, field
from typing import Optional, Callable
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
		self.stages: list[OptimizationStage] = []

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
		phases: list[dict],
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

	def format(self) -> list[str]:
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


# =============================================================================
# Optimization Analysis Table (GA/TS detailed metrics)
# =============================================================================

@dataclass
class OptimizationAnalysis:
	"""
	Detailed analysis metrics for GA or TS optimization.

	Tracks:
	- Elite vs generated performance
	- Diversity metrics
	- Improvement patterns
	"""
	method: str  # "GA" or "TS"

	# Population/seed metrics
	initial_best_ce: float
	final_best_ce: float
	initial_best_acc: Optional[float] = None
	final_best_acc: Optional[float] = None

	# Diversity metrics
	initial_ce_spread: float = 0.0  # max - min CE in initial pool
	final_ce_spread: float = 0.0    # max - min CE in final pool

	# Elite tracking (GA: elites that survived, TS: seed/elite wins)
	elite_survivals: int = 0        # How many elites made it to final
	total_elites: int = 0           # How many elites were selected
	elite_wins: int = 0             # Iterations where elite beat new offspring/neighbors
	total_iterations: int = 0

	# Improvement tracking
	improved_iterations: int = 0    # Iterations with improvement
	stagnant_iterations: int = 0    # Iterations without improvement

	# CE vs Accuracy elite overlap (GA only)
	ce_elite_count: int = 0
	acc_elite_count: int = 0
	overlap_count: int = 0          # Elites that are top in both CE and accuracy

	@property
	def improvement_pct(self) -> float:
		"""Percentage improvement from initial to final."""
		if self.initial_best_ce == 0:
			return 0.0
		return (self.initial_best_ce - self.final_best_ce) / self.initial_best_ce * 100

	@property
	def elite_win_rate(self) -> float:
		"""Percentage of iterations where elite beat new candidates."""
		if self.total_iterations == 0:
			return 0.0
		return self.elite_wins / self.total_iterations * 100

	@property
	def improvement_rate(self) -> float:
		"""Percentage of iterations with improvement."""
		if self.total_iterations == 0:
			return 0.0
		return self.improved_iterations / self.total_iterations * 100

	@property
	def diversity_change(self) -> float:
		"""Change in diversity (negative = diversity collapsed)."""
		return self.final_ce_spread - self.initial_ce_spread


class OptimizationAnalysisTable:
	"""
	Formats detailed optimization analysis into a table.

	Usage:
		analysis = OptimizationAnalysis(
			method="GA",
			initial_best_ce=10.50,
			final_best_ce=10.45,
			...
		)
		table = OptimizationAnalysisTable(analysis)
		table.print(logger)
	"""

	WIDTH = 78

	def __init__(self, analysis: OptimizationAnalysis):
		self.analysis = analysis

	def format(self) -> list[str]:
		"""Format the analysis as a list of strings."""
		a = self.analysis
		lines = []

		lines.append(f"{a.method} Optimization Analysis:")
		lines.append("-" * self.WIDTH)

		# Performance metrics
		lines.append(f"  {'Initial best CE:':<25} {a.initial_best_ce:>10.4f}  →  Final: {a.final_best_ce:>10.4f}  ({a.improvement_pct:>+.2f}%)")
		if a.initial_best_acc is not None and a.final_best_acc is not None:
			lines.append(f"  {'Initial best Acc:':<25} {a.initial_best_acc:>9.2%}  →  Final: {a.final_best_acc:>9.2%}")

		# Diversity metrics
		lines.append(f"  {'CE spread (diversity):':<25} {a.initial_ce_spread:>10.4f}  →  Final: {a.final_ce_spread:>10.4f}  ({a.diversity_change:>+.4f})")

		# Elite tracking
		if a.method == "GA":
			lines.append(f"  {'Elite survival:':<25} {a.elite_survivals:>3}/{a.total_elites:<3} elites survived to final population")
			lines.append(f"  {'Elite composition:':<25} {a.ce_elite_count} by CE + {a.acc_elite_count} by Acc ({a.overlap_count} overlap) = {a.total_elites} unique")

		lines.append(f"  {'Elite win rate:':<25} {a.elite_wins:>3}/{a.total_iterations:<3} iterations ({a.elite_win_rate:.1f}%)")
		lines.append(f"  {'Improvement rate:':<25} {a.improved_iterations:>3}/{a.total_iterations:<3} iterations ({a.improvement_rate:.1f}%)")

		lines.append("-" * self.WIDTH)

		return lines

	def print(self, log_fn: Optional[Callable[[str], None]] = None) -> None:
		"""Print the formatted table."""
		output = log_fn or print
		for line in self.format():
			output(line)

	def __str__(self) -> str:
		return "\n".join(self.format())
