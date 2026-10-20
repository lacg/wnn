"""Whole-genome neuron budget (flow param max_neurons_total, worker ABI 21).

`max_neurons` caps each cluster; `max_neurons_total` caps the genome's SUM. The
rule lives in Rust (ram_core neuron_budget.rs): an over-budget genome shrinks
every cluster in proportion, never below min_neurons, dropping neurons from the
end of each cluster. Rust applies it to every GA child / TS neighbour; these
helpers apply the SAME function to the genomes Python builds or already holds
(grid cells, immigrants, seed populations). None = no total cap (original).
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
	from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome


def validate_budget(total: Optional[int], num_clusters: int, min_neurons: int) -> None:
	"""Every cluster must be able to keep min_neurons inside the budget."""
	if total is None:
		return
	if total < num_clusters * max(min_neurons, 1):
		raise ValueError(
			f"max_neurons_total={total} < {num_clusters} clusters x min_neurons {max(min_neurons, 1)}")


def budget_counts(neurons: list[int], total: Optional[int], min_neurons: int) -> list[int]:
	"""Per-cluster counts fitted to the budget (unchanged when within it)."""
	if total is None or sum(neurons) <= total:
		return list(neurons)
	from wnn.accel import require_accel
	return list(require_accel().neuron_budget_counts(list(neurons), int(total), int(min_neurons)))


def fit_genome(genome: 'ClusterGenome', total: Optional[int], min_neurons: int) -> 'ClusterGenome':
	"""The genome fitted to the budget (the SAME object when within it)."""
	if total is None or sum(genome.neurons_per_cluster) <= total:
		return genome
	from wnn.accel import require_accel
	from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
	bits, neurons, conns = require_accel().neuron_budget_fit_genome(
		list(genome.bits_per_neuron), list(genome.neurons_per_cluster),
		list(genome.connections or []), int(total), int(min_neurons))
	return ClusterGenome(bits_per_neuron=list(bits), neurons_per_cluster=list(neurons),
	                     connections=list(conns) if genome.connections else None)
