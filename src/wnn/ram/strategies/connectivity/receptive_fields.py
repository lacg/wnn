"""
ReceptiveFields — the per-neuron feature-family connectivity constraint
(flow param `ids_receptive_fields="family"`, worker ABI 27;
docs/ids_receptive_fields_design.md).

Domain-blind: a family index per ENCODED input bit + the cross budget. The IDS
family maps that produce `bit_family` live in wnn/ids/feature_families.py.

The projection itself is Rust (ram_core receptive_fields.rs): Rust applies it
to every GA child / TS neighbour and after every genesis pass; Python calls
the SAME function here for the genomes it builds or edits itself (grid
genomes, GA initial population + immigrants, the Python mutate/crossover
fallbacks, adaptation padding). Off (`None` on the configs) never reaches this
class, so it draws nothing from any RNG.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from wnn.ram.genome import ClusterGenome


@dataclass(frozen=True)
class ReceptiveFields:
	bit_family: tuple[int, ...]
	family_names: tuple[str, ...]
	cross: float
	version: str
	source: str

	def __post_init__(self) -> None:
		if not 0.0 <= self.cross < 0.5:
			raise ValueError(f"ids_rf_cross={self.cross}: must be in [0, 0.5) (a neuron's family holds the majority)")
		if not self.bit_family or max(self.bit_family) >= len(self.family_names):
			raise ValueError("receptive fields: bit_family indexes a family that has no name")

	@property
	def total_input_bits(self) -> int:
		return len(self.bit_family)

	def rust_args(self) -> dict:
		"""The keyword args every IDSCacheWrapper search / adaptive call takes."""
		return {"rf_bit_family": list(self.bit_family), "rf_cross": self.cross}

	def project(self, bits_per_neuron: list[int], connections: list[int], seed: int) -> list[int]:
		"""Projected connections (Rust); deterministic in `seed`."""
		from wnn.accel import require_accel
		conns, _ = require_accel().rf_project_connections(
			list(bits_per_neuron), list(connections), list(self.bit_family), self.cross, seed)
		return conns

	def project_genome(self, genome: "ClusterGenome", rng: random.Random) -> "ClusterGenome":
		"""Project `genome` in place (one seed drawn from `rng`); returns it. A
		changed genome drops its cached metrics (they scored the old wiring)."""
		if not genome.connections:
			return genome
		projected = self.project(genome.bits_per_neuron, genome.connections, rng.randint(0, 2**63))
		if projected != list(genome.connections):
			genome.connections = projected
			if getattr(genome, "metrics", None) is not None:
				genome.metrics = None
		return genome

	def neuron_fields(self, bits_per_neuron: list[int], connections: list[int]) -> list[tuple[int, int, int, bool]]:
		"""Per neuron (held family or -1, in-family connections, budget, exempt)."""
		from wnn.accel import require_accel
		return require_accel().rf_neuron_fields(list(bits_per_neuron), list(connections), list(self.bit_family), self.cross)

	def family_sizes(self) -> list[int]:
		return [self.bit_family.count(f) for f in range(len(self.family_names))]

	def provenance(self) -> str:
		"""One log line: map version + source, budget, family sizes in encoded bits."""
		sizes = ", ".join(f"{n}={s}" for n, s in zip(self.family_names, self.family_sizes()))
		return (f"  [RF] ids_receptive_fields=family map={self.version}/{self.source} "
		        f"ids_rf_cross={self.cross} (budget floor({self.cross} x bits) out-of-family connections "
		        f"per neuron) over {self.total_input_bits} encoded bits: {sizes}")


def project_if_on(rf: ReceptiveFields | None, genome: "ClusterGenome", rng: random.Random) -> "ClusterGenome":
	"""The ONE Python enforcement call: project `genome` when receptive fields are
	on; when off (None) return it untouched WITHOUT drawing from `rng` — every
	default-path RNG stream stays bit-identical."""
	return genome if rf is None else rf.project_genome(genome, rng)
