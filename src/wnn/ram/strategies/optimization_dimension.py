"""OptimizationDimension — the generic axis a WNN architecture search optimizes.

This used to live (as `PhaseType`) inside the IDS-specific `adaptive_cluster`
module, but it is NOT IDS-specific: every WNN family (IDS, controller, LM) runs
phase-aware optimization where each phase touches one dimension. Promoted here
so the shared GA/TS/Lamarckian framework and the step-5 WNN-type factory can
speak one taxonomy.

NATURE OF THE DIMENSIONS
------------------------
Three are *architecture* (they define the address space + wiring); one is
*content*:

    NEURONS / BITS / CONNECTIONS   architecture
    MEMORY                         content (the stored cell values)

That split matters: MEMORY (controller paradigm B — evolving QSR cells) is
coupled to the architecture through the address space, so a MEMORY-phase
mutation operates on cells with the architecture held frozen. This is why the
enum is `OptimizationDimension` and NOT `ArchitectureDimension` — memory is a
thing you optimize, but it is not architecture.

INTEGER VALUES ARE STABLE
-------------------------
NEURONS..CLUSTER (0..3) are persisted in flow configs / the dashboard DB, so
they must never be renumbered. MEMORY is appended as 4. `PhaseType` remains as a
backward-compatible alias for all existing IDS imports.
"""

from __future__ import annotations

from enum import IntEnum


class OptimizationDimension(IntEnum):
	"""Which dimension a phase-aware mutation / crossover touches.

	- NEURONS:     change neuron counts; preserve survivors' bits + connections.
	- BITS:        change bits per neuron (synapse count); no connection drift.
	- CONNECTIONS: perturb connection targets; preserve architecture.
	- CLUSTER:     IDS-only — whole-cluster swap (crossover) / all-architecture
	               mutation. (Generic "all architecture dims" reading elsewhere.)
	- MEMORY:      evolve the stored cell *contents*, architecture frozen
	               (controller paradigm B / GA-Memory). Not architecture.
	"""
	NEURONS = 0
	BITS = 1
	CONNECTIONS = 2
	CLUSTER = 3
	MEMORY = 4


# Backward-compatible alias: IDS code imports this name from adaptive_cluster,
# which now re-exports it from here. New code should prefer OptimizationDimension.
PhaseType = OptimizationDimension


__all__ = ["OptimizationDimension", "PhaseType"]
