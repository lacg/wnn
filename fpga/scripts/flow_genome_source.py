"""FlowGenomeSource: a flow's genome, params and RECORDED validation row, read-only.

Read from the dashboard DB through a read-only URI (the exporter never writes the
DB). The recorded `threshold_metadata` is what the fidelity gate must reproduce.
Phase matching is by `experiments.phase_type` (works for PUB50 "Grid Search
(neurons x bits)" and single-genome flows alike); default prefers ga_neurons.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

PHASES = ("ga_neurons", "grid_search")

_QUERY = """
	SELECT vs.genome_hash, e.name, g.tiers_json, g.connections_json, vs.threshold_metadata
	FROM validation_summaries vs
	JOIN experiments e ON vs.experiment_id = e.id AND e.phase_type = ?
	JOIN genomes g ON g.experiment_id = e.id AND g.config_hash = vs.genome_hash
	WHERE e.flow_id = ? AND vs.validation_point = ? AND vs.genome_type = ?
	ORDER BY vs.id DESC LIMIT 1
"""


@dataclass(frozen=True)
class FlowGenomeSource:
	flow_id: int
	params: dict
	genome_hash: str
	phase_name: str
	bits_per_neuron: list[int]
	neurons_per_cluster: list[int]
	connections: list[int]
	threshold_metadata: dict

	@classmethod
	def read(cls, db_path: str, flow_id: int, genome_type: str, phase: str | None, point: str = "final") -> FlowGenomeSource:
		con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
		try:
			params = cls._params(con, flow_id)
			row = cls._row(con, flow_id, genome_type, phase, point)
		finally:
			con.close()
		genome_hash, phase_name, tiers_json, connections_json, metadata_json = row
		tiers = json.loads(tiers_json)
		return cls(
			flow_id=flow_id, params=params, genome_hash=genome_hash, phase_name=phase_name,
			bits_per_neuron=list(tiers["bits_per_neuron"]), neurons_per_cluster=list(tiers["neurons_per_cluster"]),
			connections=cls._connections(connections_json), threshold_metadata=json.loads(metadata_json or "{}"),
		)

	@staticmethod
	def _params(con, flow_id: int) -> dict:
		row = con.execute("SELECT config_json FROM flows WHERE id = ?", (flow_id,)).fetchone()
		if row is None:
			raise ValueError(f"Flow {flow_id} not found")
		return json.loads(row[0])["params"]

	@staticmethod
	def _row(con, flow_id: int, genome_type: str, phase: str | None, point: str) -> tuple:
		if phase is not None and phase not in PHASES:
			raise ValueError(f"Unknown phase: {phase}. Use one of {PHASES}.")
		for phase_type in [phase] if phase else PHASES:
			row = con.execute(_QUERY, (phase_type, flow_id, point, genome_type)).fetchone()
			if row:
				return row
		raise ValueError(f"No {point} {genome_type} genome for flow {flow_id} in {phase or 'any phase'}")

	@staticmethod
	def _connections(raw: str) -> list[int]:
		raw = raw.strip()
		return [int(c) for c in (json.loads(raw) if raw.startswith("[") else raw.split(","))]

	def genome(self):
		from wnn.ram.genome import ClusterGenome
		return ClusterGenome(bits_per_neuron=self.bits_per_neuron, neurons_per_cluster=self.neurons_per_cluster, connections=self.connections)
