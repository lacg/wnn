"""Reconstruct a GA population checkpoint from the dashboard DB.

Motivation
----------
When a worker is killed mid-GA, the in-RAM population is lost. Going forward,
`experiment.py` now configures per-generation population checkpoints so resume
is automatic. But runs that crashed *before* that fix have no on-disk
checkpoint — only the per-generation summaries persisted in the DB
(`iterations` + `genome_evaluations` + `genomes`).

This script rebuilds a `CheckpointManager`-compatible checkpoint file from those
DB rows so the worker's existing resume path (`has_checkpoint()` →
`optimize(initial_population=...)`) can pick the run back up from its last
completed generation.

Fidelity note: for neuron/bit GAs (`ga_neurons`, `ga_bits`) connectivity is NOT
an optimized dimension and is not persisted in the DB — it is regenerated from
the genome's (neurons, bits) config on resume (`initialize_connections`). The
genome's *identity* (the thing the GA optimizes) is fully captured by its tier
config, which IS in the DB. So reconstruction is faithful for those phases. For
a connection-optimizing GA (`optimize_connections=True`) the exact connectivity
cannot be recovered from the DB and resume would re-sample it — pass
--allow-connection-loss to proceed anyway.

Usage
-----
    python scripts/db_to_ga_checkpoint.py --flow-id 4042            # auto-pick GA experiment
    python scripts/db_to_ga_checkpoint.py --experiment-id 9301
    python scripts/db_to_ga_checkpoint.py --flow-id 4042 --dry-run  # report only, write nothing
    python scripts/db_to_ga_checkpoint.py --experiment-id 9301 --checkpoint-dir /path/exp_01
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Optional

# --- repo imports -----------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "wnn"))

from wnn.ram.strategies.connectivity.architecture_strategies import (
	CheckpointConfig,
	CheckpointManager,
)
from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome

DEFAULT_DB_CANDIDATES = [
	"/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db",
	"/Users/lacg/wnn/db/wnn.db",
]
CHECKPOINT_ROOTS = [
	Path("/Volumes/20260401-WDBlack-SN850X-2TB/wnn/checkpoints"),
	Path("/Users/lacg/wnn/checkpoints"),
]


def _slugify(name: str) -> str:
	"""Match the worker's checkpoint-dir slug convention (worker.py:539)."""
	return name.lower().replace(" ", "-").replace("_", "-")


def resolve_db(arg: Optional[str]) -> str:
	if arg:
		return arg
	for c in DEFAULT_DB_CANDIDATES:
		if Path(c).exists():
			return c
	raise SystemExit("Could not locate dashboard DB; pass --db")


def pick_ga_experiment(con: sqlite3.Connection, flow_id: int) -> int:
	"""Pick the GA experiment for a flow: prefer a non-completed GA phase, else
	the highest-sequence GA. Errors if none found."""
	rows = con.execute(
		"SELECT id, sequence_order, status, phase_type FROM experiments "
		"WHERE flow_id=? AND phase_type LIKE 'ga%' ORDER BY sequence_order DESC",
		(flow_id,),
	).fetchall()
	if not rows:
		raise SystemExit(f"Flow {flow_id} has no GA experiment")
	for r in rows:
		if r[2] != "completed":
			return r[0]
	return rows[0][0]


def tiers_to_arch(tiers_json: str) -> tuple[list[int], list[int]]:
	"""Expand a tiers_json descriptor into (bits_per_neuron, neurons_per_cluster)."""
	tiers = json.loads(tiers_json)
	neurons_per_cluster: list[int] = []
	bits_per_neuron: list[int] = []
	for t in tiers:
		clusters = int(t.get("clusters", 1))
		neurons = int(t["neurons"])
		bits = int(t["bits"])
		for _ in range(clusters):
			neurons_per_cluster.append(neurons)
			bits_per_neuron.extend([bits] * neurons)
	return bits_per_neuron, neurons_per_cluster


def derive_checkpoint_dir(con: sqlite3.Connection, flow_id: int, ga_seq: int) -> Path:
	"""Find the flow's checkpoint dir (sibling of the existing exp_00 grid dir)
	and return the GA experiment's exp_<seq> subdir."""
	name = con.execute("SELECT name FROM flows WHERE id=?", (flow_id,)).fetchone()[0]
	slug = _slugify(name)
	for root in CHECKPOINT_ROOTS:
		flow_dir = root / slug
		if flow_dir.exists():
			return flow_dir / f"exp_{ga_seq:02d}"
	# Fall back to the first root even if it doesn't exist yet.
	return CHECKPOINT_ROOTS[0] / slug / f"exp_{ga_seq:02d}"


def main() -> int:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	g = ap.add_mutually_exclusive_group(required=True)
	g.add_argument("--flow-id", type=int, help="Flow id; auto-picks its GA experiment")
	g.add_argument("--experiment-id", type=int, help="Specific GA experiment id")
	ap.add_argument("--db", help="Path to dashboard DB (auto-detected if omitted)")
	ap.add_argument("--checkpoint-dir", help="Explicit exp checkpoint dir (overrides derivation)")
	ap.add_argument("--seed", type=int, help="Connectivity seed (default: flow config seed)")
	ap.add_argument("--allow-connection-loss", action="store_true",
	                help="Proceed even if the phase optimizes connections (lossy)")
	ap.add_argument("--dry-run", action="store_true", help="Report only; write nothing")
	args = ap.parse_args()

	db = resolve_db(args.db)
	con = sqlite3.connect(db)
	con.row_factory = sqlite3.Row

	# Resolve experiment + flow.
	if args.experiment_id:
		exp_id = args.experiment_id
		row = con.execute("SELECT flow_id, sequence_order, phase_type, status, max_iterations FROM experiments WHERE id=?", (exp_id,)).fetchone()
		if not row:
			raise SystemExit(f"Experiment {exp_id} not found")
		flow_id = row["flow_id"]
	else:
		flow_id = args.flow_id
		exp_id = pick_ga_experiment(con, flow_id)
		row = con.execute("SELECT flow_id, sequence_order, phase_type, status, max_iterations FROM experiments WHERE id=?", (exp_id,)).fetchone()

	ga_seq = row["sequence_order"]
	phase_type = row["phase_type"]
	max_iters = row["max_iterations"] or 0

	# Guard: connection-optimizing phases can't be faithfully reconstructed.
	if "conn" in (phase_type or "").lower() and not args.allow_connection_loss:
		raise SystemExit(
			f"Phase '{phase_type}' optimizes connections, which are not in the DB. "
			"Resume would re-sample them. Re-run with --allow-connection-loss to proceed."
		)

	# Latest completed generation for this experiment.
	it = con.execute(
		"SELECT iteration_num, best_ce, best_accuracy, fitness_threshold, patience_counter, patience_max "
		"FROM iterations WHERE experiment_id=? ORDER BY iteration_num DESC LIMIT 1",
		(exp_id,),
	).fetchone()
	if not it:
		raise SystemExit(f"Experiment {exp_id} has no recorded generations to recover")
	gen = it["iteration_num"]

	# Population at that generation (elites + offspring), with their configs.
	pop_rows = con.execute(
		"SELECT ge.role, ge.elite_rank, ge.ce, ge.accuracy, ge.f1_macro, ge.fpr, "
		"       g.tiers_json, g.total_neurons "
		"FROM genome_evaluations ge "
		"JOIN iterations it ON it.id = ge.iteration_id "
		"JOIN genomes g ON g.id = ge.genome_id "
		"WHERE it.experiment_id=? AND it.iteration_num=? "
		"ORDER BY (ge.role='elite') DESC, ge.elite_rank ASC",
		(exp_id, gen),
	).fetchall()
	if not pop_rows:
		raise SystemExit(f"No genome_evaluations for experiment {exp_id} gen {gen}")

	# Seed for connectivity regeneration (informational; connections=None lets the
	# GA regenerate on resume — we record the seed for traceability only).
	seed = args.seed
	if seed is None:
		cfg = con.execute("SELECT config_json FROM flows WHERE id=?", (flow_id,)).fetchone()
		try:
			seed = json.loads(cfg["config_json"]).get("params", {}).get("seed")
		except Exception:
			seed = None

	# Build ClusterGenome population (connections left None → regenerated on resume).
	try:
		from wnn.ram.metrics import IDSMetrics, Metrics
	except Exception:
		Metrics = None
	population: list[tuple[ClusterGenome, float]] = []
	best_genome = None
	for r in pop_rows:
		bpn, npc = tiers_to_arch(r["tiers_json"])
		genome = ClusterGenome(bits_per_neuron=bpn, neurons_per_cluster=npc, connections=None)
		if Metrics is not None:
			try:
				genome.metrics = IDSMetrics(ce=r["ce"], acc=r["accuracy"], f1=r["f1_macro"], fpr=r["fpr"])
			except Exception:
				pass
		population.append((genome, float(r["ce"]) if r["ce"] is not None else 0.0))
		if best_genome is None and r["role"] == "elite" and (r["elite_rank"] or 0) == 0:
			best_genome = genome
	if best_genome is None:
		best_genome = population[0][0]

	# Target checkpoint path.
	if args.checkpoint_dir:
		ckpt_dir = Path(args.checkpoint_dir)
	else:
		ckpt_dir = derive_checkpoint_dir(con, flow_id, ga_seq)

	n_elite = sum(1 for r in pop_rows if r["role"] == "elite")
	n_off = len(pop_rows) - n_elite
	print(f"Flow {flow_id} / experiment {exp_id} ({phase_type}, seq {ga_seq})")
	print(f"  last recorded generation : {gen}  (resume → {gen + 1}; max {max_iters})")
	print(f"  population               : {len(population)} genomes ({n_elite} elite + {n_off} offspring)")
	print(f"  best                     : ce={it['best_ce']:.4f} acc={it['best_accuracy']:.4%}")
	print(f"  patience                 : {it['patience_counter']}/{it['patience_max']}")
	print(f"  threshold                : {it['fitness_threshold']}")
	print(f"  seed (connectivity)      : {seed}")
	print(f"  checkpoint dir           : {ckpt_dir}")
	print(f"  checkpoint file          : {ckpt_dir / 'ga_checkpoint_ga.json'}")

	if args.dry_run:
		print("\n[dry-run] no checkpoint written.")
		return 0

	mgr = CheckpointManager(
		config=CheckpointConfig(enabled=True, checkpoint_dir=ckpt_dir, filename_prefix="ga_checkpoint"),
		phase_name=phase_type or "GA",
		optimizer_type="GA",
		total_iterations=max_iters or (gen + 1),
		logger=print,
	)
	mgr.save(
		iteration=gen,
		population=population,
		best_genome=best_genome,
		best_fitness=(it["best_ce"], it["best_accuracy"]),
		current_threshold=it["fitness_threshold"] or 0.0,
		extra_state={
			"patience_counter": it["patience_counter"] or 0,
			"reconstructed_from_db": True,
			"complete": False,
		},
	)

	# Round-trip verification.
	state = mgr.load(ClusterGenome)
	assert state["current_iteration"] == gen, "round-trip iteration mismatch"
	assert len(state["population"]) == len(population), "round-trip population size mismatch"
	print(f"\n✓ Wrote + verified checkpoint: resume will start at generation {gen + 1} "
	      f"with {len(state['population'])} genomes.")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
