#!/usr/bin/env python3
"""Re-record the rate-0.25 FPGA fidelity reference row under the CURRENT trainer (IDS-55 follow-up).

Why: tests/test_fpga_export_fidelity.py[GAF-rate0.25] checks the exporter against
flow 6928's recorded validation row. That row is stamped worker_abi=16; since
worker ABI 25 (IDS-50) the rate < 1 training-sample mask is keyed on the neuron's
IDENTITY instead of its flat position, so the same genome trains a different
memory and the ABI-16 row cannot be reproduced by design (the ABI-24 tree + ABI-24
wheel still reproduce it bit-exact). The DB row stays as it is: it is the true
record of what that run produced.

How: the worker's OWN flow assembly + Experiment.revalidate_genome — the method
final validation calls (via scripts/ids2/revalidate_inherited.py's helpers) —
with a client that CAPTURES the summary row instead of POSTing it. Nothing is
re-implemented and nothing is written to the DB; dashboard reads only.

Usage (worker wheel installed, ABI >= 25):
  PYTHONPATH=src python scripts/ids55/rerecord_fidelity_reference.py \
      --out tests/fixtures/fpga_fidelity/gaf_6928_best_f1_ga_neurons.json
"""
import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "ids2"))
from revalidate_inherited import build_flow, load_genome, make_experiment  # noqa: E402

DB = "file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro"
FLOW, GENOME_TYPE, PHASE = 6928, "best_f1", "ga_neurons"


class CapturingClient:
	"""The one write revalidate_genome makes (the summary upsert) is captured;
	any other client call is refused loudly — this script must never write."""

	def __init__(self):
		self.rows = []

	def create_validation_summary(self, **row) -> dict:
		self.rows.append(row)
		return {"id": None}

	def __getattr__(self, name):
		raise AttributeError(f"CapturingClient refuses {name!r}: re-recording never talks to the dashboard")


def recorded_row() -> dict:
	"""The DB row the fixture replaces (experiment, genome hash, ABI, metadata)."""
	con = sqlite3.connect(DB, uri=True)
	row = con.execute("""SELECT vs.experiment_id, vs.genome_hash, vs.worker_abi, vs.threshold_metadata
		FROM validation_summaries vs JOIN experiments e ON e.id = vs.experiment_id AND e.phase_type = ?
		WHERE vs.flow_id = ? AND vs.validation_point = 'final' AND vs.genome_type = ?
		ORDER BY vs.id DESC LIMIT 1""", (PHASE, FLOW, GENOME_TYPE)).fetchone()
	return dict(experiment_id=row[0], genome_hash=row[1], worker_abi=row[2], threshold_metadata=json.loads(row[3]))


def rescore(old: dict, url: str) -> dict:
	"""The captured row of a fresh final validation of the stored genome."""
	from wnn.ram.experiments.worker import FlowWorker
	from wnn.ram.metrics import GenomeType
	# db_path=":memory:" — the worker defaults to db/wnn.db under the CWD (the LIVE DB
	# from the repo root); the tracker must not touch any file.
	worker = FlowWorker(dashboard_url=url, verify_ssl=False, db_path=":memory:")
	flow, exp_ids, _ = build_flow(worker, FLOW)
	try:
		exp = make_experiment(worker, flow, exp_ids, old["experiment_id"], FLOW)
		exp.dashboard_client = CapturingClient()
		genome = load_genome(old["genome_hash"], old["experiment_id"])
		if genome is None or exp._compute_genome_hash(genome) != old["genome_hash"]:
			raise RuntimeError("stored genome missing or hash mismatch")
		exp.revalidate_genome(genome, GenomeType(GENOME_TYPE), FLOW)
		[row] = exp.dashboard_client.rows
		return row
	finally:
		worker._cleanup_after_flow()


def main() -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("--out", required=True)
	ap.add_argument("--url", default="https://localhost:3000")
	a = ap.parse_args()
	from wnn.accel import installed_abi, require_accel
	require_accel()
	old = recorded_row()
	row = rescore(old, a.url)
	commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
	fixture = dict(
		flow_id=FLOW, genome_type=GENOME_TYPE, phase=PHASE, genome_hash=old["genome_hash"],
		worker_abi=installed_abi(), recorded_by="scripts/ids55/rerecord_fidelity_reference.py", source_commit=commit,
		replaces=dict(worker_abi=old["worker_abi"], threshold_metadata=old["threshold_metadata"]),
		why=("rate 0.25: worker ABI 25 (IDS-50) keys the training-sample mask on neuron identity, not "
		     "flat position, so the ABI-16 DB row is not reproducible by design; the DB row is untouched"),
		ce=row["ce"], accuracy=row["accuracy"], f1_macro=row["f1_macro"], fpr=row["fpr"],
		threshold_metadata=json.loads(row["threshold_metadata"]),
	)
	Path(a.out).write_text(json.dumps(fixture, indent="\t") + "\n")
	print(f"wrote {a.out} (worker_abi {fixture['worker_abi']}, {len(fixture['threshold_metadata'])} modes)")
	return 0


if __name__ == "__main__":
	sys.exit(main())
