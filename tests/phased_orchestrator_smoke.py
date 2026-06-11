"""
Smoke test for the shared phased-search machinery (D1, 11/06/2026):
CarryState rules, schema-2 checkpoint roundtrip, legacy-format loaders,
PhasedOrchestrator loop + resume, and the SIGTERM emergency dump.

Run: PYTHONPATH=src python tests/phased_orchestrator_smoke.py
"""

import gzip
import json
import os
import pickle
import signal
import sys
import tempfile
from pathlib import Path

from wnn.ram.strategies.phased import (
	CarryState, PhaseCheckpoint, PhaseOutcome, PhaseSpec, PhasedOrchestrator,
	PickleBase64Codec, save_checkpoint, load_checkpoint,
)


class ToyOrchestrator(PhasedOrchestrator):
	"""Phases 'grow' a list-genome; phase 'dead' returns an empty outcome."""
	def run_phase(self, spec, carry, index):
		if spec.key == "skip":
			return None
		if spec.key == "dead":
			# Finished but produced nothing — carry must NOT be wiped
			return PhaseOutcome(strategy_type="GA", iterations_run=3, patience=2)
		genome = (carry.genome or []) + [spec.key]
		pop = [genome, genome + ["alt"]]
		return PhaseOutcome(
			best_genome=genome, final_population=pop, final_threshold=0.1 * (index + 1),
			final_fitness=1.0 / (index + 1), iterations_run=10 + index, patience=index,
			strategy_type="GA",
		)


def main() -> int:
	checks = {}
	with tempfile.TemporaryDirectory() as td:
		codec = PickleBase64Codec()
		orch = ToyOrchestrator(td, codec, log=lambda s: None)
		phases = [PhaseSpec("1a", "Phase 1a"), PhaseSpec("dead", "Dead Phase"),
		          PhaseSpec("skip", "Skipped"), PhaseSpec("2a", "Phase 2a")]
		carry = CarryState()
		out = orch.run_all(phases, carry)

		checks["phases ran"] = set(out) == {"1a", "dead", "2a"}
		checks["carry survived dead+skip phases"] = carry.genome == ["1a", "2a"]
		checks["population carried"] = carry.population is not None and len(carry.population) == 2
		checks["gen+patience persisted"] = (
			orch.load_phase_checkpoint(phases[1]).iterations_run == 3
			and orch.load_phase_checkpoint(phases[1]).patience == 2
		)

		# --- resume: fresh orchestrator skips 1a, reloads it from disk ---
		orch2 = ToyOrchestrator(td, codec, log=lambda s: None)
		carry2 = CarryState()
		out2 = orch2.run_all(phases[:2], carry2, resume_from="dead")
		checks["resume reloads checkpoint into carry"] = carry2.genome == ["1a"]
		checks["resume outcome has fitness"] = out2["1a"].final_fitness == 1.0

		# --- legacy experiments json.gz loads ---
		legacy = {"phase_result": {
			"phase_name": "Phase 1a: GA", "strategy_type": "GA",
			"final_fitness": 9.9, "final_accuracy": 0.5, "iterations_run": 7,
			"best_genome": codec.encode(["legacy"]), "final_threshold": 0.3,
		}, "_metadata": {"phase_key": "1a"}}
		lp = Path(td) / "legacy.json.gz"
		with gzip.open(lp, "wt") as f:
			json.dump(legacy, f)
		ck = load_checkpoint(lp, codec)
		checks["legacy json loads"] = ck.final_fitness == 9.9 and ck.best_genome == ["legacy"] and ck.iterations_run == 7

		# --- legacy controller pickle loads ---
		pp = Path(td) / "stage1.pkl"
		with open(pp, "wb") as f:
			pickle.dump({"stage_num": 1, "stage_name": "NEURONS", "best_genome": ["ctl"],
			             "population": [["ctl"]], "generation": 42, "spec": "SPEC",
			             "meta": {"tilt_deg": 5}}, f)
		ck2 = load_checkpoint(pp, codec)
		checks["legacy pickle loads"] = (ck2.iterations_run == 42 and ck2.best_genome == ["ctl"]
		                                 and ck2.extra.get("spec") == "SPEC")

		# --- SIGTERM emergency dump (real signal) ---
		class Hang(ToyOrchestrator):
			def run_phase(self, spec, carry, index):
				if spec.key == "boom":
					os.kill(os.getpid(), signal.SIGTERM)  # handler dumps, then chains
				return super().run_phase(spec, carry, index)
		dumped = []
		prev = signal.signal(signal.SIGTERM, lambda *_: dumped.append("chained"))
		try:
			h = Hang(td, codec, log=lambda s: None)
			c3 = CarryState(genome=["pre"], population=[["pre"]])
			h.run_all([PhaseSpec("boom", "Boom")], c3)
		finally:
			signal.signal(signal.SIGTERM, prev)
		dumps = list(Path(td).glob("emergency_*.yaml.gz"))
		checks["SIGTERM dumped state"] = len(dumps) == 1
		checks["previous handler chained"] = dumped == ["chained"]
		if dumps:
			eck = load_checkpoint(dumps[0], codec)
			checks["dump holds the carry"] = eck.best_genome == ["pre"] and eck.extra.get("emergency_dump") is True

	failed = [k for k, ok in checks.items() if not ok]
	for k, ok in checks.items():
		print(f"  [{'PASS' if ok else 'FAIL'}] {k}")
	print("ALL PASS" if not failed else f"FAILED: {failed}")
	return 1 if failed else 0


if __name__ == "__main__":
	sys.exit(main())
