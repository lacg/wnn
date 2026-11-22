"""IDS-76 step 0 (Luiz 21/11/2026): is the GA overfitting its FIXED K-fold partition?

For each flow: rebuild the K-fold OPTIMIZER evaluator (the GA's fitness function, through
the worker's IdsEvaluatorFactory) once with the flow's own seed (the folds the GA searched
on) and once per FRESH seed (a different stratified fold partition, everything else equal),
and score the flow's final grid and GA best_f1 genomes with evaluate_batch (5-fold mean).

  edge_orig  = GA - grid on the original folds
  edge_fresh = GA - grid on fresh folds (mean over FRESH_OFFSETS)

edge_fresh << edge_orig => the GA fit its folds (rotation is the fix).
edge_fresh ~= edge_orig but TEST edge ~ 0 => the folds are fine; the loss is train->test shift.
Investigation tool; touches no flow and no DB row.

  PYTHONPATH=<repo>/src python scripts/ids_fold_overfit_precheck.py 7492 7579 ...
"""
import dataclasses
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ids57_abi_equivalence import apply_production_env, flow_params, settings_for  # noqa: E402
from ids_val_rescore_phases import PHASES, genome_for, stored_cell  # noqa: E402

DB = "file:/Users/lacg/wnn/db/wnn.db?mode=ro"
FRESH_OFFSETS = (100_003, 200_003, 300_007)


def kfold_f1(factory, dataset, s, genomes: list) -> list[float]:
	"""5-fold CV mean F1 of each genome on the fold partition `s.seed` defines."""
	optimizer = factory._optimizer(dataset, s)
	return [m.f1 for m in optimizer.evaluate_batch(genomes)]


def flow_record(db, flow_id: int) -> dict:
	from wnn.ram.experiments.IdsEvaluatorFactory import IdsEvaluatorFactory
	params = flow_params(db, flow_id)
	apply_production_env(params)
	s = settings_for(params)
	factory = IdsEvaluatorFactory(lambda m: None)
	dataset = factory.load(s)
	genomes, test = [], {}
	for phase in PHASES:
		stored, genome_hash = stored_cell(db, flow_id, phase)
		genomes.append(genome_for(db, flow_id, genome_hash))
		test[phase] = stored["macro_f1"]
	orig = kfold_f1(factory, dataset, s, genomes)
	fresh = [kfold_f1(factory, dataset, dataclasses.replace(s, seed=s.seed + off), genomes) for off in FRESH_OFFSETS]
	return {"flow": flow_id, "orig": orig, "fresh": fresh, "test": [test[p] for p in PHASES]}


def main(flow_ids: list[int]) -> None:
	db = sqlite3.connect(DB, uri=True)
	for fid in flow_ids:
		r = flow_record(db, fid)
		e_orig = (r["orig"][1] - r["orig"][0]) * 100
		e_fresh = sum((f[1] - f[0]) for f in r["fresh"]) / len(r["fresh"]) * 100
		e_test = (r["test"][1] - r["test"][0]) * 100
		print(f"{fid}  GA-grid k-fold F1: original folds {e_orig:+.3f} | fresh folds {e_fresh:+.3f} | TEST {e_test:+.3f}", flush=True)
		print("JSON", json.dumps(r), flush=True)


if __name__ == "__main__":
	main([int(x) for x in sys.argv[1:]])
