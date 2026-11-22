"""Offline VAL re-score of each phase's winner (Luiz 21/11/2026: does VAL pick the grid
or the GA genome on unswt?). Investigation tool; touches no flow and no DB row.

For each flow: rebuild its evaluators from the stored params through the worker's own
IdsEvaluatorFactory (the same pair the flow_runner uses), rebuild the final best_f1
genome of the grid_search and ga_neurons experiments from the genomes table, and run
evaluate_multiclass_at_thresholds — ONE training pass on the full train split that
scores TEST and VAL. Needs a ram_accelerator that emits the `val:` modes.

FIDELITY GATE: the re-scored TEST margin_val_cal must equal the stored test_summaries
cell (same genome, same training, same seed) before its VAL number is printed.

  PYTHONPATH=<wheel dir>:<repo>/src python scripts/ids_val_rescore_phases.py 7492 7579 ...
"""
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ids57_abi_equivalence import apply_production_env, build_pair, flow_params, settings_for  # noqa: E402

DB = "file:/Users/lacg/wnn/db/wnn.db?mode=ro"
PHASES = ("grid_search", "ga_neurons")
TOL = 1e-9


def stored_cell(db, flow_id: int, phase: str) -> tuple[dict, str]:
	"""(stored TEST margin_val_cal entry, genome_hash) of the phase's final best_f1."""
	row = db.execute("select t.threshold_metadata, t.genome_hash from test_summaries t join experiments e "
		"on e.id=t.experiment_id where t.flow_id=? and e.phase_type=? and t.test_point='final' "
		"and t.genome_type='best_f1'", (flow_id, phase)).fetchone()
	if row is None:
		raise SystemExit(f"flow {flow_id}: no final best_f1 row for {phase}")
	return json.loads(row[0])["margin_val_cal"], row[1]


def genome_for(db, flow_id: int, genome_hash: str):
	from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
	row = db.execute("select g.tiers_json, g.connections_json from genomes g join experiments e on e.id=g.experiment_id "
		"where e.flow_id=? and g.genome_hash=? limit 1", (flow_id, genome_hash)).fetchone()
	if row is None or not row[1]:
		raise SystemExit(f"flow {flow_id}: genome {genome_hash} not stored")
	tiers = json.loads(row[0])
	return ClusterGenome(bits_per_neuron=list(tiers["bits_per_neuron"]),
		neurons_per_cluster=list(tiers["neurons_per_cluster"]),
		connections=[int(c) for c in row[1].split(",")])


def rescore(report, genome, stored: dict) -> dict:
	result = report.evaluate_multiclass_at_thresholds(genome)
	test = result["modes"]["margin_val_cal"]
	for key in ("macro_f1", "benign_fpr"):
		if abs(test[key] - stored[key]) > TOL:
			raise SystemExit(f"FIDELITY FAIL {key}: re-scored {test[key]!r} vs stored {stored[key]!r}")
	val = result.get("val_partition", {}).get("margin_val_cal")
	if val is None:
		raise SystemExit("this ram_accelerator does not emit val: modes")
	return {"test_f1": test["macro_f1"], "test_fpr": test["benign_fpr"],
		"val_f1": val["macro_f1"], "val_fpr": val["benign_fpr"]}


def flow_rows(db, flow_id: int) -> dict:
	from wnn.ram.experiments.IdsEvaluatorFactory import IdsEvaluatorFactory
	params = flow_params(db, flow_id)
	apply_production_env(params)
	s = settings_for(params)
	factory = IdsEvaluatorFactory(lambda m: None)
	_, report = build_pair(factory, factory.load(s), s)
	out = {}
	for phase in PHASES:
		stored, genome_hash = stored_cell(db, flow_id, phase)
		out[phase] = rescore(report, genome_for(db, flow_id, genome_hash), stored)
	return out


def main(flow_ids: list[int]) -> None:
	db = sqlite3.connect(DB, uri=True)
	print("flow   VAL F1 grid → GA      VAL FPR grid → GA   VAL picks | TEST F1 grid → GA     TEST FPR grid → GA")
	for fid in flow_ids:
		r = flow_rows(db, fid)
		g, a = r["grid_search"], r["ga_neurons"]
		pick = "GA" if a["val_f1"] > g["val_f1"] else "grid"
		print(f"{fid}   {g['val_f1']*100:6.2f} → {a['val_f1']*100:6.2f}    {g['val_fpr']*100:5.2f} → {a['val_fpr']*100:5.2f}"
			f"      {pick:4s}    | {g['test_f1']*100:6.2f} → {a['test_f1']*100:6.2f}     {g['test_fpr']*100:5.2f} → {a['test_fpr']*100:5.2f}",
			flush=True)
		print("JSON", json.dumps({"flow": fid, **r}), flush=True)


if __name__ == "__main__":
	main([int(x) for x in sys.argv[1:]])
