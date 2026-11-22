"""Plane IDS-62 (lever A): OFFLINE seed-ensemble of the per-seed GA best_f1 winners.
Investigation tool; touches no flow and no DB row (read-only DB URI).

For each flow of a cohort: rebuild its evaluators from the stored params through the
worker's own IdsEvaluatorFactory (same pair the flow_runner uses), rebuild the final
ga_neurons best_f1 genome, and export its K-vector scores on TEST / train / VAL from ONE
training pass (IDSCacheWrapper.score_multiclass_sets, Rust).

FIDELITY GATE: decoding the exported scores with ram_accelerator.multiclass_modes_from_scores
must reproduce the stored test_summaries margin_val_cal cell (macro_f1, benign_fpr, acc)
exactly, per genome, before anything is combined.

Ensembles (members FIXED = the seed winners; VAL/TEST never pick members):
  avg  — mean of the members' per-row K-vector scores, then the SAME margin_val_cal decode
         (τ swept on the ensemble's VAL margins, applied to TEST) — multiclass_modes_from_scores.
  vote — majority of the members' own margin_val_cal decisions (each member at its own VAL τ,
         multiclass_margin_predictions); metrics by multiclass_metrics_from_predictions. Only for an
         odd number of members (2 members can tie — reported n/a).
TEST is the result; every VAL number is printed with a VAL label (tuning only).

  PYTHONPATH=<private wheel dir>:<repo>/src python scripts/ids62_seed_ensemble.py unswt=7600,7601,7602 ...
"""
import gc
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ids57_abi_equivalence import apply_production_env, build_pair, flow_params, settings_for  # noqa: E402
from ids_val_rescore_phases import genome_for  # noqa: E402

DB = "file:/Users/lacg/wnn/db/wnn.db?mode=ro"
GATE_KEYS = ("macro_f1", "benign_fpr", "acc")
# MulticlassModeTuple field order (pyapi/ids_cache_api.rs)
TUPLE_FIELDS = ("mode", "tau", "ce", "acc", "macro_f1", "weighted_f1", "benign_fpr")


def as_entry(t) -> dict:
	return dict(zip(TUPLE_FIELDS, t[:len(TUPLE_FIELDS)]))


def stored_cell(db, flow_id: int) -> tuple[dict, str, int]:
	row = db.execute("select t.threshold_metadata, t.genome_hash, t.logical_cells from test_summaries t "
		"join experiments e on e.id=t.experiment_id where t.flow_id=? and e.phase_type='ga_neurons' "
		"and t.test_point='final' and t.genome_type='best_f1'", (flow_id,)).fetchone()
	if row is None:
		raise SystemExit(f"flow {flow_id}: no final GA best_f1 row")
	return json.loads(row[0])["margin_val_cal"], row[1], row[2]


def decode(accel, test, train, val, labels, k: int, normal: int) -> dict:
	"""{mode: entry} incl. 'val:' VAL-partition entries, from flat scores."""
	_, modes = accel.multiclass_modes_from_scores(
		test.ravel().tolist(), labels["test"].tolist(), train.ravel().tolist(), labels["train"].tolist(),
		k, normal, val.ravel().tolist(), labels["val"].tolist())
	return {t[0]: as_entry(t) for t in modes}


def member(accel, db, flow_id: int) -> dict:
	"""Export one seed winner's scores and pass the fidelity gate."""
	from wnn.ram.experiments.IdsEvaluatorFactory import IdsEvaluatorFactory
	params = flow_params(db, flow_id)
	apply_production_env(params)
	s = settings_for(params)
	factory = IdsEvaluatorFactory(lambda m: None)
	optimizer, report = build_pair(factory, factory.load(s), s)
	stored, genome_hash, cells = stored_cell(db, flow_id)
	test, train, val, y_test, y_train, y_val = report.score_multiclass_sets(genome_for(db, flow_id, genome_hash))
	if val is None:
		raise SystemExit(f"flow {flow_id}: no VAL partition (not a _3way split)")
	k, normal = test.shape[1], int(report._normal_class)
	labels = {"test": np.asarray(y_test), "train": np.asarray(y_train), "val": np.asarray(y_val)}
	modes = decode(accel, test, train, val, labels, k, normal)
	got = modes["margin_val_cal"]
	for key in GATE_KEYS:
		if got[key] != stored[key]:
			raise SystemExit(f"FIDELITY FAIL flow {flow_id} {key}: re-scored {got[key]!r} vs stored {stored[key]!r}")
	print(f"[gate] flow {flow_id} genome {genome_hash}: TEST margin_val_cal reproduced exactly "
		f"(F1 {got['macro_f1']!r}, FPR {got['benign_fpr']!r}, acc {got['acc']!r}, τ {got['tau']!r})", flush=True)
	out = {"flow": flow_id, "genome": genome_hash, "cells": cells, "k": k, "normal": normal, "labels": labels,
		"test": np.array(test), "train": np.array(train), "val": np.array(val), "tau": got["tau"],
		"test_cell": got, "val_cell": modes["val:margin_val_cal"]}
	del optimizer, report, factory
	gc.collect()
	return out


def vote(accel, members: list[dict], part: str) -> np.ndarray:
	"""Majority of the members' own margin_val_cal decisions on `part` (odd member count)."""
	k, normal = members[0]["k"], members[0]["normal"]
	preds = np.array([accel.multiclass_margin_predictions(m[part].ravel().tolist(), k, m["tau"], normal) for m in members])
	counts = np.stack([(preds == c).sum(axis=0) for c in range(k)], axis=1)
	winner = counts.argmax(axis=1)
	if (counts.max(axis=1) * 2 <= len(members)).any():
		raise SystemExit(f"vote: no strict majority on some {part} rows (K={k}, {len(members)} members)")
	return winner.astype(np.uint32)


def ensembles(accel, members: list[dict]) -> dict:
	k, normal, labels = members[0]["k"], members[0]["normal"], members[0]["labels"]
	for m in members[1:]:
		for part in ("test", "train", "val"):
			if not np.array_equal(m["labels"][part], labels[part]):
				raise SystemExit(f"members disagree on {part} labels — not the same partition")
	mean = {part: np.mean([m[part] for m in members], axis=0) for part in ("test", "train", "val")}
	modes = decode(accel, mean["test"], mean["train"], mean["val"], labels, k, normal)
	out = {"avg": {"test": modes["margin_val_cal"], "val": modes["val:margin_val_cal"]}}
	if len(members) % 2 == 1:
		out["vote"] = {part: as_entry(accel.multiclass_metrics_from_predictions(
			mean[part].ravel().tolist(), vote(accel, members, part).tolist(), labels[part].tolist(), k, normal, "vote"))
			for part in ("test", "val")}
	return out


def fmt(e: dict) -> str:
	return f"{e['macro_f1']*100:7.3f} {e['benign_fpr']*100:7.3f} {e['acc']*100:7.3f}"


def report_cohort(name: str, members: list[dict], ens: dict) -> None:
	cells = sum(m["cells"] for m in members)
	print(f"\n=== {name}  (flows {','.join(str(m['flow']) for m in members)}; {len(members)} seeds) ===")
	print("row              | TEST F1   FPR     acc    | VAL F1    FPR     acc    | cells")
	for i, m in enumerate(members, 1):
		print(f"single s{i} {m['flow']}  | {fmt(m['test_cell'])} | {fmt(m['val_cell'])} | {m['cells']}")
	mean = {key: float(np.mean([m["test_cell"][key] for m in members])) for key in GATE_KEYS}
	vmean = {key: float(np.mean([m["val_cell"][key] for m in members])) for key in GATE_KEYS}
	print(f"single mean      | {fmt(mean)} | {fmt(vmean)} | {cells / len(members):.0f}")
	for kind in ("avg", "vote"):
		if kind in ens:
			print(f"{kind}-ensemble     | {fmt(ens[kind]['test'])} | {fmt(ens[kind]['val'])} | {cells}")
		else:
			print(f"{kind}-ensemble     | n/a (even member count can tie)")
	print("JSON", json.dumps({"cohort": name, "members": [{"flow": m["flow"], "genome": m["genome"], "cells": m["cells"],
		"test": m["test_cell"], "VAL": m["val_cell"]} for m in members],
		"ensembles": {k: {"test": v["test"], "VAL": v["val"]} for k, v in ens.items()}}), flush=True)


def main(specs: list[str]) -> None:
	from wnn.accel import require_accel
	accel = require_accel()
	db = sqlite3.connect(DB, uri=True)
	for spec in specs:
		name, flows = spec.split("=")
		members = [member(accel, db, int(f)) for f in flows.split(",")]
		report_cohort(name, members, ensembles(accel, members))
		del members
		gc.collect()


if __name__ == "__main__":
	main(sys.argv[1:])
