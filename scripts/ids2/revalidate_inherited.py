#!/usr/bin/env python3
"""Re-validate validation_summaries rows the pre-fix cross-flow cache inherited (10/10/2026).

Every listed row is a FINAL validation whose numbers were copied from another flow
(different memory mode / trainer ABI, or different fitness weights). This retrains
the row's OWN stored genome under its OWN flow's config and upserts the row through
the dashboard API, so it ends up with cache_key + worker_abi stamped.

Parity with the worker by construction: the flow's evaluators, experiment configs
and cache scope are built by FlowWorker._build_flow (the flow_runner path), and the
scoring is Experiment.revalidate_genome -> _validate_one_genome, the same method
_run_validation calls. Nothing is re-implemented here.

Usage:
  python scripts/ids2/revalidate_inherited.py --rows experiments/ids2_inherited_cells.json --dry-run
  python scripts/ids2/revalidate_inherited.py --rows ... --flow 6338          # smoke one flow
  python scripts/ids2/revalidate_inherited.py --rows ... --out experiments/ids2_revalidation.json
"""
import argparse
import json
import sqlite3
import sys
import time
from collections import defaultdict

DB = "file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro"
MODES = ["train_cal", "fixed_05", "platt", "beta", "empirical", "empirical_cumulative", "val_cal"]


def ro():
	return sqlite3.connect(DB, uri=True)


def load_targets(rows_path: str, only_flow: int | None) -> dict[int, list[dict]]:
	"""Row ids from the audit file -> {flow_id: [row, ...]} with the row's stored values."""
	spec = json.load(open(rows_path))
	ids = set(spec["drop_all_modes"]["row_ids"]) | set(spec["drop_empirical_cumulative_only"]["row_ids"])
	c = ro()
	q = f"""select v.id, v.flow_id, v.experiment_id, v.genome_type, v.genome_hash, v.threshold_metadata
		from validation_summaries v where v.validation_point='final' and v.id in ({','.join('?' * len(ids))})"""
	by_flow: dict[int, list[dict]] = defaultdict(list)
	for vid, fid, eid, gt, gh, tm in c.execute(q, sorted(ids)):
		if only_flow is None or fid == only_flow:
			by_flow[fid].append(dict(id=vid, flow=fid, exp=eid, gt=gt, gh=gh, old=json.loads(tm) if tm else None))
	return by_flow


def load_genome(gh: str, exp_id: int):
	"""Rebuild the stored genome (bits, neurons, connections): the genomes table first,
	else the experiment's checkpoint (genomes below the leaderboard gate are only there)."""
	from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
	row = ro().execute("""select tiers_json, connections_json from genomes
		where genome_hash=? and connections_json is not null order by id limit 1""", (gh,)).fetchone()
	if row is not None:
		tiers = json.loads(row[0])
		return ClusterGenome(bits_per_neuron=tiers["bits_per_neuron"],
			neurons_per_cluster=tiers["neurons_per_cluster"], connections=[int(x) for x in row[1].split(",")])
	return genome_from_checkpoint(gh, exp_id)


def checkpoint_genome_dicts(ckpt: dict):
	"""Every genome dict a phase checkpoint holds: best_<type>_genome, best_genome, final_population."""
	pr = ckpt.get("phase_result", {})
	yield from (v for k, v in ckpt.items() if k.startswith("best_") and k.endswith("_genome") and isinstance(v, dict))
	if isinstance(pr.get("best_genome"), dict):
		yield pr["best_genome"]
	yield from (g for g in pr.get("final_population") or [] if isinstance(g, dict))


def genome_from_checkpoint(gh: str, exp_id: int):
	import gzip
	from wnn.ram.experiments.experiment import Experiment
	from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
	for (path,) in ro().execute("select file_path from checkpoints where experiment_id=? order by id desc", (exp_id,)):
		full = path if path.startswith("/") else f"/Users/lacg/wnn/{path}"
		try:
			ckpt = json.load(gzip.open(full))
		except OSError:
			continue
		for d in checkpoint_genome_dicts(ckpt):
			g = ClusterGenome(bits_per_neuron=d["bits_per_neuron"], neurons_per_cluster=d["neurons_per_cluster"],
				connections=d.get("connections"))
			if Experiment._compute_genome_hash(None, g) == gh:
				return g
	return None


def build_flow(worker, fid: int):
	"""The worker's own flow assembly (env overrides + evaluators + configs + scope)."""
	flow_data = worker.client.get_flow(fid)
	params = flow_data["config"]["params"]
	worker._apply_env_overrides(params, fid)
	flow = worker._build_flow(flow_data, fid, flow_data["name"], params)
	exp_ids = [e["id"] for e in worker.client.list_flow_experiments(fid)]
	return flow, exp_ids, flow_data["name"]


def make_experiment(worker, flow, exp_ids: list[int], eid: int, fid: int):
	from wnn.ram.experiments.experiment import Experiment
	cfg = flow.config.experiments[exp_ids.index(eid)]
	return Experiment(config=cfg, evaluator=flow.evaluator, logger=worker._log, checkpoint_dir=None,
		dashboard_client=worker.client, experiment_id=eid, tracker=None, flow_id=fid,
		shutdown_check=None, full_evaluator=flow.full_evaluator,
		validation_scope=flow.config.validation_scope)


def stored_row(vid: int) -> tuple:
	return ro().execute("""select threshold_metadata, cache_key, worker_abi
		from validation_summaries where id=?""", (vid,)).fetchone()


def verify_row(t: dict, expect_abi: int) -> dict:
	"""The upsert must have landed with all 7 modes + scope stamped (no silent fallback)."""
	tm, key, abi = stored_row(t["id"])
	new = json.loads(tm) if tm else {}
	missing = [m for m in MODES if not isinstance(new.get(m), dict)]
	if missing or key is None or abi != expect_abi:
		raise RuntimeError(f"row {t['id']}: missing modes {missing} key={key is not None} abi={abi}")
	return new


def mode_drift(old: dict | None, new: dict) -> float:
	"""Max |old-new| (pp) over the 6 weight-independent modes: ~0 means the inherited
	row was the same model (weights-only class); large means another model."""
	if not old:
		return float("nan")
	d = 0.0
	for m in MODES:
		if m == "empirical_cumulative" or not isinstance(old.get(m), dict):
			continue
		for k in ("f1", "fpr", "acc"):
			d = max(d, 100 * abs(old[m][k] - new[m][k]))
	return d


def revalidate_flow(worker, fid: int, targets: list[dict], expect_abi: int) -> list[dict]:
	from wnn.ram.metrics import GenomeType
	flow, exp_ids, name = build_flow(worker, fid)
	out = []
	try:
		for t in targets:
			genome = load_genome(t["gh"], t["exp"])
			exp = make_experiment(worker, flow, exp_ids, t["exp"], fid)
			if genome is None or exp._compute_genome_hash(genome) != t["gh"]:
				raise RuntimeError(f"row {t['id']}: stored genome missing or hash mismatch")
			exp.revalidate_genome(genome, GenomeType(t["gt"]), fid)
			new = verify_row(t, expect_abi)
			out.append(dict(row=t["id"], flow=fid, name=name, exp=t["exp"], gt=t["gt"], gh=t["gh"],
				drift_pp=mode_drift(t["old"], new), old=t["old"], new=new))
	finally:
		worker._cleanup_after_flow()
	return out


def dry_run(by_flow: dict[int, list[dict]]) -> None:
	from wnn.ram.experiments.experiment import Experiment
	n = bad = 0
	for fid, ts in sorted(by_flow.items()):
		for t in ts:
			g = load_genome(t["gh"], t["exp"])
			ok = g is not None and Experiment._compute_genome_hash(None, g) == t["gh"]
			n += 1
			bad += not ok
			if not ok:
				print(f"  MISSING/MISMATCH row {t['id']} flow {fid} {t['gt']} {t['gh']}")
	print(f"dry-run: {len(by_flow)} flows, {n} rows, {n - bad} genomes recoverable + hash-verified, {bad} not")


def main() -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("--rows", required=True)
	ap.add_argument("--flow", type=int, default=None)
	ap.add_argument("--out", default="experiments/ids2_revalidation.json")
	ap.add_argument("--dry-run", action="store_true")
	ap.add_argument("--url", default="https://localhost:3000")
	a = ap.parse_args()
	by_flow = load_targets(a.rows, a.flow)
	if a.dry_run:
		dry_run(by_flow)
		return 0
	from wnn.accel import installed_abi, require_accel
	from wnn.ram.experiments.worker import FlowWorker
	require_accel()
	abi = installed_abi()
	worker = FlowWorker(dashboard_url=a.url, verify_ssl=False)
	results, t0 = [], time.time()
	for i, (fid, ts) in enumerate(sorted(by_flow.items()), 1):
		results += revalidate_flow(worker, fid, ts, abi)
		print(f"[reval] {i}/{len(by_flow)} flow {fid}: {len(ts)} rows ({time.time() - t0:.0f}s)", flush=True)
	json.dump(dict(worker_abi=abi, rows=results), open(a.out, "w"), indent=1)
	print(f"[reval] DONE {len(results)} rows in {time.time() - t0:.0f}s -> {a.out}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
