#!/usr/bin/env python3
"""IDS-25: re-score a flow's grid-search population on held-out TEST, writing NOTHING to the DB.

For one flow, takes the Grid Search phase checkpoint (final_population = the top-50 of the
grid points by search fitness, with connections + k-fold population_metrics) and re-scores
every genome through the worker's own validation path:
  FlowWorker._build_flow (flow_runner assembly) -> Experiment.revalidate_genome
  -> _validate_one_genome (single training pass on the 80% train, Protocol-v2 7 threshold modes)
Nothing is re-implemented. No source file is edited.

Write-safety, by construction:
  * the worker object is built WITHOUT FlowWorker.__init__ (no SqliteTracker on the live DB,
    no signal handlers); its dashboard client is a ReadOnlyClient that only forwards
    get_*/list_* calls and raises on anything else;
  * the Experiment gets a CaptureClient as dashboard_client: create_validation_summary is
    captured in memory (that is how threshold_metadata comes back), every other call raises.
    revalidate_genome runs with use_cache=False, submit_leaderboard=False.

Search-side numbers per genome (k-fold CV on the 80% train, the GA's view):
  population_metrics[i] {ce, acc, f1, fpr} + fitness_score from the grid's final iteration
  (genome_evaluations, matched by position and verified on ce).

Usage: python scripts/ids_plateau/ids25_rescore_grid.py --flow 5965 --out experiments/ids_plateau/ids25_flow_5965.json [--limit N]
"""
import argparse
import gzip
import json
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path

DB = "file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro"
GRID_CKPT = "grid_search_(neurons_x_bits).json.gz"


class ReadOnlyClient:
	"""Forwards only read calls (get_*/list_*) to the real dashboard client."""

	def __init__(self, inner):
		self._inner = inner

	def __getattr__(self, name):
		if name.startswith(("get_", "list_")):
			return getattr(self._inner, name)
		raise RuntimeError(f"ReadOnlyClient: refused dashboard call '{name}'")


class CaptureClient:
	"""Experiment dashboard_client stand-in: captures validation summaries in memory."""

	def __init__(self):
		self.rows = []

	def create_validation_summary(self, experiment_id, validation_point, genome_type, genome_hash, ce, accuracy,
			flow_id, f1_macro, fpr, threshold_metadata, scope):
		self.rows.append(dict(genome_hash=genome_hash, ce=ce, acc=accuracy, f1=f1_macro, fpr=fpr,
			tm=json.loads(threshold_metadata) if threshold_metadata else None))

	def __getattr__(self, name):
		raise RuntimeError(f"CaptureClient: refused dashboard call '{name}'")


def ro():
	return sqlite3.connect(DB, uri=True)


def make_worker(url: str):
	"""A FlowWorker with only the state _build_flow/_apply_env_overrides need."""
	from wnn.ram.experiments.worker import FlowWorker
	from wnn.ram.experiments.dashboard_client import DashboardClient, DashboardClientConfig
	from wnn.ram.experiments.tracker import NoOpTracker
	from wnn.ram.experiments import scheduler
	w = FlowWorker.__new__(FlowWorker)
	w.dashboard_url, w.poll_interval, w.checkpoint_base_dir = url, 10, Path("/nonexistent_ids25_ckpt")
	w.context_size, w.verify_ssl, w.running = 4, False, True
	w._stop_current_flow = w._pause_current_flow = False
	w.current_flow_id = None
	w.cpu_budget = scheduler.detect_budget()
	w._running = {}
	w._heartbeat_thread, w._heartbeat_stop_event = None, threading.Event()
	w._tokenizer = w._train_tokens = w._test_tokens = w._validation_tokens = None
	w._vocab_size = w._cluster_order = None
	w._log_file, w._log_dir = None, Path("/nonexistent_ids25_logs")
	w.client = ReadOnlyClient(DashboardClient(DashboardClientConfig(base_url=url, verify_ssl=False), logger=w._log))
	w.tracker = NoOpTracker()
	return w


def grid_experiment(fid: int) -> tuple[int, str]:
	r = ro().execute("select e.id, f.name from experiments e join flows f on f.id=e.flow_id where e.flow_id=? and e.phase_type='grid_search'", (fid,)).fetchone()
	return r[0], r[1]


def load_checkpoint(name: str) -> dict:
	p = Path("/Users/lacg/wnn/checkpoints") / name.lower() / "exp_00" / GRID_CKPT
	return json.load(gzip.open(p))


def fitness_scores(eid: int) -> list[tuple[int, float, float]]:
	"""(position, ce, fitness_score) of the grid's final (population) iteration."""
	c = ro()
	it = c.execute("""select i.id from iterations i join genome_evaluations ge on ge.iteration_id=i.id
		where i.experiment_id=? group by i.id having count(ge.id)>1 order by i.iteration_num desc limit 1""", (eid,)).fetchone()
	if not it:
		return []
	return c.execute("select position, ce, fitness_score from genome_evaluations where iteration_id=? order by position", (it[0],)).fetchall()


def search_side(ckpt: dict, eid: int) -> list[dict]:
	pop = ckpt["phase_result"]["final_population"]
	pm = ckpt["population_metrics"]
	fs = {p: (ce, f) for p, ce, f in fitness_scores(eid)}
	out = []
	for i, (g, m) in enumerate(zip(pop, pm)):
		ce_db, fit = fs.get(i, (None, None))
		if ce_db is not None and abs(ce_db - m["ce"]) > 1e-9:
			fit = None  # position/ce mismatch -> do not trust the match
		b = g["bits_per_neuron"]
		out.append(dict(idx=i, neurons=len(b), bits=sum(b) / len(b), kf_ce=m["ce"], kf_acc=m["acc"],
			kf_f1=m.get("f1"), kf_fpr=m.get("fpr"), fitness=fit))
	return out


def rescore(worker, fid: int, eid: int, ckpt: dict, rows: list[dict], limit: int | None) -> None:
	from wnn.ram.experiments.experiment import Experiment
	from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
	from wnn.ram.metrics import GenomeType
	flow_data = worker.client.get_flow(fid)
	params = flow_data["config"]["params"]
	worker._apply_env_overrides(params, fid)
	flow = worker._build_flow(flow_data, fid, flow_data["name"], params)
	exp_ids = [e["id"] for e in worker.client.list_flow_experiments(fid)]
	cfg = flow.config.experiments[exp_ids.index(eid)]
	cap = CaptureClient()
	exp = Experiment(config=cfg, evaluator=flow.evaluator, logger=lambda m: None, checkpoint_dir=None,
		dashboard_client=cap, experiment_id=eid, tracker=None, flow_id=fid, shutdown_check=None,
		full_evaluator=flow.full_evaluator, validation_scope=flow.config.validation_scope)
	pop = ckpt["phase_result"]["final_population"]
	for r in rows[:limit] if limit else rows:
		d = pop[r["idx"]]
		g = ClusterGenome(bits_per_neuron=d["bits_per_neuron"], neurons_per_cluster=d["neurons_per_cluster"],
			connections=d.get("connections"))
		t0 = time.time()
		exp.revalidate_genome(g, GenomeType.BEST_FITNESS, fid)
		got = cap.rows.pop()
		r.update(ho_ce=got["ce"], ho_modes={k: {kk: v[kk] for kk in ("f1", "fpr", "acc")} for k, v in got["tm"].items()
			if isinstance(v, dict) and "f1" in v}, secs=time.time() - t0)
		print(f"[ids25] flow {fid} genome {r['idx']} {r['neurons']}n x {r['bits']:.1f}b kfF1 {r['kf_f1']:.4f} "
			f"hoF1(val_cal) {r['ho_modes']['val_cal']['f1']:.4f} ({r['secs']:.1f}s)", flush=True)


def main() -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("--flow", type=int, required=True)
	ap.add_argument("--out", required=True)
	ap.add_argument("--limit", type=int, default=None)
	ap.add_argument("--url", default="https://localhost:3000")
	a = ap.parse_args()
	from wnn.accel import require_accel
	require_accel()
	eid, name = grid_experiment(a.flow)
	ckpt = load_checkpoint(name)
	rows = search_side(ckpt, eid)
	worker = make_worker(a.url)
	t0 = time.time()
	rescore(worker, a.flow, eid, ckpt, rows, a.limit)  # env overrides are process-local; process exits after
	Path(a.out).parent.mkdir(parents=True, exist_ok=True)
	json.dump(dict(flow=a.flow, name=name, grid_exp=eid, wall_s=time.time() - t0,
		genomes=[r for r in rows if "ho_modes" in r]), open(a.out, "w"), indent=1)
	print(f"[ids25] flow {a.flow} done {sum('ho_modes' in r for r in rows)} genomes in {time.time() - t0:.0f}s -> {a.out}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
