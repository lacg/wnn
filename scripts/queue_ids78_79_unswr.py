"""IDS78 (GA Connections) + IDS79 (max_neurons 500) on unswr, paired against the IDS-70 E control
(flows 7591/7592/7593, seeds r20421-23; Luiz 23/11/2026 ~20:15 EST). Prereg docs/ids78_76_79_unswr_prereg.md.

IDS78: ONE experiment, GA Connections, SEEDED from the control's own GA Neurons experiment_end checkpoint
(8426/8428/8490 = the final GA Neurons population, all 80b, connections in [0, 1104)). Grid + GA Neurons are
NOT re-flown: that would repeat the control recipe on seeds already flown. Params = the control's, verbatim.
IDS79: the control's params verbatim with max_neurons 250 -> 500 (the default grid becomes
[5, 100, 200, 300, 400, 500]), same 2 experiments.
IDS76 (fold rotation) is NOT here: no ids_fold_rotation param exists on ABI 29 (needs code + a wheel swap).

  --write-bodies   build the bodies from the read-only DB -> docs/ids78_79_flows.json (no API call)
  --dry-run        re-derive, check against the file, print the plan (no API call)
  --queue ARM      POST the bodies of ARM (ids78 | ids79) in file order, verify, PATCH queued"""
import argparse
import json
import sqlite3
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import queue_ids56_vote_share_ab as Q  # noqa: E402  (DB, API, CTX, EXPS, call, check)
from queue_ids60_unswt_ce20 import diff  # noqa: E402

CONTROLS = {20421: 7591, 20422: 7592, 20423: 7593}
WORKER_ABI = 29
CONTROL_PREFIX = "IDS70-unswr-WbCTRL-desir-BINARY-2m-max250-xo_blend_cluster-rate0.25-rklegacy-nb02-fixb80-encoff-cutE"
NAME78 = "IDS78-unswr-WbCTRL-desir-BINARY-2m-max250-xo_blend_cluster-rate0.25-rklegacy-nb02-fixb80-encoff-cutE-gaconn"
NAME79 = "IDS79-unswr-WbCTRL-desir-BINARY-2m-max500-xo_blend_cluster-rate0.25-rklegacy-nb02-fixb80-encoff-cutE"
PREREG = "docs/ids78_76_79_unswr_prereg.md"
BODIES = ROOT / "docs" / "ids78_79_flows.json"
EXP_GA_CONN = [{"name": "GA Connections", "experiment_type": "ga", "phase_type": "ga_connections",
	"optimize_bits": False, "optimize_neurons": False, "optimize_connections": True}]


def control(db, seed: int, flow_id: int) -> tuple[dict, str]:
	"""The IDS-70 E control for one seed, asserted to be the recipe both arms clone."""
	name, cfg, status = db.execute("select name, config_json, status from flows where id=?", (flow_id,)).fetchone()
	cfg = json.loads(cfg)
	p = cfg["params"]
	assert name == f"{CONTROL_PREFIX}-r{seed}" and p["seed"] == seed and status == "completed", name
	assert p["ids_split"] == "random_3way" and p["ids_k_folds"] == 5 and p["ids_kfold_per_gen"] == 5
	assert p["ids_offspring_folds"] == "all" and p["ids_cut_method"] == "info_gain_mc"
	assert p["min_bits"] == p["max_bits"] == 80 and p["bits_grid"] == [80] and p["max_neurons"] == 250
	phases = [r[0] for r in db.execute(
		"select phase_type from experiments where flow_id=? order by sequence_order", (flow_id,))]
	assert phases == ["grid_search", "ga_neurons"], phases
	return p, cfg["template"]


def ga_neurons_checkpoint(db, flow_id: int) -> int:
	"""The control's GA Neurons experiment_end checkpoint (its final population)."""
	rows = db.execute("""select c.id from checkpoints c join experiments e on e.id = c.experiment_id
		where e.flow_id=? and e.phase_type='ga_neurons' and c.checkpoint_type='experiment_end'""", (flow_id,)).fetchall()
	assert len(rows) == 1, rows
	return rows[0][0]


def body78(db, seed: int) -> dict:
	base, template = control(db, seed, CONTROLS[seed])
	Q.check(f"IDS78-r{seed}", base)
	ckpt = ga_neurons_checkpoint(db, CONTROLS[seed])
	desc = (f"IDS78 (prereg {PREREG}, Plane IDS-78): GA Connections (connections mutated, neurons/bits fixed, "
		f"TRAIN 5-fold CV fitness) seeded from control {CONTROLS[seed]}'s GA Neurons final population "
		f"(checkpoint {ckpt}); params = {CONTROLS[seed]} verbatim. Worker ABI {WORKER_ABI}.")
	return {"name": f"{NAME78}-r{seed}", "description": desc, "seed_checkpoint_id": ckpt,
		"config": {"template": template, "params": dict(base)}, "experiments": EXP_GA_CONN}


def body79(db, seed: int) -> dict:
	base, template = control(db, seed, CONTROLS[seed])
	params = {**base, "max_neurons": 500}
	assert sorted(diff(base, params)) == ["max_neurons"]
	Q.check(f"IDS79-r{seed}", params)
	desc = (f"IDS79 (prereg {PREREG}, Plane IDS-79): max_neurons (per cluster) 250 -> 500, grid "
		f"[5,100,200,300,400,500] x [80]; clone of {CONTROLS[seed]} (r{seed}) with only max_neurons changed. "
		f"Worker ABI {WORKER_ABI}.")
	return {"name": f"{NAME79}-r{seed}", "description": desc,
		"config": {"template": template, "params": params}, "experiments": Q.EXPS}


def bodies(db) -> dict:
	return {"ids78": [body78(db, s) for s in CONTROLS], "ids79": [body79(db, s) for s in CONTROLS]}


def post_once(b: dict) -> dict:
	"""POST without retry: a retried POST whose first attempt landed would create a duplicate flow."""
	req = urllib.request.Request(Q.API, method="POST", headers={"Content-Type": "application/json"},
		data=json.dumps(b).encode())
	return json.load(urllib.request.urlopen(req, context=Q.CTX, timeout=120))


def verify(db, flow_id: int, b: dict) -> None:
	"""Rule 2: experiments present with the right phase types, every param stored, seed checkpoint stored."""
	rows = db.execute("select phase_type from experiments where flow_id=? order by sequence_order",
		(flow_id,)).fetchall()
	want = [e["phase_type"] for e in b["experiments"]]
	if [r[0] for r in rows] != want:
		raise SystemExit(f"flow {flow_id}: experiments {rows}, expected {want}; left pending")
	cfg, seed_ckpt = db.execute("select config_json, seed_checkpoint_id from flows where id=?", (flow_id,)).fetchone()
	stored = json.loads(cfg)["params"]
	sent = b["config"]["params"]
	wrong = {k: [v, stored.get(k)] for k, v in sent.items() if stored.get(k) != v}
	if wrong:
		raise SystemExit(f"flow {flow_id}: stored params differ from the body: {wrong}; left pending")
	if seed_ckpt != b.get("seed_checkpoint_id"):
		raise SystemExit(f"flow {flow_id}: seed_checkpoint_id {seed_ckpt} != {b.get('seed_checkpoint_id')}; left pending")
	extra = sorted(set(stored) - set(sent))
	if extra:
		print(f"  note: flow {flow_id} stored extra keys added by the dashboard: {extra}")


def queue(db, planned: list[dict]) -> None:
	have = {r[0] for r in db.execute("select name from flows")}
	for b in planned:
		if b["name"] in have:
			print("exists, skipped", b["name"])
			continue
		f = post_once(b)
		verify(db, f["id"], b)
		Q.call("PATCH", f"{Q.API}/{f['id']}", {"status": "queued"})
		print(f["id"], b["name"], "verified + queued")


def main(mode: str, arm: str | None) -> None:
	import ram_accelerator
	if int(ram_accelerator.ABI_VERSION) != WORKER_ABI:
		raise SystemExit(f"installed ABI {ram_accelerator.ABI_VERSION} != {WORKER_ABI}")
	db = sqlite3.connect(Q.DB, uri=True)
	planned = bodies(db)
	if mode == "write":
		BODIES.write_text(json.dumps(planned, indent=1) + "\n")
		print("wrote", BODIES, {k: len(v) for k, v in planned.items()})
		return
	if json.loads(BODIES.read_text()) != planned:
		raise SystemExit(f"{BODIES} differs from the bodies re-derived from the DB; refusing")
	if mode == "dry":
		for a, bs in planned.items():
			for b in bs:
				print("[dry]", a, b["name"], "seed_ckpt", b.get("seed_checkpoint_id"),
					"max_neurons", b["config"]["params"]["max_neurons"], [e["phase_type"] for e in b["experiments"]])
		return
	queue(db, planned[arm])


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	g = ap.add_mutually_exclusive_group()
	g.add_argument("--write-bodies", action="store_true")
	g.add_argument("--dry-run", action="store_true")
	g.add_argument("--queue", choices=("ids78", "ids79"))
	a = ap.parse_args()
	main("write" if a.write_bodies else "dry" if a.dry_run else "queue", a.queue)
