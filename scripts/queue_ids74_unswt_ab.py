"""IDS74 — one-setting A/Bs on the PRODUCTION unswt recipe (Plane IDS-74; Luiz 21/11/2026 ~17:20 EST). Per seed
r20421-23: clone the IDS-56 ms01-encon control (7492/7496/7500) VERBATIM and change exactly the arm's keys:
  D   = fitness_aggregation desirability + fitness_zrank_clamp 3.0 (as IDSD 26/08)
  BC  = balance_classes false
  V02 = ids_vote_share 0.02
Controls are NOT re-flown. Arm-major order D, BC, V02. Prereg docs/ids74_unswt_ab_prereg.md.

  --write-bodies / --dry-run / (no flag) as scripts/queue_ids70_cut_method.py."""
import argparse
import json
import sqlite3
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import queue_ids56_vote_share_ab as Q  # noqa: E402  (DB, API, EXPS, call, check)
from queue_ids60_unswt_ce20 import diff  # noqa: E402

CONTROLS = {20421: 7492, 20422: 7496, 20423: 7500}
ARMS = (("D", {"fitness_aggregation": "desirability", "fitness_zrank_clamp": 3.0}),
	("BC", {"balance_classes": False}), ("V02", {"ids_vote_share": 0.02}))
WORKER_ABI = 29
PREFIX = "IDS74-unswt-BINARY-2m-max250-xo_blend_neuron-16bWb-rate0.25-ms01-encon-{arm}"
CONTROL_PREFIX = "IDS56-unswt-BINARY-2m-max250-xo_blend_neuron-16bWb-rate0.25-ms01-encon"
PREREG = "docs/ids74_unswt_ab_prereg.md"
BODIES = ROOT / "docs" / "ids74_flows.json"


def control(db, seed: int, flow_id: int) -> tuple[dict, str]:
	"""The production unswt control for one seed, asserted to be the recipe IDS74 clones."""
	name, cfg = db.execute("select name, config_json from flows where id=?", (flow_id,)).fetchone()
	cfg = json.loads(cfg)
	p = cfg["params"]
	assert name == f"{CONTROL_PREFIX}-r{seed}" and p["seed"] == seed, name
	assert p["ids_split"] == "temporal_3way" and p["ids_k_folds"] == 5 and p["ids_kfold_per_gen"] == 5
	assert p["max_bits"] == 34 and p["ids_n_bits"] == 16 and p["ids_encoding_dedupe"] == "on"
	assert "ids_cut_method" not in p and not p.get("ids_streaming", False)
	return p, cfg["template"]


def body(db, arm: str, change: dict, seed: int) -> dict:
	base, template = control(db, seed, CONTROLS[seed])
	assert base.get("fitness_aggregation") is None and base["balance_classes"] is True and base["ids_vote_share"] == 0.1
	params = {**base, **change}
	assert sorted(diff(base, params)) == sorted(change), diff(base, params)
	Q.check(f"IDS74-{arm}", params)
	desc = (f"IDS74 arm {arm} (prereg {PREREG}, Plane IDS-74): {json.dumps(change)} on the production unswt recipe; "
		f"clone of {CONTROLS[seed]} (r{seed}) with only those keys changed. Worker ABI {WORKER_ABI}.")
	return {"name": f"{PREFIX.format(arm=arm)}-r{seed}", "description": desc,
		"config": {"template": template, "params": params}, "experiments": Q.EXPS}


def bodies(db) -> list[dict]:
	return [body(db, arm, change, seed) for arm, change in ARMS for seed in CONTROLS]


def post_once(b: dict) -> dict:
	"""POST without retry: a retried POST whose first attempt landed would create a duplicate flow."""
	req = urllib.request.Request(Q.API, method="POST", headers={"Content-Type": "application/json"},
		data=json.dumps(b).encode())
	return json.load(urllib.request.urlopen(req, context=Q.CTX, timeout=120))


def verify(db, flow_id: int, b: dict) -> None:
	"""Rule 2: the flow has its experiments and stored every param we sent (incl. the arm keys)."""
	n = db.execute("select count(*) from experiments where flow_id=?", (flow_id,)).fetchone()[0]
	if n != len(b["experiments"]):
		raise SystemExit(f"flow {flow_id}: {n} experiments, expected {len(b['experiments'])}; left pending")
	stored = json.loads(db.execute("select config_json from flows where id=?", (flow_id,)).fetchone()[0])["params"]
	sent = b["config"]["params"]
	wrong = {k: [v, stored.get(k)] for k, v in sent.items() if stored.get(k) != v}
	if wrong:
		raise SystemExit(f"flow {flow_id}: stored params differ from the body: {wrong}; left pending")
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


def main(mode: str) -> None:
	import ram_accelerator
	if int(ram_accelerator.ABI_VERSION) != WORKER_ABI:
		raise SystemExit(f"installed ABI {ram_accelerator.ABI_VERSION} != {WORKER_ABI}")
	db = sqlite3.connect(Q.DB, uri=True)
	planned = bodies(db)
	if mode == "write":
		BODIES.write_text(json.dumps(planned, indent=1) + "\n")
		print("wrote", BODIES, len(planned), "bodies")
		return
	if json.loads(BODIES.read_text()) != planned:
		raise SystemExit(f"{BODIES} differs from the bodies re-derived from the DB; refusing")
	if mode == "dry":
		for b in planned:
			print("[dry]", b["name"], len(b["experiments"]), "experiments")
		return
	queue(db, planned)


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	g = ap.add_mutually_exclusive_group()
	g.add_argument("--write-bodies", action="store_true")
	g.add_argument("--dry-run", action="store_true")
	a = ap.parse_args()
	main("write" if a.write_bodies else "dry" if a.dry_run else "queue")
