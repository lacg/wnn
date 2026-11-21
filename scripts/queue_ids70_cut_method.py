"""IDS70 — supervised thermometer cut points on the unswr fixed-80b recipe (Plane IDS-70; Luiz approved 20/11/2026;
prereg docs/ids70_cut_method_prereg.md). Per seed r20421-23: clone the IDS62 control (7585/7586/7587) VERBATIM and add
exactly `ids_cut_method` (arm E = "info_gain_mc", arm B = "xgb_splits"). Controls are NOT re-flown.

  --write-bodies   build the 6 POST /api/flows bodies from the read-only DB -> docs/ids70_flows.json (no API call)
  --dry-run        re-derive the bodies, check them against docs/ids70_flows.json, print the plan (no API call)
  (no flag)        POST the bodies in file order, verify each flow (experiments + stored params), PATCH queued

Order = ARM-major: E r20421, E r20422, E r20423, then B r20421-23 (FIFO = lowest id first). E is the pre-check's
locked winner (the primary hypothesis), so its n=3 verdict lands after ~3 flows (~34 h) instead of ~5 (seed-major).
Asserts: the diff vs the control is exactly {ids_cut_method}; every key in KNOWN_PARAMS; resolvers accept the value;
worker ABI 29 installed. The worker-code gate (running worker started after ids_cut_method landed) lives in
logs/ids/queue_ids70.sh, which is the entry point."""
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

CONTROLS = {20421: 7585, 20422: 7586, 20423: 7587}
ARMS = (("E", "info_gain_mc"), ("B", "xgb_splits"))
WORKER_ABI = 29
PREFIX = "IDS70-unswr-WbCTRL-desir-BINARY-2m-max250-xo_blend_cluster-rate0.25-rklegacy-nb02-fixb80-encoff-cut{arm}"
CONTROL_PREFIX = "IDS62-unswr-WbCTRL-desir-BINARY-2m-max250-xo_blend_cluster-rate0.25-rklegacy-nb02-fixb80-encoff"
PREREG = "docs/ids70_cut_method_prereg.md"
BODIES = ROOT / "docs" / "ids70_flows.json"


def control(db, seed: int, flow_id: int) -> tuple[dict, str]:
	"""The IDS62 fixed-80b control for one seed, asserted to be the recipe IDS70 clones."""
	name, cfg = db.execute("select name, config_json from flows where id=?", (flow_id,)).fetchone()
	cfg = json.loads(cfg)
	p = cfg["params"]
	assert name == f"{CONTROL_PREFIX}-r{seed}" and p["seed"] == seed, name
	assert p["ids_split"] == "random_3way" and p["ids_k_folds"] == 5 and p["ids_kfold_per_gen"] == 5
	assert p["min_bits"] == p["max_bits"] == 80 and p["bits_grid"] == [80] and p["ids_encoding_dedupe"] == "off"
	assert "ids_cut_method" not in p and not p.get("ids_streaming", False)
	return p, cfg["template"]


def body(db, arm: str, method: str, seed: int) -> dict:
	from wnn.ram.experiments.ids_param_resolution import resolve_cut_method
	base, template = control(db, seed, CONTROLS[seed])
	params = {**base, "ids_cut_method": method}
	assert sorted(diff(base, params)) == ["ids_cut_method"]
	assert resolve_cut_method(params) == method
	Q.check(f"IDS70-{arm}", params)
	desc = (f"IDS70 arm {arm} (prereg {PREREG}, Plane IDS-70): ids_cut_method={method} on the unswr fixed-80b recipe; "
		f"clone of {CONTROLS[seed]} (r{seed}) with only ids_cut_method added. Worker ABI {WORKER_ABI}.")
	return {"name": f"{PREFIX.format(arm=arm)}-r{seed}", "description": desc,
		"config": {"template": template, "params": params}, "experiments": Q.EXPS}


def bodies(db) -> list[dict]:
	return [body(db, arm, method, seed) for arm, method in ARMS for seed in CONTROLS]


def post_once(b: dict) -> dict:
	"""POST without retry: a retried POST whose first attempt landed would create a duplicate flow."""
	req = urllib.request.Request(Q.API, method="POST", headers={"Content-Type": "application/json"},
		data=json.dumps(b).encode())
	return json.load(urllib.request.urlopen(req, context=Q.CTX, timeout=120))


def verify(db, flow_id: int, b: dict) -> None:
	"""Rule 2: the flow has its experiments and stored every param we sent (incl. ids_cut_method)."""
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
		print(f["id"], b["name"], "ids_cut_method=" + b["config"]["params"]["ids_cut_method"], "verified + queued")


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
			print("[dry]", b["name"], b["config"]["params"]["ids_cut_method"], len(b["experiments"]), "experiments")
		return
	queue(db, planned)


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	g = ap.add_mutually_exclusive_group()
	g.add_argument("--write-bodies", action="store_true")
	g.add_argument("--dry-run", action="store_true")
	a = ap.parse_args()
	main("write" if a.write_bodies else "dry" if a.dry_run else "queue")
