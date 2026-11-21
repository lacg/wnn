"""IDS73b — supervised cut E on the PRODUCTION unswt recipe, rerun after 7600-7602 failed on the encoder
guard (dedupe on + supervised cut was refused). Luiz 21/11/2026 ~17:45 EST, Plane IDS-73: three arms per seed
r20421-23, all from the IDS-56 ms01-encon control row (7492/7496/7500):
  1. control: dedupe on, quantile cuts  = 7492/7496/7500 themselves (NOT re-flown);
  2. Eoff:    dedupe OFF + ids_cut_method=info_gain_mc (== encoff control 7491/7495/7499 + cut, asserted);
  3. Eon:     dedupe on  + ids_cut_method=info_gain_mc, with the cut-then-dedupe fix (thermometer.py:
              supervised cuts fitted at the full width budget, then dedupe on the final cuts).
Every body is dry-encoded through ThermometerEncoder before anything is POSTed (the 7600-7602 lesson: the
param registry accepted a combination the encoder refused). Run with PYTHONPATH pointing at the code the
worker will run.

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
ENCOFF = {20421: 7491, 20422: 7495, 20423: 7499}
ARMS = (("Eoff", "off"), ("Eon", "on"))  # (arm, ids_encoding_dedupe); both add info_gain_mc
METHOD = "info_gain_mc"
WORKER_ABI = 29
PREFIX = "IDS73b-unswt-BINARY-2m-max250-xo_blend_neuron-16bWb-rate0.25-ms01-cut{arm}"
CONTROL_PREFIX = "IDS56-unswt-BINARY-2m-max250-xo_blend_neuron-16bWb-rate0.25-ms01-encon"
BODIES = ROOT / "docs" / "ids73b_flows.json"


def control(db, seed: int, flow_id: int) -> tuple[dict, str]:
	"""The production unswt control for one seed, asserted to be the recipe IDS73 clones."""
	name, cfg = db.execute("select name, config_json from flows where id=?", (flow_id,)).fetchone()
	cfg = json.loads(cfg)
	p = cfg["params"]
	assert name == f"{CONTROL_PREFIX}-r{seed}" and p["seed"] == seed, name
	assert p["ids_split"] == "temporal_3way" and p["ids_k_folds"] == 5 and p["ids_kfold_per_gen"] == 5
	assert p["max_bits"] == 34 and p["ids_n_bits"] == 16 and p["ids_encoding_dedupe"] == "on"
	assert "ids_cut_method" not in p and not p.get("ids_streaming", False)
	return p, cfg["template"]


def encoff_matches(db, seed: int, base: dict) -> None:
	"""The encoff control is the production control with dedupe off and nothing else."""
	cfg = json.loads(db.execute("select config_json from flows where id=?", (ENCOFF[seed],)).fetchone()[0])
	assert sorted(diff(base, cfg["params"])) == ["ids_encoding_dedupe"], ENCOFF[seed]


def dry_encode(params: dict) -> None:
	"""Build + fit the encoder exactly as the worker resolves these params (raises on a refused combo)."""
	import numpy as np
	import pandas as pd
	from wnn.ram.experiments.ids_param_resolution import resolve_cut_method, resolve_encoding_dedupe
	from wnn.representations.thermometer import ThermometerEncoder
	rng = np.random.default_rng(0)
	y = rng.integers(0, 2, 2000)
	df = pd.DataFrame({"x": rng.normal(size=2000) + y, "x2": rng.gamma(2.0, 3.0, 2000)})
	ThermometerEncoder(n_bits=params["ids_n_bits"], dedupe=resolve_encoding_dedupe(params),
		cut_method=resolve_cut_method(params)).fit(df, y_binary=y, y_multi=y)


def body(db, arm: str, dedupe: str, seed: int) -> dict:
	from wnn.ram.experiments.ids_param_resolution import resolve_cut_method
	base, template = control(db, seed, CONTROLS[seed])
	encoff_matches(db, seed, base)
	params = {**base, "ids_cut_method": METHOD, "ids_encoding_dedupe": dedupe}
	want = ["ids_cut_method"] + (["ids_encoding_dedupe"] if dedupe == "off" else [])
	assert sorted(diff(base, params)) == sorted(want)
	assert resolve_cut_method(params) == METHOD
	Q.check(f"IDS73b-{arm}", params)
	dry_encode(params)
	twin = f"{CONTROLS[seed]} with ids_encoding_dedupe=off (= {ENCOFF[seed]})" if dedupe == "off" else str(CONTROLS[seed])
	desc = (f"IDS73b arm {arm} (Plane IDS-73): ids_cut_method={METHOD}, ids_encoding_dedupe={dedupe} on the production "
		f"unswt recipe; clone of {twin} (r{seed}) + the cut. Control = {CONTROLS[seed]}. Worker ABI {WORKER_ABI}.")
	return {"name": f"{PREFIX.format(arm=arm)}-r{seed}", "description": desc,
		"config": {"template": template, "params": params}, "experiments": Q.EXPS}


def bodies(db) -> list[dict]:
	return [body(db, arm, dedupe, seed) for arm, dedupe in ARMS for seed in CONTROLS]


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
		print(f["id"], b["name"], "dedupe=" + b["config"]["params"]["ids_encoding_dedupe"], "verified + queued")


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
			p = b["config"]["params"]
			print("[dry]", b["name"], p["ids_cut_method"], "dedupe=" + p["ids_encoding_dedupe"], len(b["experiments"]), "experiments")
		return
	queue(db, planned)


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	g = ap.add_mutually_exclusive_group()
	g.add_argument("--write-bodies", action="store_true")
	g.add_argument("--dry-run", action="store_true")
	a = ap.parse_args()
	main("write" if a.write_bodies else "dry" if a.dry_run else "queue")
