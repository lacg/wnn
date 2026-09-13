"""Clone a banked cohort under a new prefix, overriding named params.

Sibling of scripts/queue_desirability_clone.py (which changes exactly the
aggregation + anchor). This one clones every flow matching a source prefix
VERBATIM — template, params, the 2-phase experiment list — and applies only the
`--set key=value` overrides given on the command line. First use (13/09/2026):
the UNSW TEMPORAL twin of the IDSXD unswr sweep, because IDSXD had no temporal
arm and arm winners do not transfer across datasets (CE20 won on unswt-16b only).

Usage:
    python scripts/queue_param_clone.py --source IDSXD-unswr- --target IDSXD-unswt- \
        --set ids_split=temporal_3way [--seeds 20403,20404,20405] [--only <substr>] \
        [--dry-run] [--expect N]

Values are parsed as JSON when they parse (numbers, booleans), else kept as
strings. Existing target names are skipped, so the script is idempotent.
Source flows are LEFT AS THEY ARE regardless of their status.
"""
import json
import sqlite3
import sys
import time

import requests
import urllib3

urllib3.disable_warnings()

DASHBOARD = "https://localhost:3000"
DB_PATH = "/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db"
EXPERIMENTS = [
	{"name": "Grid Search (neurons x bits)", "experiment_type": "grid_search", "phase_type": "grid_search"},
	{"name": "GA Neurons", "experiment_type": "ga", "phase_type": "ga_neurons"},
]
POST_TIMEOUT = 60
FLIP_RETRIES = 3


def ro():
	return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


def args_of(flag: str) -> list[str]:
	return [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == flag and i + 1 < len(sys.argv)]


def arg(flag: str, default=None):
	vals = args_of(flag)
	return vals[-1] if vals else default


def parse_value(raw: str):
	try:
		return json.loads(raw)
	except json.JSONDecodeError:
		return raw


def overrides() -> dict:
	out = {}
	for kv in args_of("--set"):
		if "=" not in kv:
			raise SystemExit(f"--set expects key=value, got {kv!r}")
		k, v = kv.split("=", 1)
		out[k] = parse_value(v)
	return out


def post_with_retry(url: str, body: dict):
	last = None
	for _ in range(FLIP_RETRIES):
		try:
			return requests.post(url, json=body, verify=False, timeout=POST_TIMEOUT)
		except requests.exceptions.RequestException as e:
			last = e
			time.sleep(5)
	raise last


def select_sources(source: str, target: str, seeds: set[int] | None, only: str | None):
	con = ro()
	rows = con.execute("SELECT name, config_json FROM flows WHERE name LIKE ? ORDER BY name",
	                   (f"{source}%",)).fetchall()
	# a cancelled target does not count as existing — re-queuing after a cancel is the point
	existing = {r[0] for r in con.execute(
		"SELECT name FROM flows WHERE name LIKE ? AND status != 'cancelled'", (f"{target}%",))}
	con.close()
	# one source per name: the same name can exist twice only via a rerun; keep the first
	seen, sources = set(), []
	for name, cj in rows:
		if name in seen:
			continue
		seen.add(name)
		p = json.loads(cj)["params"]
		if seeds is not None and int(p.get("seed", -1)) not in seeds:
			continue
		if only and only not in name:
			continue
		sources.append((name, cj))
	# SEED-MAJOR: round 1 = one flow of every arm, then round 2 ... — the worker is
	# FIFO min-id, so creation order IS run order (feedback_sweeps_always_interleave).
	sources.sort(key=lambda r: (int(json.loads(r[1])["params"].get("seed", 0)), r[0]))
	return sources, existing


def clone_one(sname: str, cj: str, source: str, target: str, sets: dict):
	cfg = json.loads(cj)
	p = dict(cfg["params"])
	p.update(sets)
	name = sname.replace(source, target, 1)
	body = {
		"name": name,
		"description": (f"Param clone of {sname}: " + ", ".join(f"{k} -> {v}" for k, v in sets.items())
		                + ". All other params byte-identical to the source."),
		"config": {"template": cfg.get("template", "ids-binary-2-phase"), "params": p},
		"experiments": EXPERIMENTS,
	}
	r = post_with_retry(f"{DASHBOARD}/api/flows", body)
	if r.status_code not in (200, 201):
		raise SystemExit(f"  x FAILED {name} ({r.status_code}) {r.text[:200]}")
	return r.json()["id"], name


def verify(created: list, sets: dict) -> int:
	con = ro()
	bad = 0
	for fid, name in created:
		st = con.execute("SELECT status FROM flows WHERE id=?", (fid,)).fetchone()[0]
		ne = con.execute("SELECT COUNT(*) FROM experiments WHERE flow_id=?", (fid,)).fetchone()[0]
		q = json.loads(con.execute("SELECT config_json FROM flows WHERE id=?", (fid,)).fetchone()[0])["params"]
		ok = ne == 2 and st == "queued" and all(q.get(k) == v for k, v in sets.items())
		if not ok:
			print(f"  ! VERIFY FAILED {name}: status={st} exps={ne} " + " ".join(f"{k}={q.get(k)}" for k in sets))
			bad += 1
	con.close()
	return bad


def main() -> int:
	source, target, sets = arg("--source"), arg("--target"), overrides()
	if not source or not target or not sets:
		print(__doc__)
		return 64
	seeds = {int(s) for s in arg("--seeds", "").split(",") if s} or None
	sources, existing = select_sources(source, target, seeds, arg("--only"))
	expect = arg("--expect")
	if not sources:
		print(f"REFUSED: no flows match {source!r}")
		return 4
	if expect and len(sources) != int(expect):
		print(f"REFUSED: expected {expect} source flows, found {len(sources)}")
		return 4

	print(f"{target} cohort: {len(sources)} flow(s) from {source}, overrides {sets}")
	if "--dry-run" in sys.argv:
		for sname, cj in sources:
			p = json.loads(cj)["params"]
			print(f"  would clone {sname}: {p.get('ids_dataset')} {p.get('ids_split')} "
			      f"{p.get('ids_n_bits')}b seed {p.get('seed')} mode {p.get('memory_mode', 'QUAD')}")
		print("DRY RUN — nothing created.")
		return 0

	created = []
	for sname, cj in sources:
		name = sname.replace(source, target, 1)
		if name in existing:
			print(f"  = exists {name}")
			continue
		fid, name = clone_one(sname, cj, source, target, sets)
		created.append((fid, name))
		print(f"  + {fid:>5}  {name}")
		time.sleep(0.2)

	print(f"flipping {len(created)} pending -> queued...")
	for fid, name in created:
		try:
			post_with_retry(f"{DASHBOARD}/api/flows/{fid}/restart", {})
		except requests.exceptions.RequestException:
			print(f"  ! {fid} {name}: flip failed after {FLIP_RETRIES} tries — still pending")
		time.sleep(0.5)

	bad = verify(created, sets)
	print(f"ALL {len(created)} VERIFIED" if not bad else f"{bad} flows FAILED verification")
	return 1 if bad else 0


if __name__ == "__main__":
	sys.exit(main())
