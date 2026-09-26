"""Clone flows into EXPLICITLY named copies with per-row param overrides.

queue_param_clone.py derives the new name from the source name, so it cannot make
FRESH-SEED clones (the "-r<seed>" suffix would lie) or re-flies that keep the seed
but need a new name. This driver takes explicit rows instead and reuses that
script's create / verify path (POST /api/flows with the 2-phase experiment list,
flip to queued, verify status + experiment count + overridden params).

Spec file: a JSON list of {"source": <flow name>, "name": <new name>,
"set": {param: value, ...}, "why": <description suffix>}. Rows are created IN FILE
ORDER — the worker is FIFO min-id, so order the file seed-major
(feedback_sweeps_always_interleave). Existing non-cancelled target names are
skipped, so the script is idempotent. Sources are never modified.

Usage: python scripts/queue_named_clones.py <spec.json> [--dry-run] [--expect N]
"""
import json
import sqlite3
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from queue_param_clone import DASHBOARD, EXPERIMENTS, post_with_retry, ro, verify  # noqa: E402

import requests  # noqa: E402


def source_config(con, name: str) -> tuple[str, dict]:
	"""The newest non-cancelled flow with this name: (template, params)."""
	row = con.execute("""SELECT config_json FROM flows WHERE name=? AND status != 'cancelled'
		ORDER BY id DESC LIMIT 1""", (name,)).fetchone()
	if row is None:
		raise SystemExit(f"REFUSED: source {name!r} not found")
	cfg = json.loads(row[0])
	return cfg.get("template", "ids-binary-2-phase"), dict(cfg["params"])


def existing_names(con) -> set[str]:
	return {r[0] for r in con.execute("SELECT name FROM flows WHERE status != 'cancelled'")}


def create(row: dict, template: str, params: dict) -> tuple[int, str]:
	params.update(row.get("set", {}))
	sets = ", ".join(f"{k} -> {v}" for k, v in row.get("set", {}).items()) or "no param change"
	body = {"name": row["name"],
	        "description": f"Clone of {row['source']}: {sets}. {row.get('why', '')}".strip(),
	        "config": {"template": template, "params": params},
	        "experiments": EXPERIMENTS}
	r = post_with_retry(f"{DASHBOARD}/api/flows", body)
	if r.status_code not in (200, 201):
		raise SystemExit(f"  x FAILED {row['name']} ({r.status_code}) {r.text[:200]}")
	fid = r.json()["id"]
	q = requests.patch(f"{DASHBOARD}/api/flows/{fid}", json={"status": "queued"}, verify=False, timeout=60)
	if q.status_code != 200:
		raise SystemExit(f"  x created {fid} but queue flip failed ({q.status_code})")
	return fid, row["name"]


def main() -> int:
	spec = json.load(open(sys.argv[1]))
	dry = "--dry-run" in sys.argv
	expect = sys.argv[sys.argv.index("--expect") + 1] if "--expect" in sys.argv else None
	con = ro()
	have = existing_names(con)
	todo = [r for r in spec if r["name"] not in have]
	if expect is not None and len(todo) != int(expect):
		print(f"REFUSED: expected {expect} new flows, spec yields {len(todo)}")
		return 4
	plan = [(r, *source_config(con, r["source"])) for r in todo]
	con.close()
	for r, _t, p in plan:
		sets = {k: (p.get(k), v) for k, v in r.get("set", {}).items()}
		print(f"  new  {r['name']:<48} <- {r['source']}  {sets}")
	if dry:
		print(f"DRY RUN — {len(plan)} flows, nothing created.")
		return 0
	created = [create(r, t, p) for r, t, p in plan]
	bad = sum(verify([c], r.get("set", {})) for c, (r, _t, _p) in zip(created, plan))
	print(f"created {len(created)} (ids {created[0][0]}-{created[-1][0]}), verify failures: {bad}")
	return 1 if bad else 0


if __name__ == "__main__":
	sys.exit(main())
