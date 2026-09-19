"""Re-run every completed QSR flow on the ABI-13 worker wheel (the fixed OI trainer).

WHY (16/09/2026, memory: project_qsr_legacy_trainer_confound). The order-independent
(OI) training gates keyed on the literal memory_mode == QUAD_WEIGHTED at 5-6 sites, so
QSR (mode 4) silently trained on the LEGACY clamped-nudge path — z=1 GPU chunking (~12x
slower) and OI_INITIAL slot defaults one nudge below QUAD-legacy semantics. Every
QSR-vs-QUAD comparison banked so far compares two TRAINERS, not two decodes. The fix
(dc801622: every gate asks CellMode::uses_oi_counters) is worker ABI 13, installed at the
worker-idle swap on 19/09/2026 11:40 EDT. QUAD_WEIGHTED results are bit-identical, so
only the QSR arms need re-running.

SELECTION is by the flow's CONFIG (params.memory_mode == "QSR"), not by name: the confound
is in the trainer, so every QSR run is affected regardless of what the GA chose. Sources
that already have a `-w64fix` clone queued are skipped — that clone will run on ABI 13
as it is. IDSX (the superseded z-score cohort, replaced by IDSXD) is excluded by default.

ORDER MATTERS: the worker's admit() takes min(id) among status='queued', so creation
order IS execution order. Clones are emitted ROUND-ROBIN across arms (seed-major within
each arm, i.e. source-id order) so stopping the queue at any point leaves every arm with
roughly equal n (memory: feedback_sweeps_always_interleave).

Sources are LEFT AS THEY ARE — never deleted, never edited. Per CLAUDE.md Rule 2 the
clones are created through POST /api/flows, never by direct SQL insert, and every clone
is verified afterwards (2 experiments, status queued, params byte-identical).

Usage:
    python scripts/queue_abi13_qsr_reruns.py --arms IDSXD-unswr-qsr,SP100-unswr-qsr,\
SP-unswt-ablqsr,SP-unswr-ablqsr,SP-cicids-ablqsr,SP-ciciot-ablqsr --expect 94 [--dry-run]
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
SUFFIX = "-abi13"
SKIP_IF_CLONED = ("-w64fix",)
EXPERIMENTS = [
	{"name": "Grid Search (neurons x bits)", "experiment_type": "grid_search", "phase_type": "grid_search"},
	{"name": "GA Neurons", "experiment_type": "ga", "phase_type": "ga_neurons"},
]
POST_TIMEOUT = 60      # the dashboard reads slowly while the worker trains
FLIP_RETRIES = 3


def ro():
	return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


def arg(flag: str, default=None):
	for i, a in enumerate(sys.argv):
		if a == flag and i + 1 < len(sys.argv):
			return sys.argv[i + 1]
	return default


def qsr_sources(con, arm: str) -> list:
	"""Completed flows in `arm` whose CONFIG says memory_mode == QSR."""
	rows = con.execute(
		"""SELECT id, name, config_json FROM flows
		   WHERE name LIKE ? AND status = 'completed'
		     AND json_extract(config_json, '$.params.memory_mode') = 'QSR'
		   ORDER BY id""", (f"{arm}%",)).fetchall()
	return [(fid, name, cfg) for fid, name, cfg in rows]


def already_cloned(con) -> set:
	"""Source names that already have a rerun clone (any known suffix, or ours)."""
	names = set()
	for suffix in SKIP_IF_CLONED + (SUFFIX,):
		for (n,) in con.execute("SELECT name FROM flows WHERE name LIKE ?", (f"%{suffix}",)):
			names.add(n[:-len(suffix)])
	return names


def interleave(per_arm: dict) -> list:
	"""Round-robin across arms so an early stop leaves every arm with data."""
	out, queues = [], {a: list(v) for a, v in per_arm.items()}
	while any(queues.values()):
		for arm in per_arm:                     # stable arm order = the --arms order
			if queues[arm]:
				out.append((arm, queues[arm].pop(0)))
	return out


def post_with_retry(url: str, body: dict):
	"""The dashboard read-times-out while the worker trains; retry rather than
	abandon a half-created cohort."""
	last = None
	for _ in range(FLIP_RETRIES):
		try:
			return requests.post(url, json=body, verify=False, timeout=POST_TIMEOUT)
		except requests.exceptions.RequestException as e:
			last = e
			time.sleep(5)
	raise last


def describe(src_id: int, src_name: str) -> str:
	return (f"RERUN of flow {src_id} ({src_name}) on the ABI-13 worker wheel: the source "
	        f"trained QSR on the LEGACY clamped-nudge path because the OI trainer gates keyed "
	        f"on the QUAD_WEIGHTED literal (project_qsr_legacy_trainer_confound, fix dc801622) "
	        f"— its QSR-vs-QUAD comparison confounds trainer with decode. Params byte-identical "
	        f"to the source. Created 19/09/2026; runs behind the queue at creation time (FIFO by id).")


def collect(con, arms: list) -> list:
	cloned = already_cloned(con)
	per_arm = {}
	for arm in arms:
		src = qsr_sources(con, arm)
		keep = [s for s in src if s[1] not in cloned]
		per_arm[arm] = keep
		print(f"  {arm:<22} {len(keep):>3} of {len(src):>3} completed QSR flows "
		      f"({len(src) - len(keep)} already have a rerun clone)")
	return interleave(per_arm)


def verify(created: list) -> int:
	con, bad = ro(), 0
	for fid, name, src_params in created:
		st = con.execute("SELECT status FROM flows WHERE id=?", (fid,)).fetchone()[0]
		ne = con.execute("SELECT COUNT(*) FROM experiments WHERE flow_id=?", (fid,)).fetchone()[0]
		got = json.loads(con.execute("SELECT config_json FROM flows WHERE id=?",
		                             (fid,)).fetchone()[0])["params"]
		if ne != 2 or st != "queued" or got != src_params:
			drift = {k: (src_params.get(k), got.get(k))
			         for k in set(src_params) | set(got) if src_params.get(k) != got.get(k)}
			print(f"  ! VERIFY FAILED {name}: status={st} exps={ne} param_drift={drift}")
			bad += 1
	con.close()
	return bad


def main() -> int:
	arms = [a for a in (arg("--arms") or "").split(",") if a]
	if not arms:
		print(__doc__)
		return 64
	expect = arg("--expect")

	con = ro()
	print(f"scanning {len(arms)} arm(s) for completed flows with memory_mode=QSR:")
	targets = collect(con, arms)
	con.close()

	if expect and len(targets) != int(expect):
		print(f"REFUSED: expected {expect} rerun targets, found {len(targets)}")
		return 4
	print(f"\n{len(targets)} rerun(s), interleaved across arms:")
	for arm, (fid, name, _) in targets:
		print(f"  {fid:>5}  {name}{SUFFIX}")
	if "--dry-run" in sys.argv:
		print("\nDRY RUN — nothing created.")
		return 0

	created = []
	for _, (fid, name, cfg_json) in targets:
		target_name = name + SUFFIX
		cfg = json.loads(cfg_json)
		params = dict(cfg["params"])                    # byte-identical: nothing changes
		body = {
			"name": target_name,
			"description": describe(fid, name),
			"config": {"template": cfg.get("template", "ids-binary-2-phase"), "params": params},
			"experiments": EXPERIMENTS,
		}
		r = post_with_retry(f"{DASHBOARD}/api/flows", body)
		if r.status_code not in (200, 201):
			print(f"  x FAILED {target_name} ({r.status_code}) {r.text[:200]}")
			return 2
		created.append((r.json()["id"], target_name, params))
		print(f"  + {r.json()['id']:>5}  {target_name}")
		time.sleep(0.2)

	print(f"\nflipping {len(created)} pending -> queued...")
	for fid, name, _ in created:
		try:
			post_with_retry(f"{DASHBOARD}/api/flows/{fid}/restart", {})
		except requests.exceptions.RequestException:
			print(f"  ! {fid} {name}: flip failed after {FLIP_RETRIES} tries — still pending")
		time.sleep(0.5)

	bad = verify(created)
	print(f"ALL {len(created)} VERIFIED" if not bad else f"{bad} flow(s) FAILED verification")
	return 1 if bad else 0


if __name__ == "__main__":
	sys.exit(main())
