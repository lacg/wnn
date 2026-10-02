#!/usr/bin/env python3
"""Requeue the 94 paused QSR `-abi13` reruns (flows 6264-6357) once IDS-20 (IDSAGG-*) drains.

The worker admits the LOWEST queued flow id first, and the QSR ids (6264-6357) are below
IDSAGG's (6380-6419) — requeueing them early would jump the cohort. So this waits until no
IDSAGG flow is queued or running, then PATCHes each still-paused `-abi13` flow to `queued`
via the dashboard API (never direct SQL), verifies, and exits. Idempotent: only touches
flows that are still `paused`.

Launch detached:
  python3 scripts/detach_launch.py /private/tmp/queue_qsr_abi13.launch.log /Users/lacg/wnn -- \
    python3 scripts/queue_qsr_abi13_after_idsagg.py
Log: /private/tmp/queue_qsr_abi13.log
"""
import json
import sqlite3
import ssl
import time
import urllib.request
from datetime import datetime, timezone

DB_URI = "file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro"
API = "https://127.0.0.1:3000/api/flows"
LOG = "/private/tmp/queue_qsr_abi13.log"
POLL_S = 300
QSR_LIKE = "%-abi13%"
GATE_LIKE = "IDSAGG%"


def log(msg: str) -> None:
	line = f"[qsr13] {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} {msg}"
	print(line, flush=True)
	with open(LOG, "a") as f:
		f.write(line + "\n")


def query(sql: str, args: tuple = ()) -> list:
	con = sqlite3.connect(DB_URI, uri=True, timeout=30)
	try:
		return con.execute(sql, args).fetchall()
	finally:
		con.close()


def gate_open() -> tuple[bool, int]:
	(n,), = query("select count(*) from flows where name like ? and status in ('queued','running')", (GATE_LIKE,))
	return n == 0, n


def paused_qsr_ids() -> list[int]:
	return [r[0] for r in query("select id from flows where name like ? and status='paused' order by id", (QSR_LIKE,))]


def patch_queued(flow_id: int, ctx: ssl.SSLContext) -> bool:
	body = json.dumps({"status": "queued"}).encode()
	req = urllib.request.Request(f"{API}/{flow_id}", data=body, method="PATCH", headers={"Content-Type": "application/json"})
	for attempt in range(10):
		try:
			with urllib.request.urlopen(req, context=ctx, timeout=30) as r:
				if r.status == 200:
					return True
		except Exception as e:
			log(f"PATCH {flow_id} attempt {attempt + 1} failed: {e}")
		time.sleep(10)
	return False


def wait_for_gate() -> None:
	beat = 0
	while True:
		ok, n = gate_open()
		if ok:
			return
		if beat % 12 == 0:
			log(f"waiting — {n} IDSAGG flow(s) still queued/running; {len(paused_qsr_ids())} QSR -abi13 paused")
		beat += 1
		time.sleep(POLL_S)


def requeue_all() -> None:
	ctx = ssl.create_default_context()
	ctx.check_hostname = False
	ctx.verify_mode = ssl.CERT_NONE
	ids = paused_qsr_ids()
	log(f"IDSAGG drained — requeueing {len(ids)} QSR -abi13 flows ({ids[0] if ids else '-'}..{ids[-1] if ids else '-'})")
	failed = [fid for fid in ids if not patch_queued(fid, ctx)]
	(nq,), = query("select count(*) from flows where name like ? and status='queued'", (QSR_LIKE,))
	log(f"DONE — {nq} QSR -abi13 now queued; PATCH failures: {failed or 'none'}")


def main() -> None:
	log("ARMED — will requeue paused QSR -abi13 reruns once no IDSAGG flow is queued/running")
	wait_for_gate()
	requeue_all()


if __name__ == "__main__":
	main()
