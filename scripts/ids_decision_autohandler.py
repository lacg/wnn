#!/usr/bin/env python3
"""Autonomous overnight handler for the 16b-vs-64b decision → grow winner to 30.

Trigger: flow 4050 (the running 16b r92774) completing.
Decision: compare 4050's best_f1 held-out genome to the current best 64b genome
(4037 = 0.939193). Branch:

  • 4050 F1 <= 64b best  → 64b WINS (expected; 16b tops at 93.75%).
      - cancel 4048 (the last queued 16b)
      - queue fresh-seed 64b-Wb flows until 30 total exist
  • 4050 F1 >  64b best  → 16b WINS (upset).
      - do NOT cancel; wait for 4049 (64b) AND 4048 (16b) to finish (evaluate them)
      - queue fresh-seed 16b-Wb flows until 30 total exist

Naming is automatic: build_flow emits XDS-unsw-random-{N}b-Wb-C35-500n34b-OI-r{seed}
so all 30 share the pattern (no rename needed). 500n34b arch via arch_override.

Self-contained + idempotent: writes a DONE marker so a restart won't double-queue;
recounts live flows before queuing so it never overshoots 30. Runs detached.
"""
import json, sqlite3, secrets, time, sys
from pathlib import Path
import requests
from queue_cross_dataset import build_flow  # same dir on PYTHONPATH

ROOT = Path("/Users/lacg/wnn")
DB = f"file:{ROOT}/db/wnn.db?mode=ro"
API = "https://localhost:3000/api/flows"
TRIGGER_FLOW = 4050
BEST_64B_F1 = 0.939193          # flow 4037, the current champion
CANCEL_IF_64B = 4048            # last queued 16b, cancelled on 64b route
WAIT_IF_16B = [4049, 4048]      # let these finish before growing 16b
TARGET = 30
DS, WEIGHT, ARCH = "unsw-random", "b", (500, 34)
LOG = ROOT / "logs/ids_decision_autohandler.log"
DONE = Path("/tmp/ids_autohandler_done")
POLL = 60


def log(msg):
	line = f"[{time.strftime('%d/%m/%Y %H:%M:%S')}] {msg}"
	print(line, flush=True)
	with open(LOG, "a") as f:
		f.write(line + "\n")


def q(sql, args=()):
	db = sqlite3.connect(DB, uri=True)
	try:
		return db.execute(sql, args).fetchall()
	finally:
		db.close()


def status_of(fid):
	r = q("SELECT status FROM flows WHERE id=?", (fid,))
	return r[0][0] if r else None


def best_f1(fid):
	r = q("SELECT MAX(vs.f1_macro) FROM validation_summaries vs "
	      "JOIN experiments e ON e.id=vs.experiment_id "
	      "WHERE vs.flow_id=? AND e.phase_type='ga_neurons' "
	      "AND vs.genome_type='best_f1' AND vs.validation_point='final'", (fid,))
	return r[0][0] if r and r[0][0] is not None else None


def live_count(thermo):
	"""Flows of this width that count toward the 30 (not cancelled/failed)."""
	r = q("SELECT COUNT(*) FROM flows WHERE "
	      "name LIKE ? AND name NOT LIKE '%OLD%' "
	      "AND status IN ('completed','running','queued','pending')",
	      (f"XDS-unsw-random-{thermo}b-Wb-C35-500n34b-OI-r%",))
	return r[0][0]


def existing_seeds():
	rows = q("SELECT DISTINCT json_extract(config_json,'$.params.seed') "
	         "FROM flows WHERE name LIKE 'XDS-unsw-random%'")
	return {int(x[0]) for x in rows if x[0] is not None}


def fresh_seeds(n, used):
	out = []
	while len(out) < n:
		s = secrets.randbelow(100000)
		if s not in used and s not in out:
			out.append(s)
	return out


def cancel(fid):
	r = requests.post(f"{API}/{fid}/stop", json={}, verify=False, timeout=20)
	log(f"cancel {fid}: HTTP {r.status_code}")


def queue_flow(thermo, seed):
	body = build_flow(DS, thermo, WEIGHT, seed, arch_override=ARCH)
	r = requests.post(API, json=body, verify=False, timeout=30)
	r.raise_for_status()
	fid = r.json().get("id")
	rr = requests.post(f"{API}/{fid}/restart", json={}, verify=False, timeout=15)
	rr.raise_for_status()
	log(f"queued flow {fid}: {body['name']}")
	return fid


def grow_to_30(thermo):
	have = live_count(thermo)
	need = max(0, TARGET - have)
	log(f"width {thermo}b: {have} live flows, need {need} more to reach {TARGET}")
	if need == 0:
		return
	seeds = fresh_seeds(need, existing_seeds())
	log(f"fresh seeds: {seeds}")
	for s in seeds:
		queue_flow(thermo, s)
	log(f"DONE growing {thermo}b to {live_count(thermo)} live flows")


def wait_for(fids):
	for fid in fids:
		while status_of(fid) not in ("completed", "cancelled", "failed", None):
			time.sleep(POLL)
		log(f"flow {fid} reached terminal status: {status_of(fid)}")


def main():
	if DONE.exists():
		log("DONE marker present — already executed; exiting.")
		return
	log(f"autohandler armed: waiting for trigger flow {TRIGGER_FLOW} to complete")
	while status_of(TRIGGER_FLOW) not in ("completed", "cancelled", "failed", None):
		time.sleep(POLL)
	st = status_of(TRIGGER_FLOW)
	log(f"trigger flow {TRIGGER_FLOW} status={st}")
	if st != "completed":
		log("trigger did not complete cleanly — aborting without action.")
		return

	f1 = best_f1(TRIGGER_FLOW)
	log(f"4050 (16b r92774) best_f1 held-out = {f1}  vs  64b champion {BEST_64B_F1}")
	if f1 is None:
		log("could not read 4050 best_f1 — aborting without action (manual review).")
		return

	if f1 <= BEST_64B_F1:
		log(f"ROUTE = 64b WINS ({f1:.4f} <= {BEST_64B_F1:.4f}).")
		cancel(CANCEL_IF_64B)
		grow_to_30(64)
	else:
		log(f"ROUTE = 16b WINS (upset! {f1:.4f} > {BEST_64B_F1:.4f}).")
		log(f"letting {WAIT_IF_16B} finish before growing 16b...")
		wait_for(WAIT_IF_16B)
		grow_to_30(16)

	DONE.write_text(time.strftime("%d/%m/%Y %H:%M:%S") + "\n")
	log("autohandler complete; DONE marker written.")


if __name__ == "__main__":
	main()
