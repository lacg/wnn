"""IDS-47 (Luiz 24/10/2026): cross-dataset A/B — BINARY lattice 2m max250 rate 0.25 x
{inherit, blend_neuron, blend_cluster} vs QUAD 1m (the original tandem config) on CICIDS2017,
CIC-IoT-2023 subsample and UNSW-NB15 random (_3way), each with its tandem cohort's fitness.
Seeds r20419-r20428, SEED-MAJOR interleave (n=5/7/10 are clean cut points). Source = the paused
tandem flow of each (cohort, seed) — cloned, never unpaused, so nothing jumps the FIFO queue.
QUAD seeds already completed in the tandem are reused, not re-run.
Usage: python queue_ids47_xds.py [--dry-run] [--cap max250|tot250|tot500] [--cancel-paused]
--cap (24/10, IDS-45 budget decision): the BINARY neuron cap. max250 = per-cluster 250 (the original);
totN = max_neurons_total N (free split). --cancel-paused first cancels every PAUSED IDS47 flow (QUAD too)
so the re-clone keeps the seed-major interleave; completed/running IDS47 flows are never touched.
"""
import sys, time, requests
sys.path.insert(0, "/Users/lacg/wnn/scripts")
import queue_param_clone as q

SEEDS = list(range(20419, 20429))
COHORTS = [("IDS4-cicids-CE20-zs", "cicids-CE20-zs"),
           ("IDS5-ciciot-B15AC-desir", "ciciot-B15AC-desir"),
           ("IDS7-unswr-WbCTRL-desir", "unswr-WbCTRL-desir")]
RULES = ["inherit", "blend_neuron", "blend_cluster"]
BINARY = {"memory_mode": "BINARY", "ids_write_rule": "lattice", "ids_single_cluster": False,
          "neuron_sample_rate": 0.25, "ids_offspring_folds": "all"}
CAPS = {"max250": {"max_neurons": 250},
        "tot250": {"max_neurons": 250, "max_neurons_total": 250},
        "tot500": {"max_neurons": 500, "max_neurons_total": 500}}


def plan(con, cap):
	out = []
	for seed in SEEDS:
		for src, short in COHORTS:
			sname = f"{src}-r{seed}"
			sid, st, cj = con.execute("SELECT id,status,config_json FROM flows WHERE name=? AND status!='cancelled' ORDER BY id LIMIT 1", (sname,)).fetchone()
			if st == "completed":
				out.append(("reuse", sname, sid, None, f"(QUAD 1m = {sid})", {}))
			else:
				out.append(("new", sname, sid, cj, f"IDS47-{short}-QUAD-1m-r{seed}", {}))
			for rule in RULES:
				out.append(("new", sname, sid, cj, f"IDS47-{short}-BINARY-lattice-2m-{cap}-xo_{rule}-r{seed}-rate0.25",
				            dict(BINARY, **CAPS[cap], ga_crossover_bits=rule)))
	return out


def arg(flag, default):
	return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def cancel_paused():
	con = q.ro()
	ids = [r[0] for r in con.execute("SELECT id FROM flows WHERE name LIKE 'IDS47-%' AND status='paused' ORDER BY id")]
	con.close()
	bad = 0
	for fid in ids:
		for _ in range(5):
			r = requests.patch(f"{q.DASHBOARD}/api/flows/{fid}", json={"status": "cancelled"}, verify=False, timeout=30)
			if r.status_code == 200: break
			time.sleep(1)
		else:
			bad += 1; print(f"  ! cancel FAILED {fid} http={r.status_code}")
		time.sleep(0.1)
	print(f"cancelled {len(ids) - bad} paused IDS47 flows, failures {bad}")
	return bad


def main():
	cap = arg("--cap", "max250")
	assert cap in CAPS, cap
	if "--cancel-paused" in sys.argv and "--dry-run" not in sys.argv and cancel_paused():
		return 1
	con = q.ro()
	p = plan(con, cap)
	existing = {r[0] for r in con.execute("SELECT name FROM flows WHERE name LIKE 'IDS47-%' AND status!='cancelled'")}
	con.close()
	new = [x for x in p if x[0] == "new"]
	print(f"{len(p)} rows: {len(new)} new flows, {len(p) - len(new)} reused QUAD")
	if "--dry-run" in sys.argv:
		for kind, sname, sid, _, name, sets in p:
			print(f"  {kind:5s} {name:72s} <- {sid} {sets.get('ga_crossover_bits', '')}")
		return 0
	created = []
	for kind, sname, sid, cj, name, sets in new:
		if name in existing:
			print(f"  = exists {name}"); continue
		fid, name = q.clone_one(sname, cj, sname, name, sets)
		for _ in range(5):
			r = requests.patch(f"{q.DASHBOARD}/api/flows/{fid}", json={"status": "queued"}, verify=False, timeout=30)
			if r.status_code == 200: break
			time.sleep(1)
		created.append((fid, name, sets)); print(f"  + {fid}  {name}"); time.sleep(0.2)
	time.sleep(1)
	bad = sum(q.verify([(f, n)], s) for f, n, s in created)
	print(f"created {len(created)}, verify failures {bad}")
	return 1 if bad else 0


if __name__ == "__main__":
	sys.exit(main())
