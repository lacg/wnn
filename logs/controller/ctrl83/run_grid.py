"""CTRL-83 stress panel runner: every (controller x cell) of cells.py, resumable, cached.

  run_grid.py [--cells c1,c2] [--controllers k1,k2] [--no-row-b] [--list]
  CTRL-87 readout: --controllers l2bnm_s1,... (cells.EXTRA_WNN) --no-row-b --cells nom,wear30,mass25,lag30,lag10,noise2

Order is INTERLEAVED (prereg §5): cell-major in cells.CELLS order (primary cells first),
and inside a cell every controller (5 WNN seeds, rows A/C, PID, LQI, then row B), so a
partial grid always holds complete cells — never all of one controller first.

Cache: data/<controller>__<cell>.npz (score_cell.py, atomic rename) and
data/mpcofB__<cell>.json (rowb.py). A job whose cache exists is skipped, so re-running
resumes. A gate failure stops the grid (exit 2) — a cell that fails a gate is not a result.
CPU: nice 15, RAYON_NUM_THREADS=6 per job, one job at a time. Report-only; no training.
"""
import argparse, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cells as C

PY = "/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
DATA = os.path.join(HERE, "data")
LOG = os.path.join(DATA, "grid.log")
EST_S = {"wnn": 20, "rival": 75, "mpcofB": 40}   # measured in the dry run (prereg §7)


def jobs(cells, ctrls, row_b):
	for c in cells:
		for k in ctrls:
			yield k, c, os.path.join(DATA, f"{k}__{c}.npz")
		if row_b:
			yield "mpcofB", c, os.path.join(DATA, f"mpcofB__{c}.json")


def cmd_for(k, c, out):
	if k == "mpcofB":
		return [PY, os.path.join(HERE, "rowb.py"), c, out]
	extra = [] if C.is_wnn(k) else ["--rival-tail"]
	return [PY, os.path.join(HERE, "score_cell.py"), k, c, out, *extra]


def est(k):
	return EST_S["mpcofB"] if k == "mpcofB" else EST_S["wnn"] if C.is_wnn(k) else EST_S["rival"]


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--cells", default=",".join(c[0] for c in C.CELLS))
	ap.add_argument("--controllers", default=",".join(C.CONTROLLERS))
	ap.add_argument("--no-row-b", action="store_true")
	ap.add_argument("--list", action="store_true", help="print the pending jobs + ETA, run nothing")
	o = ap.parse_args()
	cells, ctrls = o.cells.split(","), o.controllers.split(",")
	for c in cells:
		if c not in C.CELL:
			sys.exit(f"unknown cell {c}")
	for k in ctrls:
		if k not in C.ALL_CONTROLLERS:
			sys.exit(f"unknown controller {k}")
	os.makedirs(DATA, exist_ok=True)
	todo = [j for j in jobs(cells, ctrls, not o.no_row_b) if not os.path.exists(j[2])]
	eta = sum(est(k) for k, _, _ in todo)
	print(f"[ctrl83] {len(todo)} pending jobs, estimated {eta/3600:.2f} h serial", flush=True)
	if o.list:
		for k, c, _ in todo:
			print(f"  {c:7s} {k}")
		return
	env = {**os.environ, "RAYON_NUM_THREADS": "6"}
	for i, (k, c, out) in enumerate(todo, 1):
		t0 = time.time()
		p = subprocess.run(["nice", "-n", "15", *cmd_for(k, c, out)], env=env, capture_output=True, text=True)
		line = [l for l in p.stdout.splitlines() if l.startswith("[ctrl83]")]
		stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
		with open(LOG, "a") as f:
			f.write(f"{stamp} job {i}/{len(todo)} rc={p.returncode} {time.time()-t0:.0f}s {line[-1] if line else ''}\n")
			if p.returncode != 0:
				f.write(p.stderr[-4000:] + "\n")
		print(line[-1] if line else f"[ctrl83] {k} {c} rc={p.returncode}", flush=True)
		if p.returncode != 0:
			print(p.stderr[-2000:], file=sys.stderr)
			sys.exit(2 if p.returncode == 2 else 4)
	print("[ctrl83] grid complete", flush=True)


if __name__ == "__main__":
	main()
