#!/usr/bin/env python3
"""Micro-benchmark: what actually makes an sn>0 generation cost 4.6h?

sn=0 skips the per-motor beam-search solve ENTIRELY (controller.rs: solve_motors=0
when state_bits_in==0) and that is the ONLY training-path difference, so ~100% of the
measured 150x sn=8/sn=0 gap is that solve. This times the two levers that target it:

  RAYON_NUM_THREADS   scheduling only, no arithmetic change. The chains currently set
                      NOTHING (rayon takes all 16 cores) while the IDS worker holds a
                      13-core budget on a 16-core box — measured load average 29.3.
                      run_dfa_1layer_study.sh pinned 10 for exactly this reason.
  --topk-per-neuron   beam-search top-K per neuron (default 4). Halving it halves the
                      branching but considers fewer addresses — a SCIENCE knob.

METHOD. Each arm runs a deliberately tiny STATEFUL phased_ga and reads the grid stage's
own reported duration from the "GRID WINNER ... (Ns)" line, then the run is killed. Same
work in every arm, so the numbers are comparable; only the two variables move.

CAVEATS, stated because they bound what this can conclude:
  * folds=1 here (production is 5). Folds multiply cost linearly and identically across
    arms, so RATIOS transfer; absolute seconds do not.
  * If another controller (e.g. the P4 chain) holds the box, every arm is contended.
    The topk ratio is fairly robust to that; the THREAD comparison is NOT — pinning is
    a contention question, so measuring it under the wrong contention answers the wrong
    question. Re-run the thread arms when only the IDS worker is live.

Usage:  PYTHONPATH=src/wnn python3 scripts/bench_stateful_solve.py [--repeats N]
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = "/Users/lacg/wnn"
VP = "/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
OUTDIR = "/private/tmp/bench_stateful"
GRID_RE = re.compile(r"GRID WINNER .*\((\d+)s\)")

# Tiny but STATEFUL — sn=8 is the whole point; at sn=0 neither knob does anything.
BASE = [
	"--levels", "16", "--skip-stages", "bits,connections",
	"--grid-state-neurons", "8", "--max-state-neurons", "8",
	"--grid-bits", "30", "--max-output-neurons", "128",
	"--pop", "6", "--num-eval-folds", "1", "--runs", "1",
	"--eval-episodes", "10", "--steps", "400", "--tilt", "5.0",
	"--report-episodes", "5", "--report-seeds", "99990101",
	"--teacher", "lqr", "--memory-mode", "BINARY", "--disturbance", "L3D",
	"--obs-peraxis-p", "--obs-peraxis-i", "--no-obs-peraxis-yaw",
	"--obs-yaw-err", "--obs-yaw-err-i",
	"--base-seed", "31337002",
]

ARMS = [
	("A  threads=16 topk=4  (today's chains)", None, None),
	("B  threads=10 topk=4  (dfa1l pinning)",  "10", None),
	("C  threads=16 topk=2",                    None, "2"),
	("D  threads=10 topk=2",                    "10", "2"),
]


def run_arm(threads, topk, tag):
	"""Launch, wait for the grid line, kill. Returns grid seconds or None."""
	os.makedirs(OUTDIR, exist_ok=True)
	log = f"{OUTDIR}/{tag}.out"
	env = dict(os.environ)
	env["PYTHONPATH"] = f"{ROOT}/src/wnn:" + env.get("PYTHONPATH", "")
	if threads:
		env["RAYON_NUM_THREADS"] = threads
	else:
		env.pop("RAYON_NUM_THREADS", None)
	cmd = [VP, "-u", "-m", "wnn.control.phased_ga"] + BASE
	if topk:
		cmd += ["--topk-per-neuron", topk]
	cmd += ["--save-winner", f"{OUTDIR}/{tag}_winner.yaml.gz"]

	with open(log, "w") as fh:
		p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
		t0 = time.time()
		secs = None
		try:
			while p.poll() is None and time.time() - t0 < 1800:
				time.sleep(3)
				try:
					with open(log) as r:
						m = GRID_RE.search(r.read())
				except OSError:
					m = None
				if m:
					secs = int(m.group(1))
					break
		finally:
			p.terminate()
			try:
				p.wait(timeout=20)
			except subprocess.TimeoutExpired:
				p.kill()
	return secs


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--repeats", type=int, default=1)
	args = ap.parse_args()

	print("=" * 74)
	print("  sn=8 solve micro-benchmark — grid-stage seconds (lower = faster)")
	print(f"  box: {os.cpu_count()} cores | load {os.getloadavg()[0]:.1f} at start")
	other = subprocess.run(["pgrep", "-f", "wnn.control.phased_ga"],
	                       capture_output=True, text=True).stdout.split()
	if other:
		print(f"  ⚠️  {len(other)} other controller(s) live — thread arms are CONTENDED")
	print("=" * 74)

	results = {}
	for rep in range(args.repeats):
		for label, threads, topk in ARMS:
			tag = label.split()[0] + f"_r{rep}"
			t0 = time.time()
			secs = run_arm(threads, topk, tag)
			wall = time.time() - t0
			results.setdefault(label, []).append(secs)
			got = f"{secs}s grid" if secs else "NO GRID LINE (timeout/crash)"
			print(f"  rep{rep}  {label:<38} {got:<28} (wall {wall:.0f}s)")

	print("-" * 74)
	base = results.get(ARMS[0][0], [None])
	base_v = [v for v in base if v]
	b = sum(base_v) / len(base_v) if base_v else None
	for label, _, _ in ARMS:
		vals = [v for v in results.get(label, []) if v]
		if not vals:
			print(f"  {label:<38} no data")
			continue
		mean = sum(vals) / len(vals)
		rel = f"{b / mean:.2f}x vs A" if b else ""
		print(f"  {label:<38} mean {mean:7.1f}s   {rel}")
	print("=" * 74)
	print("  Ratios transfer to production (folds multiply every arm alike);")
	print("  absolute seconds do NOT (folds=1 here, production is 5).")
	if other:
		print("  ⚠️  Thread arms were contended by another controller — re-run them")
		print("      with only the IDS worker live before trusting B/D over A/C.")
	shutil.rmtree(OUTDIR, ignore_errors=True)
	return 0


if __name__ == "__main__":
	sys.exit(main())
