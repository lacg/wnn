"""Detached watcher: after the NEURONS full run finishes, launch the 4-PHASE
pipeline (grid → NEURONS → BITS → CONNECTIONS → MEMORY) at 5° ONLY.

Per Luiz 04/06: "NEURONS now, then all 4 phases on 5 degrees only first."
Uses run_phased_ga.py (already implements the canonical phased pipeline with
warm-start). At --tilt 5 --steps 250 it reproduces curriculum Stage A exactly
(run_phased_ga hardcodes body_rate=0.5/yaw_rate=0.3 = Stage A).

Budgets (STATED DEFAULTS — adjust if desired):
  neurons/bits/connections : 60 gens, patience 6
  memory                   : 120 gens, patience 6   (Luiz-specified)
  pop 200, eval-episodes 100, steps 250, universe-episodes 8
Fitness weights = the curriculum WINNER (read from /tmp/curric_fullmode_launched;
expected C9 = err0.40/stable0.30/jerk0.10/mono0.20) for consistency with selection.

Degree escalation (5°→…) is DEFERRED — Luiz will decide the interleave later.
Detached, idempotent.
"""
import os, sys, time, subprocess, importlib.util
from pathlib import Path

ROOT = Path("/Users/lacg/wnn")
FULL_MARKER = Path("/tmp/curric_fullmode_launched")    # "PID winner flog"
TILT, STEPS, POP, EVAL_EP, UNIV_EP = 5.0, 250, 200, 100, 8
NBC_GENS, NBC_PAT = 60, 6
MEM_GENS, MEM_PAT = 120, 6
PHASE4_SEED = 4004                                     # fresh; ≠ 42/1001/2002/3003/7007
POLL = 60
LOG = ROOT / "logs/curric_phase4_autolaunch.log"
DONE = Path("/tmp/curric_phase4_launched")


def log(msg):
	line = f"[{time.strftime('%d/%m/%Y %H:%M:%S')}] {msg}"
	print(line, flush=True)
	with open(LOG, "a") as f:
		f.write(line + "\n")


def alive(pid):
	try:
		os.kill(pid, 0); return True
	except ProcessLookupError:
		return False
	except PermissionError:
		return True


def weights_for(name):
	spec = importlib.util.spec_from_file_location("rc", str(ROOT / "tests/run_curriculum_ga.py"))
	m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
	for c in m.SWEEP_COMBOS:
		if c["name"] == name:
			return c
	raise ValueError(f"combo {name} not found")


def main():
	if DONE.exists():
		log("phase-4 already launched (marker present) — exiting."); return
	log("armed: waiting for the NEURONS full run to launch (marker), then finish")
	for _ in range(2880):  # up to ~48h for round3+full to even launch
		if FULL_MARKER.exists():
			break
		time.sleep(POLL)
	if not FULL_MARKER.exists():
		log("NEURONS full run never launched (no marker) — aborting."); return
	parts = FULL_MARKER.read_text().split()
	full_pid, winner = int(parts[0]), parts[1]
	log(f"NEURONS full run PID={full_pid}, winner={winner}; waiting for it to finish")
	while alive(full_pid):
		time.sleep(POLL)
	log(f"NEURONS full run {full_pid} exited — launching 4-phase at {TILT}°")

	w = weights_for(winner)
	ts = time.strftime("%Y%m%d_%H%M%S")
	savedir = ROOT / f"logs/controller/curriculum/phase4_{winner}_{TILT:.0f}deg_{ts}"
	savedir.mkdir(parents=True, exist_ok=True)
	plog = ROOT / f"logs/controller/curriculum/phase4_{winner}_{TILT:.0f}deg_{ts}.log"
	winner_pkl = savedir / "phase4_winner.pkl"
	Path("/tmp/curric_phase4_log.txt").write_text(str(plog) + "\n")

	cmd = [sys.executable, "-u", "tests/run_phased_ga.py",
	       "--tilt", str(TILT), "--steps", str(STEPS), "--pop", str(POP),
	       "--eval-episodes", str(EVAL_EP), "--universe-episodes", str(UNIV_EP),
	       "--neurons-gens", str(NBC_GENS), "--neurons-patience", str(NBC_PAT),
	       "--bits-gens", str(NBC_GENS), "--bits-patience", str(NBC_PAT),
	       "--conns-gens", str(NBC_GENS), "--conns-patience", str(NBC_PAT),
	       "--memory-gens", str(MEM_GENS), "--memory-patience", str(MEM_PAT),
	       "--fit-weight-err-sq", str(w["err"]), "--fit-weight-stable", str(w["stable"]),
	       "--fit-weight-jerk", str(w["jerk"]), "--fit-weight-mono", str(w["mono"]),
	       "--base-seed", str(PHASE4_SEED), "--save-winner", str(winner_pkl)]
	env = dict(os.environ)
	env["PYTHONPATH"] = f"{ROOT}/src/wnn:" + env.get("PYTHONPATH", "")
	env["RAYON_NUM_THREADS"] = "3"
	f = open(plog, "w")
	p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
	                     start_new_session=True, env=env, cwd=str(ROOT))
	DONE.write_text(f"{p.pid} {winner} {plog}\n")
	log(f"launched 4-PHASE (grid→N→B→C→M) PID {p.pid}, winner={winner} "
	    f"({w['err']}/{w['stable']}/{w['jerk']}/{w['mono']}), {TILT}°, "
	    f"N/B/C={NBC_GENS}g/p{NBC_PAT}, MEM={MEM_GENS}g/p{MEM_PAT}")
	log(f"  log: {plog}")


if __name__ == "__main__":
	main()
