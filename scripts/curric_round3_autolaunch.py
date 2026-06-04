"""Detached watcher: launch curriculum round 3 (seed C) when round 2 finishes.

Waits for the round-2 sweep process to exit, verifies it completed (SWEEP RESULT
in its log, not a crash), then launches round 3 — seed C=2002 over the 7 live
contenders (confirmation set minus the dead C3/C7/W3) — detached, and points
/tmp/curric_r3_log.txt at the new log so build_controller_results.py auto-finds it.
"""
import os, sys, time, subprocess
from pathlib import Path

ROOT = Path("/Users/lacg/wnn")
R2_PID = 63301
R3_SEED = 2002
R3_COMBOS = "W2,C2,C11,C13,W4,C9,C14"     # dropped C3, C7, W3 per Luiz 03/06
POLL = 60
LOG = ROOT / "logs/curric_round3_autolaunch.log"
DONE = Path("/tmp/curric_r3_launched")


def log(msg):
	line = f"[{time.strftime('%d/%m/%Y %H:%M:%S')}] {msg}"
	print(line, flush=True)
	with open(LOG, "a") as f:
		f.write(line + "\n")


def alive(pid):
	try:
		os.kill(pid, 0)
		return True
	except ProcessLookupError:
		return False
	except PermissionError:
		return True


def main():
	if DONE.exists():
		log("round 3 already launched (marker present) — exiting.")
		return
	log(f"armed: waiting for round-2 process {R2_PID} to exit")
	while alive(R2_PID):
		time.sleep(POLL)
	log(f"round-2 process {R2_PID} exited")

	# Verify round 2 actually completed (not a crash).
	try:
		r2log = Path("/tmp/curric_r2_log.txt").read_text().strip()
		txt = Path(r2log).read_text() if Path(r2log).exists() else ""
	except Exception:
		txt = ""
	if "SWEEP RESULT" not in txt:
		log("WARNING: round-2 log has no 'SWEEP RESULT' — round 2 may have crashed. "
		    "Launching round 3 anyway (independent seed; safe).")

	ts = time.strftime("%Y%m%d_%H%M%S")
	savedir = ROOT / f"logs/controller/curriculum/r3_seedC_{ts}"
	savedir.mkdir(parents=True, exist_ok=True)
	r3log = ROOT / f"logs/controller/curriculum/r3_seedC_{ts}.log"
	Path("/tmp/curric_r3_log.txt").write_text(str(r3log) + "\n")

	cmd = [sys.executable, "-u", "tests/run_curriculum_ga.py", "--mode", "sweep",
	       "--sweep-steps", "250", "--pop", "200", "--sweep-pop", "50",
	       "--combos", R3_COMBOS, "--train-workers", "3", "--num-eval-folds", "5",
	       "--base-seed", str(R3_SEED), "--save-dir", str(savedir)]
	env = dict(os.environ)
	env["PYTHONPATH"] = f"{ROOT}/src/wnn:" + env.get("PYTHONPATH", "")
	f = open(r3log, "w")
	p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
	                     start_new_session=True, env=env, cwd=str(ROOT))
	DONE.write_text(f"{p.pid} {r3log}\n")
	log(f"launched round 3: PID {p.pid}, combos {R3_COMBOS}, seed {R3_SEED}")
	log(f"  log: {r3log}")


if __name__ == "__main__":
	main()
