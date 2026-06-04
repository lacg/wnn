"""Detached watcher: when round 3 finishes, pick the 3-seed winner and launch
the FULL 5-stage curriculum on it.

Winner = highest mean held-out stable across rounds 1/2/3 among the round-3
contenders (W2,C2,C11,C13,W4,C9,C14). So: C13 if it holds, else whoever leads
(currently C9). Launches:
  --mode full --weights <winner> --pop 200 --base-seed 3003 --report-seed 7007
Train seed 3003 + report seed 7007 are BOTH distinct from the selection seeds
(42/1001/2002) → the held-out report is leak-free. Detached, idempotent.
"""
import os, sys, time, re, subprocess, importlib.util
from pathlib import Path

ROOT = Path("/Users/lacg/wnn")
R3_MARKER = Path("/tmp/curric_r3_launched")          # "PID logpath"
ROUND_LOGS = ["/tmp/curric_log.txt", "/tmp/curric_r2_log.txt", "/tmp/curric_r3_log.txt"]
R3_COMBOS = {"W2", "C2", "C11", "C13", "W4", "C9", "C14"}
TRAIN_SEED, REPORT_SEED = 3003, 7007
POLL = 60
LOG = ROOT / "logs/curric_fullmode_autolaunch.log"
DONE = Path("/tmp/curric_fullmode_launched")

_combo_re = re.compile(r"# COMBO (\w+):")
_best_re = re.compile(r"best: err=[\d.]+°\s+stable=([\d.]+)%")


def log(msg):
	line = f"[{time.strftime('%d/%m/%Y %H:%M:%S')}] {msg}"
	print(line, flush=True)
	with open(LOG, "a") as f:
		f.write(line + "\n")


def parse_holdout(logpath):
	"""{combo: last held-out stable%} from a sweep log."""
	out, cur = {}, None
	try:
		text = Path(logpath).read_text()
	except Exception:
		return out
	for ln in text.splitlines():
		m = _combo_re.search(ln)
		if m:
			cur = m.group(1); continue
		if cur:
			b = _best_re.search(ln)
			if b:
				out[cur] = float(b.group(1))
	return out


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
		log("full mode already launched (marker present) — exiting."); return
	# Resolve round-3 log path + PID from the marker.
	for _ in range(120):  # wait up to ~2h for round 3 to even launch
		if R3_MARKER.exists():
			break
		time.sleep(POLL)
	if not R3_MARKER.exists():
		log("round 3 never launched (no marker) — aborting."); return
	r3_pid = int(R3_MARKER.read_text().split()[0])
	log(f"waiting for round-3 process {r3_pid} to finish")
	while alive(r3_pid):
		time.sleep(POLL)
	log(f"round-3 process {r3_pid} exited")

	# Compute 3-seed means over the round-3 contenders.
	# ROUND_LOGS are /tmp POINTER files whose content is the real log path.
	logpaths = [Path(x).read_text().strip() if Path(x).exists() else x for x in ROUND_LOGS]
	rounds = [parse_holdout(lp) for lp in logpaths]
	means = {}
	for combo in R3_COMBOS:
		vals = [r[combo] for r in rounds if combo in r]
		if vals:
			means[combo] = (sum(vals) / len(vals), len(vals), vals)
	if not means:
		log("no held-out data parsed — aborting (manual review)."); return
	ranked = sorted(means.items(), key=lambda kv: -kv[1][0])
	for combo, (mean, n, vals) in ranked:
		log(f"  {combo}: mean={mean:.1f}% over {n} seeds {vals}")
	winner = ranked[0][0]
	w = weights_for(winner)
	wstr = f"err={w['err']},stable={w['stable']},jerk={w['jerk']},mono={w['mono']}"
	log(f"WINNER = {winner} ({wstr}), mean={ranked[0][1][0]:.1f}%")

	ts = time.strftime("%Y%m%d_%H%M%S")
	savedir = ROOT / f"logs/controller/curriculum/full_{winner}_{ts}"
	savedir.mkdir(parents=True, exist_ok=True)
	flog = ROOT / f"logs/controller/curriculum/full_{winner}_{ts}.log"
	Path("/tmp/curric_full_log.txt").write_text(str(flog) + "\n")
	cmd = [sys.executable, "-u", "tests/run_curriculum_ga.py", "--mode", "full",
	       "--weights", wstr, "--pop", "200",
	       "--base-seed", str(TRAIN_SEED), "--report-seed", str(REPORT_SEED),
	       "--train-workers", "3", "--num-eval-folds", "5", "--save-dir", str(savedir)]
	env = dict(os.environ); env["PYTHONPATH"] = f"{ROOT}/src/wnn:" + env.get("PYTHONPATH", "")
	f = open(flog, "w")
	p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
	                     start_new_session=True, env=env, cwd=str(ROOT))
	DONE.write_text(f"{p.pid} {winner} {flog}\n")
	log(f"launched FULL mode: PID {p.pid}, winner {winner}, train {TRAIN_SEED}, report {REPORT_SEED}")
	log(f"  log: {flog}")


if __name__ == "__main__":
	main()
