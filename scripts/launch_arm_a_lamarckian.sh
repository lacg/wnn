#!/bin/bash
# Arm A (07/06/2026 finding) — collapse the phased-GA pipeline to
#   grid → NEURONS → MEMORY,  skipping BITS + CONNECTIONS.
#
# Rationale: under --lamarckian the NEURONS stage already optimizes
# neurons+connections+memory jointly (grid covers bits), so BITS+CONNECTIONS were
# ~28h of ~43h for ~0 err gain. MEMORY (72s, +4pp stability) is essentially free,
# so we KEEP it and give it a LONGER budget to push stability past 88%.
#   4-stage result (held-out): best err 3.69° (neurons) / best stable 88% (memory)
#   goal: PID 3.40° / 98%
#
# Controller = pet project; IDS owns cores. Launch ONLY when the IDS queue is
# idle / has spare cores. RAYON=3 keeps it off IDS's back. Detached (PPID=1) so
# it survives a Claude/VS Code restart; dies on machine reboot (resume from the
# stageN checkpoints in $DIR).
#
# Usage:  bash scripts/launch_arm_a_lamarckian.sh
set -euo pipefail
cd "$(dirname "$0")/.."

TS=$(date +%Y%m%d_%H%M%S)
DIR="logs/controller/curriculum/armA_grid_neurons_memory_5deg_pop50_${TS}"
mkdir -p "$DIR"
LOG="${DIR}/armA_${TS}.log"
VENV_PY="$(pwd)/wnn/bin/python"

python3 - "$VENV_PY" "$LOG" "$DIR" <<'PY'
import subprocess, os, sys
venv_py, log_path, newdir = sys.argv[1:4]
env = dict(os.environ)
env["RAYON_NUM_THREADS"] = "3"
env["PYTHONPATH"] = os.getcwd() + "/src/wnn:" + env.get("PYTHONPATH", "")
args = [venv_py, "-u", "tests/run_phased_ga.py",
        "--tilt","5","--body-rate","0.5","--yaw-rate","0.3","--steps","250",
        "--pop","50","--elitism","0.2","--check-interval","5",
        # Stage 1 — the productive stage (proven config)
        "--neurons-gens","30","--neurons-patience","2",
        # Skip the ~28h dead weight
        "--skip-stages","bits,connections",
        # Stage 4 — MEMORY, LONGER budget (it's ~7s/gen; 300 gens ≈ 36 min worst case)
        "--memory-gens","300","--memory-patience","20",
        "--eval-episodes","100","--universe-episodes","8","--num-eval-folds","5",
        "--fit-weight-err-sq","0.40","--fit-weight-stable","0.30",
        "--fit-weight-jerk","0.10","--fit-weight-mono","0.20",
        "--base-seed","5005","--report-seed","9009","--lamarckian",
        "--save-stage-checkpoints", newdir, "--save-winner", newdir + "/winner.pkl"]
log = open(log_path, "w")
p = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT,
                     start_new_session=True, env=env, cwd=os.getcwd())
print("launched Arm A  PID", p.pid)
open(newdir + "/ARM_A_PID.txt", "w").write(str(p.pid))
print("dir:", newdir)
print("log:", log_path)
PY

echo "$DIR" > /tmp/armA_lamarckian_dir.txt
echo
echo "Arm A launched detached (RAYON=3). Tail the log with:"
echo "  tail -f $LOG"
echo "Expected: grid → NEURONS(~15h) → [skip bits,connections] → MEMORY(~minutes, 300g)"
echo "Compare held-out vs the 4-stage: best err 3.69° / best stable 88°; goal PID 3.40°/98%."
