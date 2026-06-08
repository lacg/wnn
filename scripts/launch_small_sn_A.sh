#!/bin/bash
# Small-sn LEARNED-INTEGRAL controller experiment (option A + E), 08/06/2026.
#
# The proper test of option A (project_controller_stability_diagnosis): evolve
# architectures DESIGNED to use the recurrent state, instead of retrofitting A
# onto the sn=15 winner (which the GA built for a MEMORYLESS world → huge + slow).
#   - --state-integral : train the recurrent STATE as a learned integrator
#                        (WNN_STATE_INTEGRAL_TARGET=1, direct PID-integral target).
#   - delta-control ON (default, option E): structural output integrator (+5pp).
#   - SMALL state neurons {3,6,9}: forced prefix is 2·sn, so small sn = fast
#     (sn=3 ≈ 14s/genome vs sn=15 ≈ 840s) AND the right size for an integrator
#     (3 axes; 4^sn QSR states). Grid auto-filters sn,b pairs with suffix<4.
#   - SMALLER OUTPUT (levels 16 vs the old 42): with the state carrying the
#     integral, the output specializes and can shrink.
# Pipeline: grid → NEURONS → MEMORY (skip bits+connections, per the redundancy
# finding). Goal: does a learned integrator close the gap to PID (5° 98%/3.40°)?
#
# Detached (PPID=1, survives restart; resumable from stageN checkpoints), RAYON=3
# (off IDS's cores). Controller = pet project; run when IDS idle.
# Usage:  bash scripts/launch_small_sn_A.sh
set -euo pipefail
cd "$(dirname "$0")/.."

TS=$(date +%Y%m%d_%H%M%S)
DIR="logs/controller/curriculum/smallsnA_grid_neurons_memory_5deg_${TS}"
mkdir -p "$DIR"
LOG="${DIR}/smallsnA_${TS}.log"
VENV_PY="$(pwd)/wnn/bin/python"

python3 - "$VENV_PY" "$LOG" "$DIR" <<'PY'
import subprocess, os, sys
venv_py, log_path, newdir = sys.argv[1:4]
env = dict(os.environ); env["RAYON_NUM_THREADS"] = "3"
env["PYTHONPATH"] = os.getcwd() + "/src/wnn:" + env.get("PYTHONPATH", "")
args = [venv_py, "-u", "tests/run_phased_ga.py",
        "--state-integral",                       # option A (learned integrator)
        # delta-control (E) is ON by default; small state + smaller output:
        "--grid-state-neurons", "3", "6", "9", "12",
        "--grid-bits", "18", "24", "30",
        "--levels", "16",
        "--tilt", "5", "--body-rate", "0.5", "--yaw-rate", "0.3", "--steps", "250",
        "--pop", "50", "--elitism", "0.2", "--check-interval", "5",
        "--neurons-gens", "30", "--neurons-patience", "2",
        "--skip-stages", "bits,connections",
        "--memory-gens", "60", "--memory-patience", "2",
        "--eval-episodes", "100", "--universe-episodes", "8", "--num-eval-folds", "5",
        "--fit-weight-err-sq", "0.40", "--fit-weight-stable", "0.30",
        "--fit-weight-jerk", "0.10", "--fit-weight-mono", "0.20",
        "--base-seed", "5005", "--report-seed", "9009", "--lamarckian",
        "--save-stage-checkpoints", newdir, "--save-winner", newdir + "/winner.pkl"]
log = open(log_path, "w")
p = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT,
                     start_new_session=True, env=env, cwd=os.getcwd())
print("launched small-sn A  PID", p.pid)
open(newdir + "/PID.txt", "w").write(str(p.pid))
print("dir:", newdir); print("log:", log_path)
PY

echo "$DIR" > /tmp/smallsnA_dir.txt
echo
echo "Small-sn A+E experiment launched detached (RAYON=3, resumable)."
echo "  tail:  tail -f $LOG"
echo "  Watch: grid over sn{3,6,9}×b{18,24,30} → NEURONS → MEMORY. Goal: held-out toward PID 98%/3.40°."
echo "  Baselines: memoryless 83% / E +5pp / PID 3.40°/98%."
