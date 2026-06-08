#!/bin/bash
# Stage/launch the Phase-1 pure-tilt crossover scan (8 tilts: 10..45° in 5° steps).
# Detaches the sequential driver (start_new_session -> PPID=1, survives Claude/VS
# Code restart; dies on machine reboot but the driver is RESUMABLE — re-run this
# and it skips tilts whose winner.pkl exists). RAYON=3 keeps it off IDS's cores.
#
# Controller = pet project; IDS owns cores. Run when the IDS queue is idle / has
# spare cores. ~15h/tilt × 8 ≈ 120h sequential.
#
# Usage:  bash scripts/launch_tilt_sweep_phase1.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DRIVER="$(pwd)/scripts/tilt_sweep_phase1_driver.sh"
BASE="logs/controller/tilt_sweep_phase1"
mkdir -p "$BASE"
LOG="$BASE/driver_$(date +%Y%m%d_%H%M%S).log"

python3 - "$DRIVER" "$LOG" <<'PY'
import subprocess, os, sys
driver, log_path = sys.argv[1:3]
log = open(log_path, "w")
p = subprocess.Popen(["/bin/bash", driver], stdout=log, stderr=subprocess.STDOUT,
                     start_new_session=True, cwd=os.getcwd())
print("launched tilt-sweep driver  PID", p.pid)
open(os.path.join(os.path.dirname(log_path), "DRIVER_PID.txt"), "w").write(str(p.pid))
print("driver log:", log_path)
PY

echo "$BASE" > /tmp/tilt_sweep_dir.txt
echo
echo "Phase-1 tilt sweep launched detached (RAYON=3, sequential, resumable)."
echo "  driver log : $LOG"
echo "  per-tilt   : $BASE/tilt{10..45}/tilt{T}.log"
echo "  summary    : $BASE/SUMMARY.md  (WNN-vs-PID per tilt, appended as each finishes)"
echo "Tail the live tilt with:  tail -f \$(ls -t $BASE/tilt*/*.log | head -1)"
