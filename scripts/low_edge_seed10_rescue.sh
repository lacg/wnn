#!/bin/bash
# Low-edge seed10 rescue (02/07/2026). The 8 seed10 cells crashed at launch on
# source/wheel skew (action-repeat Python landed 3728da7b while the wnn-venv wheel
# was still pre-action-repeat -> TypeError at WnnController ctor; seed09 cells were
# unaffected). The E2 driver has since installed the parity-proven wheel (N=1
# bit-identical: 22 GPU suites 0 mismatches), so a re-run is scientifically
# homogeneous with the seed09 rows. This wrapper waits for E2 to fully drain
# (one controller at a time), then re-invokes the low-edge driver, whose
# resume-skip re-runs ONLY the missing seed10 cells.
set -u
cd /Users/lacg/wnn
LOG=logs/controller/LowEdgeSeed10Rescue_20260702.log
exec >>"$LOG" 2>&1

echo "[rescue] $(date '+%Y-%m-%d %H:%M:%S') WAITING for E2 (/tmp/wnn_e2_done.json)"
while [ ! -f /tmp/wnn_e2_done.json ]; do sleep 60; done
echo "[rescue] $(date '+%Y-%m-%d %H:%M:%S') E2 done — re-running low-edge (seed10 cells only via resume-skip)"
bash scripts/low_edge_driver.sh
echo "{\"rescue_done\":true,\"ts\":\"$(date '+%Y-%m-%dT%H:%M:%S')\"}" > /tmp/wnn_rescue_done.json
echo "[rescue] $(date '+%Y-%m-%d %H:%M:%S') rescue pass complete (marker /tmp/wnn_rescue_done.json)"
