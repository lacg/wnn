#!/usr/bin/env bash
# Watch for the H3 swap marker; when it appears, launch the H3-alone controller
# run EXACTLY ONCE, then exit. Detached (PPID=1) so it survives a session clear.
#
# Chain: worker_swap.py (pid watching flow 4166) finishes the swap + writes
#   /tmp/wnn_h3_swap_done.json  -->  this watcher fires launch_controller_phased.sh.
#
# Re-arm after a session clear with:
#   python scripts/detach_launch.py /Users/lacg/wnn/logs/controller/h3_watcher.out \
#     /Users/lacg/wnn -- bash scripts/h3_launch_watcher.sh
set -uo pipefail
PROJ=/Users/lacg/wnn
SWAP_MARKER=/tmp/wnn_h3_swap_done.json
LAUNCH_MARKER=/tmp/wnn_h3_launched.json
RUN_NAME=c10_delta_decouple_20260618
BASE_SEED=31337002

echo "[h3-watcher] armed $(date -u +%Y-%m-%dT%H:%M:%SZ); waiting for $SWAP_MARKER"
# Idempotency: if H3 was already launched, do nothing.
if [ -f "$LAUNCH_MARKER" ]; then
	echo "[h3-watcher] $LAUNCH_MARKER already present -> H3 already launched; exiting."
	exit 0
fi

while [ ! -f "$SWAP_MARKER" ]; do
	sleep 60
done

echo "[h3-watcher] swap marker detected $(date -u +%Y-%m-%dT%H:%M:%SZ):"
cat "$SWAP_MARKER" || true

# Give the freshly-restarted worker a moment to settle before adding the controller load.
sleep 20

echo "[h3-watcher] launching H3-alone: $RUN_NAME (--decouple-outputs, delta default, NO obs)"
bash "$PROJ/scripts/launch_controller_phased.sh" "$RUN_NAME" "$BASE_SEED" --decouple-outputs

date -u +%Y-%m-%dT%H:%M:%SZ > "$LAUNCH_MARKER"
echo "[h3-watcher] wrote $LAUNCH_MARKER; H3 launched. Watcher exiting."
