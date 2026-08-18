#!/bin/bash
# One sequential job: re-headline the already-flown widths, THEN re-arm the ladder.
#
# Sequential on purpose — only ONE controller process may run at a time, and the
# round-1 cull ranks widths by headline steady, so every banked width must be
# re-selected over the full candidate pool BEFORE the chain reaches the cull.
#
# The chain is idempotent per-run (run_controller_arm skips any tag whose marker
# exists), so re-arming replays instantly through b10/b12/b14 and starts flying
# at the first width with no marker — b16, which was killed mid-flight for the
# stage-select fix and left no marker.
set -u
ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
LOG="/private/tmp/sweep_ladder.log"

echo "[wrap] $(date -u +%FT%TZ) ===== RE-HEADLINE banked widths (no re-fly) =====" >> "$LOG"
bash "$ROOT/scripts/recalc_sweep_headlines.sh" >> "$LOG" 2>&1
rc=$?
echo "[wrap] $(date -u +%FT%TZ) recalc finished rc=$rc" >> "$LOG"
if [ $rc -ne 0 ]; then
	echo "[wrap] $(date -u +%FT%TZ) ABORT: recalc failed — ladder NOT re-armed" >> "$LOG"
	exit $rc
fi

echo "[wrap] $(date -u +%FT%TZ) ===== RE-ARM ladder (skips banked markers, resumes at b16) =====" >> "$LOG"
exec bash "$ROOT/scripts/sweep_ladder_chain.sh"
