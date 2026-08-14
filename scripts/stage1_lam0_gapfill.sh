#!/usr/bin/env bash
# STAGE 1 GAP-FILL — re-fly the ONE arm lost to the 14/08/2026 edit-window crash.
#
# WHAT WAS LOST. S1L_lam0_..._s31337003 (the replication seed's CONTROL arm) ran
# 3h08m and died in its final report with "not enough values to unpack (expected
# 5, got 4)". Cause: chunk D's Python landed a strict 5-unpack, and phased_ga
# imports classical_baseline LAZILY — at the GRID stage, not just at held-out. The
# grid finished ~6 min into the window between that edit and the arity shim that
# would have absorbed it, so the run bound the broken intermediate version and
# carried it to the end. rc=1 => no marker, which is correct: marker absence means
# "re-run me", and this script is the re-run.
#
# WHY A WAITER AND NOT A DIRECT LAUNCH. The sweep chain iterates the lambda list
# ONCE, so it will not come back for a skipped arm on its own. Relaunching the
# sweep script itself is idempotent — run_controller_arm skips any tag whose
# marker exists — so once the current chain is done, one relaunch re-flies exactly
# the missing lambda=0 and nothing else.
#
# ORDERING WITH THE INSTALL GUARD. chunk_d_install_guard.sh waits for 10 markers +
# 0 controllers. Without this gap-fill the sweep tops out at 9 and the guard
# stalls forever. With it, the guard stays parked while this arm flies (9 markers)
# and opens the moment it lands (10). No race: the two gates are disjoint.
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1

LOG="/private/tmp/stage1_lambda_sweep.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
SWEEP_PID="${GAP_SWEEP_PID:-0}"
MARKERS="$ROOT/experiments/stage1lambda_markers"
MISSING="S1L_lam0_mpcof_cf21_brushless_L4C_s31337003.json"

log() { echo "[gapfill] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controllers() { ps -axo pid,command 2>/dev/null | grep "wnn.control.phased_ga" \
	| grep -v "/usr/bin/time" | grep -v grep | grep -c python; }

log "########## ARMED — will re-fly the missing lambda=0 seed 31337003 once sweep PID $SWEEP_PID exits ##########"

# 48 h ceiling: 4 remaining arms at ~3h30m is ~14 h. Only fires if the sweep died.
WAITED=0
while true; do
	ALIVE=0
	[ "$SWEEP_PID" != "0" ] && ps -p "$SWEEP_PID" >/dev/null 2>&1 && ALIVE=1
	if [ "$ALIVE" -eq 0 ] && [ "$(controllers)" -eq 0 ]; then
		log "gate open: sweep chain gone, box idle (waited ${WAITED}s)"
		break
	fi
	if [ "$WAITED" -ge 172800 ]; then
		log "ABORT: sweep still alive after 48 h — not gap-filling."
		exit 1
	fi
	sleep 300
	WAITED=$((WAITED + 300))
done

if [ -f "$MARKERS/$MISSING" ]; then
	log "nothing to do — $MISSING already exists"
	exit 0
fi

log "relaunching the sweep for seed 31337003 (skips the 4 markered arms, re-flies lambda=0)"
exec env S1_SEEDS=31337003 bash "$ROOT/scripts/stage1_lambda_alt_sweep.sh"
