#!/usr/bin/env bash
# wheel_tail_install.sh — install the narrowed-columns ram_controller wheel at
# the ARMED-CHAIN TAIL, then smoke ONE tiny launch. Armed 11/08/2026.
#
# WHY GATED. Three chains are armed (sn>0 37795 -> dob-lqr 95222 -> dob-mpc
# 95298). The sn arm's controls lean on "controller source byte-identical
# since 5f3d113c", and the DOB arms interleave off/on cells WITHIN each arm —
# so a wheel landing mid-anything breaks a comparison. The only safe install
# point is after ALL THREE chains exit. (The wheel is a pure-representation
# change — 125 Rust tests incl. 14 parity sweeps + 4 narrowing tests pass —
# but the invariant is cheap to keep and expensive to argue about.)
#
# WHY SMOKE. feedback_never_deploy_while_chain_armed: 3 cohorts died in a day;
# a 60s pop-6 launch catches what green unit tests cannot. This installs, then
# smokes, then STOPS — it launches no chain and touches nothing else.
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
LOG="/private/tmp/wheel_tail_install.log"
WHEEL="/Volumes/20260401-WDBlack-SN850X-2TB/cargo-target/wheels/ram_controller-2026.212.37-cp313-cp313-macosx_11_0_arm64.whl"
VENV_PIP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/pip"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
CHAIN_PIDS="37795 95222 95298"

log() { echo "[wheel-tail] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controllers() { ps -axo pid,command 2>/dev/null \
	| grep -E "wnn.control.phased_ga|e5_residual_proof" \
	| grep -v "/usr/bin/time" | grep -v grep | grep -c python; }

log "ARMED — waiting for chains [$CHAIN_PIDS] to exit, then installing $(basename "$WHEEL")"

# 1. All three chains gone (a dead-early chain also counts as gone — the gate
#    is "no armed chain can launch another run", which pid-exit exactly means).
waited=0
while :; do
	alive=0
	for p in $CHAIN_PIDS; do kill -0 "$p" 2>/dev/null && alive=$((alive + 1)); done
	[ "$alive" -eq 0 ] && break
	[ $((waited % 3600)) -eq 0 ] && log "waiting: $alive chain(s) alive (${waited}s)"
	sleep 120; waited=$((waited + 120))
done
log "all chains exited after ${waited}s"

# 2. No controller running (the last run may outlive its chain briefly).
while [ "$(controllers)" -gt 0 ]; do
	log "waiting: controller still running"
	sleep 120
done
log "no controller running — installing"

# 3. Install + verify import.
if ! "$VENV_PIP" install --force-reinstall --no-deps "$WHEEL" >> "$LOG" 2>&1; then
	log "ABORT: pip install failed — old wheel remains active"
	exit 1
fi
abi=$("$VP" -c "import ram_controller as c; print(c.ABI_VERSION)" 2>>"$LOG") || {
	log "ABORT: import ram_controller failed after install"
	exit 1
}
log "installed OK — ram_controller ABI $abi"

# 4. Smoke ONE tiny launch (pop 6, 1 gen, ~60s) — the pre-flight recipe shape.
SMOKE_OUT="logs/controller/sn_state/WHEEL_SMOKE.out"
export WNN_STATE_SPLIT=1
if PYTHONPATH=src/wnn "$VP" -u -m wnn.control.phased_ga \
		--levels 16 --threshold-calib-tilt 30 --skip-stages bits,connections \
		--neurons-gens 1 --neurons-patience 1 --memory-gens 1 --memory-patience 1 \
		--pop 6 --num-eval-folds 1 --eval-episodes 2 --memory-eval-episodes 2 \
		--steps 200 --tilt 5.0 --report-episodes 2 --holdout-pop-sample 2 \
		--grid-bits 24 --grid-state-neurons 4 --max-state-neurons 4 \
		--max-output-neurons 128 --runs 1 --memory-mode BINARY \
		--airframe cf21_brushless --disturbance L4C --teacher lqi \
		--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i \
		--report-seeds 99990101 --base-seed 31337002 > "$SMOKE_OUT" 2>&1; then
	if grep -qE "\(state 0/[0-9]+" "$SMOKE_OUT"; then
		log "SMOKE FAILED THE MECHANISM CHECK: zero state cells populated — investigate before any launch"
		exit 2
	fi
	log "SMOKE PASSED — rc=0 and state cells populated ($SMOKE_OUT). Wheel is LIVE; safe to launch."
else
	log "SMOKE FAILED rc=$? — see $SMOKE_OUT. Do NOT launch anything on this wheel."
	exit 2
fi
