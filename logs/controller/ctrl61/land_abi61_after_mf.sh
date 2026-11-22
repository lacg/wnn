#!/bin/bash
# CTRL-67: land the f32 Mahony estimator (ABI 61) once the model-free block (CTRL-66) ends.
# Luiz 22/11: the new-lineage control will be a MODEL-FREE arm (N0 vs NR picked from the
# n=5 readout) -> this script ONLY merges + installs + verifies. It NEVER flies anything.
# Log lines go to logs/controller/ctrl48/runs.log prefixed 'F32:'.
set -u
ROOT=/Users/lacg/wnn
RUNS=$ROOT/logs/controller/ctrl48/runs.log
PY=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python
WHEEL=$ROOT/logs/controller/ctrl61/ram_controller-2026.212.37-cp311-abi3-macosx_11_0_arm64.whl
START_LINE=${START_LINE:?set START_LINE = runs.log line count at arm time}
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) F32: $*" >> "$RUNS"; }

log "armed (pid $$): waiting for the block end marker after line $START_LINE"
while :; do
	NEWTAIL=$(tail -n +"$((START_LINE + 1))" "$RUNS" | grep -v " F32: ")
	if echo "$NEWTAIL" | grep -q "MF ABORT"; then log "stand down: MF ABORT seen, nothing landed"; exit 1; fi
	if echo "$NEWTAIL" | grep -q "MF DONE"; then break; fi
	sleep 60
done
# the block driver must be gone and no controller child alive
while pgrep -f "ab_n0_nr.s[h]" >/dev/null || pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null; do sleep 30; done

cd "$ROOT" || exit 1
if ! git merge --no-ff ctrl-f32-estimator -m "Merge ctrl-f32-estimator: ABI 61 f32 Mahony estimator (CTRL-67)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Exc8QwHhxZqBnX4jPhZ3aZ" >> "$RUNS" 2>&1; then
	git merge --abort 2>/dev/null
	log "REFUSED: merge failed (aborted), ABI 60 left installed"; exit 1
fi
log "merged $(git rev-parse --short HEAD)"

if ! "$PY" -m pip install --force-reinstall --no-deps -q "$WHEEL" >> "$RUNS" 2>&1; then
	log "REFUSED: pip install failed — tree merged but wheel NOT installed (fix by hand)"; exit 1
fi
ABI=$(cd /tmp && "$PY" -c "import ram_controller as c; print(c.ABI_VERSION)" 2>&1)
if [ "$ABI" != "61" ]; then log "REFUSED: installed ABI reads '$ABI', expected 61"; exit 1; fi
FACADE=$(cd /tmp && PYTHONPATH="$ROOT/src" "$PY" -c "import wnn.control._accel as a; print(a.EXPECTED_ABI, a.ABI_VERSION)" 2>&1 | tail -1)
if [ "$FACADE" != "61 61" ]; then log "REFUSED: facade (EXPECTED_ABI ABI_VERSION) reads '$FACADE', expected '61 61'"; exit 1; fi
log "facade import OK: EXPECTED_ABI=61 ABI_VERSION=61"

git push origin HEAD >> "$RUNS" 2>&1 && log "pushed" || log "push FAILED (merge is local only)"
log "ABI 61 LANDED — no flight armed; re-fly waits for Luiz's model-free arm pick (N0 vs NR)"
