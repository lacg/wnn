#!/usr/bin/env bash
# Post-P2 gate: install the controller wheel, run the CPU/GPU parity sweep that the
# QUAD+ANTAGONIST rows added, and only then hand the box to P3.
#
# WHY THIS EXISTS AT ALL. p3_chain.sh triggers on P2_ALL_DONE.marker, and P2 writes
# that marker the moment its last seed lands. P3 then waits 60s, checks the box is
# clear, and starts. Sixty seconds is not enough to install a wheel and run a parity
# sweep, and worse: `maturin develop` replacing ram_controller while P3 spawns its
# first cell is the source/wheel-skew failure — P3's later seeds would run a
# different binary than its first. So P3 is held (its parked chain is stopped before
# this arms) and relaunched HERE, after the wheel is settled.
#
# WHY THE INSTALL HAS TO WAIT FOR P2 AT ALL. The parity rows added in ee0d3812 are
# source-only; executing them needs an installed wheel. Installing mid-P2 would have
# given P2's seed 2 a different binary than seed 1 — the same skew, inside one
# experiment instead of between two.
#
# ORDER IS THE POINT: P2 done -> install -> parity -> P3. A parity failure STOPS the
# chain and does NOT start P3, because a controller whose GPU and CPU disagree is not
# something to hand nine more cells to.
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
LOG="/private/tmp/p2_post_chain.log"
DONE_MARKER="experiments/p2_markers/P2_ALL_DONE.marker"
ACCEL="src/wnn/ram/strategies/accelerator"
PARITY_OUT="/private/tmp/p2_post_parity.txt"
# set -u + a detached process = PYTHONPATH is unset, and "$PYTHONPATH" is then a
# HARD ERROR, not an empty string. That killed the first run of this gate after the
# wheel had already installed. Normalise once here so no later use can trip on it.
PYTHONPATH="${PYTHONPATH:-}"

log() { echo "[post] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controllers() { ps -axo pid,command 2>/dev/null | grep "wnn.control.phased_ga" \
	| grep -v "/usr/bin/time" | grep -v grep | grep -c python; }

log "########## ARMED — waiting for $DONE_MARKER ##########"

# ---- 1. wait for P2 to publish a COMPLETE result ----------------------------
while [ ! -f "$DONE_MARKER" ]; do
	sleep 120
done
log "P2_ALL_DONE present"

# ---- 2. the box must be genuinely idle before a wheel swap ------------------
sleep 45
if [ "$(controllers)" -gt 0 ]; then
	log "ABORT: P2 says done but $(controllers) controller(s) alive — refusing to swap the \
wheel under a live process. P3 NOT started. Needs a human."
	exit 3
fi
log "box clear: controllers=0"

# ---- 3. install the controller wheel ----------------------------------------
# Controller wheel only: the IDS worker never imports ram_controller, so this does
# not need a worker swap and the live worker is unaffected.
log "installing ram_controller..."
if ! ( unset CONDA_PREFIX; . "$ROOT/wnn/bin/activate"; cd "$ROOT/$ACCEL" \
       && maturin develop --release -m controller/Cargo.toml ) >> "$LOG" 2>&1; then
	log "ABORT: wheel install FAILED — P3 NOT started."
	exit 4
fi
abi=$( ( . "$ROOT/wnn/bin/activate"; python3 -c "import ram_controller as c; print(c.ABI_VERSION)" ) 2>/dev/null )
log "installed, ram_controller ABI=$abi"

# ---- 4. the parity sweep, including the QUAD+ANTAGONIST rows ----------------
log "running split_train_loop parity sweep..."
( . "$ROOT/wnn/bin/activate"; export PYTHONPATH="$ROOT/src/wnn:$PYTHONPATH"
  python3 - <<'PY'
import ram_controller as c
rows = c.run_controller_split_train_loop_parity_test()
bad = [r for r in rows if not r[1]]
for name, ok, detail in rows:
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}")
print(f"\n{len(rows)-len(bad)}/{len(rows)} passed")
raise SystemExit(1 if bad else 0)
PY
) > "$PARITY_OUT" 2>&1
rc=$?
cat "$PARITY_OUT" >> "$LOG"
if [ $rc -ne 0 ]; then
	log "########## PARITY FAILED (rc=$rc) — P3 NOT STARTED. See $PARITY_OUT ##########"
	exit 5
fi
log "parity PASSED — $(grep -c '^ok  ' "$PARITY_OUT") rows"

# ---- 5. hand the box to P3 --------------------------------------------------
# P2_ALL_DONE already exists, so p3_chain falls straight through its wait loop into
# its own box check. Launched detached so it survives this script exiting.
log "starting P3"
P3_CONFIRMED=1 nohup env P3_CONFIRMED=1 bash "$ROOT/scripts/p3_chain.sh" \
	>> /private/tmp/p3_chain.out 2>&1 &
sleep 10
if pgrep -f p3_chain.sh >/dev/null 2>&1; then
	log "########## POST-P2 CHAIN DONE — P3 running ##########"
else
	log "WARNING: P3 did not come up — needs a human."
	exit 6
fi
