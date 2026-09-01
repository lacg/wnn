#!/usr/bin/env bash
# STAGE C, GAMMA=2 ARM — the levels ladder re-run with the shaped alphabet ON
# (01/09/2026, Luiz's call).
#
# WHY THIS EXISTS. Phase 1's gate went 1-1 and was settled only by the MAGNITUDE
# of the summed gate-distance delta, which b=36's large loss dominated. b=32
# genuinely PREFERRED gamma=2 (hd 0.9976 vs 1.1440). So "phase 2 carries
# gamma=1.0" is the tiebreak rule firing on n=1 per width, NOT a refutation of
# alphabet shaping — and because phase 2 then ran entirely at gamma=1, nothing in
# the levels ladder re-tests it. This script runs the SAME six points with
# gamma=2 so the ladder becomes a full 2 x 2 x 3 factorial (gamma x width x
# levels) instead of a gamma=1 slice through it.
#
# THE HYPOTHESIS THIS ARM TESTS: does shaping INTERACT with resolution? Gamma's
# whole claim is finer steps near zero at ZERO extra footprint, and n=64 (16
# levels/motor) just got inside the gate for the first time (hd 0.4034) by
# spending footprint instead. If gamma=2 at LOW n reaches what high n reaches,
# the FPGA/MCU claim keeps its cells. If gamma's benefit shrinks as levels rise
# (the alphabet is already fine near zero), the two are substitutes and the
# ladder's gamma=1 slice was the right one to run first.
#
# NO DUPLICATED RECIPE. This script does NOT re-implement the arm — a copied
# recipe block is exactly how S16 silently substituted C10. It relaunches
# scripts/sweep_ladder_gamma.sh itself with SL_FORCE_PHASE2_GAMMA=2.0, so every
# flag stays byte-identical to the gamma=1 arm and --delta-gamma is the ONLY
# thing that differs. Phase 1 re-runs nothing (its two g20 markers already
# exist, run_point skips on marker presence) and the gate re-logs the same
# verdict; the override then overrules it for phase 2 only.
#
# ⚠️ THE PATCH WINDOW. sweep_ladder_gamma.sh has no override hook yet, and it
# CANNOT be edited while bash is executing it (bash resumes at a byte offset).
# This script therefore applies the hook only AFTER the chain has exited — the
# one window in which the edit is safe. The patch is idempotent (grep-guarded)
# and is syntax-checked with `bash -n` before anything is launched.
#
# THE COMPLETION SIGNAL IS THE MARKERS, NOT THE PROCESS (controller_arm_lib.sh
# withholds a marker on watchdog kill, crash, or a clean exit with no MEMORY
# triple). If the chain is gone but its six markers are not all present, a run
# needs a human: this script FAILS CLOSED, logs which are missing, and leaves the
# box idle rather than stacking a second ladder on top of a crash.
set -u
ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1

LOG="/private/tmp/sweep_ladder_gamma2.log"
CHAIN="scripts/sweep_ladder_gamma.sh"
MARKDIR="experiments/sweepladder_markers"
SEED="${SL_SEED:-31337002}"
WIDTHS="${SL_WIDTHS:-36 32}"
# n=256 is deliberately NOT in the gamma=2 arm (Luiz, 01/09). The alphabet probe
# already banked n=256 as REFUTED on footprint, and gamma's entire claim is finer
# resolution at ZERO extra footprint — so a gamma=2 point at a shape the FPGA/MCU
# claim cannot spend tests nothing either arm can use. levels {16,24} is where
# gamma and levels actually compete, and the factorial is complete over those.
G2_NEURONS="${SL_G2_NEURONS:-64 96}"        # the gamma=2 arm's ladder
G1_NEURONS="${SL_G1_NEURONS:-64 96 256}"   # what we WAIT for (the gamma=1 arm)

log() { echo "[ladder-gamma2] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controller_pids() { pgrep -f -- "-m wnn.control.phased_ga" 2>/dev/null || true; }
chain_pids() { pgrep -f "$CHAIN" 2>/dev/null | grep -v "^$$\$" || true; }

# Single-instance guard: a second copy would double-launch the ladder.
if [ "$(pgrep -fc "sweep_ladder_gamma2_supervisor.sh" 2>/dev/null || echo 1)" -gt 1 ]; then
	log "ABORT — another instance of this supervisor is already running."
	exit 1
fi

g1_missing() {
	local miss="" n b t
	for n in $G1_NEURONS; do
		for b in $WIDTHS; do
			t="SL_C_b${b}n${n}_cf21_brushless_L4C_g10_s${SEED}.json"
			[ -f "${MARKDIR}/${t}" ] || miss="${miss} ${t}"
		done
	done
	echo "$miss"
}

log "########## ARMED — waiting for the gamma=1 ladder to complete ##########"
log "wait-for: $(g1_missing | wc -w) markers still missing; then patch + relaunch at gamma=2"

# ---- 1. WAIT for the gamma=1 chain to finish. Pure wait, never preempts.
while [ -n "$(chain_pids)" ]; do sleep 60; done
log "chain $CHAIN has exited."

# ---- 2. FAIL CLOSED if it did not bank all six markers.
MISS="$(g1_missing)"
if [ -n "$MISS" ]; then
	log "ABORT — chain gone but markers MISSING:${MISS}"
	log "A run needs a human (watchdog kill / crash / no MEMORY triple). Box left idle deliberately."
	exit 1
fi
log "all $(echo $G1_NEURONS $WIDTHS | wc -w) gamma=1 ladder markers present."

# ---- 3. Let any straggler controller finish. Pure wait — one at a time.
while [ -n "$(controller_pids)" ]; do sleep 30; done

# ---- 4. PATCH the override hook in. Safe now: nothing is executing the file.
if grep -q "SL_FORCE_PHASE2_GAMMA" "$CHAIN"; then
	log "override hook already present — no patch needed."
else
	cp "$CHAIN" "${CHAIN}.pre-g2-$(date -u +%Y%m%dT%H%M%SZ).bak"
	"/usr/bin/python3" - "$CHAIN" <<'PY'
import sys
p = sys.argv[1]
src = open(p).read()
anchor = '# ---- PHASE 2: the levels ladder, NEURON-MAJOR, cheapest resolution first.'
hook = (
	'# An explicit override for a DELIBERATE re-run of phase 2 at another gamma\n'
	'# (stage C gamma=2 arm, 01/09/2026). The gate stays exactly as it was and its\n'
	'# verdict is still logged above; this only overrules what phase 2 CARRIES, and\n'
	'# only when the operator asks for it by name.\n'
	'if [ -n "${SL_FORCE_PHASE2_GAMMA:-}" ]; then\n'
	'\tPHASE2_GAMMA="$SL_FORCE_PHASE2_GAMMA"\n'
	'\tlog "OVERRIDE: SL_FORCE_PHASE2_GAMMA set — phase 2 carries gamma=${PHASE2_GAMMA} regardless of the verdict."\n'
	'fi\n\n'
)
assert src.count(anchor) == 1, "anchor not found exactly once"
open(p, 'w').write(src.replace(anchor, hook + anchor))
PY
	if ! bash -n "$CHAIN"; then
		log "ABORT — patched $CHAIN FAILS bash -n. Restoring backup, launching nothing."
		cp "$(ls -t ${CHAIN}.pre-g2-*.bak | head -1)" "$CHAIN"
		exit 1
	fi
	log "override hook patched in and syntax-checked."
fi

# ---- 5. RELAUNCH the same chain at gamma=2. Phase 1 skips (markers exist).
log "===== LAUNCHING the gamma=2 ladder: widths=[$WIDTHS] neurons=[$G2_NEURONS] ====="
# macOS has NO setsid; nohup + & is the house idiom (the chain re-parents to
# PID 1 when this supervisor exits, which it does immediately after).
SL_FORCE_PHASE2_GAMMA=2.0 SL_NEURONS="$G2_NEURONS" SL_WIDTHS="$WIDTHS" SL_SEED="$SEED" \
	nohup bash "$CHAIN" >/dev/null 2>&1 &
log "launched pid $! — tags will be SL_C_b{W}n{N}_cf21_brushless_L4C_g20_s${SEED}"
