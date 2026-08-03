#!/usr/bin/env bash
# P2 handoff: sweep the OUTPUT-NEURON axis (PWM decode resolution) on the nf=17
# pidmix+tilt arm, starting automatically once the dfa1l 40-cell sweep finishes.
#
# TRIGGER — ALL_DONE.marker, not a cell count.
# run_dfa_1layer_study.sh publishes experiments/dfa1l_markers/ALL_DONE.marker after
# its last cell (study:187); the supervisor keys on the same file and exits 10 when
# it appears (supervisor:63,109). Counting *.json instead would fire early, because
# the marker dir also holds 13 auxiliary files (baselines, gap/rescore probes, split
# smokes) that no cell ever writes — 40 files there is NOT 40 finished cells.
#
# WHY THIS WAITS INSTEAD OF SEIZING (the opposite of p1_chain_20260802.sh):
# P1 had a deadline — one pre-registered question blocking the feature decision —
# so it killed a live cell and paid for it. P2 has no such urgency, and the sweep
# is the paper's main table. So this takes the box only when the sweep has finished
# with it. No cell is ever killed by this script; if it never fires, nothing is lost.
#
# RECIPE PROVENANCE: reconstructed from the A8 command line after
# scratchpad/RECIPE_phase2.md was lost with the scratchpad, confirmed by Luiz
# 03/08/2026. --grid-output-neurons 64 96 128 = 16/24/32 levels per motor; all are
# multiples of the BINARY output quantum (2·num_motors = 8), so none floor-divides
# into an odd level count the antagonist E/I split cannot hold. K stays at the
# default 4 — the window is P3's variable, not P2's.
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
# shellcheck source=controller_arm_lib.sh
. "$ROOT/scripts/controller_arm_lib.sh"

LOG="/private/tmp/p2_chain.log"
DONE_MARKER="experiments/dfa1l_markers/ALL_DONE.marker"
STUDY="scripts/run_dfa_1layer_study.sh"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
OUTDIR="logs/controller/p2"
MARKDIR="experiments/p2_markers"
SEEDS="${P2_SEEDS:-31337002 31337003 31337004}"
REPORT_SEEDS="99990101 99990102 99990103 99990104 99990105"
P2_CONFIRMED="${P2_CONFIRMED:-0}"

log() { echo "[p2] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controllers() { ps -axo pid,command 2>/dev/null | grep "wnn.control.phased_ga" \
	| grep -v "/usr/bin/time" | grep -v grep | grep -c python; }

if [ "$P2_CONFIRMED" != "1" ]; then
	log "REFUSING TO ARM — set P2_CONFIRMED=1 once the recipe is signed off"
	exit 2
fi

mkdir -p "$OUTDIR" "$MARKDIR"
log "########## ARMED — waiting for $DONE_MARKER ##########"

# ---- 1. wait for the sweep to declare itself finished -----------------------
while [ ! -f "$DONE_MARKER" ]; do
	sleep 300
done
log "ALL_DONE.marker present — the sweep is finished"

# ---- 2. confirm the box is actually free ------------------------------------
# The supervisor exits itself on ALL_DONE; the driver has returned. Give both a
# moment, then verify rather than assume. If anything is still running we do NOT
# kill it — an unexpected live controller means the world is not what we think.
sleep 60
if [ "$(controllers)" -gt 0 ] || pgrep -f "$STUDY" >/dev/null 2>&1; then
	log "ABORT: sweep says done but controllers=$(controllers) drivers=$(pgrep -fc "$STUDY" 2>/dev/null || echo 0) \
— refusing to run alongside. Needs a human."
	exit 3
fi
log "box clear: controllers=0 drivers=0"

# ---- 3. P2: the output-neuron (decode-resolution) sweep ---------------------
# --max-output-neurons must stay >= the grid max or phased_ga refuses outright
# (it will not silently clamp a shape the grid was told to explore).
for seed in $SEEDS; do
	run_controller_arm "P2_on_L2D_s${seed}" "$MARKDIR" "$OUTDIR" "$VP" log \
		"\"arm\":\"P2\",\"axis\":\"output_neurons\",\"disturbance\":\"L2D\",\"features\":\"pidmix_tilt\",\"mode\":\"BINARY\",\"seed\":${seed}" \
		-- \
		--levels 16 --skip-stages bits,connections --lamarckian \
		--max-cells 180000 \
		--neurons-gens 60 --neurons-patience 3 \
		--memory-gens 120 --memory-patience 2 \
		--pop 50 --num-eval-folds 5 --check-interval 2 --magnitude-aware-patience \
		--eval-episodes 100 --memory-eval-episodes 200 \
		--steps 2000 --tilt 5.0 \
		--fit-weight-err-sq 0.4 --fit-weight-stable 0.3 \
		--fit-weight-jerk 0.2 --fit-weight-mono 0.1 \
		--report-episodes 100 --holdout-pop-sample 8 \
		--grid-bits 24 30 \
		--grid-output-neurons 64 96 128 \
		--max-output-neurons 128 \
		--grid-state-neurons 0 --max-state-neurons 0 \
		--runs 1 --teacher lqr --memory-mode BINARY \
		--disturbance L2D \
		--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw \
		--obs-yaw-err --obs-yaw-err-i --obs-tilt-p --obs-tilt-i \
		--report-seeds $REPORT_SEEDS \
		--base-seed "$seed"
done

# ---- 4. publish completion, so p3_chain can key on it -----------------------
# Only when EVERY seed has a marker. A partial P2 must not release P3: P3 is a
# 9-cell run and starting it on top of an unfinished P2 would lose both.
got=$(ls -1 "$MARKDIR"/P2_on_L2D_s*.json 2>/dev/null | wc -l | tr -d ' ')
want=$(echo $SEEDS | wc -w | tr -d ' ')
if [ "$got" = "$want" ]; then
	date -u +%FT%TZ > "${MARKDIR}/P2_ALL_DONE.marker"
	log "########## P2 CHAIN DONE — ${got}/${want} markers, P2_ALL_DONE published ##########"
else
	log "########## P2 INCOMPLETE — ${got}/${want} markers, P2_ALL_DONE withheld (P3 stays parked) ##########"
	exit 4
fi
