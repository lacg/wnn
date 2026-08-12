#!/usr/bin/env bash
# resume_sn_cell.sh — MANUAL resume of ONE sn>0 cell from its emergency dump.
#
# PREPPED 11/08/2026 for the case the chain's own retry budget (2) is exhausted:
# run_controller_arm now auto-resumes from emergency_stage*.yaml.gz on rc=137/143,
# but a run killed 3x is abandoned by design ("needs a fix, not a retry") and the
# chain moves on. THIS script is the fix-then-rerun half of that contract: it
# re-runs a single cell through the SAME run_controller_arm (same marker schema,
# same R1-R3 guards, same retry-resume), so a manual rescue is indistinguishable
# from a chain-run cell in the study table.
#
# Args MUST mirror scripts/sn_state_neurons_chain.sh EXACTLY (one cell of it) —
# if you change the chain's recipe, change this file too.
#
# Usage:  scripts/resume_sn_cell.sh [sn] [seed]      # defaults: 8 31337002
#   e.g.  nohup scripts/resume_sn_cell.sh 8 31337002 >/dev/null 2>&1 &
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
. "$ROOT/scripts/controller_arm_lib.sh"

SN="${1:-8}"
SEED="${2:-31337002}"
LOG="/private/tmp/sn_state.log"          # same log as the chain — one timeline
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
OUTDIR="logs/controller/sn_state"
MARKDIR="experiments/sn_state_markers"
AIRFRAME="cf21_brushless"
DIST="L4C"
TEACHER="${SN_TEACHER:-lqi}"
NEURONS_GENS=5
REPORT_SEEDS="99990101 99990102 99990103 99990104 99990105"
FEAT_PIDMIX="--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i"

export WNN_STATE_SPLIT=1   # REQUIRED — sn>0 does NOT imply it (reward_gated.py:477)

log() { echo "[sn-manual] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controllers() { ps -axo pid,command 2>/dev/null \
	| grep -E "wnn.control.phased_ga|e5_residual_proof" \
	| grep -v "/usr/bin/time" | grep -v grep | grep -c python; }

# ONE controller at a time — hard rule. Refuse to start into a busy box.
if [ "$(controllers)" -gt 0 ]; then
	log "ABORT: a controller is already running ($(controllers)) — one at a time."
	exit 3
fi

tag="S1_${TEACHER}_sn${SN}_${AIRFRAME}_${DIST}_s${SEED}"
log "########## MANUAL RESUME $tag (run_controller_arm handles dump discovery) ##########"

# run_controller_arm finds the newest emergency dump itself on retry; for the
# FIRST launch here, hand it the dump explicitly if one exists so we resume
# rather than restart (marker absent + dump present is exactly the rescue case).
RESUME_ARGS=""
dump=$(ls -t "$OUTDIR/${tag}_stages"/emergency_stage*.yaml.gz 2>/dev/null | head -1)
if [ -n "$dump" ]; then
	log "$tag: resuming from $(basename "$dump")"
	RESUME_ARGS="--resume-from-emergency $dump --resume-mode same"
else
	log "$tag: NO emergency dump found — this will run from scratch"
fi

run_controller_arm "$tag" \
	"$MARKDIR" "$OUTDIR" "$VP" log \
	"\"arm\":\"SN\",\"teacher\":\"${TEACHER}\",\"state_neurons\":${SN},\"split\":true,\"airframe\":\"${AIRFRAME}\",\"disturbance\":\"${DIST}\",\"mode\":\"BINARY\",\"neurons_gens\":${NEURONS_GENS},\"seed\":${SEED},\"code\":\"post-5f3d113c\",\"manual_resume\":true" \
	-- \
	--levels 16 --threshold-calib-tilt 30 \
	--skip-stages bits,connections --lamarckian \
	--max-cells 180000 --max-cells-strict \
	--neurons-gens "$NEURONS_GENS" --neurons-patience 3 \
	--memory-gens 120 --memory-patience 2 \
	--pop 50 --num-eval-folds 5 --check-interval 2 --magnitude-aware-patience \
	--eval-episodes 100 --memory-eval-episodes 200 \
	--steps 2000 --tilt 5.0 \
	--fit-weight-err-sq 0.4 --fit-weight-stable 0.3 \
	--fit-weight-jerk 0.2 --fit-weight-mono 0.1 \
	--report-episodes 100 --holdout-pop-sample 8 \
	--grid-bits 24 30 \
	--grid-state-neurons "$SN" --max-state-neurons "$SN" \
	--max-output-neurons 128 \
	--runs 1 --memory-mode BINARY \
	--airframe "$AIRFRAME" --disturbance "$DIST" --teacher "$TEACHER" \
	$FEAT_PIDMIX \
	--save-stage-checkpoints "$OUTDIR/${tag}_stages" \
	--report-seeds $REPORT_SEEDS \
	--base-seed "$SEED" \
	$RESUME_ARGS
rc=$?
log "$tag: manual resume finished rc=$rc"
exit "$rc"
