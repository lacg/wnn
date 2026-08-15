#!/usr/bin/env bash
# SPECIALIST PROGRAMME ROUND 1 (14/08/2026, Luiz) — 5 arms, one run each,
# seed 31337002, stage-1 plant (translation ON, lambda_alt=16 = the s31337002
# winner so the axis under test is CONNECTIVITY, nothing else).
#
#   B   b=15 SPREAD, 256-neuron cap    — the size/count trade alone
#   C2  b=15 MIN_PER_CLUSTER(2)        — Luiz's min-2-with-donation (interval floor)
#   C3  b=15 MIN_PER_CLUSTER(3)        — the dose response: is min-3 > min-2?
#   D   b=30 SPREAD + FULL K-WINDOW    — does raw 4ms history help at all?
#   E   sn>0 (search 4..8) + k=1       — the state layer as the ONLY memory (DFA)
#
# Arm A (the banked S1L_lam16_..._s31337002 marker: 128n b30 spread, 97.8/1.76/1.34)
# is the shared control — free, not re-flown.
#
# PRE-REGISTERED READ (task #15): headline held-out triple + pos= column, arms
# B/C2/C3 footprint-compared on total populated cells (the FPGA-real budget, in
# each marker's fpga field), C2 vs B isolates specialization from size, C3 vs C2
# is the dose. n=1 => measurement, not verdict; round 2 = second seed on survivors.
#
# INTERLEAVED per feedback_sweeps_always_interleave: one run of EVERY arm before
# any arm earns a second seed.
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
. "$ROOT/scripts/controller_arm_lib.sh"

LOG="/private/tmp/specialist_round1.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
OUTDIR="logs/controller/specialist1"
MARKDIR="experiments/specialist1_markers"
AIRFRAME="cf21_brushless"
DIST="L4C"
SEED="${SP1_SEED:-31337002}"
REPORT_SEEDS="99990101 99990102 99990103 99990104 99990105"

FEAT_STAGE1="--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i \
	--obs-collective-cmd --obs-alt-err --obs-vz"
S16_WEIGHTS="--fit-weight-err-sq 0.25 --fit-weight-steady 0.35 --fit-weight-stable 0.20 --fit-weight-jerk 0.15 --fit-weight-mono 0.05"

log() { echo "[spec1] $(date -u +%FT%TZ) $*" >> "$LOG"; }

mkdir -p "$OUTDIR" "$MARKDIR"
log "########## ARMED — specialist round 1, 5 arms, seed=$SEED ##########"

run_arm_common() {
	local tag="$1" extra_json="$2"; shift 2
	mkdir -p "$OUTDIR/ckpt/$tag"
	# shellcheck disable=SC2086
	run_controller_arm "$tag" \
		"$MARKDIR" "$OUTDIR" "$VP" log \
		"$extra_json" \
		-- \
		--levels 16 --skip-stages bits,connections --lamarckian \
		--max-cells 180000 --max-cells-strict \
		--save-stage-checkpoints "$OUTDIR/ckpt/$tag" \
		--neurons-gens 5 --neurons-patience 3 \
		--memory-gens 120 --memory-patience 2 \
		--pop 50 --num-eval-folds 5 --check-interval 2 --magnitude-aware-patience \
		--eval-episodes 100 --memory-eval-episodes 200 \
		--steps 2000 --tilt 5.0 \
		$S16_WEIGHTS \
		--report-episodes 100 --holdout-pop-sample 8 \
		--runs 1 --memory-mode BINARY \
		--airframe "$AIRFRAME" --disturbance "$DIST" --teacher mpcof \
		$FEAT_STAGE1 \
		--translation --fit-weight-alt 16 \
		--report-seeds $REPORT_SEEDS \
		--base-seed "$SEED" \
		"$@"
	log "$tag finished rc=$?"
}

# --- B: b=15 spread, more neurons (size/count trade, no structure) -----------
log "===== ARM B (b15 spread, 256n cap) ====="
run_arm_common "SP1_B_b15spread_${AIRFRAME}_${DIST}_s${SEED}" \
	"\"arm\":\"SPEC1_B\",\"conn_policy\":\"spread\",\"bits\":15,\"seed\":${SEED}" \
	--grid-bits 15 --grid-state-neurons 0 --max-state-neurons 0 --max-output-neurons 256

# --- C2: b=15 MIN_PER_CLUSTER(2) ---------------------------------------------
log "===== ARM C2 (b15 min2) ====="
run_arm_common "SP1_C2_b15min2_${AIRFRAME}_${DIST}_s${SEED}" \
	"\"arm\":\"SPEC1_C2\",\"conn_policy\":\"min2\",\"bits\":15,\"seed\":${SEED}" \
	--grid-bits 15 --grid-state-neurons 0 --max-state-neurons 0 --max-output-neurons 256 \
	--conn-policy min2

# --- C3: b=15 MIN_PER_CLUSTER(3) ---------------------------------------------
log "===== ARM C3 (b15 min3) ====="
run_arm_common "SP1_C3_b15min3_${AIRFRAME}_${DIST}_s${SEED}" \
	"\"arm\":\"SPEC1_C3\",\"conn_policy\":\"min3\",\"bits\":15,\"seed\":${SEED}" \
	--grid-bits 15 --grid-state-neurons 0 --max-state-neurons 0 --max-output-neurons 256 \
	--conn-policy min3

# --- D: b=30 spread over the FULL K-frame window ------------------------------
log "===== ARM D (b30 full-window) ====="
run_arm_common "SP1_D_b30fullwin_${AIRFRAME}_${DIST}_s${SEED}" \
	"\"arm\":\"SPEC1_D\",\"conn_policy\":\"spread\",\"bits\":30,\"output_full_window\":true,\"seed\":${SEED}" \
	--grid-bits 24 30 --grid-state-neurons 0 --max-state-neurons 0 --max-output-neurons 128 \
	--output-full-window

# --- E: sn>0 with k=1 — the state layer as the ONLY memory --------------------
log "===== ARM E (sn>0, k=1) ====="
run_arm_common "SP1_E_snk1_${AIRFRAME}_${DIST}_s${SEED}" \
	"\"arm\":\"SPEC1_E\",\"conn_policy\":\"spread\",\"bits\":30,\"input_window_k\":1,\"seed\":${SEED}" \
	--grid-bits 24 30 --grid-state-neurons "4 8" --max-state-neurons 8 --max-output-neurons 128 \
	--input-window-k 1

log "########## ROUND 1 COMPLETE — $(ls "$MARKDIR" | wc -l) markers ##########"
