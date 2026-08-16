#!/usr/bin/env bash
# SPECIALIST PROGRAMME ROUND 2 (16/08/2026, Luiz) — the width/resolution/redundancy
# factorial. 6 arms, one run each, seed 31337002, same stage-1 plant as round 1
# (translation ON, lambda_alt=16), same 180k cell budget — so B and A stay valid
# comparators. Runs on the 16/08 wheel (alt_err row 14 + --target-levels); legacy
# semantics are bit-identical on that wheel (parity anchors green), so cross-round
# comparison is honest.
#
#   H    b=15 SPREAD, 1024n            — resolution ALONE vs B (256n b15): does 4x
#                                        finer PWM help when the address width is
#                                        what b15 gives you?
#   R32  H + --target-levels 32        — Luiz's decoupling, dose endpoint: 1024
#                                        neurons, 32 coarse thresholds/motor, x8
#                                        redundancy averaging in the sum decode
#   Q32  b=15 SPREAD, 256n + T=32      — the averaging pair: R32 vs Q32 at EQUAL
#                                        T isolates pure population averaging
#   G    b=20 SPREAD, 256n             — +5 address bits vs B, all else equal
#   R64  H + --target-levels 64        — the dose middle point
#   F    b=20 SPREAD, 512n             — width x size middle
#
# PRE-REGISTERED READ: headline held-out triple + pos= + MONO/NEURON (the raw
# mono_viol count scales with neuron count — 1024n arms MUST be read per-neuron
# or averaging looks like it increases violations by bookkeeping).
#   - mechanism (thresholds too fine for the addressing): mono/n falls
#     monotonically H -> R64 -> R32 AND steady follows
#   - averaging (Luiz's hypothesis): R32 beats Q32 at equal T
#   - width: G - B = +5 bits alone; F - G = size at b20
# n=1 => measurement, not verdict.
#
# INTERLEAVED per feedback_sweeps_always_interleave: the four hypotheses each get
# a run before any gets its second point (H, R32, Q32, G, then R64, F).
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
. "$ROOT/scripts/controller_arm_lib.sh"

LOG="/private/tmp/specialist_round2.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
OUTDIR="logs/controller/specialist2"
MARKDIR="experiments/specialist2_markers"
AIRFRAME="cf21_brushless"
DIST="L4C"
SEED="${SP2_SEED:-31337002}"
REPORT_SEEDS="99990101 99990102 99990103 99990104 99990105"

FEAT_STAGE1="--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i \
	--obs-collective-cmd --obs-alt-err --obs-vz"
S16_WEIGHTS="--fit-weight-err-sq 0.25 --fit-weight-steady 0.35 --fit-weight-stable 0.20 --fit-weight-jerk 0.15 --fit-weight-mono 0.05"

log() { echo "[spec2] $(date -u +%FT%TZ) $*" >> "$LOG"; }

mkdir -p "$OUTDIR" "$MARKDIR"
log "########## ARMED — specialist round 2 (width/resolution/redundancy factorial), 6 arms, seed=$SEED ##########"

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
		--grid-state-neurons 0 --max-state-neurons 0 \
		--report-seeds $REPORT_SEEDS \
		--base-seed "$SEED" \
		"$@"
	log "$tag finished rc=$?"
}

# --- H: b=15 spread, 1024n — resolution alone vs B ---------------------------
log "===== ARM H (b15 spread, 1024n — resolution alone) ====="
run_arm_common "SP2_H_b15n1024_${AIRFRAME}_${DIST}_s${SEED}" \
	'"arm":"SPEC2_H","conn_policy":"spread","bits":15,"neurons":1024,"target_levels":0' \
	--grid-bits 15 --max-output-neurons 1024

# --- R32: 1024n, T=32 — the decoupling, dose endpoint ------------------------
log "===== ARM R32 (b15 1024n, target-levels 32) ====="
run_arm_common "SP2_R32_b15n1024t32_${AIRFRAME}_${DIST}_s${SEED}" \
	'"arm":"SPEC2_R32","conn_policy":"spread","bits":15,"neurons":1024,"target_levels":32' \
	--grid-bits 15 --max-output-neurons 1024 --target-levels 32

# --- Q32: 256n, T=32 — the averaging pair for R32 ----------------------------
log "===== ARM Q32 (b15 256n, target-levels 32) ====="
run_arm_common "SP2_Q32_b15n256t32_${AIRFRAME}_${DIST}_s${SEED}" \
	'"arm":"SPEC2_Q32","conn_policy":"spread","bits":15,"neurons":256,"target_levels":32' \
	--grid-bits 15 --max-output-neurons 256 --target-levels 32

# --- G: b=20 spread, 256n — +5 address bits vs B -----------------------------
log "===== ARM G (b20 spread, 256n — +5 bits vs B) ====="
run_arm_common "SP2_G_b20n256_${AIRFRAME}_${DIST}_s${SEED}" \
	'"arm":"SPEC2_G","conn_policy":"spread","bits":20,"neurons":256,"target_levels":0' \
	--grid-bits 20 --max-output-neurons 256

# --- R64: 1024n, T=64 — the dose middle --------------------------------------
log "===== ARM R64 (b15 1024n, target-levels 64) ====="
run_arm_common "SP2_R64_b15n1024t64_${AIRFRAME}_${DIST}_s${SEED}" \
	'"arm":"SPEC2_R64","conn_policy":"spread","bits":15,"neurons":1024,"target_levels":64' \
	--grid-bits 15 --max-output-neurons 1024 --target-levels 64

# --- F: b=20 spread, 512n — width x size middle ------------------------------
log "===== ARM F (b20 spread, 512n) ====="
run_arm_common "SP2_F_b20n512_${AIRFRAME}_${DIST}_s${SEED}" \
	'"arm":"SPEC2_F","conn_policy":"spread","bits":20,"neurons":512,"target_levels":0' \
	--grid-bits 20 --max-output-neurons 512

log "########## SPECIALIST ROUND 2 COMPLETE — $(ls "$MARKDIR" | wc -l) markers ##########"
