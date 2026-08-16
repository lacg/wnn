#!/usr/bin/env bash
# SPECIALIST PROGRAMME ROUND 2 (16/08/2026, Luiz) — the width/resolution/redundancy
# factorial + the coverage + temporal-coverage arms. 8 arms, one run each, seed 31337002, same stage-1 plant as round 1
# (translation ON, lambda_alt=16), same 180k cell budget — so B and A stay valid
# comparators. Runs on the 16/08 wheel (alt_err row 14 + --target-levels); legacy
# semantics are bit-identical on that wheel (parity anchors green), so cross-round
# comparison is honest.
#
#   FR1  b=18 FRAMED1, 240n, 40 ms     — each neuron covers ONE frame COMPLETELY
#                                        (all 18 features, 1 threshold each); the
#                                        population covers time 8:4:2:1. The repair
#                                        for arm D's dilution, from the other side.
#   H    b=15 SPREAD, 1024n            — resolution ALONE vs B (256n b15): does 4x
#                                        finer PWM help when the address width is
#                                        what b15 gives you?
#   R32  H + --target-levels 32        — Luiz's decoupling, dose endpoint: 1024
#                                        neurons, 32 coarse thresholds/motor, x8
#                                        redundancy averaging in the sum decode
#   Q32  b=15 SPREAD, 256n + T=32      — the averaging pair: R32 vs Q32 at EQUAL
#                                        T isolates pure population averaging
#   G    b=20 SPREAD, 256n             — +5 address bits vs B, all else equal
#   K18  b=18 MIN1, 128n               — FULL FEATURE COVERAGE (Luiz 16/08): with
#                                        18 features x 8 bpf, b=18 under min1 gives
#                                        every feature EXACTLY ONE threshold. The
#                                        opposite extreme from min2/min3: maximum
#                                        feature breadth, minimum per-feature depth.
#   K18s b=18 SPREAD, 128n             — the CONTROL for K18. Same width, same
#                                        neurons, uniform draw: covers only ~12.0
#                                        of 18 features in expectation, so
#                                        K18 - K18s isolates COVERAGE from width.
#   K15  b=15 MIN1, 128n               — coverage dose: 15 of 18 features, 1 each
#
# PRE-REGISTERED READ: headline held-out triple + pos= + MONO/NEURON (the raw
# mono_viol count scales with neuron count — 1024n arms MUST be read per-neuron
# or averaging looks like it increases violations by bookkeeping).
#   - mechanism (thresholds too fine for the addressing): mono/n falls from
#     H (native 256 thresholds) to R32 (32) AND steady follows. ENDPOINTS ONLY —
#     R64, the dose middle, was dropped 16/08 to keep the round at ~31h, so a
#     non-monotone dose cannot be detected here; a positive H->R32 result should
#     be followed by R64 before the mechanism is called established.
#   - averaging (Luiz's hypothesis): R32 beats Q32 at equal T
#   - width: G - B = +5 address bits alone (F, the b20 x 512n size point, was
#     dropped 16/08 — size at b20 is untested this round)
#   - temporal coverage: FR1 vs arm D (SP1, b30 full window, 91,993 B, steady
#     10.17) — same wide input space, opposite budget policy. FR1 vs arm A
#     (1-frame b30, steady 1.34) says whether time is worth ANY of the budget.
#   - coverage: K18 - K18s = feature coverage at fixed width/neurons; K18 - K15 =
#     the coverage dose (18/18 vs 15/18 features). Spread's expected coverage:
#     b=15 -> 10.7/18, b=18 -> 12.0/18, b=30 -> 15.4/18 (so 33% of
#     (neuron,feature) pairs see NOTHING at b=18 under the legacy draw).
# n=1 => measurement, not verdict.
#
# INTERLEAVED per feedback_sweeps_always_interleave: each hypothesis gets one run
# before any gets a second point (FR1, H, R32, Q32, G, then K18, K18s, K15).
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
log "########## ARMED — specialist round 2 (width/resolution/redundancy/coverage/temporal), 8 arms, seed=$SEED ##########"

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

# --- FR1: framed1 — each neuron covers ONE frame completely -------------------
# Luiz's temporal-coverage hypothesis (16/08). Arm D refuted SPREADING a neuron's
# budget across the window (steady 1.34 -> 10.17 at 4 ms, 4x per-frame coverage
# lost). FR1 spends it the other way: every neuron picks ONE frame and covers all
# 18 features of it (b=18 min1 within the frame), and the POPULATION covers time
# via recency weights 8:4:2:1 => ~128/64/32/16 of 240 neurons on frames
# newest..oldest. 240n = 60 levels/motor (even, BINARY antagonist). Frame drawn
# PER NEURON, not by index: the layout is motor-major, so an index schedule would
# leave motor 3 commanding on 30 ms-old state.
log "===== ARM FR1 (framed1 b18, 240n, k4 stride10 = 40 ms — one frame per neuron, fully covered) ====="
run_arm_common "SP2_FR1_framed1b18n240_${AIRFRAME}_${DIST}_s${SEED}" \
	'"arm":"SPEC2_FR1","conn_policy":"framed1","bits":18,"neurons":240,"input_window_k":4,"frame_stride":10,"target_levels":0' \
	--grid-bits 18 --max-output-neurons 240 --conn-policy framed1 \
	--output-full-window --input-window-k 4 --frame-stride 10

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

# --- K18: b=18 min1 — every feature exactly one threshold ---------------------
log "===== ARM K18 (b18 min1, 128n — full feature coverage, 1 threshold each) ====="
run_arm_common "SP2_K18_b18min1_${AIRFRAME}_${DIST}_s${SEED}" \
	'"arm":"SPEC2_K18","conn_policy":"min1","bits":18,"neurons":128,"target_levels":0' \
	--grid-bits 18 --max-output-neurons 128 --conn-policy min1

# --- K18s: the spread control at the same width/neurons ----------------------
log "===== ARM K18s (b18 spread, 128n — the coverage control for K18) ====="
run_arm_common "SP2_K18s_b18spread_${AIRFRAME}_${DIST}_s${SEED}" \
	'"arm":"SPEC2_K18s","conn_policy":"spread","bits":18,"neurons":128,"target_levels":0' \
	--grid-bits 18 --max-output-neurons 128

# --- K15: coverage dose, 15 of 18 features -----------------------------------
log "===== ARM K15 (b15 min1, 128n — 15 of 18 features, 1 threshold each) ====="
run_arm_common "SP2_K15_b15min1_${AIRFRAME}_${DIST}_s${SEED}" \
	'"arm":"SPEC2_K15","conn_policy":"min1","bits":15,"neurons":128,"target_levels":0' \
	--grid-bits 15 --max-output-neurons 128 --conn-policy min1

log "########## SPECIALIST ROUND 2 COMPLETE — $(ls "$MARKDIR" | wc -l) markers ##########"
