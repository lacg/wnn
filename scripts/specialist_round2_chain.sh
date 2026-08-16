#!/usr/bin/env bash
# SPECIALIST PROGRAMME ROUND 2 (16/08/2026, Luiz) — the width/resolution/redundancy
# factorial + the coverage + temporal-coverage arms. 10 arms / 13 runs, same
# stage-1 plant as round 1 (translation ON, lambda_alt=16), same 180k cell budget
# — so B and A stay valid comparators. Runs on the 16/08 wheel (alt_err row 14 +
# --target-levels); legacy semantics are bit-identical on that wheel (parity
# anchors green), so cross-round comparison is honest.
#
#   FQ18/FQ36/FQ30 (16/08 evening, Luiz — REPLACES FR1) — the framed-window
#     family, 2 seeds x 3 widths. 240n, k=4 stride 10 (frames at t-0/-10/-20/-30
#     ms), EXACT window quotas newest->oldest 128/64/32/16 (32/16/8/4 per motor,
#     deterministic — _framed1_slot_schedule, not the weighted draw). Every
#     neuron is frame-pure and min1-covers its frame; the width is the dose:
#       FQ18  b=18  all 18 features x exactly 1 threshold
#       FQ30  b=30  12 features x 2 + 6 x 1 (1.67/feature)  — arm A's width
#       FQ36  b=36  all 18 features x exactly 2 thresholds
#     D40 (steady 36.70 vs D 10.17) showed mixing stale frames into ONE address
#     actively misleads; FQ asks whether frame-PURE neurons make 40 ms pay.
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
#   - temporal coverage (FQ family, n=2/width): FQ vs D40 (same 40 ms window,
#     same 576-bit space, opposite budget policy — D40 steady 36.70) is the
#     frame-purity test; FQ vs arm A (1-frame b30, steady 1.34) says whether
#     time is worth ANY of the budget; FQ18->FQ30->FQ36 is the within-frame
#     depth dose at full coverage (1.0/1.67/2.0 thresholds per feature).
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

# --- FQ family: framed windows with EXACT quotas, width dose (replaces FR1) ---
# Luiz's 16/08-evening spec: 2 seeds x (4 windows, 128/64/32/16 newest->oldest,
# deterministic per motor block) x [b18 = 1 thr/feature; b36 = 2 thr/feature;
# b30 = arm A's width, 12x2+6x1]. Every neuron frame-pure + min1 over its frame;
# 240n = 60 levels/motor (even, BINARY antagonist). Seed-1 runs fly first so
# every hypothesis in the round gets one run before any second seed (interleave).
run_fq() {
	local width="$1" seed="$2" saved="$SEED"
	SEED="$seed"   # run_arm_common reads the global; restore after
	run_arm_common "SP2_FQ${width}_framedq_b${width}n240_${AIRFRAME}_${DIST}_s${seed}" \
		"\"arm\":\"SPEC2_FQ${width}\",\"conn_policy\":\"framed1\",\"bits\":${width},\"neurons\":240,\"input_window_k\":4,\"frame_stride\":10,\"quota\":\"128/64/32/16\",\"target_levels\":0,\"seed\":${seed}" \
		--grid-bits "$width" --max-output-neurons 240 --conn-policy framed1 \
		--output-full-window --input-window-k 4 --frame-stride 10
	SEED="$saved"
}

log "===== ARM FQ18 s${SEED} (framed-quota b18 — full coverage, 1 threshold/feature, 40 ms) ====="
run_fq 18 "$SEED"
log "===== ARM FQ36 s${SEED} (framed-quota b36 — full coverage, 2 thresholds/feature, 40 ms) ====="
run_fq 36 "$SEED"
log "===== ARM FQ30 s${SEED} (framed-quota b30 — arm A's width, 12x2+6x1 coverage, 40 ms) ====="
run_fq 30 "$SEED"

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

# --- FQ family, SECOND SEED (interleave satisfied: every arm has one run) -----
SEED2="${SP2_SEED2:-31337003}"
log "===== ARM FQ18 s${SEED2} (framed-quota b18, second seed) ====="
run_fq 18 "$SEED2"
log "===== ARM FQ36 s${SEED2} (framed-quota b36, second seed) ====="
run_fq 36 "$SEED2"
log "===== ARM FQ30 s${SEED2} (framed-quota b30, second seed) ====="
run_fq 30 "$SEED2"

log "########## SPECIALIST ROUND 2 COMPLETE — $(ls "$MARKDIR" | wc -l) markers ##########"
