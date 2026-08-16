#!/usr/bin/env bash
# SPECIALIST PROGRAMME ROUND 2 (16/08/2026, Luiz) — the 1-LAYER LADDER, then the
# independent arms. 13 runs, same stage-1 plant as round 1 (translation ON,
# lambda_alt=16), same 180k cell budget — so B and A stay valid comparators.
# Runs on the 16/08 wheel (alt_err row 14 + --target-levels + Rust conn-policy
# samplers); legacy semantics are bit-identical on that wheel (parity anchors
# green), so cross-round comparison is honest.
#
# THE LADDER (Luiz 16/08 evening): the DFA-era b30 was TOTAL address width
# (prefix included), so the 1-layer sweet spot was never measured. Sweep it
# FIRST, then neurons at the width winner, then the framed family at both
# winners:
#
#   WSB  bits sweep, 2 seeds — ONE grid run sweeps b=10..36 step 2 (14 points,
#        sn=0, 128n cap, spread, 1 window). The full population (all widths)
#        carries into NEURONS/MEMORY and competes; stage-select headlines the
#        honest winner, and the grid's per-width val triples are the sweep data.
#   WSN  neuron sweep at the WSB winner width, 2 seeds — --grid-output-neurons
#        16 32 64 128 256 512 (levels/motor 4..128, all multiples of the BINARY
#        quantum 8). output_neurons IS the PWM decode resolution.
#   FQ   framed windows at BOTH winners, 2 seeds — k=4 stride 10 (frames at
#        t-0/-10/-20/-30 ms), EXACT window quotas newest->oldest 8:4:2:1 per
#        motor block (arch_ops::framed1_slot_schedule), every neuron frame-pure
#        + min1 over its frame. D40 (steady 36.70 vs D 10.17) showed mixing
#        stale frames into ONE address actively misleads; FQ asks whether
#        frame-PURE neurons make 40 ms pay, at the measured 1-layer optimum.
#        Winner bits/neurons are parsed from the WSB/WSN markers (ob=/on= in
#        the FPGA field, lower headline steady wins); fallback b30/240n logged
#        LOUDLY if parsing ever fails.
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
#   - width: WSB is the direct 1-layer width sweep (grid val triples per width
#     x 2 seeds + 2 full-pipeline headlines); G - B stays as the 256n check
#   - capacity/resolution: WSN sweeps 16n..512n at the width winner
#   - temporal coverage (FQ, 2 seeds at the measured optimum): FQ vs D40 (same
#     40 ms window, opposite budget policy — D40 steady 36.70) is the
#     frame-purity test; FQ vs arm A (1-frame b30, steady 1.34) says whether
#     time is worth ANY of the budget.
#   - follow-up protocol (Luiz): if the top-3 widths are statistically close in
#     the WSB stage-select lists, fly them as individual arms on more seeds
#     before trusting the winner.
#   - coverage: K18 - K18s = feature coverage at fixed width/neurons; K18 - K15 =
#     the coverage dose (18/18 vs 15/18 features). Spread's expected coverage:
#     b=15 -> 10.7/18, b=18 -> 12.0/18, b=30 -> 15.4/18 (so 33% of
#     (neuron,feature) pairs see NOTHING at b=18 under the legacy draw).
# n=1 => measurement, not verdict.
#
# ORDER: the ladder is inherently sequential (WSN needs WSB's winner, FQ needs
# both), so it runs first — WSB x2, WSN x2, FQ x2 — then the independent arms
# (H, R32, Q32, G, K18, K18s, K15) each get their single run. The interleave
# rule applies WITHIN the ladder via its 2-seed pairs.
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

SEED2="${SP2_SEED2:-31337003}"

# Run one arm at an explicit seed (run_arm_common reads the SEED global).
run_at_seed() {
	local seed="$1" saved="$SEED"; shift
	SEED="$seed"
	run_arm_common "$@"
	SEED="$saved"
}

# Winner field from the LOWER-headline-steady of two markers ("ob=" / "on=" in
# the FPGA field). Echoes empty on any failure — callers must fall back LOUDLY.
pick_winner() {
	"$VP" - "$1" "$2" "$3" <<'PY'
import json, re, sys
best = None
for p in sys.argv[1:3]:
	try:
		d = json.load(open(p))
	except Exception:
		continue
	m = re.search(r"steady=([0-9.]+)", d.get("headline_holdout", ""))
	s = float(m.group(1)) if m else 1e9
	if best is None or s < best[0]:
		best = (s, d)
m = re.search(sys.argv[3], best[1].get("fpga", "")) if best else None
print(m.group(1) if m else "")
PY
}

# --- LADDER 1/3: WSB — the 1-layer width sweep (2 seeds) ----------------------
# One grid run sweeps every width; all widths carry into NEURONS/MEMORY and
# compete. Arm A's protocol otherwise (sn=0, 128n cap, spread, 1 window).
WSB_BITS="10 12 14 16 18 20 22 24 26 28 30 32 34 36"
for s in "$SEED" "$SEED2"; do
	log "===== ARM WSB s${s} (1-layer width sweep b10..36 step 2 — where is the 1-layer sweet spot?) ====="
	run_at_seed "$s" "SP2_WSB_bsweep_${AIRFRAME}_${DIST}_s${s}" \
		"\"arm\":\"SPEC2_WSB\",\"conn_policy\":\"spread\",\"bits\":\"10..36x2\",\"neurons\":128,\"target_levels\":0,\"seed\":${s}" \
		--grid-bits $WSB_BITS --max-output-neurons 128
done

WB=$(pick_winner "$MARKDIR/SP2_WSB_bsweep_${AIRFRAME}_${DIST}_s${SEED}.json" \
                 "$MARKDIR/SP2_WSB_bsweep_${AIRFRAME}_${DIST}_s${SEED2}.json" 'ob=([0-9]+)')
if [ -z "$WB" ]; then
	WB=30
	log "!!!!! WSB winner parse FAILED — falling back to b=${WB} (arm A's width). FIX THE PARSE."
else
	log "===== WSB WINNER: b=${WB} ====="
fi

# --- LADDER 2/3: WSN — the neuron sweep at the width winner (2 seeds) ---------
for s in "$SEED" "$SEED2"; do
	log "===== ARM WSN s${s} (neuron sweep 16n..512n at b${WB} — capacity/PWM resolution at the width winner) ====="
	run_at_seed "$s" "SP2_WSN_nsweep_b${WB}_${AIRFRAME}_${DIST}_s${s}" \
		"\"arm\":\"SPEC2_WSN\",\"conn_policy\":\"spread\",\"bits\":${WB},\"neurons\":\"16..512\",\"target_levels\":0,\"seed\":${s}" \
		--grid-bits "$WB" --grid-output-neurons 16 32 64 128 256 512 \
		--max-output-neurons 512
done

WN=$(pick_winner "$MARKDIR/SP2_WSN_nsweep_b${WB}_${AIRFRAME}_${DIST}_s${SEED}.json" \
                 "$MARKDIR/SP2_WSN_nsweep_b${WB}_${AIRFRAME}_${DIST}_s${SEED2}.json" 'on=([0-9]+)')
if [ -z "$WN" ]; then
	WN=240
	log "!!!!! WSN winner parse FAILED — falling back to ${WN}n. FIX THE PARSE."
else
	log "===== WSN WINNER: ${WN}n ====="
fi

# --- LADDER 3/3: FQ — framed windows at BOTH winners (2 seeds) ----------------
# Exact 8:4:2:1 window quotas newest->oldest per motor block; every neuron
# frame-pure + min1 over its frame; the population covers time.
for s in "$SEED" "$SEED2"; do
	log "===== ARM FQ s${s} (framed-quota b${WB} ${WN}n, 40 ms — does frame-purity make time pay at the 1-layer optimum?) ====="
	run_at_seed "$s" "SP2_FQ_framedq_b${WB}n${WN}_${AIRFRAME}_${DIST}_s${s}" \
		"\"arm\":\"SPEC2_FQ\",\"conn_policy\":\"framed1\",\"bits\":${WB},\"neurons\":${WN},\"input_window_k\":4,\"frame_stride\":10,\"quota\":\"8:4:2:1\",\"target_levels\":0,\"seed\":${s}" \
		--grid-bits "$WB" --grid-output-neurons "$WN" --max-output-neurons "$WN" \
		--conn-policy framed1 --output-full-window --input-window-k 4 --frame-stride 10
done

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
