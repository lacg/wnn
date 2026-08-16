#!/usr/bin/env bash
# SPECIALIST PROGRAMME ROUND 2 (16/08/2026 evening, Luiz's redesign) — the
# 1-LAYER SWEEP LADDER with the pipeline ablation. 8 runs, same stage-1 plant
# as round 1 (translation ON, lambda_alt=16), same 180k cell budget. Runs on
# the 16/08 wheel (alt_err row 14 + --target-levels + Rust samplers + scoped
# axonogenesis).
#
# WHY: DFA-era --grid-bits was the TOTAL address width (prefix included), so
# the tuned "b30" was really 14-22 input-facing bits — the 1-layer width sweet
# spot has never been measured. The 40 ms framed family is DEFERRED until the
# sweeps close ("forget the 40ms" — Luiz); it will fly at the measured optimum.
#
# THE LADDER — each sweep is flown through TWO pipelines (2 seeds each):
#
#   P1  grid -> GA-NEURONS      -> GA-MEMORY   (the round-1 protocol: ablation
#                                               baseline, comparable to A..E)
#   P2  grid -> GA-CONNECTIVITY -> GA-MEMORY   (Luiz's point: when the axes
#       under study are bits/neurons, the honest GA is CONNECTIVITY — GA-neurons
#       varies neuron count, GA-bits varies width; only GA-connectivity tunes
#       the wiring at FIXED shape. --conn-mutation-scope feature: rewires move
#       WHERE on the feature a bit sits, never which feature — at 1 window the
#       "window" scope degenerates to free, so feature-scope is the new physics.)
#
#   WSB  bits sweep — ONE grid run sweeps b=10..36 step 2 (14 points, sn=0,
#        32n FIXED so it is fast; 32n = 8 levels/motor, BINARY-quantum-legal).
#        All widths carry into the GA stage and compete; stage-select headlines
#        the honest winner. 2 pipelines x 2 seeds = 4 runs.
#   WSN  neuron sweep at the WSB winner width — --grid-output-neurons
#        16 32 64 128 256 512 (levels/motor 4..128). 2 x 2 = 4 runs.
#
# WINNER PARSE: lowest headline-holdout steady across a stage's 4 markers;
# bits from ob=, neurons from on= (both in the FPGA field since 16/08). A
# failed parse falls back LOUDLY (b30 / 240n) rather than dying silently.
#
# PRE-REGISTERED READ: headline triple + pos= per run; P2-P1 at equal seed =
# what connectivity-only optimization buys over the standard pipeline; the
# per-width/per-n grid val triples (stage-select candidate lists) are the sweep
# curves; 2 seeds => spread, not verdict. Follow-up (Luiz): top-3 widths
# statistically close -> individual arms on more seeds before trusting the
# winner.
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
SEED2="${SP2_SEED2:-31337003}"
REPORT_SEEDS="99990101 99990102 99990103 99990104 99990105"

FEAT_STAGE1="--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i \
	--obs-collective-cmd --obs-alt-err --obs-vz"
S16_WEIGHTS="--fit-weight-err-sq 0.25 --fit-weight-steady 0.35 --fit-weight-stable 0.20 --fit-weight-jerk 0.15 --fit-weight-mono 0.05"

log() { echo "[spec2] $(date -u +%FT%TZ) $*" >> "$LOG"; }

mkdir -p "$OUTDIR" "$MARKDIR"
log "########## ARMED — round 2 sweep ladder (bits then neurons, P1 vs P2), 8 runs ##########"

# run_arm <seed> <tag> <extra-json> <phased_ga args...>
run_arm() {
	local seed="$1" tag="$2" extra="$3"; shift 3
	mkdir -p "$OUTDIR/ckpt/$tag"
	# shellcheck disable=SC2086
	run_controller_arm "$tag" \
		"$MARKDIR" "$OUTDIR" "$VP" log \
		"$extra" \
		-- \
		--levels 16 --lamarckian \
		--max-cells 180000 --max-cells-strict \
		--save-stage-checkpoints "$OUTDIR/ckpt/$tag" \
		--neurons-gens 5 --neurons-patience 3 \
		--conns-gens 5 --conns-patience 3 \
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
		--base-seed "$seed" \
		"$@"
	log "$tag finished rc=$?"
}

# Winner field from the LOWEST-headline-steady of N markers. Last arg is the
# regex; echoes empty on any failure — callers must fall back LOUDLY.
pick_winner() {
	"$VP" - "$@" <<'PY'
import json, re, sys
*paths, pattern = sys.argv[1:]
best = None
for p in paths:
	try:
		d = json.load(open(p))
	except Exception:
		continue
	m = re.search(r"steady=([0-9.]+)", d.get("headline_holdout", ""))
	s = float(m.group(1)) if m else 1e9
	if best is None or s < best[0]:
		best = (s, d)
m = re.search(pattern, best[1].get("fpga", "")) if best else None
print(m.group(1) if m else "")
PY
}

# --- LADDER 1/2: WSB — the 1-layer width sweep at 32n fixed -------------------
WSB_BITS="10 12 14 16 18 20 22 24 26 28 30 32 34 36"
for s in "$SEED" "$SEED2"; do
	log "===== WSB-P1 s${s} (width sweep b10..36 @32n, grid->GA-neurons->GA-memory — where is the 1-layer width sweet spot?) ====="
	run_arm "$s" "SP2_WSBP1_bsweep32n_${AIRFRAME}_${DIST}_s${s}" \
		"\"arm\":\"SPEC2_WSB_P1\",\"pipeline\":\"neurons\",\"conn_policy\":\"spread\",\"bits\":\"10..36x2\",\"neurons\":32,\"seed\":${s}" \
		--skip-stages bits,connections \
		--grid-bits $WSB_BITS --grid-output-neurons 32 --max-output-neurons 32

	log "===== WSB-P2 s${s} (width sweep b10..36 @32n, grid->GA-CONNECTIVITY(feature)->GA-memory — does wiring-only tuning beat the standard pipeline?) ====="
	run_arm "$s" "SP2_WSBP2_bsweep32n_${AIRFRAME}_${DIST}_s${s}" \
		"\"arm\":\"SPEC2_WSB_P2\",\"pipeline\":\"connectivity-feature\",\"conn_policy\":\"spread\",\"bits\":\"10..36x2\",\"neurons\":32,\"seed\":${s}" \
		--skip-stages neurons,bits --conn-mutation-scope feature \
		--grid-bits $WSB_BITS --grid-output-neurons 32 --max-output-neurons 32
done

WB=$(pick_winner \
	"$MARKDIR/SP2_WSBP1_bsweep32n_${AIRFRAME}_${DIST}_s${SEED}.json" \
	"$MARKDIR/SP2_WSBP1_bsweep32n_${AIRFRAME}_${DIST}_s${SEED2}.json" \
	"$MARKDIR/SP2_WSBP2_bsweep32n_${AIRFRAME}_${DIST}_s${SEED}.json" \
	"$MARKDIR/SP2_WSBP2_bsweep32n_${AIRFRAME}_${DIST}_s${SEED2}.json" \
	'ob=([0-9]+)')
if [ -z "$WB" ]; then
	WB=30
	log "!!!!! WSB winner parse FAILED — falling back to b=${WB} (arm A's width). FIX THE PARSE."
else
	log "===== WSB WINNER: b=${WB} ====="
fi

# --- LADDER 2/2: WSN — the neuron sweep at the width winner -------------------
WSN_NEURONS="16 32 64 128 256 512"
for s in "$SEED" "$SEED2"; do
	log "===== WSN-P1 s${s} (neuron sweep 16n..512n @b${WB}, grid->GA-neurons->GA-memory — how much PWM resolution does the width winner want?) ====="
	run_arm "$s" "SP2_WSNP1_nsweep_b${WB}_${AIRFRAME}_${DIST}_s${s}" \
		"\"arm\":\"SPEC2_WSN_P1\",\"pipeline\":\"neurons\",\"conn_policy\":\"spread\",\"bits\":${WB},\"neurons\":\"16..512\",\"seed\":${s}" \
		--skip-stages bits,connections \
		--grid-bits "$WB" --grid-output-neurons $WSN_NEURONS --max-output-neurons 512

	log "===== WSN-P2 s${s} (neuron sweep 16n..512n @b${WB}, grid->GA-CONNECTIVITY(feature)->GA-memory — connectivity-only at each capacity) ====="
	run_arm "$s" "SP2_WSNP2_nsweep_b${WB}_${AIRFRAME}_${DIST}_s${s}" \
		"\"arm\":\"SPEC2_WSN_P2\",\"pipeline\":\"connectivity-feature\",\"conn_policy\":\"spread\",\"bits\":${WB},\"neurons\":\"16..512\",\"seed\":${s}" \
		--skip-stages neurons,bits --conn-mutation-scope feature \
		--grid-bits "$WB" --grid-output-neurons $WSN_NEURONS --max-output-neurons 512
done

WN=$(pick_winner \
	"$MARKDIR/SP2_WSNP1_nsweep_b${WB}_${AIRFRAME}_${DIST}_s${SEED}.json" \
	"$MARKDIR/SP2_WSNP1_nsweep_b${WB}_${AIRFRAME}_${DIST}_s${SEED2}.json" \
	"$MARKDIR/SP2_WSNP2_nsweep_b${WB}_${AIRFRAME}_${DIST}_s${SEED}.json" \
	"$MARKDIR/SP2_WSNP2_nsweep_b${WB}_${AIRFRAME}_${DIST}_s${SEED2}.json" \
	'on=([0-9]+)')
[ -n "$WN" ] && log "===== WSN WINNER: ${WN}n (at b=${WB}) — the framed/40ms family flies at these winners NEXT round =====" \
             || log "!!!!! WSN winner parse FAILED — fix before arming the framed family."

log "########## SPECIALIST ROUND 2 COMPLETE — $(ls "$MARKDIR" | wc -l) markers ##########"
