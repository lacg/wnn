#!/usr/bin/env bash
# THE 1-LAYER SWEEP LADDER (16/08/2026 ~21:40 EDT, Luiz's consolidation —
# supersedes specialist rounds 2 and 3 and the sequencer; "the sweep is the
# most important one, it will tell us which experiments matter").
#
# Four stages, strictly ordered, each feeding the next. ONE pipeline for the
# sweeps (P1 = grid -> GA-NEURONS -> GA-MEMORY, the round-1 protocol, so every
# point is comparable to the banked arms); the PIPELINE question itself is
# stage D, asked once, at the discovered optimum.
#
#   A  WSB   bits 10..36 step 2 in ONE grid run, 32n fixed (fast), 2 seeds.
#            The grid trains every width; the whole population carries into
#            the GA stages and competes; stage-select headlines the winner.
#            Per-width val triples (stage-select lists) are the sweep curve.
#   B  WSN   neurons 16..512 (x2 steps) at the width winner, 2 seeds.
#   C  WK    the window sweep AT FIXED CAPACITY: total stays WN for every arm
#            (levels/motor constant => PWM resolution never moves). k=1 is the
#            control (min1); k=2/3/4 are framed1 with the scheduler's actual
#            largest-remainder splits (recorded in the marker, e.g. 128n ->
#            88/40, 72/40/16, 72/32/16/8), 10 ms between windows. GATED: k+1
#            flies only if k improved the mean headline steady (2 seeds).
#   D  A/B   (grid -> GA-NEURONS -> GA-MEMORY) vs (grid -> GA-CONNECTIVITY
#            (feature-scope) -> GA-MEMORY) at the full winner (WB, WN, WK).
#            The P1 side already exists — it IS the stage-C winner pair — so
#            only the P2 runs fly; the contrast is within-shape, within-seed.
#
# Gates compare mean headline-holdout steady across both seeds; n=2 makes every
# gate a screen, not a verdict — a human reads the ladder before publishing.
# All runs: stage-1 plant (translation ON, lambda_alt=16), 180k cell budget,
# mpcof teacher, seed pair 31337002/31337003. Markers are idempotent: re-run
# this chain and it resumes wherever it stopped.
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
. "$ROOT/scripts/controller_arm_lib.sh"

LOG="/private/tmp/sweep_ladder.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
OUTDIR="logs/controller/sweep_ladder"
MARKDIR="experiments/sweepladder_markers"
AIRFRAME="cf21_brushless"
DIST="L4C"
SEED="${SL_SEED:-31337002}"
SEED2="${SL_SEED2:-31337003}"
REPORT_SEEDS="99990101 99990102 99990103 99990104 99990105"
QUANTUM=8   # output quantum under BINARY antagonist decode (2*num_motors)

FEAT_STAGE1="--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i \
	--obs-collective-cmd --obs-alt-err --obs-vz"
S16_WEIGHTS="--fit-weight-err-sq 0.25 --fit-weight-steady 0.35 --fit-weight-stable 0.20 --fit-weight-jerk 0.15 --fit-weight-mono 0.05"

log() { echo "[ladder] $(date -u +%FT%TZ) $*" >> "$LOG"; }

mkdir -p "$OUTDIR" "$MARKDIR"
log "########## ARMED — 1-layer sweep ladder: WSB -> WSN -> windows -> pipeline A/B ##########"

# run_arm <seed> <tag> <extra-json> <phased_ga args...>   (P1 unless overridden)
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

pick_winner() {
	"$VP" - "$@" <<'PY'
import glob, json, re, sys
pattern = sys.argv[-1]
best = None
for pat in sys.argv[1:-1]:
	for p in glob.glob(pat):
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

mean_steady() {
	"$VP" - "$1" <<'PY'
import glob, json, re, sys, statistics
vals = []
for p in glob.glob(sys.argv[1]):
	try:
		d = json.load(open(p))
	except Exception:
		continue
	m = re.search(r"steady=([0-9.]+)", d.get("headline_holdout", ""))
	if m:
		vals.append(float(m.group(1)))
print(f"{statistics.mean(vals):.4f}" if vals else "")
PY
}

split_for_k() {  # actual scheduler split of $WN neurons over $1 windows
	"$VP" -c "
import ram_controller as c
from collections import Counter
cnt = Counter(c.arch_framed1_slot_schedule(int('$WN'), int('$1'), int('$QUANTUM'), 42, 0, 0, 0))
print('/'.join(str(cnt[i]) for i in range(int('$1') - 1, -1, -1)))
" 2>/dev/null
}

# ===== STAGE A: WSB — the width sweep ========================================
WSB_BITS="10 12 14 16 18 20 22 24 26 28 30 32 34 36"
for s in "$SEED" "$SEED2"; do
	log "===== A: WSB s${s} (b10..36 step 2 @32n, 1 window — where is the 1-layer width sweet spot?) ====="
	run_arm "$s" "SL_WSB_bsweep32n_${AIRFRAME}_${DIST}_s${s}" \
		"\"arm\":\"SL_WSB\",\"stage\":\"A\",\"conn_policy\":\"spread\",\"bits\":\"10..36x2\",\"neurons\":32,\"seed\":${s}" \
		--skip-stages bits,connections \
		--grid-bits $WSB_BITS --grid-output-neurons 32 --max-output-neurons 32
done
WB=$(pick_winner "$MARKDIR/SL_WSB_*.json" 'ob=([0-9]+)')
if [ -z "$WB" ]; then
	log "ABORT: WSB winner parse failed — no width to carry forward. Fix the parse and re-run."
	exit 1
fi
log "########## A DONE — width winner b=${WB} ##########"

# ===== STAGE B: WSN — the neuron sweep at the width winner ===================
for s in "$SEED" "$SEED2"; do
	log "===== B: WSN s${s} (16n..512n @b${WB}, 1 window — how much capacity/PWM resolution does the width want?) ====="
	run_arm "$s" "SL_WSN_nsweep_b${WB}_${AIRFRAME}_${DIST}_s${s}" \
		"\"arm\":\"SL_WSN\",\"stage\":\"B\",\"conn_policy\":\"spread\",\"bits\":${WB},\"neurons\":\"16..512\",\"seed\":${s}" \
		--skip-stages bits,connections \
		--grid-bits "$WB" --grid-output-neurons 16 32 64 128 256 512 --max-output-neurons 512
done
WN=$(pick_winner "$MARKDIR/SL_WSN_*.json" 'on=([0-9]+)')
if [ -z "$WN" ]; then
	log "ABORT: WSN winner parse failed — no neuron count to carry forward."
	exit 1
fi
log "########## B DONE — neuron winner ${WN}n (at b=${WB}) ##########"

# ===== STAGE C: the window sweep at FIXED capacity ===========================
# k=1 control: min1 (full feature coverage at 1 window; framed1 degenerates to
# exactly this, so it is the honest k=1 member of the same family).
for s in "$SEED" "$SEED2"; do
	log "===== C: K1 s${s} (${WN}n b${WB} min1, ONE window — the fixed-capacity control) ====="
	run_arm "$s" "SL_K1_b${WB}n${WN}_${AIRFRAME}_${DIST}_s${s}" \
		"\"arm\":\"SL_K1\",\"stage\":\"C\",\"conn_policy\":\"min1\",\"bits\":${WB},\"neurons\":${WN},\"input_window_k\":1,\"seed\":${s}" \
		--skip-stages bits,connections --conn-policy min1 \
		--grid-bits "$WB" --grid-output-neurons "$WN" --max-output-neurons "$WN"
done
BEST=$(mean_steady "$MARKDIR/SL_K1_*.json")
if [ -z "$BEST" ]; then
	log "ABORT: K1 produced no readable markers — nothing to gate against."
	exit 1
fi
WK=1
log "===== C: K1 mean headline steady ${BEST}° ====="

for K in 2 3 4; do
	SPLIT=$(split_for_k "$K")
	if [ -z "$SPLIT" ]; then
		log "STOP: scheduler could not split ${WN}n over k=${K} — window ladder ends."
		break
	fi
	for s in "$SEED" "$SEED2"; do
		log "===== C: K${K} s${s} (SAME ${WN}n spread ${SPLIT} newest->oldest, 10 ms apart — does old state pay at zero capacity cost?) ====="
		run_arm "$s" "SL_K${K}_b${WB}n${WN}_${AIRFRAME}_${DIST}_s${s}" \
			"\"arm\":\"SL_K${K}\",\"stage\":\"C\",\"conn_policy\":\"framed1\",\"bits\":${WB},\"neurons\":${WN},\"input_window_k\":${K},\"frame_stride\":10,\"quota\":\"${SPLIT}\",\"seed\":${s}" \
			--skip-stages bits,connections \
			--conn-policy framed1 --output-full-window --input-window-k "$K" --frame-stride 10 \
			--grid-bits "$WB" --grid-output-neurons "$WN" --max-output-neurons "$WN"
	done
	WS=$(mean_steady "$MARKDIR/SL_K${K}_*.json")
	if [ -z "$WS" ]; then
		log "STOP: K${K} produced no readable markers — fix before going deeper."
		break
	fi
	log "===== C: K${K} GATE — ${WS}° vs best-so-far K${WK} ${BEST}° ====="
	if [ "$("$VP" -c "print(1 if float('$WS') < float('$BEST') else 0)")" = "1" ]; then
		log "K${K} IMPROVED — proceeding to k=$((K + 1))"
		BEST="$WS"; WK="$K"
	else
		log "K${K} did NOT improve at fixed capacity — window ladder STOPS (deeper windows starve the recent frame)."
		break
	fi
done
log "########## C DONE — window winner k=${WK} @ ${BEST}° ##########"

# ===== STAGE D: pipeline A/B at the full winner (WB, WN, WK) =================
# P1 at this shape already exists: it IS the stage-C winner pair (SL_K${WK}).
# Only P2 flies; the contrast is within-shape, within-seed.
if [ "$WK" -gt 1 ]; then
	SHAPE_ARGS=(--conn-policy framed1 --output-full-window --input-window-k "$WK" --frame-stride 10)
	SHAPE_JSON="\"conn_policy\":\"framed1\",\"input_window_k\":${WK},\"frame_stride\":10"
else
	SHAPE_ARGS=(--conn-policy min1)
	SHAPE_JSON="\"conn_policy\":\"min1\",\"input_window_k\":1"
fi
for s in "$SEED" "$SEED2"; do
	log "===== D: P2 s${s} (grid->GA-CONNECTIVITY(feature)->GA-memory at b${WB} ${WN}n k${WK} — does wiring-only tuning beat the standard pipeline?) ====="
	run_arm "$s" "SL_P2_b${WB}n${WN}k${WK}_${AIRFRAME}_${DIST}_s${s}" \
		"\"arm\":\"SL_P2\",\"stage\":\"D\",\"pipeline\":\"connectivity-feature\",${SHAPE_JSON},\"bits\":${WB},\"neurons\":${WN},\"seed\":${s}" \
		--skip-stages neurons,bits --conn-mutation-scope feature \
		"${SHAPE_ARGS[@]}" \
		--grid-bits "$WB" --grid-output-neurons "$WN" --max-output-neurons "$WN"
done
P2S=$(mean_steady "$MARKDIR/SL_P2_*.json")
log "########## D DONE — pipeline A/B at (b${WB}, ${WN}n, k${WK}): P1 ${BEST}° vs P2 ${P2S:-unreadable}° ##########"
log "########## LADDER COMPLETE — $(ls "$MARKDIR" | wc -l) markers · winners: b=${WB} n=${WN} k=${WK} ##########"
