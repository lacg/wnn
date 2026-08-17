#!/usr/bin/env bash
# SPECIALIST ROUND 3 — THE WINDOW SWEEP AT FIXED CAPACITY (16/08/2026 late
# evening, Luiz's second cut, replacing the 1.5N + matched-control design).
#
# "Keep X as the total and spread it over the windows — for 128n at 2:1 that is
#  ~85/43 — so the thermometer levels are not impacted and we can compare 128n
#  before and after."
#
# WHY THIS IS THE SHARPER DESIGN: output_neurons IS the PWM decode resolution
# (levels_per_motor = output_neurons / num_motors). Holding the TOTAL at N=128
# keeps 32 levels/motor for every arm, so "more windows" is the ONLY moving
# axis — no matched-capacity control arm needed, ONE 1-window control (C1)
# serves the whole ladder, and every arm is directly comparable to the banked
# 128n rows.
#
# THE SPLITS (what the Rust scheduler actually produces at N=128, quantum 8 —
# largest-remainder per 16-neuron block, so per-motor balance is exact):
#   k=2  88/40        (ideal 2:1     = 85.3/42.7)
#   k=3  72/40/16     (ideal 4:2:1   = 73.1/36.6/18.3)
#   k=4  72/32/16/8   (ideal 8:4:2:1 = 68.3/34.1/17.1/8.5)
# The marker JSON records the ACTUAL split, not the ideal.
#
# ARMS (windows 10 ms apart, all through grid -> GA-CONNECTIVITY(feature) ->
# GA-MEMORY):
#   C1   128n, 1 window, min1        — the ladder's control. Flown (not
#        borrowed from arm A) because the pipeline differs from round 1's
#        grid->NEURONS->MEMORY; every W-C contrast must be within-pipeline.
#   W2   128n over 2 windows (88/40) — does giving a third of the population
#        10 ms-old state pay, at zero capacity cost?
#   W3   128n over 3 windows         — only if W2 beat C1 (mean steady, 2 seeds)
#   W4   128n over 4 windows         — only if W3 beat W2
#
# feature-scope connectivity for every arm: framed1 init makes neurons
# frame-pure and a free rewire would scatter that mid-search; a thermometer run
# lives inside one frame, so feature scope preserves purity by construction and
# gives C1 identical optimizer pressure.
#
# GATES at n=2 are a screen, not a verdict — a human reads the ladder before
# anything is published.
#
# Winners come from round-2 markers OR the SP3_BITS / SP3_NEURONS overrides
# (Luiz 16/08 21:2x: armed with SP3_BITS=30 SP3_NEURONS=128 — arm A's shape —
# so this round does NOT wait on round 2; the sequencer flies it FIRST).
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
. "$ROOT/scripts/controller_arm_lib.sh"

LOG="/private/tmp/specialist_round3.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
OUTDIR="logs/controller/specialist3"
MARKDIR="experiments/specialist3_markers"
R2MARKS="experiments/specialist2_markers"
AIRFRAME="cf21_brushless"
DIST="L4C"
SEED="${SP3_SEED:-31337002}"
SEED2="${SP3_SEED2:-31337003}"
REPORT_SEEDS="99990101 99990102 99990103 99990104 99990105"
QUANTUM=8   # output quantum under BINARY antagonist decode (2*num_motors)

FEAT_STAGE1="--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i \
	--obs-collective-cmd --obs-alt-err --obs-vz"
S16_WEIGHTS="--fit-weight-err-sq 0.25 --fit-weight-steady 0.35 --fit-weight-stable 0.20 --fit-weight-jerk 0.15 --fit-weight-mono 0.05"

log() { echo "[spec3] $(date -u +%FT%TZ) $*" >> "$LOG"; }

mkdir -p "$OUTDIR" "$MARKDIR"

run_arm() {
	local seed="$1" tag="$2" extra="$3"; shift 3
	mkdir -p "$OUTDIR/ckpt/$tag"
	# shellcheck disable=SC2086
	run_controller_arm "$tag" \
		"$MARKDIR" "$OUTDIR" "$VP" log \
		"$extra" \
		-- \
		--levels 16 --lamarckian \
		--skip-stages neurons,bits --conn-mutation-scope feature \
		--max-cells 180000 --max-cells-strict \
		--save-stage-checkpoints "$OUTDIR/ckpt/$tag" \
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

WB="${SP3_BITS:-$(pick_winner "$R2MARKS/SP2_WSB*.json" 'ob=([0-9]+)')}"
WN="${SP3_NEURONS:-$(pick_winner "$R2MARKS/SP2_WSN*.json" 'on=([0-9]+)')}"
if [ -z "$WB" ] || [ -z "$WN" ]; then
	log "ABORT: no width/neuron winners (bits='$WB' neurons='$WN'). Land round 2 or pass SP3_BITS/SP3_NEURONS."
	exit 1
fi
log "########## ARMED — round 3 window sweep at FIXED n=${WN}, b=${WB} (k=2, gated to 3, 4) ##########"

# The ACTUAL split the Rust scheduler will produce for n=WN at k windows.
split_for_k() {
	"$VP" -c "
import ram_controller as c
from collections import Counter
cnt = Counter(c.arch_framed1_slot_schedule(int('$WN'), int('$1'), int('$QUANTUM'), 42, 0, 0, 0))
print('/'.join(str(cnt[i]) for i in range(int('$1') - 1, -1, -1)))
" 2>/dev/null
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

# ---- C1: the 1-window control at the SAME capacity and pipeline --------------
for s in "$SEED" "$SEED2"; do
	log "===== C1 s${s} (${WN}n b${WB} min1, ONE window — the fixed-capacity control every W{k} is read against) ====="
	run_arm "$s" "SP3_C1_1win_b${WB}n${WN}_${AIRFRAME}_${DIST}_s${s}" \
		"\"arm\":\"SPEC3_C1\",\"conn_policy\":\"min1\",\"bits\":${WB},\"neurons\":${WN},\"input_window_k\":1,\"seed\":${s}" \
		--conn-policy min1 \
		--grid-bits "$WB" --grid-output-neurons "$WN" --max-output-neurons "$WN"
done
BEST=$(mean_steady "$MARKDIR/SP3_C1_*.json")
if [ -z "$BEST" ]; then
	log "ABORT: C1 produced no readable markers — nothing to gate against."
	exit 1
fi
BEST_NAME="C1"
log "===== C1 mean headline steady: ${BEST}° ====="

# ---- the k ladder at FIXED total -------------------------------------------
for K in 2 3 4; do
	SPLIT=$(split_for_k "$K")
	if [ -z "$SPLIT" ]; then
		log "STOP: scheduler could not split n=${WN} over k=${K} — ladder ends."
		break
	fi
	log "===== k=${K}: SAME ${WN}n spread ${SPLIT} (newest->oldest), ${WB}b, 10 ms apart ====="
	for s in "$SEED" "$SEED2"; do
		log "===== W${K} s${s} (${WN}n as ${SPLIT} over ${K} windows — does giving part of a FIXED population old state pay?) ====="
		run_arm "$s" "SP3_W${K}_framed_b${WB}n${WN}_${AIRFRAME}_${DIST}_s${s}" \
			"\"arm\":\"SPEC3_W${K}\",\"conn_policy\":\"framed1\",\"bits\":${WB},\"neurons\":${WN},\"input_window_k\":${K},\"frame_stride\":10,\"quota\":\"${SPLIT}\",\"seed\":${s}" \
			--conn-policy framed1 --output-full-window --input-window-k "$K" --frame-stride 10 \
			--grid-bits "$WB" --grid-output-neurons "$WN" --max-output-neurons "$WN"
	done

	WS=$(mean_steady "$MARKDIR/SP3_W${K}_*.json")
	if [ -z "$WS" ]; then
		log "STOP: W${K} produced no readable markers — fix before going deeper."
		break
	fi
	log "===== k=${K} GATE: W${K} mean steady ${WS}° vs best-so-far ${BEST_NAME} ${BEST}° ====="
	if [ "$("$VP" -c "print(1 if float('$WS') < float('$BEST') else 0)")" = "1" ]; then
		log "W${K} IMPROVED on ${BEST_NAME} — proceeding to k=$((K + 1))"
		BEST="$WS"; BEST_NAME="W${K}"
	else
		log "W${K} did NOT beat ${BEST_NAME} at fixed capacity — STOPPING the ladder (deeper windows are starving the recent frame)."
		break
	fi
done

log "########## ROUND 3 COMPLETE — $(ls "$MARKDIR" | wc -l) markers · best ${BEST_NAME} @ ${BEST}° ##########"
