#!/usr/bin/env bash
# SPECIALIST ROUND 3 — THE WINDOW SWEEP (16/08/2026 evening, Luiz).
#
# "Sweep the windows: 1, then 2 windows using the number of neurons + half of it
#  on the second window (10 ms apart); if it improves try 3, then 4."
#
# THE RATIO IS ALREADY THE MACHINERY: framed1 draws frames with recency weights
# 2^slot, which at k=2 is exactly 2:1 — so a population of 1.5N apportions to N
# on the newest frame and N/2 on the older one. Verified exact (Rust
# framed1_slot_schedule, per motor block) for every WSN grid point:
#   k=2 -> 1.5N   = N + N/2                exact at N = 16..512
#   k=3 -> 1.75N  = N + N/2 + N/4          exact at N >= 32
#   k=4 -> 1.875N = N + N/2 + N/4 + N/8    exact at N >= 64
# Below those the BINARY output quantum (2*num_motors = 8) cannot express the
# smallest window; the chain REFUSES rather than silently mis-splitting.
#
# THE TRIAD PER k (this is the part Luiz's sketch needs to be honest):
#   W{k}  framed1, k windows, stride 10 (10 ms apart), TOTAL = the multiple above
#   C{k}  the MATCHED-CAPACITY CONTROL: the same total neurons, ONE window
#   (W1 = the WSN winner run itself — already banked, free)
# W{k} vs W1 measures the PACKAGE (more neurons AND they look at the past).
# W{k} vs C{k} isolates what TIME bought, because output_neurons IS the PWM
# decode resolution — 1.5N is finer-grained regardless of where it looks. Without
# C{k} a win is unattributable.
#
# PIPELINE: grid -> GA-CONNECTIVITY(scope=feature) -> GA-MEMORY for every arm.
# Why feature scope and not free: framed1 init makes each neuron frame-pure, and
# a free-scope rewire would scatter it back across frames mid-search — the arm
# would stop testing what it was built to test. A bit's thermometer run lives
# inside one frame, so feature scope preserves purity by construction, and the
# k=1 control gets the IDENTICAL optimizer pressure (at k=1 the window scope
# would degenerate to free, which is why it is not used here).
#
# GATE: k+1 is attempted only if W{k} beat C{k} on MEAN headline steady across
# both seeds. The decision and both numbers are logged. n=2 => this gate is a
# screen, not a verdict; a human reads the triad before anything is published.
#
# NOT AUTO-ARMED. It depends on round-2 winners, and the gate deserves a human
# look at n=2. Arm with:
#   nohup bash scripts/specialist_round3_windows.sh >/dev/null 2>&1 &
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
QUANTUM=8   # 2*num_motors under BINARY (antagonist decode needs even levels)

FEAT_STAGE1="--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i \
	--obs-collective-cmd --obs-alt-err --obs-vz"
S16_WEIGHTS="--fit-weight-err-sq 0.25 --fit-weight-steady 0.35 --fit-weight-stable 0.20 --fit-weight-jerk 0.15 --fit-weight-mono 0.05"

log() { echo "[spec3] $(date -u +%FT%TZ) $*" >> "$LOG"; }

mkdir -p "$OUTDIR" "$MARKDIR"
log "########## ARMED — round 3 window sweep (k=2, gated on to 3 and 4) ##########"

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

# ---- round-2 winners ---------------------------------------------------------
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
	log "ABORT: could not read round-2 winners (bits='$WB' neurons='$WN'). Round 2 must "
	log "       have landed its markers, or pass SP3_BITS=.. SP3_NEURONS=.. explicitly."
	exit 1
fi
log "round-2 winners: b=${WB} n=${WN}"

# total_for_k <k>  ->  neurons rounded to the output quantum, but ONLY if the
# scheduler actually produces the requested N, N/2, N/4... split. A threshold
# test is not enough: at n=16 k=3 the quantum rounds the split to 16/8/8 while
# the arm claims 16/8/4, which would publish a mis-described experiment. Ask the
# real scheduler and compare.
total_for_k() {
	"$VP" -c "
import ram_controller as c
from collections import Counter
k, n, q = int('$1'), int('$WN'), int('$QUANTUM')
tot = int(round(round(n * sum(2.0 ** -i for i in range(k))) / q)) * q
if tot <= 0:
	raise SystemExit
got = Counter(c.arch_framed1_slot_schedule(tot, k, q, 42, 0, 0, 0))
got = [got[i] for i in range(k - 1, -1, -1)]     # newest -> oldest
want = [n >> i for i in range(k)]
print(tot if got == want else '')
" 2>/dev/null
}

# actual per-window split the Rust scheduler will produce, for the log
split_for() {
	"$VP" -c "
import ram_controller as c
from collections import Counter
s = c.arch_framed1_slot_schedule(int('$1'), int('$2'), int('$QUANTUM'), 42, 0, 0, 0)
cnt = Counter(s)
print('/'.join(str(cnt[i]) for i in range(int('$2') - 1, -1, -1)))
" 2>/dev/null
}

# mean headline steady over a tag glob (empty if nothing parseable)
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

# ---- the k ladder ------------------------------------------------------------
for K in 2 3 4; do
	TOT=$(total_for_k "$K")
	if [ -z "$TOT" ]; then
		log "STOP: at n=${WN}, k=${K} cannot be apportioned as $(printf '%s' "$WN")/2/4... under the output quantum "
		log "       (${QUANTUM}) — the scheduler's split would not match the arm's claim. Ladder ends here."
		break
	fi
	SPLIT=$(split_for "$TOT" "$K")
	log "===== k=${K}: total ${TOT}n, windows (newest->oldest) ${SPLIT}, ${WB}b, 10 ms apart ====="

	for s in "$SEED" "$SEED2"; do
		log "===== W${K} s${s} (${TOT}n framed over ${K} windows ${SPLIT} — does looking ${K} steps into the past pay?) ====="
		run_arm "$s" "SP3_W${K}_framed_b${WB}n${TOT}_${AIRFRAME}_${DIST}_s${s}" \
			"\"arm\":\"SPEC3_W${K}\",\"conn_policy\":\"framed1\",\"bits\":${WB},\"neurons\":${TOT},\"input_window_k\":${K},\"frame_stride\":10,\"quota\":\"${SPLIT}\",\"seed\":${s}" \
			--conn-policy framed1 --output-full-window --input-window-k "$K" --frame-stride 10 \
			--grid-bits "$WB" --grid-output-neurons "$TOT" --max-output-neurons "$TOT"

		log "===== C${K} s${s} (${TOT}n on ONE window — matched-capacity control: is it the neurons or the time?) ====="
		run_arm "$s" "SP3_C${K}_ctl1win_b${WB}n${TOT}_${AIRFRAME}_${DIST}_s${s}" \
			"\"arm\":\"SPEC3_C${K}\",\"conn_policy\":\"min1\",\"bits\":${WB},\"neurons\":${TOT},\"input_window_k\":1,\"seed\":${s}" \
			--conn-policy min1 \
			--grid-bits "$WB" --grid-output-neurons "$TOT" --max-output-neurons "$TOT"
	done

	WS=$(mean_steady "$MARKDIR/SP3_W${K}_*.json")
	CS=$(mean_steady "$MARKDIR/SP3_C${K}_*.json")
	if [ -z "$WS" ] || [ -z "$CS" ]; then
		log "STOP: k=${K} produced no comparable markers (W='$WS' C='$CS') — fix before going deeper."
		break
	fi
	log "===== k=${K} GATE: W${K} mean steady ${WS}° vs matched control C${K} ${CS}° ====="
	if [ "$("$VP" -c "print(1 if float('$WS') < float('$CS') else 0)")" = "1" ]; then
		log "k=${K} IMPROVED over matched capacity — proceeding to k=$((K + 1))"
	else
		log "k=${K} did NOT beat its matched-capacity control — STOPPING the ladder here (time is not paying; the extra neurons are)."
		break
	fi
done

log "########## ROUND 3 COMPLETE — $(ls "$MARKDIR" | wc -l) markers ##########"
