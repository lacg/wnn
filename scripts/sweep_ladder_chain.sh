#!/usr/bin/env bash
# THE 1-LAYER SWEEP LADDER — CONTROLLED EDITION (17/08/2026, Luiz's correction).
#
# v1 was a COMPETITIVE sweep: all 14 widths in one grid, one population, the GA
# culling widths against each other. Luiz: "the sweep freezes one thing and
# checks its results... 2 seeds using 10b and get their baseline, then 12b, then
# 14b... this way we have enough population from those bits and no interference
# from others." So each point now gets its OWN run and its OWN full population.
#
# FREEZING IS ENFORCED BY STAGE SELECTION, NOT BY CAPS: every sweep run is
#   grid (ONE point) -> GA-CONNECTIVITY -> GA-MEMORY
# because GA-NEURONS mutates neuron COUNTS (_mutate_neurons: set_state_neurons /
# set_output_neurons) — harmless in a bits sweep but it would silently un-freeze
# the neuron sweep. GA-CONNECTIVITY rewires at FIXED (n, b), GA-MEMORY only
# writes cells, so the swept variable cannot drift in either sweep and both are
# measured under identical optimizer pressure. GA-Neurons vs GA-Connectivity is
# then asked once, on its own, in stage D — where it is the question rather than
# a confound.
#
# INTERLEAVED (feedback_sweeps_always_interleave): round 1 = ONE seed of every
# point, round 2 = the second seed. A stall still leaves a complete low-res
# picture of the whole axis instead of a perfect answer for the first few points.
#
# CULL AFTER ROUND 1 (17/08/2026, Luiz — "throw the fishing net very broad, then
# go fine-grained on the promising values"). Round 1 IS the broad net: 14 widths
# at one seed each. Round 2 then spends its budget only on the survivors instead
# of re-flying widths already shown to be hopeless. Rule (pre-registered, applied
# by _cull below):
#   keep a width if it is in the top SL_CULL_K (default 6) by round-1 headline
#   steady, OR within SL_CULL_RATIO (default 1.25x) of the best width — the
#   ratio clause exists because round 1 is n=1 and a hard top-K would drop a
#   width that lost by less than seed noise.
# A width with no readable round-1 marker is dropped LOUDLY (it did not produce
# a number, so it cannot be defended). Culled widths keep their round-1 marker:
# the broad-net curve stays complete and publishable, they simply never get a
# second seed and cannot win the stage.
# The freed budget (~8 runs, ~28 h) can buy survivors a THIRD seed instead:
# set SL_SEEDS="31337002 31337003 31337004" — rounds 2+ only fly survivors.
#
#   A  BITS    b = 10,12,...,36 at 32n           14 points x 2 seeds = 28 runs
#   B  NEURONS n = 16..512 at the bits winner     6 points x 2 seeds = 12 runs
#   C  WINDOWS k = 1,2,3,4 at FIXED capacity      gated, 2 seeds     = 2..8 runs
#      (total neurons held at the stage-B winner so levels/motor never moves;
#       k=1 = min1 control, k>=2 = framed1 at the scheduler's actual splits,
#       10 ms apart; k+1 only if k improved mean headline steady)
#   D  PIPELINE (grid->GA-NEURONS->GA-MEMORY) vs (grid->GA-CONNECTIVITY->
#      GA-MEMORY) at the full winner (b*, n*, k*)                   = 4 runs
#
# Winners = LOWEST MEAN headline-holdout steady across that point's seeds (the
# mean, not the best seed — best-of-N inflates). Every point's per-seed numbers
# stay in its own marker, so the full curve is recoverable regardless of which
# point won.
#
# COST: 46-52 runs at ~3.2 h = ~150-170 h (~6-7 days). To trim, set
# SL_BITS="10 14 18 22 26 30 34" (Luiz's own 4-by-4 alternative): 7 points x 2
# seeds = 14 runs, ~46 h for stage A.
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
SEEDS="${SL_SEEDS:-31337002 31337003}"
BITS="${SL_BITS:-10 12 14 16 18 20 22 24 26 28 30 32 34 36}"
NEURONS="${SL_NEURONS:-16 32 64 128 256 512}"
SWEEP_N="${SL_SWEEP_NEURONS:-32}"   # the fixed neuron count during the BITS sweep
REPORT_SEEDS="99990101 99990102 99990103 99990104 99990105"
QUANTUM=8

FEAT_STAGE1="--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i \
	--obs-collective-cmd --obs-alt-err --obs-vz"
# ---------------------------------------------------------------------------
# FITNESS REGIME (refreshed 24/08/2026 — was S16 + legacy combine + no gate).
#
# PROVISIONAL: these are the leading arm of the GATED WEIGHT SWEEP, which is
# still flying (21/30 markers at the time of writing). The C10 pair is SEALED
# (C10noJM beat C10 3-1 on both primaries, pre-registered paired majority), but
# C10noJM vs S16noJM is NOT adjudicated — it is split 1-1-1 over three seeds.
# When the sweep ends, change ONE line: LADDER_WEIGHTS.
#
#   C10noJM  --fit-weight-err-sq 0.57   --fit-weight-stable 0.43
#   S16noJM  --fit-weight-err-sq 0.3125 --fit-weight-stable 0.25 --fit-weight-steady 0.4375
#
# Both are their parent's weights RENORMALIZED to sum to 1 after dropping jerk
# and mono — the terms the sweep showed actively hurt (and the noJM arms post
# LOWER jerk than their jerk-weighted parents anyway). Do NOT re-add them; if
# mono ever matters it belongs in the viability gate as a CONSTRAINT, not here.
# ---------------------------------------------------------------------------
LADDER_WEIGHTS="${LADDER_WEIGHTS:---fit-weight-err-sq 0.57 --fit-weight-stable 0.43}"

# Aggregation + viability gate: byte-identical to the regime that SELECTED the
# weights above. Both were absent before 24/08 — and an absent --fit-aggregation
# is NOT a no-op: its default is None = "harmonic in-search + arithmetic
# stage-select", i.e. the legacy combine. The ladder would have produced a
# complete, plausible bits curve measured under the combine we replaced, with
# nothing in the output saying so.
AGG="--fit-aggregation zscore --zrank-clamp 3.0"
GATE="--gate-stable 0.70 --gate-err 8.0"

# Altitude reward shaping. The sweep that chose LADDER_WEIGHTS flew
# --reward-lambda-alt 0; this file carried 16 (pre-24/08). Defaulting to 0 keeps
# the ladder coherent with the objective the weights were selected under.
# Override with SL_LAMBDA_ALT=16 if the translation task deliberately wants it.
SL_LAMBDA_ALT="${SL_LAMBDA_ALT:-0}"

log() { echo "[ladder] $(date -u +%FT%TZ) $*" >> "$LOG"; }

mkdir -p "$OUTDIR" "$MARKDIR"
log "########## ARMED — CONTROLLED sweep ladder (one point per run, 2 seeds, interleaved) ##########"
log "bits=[$BITS] neurons=[$NEURONS] seeds=[$SEEDS]"

# run_arm <seed> <tag> <extra-json> <extra phased_ga args...>
# Sweep default: grid(1 point) -> GA-CONNECTIVITY -> GA-MEMORY. Callers may
# append --skip-stages to override (argparse last-wins) — stage D does exactly that.
run_arm() {
	local seed="$1" tag="$2" extra="$3"; shift 3
	mkdir -p "$OUTDIR/ckpt/$tag"
	# shellcheck disable=SC2086
	run_controller_arm "$tag" \
		"$MARKDIR" "$OUTDIR" "$VP" log \
		"$extra" \
		-- \
		--levels 16 --lamarckian \
		--skip-stages neurons,bits \
		--max-cells 180000 --max-cells-strict \
		--save-stage-checkpoints "$OUTDIR/ckpt/$tag" \
		--neurons-gens 5 --neurons-patience 3 \
		--conns-gens 5 --conns-patience 3 \
		--memory-gens 120 --memory-patience 2 \
		--pop 50 --num-eval-folds 5 --check-interval 2 --magnitude-aware-patience \
		--eval-episodes 100 --memory-eval-episodes 200 \
		--steps 2000 --tilt 5.0 \
		$LADDER_WEIGHTS \
		$AGG $GATE \
		--report-episodes 100 --holdout-pop-sample 8 \
		--runs 1 --memory-mode BINARY \
		--airframe "$AIRFRAME" --disturbance "$DIST" --teacher mpcof \
		$FEAT_STAGE1 \
		--translation --reward-lambda-alt "$SL_LAMBDA_ALT" \
		--grid-state-neurons 0 --max-state-neurons 0 \
		--report-seeds $REPORT_SEEDS \
		--base-seed "$seed" \
		"$@"
	log "$tag finished rc=$?"
}

# mean headline steady over a marker glob (empty when nothing is parseable)
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

# _cull <marker-glob-prefix> <marker-glob-suffix> <points...>
# Echoes the surviving points (space separated), best first. Keeps the top
# SL_CULL_K by headline steady plus anything within SL_CULL_RATIO of the best;
# drops points with no readable marker. Diagnostics go to stderr (the caller
# logs them) so stdout stays parseable.
_cull() {
	local pre="$1" suf="$2"; shift 2
	SL_CULL_K="${SL_CULL_K:-6}" SL_CULL_RATIO="${SL_CULL_RATIO:-1.25}" \
	"$VP" - "$pre" "$suf" "$@" <<'PY'
import glob, json, os, re, sys
pre, suf, points = sys.argv[1], sys.argv[2], sys.argv[3:]
K = int(os.environ.get("SL_CULL_K", "6"))
RATIO = float(os.environ.get("SL_CULL_RATIO", "1.25"))
scored, missing = [], []
for p in points:
	vals = []
	for f in glob.glob(f"{pre}{p}{suf}"):
		try:
			d = json.load(open(f))
		except Exception:
			continue
		m = re.search(r"steady=([0-9.]+)", d.get("headline_holdout", ""))
		if m:
			vals.append(float(m.group(1)))
	if vals:
		scored.append((min(vals), p))     # round 1 is n=1; min == that seed
	else:
		missing.append(p)
if not scored:
	print("", end="")
	sys.exit(0)
scored.sort()
best = scored[0][0]
keep = [p for i, (v, p) in enumerate(scored) if i < K or v <= best * RATIO]
for v, p in scored:
	mark = "KEEP" if p in keep else "cull"
	print(f"  [cull] {mark} {p}: steady {v:.2f}° ({v / best:.2f}x best)", file=sys.stderr)
for p in missing:
	print(f"  [cull] DROP {p}: no readable round-1 marker", file=sys.stderr)
print(" ".join(keep))
PY
}

# argmin over "<label> <value>" lines on stdin; echoes the winning label
argmin_label() {
	"$VP" -c "
import sys
best = None
for line in sys.stdin:
	parts = line.split()
	if len(parts) == 2:
		try:
			v = float(parts[1])
		except ValueError:
			continue
		if best is None or v < best[1]:
			best = (parts[0], v)
print(best[0] if best else '')
"
}

# ===== STAGE A: the BITS sweep, one width per run ============================
run_bits_round() {   # <seed> <widths...>
	local seed="$1"; shift
	for b in "$@"; do
		log "===== A: b=${b} s${seed} (${SWEEP_N}n, 1 window — this width's OWN baseline, no other width in the population) ====="
		run_arm "$seed" "SL_A_b${b}n${SWEEP_N}_${AIRFRAME}_${DIST}_s${seed}" \
			"\"stage\":\"A\",\"sweep\":\"bits\",\"bits\":${b},\"neurons\":${SWEEP_N},\"input_window_k\":1,\"seed\":${seed}" \
			--grid-bits "$b" --grid-output-neurons "$SWEEP_N" --max-output-neurons "$SWEEP_N"
	done
}

# Round 1 = THE BROAD NET: every width, one seed.
SEED1="${SEEDS%% *}"
LATER_SEEDS="${SEEDS#* }"
[ "$LATER_SEEDS" = "$SEEDS" ] && LATER_SEEDS=""
# shellcheck disable=SC2086
run_bits_round "$SEED1" $BITS
log "########## A: round 1 complete (broad net, ${SEED1}) — culling ##########"

# Cull to the promising widths; survivors get every later seed.
# shellcheck disable=SC2086
SURVIVORS=$(_cull "$MARKDIR/SL_A_b" "n${SWEEP_N}_*.json" $BITS 2>/tmp/sl_cull.txt)
while IFS= read -r line; do log "$line"; done < /tmp/sl_cull.txt
if [ -z "$SURVIVORS" ]; then
	log "ABORT: cull kept nothing — round 1 produced no readable markers."
	exit 1
fi
log "########## A: SURVIVORS = [${SURVIVORS}] (of [${BITS}]) — later seeds fly only these ##########"

for seed in $LATER_SEEDS; do
	# shellcheck disable=SC2086
	run_bits_round "$seed" $SURVIVORS
	log "########## A: seed ${seed} round complete (survivors only) ##########"
done

# The published curve keeps EVERY width (culled ones at n=1, survivors at n>=2)
# so the broad net stays readable; only survivors are eligible to win.
for b in $BITS; do
	log "A curve: b=${b} mean headline steady = $(mean_steady "$MARKDIR/SL_A_b${b}n${SWEEP_N}_*.json")"
done
for b in $SURVIVORS; do
	printf '%s %s\n' "$b" "$(mean_steady "$MARKDIR/SL_A_b${b}n${SWEEP_N}_*.json")"
done > /tmp/sl_a_curve.txt
WB=$(argmin_label < /tmp/sl_a_curve.txt)
if [ -z "$WB" ]; then
	log "ABORT: stage A produced no readable curve — nothing to carry forward."
	exit 1
fi
log "########## A DONE — bits winner b=${WB} (lowest MEAN steady across seeds, survivors only) ##########"

# ===== STAGE B: the NEURON sweep at the bits winner ==========================
for seed in $SEEDS; do
	for n in $NEURONS; do
		log "===== B: n=${n} s${seed} (b${WB}, 1 window — this capacity's OWN baseline) ====="
		run_arm "$seed" "SL_B_b${WB}n${n}_${AIRFRAME}_${DIST}_s${seed}" \
			"\"stage\":\"B\",\"sweep\":\"neurons\",\"bits\":${WB},\"neurons\":${n},\"input_window_k\":1,\"seed\":${seed}" \
			--grid-bits "$WB" --grid-output-neurons "$n" --max-output-neurons "$n"
	done
	log "########## B: seed ${seed} round complete ##########"
done

for n in $NEURONS; do
	printf '%s %s\n' "$n" "$(mean_steady "$MARKDIR/SL_B_b${WB}n${n}_*.json")"
	log "B curve: n=${n} mean headline steady = $(mean_steady "$MARKDIR/SL_B_b${WB}n${n}_*.json")"
done > /tmp/sl_b_curve.txt
WN=$(argmin_label < /tmp/sl_b_curve.txt)
if [ -z "$WN" ]; then
	log "ABORT: stage B produced no readable curve."
	exit 1
fi
log "########## B DONE — neuron winner n=${WN} (at b=${WB}) ##########"

# ===== STAGE C: the WINDOW sweep at FIXED capacity, gated ====================
# Total neurons stay at WN for every k, so levels/motor (the PWM decode
# resolution) never moves and "more windows" is the only axis.
split_for_k() {
	"$VP" -c "
import ram_controller as c
from collections import Counter
cnt = Counter(c.arch_framed1_slot_schedule(int('$WN'), int('$1'), int('$QUANTUM'), 42, 0, 0, 0))
print('/'.join(str(cnt[i]) for i in range(int('$1') - 1, -1, -1)))
" 2>/dev/null
}

for seed in $SEEDS; do
	log "===== C: k=1 s${seed} (${WN}n b${WB} min1, ONE window — the fixed-capacity control) ====="
	run_arm "$seed" "SL_C_k1_b${WB}n${WN}_${AIRFRAME}_${DIST}_s${seed}" \
		"\"stage\":\"C\",\"sweep\":\"windows\",\"conn_policy\":\"min1\",\"bits\":${WB},\"neurons\":${WN},\"input_window_k\":1,\"seed\":${seed}" \
		--conn-policy min1 \
		--grid-bits "$WB" --grid-output-neurons "$WN" --max-output-neurons "$WN"
done
BEST=$(mean_steady "$MARKDIR/SL_C_k1_*.json")
WK=1
if [ -z "$BEST" ]; then
	log "ABORT: k=1 control produced no readable markers."
	exit 1
fi
log "===== C: k=1 mean headline steady ${BEST}° ====="

for K in 2 3 4; do
	SPLIT=$(split_for_k "$K")
	if [ -z "$SPLIT" ]; then
		log "STOP: scheduler could not split ${WN}n over k=${K} — window ladder ends."
		break
	fi
	for seed in $SEEDS; do
		log "===== C: k=${K} s${seed} (SAME ${WN}n spread ${SPLIT} newest->oldest, 10 ms apart — does old state pay at zero capacity cost?) ====="
		run_arm "$seed" "SL_C_k${K}_b${WB}n${WN}_${AIRFRAME}_${DIST}_s${seed}" \
			"\"stage\":\"C\",\"sweep\":\"windows\",\"conn_policy\":\"framed1\",\"bits\":${WB},\"neurons\":${WN},\"input_window_k\":${K},\"frame_stride\":10,\"quota\":\"${SPLIT}\",\"seed\":${seed}" \
			--conn-policy framed1 --output-full-window --input-window-k "$K" --frame-stride 10 \
			--conn-mutation-scope window \
			--grid-bits "$WB" --grid-output-neurons "$WN" --max-output-neurons "$WN"
	done
	KS=$(mean_steady "$MARKDIR/SL_C_k${K}_*.json")
	if [ -z "$KS" ]; then
		log "STOP: k=${K} produced no readable markers."
		break
	fi
	log "===== C: k=${K} GATE — ${KS}° vs best-so-far k=${WK} ${BEST}° ====="
	if [ "$("$VP" -c "print(1 if float('$KS') < float('$BEST') else 0)")" = "1" ]; then
		log "k=${K} IMPROVED — proceeding to k=$((K + 1))"
		BEST="$KS"; WK="$K"
	else
		log "k=${K} did NOT improve at fixed capacity — window ladder STOPS (deeper windows starve the recent frame)."
		break
	fi
done
log "########## C DONE — window winner k=${WK} @ ${BEST}° ##########"

# ===== STAGE D: GA-NEURONS vs GA-CONNECTIVITY at the full winner =============
# Both pipelines fly here, fresh, at the SAME (b*, n*, k*) and the same seeds —
# the stage-C runs are not reused as the P2 arm because they were selected ON
# that metric, which would hand P2 a selection advantage.
if [ "$WK" -gt 1 ]; then
	SHAPE=(--conn-policy framed1 --output-full-window --input-window-k "$WK" --frame-stride 10)
	SHAPE_JSON="\"conn_policy\":\"framed1\",\"input_window_k\":${WK},\"frame_stride\":10"
else
	SHAPE=(--conn-policy min1)
	SHAPE_JSON="\"conn_policy\":\"min1\",\"input_window_k\":1"
fi
for seed in $SEEDS; do
	log "===== D: P1 s${seed} (grid -> GA-NEURONS -> GA-MEMORY at b${WB} ${WN}n k${WK}) ====="
	run_arm "$seed" "SL_D_P1_b${WB}n${WN}k${WK}_${AIRFRAME}_${DIST}_s${seed}" \
		"\"stage\":\"D\",\"pipeline\":\"neurons\",${SHAPE_JSON},\"bits\":${WB},\"neurons\":${WN},\"seed\":${seed}" \
		"${SHAPE[@]}" --skip-stages bits,connections \
		--grid-bits "$WB" --grid-output-neurons "$WN" --max-output-neurons "$WN"

	log "===== D: P2 s${seed} (grid -> GA-CONNECTIVITY(feature) -> GA-MEMORY at b${WB} ${WN}n k${WK}) ====="
	run_arm "$seed" "SL_D_P2_b${WB}n${WN}k${WK}_${AIRFRAME}_${DIST}_s${seed}" \
		"\"stage\":\"D\",\"pipeline\":\"connectivity-feature\",${SHAPE_JSON},\"bits\":${WB},\"neurons\":${WN},\"seed\":${seed}" \
		"${SHAPE[@]}" --conn-mutation-scope feature \
		--grid-bits "$WB" --grid-output-neurons "$WN" --max-output-neurons "$WN"
done
P1S=$(mean_steady "$MARKDIR/SL_D_P1_*.json")
P2S=$(mean_steady "$MARKDIR/SL_D_P2_*.json")
log "########## D DONE — at (b${WB}, ${WN}n, k${WK}): GA-neurons ${P1S:-?}° vs GA-connectivity ${P2S:-?}° ##########"
log "########## LADDER COMPLETE — $(ls "$MARKDIR" | wc -l) markers · winners b=${WB} n=${WN} k=${WK} ##########"
