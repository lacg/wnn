#!/usr/bin/env bash
# STAGE-A BITS SWEEP as the DESIRABILITY A/B (26/08/2026, Luiz).
#
# Two arms per (width, seed), everything identical except the aggregation:
#   arm GATE  = the shipped regime: --fit-aggregation zscore --zrank-clamp 3.0
#               --gate-stable 0.70 --gate-err 8.0   (control; ABI-24 behavior)
#   arm DESIR = --fit-aggregation desirability, NO gate flags (the gate is
#               emergent; the calculator REFUSES gate+desirability).
# Same S16noJM weights on both arms so the A/B isolates the aggregation
# (docs/DESIRABILITY_FITNESS_SHAPES.md). Interleaved WIDTH-MAJOR, arm pairs
# adjacent, so a paired read exists as early as possible.
#
# WHY the restart (25-26/08 findings): 0/686 during-search samples were
# feasible at 32n, so the gated arm's weights never applied — every ladder
# search ranked on the violation function, whose stable term is bounded and
# err term is not (memory: project_gate_violation_incommensurable). b=10 is
# TRIMMED (measured dead: 0.8% stable held-out). Existing SL_A markers for
# b12-b18 s31337002 are VALID gate-arm points and are reused (idempotent skip).
#
# RANKING (Luiz 26/08): stage A ranks on GATE-DISTANCE of the headline
# held-out, NOT steady — the search optimizes distance-to-flying, so the
# scoreboard must measure the same thing. Distance = desirability half-lives
# over the GATE PAIR only (err .3125 / stable .25, renormalized .5556/.4444;
# anchors 8.0 deg / 0.70) — commensurable, no new constants.
#
# STAGE A ONLY. The A/B verdict is read ONCE by Luiz before stages B-D relaunch
# under the winning aggregation (scripts/sweep_ladder_chain.sh remains for that).
set -u
ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
. "$ROOT/scripts/controller_arm_lib.sh"

LOG="/private/tmp/sweep_ladder_ab.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
OUTDIR="logs/controller/sweep_ladder"
MARKDIR="experiments/sweepladder_markers"
AIRFRAME="cf21_brushless"
DIST="L4C"
SEEDS="${SL_SEEDS:-31337002 31337003}"
BITS="${SL_BITS:-12 14 16 18 20 22 24 26 28 30 32 34 36}"   # b=10 trimmed (dead floor, measured)
SWEEP_N="${SL_SWEEP_NEURONS:-32}"
REPORT_SEEDS="99990101 99990102 99990103 99990104 99990105"

FEAT_STAGE1="--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i \
--obs-collective-cmd --obs-alt-err --obs-vz"

LADDER_WEIGHTS="${LADDER_WEIGHTS:---fit-weight-err-sq 0.3125 --fit-weight-stable 0.25 --fit-weight-steady 0.4375}"
AGG_GATE="--fit-aggregation zscore --zrank-clamp 3.0 --gate-stable 0.70 --gate-err 8.0"
AGG_DESIR="--fit-aggregation desirability"
SL_LAMBDA_ALT="${SL_LAMBDA_ALT:-0}"

log() { echo "[ladder-ab] $(date -u +%FT%TZ) $*" >> "$LOG"; }

mkdir -p "$OUTDIR" "$MARKDIR"
log "########## ARMED — stage-A bits sweep as DESIRABILITY A/B (arms paired per width) ##########"
log "bits=[$BITS] seeds=[$SEEDS] arms=[gate|desir] weights unchanged (S16noJM)"

# run_one <seed> <tag> <extra-json> <agg flags...>
run_one() {
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
		"$@" \
		--report-episodes 100 --holdout-pop-sample 8 \
		--runs 1 --memory-mode BINARY \
		--airframe "$AIRFRAME" --disturbance "$DIST" --teacher mpcof \
		$FEAT_STAGE1 \
		--translation --reward-lambda-alt "$SL_LAMBDA_ALT" \
		--grid-state-neurons 0 --max-state-neurons 0 \
		--report-seeds $REPORT_SEEDS \
		--base-seed "$seed"
	log "$tag finished rc=$?"
}

# One width, one seed, BOTH arms back-to-back (gate first: its early widths
# already carry markers, so the desir run usually starts sooner in wall-clock).
run_pair() {
	local seed="$1" b="$2"
	local base="SL_A_b${b}n${SWEEP_N}_${AIRFRAME}_${DIST}"
	log "===== A/B: b=${b} s${seed} arm=GATE ====="
	# shellcheck disable=SC2086
	run_one "$seed" "${base}_s${seed}" \
		"\"stage\":\"A\",\"sweep\":\"bits\",\"arm\":\"gate\",\"bits\":${b},\"neurons\":${SWEEP_N},\"input_window_k\":1,\"seed\":${seed}" \
		$AGG_GATE --grid-bits "$b" --grid-output-neurons "$SWEEP_N" --max-output-neurons "$SWEEP_N"
	log "===== A/B: b=${b} s${seed} arm=DESIR ====="
	# shellcheck disable=SC2086
	run_one "$seed" "${base}_desir_s${seed}" \
		"\"stage\":\"A\",\"sweep\":\"bits\",\"arm\":\"desir\",\"bits\":${b},\"neurons\":${SWEEP_N},\"input_window_k\":1,\"seed\":${seed}" \
		$AGG_DESIR --grid-bits "$b" --grid-output-neurons "$SWEEP_N" --max-output-neurons "$SWEEP_N"
}

# Gate-distance of a marker glob's headline held-out: renormalized desirability
# half-lives over the GATE PAIR (err .5556 @ anchor 8 deg, stable .4444 @ 0.70).
# min over matching markers (round 1 is n=1 per arm).
gate_distance() {
	"$VP" - "$1" <<'PY'
import glob, json, math, re, sys
K = math.log(0.5) / math.log(0.70)
best = None
for p in glob.glob(sys.argv[1]):
	try:
		d = json.load(open(p))
	except Exception:
		continue
	h = d.get("headline_holdout", "")
	ms = re.search(r"stable=([0-9.]+)%", h)
	me = re.search(r"err=([0-9.]+)", h)
	if not (ms and me):
		continue
	s = max(float(ms.group(1)) / 100.0, 1e-6)
	e = float(me.group(1))
	hd = 0.5556 * (e / 8.0) + 0.4444 * min(K * -math.log2(s), 20.0)
	best = hd if best is None else min(best, hd)
print("" if best is None else f"{best:.4f}")
PY
}

# ===== round 1: every width, first seed, both arms ===========================
SEED1="${SEEDS%% *}"
LATER_SEEDS="${SEEDS#* }"
[ "$LATER_SEEDS" = "$SEEDS" ] && LATER_SEEDS=""
for b in $BITS; do
	run_pair "$SEED1" "$b"
done
log "########## round 1 complete (both arms, ${SEED1}) — culling on GATE-DISTANCE ##########"

# Cull: a width survives if EITHER arm is promising (min across arms).
CURVE=/tmp/sl_ab_curve.txt
: > "$CURVE"
for b in $BITS; do
	gd=$(gate_distance "$MARKDIR/SL_A_b${b}n${SWEEP_N}_*.json")
	log "round-1 curve: b=${b} min gate-distance = ${gd:-unreadable}"
	[ -n "$gd" ] && printf '%s %s\n' "$b" "$gd" >> "$CURVE"
done
SURVIVORS=$("$VP" - "$CURVE" <<'PY'
import sys
K, RATIO = 6, 1.25
rows = []
for line in open(sys.argv[1]):
	p, v = line.split()
	rows.append((float(v), p))
rows.sort()
if not rows:
	sys.exit(0)
best = rows[0][0]
keep = [p for i, (v, p) in enumerate(rows) if i < K or v <= best * RATIO]
print(" ".join(keep))
PY
)
if [ -z "$SURVIVORS" ]; then
	log "ABORT: cull kept nothing — round 1 produced no readable markers."
	exit 1
fi
log "########## SURVIVORS = [${SURVIVORS}] (of [${BITS}]) — later seeds fly only these, both arms ##########"

for seed in $LATER_SEEDS; do
	for b in $SURVIVORS; do
		run_pair "$seed" "$b"
	done
	log "########## seed ${seed} round complete (survivors, both arms) ##########"
done

log "########## STAGE A A/B COMPLETE — read ONCE: paired per (width, seed), gate vs desir, on the held-out triple. Stages B-D relaunch AFTER the verdict. ##########"
