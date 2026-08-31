#!/usr/bin/env bash
# STAGE C — THE GAMMA A/B, THEN THE LEVELS LADDER (31/08/2026, Luiz's design).
#
# PHASE 1 rules PHASE 2. Phase 2 runs ONLY if gamma is not detrimental, and this
# script decides that from the markers rather than asking a human mid-chain.
#
# WHY GAMMA FIRST. --delta-control is ON, so the network emits a per-step PWM
# DELTA into a leaky accumulator, and controller.rs fixes the quantum:
#     smallest nonzero correction = delta_max / (levels/2)
# The help text's "delta_max/8" is that formula AT 16 LEVELS, not a constant. At
# our 8 levels/motor it is 0.1/4 = 0.025 PWM, and with --delta-leak 0.95 the
# smallest SUSTAINABLE offset is 0.025/(1-0.95) = 0.5 PWM — half of full
# authority. "Holding an equilibrium means orbiting it in a limit cycle of that
# amplitude" (controller.rs). That is a mechanical account of why `steady` is the
# sweep's worst column (6.53° at b=36) and why more input bits barely move it:
# steady-state error here is set by the OUTPUT ALPHABET, not the input lens.
#
# --delta-gamma shapes |t|^gamma before scaling: same range, same neutral, same
# level count, SAME FOOTPRINT — resolution concentrated near zero where the limit
# cycle lives, coarser near full authority where the transient dominates and
# precision is worthless. At 8 levels gamma=2 takes the finest step 0.025 ->
# 0.00625 PWM: twice as fine as DOUBLING the neurons would get, for zero extra
# neurons and zero extra cells. Footprint matters — the alphabet probe's L64 was
# banked REFUTED partly on "footprint the FPGA/MCU claim cannot spend".
#
# PHASE 1 — the A/B, 2 runs. b in {36,32} at n=32 (8 levels/motor), gamma=2. The
# CONTROLS ARE ALREADY BANKED: the same two shapes at gamma=1 (SL_A markers,
# b36 66.6%/5.94/6.53 hd 0.919, b32 57.2%/6.45/5.99 hd 1.144). Nothing is re-run.
#
# THE GATE. Gamma is DETRIMENTAL if its gate-distance is worse than its gamma=1
# control on BOTH widths. Then phase 2 does not run and the box goes idle for
# Luiz — fail closed, exactly as the conditional was given. If it wins on either
# width, phase 2 proceeds WITH gamma.
#
# PHASE 2 — the levels ladder, 6 runs. b in {36,32} x n in {64,96,256}, i.e.
# 16 / 24 / 64 levels per motor, neuron-major so both widths see each resolution
# before any resolution finishes (the interleave rule).
#
# ⚠️ n=256 IS THE EXPENSIVE ONE. Cells scale ~linearly in neurons (measured: the
# alphabet probe went 136k -> 597k for 4x the neurons), so b=36 at n=256 projects
# to ~1.2M cells/genome, ~4.5x the n=32 point. It is also the shape the alphabet
# probe banked as refuted on footprint. It runs LAST so the cheaper points are
# banked first if it has to be killed.
set -u
ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
. "$ROOT/scripts/controller_arm_lib.sh"

LOG="/private/tmp/sweep_ladder_gamma.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
OUTDIR="logs/controller/sweep_ladder"
MARKDIR="experiments/sweepladder_markers"
AIRFRAME="cf21_brushless"; DIST="L4C"; SEED="${SL_SEED:-31337002}"
WIDTHS="${SL_WIDTHS:-36 32}"
PHASE2_NEURONS="${SL_NEURONS:-64 96 256}"   # 16 / 24 / 64 levels per motor
GAMMA="${SL_GAMMA:-2.0}"
REPORT_SEEDS="99990101 99990102 99990103 99990104 99990105"

FEAT="--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i \
--obs-collective-cmd --obs-alt-err --obs-vz"
WEIGHTS="--fit-weight-err-sq 0.3125 --fit-weight-stable 0.25 --fit-weight-steady 0.4375"
AGG_GATE="--fit-aggregation zscore --zrank-clamp 3.0 --gate-stable 0.70 --gate-err 8.0"

log() { echo "[ladder-gamma] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controller_pids() { pgrep -f -- "-m wnn.control.phased_ga" 2>/dev/null || true; }

# Gate-distance of a marker's headline held-out, in desirability half-lives over
# the GATE PAIR (err .5556 @ 8.0deg, stable .4444 @ 0.70). Lifted verbatim from
# sweep_ladder_probe_wide.sh so stage C ranks on exactly the scale stage A did.
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

# run_point <bits> <neurons> <gamma>
run_point() {
	local b="$1" n="$2" g="$3"
	local gtag; gtag="g$(echo "$g" | tr -d '.')"
	local tag="SL_C_b${b}n${n}_${AIRFRAME}_${DIST}_${gtag}_s${SEED}"
	if [ -f "${MARKDIR}/${tag}.json" ]; then
		log "SKIP $tag (marker exists)"
		return 0
	fi
	# PURE WAIT — one controller at a time; whatever is flying finishes untouched.
	while [ -n "$(controller_pids)" ]; do sleep 20; done
	mkdir -p "$OUTDIR/ckpt/$tag"
	log "===== START $tag (b=${b}, n=${n} = $((n / 4)) levels/motor, gamma=${g}) ====="
	# shellcheck disable=SC2086
	run_controller_arm "$tag" "$MARKDIR" "$OUTDIR" "$VP" log \
		"\"stage\":\"C\",\"sweep\":\"gamma-levels\",\"arm\":\"gate\",\"bits\":${b},\"neurons\":${n},\"levels_per_motor\":$((n / 4)),\"delta_gamma\":${g},\"input_window_k\":1,\"seed\":${SEED}" \
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
		$WEIGHTS $AGG_GATE \
		--delta-gamma "$g" \
		--grid-bits "$b" --grid-output-neurons "$n" --max-output-neurons "$n" \
		--report-episodes 100 --holdout-pop-sample 8 \
		--runs 1 --memory-mode BINARY \
		--airframe "$AIRFRAME" --disturbance "$DIST" --teacher mpcof \
		$FEAT \
		--translation --reward-lambda-alt 0 \
		--grid-state-neurons 0 --max-state-neurons 0 \
		--report-seeds $REPORT_SEEDS \
		--base-seed "$SEED"
	log "$tag finished rc=$?"
}

log "########## ARMED — STAGE C: gamma A/B (n=32) then the levels ladder ##########"
log "widths=[$WIDTHS] gamma=$GAMMA phase2_neurons=[$PHASE2_NEURONS] seed=$SEED arm=GATE budget=5/3"

# ---- PHASE 1: gamma=2 at the banked shape. Controls are the SL_A gamma=1 markers.
for b in $WIDTHS; do
	run_point "$b" 32 "$GAMMA"
done

# ---- THE GATE. Compare each width against its own gamma=1 control.
log "---------- PHASE 1 VERDICT ----------"
wins=0
for b in $WIDTHS; do
	gtag="g$(echo "$GAMMA" | tr -d '.')"
	hd_g=$(gate_distance "${MARKDIR}/SL_C_b${b}n32_${AIRFRAME}_${DIST}_${gtag}_s${SEED}.json")
	hd_1=$(gate_distance "${MARKDIR}/SL_A_b${b}n32_${AIRFRAME}_${DIST}_s${SEED}.json")
	if [ -z "$hd_g" ] || [ -z "$hd_1" ]; then
		log "b=${b}: MISSING gate-distance (gamma='${hd_g:-none}' control='${hd_1:-none}') — cannot judge."
		continue
	fi
	verdict=$(awk -v a="$hd_g" -v b="$hd_1" 'BEGIN{print (a<b) ? "BETTER" : "worse"}')
	log "b=${b}: gamma=${GAMMA} hd=${hd_g}  vs  gamma=1 hd=${hd_1}  -> ${verdict}"
	[ "$verdict" = "BETTER" ] && wins=$((wins + 1))
done

if [ "$wins" = "0" ]; then
	log "########## STOPPED (fail-closed): gamma is DETRIMENTAL on every width judged."
	log "Phase 2 NOT launched — the levels ladder was conditional on gamma not hurting."
	log "Box is idle. Luiz's call: re-run phase 2 at gamma=1 with SL_GAMMA=1.0, or drop the axis. ##########"
	exit 0
fi
log "gamma wins on ${wins}/$(echo $WIDTHS | wc -w | tr -d ' ') widths — PHASE 2 PROCEEDS with gamma=${GAMMA}."

# ---- PHASE 2: the levels ladder, NEURON-MAJOR, cheapest resolution first.
for n in $PHASE2_NEURONS; do
	for b in $WIDTHS; do
		run_point "$b" "$n" "$GAMMA"
	done
done
log "########## STAGE C COMPLETE — rank on gate-distance against the SL_A n=32 gamma=1 controls ##########"
