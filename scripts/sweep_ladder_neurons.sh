#!/usr/bin/env bash
# STAGE B — THE NEURONS LADDER (31/08/2026). SPEC ONLY, NOT ARMED.
#
# ⚠️ Nothing launches this. Arm it only after the b=48 long-budget run has banked
# its marker and Luiz has ruled on the top-2-vs-top-3 width carry-forward.
#
# WHAT THE AXIS ACTUALLY IS. output_neurons = num_motors · levels_per_motor
# (phased_ga --grid-output-neurons help). cf21_brushless is a quad, so this is a
# PWM DECODE RESOLUTION sweep, not an ensemble-size sweep:
#     n=32  ->  8 levels/motor   <- what ALL 33 bits-sweep markers ran at
#     n=48  -> 12 levels/motor
#     n=64  -> 16 levels/motor
#     n=96  -> 24 levels/motor
# Every value MUST be a multiple of num_motors (4). This is the axis the banked
# delta-control finding points at — the alphabet quantizes the INCREMENT, and the
# whole bits sweep has been run at the coarsest resolution we ever use.
#
# WHY NEURONS NOW RATHER THAN MORE BITS BUDGET. 33 markers, ONE neuron count. That
# is the bits-major failure feedback_sweeps_always_interleave was written about
# (IDSXD burned 37.7h dataset-major on a saturated cell while two datasets had no
# data at all). The bits region is already mapped at 13 widths; neurons is n=1.
#
# THE CELL BUDGET IS NOT A CONSTRAINT HERE — MEASURED, not assumed:
#   * --max-cells 180000 --max-cells-strict gates STRUCTURAL GROWS at mutation
#     time only. Its own docstring excludes neurogenesis ("scales cells roughly
#     linearly rather than x2^d") and cells written during training/DAgger
#     ("which no mutation gate sees").
#   * These runs pass --skip-stages neurons,bits and pin n via the grid, so NO
#     structural grow ever happens and the gate never fires. Confirmed: b=34
#     (mean 214k cells/genome) and b=36 (270k) already ran far ABOVE the 180k
#     budget with ZERO suppression events logged.
#   * Measured cells/genome at n=32: b32 166k · b34 214k · b36 270k · b48 651k
#     · b64 897k. The 8 banked levels=16 (n=64) runs at b=30 sit at 122k-266k,
#     the same order as n=32 at b=32 — so cells scale SUB-linearly in neurons.
#     Even pessimistic linear scaling puts n=96 at b=36 near 810k, under the
#     897k that b=64 already ran without trouble.
#   * If NEURONS is ever un-skipped, the cap becomes live and this note expires.
#
# TIMING IS THE UNKNOWN. Within the bits sweep s/gen was FLAT (1798-2342s) across
# a 450x range in cells, so memory size is not the time driver — rollouts are. But
# no n>32 run exists at THIS episode/step budget, so the first run of the ladder
# is also the timing measurement. Read its gen-1 s/gen before trusting the ETA.
#
# DESIGN. Top-2 widths only (b=36 hd 0.919, b=32 hd 1.144; b=34 at 1.507 is 60%
# further out and already showed a dip-and-recover). GATE arm only — it won 5 of
# 6 widths, and carrying both arms doubles the cost of an axis we know nothing
# about. Budget stays the probe's 5/3: fairness in a sweep comes from every point
# SHARING a budget, not from the budget being generous. Long-budget runs are for
# the winner, once, at the end.
#
# ORDER IS NEURON-MAJOR so every width gets a point at each resolution before any
# resolution is finished — the interleave rule. n=32 is NOT re-run: both points
# are already banked as the controls (b36 66.6%/5.94/6.53, b32 57.2%/6.45/5.99).
#
# COST: 6 runs. At the bits sweep's ~2000s/gen and ~1800s grid, ~3.4h each if the
# timing holds => ~20h total. VERIFY against run 1 before assuming the rest.
set -u
ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
. "$ROOT/scripts/controller_arm_lib.sh"

LOG="/private/tmp/sweep_ladder_neurons.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
OUTDIR="logs/controller/sweep_ladder"
MARKDIR="experiments/sweepladder_markers"
AIRFRAME="cf21_brushless"
DIST="L4C"
SEED="${SL_SEED:-31337002}"
WIDTHS="${SL_WIDTHS:-36 32}"       # top 2 by gate-distance; b=34 held in reserve
NEURONS="${SL_NEURONS:-64 96 48}"  # 16 / 24 / 12 levels per motor; n=32 already banked
REPORT_SEEDS="99990101 99990102 99990103 99990104 99990105"

FEAT_STAGE1="--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i \
--obs-collective-cmd --obs-alt-err --obs-vz"
LADDER_WEIGHTS="--fit-weight-err-sq 0.3125 --fit-weight-stable 0.25 --fit-weight-steady 0.4375"
AGG_GATE="--fit-aggregation zscore --zrank-clamp 3.0 --gate-stable 0.70 --gate-err 8.0"

log() { echo "[ladder-neurons] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controller_pids() { pgrep -f -- "-m wnn.control.phased_ga" 2>/dev/null || true; }

mkdir -p "$OUTDIR" "$MARKDIR"
log "########## ARMED — STAGE B neurons ladder ##########"
log "widths=[$WIDTHS] neurons=[$NEURONS] seed=$SEED arm=GATE budget=5/3 (n=32 controls already banked)"

run_point() {
	local b="$1" n="$2"
	local tag="SL_B_b${b}n${n}_${AIRFRAME}_${DIST}_s${SEED}"
	if [ -f "${MARKDIR}/${tag}.json" ]; then
		log "SKIP $tag (marker exists)"
		return 0
	fi
	# PURE WAIT — one controller at a time; whatever is flying finishes on its own.
	while [ -n "$(controller_pids)" ]; do sleep 20; done
	mkdir -p "$OUTDIR/ckpt/$tag"
	log "===== START $tag (b=${b}, n=${n} = $((n / 4)) levels/motor) ====="
	# shellcheck disable=SC2086
	run_controller_arm "$tag" "$MARKDIR" "$OUTDIR" "$VP" log \
		"\"stage\":\"B\",\"sweep\":\"neurons\",\"arm\":\"gate\",\"bits\":${b},\"neurons\":${n},\"levels_per_motor\":$((n / 4)),\"input_window_k\":1,\"seed\":${SEED}" \
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
		$LADDER_WEIGHTS $AGG_GATE \
		--grid-bits "$b" --grid-output-neurons "$n" --max-output-neurons "$n" \
		--report-episodes 100 --holdout-pop-sample 8 \
		--runs 1 --memory-mode BINARY \
		--airframe "$AIRFRAME" --disturbance "$DIST" --teacher mpcof \
		$FEAT_STAGE1 \
		--translation --reward-lambda-alt 0 \
		--grid-state-neurons 0 --max-state-neurons 0 \
		--report-seeds $REPORT_SEEDS \
		--base-seed "$SEED"
	log "$tag finished rc=$?"
}

# NEURON-MAJOR: every width sees each resolution before any resolution completes.
for n in $NEURONS; do
	for b in $WIDTHS; do
		run_point "$b" "$n"
	done
done
log "########## STAGE B COMPLETE — rank on gate-distance against the n=32 controls ##########"
