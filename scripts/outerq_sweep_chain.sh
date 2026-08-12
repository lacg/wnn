#!/usr/bin/env bash
# CALIB x LEVELS — perception resolution, alone and coupled with action resolution.
#
# WHY (09/08/2026). Two quantizers sit between the error and the correction, and
# until today only one was known:
#   ACTION      the delta alphabet — smallest nonzero correction delta_max/(levels/2).
#               Refined uniformly by the alphabet probe (L32/L64) and REFUTED at its
#               bar: L64 halved hold on s31337002 (0.53 -> 0.31) and lost on s31337003.
#   PERCEPTION  the thermometer ladder. Found mis-calibrated: thresholds were
#               quantile-fit at a hardcoded 30 deg initial tilt while every recipe
#               flies --tilt 5.0, so ladder spans ran ~6x too wide (ratio 0.17 = 5/30)
#               and the finest bin 6-17x too coarse, on 12/15 features.
#
# A lever that helps on one seed and not the other is the signature of a SECOND
# binding constraint. The two limits are MULTIPLICATIVE: however finely you can act,
# you cannot correct an error the encoder cannot resolve. This sweep tests the
# perception axis and the coupling in one shot.
#
# THE 2x2x2. calib_tilt in {5.0, 2.5} x levels in {16, 64} x seeds {002, 003}.
#   calib 5.0  = calibrate on the regime we fly (today's default after the fix)
#   calib 2.5  = deliberately NARROWER: finer near-zero bins for the hold window,
#                paid for with transient saturation. Measured saturation against the
#                flown 5 deg distribution: 30deg 11.3% of states outside the ladder,
#                5deg 31.0%, 2.5deg 39.6%, 1deg 59.1%. 1 deg is over the cliff and is
#                NOT flown; 2.5 deg is the live candidate. Saturating the transient is
#                cheap (a large error only needs "big, this sign", not its magnitude);
#                losing stable% is how this shows up if the trade is wrong.
#
# CONTROLS: already flown at calib 30 deg (the old hardcoded value) —
#   L16: CMT_lqi_..._s31337002 = 99.8/1.11/0.53 · s31337003 = 100.0/1.58/0.81
#   L64: ALP_lqi_L64_...       = 100.0/0.86/0.31 · 100.0/1.43/0.91
# No control re-flies needed; the calib axis IS the comparison against them.
#
# BAR (pre-registered): an arm beats the same-seed, same-levels calib-30 control's
# steady on BOTH base seeds without losing stable. REFUTATION: no arm does => the
# hold floor is not a resolution limit in either channel, and the structural route
# (sn>0 / output-side disturbance observer) is what remains.
#
# ORDER: interleaved — one of each (calib, levels) combo per round, both seeds of a
# combo together, cheap combos first. The standing sweep rule: round 1 yields one
# reading of every cell so a dead cell is culled before the second seed is spent.
#
# COST ~12h: L16 ~30m, L64 ~2-2.5h, x2 calib x2 seeds. One controller at a time.
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
# shellcheck source=controller_arm_lib.sh
. "$ROOT/scripts/controller_arm_lib.sh"

LOG="/private/tmp/outerq_sweep.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
OUTDIR="logs/controller/outerq_sweep"
MARKDIR="experiments/outerq_sweep_markers"
AIRFRAME="cf21_brushless"
DIST="L4C"
TEACHER="lqi"
NEURONS_GENS=5
REPORT_SEEDS="99990101 99990102 99990103 99990104 99990105"
# cell = calib:levels:seed. Interleaved: cheap L16 pair of each calib first.
# cell = q:seed. q=none is the 30deg legacy CONTROL (re-flown on current code so
# the arm is internally uniform — the old CMT_lqi control predates the
# disturbance-aware fit and is not comparable).
RUNS="${OQ_RUNS:-${OQ_CELLS:-none:31337002 none:31337003 0.02:31337002 0.02:31337003 0.005:31337002 0.005:31337003}}"
WAIT_PID="${OQ_WAIT_PID:-}"
WAIT_CEIL="${OQ_WAIT_CEIL:-259200}"

FEAT_PIDMIX="--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i"

log() { echo "[outerq] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controllers() { ps -axo pid,command 2>/dev/null \
	| grep -E "wnn.control.phased_ga|e5_residual_proof" \
	| grep -v "/usr/bin/time" | grep -v grep | grep -c python; }
# output-neuron ceiling = max(4*levels, 128) — 128 is the committee recipe's cap,
# non-binding there (on=64), so L16 stays byte-identical to the control.
max_out() { local m=$((4 * $1)); [ "$m" -lt 128 ] && m=128; echo "$m"; }

mkdir -p "$OUTDIR" "$MARKDIR"
log "########## ARMED — CALIBxLEVELS cells=[$RUNS] wait_pid=${WAIT_PID:-none} ##########"

if [ -n "$WAIT_PID" ]; then
	waited_pid=0
	while kill -0 "$WAIT_PID" 2>/dev/null; do
		[ $((waited_pid % 1800)) -eq 0 ] && log "waiting for gate PID $WAIT_PID (${waited_pid}s)"
		sleep 60; waited_pid=$((waited_pid + 60))
		[ "$waited_pid" -ge "$WAIT_CEIL" ] && { log "ABORT: gate alive after ${waited_pid}s"; exit 3; }
	done
	log "gate PID $WAIT_PID exited after ${waited_pid}s"
fi

waited=0
while [ "$(controllers)" -gt 0 ]; do
	log "waiting for the box: controllers=$(controllers) (${waited}s)"
	sleep 60; waited=$((waited + 60))
	[ "$waited" -ge "$WAIT_CEIL" ] && { log "ABORT: box busy after ${waited}s"; exit 3; }
done
log "box clear: controllers=0"

for run_id in $RUNS; do
	q="${run_id%%:*}"; seed="${run_id##*:}"
	if [ "$q" = "none" ]; then
		qtag="c30"; QFLAGS="--threshold-calib-tilt 30"
	else
		qtag="q$(echo "$q" | tr -d '.')"; QFLAGS="--threshold-calib-tilt 5.0 --threshold-outer-quantile $q"
	fi
	tag="OQ_${TEACHER}_${qtag}_${AIRFRAME}_${DIST}_s${seed}"
	run_controller_arm "$tag" \
		"$MARKDIR" "$OUTDIR" "$VP" log \
		"\"arm\":\"OUTERQ\",\"teacher\":\"${TEACHER}\",\"outer_q\":\"${q}\",\"airframe\":\"${AIRFRAME}\",\"disturbance\":\"${DIST}\",\"mode\":\"BINARY\",\"state_neurons\":0,\"neurons_gens\":${NEURONS_GENS},\"seed\":${seed},\"code\":\"post-5f3d113c\"" \
		-- \
		--levels 16 $QFLAGS \
		--skip-stages bits,connections --lamarckian \
		--max-cells 180000 --max-cells-strict \
		--neurons-gens "$NEURONS_GENS" --neurons-patience 3 \
		--memory-gens 120 --memory-patience 2 \
		--pop 50 --num-eval-folds 5 --check-interval 2 --magnitude-aware-patience \
		--eval-episodes 100 --memory-eval-episodes 200 \
		--steps 2000 --tilt 5.0 \
		--fit-weight-err-sq 0.4 --fit-weight-stable 0.3 \
		--fit-weight-jerk 0.2 --fit-weight-mono 0.1 \
		--report-episodes 100 --holdout-pop-sample 8 \
		--grid-bits 24 30 \
		--grid-state-neurons 0 --max-state-neurons 0 \
		--max-output-neurons 128 \
		--runs 1 --memory-mode BINARY \
		--airframe "$AIRFRAME" --disturbance "$DIST" --teacher "$TEACHER" \
		$FEAT_PIDMIX \
		--save-stage-checkpoints "$OUTDIR/${tag}_stages" \
		--report-seeds $REPORT_SEEDS \
		--base-seed "$seed"
	log "cell $run_id finished rc=$?"
done

log "########## OUTER-Q ARM DONE — markers in $MARKDIR ##########"
log "NEXT: per q, compare HEADLINE steady vs the c30 control cells IN THIS ARM (not the old CMT_lqi — that predates the disturbance-aware fit). Bar: beat BOTH seeds without losing stable. The hypothesis under test is that % of the FLOWN distribution left OUTSIDE the ladder is what orders hold: c30 13.3%, q=.02 4.7%, q=.005 5.6% (4-episode fits, +-1pp noise)."
