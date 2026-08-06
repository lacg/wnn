#!/usr/bin/env bash
# L2 — does a residual on the FIRMWARE CASCADE break the hold floor?
#
# WHY THIS RUN. Same finding as L1 (docs/l4_teacher_screen_results.md §"Where the
# error lives"): students are teacher-grade in recovery and hit an absolute
# hold-attitude floor (steady 0.57-0.87 deg) set by disturbance observability. L1
# attacks it by making the disturbance OBSERVABLE (d̂ feature). L2 attacks it by
# making it IRRELEVANT — the cascade's integral action cancels the sustained bias
# before the student ever sees the error, leaving the student the transient it is
# already good at.
#
# The two are a DISCRIMINATING PAIR, not two shots at one target: if L1 wins the
# floor was an input-representation limit; if only L2 wins the limit is in the
# learning/credit-assignment path and no feature would have fixed it.
#
# WHAT LANDED FIRST (commit c7fa0a2d): the residual baseline now RUNS the firmware
# cascade on an airframe that registers one — dagger.py::make_residual_baseline
# returns AttitudePidFirmware, score_controllers_metal takes af_pid_* and builds the
# kernel cascade from AttitudePidFirmwareRs itself, and the old
# _refuse_cascade_on_residual guard is gone. e5_residual_proof.py gained an airframe
# argument (argv[8]) — without it the script flies the SYNTHETIC plant and could
# never have exercised any of this.
#
# DOUBLE BASELINE — the hybrid must clear BOTH or it is not a win:
#   (a) the cascade alone     err 1.78 / stable 100.0 / steady 1.03   (classical ruler)
#   (b) the direct-student band  err ~1.29 / steady ~0.7              (L4 screen)
# HONEST CAVEAT, stated before the run: the cascade's own hold (1.03 deg) is WORSE
# than the student floor (0.57-0.87), so this is NOT "the baseline holds better".
# The bet is COMBINATION — baseline integral absorbs the bias, student supplies the
# recovery the cascade lacks. A result landing BETWEEN the two parents instead of
# below both is a real, reportable negative, not a tuning problem.
#
# Independent of the win: the cascade is the SHIPPED Crazyflie firmware loop, so a
# residual on it is the only variant flyable on hardware without replacing the stock
# controller.
#
# ARGV of e5_residual_proof.py:
#   1 seed  2 baseline  3 clamp  4 dist  5 expert  6 state_neurons  7 state_bits  8 airframe
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1

LOG="/private/tmp/l2_cascade_residual.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
OUTDIR="logs/controller/l2cascade"
MARKDIR="experiments/l2cascade_markers"
AIRFRAME="${L2_AIRFRAME:-cf21_brushless}"
DIST="${L2_DIST:-L4C}"
SEEDS="${L2_SEEDS:-31337002 31337003}"
BASELINE="${L2_BASELINE:-stock_pid}"   # stock_pid => the CASCADE on a cascade airframe.
                                       # 'pd' deliberately stays legacy (Ki=0 ablation floor).
EXPERT="${L2_EXPERT:-mpcof}"           # the screen's best teacher, matching L1's arm
CLAMP="${L2_CLAMP:-0.4}"
SN="${L2_STATE_NEURONS:-16}"
SB="${L2_STATE_BITS:-32}"
# Poll ceiling while waiting for L1 to finish (12 h; L1 is 2 x ~3 h).
WAIT_CEIL="${L2_WAIT_CEIL:-43200}"

export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1

log() { echo "[l2cascade] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controllers() { ps -axo pid,command 2>/dev/null \
	| grep -E "wnn.control.phased_ga|e5_residual_proof" \
	| grep -v "/usr/bin/time" | grep -v grep | grep -c python; }

mkdir -p "$OUTDIR" "$MARKDIR"
log "########## ARMED — L2 cascade-residual airframe=$AIRFRAME dist=$DIST baseline=$BASELINE expert=$EXPERT seeds=[$SEEDS] ##########"

# ONE CONTROLLER AT A TIME (the standing box rule). Wait for L1 rather than contend:
# two controllers on this box means both run slower AND the memory watchdog starts
# making choices for us.
waited=0
while [ "$(controllers)" -gt 0 ]; do
	if [ "$waited" -ge "$WAIT_CEIL" ]; then
		log "ABORT: still $(controllers) controller(s) after ${waited}s — giving up rather than contend."
		exit 3
	fi
	[ $((waited % 900)) -eq 0 ] && log "waiting for the box: controllers=$(controllers) (${waited}s)"
	sleep 60
	waited=$((waited + 60))
done
log "box clear after ${waited}s: controllers=0"

run_cell() {
	local seed="$1"
	local tag="L2C_${BASELINE}_${AIRFRAME}_${DIST}_s${seed}"
	local marker="${MARKDIR}/${tag}.json"
	local out="${OUTDIR}/${tag}.out"

	if [ -f "$marker" ]; then
		log "$tag: marker exists — skip"
		return 0
	fi
	log "===== START $tag ====="
	local t0=$SECONDS
	"$VP" -u "${ROOT}/scripts/e5_residual_proof.py" \
		"$seed" "$BASELINE" "$CLAMP" "$DIST" "$EXPERT" "$SN" "$SB" "$AIRFRAME" \
		> "$out" 2>&1
	local rc=$? dur=$((SECONDS - t0))

	# Marker = "this cell genuinely finished", same contract as controller_arm_lib:
	# rc=0 AND the VERDICT line actually printed. A truncated run exits clean from a
	# watchdog TERM but never reaches its verdict, and a marker there would enter a
	# hole into the table as if it were a measurement.
	if [ "$rc" -ne 0 ]; then
		log "$tag: rc=$rc after ${dur}s — NO marker (re-runnable)"
		return 1
	fi
	if ! grep -q "===== VERDICT =====" "$out"; then
		log "$tag: rc=0 but NO verdict line after ${dur}s — NO marker (truncated)"
		return 1
	fi
	printf '{"tag":"%s","arm":"L2CASCADE","baseline":"%s","airframe":"%s","disturbance":"%s","expert":"%s","clamp":%s,"state_neurons":%s,"state_bits":%s,"seed":%s,"rc":0,"dur_s":%d,"done":"%s"}\n' \
		"$tag" "$BASELINE" "$AIRFRAME" "$DIST" "$EXPERT" "$CLAMP" "$SN" "$SB" "$seed" "$dur" \
		"$(date -u +%FT%TZ)" > "$marker"
	log "$tag: rc=0 dur=${dur}s — marker written"
	return 0
}

for seed in $SEEDS; do
	log "===== ROUND seed=$seed ====="
	run_cell "$seed"
	log "seed=$seed finished rc=$?"
done

log "########## L2 CHAIN DONE — markers in $MARKDIR ##########"
