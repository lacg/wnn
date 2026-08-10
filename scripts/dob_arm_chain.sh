#!/usr/bin/env bash
# DOB — the output-side disturbance observer, the ONE line that makes mpcof 0.00.
#
# WHY. mpcof posts 0.00±0.00 steady while every student floors at ~0.5°. The reason
# is not perception: optimal.rs::step_rs computes u_cmd = u_policy − clamp(d̂/b) in
# f64, DOWNSTREAM of the policy. L1 handed the student that same d̂ as an INPUT
# feature and lost 4/4 — a quantized LUT cannot learn to subtract a continuous
# bias, and asking it to was the wrong request. `--dob` (ABI: dhat_ff) moves the
# teacher's line into the student's actuator path: the LUT stays a memoryless
# quantized policy and the bias cancellation happens continuously after it.
# ~6 flops/axis/step against the measured 820 instr/step, so the MCU claim
# survives. NOT the same as L2, which replaced the policy with a cascade and
# DOUBLED hold error.
#
# THE 2x2. dob ∈ {off, on} × seeds {31337002, 31337003}, teacher lqi, otherwise
# the committee control shape. The `off` cells are flown FRESH rather than reusing
# CMT_lqi because both --dhat-b and the observer must be live in both arms — the
# only difference is whether its estimate reaches the actuator. Reusing the old
# control would confound the DOB with the observer's mere presence.
#
# CALIBRATION: --threshold-calib-tilt 30 (the LEGACY value), deliberately. The
# calib sweep's first cell showed the "matched" 5° fit is 2.9x WORSE on steady and
# degrades even the GRID stage, because the thermometer is fitted on PID rollouts
# on a CLEAN plant while the STUDENT flies a worse policy under L4C — the ladder
# must cover the student's excursions, not the teacher's. Until that sweep reports,
# the DOB arm uses the calibration every published number used, so its A/B is not
# entangled with an unsettled encoder question.
#
# BAR (pre-registered): DOB-on beats DOB-off steady on BOTH seeds without losing
# stable. The interesting magnitude is large: if the observer loop is really what
# separates 0.5° from 0.00°, this should move hold by more than any lever tried so
# far. REFUTATION: no improvement ⇒ the gap is not the cancellation either, and
# what remains is genuine state (sn>0) — or the student's own policy error already
# dominates the bias term.
#
# COST ~2h: lqi ≈ 25-30 min/run × 4. One controller at a time.
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
# shellcheck source=controller_arm_lib.sh
. "$ROOT/scripts/controller_arm_lib.sh"

LOG="/private/tmp/dob_arm_chain.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
OUTDIR="logs/controller/dob_arm"
MARKDIR="experiments/dob_arm_markers"
AIRFRAME="cf21_brushless"
DIST="L4C"
TEACHER="lqi"
NEURONS_GENS=5
REPORT_SEEDS="99990101 99990102 99990103 99990104 99990105"
CALIB_TILT="${DOB_CALIB_TILT:-30}"
# cell = dob:seed. Interleaved: both arms of a seed before the next seed.
CELLS="${DOB_CELLS:-off:31337002 on:31337002 off:31337003 on:31337003}"
WAIT_PID="${DOB_WAIT_PID:-}"
WAIT_CEIL="${DOB_WAIT_CEIL:-259200}"

FEAT_PIDMIX="--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i"

log() { echo "[dob] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controllers() { ps -axo pid,command 2>/dev/null \
	| grep -E "wnn.control.phased_ga|e5_residual_proof" \
	| grep -v "/usr/bin/time" | grep -v grep | grep -c python; }

mkdir -p "$OUTDIR" "$MARKDIR"
log "########## ARMED — DOB cells=[$CELLS] calib=$CALIB_TILT wait_pid=${WAIT_PID:-none} ##########"

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

# PRE-FLIGHT (the rule three dead cohorts bought on 09-10/08): launch ONE tiny run
# with this arm's exact flag shape before committing the box to 4 cells. A 60s
# pop-6 invocation catches signature skew, wheel skew and flag typos that a green
# unit-test suite cannot — the Rust suite was 125/125 while every launch died.
log "pre-flight: tiny --dob launch"
if ! PYTHONPATH=src/wnn "$VP" -u -m wnn.control.phased_ga \
		--levels 16 --threshold-calib-tilt "$CALIB_TILT" --skip-stages bits,connections \
		--neurons-gens 1 --neurons-patience 1 --memory-gens 1 --memory-patience 1 \
		--pop 6 --num-eval-folds 1 --eval-episodes 2 --memory-eval-episodes 2 \
		--steps 200 --tilt 5.0 --report-episodes 2 --holdout-pop-sample 2 \
		--grid-bits 24 --grid-state-neurons 0 --max-state-neurons 0 \
		--max-output-neurons 128 --runs 1 --memory-mode BINARY \
		--airframe "$AIRFRAME" --disturbance "$DIST" --teacher "$TEACHER" \
		--dhat-b --dob $FEAT_PIDMIX \
		--report-seeds 99990101 --base-seed 31337002 > "$OUTDIR/PREFLIGHT.out" 2>&1; then
	log "ABORT: pre-flight FAILED — see $OUTDIR/PREFLIGHT.out. Arming nothing."
	exit 4
fi
log "pre-flight OK"

for cell in $CELLS; do
	dob="${cell%%:*}"; seed="${cell##*:}"
	tag="DOB_${TEACHER}_${dob}_${AIRFRAME}_${DIST}_s${seed}"
	DOB_FLAG=""
	[ "$dob" = "on" ] && DOB_FLAG="--dob"
	run_controller_arm "$tag" \
		"$MARKDIR" "$OUTDIR" "$VP" log \
		"\"arm\":\"DOB\",\"teacher\":\"${TEACHER}\",\"dob\":\"${dob}\",\"calib_tilt\":${CALIB_TILT},\"airframe\":\"${AIRFRAME}\",\"disturbance\":\"${DIST}\",\"obs_dhat\":false,\"mode\":\"BINARY\",\"state_neurons\":0,\"neurons_gens\":${NEURONS_GENS},\"seed\":${seed}" \
		-- \
		--levels 16 --threshold-calib-tilt "$CALIB_TILT" \
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
		--dhat-b $DOB_FLAG \
		$FEAT_PIDMIX \
		--save-stage-checkpoints "$OUTDIR/${tag}_stages" \
		--report-seeds $REPORT_SEEDS \
		--base-seed "$seed"
	log "cell $cell finished rc=$?"
done

log "########## DOB DONE — markers in $MARKDIR ##########"
log "NEXT: per seed, DOB-on vs DOB-off HEADLINE steady (both arms carry the observer; only the cancellation differs). Bar: on beats off on BOTH seeds without losing stable. Compare the gap to mpcof's 0.00 — the DOB is the teacher's own mechanism, so a null here says the student's POLICY error dominates the bias term."
