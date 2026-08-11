#!/usr/bin/env bash
# S1 — sn>0 / STATE NEURONS: the last untested route to the hold floor.
#
# WHY NOW (10/08/2026). Every "give the substrate finer numbers" lever has failed:
#   perception  outer-quantile 0.02/0.005   REFUTED 4/4 (worse steady AND lost stable)
#   action      L64 uniform alphabet        1 seed each way (UNRESOLVED at its n)
#   action      gamma=2 warped alphabet     REFUTED 3/3, CI [+0.27, +1.05]
#   perception  student-state refit         OPEN, effect ~ 0
# Resolution is not the binding constraint in either channel. L1 already showed a
# bias estimate cannot be INJECTED as an input feature (refuted 4/4); L1b showed the
# floor survives a ranking that explicitly targets steady. What is left is
# EXPRESSIVENESS: an observer needs somewhere to CARRY its estimate. sn>0 asks
# whether the substrate can LEARN the integrator it cannot be handed.
#
# ⚠️ WNN_STATE_SPLIT=1 IS REQUIRED AND IS NOT IMPLIED BY sn>0. reward_gated.py:477
# reads `use_split = os.environ.get("WNN_STATE_SPLIT") == "1"`. Without it the legacy
# bptt path runs and the conflict-driven split trainer — the thing that actually
# writes state under cross-episode conflict, worth +20pp when it landed — never
# engages. This is the third env/flag trap in two days (after the refit placebo and
# --dob's silent no-op), so the pre-flight ASSERTS the state cells were populated
# rather than trusting the exit code.
#
# TEACHER = lqi, NOT mpcof (deviates from reassessment section 5, deliberately):
#   * The sn=0 CONTROLS ALREADY EXIST for lqi at n=5, flown on THIS code
#     (E1 refit-off cells, 0.36-0.93 deg headline), verified by
#     `git log 5f3d113c..HEAD -- src/wnn/control/` being empty. mpcof's sn=0 controls
#     predate the encoder fixes and would ALL need re-flying.
#   * mpcof runs 2.7-3.2 h; lqi runs ~0.5 h. lqi: 6 runs ~3 h with free controls.
#     mpcof: 9 runs ~27 h. That is 9x the cost for the same number of seeds.
#   * Today's variance lesson says spend budget on SEEDS, not on replicates.
#   ⚠️ HONEST CAVEAT: a NULL ON lqi DOES NOT REFUTE mpcof. The section-5 argument for
#   mpcof is real — it is the 0.00 deg steady teacher whose entire advantage IS the
#   observer loop, so state may only pay off there. lqi is the cheap POWERED SCREEN;
#   mpcof round 2 is planned either way and its result is what decides the route.
#
# THE ARM. sn in {4, 8} x seeds {002, 003, 004} = 6 runs, interleaved (both sn on the
# first seed, then the next). Controls are the EXISTING E1 refit-off cells on the same
# seeds — not re-flown, because the controller source is byte-identical since 5f3d113c
# and that control reproduced bit-for-bit across three independently launched chains.
#
# BAR — the CI rule, NOT unanimity (the all-seeds bar was withdrawn 10/08; it grows
# stricter with N, so more evidence made a real effect harder to show). Decide on the
# 95% CI of the paired delta on HEADLINE steady:
#   CI entirely below 0 -> PROMOTE ; entirely above -> REFUTE ;
#   spans 0, half-width <= 0.15 deg -> genuine null ; spans 0 and wide -> more seeds.
#
# SECONDARY READ-OUT (section 5, kept): does the advantage GROW WITH EPISODE LENGTH?
# An integrator needs time to converge; a pure quantization effect would not show
# length-dependence. Not measured by this chain — it is a re-score of the winners.
#
# CELL BUDGET IS HELD AT THE CONTROL'S 180k, deliberately. State neurons add address
# bits, so the cap may BIND HARDER for sn>0 than for sn=0 — a real confound. Raising
# it would break comparability with the free controls, which is worse. If an sn>0 cell
# reports cells at the cap, say so in the write-up rather than silently comparing.
#
# DEPLOYMENT HONESTY (section 5): sn>0 adds a recurrent read-modify-write to the
# measured 820-instr/step loop. The MCU harness MUST re-measure before any sn>0 number
# sits next to the compute claim in a paper table.
#
# ARMING:  SN_WAIT_PID=<last chain pid> nohup scripts/sn_state_neurons_chain.sh &
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
# shellcheck source=controller_arm_lib.sh
. "$ROOT/scripts/controller_arm_lib.sh"

LOG="/private/tmp/sn_state.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
OUTDIR="logs/controller/sn_state"
MARKDIR="experiments/sn_state_markers"
AIRFRAME="cf21_brushless"
DIST="L4C"
TEACHER="${SN_TEACHER:-lqi}"
NEURONS_GENS=5
REPORT_SEEDS="99990101 99990102 99990103 99990104 99990105"
SEEDS="${SN_SEEDS:-31337002 31337003 31337004}"
SN_LEVELS="${SN_LEVELS:-4 8}"
WAIT_PID="${SN_WAIT_PID:-}"
WAIT_CEIL="${SN_WAIT_CEIL:-259200}"

FEAT_PIDMIX="--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i"

export WNN_STATE_SPLIT=1   # REQUIRED — see the header. sn>0 does NOT imply it.

log() { echo "[sn] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controllers() { ps -axo pid,command 2>/dev/null \
	| grep -E "wnn.control.phased_ga|e5_residual_proof" \
	| grep -v "/usr/bin/time" | grep -v grep | grep -c python; }

mkdir -p "$OUTDIR" "$MARKDIR"
log "########## ARMED — sn>0 teacher=$TEACHER sn=[$SN_LEVELS] seeds=[$SEEDS] WNN_STATE_SPLIT=1 wait_pid=${WAIT_PID:-none} ##########"

if [ -n "$WAIT_PID" ]; then
	waited=0
	while kill -0 "$WAIT_PID" 2>/dev/null; do
		[ $((waited % 1800)) -eq 0 ] && log "waiting for gate PID $WAIT_PID (${waited}s)"
		sleep 60; waited=$((waited + 60))
		[ "$waited" -ge "$WAIT_CEIL" ] && { log "ABORT: gate alive after ${waited}s"; exit 3; }
	done
	log "gate PID $WAIT_PID exited after ${waited}s"
fi

waited=0
while [ "$(controllers)" -gt 0 ]; do
	log "waiting for the box: controllers=$(controllers) (${waited}s)"
	sleep 60; waited=$((waited + 60))
	[ "$waited" -ge "$WAIT_CEIL" ] && { log "ABORT: box busy after ${waited}s"; exit 3; }
done
log "box clear: controllers=0"

# PRE-FLIGHT. Parsing is not engaging — assert the STATE CELLS were actually written.
# phased_ga's FPGA line reports "(state P/T, output ...)"; P==0 means the state
# neurons exist in the architecture but carry nothing, i.e. the arm would measure a
# perfect null while completing cleanly. Same guard shape as E1's REGRIDDING check
# and DOB's observer-b check.
log "pre-flight: tiny sn=4 launch with WNN_STATE_SPLIT=1"
if ! PYTHONPATH=src/wnn "$VP" -u -m wnn.control.phased_ga \
		--levels 16 --threshold-calib-tilt 30 --skip-stages bits,connections \
		--neurons-gens 1 --neurons-patience 1 --memory-gens 1 --memory-patience 1 \
		--pop 6 --num-eval-folds 1 --eval-episodes 2 --memory-eval-episodes 2 \
		--steps 200 --tilt 5.0 --report-episodes 2 --holdout-pop-sample 2 \
		--grid-bits 24 --grid-state-neurons 4 --max-state-neurons 4 \
		--max-output-neurons 128 --runs 1 --memory-mode BINARY \
		--airframe "$AIRFRAME" --disturbance "$DIST" --teacher "$TEACHER" \
		$FEAT_PIDMIX \
		--report-seeds 99990101 --base-seed 31337002 > "$OUTDIR/PREFLIGHT.out" 2>&1; then
	log "ABORT: pre-flight FAILED — see $OUTDIR/PREFLIGHT.out. Arming nothing."
	exit 4
fi
if grep -qE "\(state 0/[0-9]+" "$OUTDIR/PREFLIGHT.out"; then
	log "ABORT: pre-flight ran but populated ZERO state cells — the state neurons are"
	log "       dead weight and the arm would measure a perfect null. Check that"
	log "       WNN_STATE_SPLIT=1 reached the trainer. See $OUTDIR/PREFLIGHT.out"
	exit 5
fi
if ! grep -q "sn=4" "$OUTDIR/PREFLIGHT.out"; then
	log "ABORT: pre-flight produced no sn=4 architecture — the state axis did not take."
	exit 6
fi
log "pre-flight OK — sn=4 architecture built and state cells populated"

CELLS=""
for s in $SEEDS; do for n in $SN_LEVELS; do CELLS="$CELLS $n:$s"; done; done
log "cells (interleaved): [$CELLS]"

for cell in $CELLS; do
	sn="${cell%%:*}"; seed="${cell##*:}"
	tag="S1_${TEACHER}_sn${sn}_${AIRFRAME}_${DIST}_s${seed}"
	run_controller_arm "$tag" \
		"$MARKDIR" "$OUTDIR" "$VP" log \
		"\"arm\":\"SN\",\"teacher\":\"${TEACHER}\",\"state_neurons\":${sn},\"split\":true,\"airframe\":\"${AIRFRAME}\",\"disturbance\":\"${DIST}\",\"mode\":\"BINARY\",\"neurons_gens\":${NEURONS_GENS},\"seed\":${seed},\"code\":\"post-5f3d113c\"" \
		-- \
		--levels 16 --threshold-calib-tilt 30 \
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
		--grid-state-neurons "$sn" --max-state-neurons "$sn" \
		--max-output-neurons 128 \
		--runs 1 --memory-mode BINARY \
		--airframe "$AIRFRAME" --disturbance "$DIST" --teacher "$TEACHER" \
		$FEAT_PIDMIX \
		--save-stage-checkpoints "$OUTDIR/${tag}_stages" \
		--report-seeds $REPORT_SEEDS \
		--base-seed "$seed"
	log "cell $cell finished rc=$?"
done

log "########## sn>0 ARM DONE — markers in $MARKDIR ##########"
log "NEXT: pair each sn cell against the SAME-SEED E1 refit-off control (experiments/e1_coverage_markers) on HEADLINE steady; compute the paired 95% CI per sn level. Report whether any sn cell hit the 180k cell cap (it binds harder with state bits — a confound to disclose, not hide). If lqi moves, run mpcof to test the mechanism; if lqi is null, that does NOT refute mpcof and the section-5 argument still stands."
