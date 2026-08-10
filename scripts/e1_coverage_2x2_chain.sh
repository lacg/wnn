#!/usr/bin/env bash
# E1 — COVERAGE 2x2: encoder package x student-state refit, 3 base seeds.
#
# WHY A 2x2 AND NOT A GREEDY STACK (10/08/2026). Two fixes attack the SAME defect —
# the fraction of the FLOWN distribution that falls outside the thermometer ladder and
# saturates to an all-0/all-1 code:
#
#   ENCODER PACKAGE  calibrate on the flown regime (tilt 5) and widen the outer
#                    quantiles into the tails. Reduces outside% by moving the ladder.
#   STUDENT REFIT    --threshold-refit-from-student: roll the grid winner out, refit
#                    on the STUDENT's own visited states (concatenated with the
#                    teacher pool), REGRID. Reduces outside% by fixing the SAMPLE the
#                    ladder is fitted on — DAgger covariate shift in the input
#                    encoding, where no amount of training can repair it.
#
# Stacking two coverage fixes can OVER-correct: the 8-bit budget is finite, so buying
# tail coverage twice coarsens the near-zero region — the exact failure that made
# calib=5 come in 2.9x worse on steady and degrade even GRID (1.39 -> 5.34). A greedy
# "refit on top of the q winner" chain cannot see that; it reads a joint
# over-correction as "refit does not work". The interaction term is the whole point.
#
# ⚠️ FACTOR A IS A PACKAGE, NOT A CLEAN QUANTILE EFFECT. c30 = --threshold-calib-tilt
# 30 (legacy positions); the q arm = --threshold-calib-tilt 5.0 + --threshold-outer-
# quantile Q. Those differ in BOTH calibration tilt and quantile position, exactly as
# the in-flight outer-q arm does. This is deliberate — it keeps continuity with that
# arm and asks the question that actually gates the recipe freeze ("does the new
# encoder package beat the legacy one, and does refit add anything on top of either")
# — but it must NEVER be written up as "the outer quantile is the lever". Isolating
# the quantile from the tilt needs its own arm and is not what this one measures.
#
# THE 2x2x3. enc in {c30, qQ} x refit in {off, on} x seeds {002, 003, 004} = 12 runs.
#   3 SEEDS, not 2. The committee closure measured a 0.91 deg BETWEEN-seed range
#   against 0.67-1.04 deg within-seed spreads, and both L3 and the alphabet probe
#   died on "one seed each way". s004 is systematically harder (~0.5-0.7 deg worse on
#   every teacher) — that is a feature: a coverage fix that only works on easy seeds
#   is not a fix.
#
# THE REGRID IS NOT AN EXTRA DRAW. --threshold-refit-from-student re-runs stage0 as
# stage0_grid(args, ec, seed, thresholds_override=thr2) — SAME seed, SAME grid points,
# only the address function differs. If the refit were a no-op the second grid would
# reproduce the first bit-for-bit. So refit-on does not get a second chance at the
# grid; the only asymmetry is wall-clock. (Verified at phased_ga.py:1712 before this
# arm was designed — preserve that property if the refit path is ever touched.)
#
# BAR (pre-registered, written BEFORE any cell ran): the winning combination must beat
# the c30/refit-off control cell's HEADLINE steady on ALL THREE seeds without losing
# stable. Compare HEADLINE-to-HEADLINE, never NEURONS-to-NEURONS: stage selection
# routinely headlines different stages across seeds (the outer-q controls headlined
# NEURONS#0 and MEMORY#1 respectively), so a fixed-stage comparison is comparing two
# different objects. REFUTATION: no combination clears all three => coverage is not
# what orders the hold floor, the outside% proxy is correlational only, and the
# structural route (sn>0 / state neurons) is what remains.
#
# GRID IS NOT THE READ-OUT. Across the two outer-q control seeds GRID moved OPPOSITE
# to the trained stages (+0.83 vs -0.27 on steady) with a ~35x fatter spread
# (+-0.74 vs +-0.02). An effect visible only in GRID is an encoder artifact, not a
# control win.
#
# ORDER: interleaved — round 1 is one reading of all FOUR combos on s002, then s003,
# then s004. The standing sweep rule: a dead combo is culled before the second and
# third seeds are spent on it.
#
# COST ~6h: ~30 min/cell measured (outer-q cells ran 1532s and 2082s), x12, plus the
# refit cells' extra GRID stage. One controller at a time.
#
# ARMING. Factor A is resolved from the outer-q markers AFTER the gate clears (see
# the resolver block below); no winner => the arm degenerates to refit-only, 6 cells.
#   E1_WAIT_PID=<dob chain pid> nohup scripts/e1_coverage_2x2_chain.sh &
# E1_Q is RESOLVED from the outer-q markers after the gate; set it only to override.
# (macOS has no setsid — use `nohup ... &` then verify PPID=1, per
# feedback_detach_background_processes.)
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
# shellcheck source=controller_arm_lib.sh
. "$ROOT/scripts/controller_arm_lib.sh"

LOG="/private/tmp/e1_coverage.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
OUTDIR="logs/controller/e1_coverage"
MARKDIR="experiments/e1_coverage_markers"
OQ_MARKDIR="${E1_OQ_MARKDIR:-experiments/outerq_sweep_markers}"
AIRFRAME="cf21_brushless"
DIST="L4C"
TEACHER="lqi"
NEURONS_GENS=5
REPORT_SEEDS="99990101 99990102 99990103 99990104 99990105"
REFIT_EPISODES="${E1_REFIT_EPISODES:-10}"

SEEDS="${E1_SEEDS:-31337002 31337003 31337004}"
WAIT_PID="${E1_WAIT_PID:-}"
WAIT_CEIL="${E1_WAIT_CEIL:-259200}"

FEAT_PIDMIX="--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i"

log() { echo "[e1] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controllers() { ps -axo pid,command 2>/dev/null \
	| grep -E "wnn.control.phased_ga|e5_residual_proof" \
	| grep -v "/usr/bin/time" | grep -v grep | grep -c python; }

mkdir -p "$OUTDIR" "$MARKDIR"
# NB: E1_Q and CELLS are resolved AFTER the gate (see the resolver block), so they are
# deliberately NOT referenced here — under `set -u` that would abort the chain at arm
# time with an unbound-variable error instead of running it.
log "########## ARMED — E1 COVERAGE seeds=[$SEEDS] refit_eps=$REFIT_EPISODES q=${E1_Q:-<resolve after gate>} wait_pid=${WAIT_PID:-none} ##########"

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

# RESOLVE FACTOR A FROM THE OUTER-Q MARKERS, AFTER the gate (10/08/2026).
#
# E1_Q was originally a required parameter with no default, so that a human had to
# read the outer-q result before this arm could run. That is still the intent — but a
# 6-hour arm finishing at 03:00 should not have to wait for someone to wake up. The
# resolver applies the SAME pre-registered rule a human would (beat the control's
# headline steady on EVERY seed without losing stable) and logs the full comparison
# table, so the reasoning is in the chain log rather than in someone's head.
#
# Resolution happens AFTER the gate wait, not at arm time: the outer-q arm is still
# flying when this chain is armed, so resolving early would read an incomplete arm.
#
# NO WINNER IS A FIRST-CLASS OUTCOME. If no quantile clears the bar, factor A has no
# live level and the 2x2 DEGENERATES to the refit-only arm on the control encoder —
# 6 runs instead of 12. That is the honest response: the interaction this 2x2 exists
# to detect is only meaningful between two WORKING fixes, and half the cells would
# otherwise re-measure a refuted encoder. Never fall back to the least-bad quantile;
# best-of-N on a refuted arm is how a failed lever gets promoted into a paper.
if [ -n "${E1_Q:-}" ]; then
	log "factor A: E1_Q=$E1_Q set explicitly — skipping resolution"
else
	log "resolving factor A from $OQ_MARKDIR (bar: beat the c30 control on EVERY seed, no stable loss)"
	E1_Q="$(PYTHONPATH=src/wnn "$VP" scripts/pick_best_arm_config.py \
		"$OQ_MARKDIR" outer_q none 2>>"$LOG")"
	E1_Q="$(echo "$E1_Q" | tr -d '[:space:]')"
fi

if [ -n "$E1_Q" ]; then
	ENC_LEVELS="c30 q"
	log "factor A RESOLVED: q=$E1_Q beat the control — running the FULL 2x2 (12 cells)"
else
	ENC_LEVELS="c30"
	log "factor A REFUTED: no quantile beat the control — DEGENERATING to the refit-only"
	log "  arm on the c30 control encoder (6 cells). The 2x2 interaction is only"
	log "  meaningful between two working fixes; re-measuring a refuted encoder is waste."
fi

# Interleaved: every combo once on the first seed, then the next — the standing sweep
# rule, so a dead combo is culled before later seeds are spent on it.
CELLS=""
for s in $SEEDS; do
	for e in $ENC_LEVELS; do
		for r in off on; do CELLS="$CELLS $e:$r:$s"; done
	done
done
log "cells (interleaved): [$CELLS]"

# PRE-FLIGHT (the rule three dead cohorts bought on 09-10/08): launch ONE tiny run
# with this arm's exact flag shape before committing the box to 12 cells. A 60s pop-6
# invocation catches signature skew, wheel skew and flag typos that a green unit-test
# suite cannot — the Rust suite was 125/125 while every launch died, twice. This arm
# is the FIRST flight of --threshold-refit-from-student, so the refit flag is in the
# pre-flight shape deliberately: an unflown code path is exactly what this catches.
log "pre-flight: tiny --threshold-refit-from-student launch"
if ! PYTHONPATH=src/wnn "$VP" -u -m wnn.control.phased_ga \
		--levels 16 --threshold-calib-tilt 30 \
		--threshold-refit-from-student --threshold-refit-episodes 2 \
		--skip-stages bits,connections \
		--neurons-gens 1 --neurons-patience 1 --memory-gens 1 --memory-patience 1 \
		--pop 6 --num-eval-folds 1 --eval-episodes 2 --memory-eval-episodes 2 \
		--steps 200 --tilt 5.0 --report-episodes 2 --holdout-pop-sample 2 \
		--grid-bits 24 --grid-state-neurons 0 --max-state-neurons 0 \
		--max-output-neurons 128 --runs 1 --memory-mode BINARY \
		--airframe "$AIRFRAME" --disturbance "$DIST" --teacher "$TEACHER" \
		$FEAT_PIDMIX \
		--report-seeds 99990101 --base-seed 31337002 > "$OUTDIR/PREFLIGHT.out" 2>&1; then
	log "ABORT: pre-flight FAILED — see $OUTDIR/PREFLIGHT.out. Arming nothing."
	exit 4
fi
# The refit is a PLACEBO if the student's samples are swamped by the teacher pool
# (measured: 2 samples/feature moved the ladder 1.00x). Prove the collector actually
# fired in the pre-flight rather than silently falling back to the teacher fit.
if ! grep -q "thr-refit.*REGRIDDING" "$OUTDIR/PREFLIGHT.out"; then
	log "ABORT: pre-flight ran but the refit never REGRIDDED — the flag is inert."
	log "       (grep '\[thr-refit\]' $OUTDIR/PREFLIGHT.out for which guard returned early)"
	exit 5
fi
log "pre-flight OK — refit collected samples and regridded"

for cell in $CELLS; do
	enc="${cell%%:*}"; rest="${cell#*:}"; refit="${rest%%:*}"; seed="${rest##*:}"
	if [ "$enc" = "c30" ]; then
		ENCFLAGS="--threshold-calib-tilt 30"
	else
		ENCFLAGS="--threshold-calib-tilt 5.0 --threshold-outer-quantile $E1_Q"
	fi
	if [ "$refit" = "on" ]; then
		REFITFLAGS="--threshold-refit-from-student --threshold-refit-episodes $REFIT_EPISODES"
	else
		REFITFLAGS=""
	fi
	tag="E1_${TEACHER}_${enc}_refit${refit}_${AIRFRAME}_${DIST}_s${seed}"
	run_controller_arm "$tag" \
		"$MARKDIR" "$OUTDIR" "$VP" log \
		"\"arm\":\"E1COV\",\"teacher\":\"${TEACHER}\",\"enc\":\"${enc}\",\"outer_q\":\"${E1_Q}\",\"refit\":\"${refit}\",\"refit_episodes\":${REFIT_EPISODES},\"airframe\":\"${AIRFRAME}\",\"disturbance\":\"${DIST}\",\"mode\":\"BINARY\",\"state_neurons\":0,\"neurons_gens\":${NEURONS_GENS},\"seed\":${seed},\"code\":\"post-5f3d113c\"" \
		-- \
		--levels 16 $ENCFLAGS $REFITFLAGS \
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
	log "cell $cell finished rc=$?"
done

log "########## E1 COVERAGE 2x2 DONE — markers in $MARKDIR ##########"
log "NEXT: per seed, tabulate the four HEADLINE triples. Report the two main effects AND the interaction (enc x refit); a positive main effect for each with a negative interaction is the over-correction this arm exists to detect. Bar: one combination beats c30/refit-off on ALL THREE seeds without losing stable. Then E2 (--delta-gamma 2.0 vs 1.0) flies on the winning encoder."
