#!/usr/bin/env bash
# L3 — is the hold floor an ACTUATION-RESOLUTION limit?
#
# WHY THIS RUN. L1, L1b and L2 are all flown and all refuted
# (docs/l4_teacher_screen_results.md section "Hold-floor levers"):
#   L1   gave the student the disturbance as input   -> steady got WORSE (4/4 with L1b)
#   L2   handed the disturbance to a controller that
#        provably cancels it (firmware cascade I-term) -> steady roughly DOUBLED
#   L1b  ranked genomes explicitly on steady (S16)     -> steady did not reliably move
# Neither more input, nor a better substrate, nor a hold-targeted objective moves the
# floor => it is STRUCTURAL, and the remaining structure is how the student ACTUATES.
#
# THE MECHANISM UNDER TEST. The deployed students run delta control:
#     pwm <- 0.5 + delta_leak*(pwm - 0.5) + Delta
# with the 17-value alphabet quantizing DELTA (step = delta_max/8), not the throttle.
# Holding a constant Delta settles the accumulator at Delta/(1-delta_leak), so the
# SMALLEST NON-ZERO SUSTAINED THROTTLE OFFSET is
#     (delta_max/8)/(1-delta_leak) = 0.0125/0.05 = 0.25 pwm
# at the defaults — enormous next to what a 0.5 deg hold needs. The rollout trace shows
# the student evading it by duty-cycling the increment (70-82% of steady-tail steps emit
# a non-zero Delta, mixed signs) so the leaky accumulator averages it: a Delta-Sigma
# modulator that buys effectively continuous actuation AT THE PRICE OF A LIMIT-CYCLE
# RIPPLE. Hypothesis: THE RIPPLE IS THE HOLD FLOOR.
#
# THE DESIGN'S POINT IS THE PAIRING, NOT A SWEEP. Granularity falls if you shrink
# delta_max OR if you make the accumulator leakier. The two arms reach the IDENTICAL
# predicted granularity by opposite means and pay OPPOSITE prices:
#
#   arm        delta_max  delta_leak  smallest sustained offset  price
#   control    0.1        0.95        0.2500 pwm  (already flown)  --
#   A (step)   0.025      0.95        0.0625 pwm  (4x finer)      slew authority (max
#                                                                  Delta/step drops 4x)
#   B (leak)   0.1        0.80        0.0625 pwm  (4x finer)      integrator memory
#                                                                  (offsets decay 4x faster)
#
#   BOTH improve steady, similarly  => granularity IS the mechanism; the floor is an
#                                      actuation-resolution limit. That is the finding.
#   ONLY A improves                 => not granularity but SLEW AUTHORITY.
#   ONLY B improves                 => not granularity but INTEGRATOR MEMORY.
#   NEITHER improves                => actuation is ruled out too, and L4
#                                      (magnitude-weighted DAGGER conflict writes) is the
#                                      last standing candidate.
# No previous experiment separates those three.
#
# THE CONTROL ARM IS ALREADY FLOWN — it is the L4 screen's mpcof arm, the SAME cell as
# L1b's "C10 + no d-hat" — so this chain is 4 runs, not 6:
#   control  s31337002  err 1.21 / stable 100.0 / steady 0.64
#            s31337003  err 1.58 / stable 100.0 / steady 0.95
#
# SUCCESS (pre-registered, same bar as L1): steady drops below ~0.35 deg on BOTH seeds
# for at least one arm, i.e. clears the 0.57-0.87 floor band by more than the seed spread.
# REFUTATION: steady stays inside the band on both seeds for BOTH arms => the floor is not
# actuation granularity either, and attention moves to L4.
#
# READ STEADY AS THE PRIMARY. err is ~80% recovery term and recovery is already
# teacher-grade (0.88-1.21x per D1/D2). EXPECT BOTH ARMS TO COST err — they reduce
# actuation authority in the transient; that is the bargain being tested. Report the FULL
# TRIPLE (err/stable/steady) for every run and never declare a winner on one metric.
# WATCH stable ON ARM B: a leakier accumulator may fail to hold against the L4C sustained
# bias at all. A stability collapse there is INFORMATIVE (the integrator memory is
# load-bearing), not a failed run.
#
# Every other flag is COPIED from scripts/l1_dhat_chain.sh, including C10 weights and the
# 5-gen NEURONS cap, so the only difference from the control is the delta pair.
#
# NB: --delta-max did not exist before 07/08/2026. delta_max was a ControllerSpec field
# phased_ga never passed, so it was UNREACHABLE from a run rather than merely unsearched.
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
# shellcheck source=controller_arm_lib.sh
. "$ROOT/scripts/controller_arm_lib.sh"

LOG="/private/tmp/l3_delta_granularity_chain.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
OUTDIR="logs/controller/l3delta"
MARKDIR="experiments/l3delta_markers"
AIRFRAME="${L3_AIRFRAME:-cf21_brushless}"
DIST="${L3_DIST:-L4C}"
SEEDS="${L3_SEEDS:-31337002 31337003}"
# MPC-family cap, copied from the control arm. NOT a free choice.
NEURONS_GENS="${L3_NEURONS_GENS:-5}"
REPORT_SEEDS="99990101 99990102 99990103 99990104 99990105"
# Gate: wait for THIS chain process to exit before touching the box. A bare
# controllers() poll cannot tell an inter-run gap from "the chain finished" — that is how
# L2-v2 took the box in a 1 s window on 07/08. Unset = poll only.
WAIT_PID="${L3_WAIT_PID:-}"
WAIT_CEIL="${L3_WAIT_CEIL:-43200}"

# nf=15 pidmix. NO d-hat: L1b refuted it in 4/4 comparisons, so it stays off.
FEAT_PIDMIX="--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i"

log() { echo "[l3delta] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controllers() { ps -axo pid,command 2>/dev/null \
	| grep -E "wnn.control.phased_ga|e5_residual_proof" \
	| grep -v "/usr/bin/time" | grep -v grep | grep -c python; }

mkdir -p "$OUTDIR" "$MARKDIR"
log "########## ARMED — L3 delta granularity airframe=$AIRFRAME dist=$DIST seeds=[$SEEDS] gens=$NEURONS_GENS ##########"

if [ -n "$WAIT_PID" ]; then
	waited_pid=0
	while kill -0 "$WAIT_PID" 2>/dev/null; do
		[ $((waited_pid % 900)) -eq 0 ] && log "waiting for chain PID $WAIT_PID to exit (${waited_pid}s)"
		sleep 60
		waited_pid=$((waited_pid + 60))
		if [ "$waited_pid" -ge "$WAIT_CEIL" ]; then
			log "ABORT: chain PID $WAIT_PID still alive after ${waited_pid}s — refusing to contend."
			exit 3
		fi
	done
	log "chain PID $WAIT_PID has exited after ${waited_pid}s"
fi

waited=0
while [ "$(controllers)" -gt 0 ]; do
	log "waiting for the box: controllers=$(controllers) (${waited}s)"
	sleep 60
	waited=$((waited + 60))
	if [ "$waited" -ge "$WAIT_CEIL" ]; then
		log "ABORT: box still busy after ${waited}s — refusing to contend."
		exit 3
	fi
done
log "box clear: controllers=0"

# run_arm <seed> <arm:A|B>
run_arm() {
	local seed="$1" arm="$2"
	local variant dmax dleak
	if [ "$arm" = "A" ]; then
		variant="dstep"   ; dmax="0.025" ; dleak="0.95"
	else
		variant="dleak"   ; dmax="0.1"   ; dleak="0.80"
	fi

	run_controller_arm "L3D_${variant}_mpcof_${AIRFRAME}_${DIST}_s${seed}" \
		"$MARKDIR" "$OUTDIR" "$VP" log \
		"\"arm\":\"L3DELTA\",\"variant\":\"${variant}\",\"delta_max\":${dmax},\"delta_leak\":${dleak},\"teacher\":\"mpcof\",\"airframe\":\"${AIRFRAME}\",\"disturbance\":\"${DIST}\",\"obs_dhat\":false,\"mode\":\"BINARY\",\"state_neurons\":0,\"neurons_gens\":${NEURONS_GENS},\"seed\":${seed}" \
		-- \
		--levels 16 --skip-stages bits,connections --lamarckian \
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
		--airframe "$AIRFRAME" \
		--disturbance "$DIST" \
		--teacher mpcof \
		$FEAT_PIDMIX \
		--delta-max "$dmax" --delta-leak "$dleak" \
		--report-seeds $REPORT_SEEDS \
		--base-seed "$seed"
}

# INTERLEAVED: each seed flies BOTH arms before the next seed starts, so the first two
# runs already answer "did granularity move steady at all" and a dead arm can be culled
# before the second seed is spent.
for seed in $SEEDS; do
	log "===== ROUND seed=$seed (both delta arms) ====="
	for arm in A B; do
		run_arm "$seed" "$arm"
		log "seed=$seed arm=$arm finished rc=$?"
	done
done

log "########## L3 CHAIN DONE — markers in $MARKDIR ##########"
