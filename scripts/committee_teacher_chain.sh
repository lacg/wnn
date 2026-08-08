#!/usr/bin/env bash
# COMMITTEE — one student per teacher, on the CURRENT recipe, then vote them.
#
# WHY THIS RUN. The 11/07 ensemble result (lqr+mpc PWM committee 90.5±2.9% stable /
# 3.23 deg / steady 2.85 deg, +7.5pp over best solo at HALF the sigma) proved the
# committee MECHANISM on the July recipe — but at an operating point that today's solo
# control (1.21 deg / 100% / steady 0.64) has already passed outright. The +7.5pp was
# won from stability headroom that no longer exists; whether a committee buys anything
# at 100% stable — where the ONLY currency left is steady — has never been tested.
# Members must therefore be re-flown on the current recipe; the July winners are not
# comparable (old scorer protocol, old feature set, and lqi/mpcof were silently
# training as PID before the 05/08 TeacherBank fix — no valid history exists for them).
#
# THE DESIGN. 5 teachers x 2 base seeds = 10 runs, IDENTICAL recipe except --teacher
# (the L3/L1b control shape: C10 weights, 5-gen NEURONS cap, sn=0 BINARY, default
# delta pair, no d-hat, no L4 write flags — the committee question is orthogonal to
# the write-rule question L4 is testing, so the two must not be mixed in one cohort).
# Every run dumps per-stage checkpoints, so each contributes its GRID, NEURONS and
# MEMORY winners: 30 candidate members, not 10 — the "winners from either grid,
# neurons and memory from each teacher" pool.
#
# Teacher diversity is the committee's whole bet: LQR/MPC (memoryless optimal),
# LQI/MPCOF (stateful, bias-rejecting), PID (integral-laden hand tuning) imprint
# DIFFERENT error->PWM mappings; a per-motor PWM vote can cancel their uncorrelated
# mistakes. The July trio lesson stands: a weak member DRAGS the vote (adding pid
# hurt), so scoring must rank pairs before larger sets.
#
# CONTROL (already flown, = the mpcof cell): s31337002 1.21/100.0/0.64 ;
# s31337003 1.58/100.0/0.95. The mpcof run here is a REPRODUCTION of that cell (same
# flags, fresh run) — it validates comparability AND supplies mpcof's stage winners,
# which --save-winner alone never kept.
#
# SUCCESS (pre-registered): a committee (pair or larger, scored by
# scripts/ensemble_teachers.py on the 5 report seeds) beats the best SOLO member's
# steady on BOTH base seeds without losing stable. The solo table is a finding in
# itself either way: it is the first teacher screen on the current recipe since the
# TeacherBank fix, at n=2 seeds.
# REFUTATION: no committee beats its best member => vote diversity does not buy hold
# precision at this operating point.
#
# SCORING (after all 10 markers): scripts/ensemble_teachers.py per base seed with the
# 5 MEMORY winners (solos + --pairs); stage-winner (GRID/NEURONS) members and trios
# are follow-up analysis from the same checkpoint pool. Scoring is minutes, not hours.
#
# COST. 10 runs x ~2.5h ≈ 25h, sequential, gated behind the L4 chain (~10h). One
# controller at a time throughout.
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
# shellcheck source=controller_arm_lib.sh
. "$ROOT/scripts/controller_arm_lib.sh"

LOG="/private/tmp/committee_teacher_chain.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
OUTDIR="logs/controller/committee"
MARKDIR="experiments/committee_markers"
AIRFRAME="${CMT_AIRFRAME:-cf21_brushless}"
DIST="${CMT_DIST:-L4C}"
SEEDS="${CMT_SEEDS:-31337002 31337003}"
TEACHERS="${CMT_TEACHERS:-mpcof lqi lqr mpc pid}"
# Budget-matched across ALL teachers (the L1/L3 cap) — committee members must be
# comparable to each other and to the control cell.
NEURONS_GENS="${CMT_NEURONS_GENS:-5}"
REPORT_SEEDS="99990101 99990102 99990103 99990104 99990105"
# Gate on the L4 chain PID (armed 07/08 ~21:19 EDT). The L2-v2 lesson: a bare
# controllers() poll cannot tell an inter-run gap from "chain finished".
WAIT_PID="${CMT_WAIT_PID:-}"
WAIT_CEIL="${CMT_WAIT_CEIL:-86400}"

# nf=15 pidmix, no d-hat, default delta pair, NO L4 write flags — control shape.
FEAT_PIDMIX="--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i"

log() { echo "[committee] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controllers() { ps -axo pid,command 2>/dev/null \
	| grep -E "wnn.control.phased_ga|e5_residual_proof" \
	| grep -v "/usr/bin/time" | grep -v grep | grep -c python; }

mkdir -p "$OUTDIR" "$MARKDIR"
log "########## ARMED — COMMITTEE teachers=[$TEACHERS] airframe=$AIRFRAME dist=$DIST seeds=[$SEEDS] gens=$NEURONS_GENS ##########"

if [ -n "$WAIT_PID" ]; then
	waited_pid=0
	while kill -0 "$WAIT_PID" 2>/dev/null; do
		[ $((waited_pid % 1800)) -eq 0 ] && log "waiting for chain PID $WAIT_PID to exit (${waited_pid}s)"
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

# run_member <seed> <teacher>
run_member() {
	local seed="$1" teacher="$2"
	local tag="CMT_${teacher}_${AIRFRAME}_${DIST}_s${seed}"

	run_controller_arm "$tag" \
		"$MARKDIR" "$OUTDIR" "$VP" log \
		"\"arm\":\"COMMITTEE\",\"teacher\":\"${teacher}\",\"airframe\":\"${AIRFRAME}\",\"disturbance\":\"${DIST}\",\"obs_dhat\":false,\"mode\":\"BINARY\",\"state_neurons\":0,\"neurons_gens\":${NEURONS_GENS},\"seed\":${seed}" \
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
		--teacher "$teacher" \
		$FEAT_PIDMIX \
		--save-stage-checkpoints "$OUTDIR/${tag}_stages" \
		--report-seeds $REPORT_SEEDS \
		--base-seed "$seed"
}

# INTERLEAVED by seed: each base seed flies ALL 5 teachers before the next seed
# starts, so the first round already yields a complete 5-member committee to score
# and a weak teacher shows itself before the second seed is spent on it.
for seed in $SEEDS; do
	log "===== ROUND seed=$seed (all teachers) ====="
	for teacher in $TEACHERS; do
		run_member "$seed" "$teacher"
		log "seed=$seed teacher=$teacher finished rc=$?"
	done
done

log "########## COMMITTEE CHAIN DONE — markers in $MARKDIR ##########"
log "NEXT: score with ensemble_teachers.py per base seed (5 MEMORY winners, solos + --pairs, --steps 2000 --episodes 100 --seeds ${REPORT_SEEDS// /,})"
