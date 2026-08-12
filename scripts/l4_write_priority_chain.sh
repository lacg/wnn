#!/usr/bin/env bash
# L4 — is the hold floor a CREDIT-ASSIGNMENT limit in the write rule?
#
# WHY THIS RUN. L1, L1b, L2 and L3 are all flown and all refuted
# (docs/l4_teacher_screen_results.md section "Hold-floor levers"):
#   L1/L1b  disturbance as input / steady-ranked objective  -> steady WORSE or unmoved
#   L2      residual on the self-cancelling firmware cascade -> steady ~DOUBLED
#   L3      4x finer actuation granularity by BOTH routes    -> steady WORSE or unmoved
# More input, a better substrate, a hold-targeted objective and finer actuation are all
# ruled out. The remaining structure is HOW THE TEACHER'S DEMONSTRATIONS BECOME RUNS.
#
# THE MECHANISM UNDER TEST. The deployed students are single-layer (sn=0) BINARY:
# training is the direct write in controller.rs section (d) — visited address -> teacher
# target bit — and BINARY is LAST-WRITER-WINS. The backward BPTT walk runs d = n_rec-1
# down to 0, so the EARLIEST record in every window commits last and owns any contested
# cell. Which record survives a collision is therefore decided by its POSITION in the
# window — arbitrary w.r.t. how far the drone actually was from level. The near-hover
# mass (most of every rollout under L4C) freely overwrites the rare large-error
# corrections. Hypothesis: THAT mis-assignment is the hold floor.
#
# Evidence pointing here: L2's DAgger trace degrades as beta anneals (best iters 4-5,
# scored iter 8 worse) — a credit-assignment signature, not an authority one.
#
# THE DESIGN'S POINT IS THE PAIRING (same shape as L3). Two default-off flags reach the
# same "corrections beat hover mass" end by opposite means and pay OPPOSITE prices:
#
#   arm        flag                     mechanism                        price
#   control    (neither)                earliest record wins, arbitrary  -- (already flown)
#   A (order)  --write-priority-err     commits ascend by |err|: the     none in coverage —
#                                       HIGHEST-error record writes      every record still
#                                       last and owns contested cells    trains
#   B (floor)  --write-err-floor 0.5    records under 0.5 deg |err|      hover coverage —
#                                       never commit at all              settled states go
#                                                                        untrained
#
#   BOTH improve steady   => magnitude-blind credit assignment IS the floor. The finding.
#   ONLY A improves       => ordering (who wins collisions) is the mechanism; coverage
#                            is fine.
#   ONLY B improves       => the hover MASS itself is the problem (dilution, not
#                            collisions).
#   NEITHER improves      => the write rule is ruled out too, and the programme's four
#                            levers are exhausted: the floor sits deeper than input,
#                            substrate, objective, actuation, or credit assignment.
#
# CONTROL IS ALREADY FLOWN (the L4 screen's mpcof arm = L1b's "C10 + no d-hat" cell):
#   control  s31337002  err 1.21 / stable 100.0 / steady 0.64
#            s31337003  err 1.58 / stable 100.0 / steady 0.95
#
# SUCCESS (pre-registered, same bar as L1/L3): steady below ~0.35 deg on BOTH seeds for
# at least one arm. REFUTATION: steady stays in the 0.57-0.87 band on both seeds for
# both arms. n=1 seed ranks nothing.
#
# READ STEADY AS THE PRIMARY; report the FULL TRIPLE always. WATCH stable ON ARM B: a
# 0.5 deg floor removes the settled-hover demonstrations entirely — if stable collapses,
# that is INFORMATIVE (hover coverage is load-bearing), not a failed run. Same reading
# discipline as L3's arm-B note.
#
# NEW THIS CHAIN: --save-stage-checkpoints per run (async dump after each stage) — the
# GRID/NEURONS/MEMORY winners all survive, instead of only MEMORY via --save-winner.
# Feeds the identical-draw stage comparison AND the teacher-committee member pool
# (.claude/plans/grid-holdout-instrumentation.md Change 2). File-write only; no effect
# on the search.
#
# Every other flag is COPIED from scripts/l3_delta_granularity_chain.sh's control shape
# (which copied l1_dhat_chain.sh), including C10 weights, the 5-gen NEURONS cap and the
# DEFAULT delta pair (0.1/0.95) — the only difference from the control is the write rule.
#
# NB: --write-priority-err / --write-err-floor did not exist before 07/08/2026
# (ram_controller ABI 22 wheel, installed 07/08 ~21:15 EDT). The wheel's flags are
# default-off and parity-gated (sn0_l4_flags_off_bit_identical), so the control cell
# remains comparable.
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
# shellcheck source=controller_arm_lib.sh
. "$ROOT/scripts/controller_arm_lib.sh"

LOG="/private/tmp/l4_write_priority_chain.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
OUTDIR="logs/controller/l4writes"
MARKDIR="experiments/l4writes_markers"
AIRFRAME="${L4_AIRFRAME:-cf21_brushless}"
DIST="${L4_DIST:-L4C}"
SEEDS="${L4_SEEDS:-31337002 31337003}"
# MPC-family cap, copied from the control arm. NOT a free choice.
NEURONS_GENS="${L4_NEURONS_GENS:-5}"
REPORT_SEEDS="99990101 99990102 99990103 99990104 99990105"
# Arm B floor (deg). 0.5 sits under the 0.57-0.87 steady floor band, so it cuts
# only the settled-hover mass while keeping the whole correction range trainable.
ERR_FLOOR="${L4_ERR_FLOOR:-0.5}"
# Gate: wait for a chain PID to exit before touching the box (the L2-v2 lesson —
# a bare controllers() poll cannot tell an inter-run gap from "chain finished").
# Unset = poll only, correct when the box is already idle at arm time.
WAIT_PID="${L4_WAIT_PID:-}"
WAIT_CEIL="${L4_WAIT_CEIL:-43200}"

# nf=15 pidmix. NO d-hat (L1b refuted it 4/4); DEFAULT delta pair (L3 refuted both
# granularity arms, so the control actuation is the right baseline).
FEAT_PIDMIX="--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i"

log() { echo "[l4writes] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controllers() { ps -axo pid,command 2>/dev/null \
	| grep -E "wnn.control.phased_ga|e5_residual_proof" \
	| grep -v "/usr/bin/time" | grep -v grep | grep -c python; }

mkdir -p "$OUTDIR" "$MARKDIR"
log "########## ARMED — L4 write-rule airframe=$AIRFRAME dist=$DIST seeds=[$SEEDS] gens=$NEURONS_GENS floor=${ERR_FLOOR}deg ##########"

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
	local variant armflags floor_json
	if [ "$arm" = "A" ]; then
		variant="worder" ; armflags="--write-priority-err" ; floor_json="0.0"
	else
		variant="wfloor" ; armflags="--write-err-floor ${ERR_FLOOR}" ; floor_json="${ERR_FLOOR}"
	fi
	local tag="L4W_${variant}_mpcof_${AIRFRAME}_${DIST}_s${seed}"

	run_controller_arm "$tag" \
		"$MARKDIR" "$OUTDIR" "$VP" log \
		"\"arm\":\"L4WRITES\",\"variant\":\"${variant}\",\"write_priority_err\":$([ "$arm" = "A" ] && echo true || echo false),\"write_err_floor_deg\":${floor_json},\"teacher\":\"mpcof\",\"airframe\":\"${AIRFRAME}\",\"disturbance\":\"${DIST}\",\"obs_dhat\":false,\"mode\":\"BINARY\",\"state_neurons\":0,\"neurons_gens\":${NEURONS_GENS},\"seed\":${seed}" \
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
		$armflags \
		--save-stage-checkpoints "$OUTDIR/${tag}_stages" \
		--report-seeds $REPORT_SEEDS \
		--base-seed "$seed"
}

# INTERLEAVED: each seed flies BOTH arms before the next seed starts, so the first two
# runs already answer "did the write rule move steady at all" and a dead arm can be
# culled before the second seed is spent.
for seed in $SEEDS; do
	log "===== ROUND seed=$seed (both write-rule arms) ====="
	for arm in A B; do
		run_arm "$seed" "$arm"
		log "seed=$seed arm=$arm finished rc=$?"
	done
done

log "########## L4 CHAIN DONE — markers in $MARKDIR ##########"
