#!/usr/bin/env bash
# L4 TEACHER SCREEN — which classical teacher makes the best WNN student, on a
# SOURCED plant under a SOURCED disturbance ladder?
#
# WHY THIS RUN. Two independent findings forced it. (1) The teacher ranking is not
# portable: LQI won at L3D, MPCOF was WORST there and is BEST at L4C on the
# Crazyflie — so a teacher chosen on one regime says nothing about another, and the
# lqr->lqi swap we measured earlier came back a null for exactly that reason.
# (2) The plant and the disturbances are both new (Crazyflie 2.1 Brushless from
# Bitcraze firmware; L4 from Molchanov/RotorS/datasheets), so nothing measured
# before transfers. See docs/disturbance_param_sources.md.
#
# The classical baselines on this plant (5-seed held-out, already measured, in
# experiments/l4teach_markers/baselines_L4C_cf21bl.json — recomputed 05/08 on the
# FIXED moment arm; the old dfa1l_markers copy was deleted with the pre-fix numbers):
#   MPCOF 100.0% / 0.69 deg  |  LQI 100.0% / 0.81  |  LQR 100.0% / 0.93
#   MPC   100.0% / 1.04      |  PID 100.0% / 1.64
# Every classical controller holds 100% stable across the whole ladder, so THE
# METRIC HERE IS ERROR, not stability. A student that reaches ~1 deg is competitive;
# stability alone will not separate anything.
#
# This screen doubles as the COMMITTEE POOL: five students trained from five
# structurally different teachers have decorrelated failure modes in a way five
# seeds of one teacher do not (the lqr+mpc ensemble already measured 90.5%).
#
# INTERLEAVED BY SEED (the sweep rule): round r fires ALL FIVE teachers at seed r.
# Losing the box partway then leaves a complete low-n version of the whole contrast
# rather than every seed of one teacher and nothing of the others.
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
# shellcheck source=controller_arm_lib.sh
. "$ROOT/scripts/controller_arm_lib.sh"

LOG="/private/tmp/l4_teacher_chain.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
OUTDIR="logs/controller/l4teach"
MARKDIR="experiments/l4teach_markers"
AIRFRAME="${L4_AIRFRAME:-cf21_brushless}"
DIST="${L4_DIST:-L4C}"
TEACHERS="${L4_TEACHERS:-mpcof lqi lqr mpc pid}"
SEEDS="${L4_SEEDS:-31337002 31337003}"
# NEURONS generation budget. Default 60 = run to natural early-stop, which is
# what the closed-form arm did (it settled at 8-14 gens). Override to cap an
# arm whose per-generation cost makes the full budget impractical — the MPC
# family measures ~20 min/gen against the closed-form ~4 min/gen.
#
# A CAPPED ARM IS NOT BUDGET-MATCHED to the closed-form results and the
# comparison is therefore ASYMMETRIC: if a capped MPC student BEATS the closed-form
# arm's mean student error that is conclusive — better teacher, less search. If it
# loses, the result is ambiguous between teacher quality and search budget and must be
# reported that way, not as "MPC is a worse teacher".
# (The old "1.11 deg" figure cited here was measured on the pre-05/08 broken moment arm
# and was deleted with the rest of that screen — compare against the CLOSED-FORM ARM OF
# THIS RERUN, never against a remembered number.)
NEURONS_GENS="${L4_NEURONS_GENS:-60}"
REPORT_SEEDS="99990101 99990102 99990103 99990104 99990105"

# nf=15 pidmix — the A2 probe winner, and the only feature set with a matched
# head-to-head win (beat pidmix_tilt on err at all three paired seeds).
FEAT_PIDMIX="--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i"

log() { echo "[l4teach] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controllers() { ps -axo pid,command 2>/dev/null | grep "wnn.control.phased_ga" \
	| grep -v "/usr/bin/time" | grep -v grep | grep -c python; }

mkdir -p "$OUTDIR" "$MARKDIR"
log "########## ARMED — airframe=$AIRFRAME dist=$DIST teachers=[$TEACHERS] seeds=[$SEEDS] ##########"

# Never run alongside another controller: abort rather than contend.
if [ "$(controllers)" -gt 0 ]; then
	log "ABORT: controllers=$(controllers) already running — refusing to contend."
	exit 3
fi
log "box clear: controllers=0"

run_teacher() {
	local teacher="$1" seed="$2"
	# Teacher AND airframe AND disturbance are all in the tag. A marker that
	# recorded them only internally would still be skipped by name on a re-run
	# under a different combination, silently returning one configuration's cell
	# as another's result.
	run_controller_arm "L4T_${teacher}_${AIRFRAME}_${DIST}_s${seed}" \
		"$MARKDIR" "$OUTDIR" "$VP" log \
		"\"arm\":\"L4TEACH\",\"teacher\":\"${teacher}\",\"airframe\":\"${AIRFRAME}\",\"disturbance\":\"${DIST}\",\"features\":\"pidmix\",\"mode\":\"BINARY\",\"state_neurons\":0,\"neurons_gens\":${NEURONS_GENS},\"seed\":${seed}" \
		-- \
		--levels 16 --skip-stages bits,connections --lamarckian \
		`# --max-cells-strict (05/08/2026): without it the budget is a THRESHOLD,` \
		`# not a ceiling — grows are suppressed only once ALREADY at budget, so a` \
		`# genome just under the line takes a bits-grow that replicates its layer` \
		`# x2^delta and lands far above. Measured on the lqr/lqi runs of this very` \
		`# screen: mu237k held for five MEMORY generations against a 180k budget,` \
		`# max-in-population 468k (the weak-teacher runs never exceeded mu59k, so it` \
		`# bound precisely on the arms searching hardest). strict clamps the grow to` \
		`# the largest delta that still fits POST-grow.` \
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
		--report-seeds $REPORT_SEEDS \
		--base-seed "$seed"
}

for seed in $SEEDS; do
	log "===== ROUND seed=$seed ====="
	for t in $TEACHERS; do
		# Abort the whole chain if something else grabbed the box between runs —
		# two controllers on one machine is the documented way to lose both.
		if [ "$(controllers)" -gt 0 ]; then
			log "ABORT mid-round: a controller appeared (controllers=$(controllers))"
			exit 4
		fi
		run_teacher "$t" "$seed"
	done
done

got=0
for seed in $SEEDS; do
	for t in $TEACHERS; do
		[ -f "${MARKDIR}/L4T_${t}_${AIRFRAME}_${DIST}_s${seed}.json" ] && got=$((got+1))
	done
done
want=$(( $(echo $SEEDS | wc -w) * $(echo $TEACHERS | wc -w) ))
if [ "$got" = "$want" ]; then
	: > "${MARKDIR}/L4TEACH_ALL_DONE.marker"
	log "########## L4 TEACHER SCREEN DONE — ${got}/${want} markers ##########"
else
	log "########## L4 TEACHER SCREEN INCOMPLETE — ${got}/${want} markers ##########"
	exit 5
fi
