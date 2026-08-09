#!/usr/bin/env bash
# COMMITTEE SCORING — the pass that answers the pre-registered question.
#
# WHY THIS EXISTS AS A CHAIN, NOT A COMMAND. The 10 committee members take ~25h to
# fly; the scoring itself is minutes. Running it by hand means someone has to be
# awake at the moment the last marker lands. This arms behind the committee chain's
# PID (the same gate pattern committee_teacher_chain.sh uses on the L4 chain — a bare
# controllers() poll cannot tell an inter-run gap from "chain finished", the L2-v2
# lesson) and fires the scoring the moment the box goes idle.
#
# WHAT IT SCORES. Per base seed: each teacher's HEADLINE winner -- the stage chosen by
# the union ranking over 5 val seeds (seeds.val + 0..4, disjoint from the report seeds)
# -- SOLO and as PWM committees, with --pairs so every 2-member vote is ranked before
# larger sets.
#
# HEADLINE, NOT MEMORY-ONLY (corrected 08/08/2026). The chain's own comment block
# pre-registered "the 5 MEMORY winners", but that text predates the stage-selection
# work: every run now publishes GRID/NEURONS/MEMORY and headlines the val-selected one,
# so the headline IS what the run ships. Assembling the committee from MEMORY-only
# would be a committee of something we do not ship, picked by a hand-chosen stage rule
# -- exactly what the union ranking replaced. It differs in real members: mpc s31337002
# headlines at NEURONS while the other eight markers headline at MEMORY.
#
# The earlier defence of MEMORY-only ("mpc's MEMORY winner is better on report-seed
# steady than its NEURONS headline, 0.78 vs 0.91") was WRONG IN KIND and is retracted:
# it settles a member-selection question with the very partition the pass publishes.
# Selection must ride on val seeds or an a-priori rule. The union ranking is the former;
# both are unbiased w.r.t. the report seeds, and only the headline is the run's product.
#
# A MEMORY-only pass remains available afterwards as a DISCLOSED sensitivity check
# (CMT_MEMBER_STAGE=memory). It is deliberately not run automatically -- scoring both
# and reporting the better one is best-of-N inflation wearing a different hat.
#
# PID STAYS IN. On seed 31337002 pid is decisively the worst member (94.0% stable /
# 2.56 deg err / 2.16 deg steady vs lqi's 99.8 / 1.11 / 0.53) and the July trio lesson
# says a weak member DRAGS the vote. Dropping it now would be selection on the very
# data the pass reports. --pairs is the honest instrument: it shows what pid costs
# instead of assuming it.
#
# NO GRID MEMBERS. --save-stage-checkpoints wrote stage1_neurons and stage4_memory
# only; there is no GRID checkpoint to score. NEURONS members remain available for
# follow-up from the same pool.
#
# PRE-REGISTERED BAR. A committee must beat its best SOLO member's steady on BOTH
# base seeds without losing stable. REFUTATION: no committee beats its best member =>
# vote diversity does not buy hold precision at this operating point. July's
# lqr+mpc (90.5% / 3.23 deg / 2.85 deg) does NOT transfer -- it beat solos at an
# operating point today's solo control already passes.
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1

LOG="/private/tmp/committee_scoring_chain.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
MARKDIR="experiments/committee_markers"
CKROOT="logs/controller/committee"
OUTDIR="logs/controller/committee_scoring"
SCOREMARK="experiments/committee_score_markers"

AIRFRAME="${CMT_AIRFRAME:-cf21_brushless}"
DIST="${CMT_DIST:-L4C}"
SEEDS="${CMT_SEEDS:-31337002 31337003}"
TEACHERS="${CMT_TEACHERS:-mpcof lqi lqr mpc pid}"
# The report seeds the members were held out on -- same 5, so solo numbers here are
# directly comparable to the per-run MEMORY MULTI-SEED blocks in the markers.
REPORT_SEEDS="${CMT_REPORT_SEEDS:-99990101,99990102,99990103,99990104,99990105}"
# --steps 2000 --episodes 100 = the member runs' own protocol. Do not shorten:
# a committee scored at a different episode length is not comparable to its members.
STEPS="${CMT_STEPS:-2000}"
EPISODES="${CMT_EPISODES:-100}"
AGG="${CMT_AGG:-both}"
# Exhaustive committee sizes to sweep. 3,4 costs ~20 min and answers "is a trio
# enough?" in the same pass — measured 09/08: k=3 matches or beats k=4/5, and
# adding pid to any trio loses 8/8 paired comparisons.
COMBO_SIZES="${CMT_COMBO_SIZES:-3,4}"

WAIT_PID="${CMT_SCORE_WAIT_PID:-}"
WAIT_CEIL="${CMT_SCORE_WAIT_CEIL:-172800}"

log() { echo "[cmt-score] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controllers() { ps -axo pid,command 2>/dev/null \
	| grep -E "wnn.control.phased_ga|e5_residual_proof" \
	| grep -v "/usr/bin/time" | grep -v grep | grep -c python; }

# Which stage each member contributes: "headline" (default, = the run's product) or a
# forced stage name for a disclosed sensitivity pass (grid|neurons|memory).
MEMBER_STAGE="${CMT_MEMBER_STAGE:-headline}"

# Stage this teacher/seed's member comes from. For "headline", read it out of the
# marker's headline_stage line ("... HEADLINE stage=MEMORY (union rank ...)") -- the
# marker is the only record of what the val ranking chose, so it is the authority.
member_stage() {
	local seed="$1" teacher="$2"
	if [ "$MEMBER_STAGE" != "headline" ]; then
		echo "$MEMBER_STAGE"; return
	fi
	local mk="$MARKDIR/CMT_${teacher}_${AIRFRAME}_${DIST}_s${seed}.json"
	[ -f "$mk" ] || return
	sed -n 's/.*HEADLINE stage=\([A-Z]*\).*/\1/p' "$mk" | head -1 \
		| tr '[:upper:]' '[:lower:]'
}

# Which genome index within the headline stage's population the run published.
# Since the top-3 stage selection (d3f64e03) the marker's headline line may name a
# RUNNER-UP, e.g. "HEADLINE stage=MEMORY genome=MEMORY#2 (...)" — the published
# triple belongs to final_population[2], and scoring pop[0] would re-create the
# wrong-genome bug one level up. Markers from before d3f64e03 have no "genome="
# and default to 0 (their published row IS pop[0]).
member_index() {
	local seed="$1" teacher="$2"
	local mk="$MARKDIR/CMT_${teacher}_${AIRFRAME}_${DIST}_s${seed}.json"
	[ -f "$mk" ] || { echo 0; return; }
	local i
	i="$(sed -n 's/.*HEADLINE stage=[A-Z]* genome=[A-Z]*#\([0-9]*\).*/\1/p' "$mk" | head -1)"
	echo "${i:-0}"
}

# Path to this teacher's member checkpoint for a base seed ("" if absent), with the
# published genome index appended as "#N" when it is not 0 (ensemble_teachers.py
# understands `path#N` since 09/08/2026). GRID is stage0 (only written since the
# 08/08/2026 fix), NEURONS stage1, MEMORY stage4 -- a headline naming a stage whose
# checkpoint predates that fix yields "", which the strict gate turns into a logged
# skip rather than a silent substitution.
member_ckpt() {
	local seed="$1" teacher="$2" st file idx
	st="$(member_stage "$seed" "$teacher")"
	case "$st" in
		grid)    file="stage0_grid.yaml.gz" ;;
		neurons) file="stage1_neurons.yaml.gz" ;;
		memory)  file="stage4_memory.yaml.gz" ;;
		*)       return ;;
	esac
	local p="$CKROOT/CMT_${teacher}_${AIRFRAME}_${DIST}_s${seed}_stages/$file"
	[ -f "$p" ] || return
	idx="$(member_index "$seed" "$teacher")"
	if [ "$idx" != "0" ]; then p="${p}#${idx}"; fi
	echo "$p"
}

# How many of this seed's 5 members have a MEMORY winner on disk.
members_present() {
	local seed="$1" n=0 t
	for t in $TEACHERS; do
		[ -n "$(member_ckpt "$seed" "$t")" ] && n=$((n + 1))
	done
	echo "$n"
}

# READINESS GATE: STRICT 5 (decided 08/08/2026).
#
# A seed is scored only when all five members are on disk. The alternatives were
# considered and rejected: scoring a partial seed always yields numbers, but a
# 4-member vote is not the same object as a 5-member one, and comparing seed 002's
# 5-way against seed 003's 4-way would breach the pre-registered "beats its best solo
# on BOTH seeds" bar without anyone noticing it had happened. Unattended degradation
# is exactly the failure this bar exists to prevent.
#
# The cost is accepted and bounded: one crashed member forfeits that seed's
# UNATTENDED scoring, not the work -- the other members' checkpoints survive, the
# skip is logged with its member count, and the seed can be re-flown or scored by
# hand. score_seed() is idempotent on its marker, so re-running the chain after a
# repair picks up exactly the seed that was skipped.
seed_is_ready() {
	local seed="$1"
	[ "$(members_present "$seed")" -eq 5 ] && echo "yes"
}

score_seed() {
	local seed="$1"
	local mark="$SCOREMARK/CMTSCORE_s${seed}.json"
	local out="$OUTDIR/CMTSCORE_s${seed}.out"

	if [ -f "$mark" ]; then
		log "s$seed: score marker exists — skip"
		return 0
	fi

	local args="" provenance="" t p st
	for t in $TEACHERS; do
		p="$(member_ckpt "$seed" "$t")"
		st="$(member_stage "$seed" "$t")"
		[ -n "$p" ] && args="$args ${t}=${p}"
		provenance="$provenance ${t}:${st:-MISSING}"
	done
	local n
	n="$(members_present "$seed")"

	# The per-member stage is part of the result, not a detail: a reader must be able
	# to see that (say) mpc contributed its NEURONS winner while the rest contributed
	# MEMORY, without re-deriving it from the markers.
	log "===== SCORE s$seed (${n} members, stage=$MEMBER_STAGE):$provenance ====="
	local t0=$SECONDS
	# --train-seed / --disturbance / --airframe are REQUIRED for a valid score:
	# thresholds must be fitted on the seed the cells were written under, and the
	# plant + conditions must match the members' own held-out. The 08/08 run
	# omitted all three and produced solos 20-40x worse than their own runs.
	PYTHONPATH=src/wnn "$VP" scripts/ensemble_teachers.py \
		--winners $args \
		--pairs --combo-sizes "$COMBO_SIZES" --agg "$AGG" \
		--steps "$STEPS" --episodes "$EPISODES" \
		--seeds "$REPORT_SEEDS" \
		--base-seed "$seed" --disturbance "$DIST" --airframe "$AIRFRAME" > "$out" 2>&1
	local rc=$? dur=$((SECONDS - t0))

	printf '{"tag":"CMTSCORE_s%s","seed":%s,"members":%s,"teachers":"%s","member_stage":"%s","member_provenance":"%s","agg":"%s","steps":%s,"episodes":%s,"report_seeds":"%s","rc":%s,"dur_s":%s,"out":"%s","done":true}\n' \
		"$seed" "$seed" "$n" "$TEACHERS" "$MEMBER_STAGE" "${provenance# }" "$AGG" \
		"$STEPS" "$EPISODES" "$REPORT_SEEDS" "$rc" "$dur" "$out" > "$mark"
	log "s$seed: finished rc=$rc dur=${dur}s members=$n -> $mark"
}

mkdir -p "$OUTDIR" "$SCOREMARK"
log "########## ARMED — COMMITTEE SCORING seeds=[$SEEDS] teachers=[$TEACHERS] member_stage=$MEMBER_STAGE combo_sizes=$COMBO_SIZES wait_pid=${WAIT_PID:-none} ##########"

if [ -n "$WAIT_PID" ]; then
	waited_pid=0
	while kill -0 "$WAIT_PID" 2>/dev/null; do
		[ $((waited_pid % 1800)) -eq 0 ] && log "waiting for committee chain PID $WAIT_PID to exit (${waited_pid}s)"
		sleep 60
		waited_pid=$((waited_pid + 60))
		if [ "$waited_pid" -ge "$WAIT_CEIL" ]; then
			log "ABORT: chain PID $WAIT_PID still alive after ${waited_pid}s — refusing to contend."
			exit 3
		fi
	done
	log "committee chain PID $WAIT_PID has exited after ${waited_pid}s"
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

for seed in $SEEDS; do
	if [ "$(seed_is_ready "$seed")" = "yes" ]; then
		score_seed "$seed"
	else
		log "s$seed: NOT ready (members_present=$(members_present "$seed")) — skipped"
	fi
done

log "########## COMMITTEE SCORING DONE — markers in $SCOREMARK ##########"
log "NEXT: read $OUTDIR/CMTSCORE_s*.out — for each seed, does ANY committee beat its best SOLO member's steady WITHOUT losing stable? Bar requires BOTH seeds."
