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
# WHAT IT SCORES. Per base seed: the 5 MEMORY-stage winners (one per teacher), SOLO
# and as PWM committees, with --pairs so every 2-member vote is ranked before larger
# sets. MEMORY for all five is the PRE-REGISTERED, stage-uniform choice: picking each
# teacher's val-selected headline stage instead would mix stages across members (mpc
# headlines at NEURONS, the other four at MEMORY) and make the members incomparable.
# It is also not a choice that flatters the design — mpc's MEMORY winner is BETTER on
# report-seed steady than its NEURONS headline (0.78 vs 0.91), so uniform-MEMORY is
# if anything generous to the stage it did not select.
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

WAIT_PID="${CMT_SCORE_WAIT_PID:-}"
WAIT_CEIL="${CMT_SCORE_WAIT_CEIL:-172800}"

log() { echo "[cmt-score] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controllers() { ps -axo pid,command 2>/dev/null \
	| grep -E "wnn.control.phased_ga|e5_residual_proof" \
	| grep -v "/usr/bin/time" | grep -v grep | grep -c python; }

# Path to one teacher's MEMORY-stage winner for a base seed ("" if absent).
member_ckpt() {
	local seed="$1" teacher="$2"
	local p="$CKROOT/CMT_${teacher}_${AIRFRAME}_${DIST}_s${seed}_stages/stage4_memory.yaml.gz"
	[ -f "$p" ] && echo "$p"
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

	local args="" t p
	for t in $TEACHERS; do
		p="$(member_ckpt "$seed" "$t")"
		[ -n "$p" ] && args="$args ${t}=${p}"
	done
	local n
	n="$(members_present "$seed")"

	log "===== SCORE s$seed (${n} members: $TEACHERS) ====="
	local t0=$SECONDS
	PYTHONPATH=src/wnn "$VP" scripts/ensemble_teachers.py \
		--winners $args \
		--pairs --agg "$AGG" \
		--steps "$STEPS" --episodes "$EPISODES" \
		--seeds "$REPORT_SEEDS" > "$out" 2>&1
	local rc=$? dur=$((SECONDS - t0))

	printf '{"tag":"CMTSCORE_s%s","seed":%s,"members":%s,"teachers":"%s","agg":"%s","steps":%s,"episodes":%s,"report_seeds":"%s","rc":%s,"dur_s":%s,"out":"%s","done":true}\n' \
		"$seed" "$seed" "$n" "$TEACHERS" "$AGG" "$STEPS" "$EPISODES" \
		"$REPORT_SEEDS" "$rc" "$dur" "$out" > "$mark"
	log "s$seed: finished rc=$rc dur=${dur}s members=$n -> $mark"
}

mkdir -p "$OUTDIR" "$SCOREMARK"
log "########## ARMED — COMMITTEE SCORING seeds=[$SEEDS] teachers=[$TEACHERS] wait_pid=${WAIT_PID:-none} ##########"

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
