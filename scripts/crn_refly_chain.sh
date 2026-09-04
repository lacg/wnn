#!/usr/bin/env bash
# CRN RE-FLY — the clean CRN-vs-rotation read (04/09/2026, Luiz: "queue the b24
# s31337002 CRN re-fly after the A/B").
#
# THE QUESTION. CRN is the FIX (Luiz 04/09: "we are not going back") — this is
# not an A/B, it is the paired MEASUREMENT of what the fix changed. The CRN
# fitness landed mid bits-round-2 (03/09 21:05 EDT), so every CRN-vs-rotation
# pair banked so far ALSO differs by seed (b24: 0.1972 at s31337002 rotation vs
# 0.1271 at s31337003 CRN). This flies the SAME shapes at the SAME seed with only
# the scorer changed: b in {24, 32} at n=256 gamma=1 seed 31337002 (the width
# curve's narrow end and its winner), tagged ..._s31337002_crn so the
# rotation-era markers are untouched. Sequential in ONE ladder instance.
#
# HOW. Via the ladder's own script (scripts/sweep_ladder_gamma.sh) with the
# SL_TAG_SUFFIX hook — the recipe is never copied. Byte-identical to the banked
# run except that phased_ga now defaults to --score-crn.
#
# GATING. After the translation A/B banks all 10 of its markers (any shape).
# Pure wait, FAIL CLOSED if the A/B chain died short (same posture as the rest
# of the queue). The leak revisit chain gates on THIS chain's marker, so the
# order is A/B -> this -> leak with no race for the box.
set -u
ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1

LOG="/private/tmp/crn_refly.log"
CHAIN="scripts/sweep_ladder_gamma.sh"
LADDERMARK="experiments/sweepladder_markers"
TABMARK="experiments/translationab_markers"
AIRFRAME="cf21_brushless"; DIST="L4C"
BITS="${CRN_BITS:-24 32}"; NEURONS="${CRN_NEURONS:-256}"; SEED="${CRN_SEED:-31337002}"
tag_of() { echo "SL_C_b${1}n${NEURONS}_${AIRFRAME}_${DIST}_g10_s${SEED}"; }
TAB_SEEDS="31337002 31337003 31337004 31337005 31337006"

log() { echo "[crn-refly] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controller_pids() { pgrep -f -- "-m wnn.control.phased_ga" 2>/dev/null || true; }
tab_pids() { pgrep -f "scripts/translation_ab_chain.sh" 2>/dev/null || true; }
ladder_pids() { pgrep -f "scripts/sweep_ladder_gamma.sh" 2>/dev/null || true; }

tab_missing() {
	local miss="" arm seed
	for seed in $TAB_SEEDS; do
		for arm in on off; do
			ls "${TABMARK}"/TAB_${arm}_b*n*_${AIRFRAME}_${DIST}_s${seed}.json >/dev/null 2>&1 \
				|| miss="${miss} TAB_${arm}_s${seed}"
		done
	done
	echo "$miss"
}

log "########## ARMED — CRN re-fly of b in [${BITS}] n=${NEURONS} seed ${SEED} (controls: the rotation-era markers) ##########"
for b in $BITS; do
	[ -f "${LADDERMARK}/$(tag_of "$b").json" ] || { log "ABORT — control marker $(tag_of "$b").json missing; nothing to pair against."; exit 1; }
done
grep -q 'SL_TAG_SUFFIX' "$CHAIN" || { log "ABORT — $CHAIN has no SL_TAG_SUFFIX hook."; exit 1; }

# ---- GATE on the A/B's 10 markers; fail closed if its chain died short.
beat=0
while :; do
	miss="$(tab_missing)"
	[ -z "$miss" ] && break
	if [ -z "$(tab_pids)" ]; then
		log "ABORT — translation A/B chain gone with markers MISSING:${miss}"
		log "A run needs a human. Launching nothing; box left idle deliberately."
		exit 1
	fi
	[ $((beat % 30)) = 0 ] && log "waiting on the translation A/B — still missing:${miss}"
	beat=$((beat + 1)); sleep 60
done
log "translation A/B complete — all 10 markers present."
while [ -n "$(controller_pids)" ]; do sleep 30; done

# ---- FLY via the ladder script (SL_SKIP_PHASE1: no gamma A/B; forced gamma=1).
log "===== LAUNCHING widths [${BITS}] as $(tag_of '{W}')_crn via $CHAIN (sequential, ladder SKIPs any existing marker) ====="
SL_SKIP_PHASE1=1 SL_SWEEP_LABEL="crn-refly" SL_FORCE_PHASE2_GAMMA="1.0" \
	SL_WIDTHS="$BITS" SL_NEURONS="$NEURONS" SL_SEED="$SEED" SL_TAG_SUFFIX="_crn" \
	nohup bash "$CHAIN" >/dev/null 2>&1 &
log "launched pid $!"
sleep 30
while [ -n "$(ladder_pids)" ]; do sleep 60; done
while [ -n "$(controller_pids)" ]; do sleep 30; done

miss=""
for b in $BITS; do
	[ -f "${LADDERMARK}/$(tag_of "$b")_crn.json" ] || miss="${miss} $(tag_of "$b")_crn.json"
done
if [ -n "$miss" ]; then
	log "ABORT — ladder exited but markers MISSING:${miss} (watchdog kill / crash / no MEMORY triple). A run needs a human."
	exit 1
fi
log "markers banked for widths [${BITS}]"
# ---- VERDICT: the leaderboard is the bar (never a hand-derived hd).
python3 scripts/gate_distance_leaderboard.py > docs/controller_gate_distance_leaderboard.md
log "PAIR (same shape, same seed, only the scorer differs) — leaderboard rows:"
for b in $BITS; do
	grep -E "$(tag_of "$b")(_crn)?$" docs/controller_gate_distance_leaderboard.md | while IFS= read -r row; do log "  $row"; done
done
log "########## DONE ##########"
