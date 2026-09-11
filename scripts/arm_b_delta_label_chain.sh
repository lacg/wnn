#!/usr/bin/env bash
# ARM B — the TRUE-DELTA DAgger label, 4 seeds at b24 n256, vs the banked s=1 controls.
#
# THE LEVER. Arm A rescaled the label (--delta-label-scale) but kept its SEMANTICS:
# the teacher's ABSOLUTE pwm, floored to the 1/L grid, integrated by the leaky
# accumulator. Arm B changes the semantics instead: --dagger-label-delta labels the
# STEP the teacher took, not the level it reached.
#
# ⚠️ THIS IS A TWO-FLAG BUNDLE AND CANNOT BE ONE FLAG. phased_ga's own help says the
# delta label "depends on the accumulator: pair with --obs-pwm" — without the pwm
# observation the student cannot know the level it is stepping from, so the label is
# unreadable. The arm therefore differs from its control by BOTH the label semantics
# AND one extra observation channel. An arm win reads "the delta label AND/OR seeing
# its own pwm helps". Quote that caveat with every number, exactly as the translation
# A/B does with its four-flag bundle.
#
# ⚠️ POWER, STATED UP FRONT (scripts/paired_power.py on arm A, docs/controller_paired_power.txt):
# at the observed paired SD this design can resolve
#     altitude  ~0.10 m   (needs <=5 seeds)   -> ADEQUATELY POWERED
#     err       ~0.60 deg (needs 4 for 0.5)   -> only a large effect
#     steady    ~0.65 deg (needs 10 for 0.3)  -> UNDER-POWERED, cannot see a small effect
#     stable    ~1.3 pp                       -> UNDER-POWERED
# So a null result on steady from this arm is INDETERMINATE, never a refutation. Say so
# in the verdict. If the steady question must be answered, it needs ~10-21 seeds, not 4.
#
# VERDICT is the mean paired delta and its 95% CI from scripts/paired_power.py — NOT a
# win/loss tally. A "k of n wins" rule is a sign test that fires 31% of the time on a
# dead lever at n=4 (see commit 4b2bb46a).
#
# Marker-gated and idempotent (the ladder SKIPs an existing marker). FAILS CLOSED: a
# missing marker after a run means a human is needed — the chain stops and leaves the
# box idle rather than stacking work on a crash.
set -u

cd "$(dirname "$0")/.." || exit 1
LOG="${ARMB_LOG:-/private/tmp/arm_b_delta_label.log}"
MARK="experiments/sweepladder_markers"
LADDER="scripts/sweep_ladder_gamma.sh"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
BITS="${ARMB_BITS:-24}"
NEURONS="${ARMB_NEURONS:-256}"
AIRFRAME="cf21_brushless"
DIST="L4C"
SEEDS="${ARMB_SEEDS:-31337002 31337003 31337004 31337005}"
SUFFIX="_bd"

log() { echo "[arm-b] $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
controller_pids() { pgrep -f "MacOS/Python -u -m wnn.control.phased_ga" 2>/dev/null || true; }
ladder_pids() { pgrep -f "scripts/sweep_ladder_gamma.sh" 2>/dev/null || true; }
busy() { [ -n "$(controller_pids)" ] || [ -n "$(ladder_pids)" ]; }

wait_box_clear() {
	local beat=0
	while busy; do
		sleep 30; beat=$((beat + 1))
		[ $((beat % 30)) = 0 ] && log "waiting — box busy"
	done
}

# The control for seed 31337002 is the CRN re-fly; the others are the plain markers.
ctrl_tag() {
	local seed="$1" base="SL_C_b${BITS}n${NEURONS}_${AIRFRAME}_${DIST}_g10_s${seed}"
	[ "$seed" = "31337002" ] && echo "${base}_crn" || echo "$base"
}

preflight() {
	busy && { log "ABORT — the box is NOT idle. Never launch this chain beside another."; exit 1; }
	# D0 (11/09): in delta-label mode the label is pid_pwms - leaked_baseline (~0.694 under
	# --translation) while the mpcof teacher emits ~0.5 at level -> every level step labelled
	# "max descend". Refuse until the label re-base has landed (sentinel touched by its deploy).
	[ -f "experiments/labelscale_markers/LABEL_REBASE_LANDED.json" ] \
		|| { log "ABORT — label re-base not landed (spec docs/multi_axis_programme_spec.md §0.9). Arm B would train on 'max descend' labels."; exit 1; }
	if ! PYTHONPATH=src/wnn $VP -m wnn.control.phased_ga --help 2>/dev/null | grep -q -- "--dagger-label-delta"; then
		log "ABORT — the Python tree has no --dagger-label-delta (source/wheel skew)."; exit 1
	fi
	if ! PYTHONPATH=src/wnn $VP -m wnn.control.phased_ga --help 2>/dev/null | grep -q -- "--obs-pwm"; then
		log "ABORT — the Python tree has no --obs-pwm; the delta label needs it."; exit 1
	fi
	local missing=""
	for seed in $SEEDS; do
		[ -f "${MARK}/$(ctrl_tag "$seed").json" ] || missing="${missing} $(ctrl_tag "$seed")"
	done
	[ -z "$missing" ] || { log "ABORT — control markers missing:${missing}. Nothing to pair against."; exit 1; }
	log "preflight OK — box idle, both flags present, 4 controls banked."
}

run_seed() {
	local seed="$1"
	local tag="SL_C_b${BITS}n${NEURONS}_${AIRFRAME}_${DIST}_g10_s${seed}${SUFFIX}"
	if [ -f "${MARK}/${tag}.json" ]; then
		log "SKIP — ${tag} already banked."; return 0
	fi
	wait_box_clear
	log "===== START ${tag} (--dagger-label-delta --obs-pwm, TWO-FLAG bundle; control $(ctrl_tag "$seed")) ====="
	SL_SKIP_PHASE1=1 SL_SWEEP_LABEL="arm-b-delta-label" SL_FORCE_PHASE2_GAMMA="1.0" \
		SL_WIDTHS="$BITS" SL_NEURONS="$NEURONS" SL_SEED="$seed" \
		SL_TAG_SUFFIX="$SUFFIX" \
		SL_EXTRA_ARGS="--dagger-label-delta --obs-pwm" \
		SL_EXTRA_MARKER_JSON=",\"arm\":\"delta-label\",\"dagger_label_delta\":true,\"obs_pwm\":true,\"flag_bundle\":2,\"control_tag\":\"$(ctrl_tag "$seed")\"" \
		bash "$LADDER"
	log "ladder exited rc=$? for ${tag}"
	while [ -n "$(controller_pids)" ]; do sleep 30; done
	[ -f "${MARK}/${tag}.json" ] || { log "ABORT — marker ${tag}.json MISSING. A run needs a human. Box left idle."; exit 1; }
	log "banked: ${tag}"
}

verdict() {
	$VP scripts/gate_distance_leaderboard.py > docs/controller_gate_distance_leaderboard.md 2>/dev/null
	log "---------- VERDICT: mean paired delta + 95% CI (NOT a win tally) ----------"
	PYTHONPATH=src/wnn $VP scripts/paired_power.py \
		--arm "$SUFFIX" \
		--base "SL_C_b${BITS}n${NEURONS}_${AIRFRAME}_${DIST}_g10_s{seed}" \
		$(for s in $SEEDS; do printf -- '--seed %s ' "$s"; done) \
		--control-override "31337002=$(ctrl_tag 31337002)" 2>&1 | tee -a "$LOG"
	log "CAVEAT to quote with every number: two-flag bundle (delta label + pwm observation)."
	log "CAVEAT to quote with every number: n=4 is under-powered for steady/stable; a null there is INDETERMINATE."
}

log "########## ARMED — ARM B (true-delta label) at b${BITS} n${NEURONS}, seeds [${SEEDS}] ##########"
preflight
for seed in $SEEDS; do run_seed "$seed"; done
verdict
log "########## ARM B COMPLETE — 4/4 markers ##########"
