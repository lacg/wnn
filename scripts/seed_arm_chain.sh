#!/usr/bin/env bash
# SEED-ARM CHAIN — ONE arm x N seeds at a fixed shape, paired against banked controls.
#
# The generic form of arm_b_delta_label_chain.sh (15/09/2026): every "flag bundle vs
# the _hd controls at b24 n256" arm since 11/09 was a copy of that script with three
# strings changed, so the strings are now parameters. The ladder (sweep_ladder_gamma.sh)
# still owns the recipe — this only names the arm, its extra flags, its marker fields
# and its control — so a recipe change never has to be copied into an arm.
#
# REQUIRED env:
#   ARM_SUFFIX       marker/tag suffix, e.g. _pipeN            (tag = SL_C_b24n256_..._s<seed><SUFFIX>)
#   ARM_EXTRA_ARGS   phased_ga flags appended AFTER the ladder's own (argparse last-wins,
#                    so "--skip-stages connections,bits" overrides the ladder's neurons,bits)
#   ARM_LABEL        one clause for the log + marker "arm" field, e.g. pipeline-neurons
#   ARM_MARKER_JSON  extra marker fields (raw JSON, leading comma), may be empty
# OPTIONAL env:
#   ARM_CTRL_SUFFIX  control suffix for the paired verdict (default _hd = D0 derived-hover runs)
#   ARM_SEEDS        default "31337002 31337003 31337004 31337005"
#   ARM_BITS/ARM_NEURONS   default 24/256
#   ARM_REQUIRE_FLAGS      space-separated flags that must exist in phased_ga --help (skew guard)
#   ARM_LOG          default /private/tmp/seed_arm<SUFFIX>.log
#   ARM_NO_CONTROL   =1 → no paired control exists for these seeds (e.g. a NEW seed whose
#                    control flies in this very chain as the first step): skip the control
#                    preflight and the paired verdict; the marker still records control_tag.
#                    Added 19/09/2026 for the CTRL-7 5th seed (s31337006: _hd29 first, then _pipeN).
#
# VERDICT is the mean paired delta and its 95% CI (scripts/paired_power.py), never a
# win tally. n=4 resolves ~0.3 deg err / ~0.1 m alt / ~0.6 deg steady; a steady or
# stable null at n=4 is INDETERMINATE.
#
# Marker-gated and idempotent (the ladder SKIPs an existing marker). FAILS CLOSED: a
# missing marker after a run means a human is needed. Honours experiments/HOLD_CONTROLLER
# through the ladder's wait_while_held.
set -u

cd "$(dirname "$0")/.." || exit 1
: "${ARM_SUFFIX:?ARM_SUFFIX required}"; : "${ARM_EXTRA_ARGS:?ARM_EXTRA_ARGS required}"; : "${ARM_LABEL:?ARM_LABEL required}"
SUFFIX="$ARM_SUFFIX"
LOG="${ARM_LOG:-/private/tmp/seed_arm${SUFFIX}.log}"
MARK="experiments/sweepladder_markers"
LADDER="scripts/sweep_ladder_gamma.sh"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
BITS="${ARM_BITS:-24}"
NEURONS="${ARM_NEURONS:-256}"
AIRFRAME="cf21_brushless"
DIST="L4C"
SEEDS="${ARM_SEEDS:-31337002 31337003 31337004 31337005}"
CTRL="${ARM_CTRL_SUFFIX:-_hd}"
HOVER="${TEACHER_HOVER:-derived}"

log() { echo "[arm${SUFFIX}] $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
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

ctrl_tag() { echo "SL_C_b${BITS}n${NEURONS}_${AIRFRAME}_${DIST}_g10_s$1${CTRL}"; }
arm_tag()  { echo "SL_C_b${BITS}n${NEURONS}_${AIRFRAME}_${DIST}_g10_s$1${SUFFIX}"; }

preflight() {
	busy && { log "ABORT — the box is NOT idle. Never launch this chain beside another."; exit 1; }
	local help; help="$(PYTHONPATH=src/wnn $VP -m wnn.control.phased_ga --help 2>/dev/null)"
	for f in ${ARM_REQUIRE_FLAGS:-}; do
		echo "$help" | grep -q -- "$f" || { log "ABORT — the Python tree has no $f (source/wheel skew)."; exit 1; }
	done
	if [ "${ARM_NO_CONTROL:-0}" = "1" ]; then
		log "preflight OK — box idle, flags present; ARM_NO_CONTROL=1: no paired control for these seeds (verdict skipped)."
		return 0
	fi
	local missing=""
	for seed in $SEEDS; do
		[ -f "${MARK}/$(ctrl_tag "$seed").json" ] || missing="${missing} $(ctrl_tag "$seed")"
	done
	[ -z "$missing" ] || { log "ABORT — control markers missing:${missing}. Nothing to pair against."; exit 1; }
	log "preflight OK — box idle, flags present, $(echo $SEEDS | wc -w | tr -d ' ') controls (${CTRL}) banked."
}

run_seed() {
	local seed="$1" tag; tag="$(arm_tag "$seed")"
	if [ -f "${MARK}/${tag}.json" ]; then
		log "SKIP — ${tag} already banked."; return 0
	fi
	wait_box_clear
	log "===== START ${tag} (${ARM_LABEL}; flags: ${ARM_EXTRA_ARGS}; control $(ctrl_tag "$seed")) ====="
	SL_SKIP_PHASE1=1 SL_SWEEP_LABEL="${ARM_LABEL}" SL_FORCE_PHASE2_GAMMA="1.0" \
		SL_WIDTHS="$BITS" SL_NEURONS="$NEURONS" SL_SEED="$seed" \
		SL_TAG_SUFFIX="$SUFFIX" \
		SL_EXTRA_ARGS="${ARM_EXTRA_ARGS} --teacher-hover ${HOVER}" \
		SL_EXTRA_MARKER_JSON=",\"arm\":\"${ARM_LABEL}\",\"teacher_hover\":\"${HOVER}\",\"control_tag\":\"$(ctrl_tag "$seed")\"${ARM_MARKER_JSON:-}" \
		bash "$LADDER"
	log "ladder exited rc=$? for ${tag}"
	while [ -n "$(controller_pids)" ]; do sleep 30; done
	[ -f "${MARK}/${tag}.json" ] || { log "ABORT — marker ${tag}.json MISSING. A run needs a human. Box left idle."; exit 1; }
	log "banked: ${tag}"
}

verdict() {
	$VP scripts/gate_distance_leaderboard.py > docs/controller_gate_distance_leaderboard.md 2>/dev/null
	if [ "${ARM_NO_CONTROL:-0}" = "1" ]; then
		log "---------- no paired verdict (ARM_NO_CONTROL=1) — leaderboard refreshed only ----------"; return 0
	fi
	log "---------- VERDICT: mean paired delta + 95% CI vs ${CTRL} (NOT a win tally) ----------"
	PYTHONPATH=src/wnn $VP scripts/paired_power.py \
		--arm "$SUFFIX" \
		--base "SL_C_b${BITS}n${NEURONS}_${AIRFRAME}_${DIST}_g10_s{seed}" \
		$(for s in $SEEDS; do printf -- '--seed %s ' "$s"; done) \
		--control-suffix "$CTRL" 2>&1 | tee -a "$LOG"
	log "CAVEAT: n=4 is under-powered for steady/stable; a null there is INDETERMINATE."
	log "CAVEAT: controls ${CTRL} flew on the pre-ABI-28 trainer; the arm flies on the installed wheel — era differs, exactly as arm B's pair did."
}

log "########## ARMED — ${ARM_LABEL} (${SUFFIX}) at b${BITS} n${NEURONS}, seeds [${SEEDS}], control ${CTRL} ##########"
preflight
for seed in $SEEDS; do run_seed "$seed"; done
verdict
log "########## ${ARM_LABEL} COMPLETE — $(echo $SEEDS | wc -w | tr -d ' ')/$(echo $SEEDS | wc -w | tr -d ' ') markers ##########"
