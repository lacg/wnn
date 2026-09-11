#!/usr/bin/env bash
# D0 A/B — the DERIVED teacher hover (--teacher-hover derived) vs the banked LEGACY anchors,
# 4 seeds at b24 n256, same recipe otherwise. Spec docs/multi_axis_programme_spec.md §0.
#
# WHAT IT TESTS. The DAgger training teachers were built at a hard-coded hover 0.5 while the
# plant hovers at sqrt(m·g/4k) = 0.694 (the scorer's rivals already used the derived value).
# With the switch ON the trainer's teachers are built at the derived hover and the label is
# re-based on the teacher's own hover (label = neutral + (p − hover_teacher)), so trainer and
# rival are the same law and PID / delta-label students become trainable.
#
# EXPECTED EFFECT ≈ 0 (probe docs/d0_hover_anchor_probe.txt: label ratio 1.010 mpcof, 1.002
# lqr). So this is an EQUIVALENCE check, not a lever: PRIMARY = ERR (n=4 MDE ~0.6-0.7°),
# ALT is the pre-registered NO-REGRESSION check (MDE ~0.08-0.16 m; the collective channel is
# untouched, so alt is the bug detector), steady secondary, stable descriptive.
# Verdict = scripts/paired_power.py mean delta + CI, never a win tally.
#
# Marker-gated, idempotent, fails closed. Controls = the banked s=1 anchors (CRN for seed 2).
set -u

cd "$(dirname "$0")/.." || exit 1
LOG="${ABHD_LOG:-/private/tmp/d0_hover_ab.log}"
MARK="experiments/sweepladder_markers"
LADDER="scripts/sweep_ladder_gamma.sh"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
BITS="${ABHD_BITS:-24}"
NEURONS="${ABHD_NEURONS:-256}"
AIRFRAME="cf21_brushless"
DIST="L4C"
SEEDS="${ABHD_SEEDS:-31337002 31337003 31337004 31337005}"
SUFFIX="_hd"

log() { echo "[d0-ab] $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
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
	if ! PYTHONPATH=src/wnn $VP -m wnn.control.phased_ga --help 2>/dev/null | grep -q -- "--teacher-hover"; then
		log "ABORT — the Python tree has no --teacher-hover (source/wheel skew)."; exit 1
	fi
	if ! PYTHONPATH=src/wnn $VP -m wnn.control.phased_ga --help 2>/dev/null | grep -q -- "derived"; then
		log "ABORT — --teacher-hover has no derived choice."; exit 1
	fi
	local missing=""
	for seed in $SEEDS; do
		[ -f "${MARK}/$(ctrl_tag "$seed").json" ] || missing="${missing} $(ctrl_tag "$seed")"
	done
	[ -z "$missing" ] || { log "ABORT — control markers missing:${missing}. Nothing to pair against."; exit 1; }
	PYTHONPATH=src/wnn $VP -c "import wnn.control._accel as a; assert a.EXPECTED_ABI >= 27" 2>/dev/null \
		|| { log "ABORT — installed wheel/facade predate ABI 27 (teacher_hover_mode)."; exit 1; }
	log "preflight OK — box idle, --teacher-hover derived present, ABI>=27, 4 controls banked."
}

run_seed() {
	local seed="$1"
	local tag="SL_C_b${BITS}n${NEURONS}_${AIRFRAME}_${DIST}_g10_s${seed}${SUFFIX}"
	if [ -f "${MARK}/${tag}.json" ]; then
		log "SKIP — ${tag} already banked."; return 0
	fi
	wait_box_clear
	log "===== START ${tag} (--teacher-hover derived, equivalence A/B; control $(ctrl_tag "$seed")) ====="
	SL_SKIP_PHASE1=1 SL_SWEEP_LABEL="d0-derived-hover-ab" SL_FORCE_PHASE2_GAMMA="1.0" \
		SL_WIDTHS="$BITS" SL_NEURONS="$NEURONS" SL_SEED="$seed" \
		SL_TAG_SUFFIX="$SUFFIX" \
		SL_EXTRA_ARGS="--teacher-hover derived" \
		SL_EXTRA_MARKER_JSON=",\"arm\":\"d0-derived-hover\",\"teacher_hover\":\"derived\",\"control_tag\":\"$(ctrl_tag "$seed")\"" \
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
		--primary err --arm "$SUFFIX" \
		--base "SL_C_b${BITS}n${NEURONS}_${AIRFRAME}_${DIST}_g10_s{seed}" \
		$(for s in $SEEDS; do printf -- '--seed %s ' "$s"; done) \
		--control-override "31337002=$(ctrl_tag 31337002)" 2>&1 | tee -a "$LOG"
	log "READ: equivalence check — ERR primary, ALT no-regression. A CI that excludes zero on alt is a BUG, not a result."
	log "CAVEAT: n=4 bounds the effect to ~0.6° err / ~0.1 m alt; if both straddle zero, derived becomes the DEFAULT."
}

log "########## ARMED — D0 A/B (derived teacher hover) at b${BITS} n${NEURONS}, seeds [${SEEDS}] ##########"
preflight
for seed in $SEEDS; do run_seed "$seed"; done
verdict
log "########## D0 A/B COMPLETE — 4/4 markers ##########"
