#!/usr/bin/env bash
# THE POST-ARM-A QUEUE — Luiz's order, 11/09/2026. Replaces the bare
# launch_leak_x_labelscale_when_ready.sh launcher (which would have let the 2x2 abort).
#
#   STEP 1  2x2 leak x label-scale, FORCED on s=2 via LS_STAR=2        4 runs  ~20 h
#   STEP 2  ARM B, the true-delta label (--dagger-label-delta --obs-pwm) 4 runs ~20 h
#   STEP 3  window-k FRAMED, runs 2..12 (re-enters the marker-gated sequencer) 11 runs ~50 h
#   STEP 4  multi-axis programme — DESIGN ONLY, no script exists. The queue STOPS here
#           and says so rather than inventing an experiment.
#
# WHY LS_STAR=2 IS AN OVERRIDE, NOT A RESULT. Arm A's own s* rule (>=3/4 paired steady
# wins) selects NOTHING: s=2 went 2/4, s=4 1/4, s=8 0/4. Luiz chose to fly the cross on
# s=2 anyway. That is legitimate — the rule it fails is a sign test that fires 31% of the
# time on a dead lever (commit 4b2bb46a) — but it means the 2x2's input rung is NOT an
# established winner. Say "s=2 was forced, not selected" wherever the 2x2 is reported.
#
# EVERY STEP: waits for an idle box, is marker-gated and idempotent, and FAILS CLOSED —
# a step whose markers are missing when its chain exits stops the queue and leaves the
# box idle for a human. Nothing here can preempt a live run.
set -u

cd "$(dirname "$0")/.." || exit 1
LOG="${PQ_LOG:-/private/tmp/post_arma_queue.log}"
MARK="experiments/sweepladder_markers"
BASE="SL_C_b24n256_cf21_brushless_L4C_g10_s"
SEEDS="31337002 31337003 31337004 31337005"

log() { echo "[post-arma] $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
controller_pids() { pgrep -f "MacOS/Python -u -m wnn.control.phased_ga" 2>/dev/null || true; }
chain_pids() { pgrep -f "scripts/(label_scale_arm_chain|sweep_ladder_gamma|leak_x_labelscale_chain|arm_b_delta_label_chain|queue_after_ab_chain)\.sh" 2>/dev/null || true; }
busy() { [ -n "$(controller_pids)" ] || [ -n "$(chain_pids)" ]; }

arm_a_done() {
	[ "$(ls "$MARK" | grep -cE '_ls[248]\.json$')" -ge 12 ] \
		&& ! pgrep -f "scripts/label_scale_arm_chain.sh" >/dev/null 2>&1
}

wait_box_clear() {
	local beat=0
	while busy; do
		sleep 60; beat=$((beat + 1))
		[ $((beat % 30)) = 0 ] && log "waiting — box busy"
	done
	sleep 120
}

count_markers() { ls "$MARK" 2>/dev/null | grep -cE "$1" ; }

# run_step <name> <marker regex> <expected count> <command...>
run_step() {
	local name="$1" regex="$2" want="$3"; shift 3
	local have; have="$(count_markers "$regex")"
	if [ "$have" -ge "$want" ]; then
		log "SKIP ${name} — already complete (${have}/${want} markers)."
		return 0
	fi
	wait_box_clear
	log "===== STEP ${name}: starting (${have}/${want} markers present) ====="
	"$@"
	local rc=$?
	log "${name} chain exited rc=${rc}"
	while [ -n "$(controller_pids)" ]; do sleep 30; done
	have="$(count_markers "$regex")"
	if [ "$have" -lt "$want" ]; then
		log "ABORT — ${name} finished with ${have}/${want} markers. A run needs a human."
		log "The box is left IDLE. Nothing further in this queue runs."
		exit 1
	fi
	log "===== STEP ${name}: COMPLETE (${have}/${want} markers) ====="
}

log "########## POST-ARM-A QUEUE ARMED — 2x2(s=2 forced) -> arm B -> window-k -> stop ##########"
log "waiting for arm A to complete (12 _ls markers) and the box to go idle"
while ! arm_a_done; do sleep 60; done
log "arm A complete."

run_step "1 2x2-leak-x-labelscale(s=2 FORCED)" '_l090_ls2\.json$' 4 \
	env LS_STAR=2 bash scripts/leak_x_labelscale_chain.sh

run_step "2 arm-B-true-delta-label" '_bd\.json$' 4 \
	bash scripts/arm_b_delta_label_chain.sh

run_step "3 window-k-FRAMED runs 2..12" '_win[234]\.json$' 12 \
	bash scripts/queue_after_ab_chain.sh

log "########## QUEUE DRAINED ##########"
log "STEP 4 (multi-axis programme) has NO SCRIPT and NO SPEC — only a one-line"
log "placeholder (teacher/disturbance/airframe/levels/sn>0 x altitude). It is design"
log "work, not compute. The box is now IDLE and waiting for a written design."
