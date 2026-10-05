#!/usr/bin/env bash
# queue_tilt_width_sweep — DAgger tilt-curriculum width ladder below 30° (Luiz 05/10/2026):
# _tw7 (3->7), _tw10 (5->10), _tw15 (8->15) x s2-s5, paired vs _op31 (8->30); _tc31 (5->5) is the other
# baseline. Rule PRE-REGISTERED in experiments/ctrl_tilt_width_rule.json.
#
# SLOT: after b20 ("b20 DONE 4/4" in /private/tmp/queue_b20.log), inside queue_2009c's tilt PARK, BEFORE
# LEVELS. queue_2009c and queue_b20 are running bash scripts and must never be edited, so this follow-on
# supervisor waits for b20's DONE line. LEVELS starts the moment experiments/tilt_decision exists — DO NOT
# write it until this logs "TILT-WIDTH DONE 12/12".
#
# ORDER: seed-major interleaved — every seed flies the whole ladder before the next seed starts, so a
# partial result is always a complete per-seed ladder. Each (seed, arm) is one seed_arm_chain call; the
# chain is marker-gated (an existing marker SKIPs), so a relaunch resumes where it stopped.
# Honours experiments/HOLD_CONTROLLER via the ladder's wait_while_held. FAILS CLOSED on a missing marker.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG="/private/tmp/queue_tilt_width.log"
B20LOG="/private/tmp/queue_b20.log"
MARK="experiments/sweepladder_markers"
BASE="SL_C_b24n256_cf21_brushless_L4C_g10"
SEEDS="31337002 31337003 31337004 31337005"
ARMS="_tw7:3:7 _tw10:5:10 _tw15:8:15"
RECIPE="--obs-pwm --skip-stages connections,bits --max-output-neurons 512 --max-cells 1000000000"
REQ="--obs-pwm --skip-stages --max-output-neurons --rg-easy-tilt-deg --rg-full-tilt-deg"
RJ=",\"recipe\":\"grid-neurons-memory+obs-pwm\",\"anchor\":\"_op31\",\"obs_pwm\":true,\"trainer\":\"ABI31 stage-2 fixes (CTRL-17 G1-G4, altitude side incl.)\",\"rule\":\"experiments/ctrl_tilt_width_rule.json\",\"control_curriculum\":\"8->30\""
log() { echo "[qtw] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
busy() { pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null || pgrep -f "scripts/sweep_ladder_gamm[a].sh" >/dev/null || pgrep -f "scripts/seed_arm_chai[n].sh" >/dev/null; }
banked() { local n=0 s a; for s in $SEEDS; do for a in $ARMS; do [ -f "${MARK}/${BASE}_s${s}${a%%:*}.json" ] && n=$((n + 1)); done; done; echo $n; }
b20_done() { grep -q "b20 DONE 4/4" "$B20LOG" 2>/dev/null; }

fly() {  # fly <seed> <suffix> <easy> <full>
	local seed="$1" sfx="$2" easy="$3" full="$4"
	[ -f "${MARK}/${BASE}_s${seed}${sfx}.json" ] && { log "SKIP — ${sfx} s${seed} already banked"; return 0; }
	while busy; do sleep 60; done
	log "START ${sfx} s${seed} (curriculum ${easy}->${full}) — $(banked)/12 banked"
	ARM_SUFFIX="$sfx" ARM_LABEL="tilt-width-${easy}-${full}-abi31" ARM_SEEDS="$seed" \
		ARM_EXTRA_ARGS="${RECIPE} --rg-easy-tilt-deg ${easy} --rg-full-tilt-deg ${full}" ARM_REQUIRE_FLAGS="$REQ" \
		ARM_CTRL_SUFFIX="_op31" ARM_MARKER_JSON=",\"tilt_curriculum\":\"${easy}->${full}\"${RJ}" \
		ARM_LOG="/private/tmp/seed_arm${sfx}.log" bash scripts/seed_arm_chain.sh
	[ -f "${MARK}/${BASE}_s${seed}${sfx}.json" ] || { log "ABORT — ${sfx} s${seed} marker MISSING. A run needs a human. Box left idle."; exit 1; }
	log "banked ${sfx} s${seed} — $(banked)/12"
}

log "########## ARMED — tilt-width ladder _tw7/_tw10/_tw15 x s2-s5 (seed-major), slot = after b20, before LEVELS ##########"
beat=0
until b20_done; do
	[ -f experiments/tilt_decision ] && { log "ABORT — tilt_decision already written; LEVELS may be running. The ladder needs a human slot."; exit 1; }
	[ $((beat % 30)) = 0 ] && log "waiting for 'b20 DONE 4/4' in ${B20LOG} (ladder $(banked)/12)"
	sleep 60; beat=$((beat + 1))
done
[ -f experiments/tilt_decision ] && { log "ABORT — tilt_decision exists at b20 DONE; LEVELS may be running."; exit 1; }
log "b20 DONE seen — flying the ladder (do NOT write experiments/tilt_decision until 'TILT-WIDTH DONE')"
for seed in $SEEDS; do
	for a in $ARMS; do
		IFS=: read -r sfx easy full <<< "$a"
		fly "$seed" "$sfx" "$easy" "$full"
	done
	log "---------- seed ${seed} ladder complete ($(banked)/12) ----------"
done
[ "$(banked)" -ge 12 ] || { log "ABORT — ladder $(banked)/12 after the loop. A run needs a human."; exit 1; }
log "TILT-WIDTH DONE 12/12 — apply experiments/ctrl_tilt_width_rule.json (+ ctrl_tilt_rule.json for _tc31, + 15°/30° stress re-score), THEN write experiments/tilt_decision (Luiz)"
