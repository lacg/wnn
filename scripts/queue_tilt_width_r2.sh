#!/usr/bin/env bash
# queue_tilt_width_r2 — ROUND 2 of the DAgger tilt-curriculum width ladder (Luiz 12/10/2026, CTRL-25):
# _tw515 (5->15), _tw20 (8->20), _tw35 (8->35) x s2-s5, paired vs _op31 (8->30). Round 1 (_tw7/_tw10/_tw15)
# is banked 12/12. Rule PRE-REGISTERED in experiments/ctrl_tilt_width_r2_rule.json (round-1 bar, unchanged).
#
# SLOT: after the round-1 stress re-score ("STRESS DONE" in logs/controller/tilt_stress.log), before LEVELS.
# queue_2009c (LEVELS) is NOT running since the 12/10 power cut; it starts only when experiments/tilt_decision
# exists, so this supervisor stops before any run if that file appears.
#
# ORDER: seed-major interleaved — every seed flies all three arms before the next seed starts. Each
# (seed, arm) is one seed_arm_chain call; the chain is marker-gated (an existing marker SKIPs), so a
# relaunch resumes where it stopped. Honours experiments/HOLD_CONTROLLER via the ladder's wait_while_held.
# FAILS CLOSED on a missing marker.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG="logs/controller/queue_tilt_width_r2.log"
STRESSLOG="logs/controller/tilt_stress.log"
MARK="experiments/sweepladder_markers"
BASE="SL_C_b24n256_cf21_brushless_L4C_g10"
SEEDS="31337002 31337003 31337004 31337005"
ARMS="_tw515:5:15 _tw20:8:20 _tw35:8:35"
TOTAL=12
RECIPE="--obs-pwm --skip-stages connections,bits --max-output-neurons 512 --max-cells 1000000000"
REQ="--obs-pwm --skip-stages --max-output-neurons --rg-easy-tilt-deg --rg-full-tilt-deg"
RJ=",\"recipe\":\"grid-neurons-memory+obs-pwm\",\"anchor\":\"_op31\",\"obs_pwm\":true,\"trainer\":\"ABI31 stage-2 fixes (CTRL-17 G1-G4, altitude side incl.)\",\"rule\":\"experiments/ctrl_tilt_width_r2_rule.json\",\"control_curriculum\":\"8->30\",\"round\":2"
log() { echo "[qtw2] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
busy() { pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null || pgrep -f "scripts/sweep_ladder_gamm[a].sh" >/dev/null \
	|| pgrep -f "scripts/seed_arm_chai[n].sh" >/dev/null || pgrep -f "tilt_stress/run_stres[s].sh" >/dev/null; }
banked() { local n=0 s a; for s in $SEEDS; do for a in $ARMS; do [ -f "${MARK}/${BASE}_s${s}${a%%:*}.json" ] && n=$((n + 1)); done; done; echo $n; }
stress_done() { grep -q "STRESS DONE" "$STRESSLOG" 2>/dev/null; }

fly() {  # fly <seed> <suffix> <easy> <full>
	local seed="$1" sfx="$2" easy="$3" full="$4"
	[ -f "${MARK}/${BASE}_s${seed}${sfx}.json" ] && { log "SKIP — ${sfx} s${seed} already banked"; return 0; }
	while busy; do sleep 60; done
	[ -f experiments/tilt_decision ] && { log "STOP — experiments/tilt_decision exists; LEVELS owns the box. Round 2 at $(banked)/${TOTAL}."; exit 1; }
	log "START ${sfx} s${seed} (curriculum ${easy}->${full}) — $(banked)/${TOTAL} banked"
	ARM_SUFFIX="$sfx" ARM_LABEL="tilt-width-${easy}-${full}-abi31" ARM_SEEDS="$seed" \
		ARM_EXTRA_ARGS="${RECIPE} --rg-easy-tilt-deg ${easy} --rg-full-tilt-deg ${full}" ARM_REQUIRE_FLAGS="$REQ" \
		ARM_CTRL_SUFFIX="_op31" ARM_MARKER_JSON=",\"tilt_curriculum\":\"${easy}->${full}\"${RJ}" \
		ARM_LOG="logs/controller/seed_arm${sfx}.log" bash scripts/seed_arm_chain.sh
	[ -f "${MARK}/${BASE}_s${seed}${sfx}.json" ] || { log "ABORT — ${sfx} s${seed} marker MISSING. A run needs a human. Box left idle."; exit 1; }
	log "banked ${sfx} s${seed} — $(banked)/${TOTAL}"
}

log "########## ARMED — tilt-width ROUND 2 _tw515/_tw20/_tw35 x s2-s5 (seed-major), slot = after the stress re-score, before LEVELS ##########"
beat=0
until stress_done; do
	[ $((beat % 30)) = 0 ] && log "waiting for 'STRESS DONE' in ${STRESSLOG} (round 2 $(banked)/${TOTAL})"
	sleep 60; beat=$((beat + 1))
done
log "STRESS DONE seen — flying round 2 (do NOT write experiments/tilt_decision until 'TILT-WIDTH R2 DONE')"
for seed in $SEEDS; do
	for a in $ARMS; do
		IFS=: read -r sfx easy full <<< "$a"
		fly "$seed" "$sfx" "$easy" "$full"
	done
	log "---------- seed ${seed} round 2 complete ($(banked)/${TOTAL}) ----------"
done
[ "$(banked)" -ge "$TOTAL" ] || { log "ABORT — round 2 $(banked)/${TOTAL} after the loop. A run needs a human."; exit 1; }
log "TILT-WIDTH R2 DONE ${TOTAL}/${TOTAL} — apply experiments/ctrl_tilt_width_r2_rule.json (+ 15°/30° stress re-score with --trained-tilt 5), THEN Luiz writes experiments/tilt_decision"
