#!/usr/bin/env bash
# Unattended chain for the 01/08 → 02/08 overnight window (~16h).
#
# ORDER MATTERS AND IS THE WHOLE POINT:
#   1. wait for the RUNNING probe (round 1, seed 31337002) to finish on its own
#   2. probe rounds 2 + 3 (seeds 31337003, 31337004) → n=3 on all four arms
#   3. ONLY THEN resume the dfa1l sweep, and only then re-arm its supervisor
#
# Why the supervisor is armed LAST: it relaunches the dfa1l driver whenever it sees
# zero drivers. Armed during the probe it would start a second controller alongside
# it — the double-run this whole design exists to avoid. It is therefore the last
# thing started, after the sweep is already back up.
#
# Why no culling between rounds: culling needs judgment (is A3 at 0% a real negative
# or a broken arm?) and nobody is awake to apply it. Running all three rounds costs
# at most a couple of hours of arms we might have dropped, and buys n=3 on every arm
# — which is a stronger negative result than n=1 if L3D does come back at zero.
#
# Each probe round is RESUMABLE (per-arm marker skip-gate), so the retry loop below
# re-enters and continues rather than restarting. Retries are bounded: a genuinely
# broken arm must not burn the night in a relaunch loop.
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
LOG="/private/tmp/overnight_chain.log"
PROBE="scripts/run_l3d_feature_probe.sh"
STUDY="scripts/run_dfa_1layer_study.sh"
MAX_TRIES=3

log() { echo "[chain] $(date -u +%FT%TZ) $*" >> "$LOG"; }

controllers() { ps -axo pid,command 2>/dev/null | grep "wnn.control.phased_ga" \
	| grep -v "/usr/bin/time" | grep -v grep | grep -c python; }

# Block until no controller AND no probe driver is alive. Belt and braces: keying on
# the driver alone would race the last arm's python, and keying on the python alone
# would fire in the gap between two arms.
wait_for_quiet() {
	local what="$1"
	while pgrep -f "$PROBE" >/dev/null 2>&1 || [ "$(controllers)" -gt 0 ]; do
		sleep 60
	done
	log "quiet: $what"
}

run_probe_round() {
	local seed="$1" try=1
	while [ "$try" -le "$MAX_TRIES" ]; do
		log "probe seed=$seed attempt $try/$MAX_TRIES"
		bash "$PROBE" "$seed" >> /private/tmp/l3dfeat_driver.log 2>&1
		# 4 arms per seed; the marker skip-gate makes a re-entry cheap.
		local n
		n=$(ls -1 "experiments/l3dfeat_markers/"*"_s${seed}.json" 2>/dev/null | wc -l | tr -d ' ')
		log "probe seed=$seed produced ${n}/4 markers"
		[ "${n:-0}" -ge 4 ] && return 0
		try=$((try+1))
	done
	log "probe seed=$seed INCOMPLETE after $MAX_TRIES tries — moving on, not blocking the chain"
	return 1
}

log "########## CHAIN ARMED ##########"

# ---- 1. let round 1 finish -------------------------------------------------
log "waiting for the in-flight round 1 (seed 31337002) to finish"
wait_for_quiet "round 1 done"

# ---- 2. rounds 2 and 3 -----------------------------------------------------
for seed in 31337003 31337004; do
	run_probe_round "$seed"
	wait_for_quiet "probe seed=$seed settled"
done
log "PROBE COMPLETE — markers: $(ls -1 experiments/l3dfeat_markers/*.json 2>/dev/null | wc -l | tr -d ' ')"

# ---- 3. resume the dfa1l sweep, THEN arm its supervisor --------------------
# dfa_10feat_BINARY_s31337004 has no marker (killed at 4:31 for the probe), so the
# driver re-enters the loop and re-runs it from the top. Nothing else is lost.
log "resuming the dfa1l sweep"
nohup bash "$STUDY" >> /private/tmp/dfa1l_driver.log 2>&1 &
sleep 120
if pgrep -f "$STUDY" >/dev/null 2>&1; then
	log "sweep driver up — arming the supervisor"
	nohup bash scripts/dfa1l_sweep_supervisor.sh 300 >> /private/tmp/dfa1l_supervisor.out 2>&1 &
	log "supervisor armed"
else
	log "WARNING: sweep driver did NOT come up — NOT arming the supervisor (it would \
relaunch blindly). Needs a human."
fi

log "########## CHAIN DONE ##########"
