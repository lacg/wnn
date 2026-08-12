#!/usr/bin/env bash
# P1 handoff: run pidmix+tilt (nf=17) once the LIVE dfa1l cell completes, then give
# the box back to the sweep.
#
# WHY NOT dfa1l_restart_at_cell_boundary.sh --handoff: that watcher ABORTS (exit 8)
# if the sweep supervisor is alive when the boundary fires, so using it would mean
# disarming self-healing NOW and leaving the sweep unprotected for however many hours
# the run has left. This polls instead and stops the supervisor at the moment of
# action, so the sweep keeps its watchdog right up until we take the box.
#
# COST OF THAT CHOICE, stated plainly: the driver starts the NEXT cell within seconds
# of the current one finishing, so this kills a run that is a few minutes in. That
# cell writes no marker (R4), so the sweep simply re-runs it later from the top.
# Losing minutes is the right trade against hours of an unsupervised sweep.
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
LOG="/private/tmp/p1_chain.log"
MARKER="experiments/dfa1l_markers/dfa_10feat_BINARY_s31337004.json"
PROBE="scripts/run_l3d_feature_probe.sh"
STUDY="scripts/run_dfa_1layer_study.sh"
SEEDS="31337002 31337003 31337004"
ARMS_PER_SEED=8          # A1..A8; A1-A7 skip on their markers, only A8 runs

log() { echo "[p1] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controllers() { ps -axo pid,command 2>/dev/null | grep "wnn.control.phased_ga" \
	| grep -v "/usr/bin/time" | grep -v grep | grep -c python; }

log "########## ARMED — waiting for $MARKER ##########"

# ---- 1. wait for the live cell to write its completion marker ---------------
# Keyed on the MARKER, not on the process: the driver replaces the python within
# seconds, so a process-death test would fire on the wrong thing. The marker is the
# driver's own statement that the run genuinely finished (rc=0 + a MEMORY triple).
while [ ! -f "$MARKER" ]; do
	sleep 120
done
log "run 20 marker present — taking the box"

# ---- 2. supervisor FIRST, then the driver -----------------------------------
# Order matters: kill the driver while the supervisor lives and it relaunches one
# within a poll, and we would be fighting it.
pkill -f dfa1l_sweep_supervisor.sh 2>/dev/null; sleep 3
log "supervisor stopped: $(pgrep -f dfa1l_sweep_supervisor >/dev/null && echo STILL-ALIVE || echo ok)"
pkill -f "$STUDY" 2>/dev/null; sleep 2
pkill -f "wnn.control.phased_ga" 2>/dev/null; sleep 8
if [ "$(controllers)" -gt 0 ]; then
	log "SIGTERM left a controller alive — escalating"
	pkill -9 -f "wnn.control.phased_ga" 2>/dev/null; sleep 5
fi
log "box clear: controllers=$(controllers) drivers=$(pgrep -fc "$STUDY" 2>/dev/null || echo 0)"

# ---- 3. P1 ------------------------------------------------------------------
for seed in $SEEDS; do
	log "P1 seed=$seed"
	bash "$PROBE" "$seed" >> /private/tmp/l3dfeat_driver.log 2>&1
	n=$(ls -1 "experiments/l3dfeat_markers/"*"_s${seed}.json" 2>/dev/null | wc -l | tr -d ' ')
	log "P1 seed=$seed → ${n}/${ARMS_PER_SEED} markers"
done
log "P1 COMPLETE"

# ---- 4. give the box back — sweep first, supervisor last --------------------
# Same ordering rule as the overnight chain: a supervisor armed before the driver is
# up sees zero drivers and launches its own, racing ours.
log "resuming the dfa1l sweep"
nohup bash "$STUDY" >> /private/tmp/dfa1l_driver.log 2>&1 &
sleep 120
if pgrep -f "$STUDY" >/dev/null 2>&1; then
	nohup bash scripts/dfa1l_sweep_supervisor.sh 300 >> /private/tmp/dfa1l_supervisor.out 2>&1 &
	log "sweep up, supervisor re-armed"
else
	log "WARNING: sweep driver did NOT come up — supervisor NOT armed (it would \
relaunch blindly). Needs a human."
fi
log "########## P1 CHAIN DONE ##########"
