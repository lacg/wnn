#!/usr/bin/env bash
# Resume the dfa1l sweep once the ceiling pipeline finishes, so the box does not
# idle for days after a multi-day pipeline ends. Companion to
# dfa1l_handoff_to_ceiling.sh, which stopped the sweep at a cell boundary.
#
# WHY IT WATCHES THE PROCESS, NOT A MARKER: run_ceiling_pipeline.sh writes no
# completion marker, and its final `log "pipeline (PHASE) done"` runs even when a
# phase FAILED — `all` is `phase_S && phase_B && phase_A && phase_C`, so a phase-S
# failure short-circuits B/A/C and still reaches that line. Grepping for "done"
# would therefore resume the sweep on top of a failed pipeline and bury it.
#
# WHEN IT REFUSES (logs loudly, resumes NOTHING, leaves it for a human):
#   * the pipeline never appears within APPEAR_TIMEOUT_S — the handoff aborted
#     (IDS worker dead, phased_ga survivors, supervisor still up). Every one of
#     those is a reason NOT to start a controller, so idling is the correct
#     outcome and a silent resume would be the wrong one.
#   * the pipeline exits sooner than MIN_PIPELINE_S — that is a crash, not a run.
#     Resuming would hide the failure under a sweep that looks like progress.
#   * the IDS worker is gone, a phased_ga survives, or a driver already exists.
#     IDS is priority; a second driver is the double-run OOM risk.
#
# Usage: dfa1l_resume_after_ceiling.sh [appear_timeout_s] [min_pipeline_s] [poll_s]
set -u
ROOT="${RESUMER_ROOT:-/Users/lacg/wnn}"
cd "$ROOT" || exit 1

APPEAR_TIMEOUT_S="${1:-3600}"   # how long to wait for the pipeline to show up
MIN_PIPELINE_S="${2:-600}"      # shorter than this ⇒ crash, do not resume
POLL="${3:-60}"

SCRIPT="${RESUMER_SCRIPT:-scripts/run_dfa_1layer_study.sh}"
SUPERVISOR="${RESUMER_SUPERVISOR:-scripts/dfa1l_sweep_supervisor.sh}"
PIPELINE_PAT="${RESUMER_PIPELINE_PAT:-run_ceiling_pipeline.sh}"
PHASED_PAT="${RESUMER_PHASED_PAT:-wnn.control.phased_ga}"
DRIVER_LOG="${RESUMER_DRIVER_LOG:-/private/tmp/dfa1l_driver.log}"
SUP_LOG="${RESUMER_SUP_LOG:-/private/tmp/dfa1l_supervisor.log}"
LOG="${RESUMER_LOG:-/private/tmp/dfa1l_resumer.log}"
IDS_WORKER="${RESUMER_IDS_WORKER:-77344}"

log() { echo "[resumer] $(date -u +%FT%TZ) $*" >> "$LOG"; }

pipeline_count() { ps -axo command= 2>/dev/null | grep -c "[b]ash .*$PIPELINE_PAT"; }
driver_count()   { ps -axo command= 2>/dev/null | grep -c "[b]ash .*$SCRIPT"; }
# Count real phased_ga PYTHONs only; the /usr/bin/time wrapper is not a controller.
phased_count() {
	ps -axo pid,command 2>/dev/null | grep "$PHASED_PAT" | grep -v "/usr/bin/time" \
		| grep -v grep | grep -c python
}

log "ARMED: waiting up to ${APPEAR_TIMEOUT_S}s for '$PIPELINE_PAT' to appear (poll ${POLL}s)"

# ---- 1. wait for the pipeline to APPEAR --------------------------------------
waited=0
while [ "$(pipeline_count)" -eq 0 ]; do
	if [ "$waited" -ge "$APPEAR_TIMEOUT_S" ]; then
		log "ABORT: pipeline never started within ${APPEAR_TIMEOUT_S}s. The handoff \
most likely aborted (IDS worker down / phased_ga survivors / supervisor still up) — \
every one of those is a reason NOT to start a controller. Resuming NOTHING; human call."
		exit 2
	fi
	sleep "$POLL"
	waited=$((waited + POLL))
done
started_at=$(date +%s)
log "pipeline is running — waiting for it to finish (this is expected to take days)"

# ---- 2. wait for it to FINISH ------------------------------------------------
while [ "$(pipeline_count)" -gt 0 ]; do sleep "$POLL"; done
ran_for=$(( $(date +%s) - started_at ))
log "pipeline exited after ${ran_for}s"

if [ "$ran_for" -lt "$MIN_PIPELINE_S" ]; then
	log "ABORT: pipeline lasted ${ran_for}s < ${MIN_PIPELINE_S}s — that is a crash, not \
a run. NOT resuming the sweep (it would bury the failure under apparent progress). \
Check /private/tmp/ceiling_pipeline.log and logs/controller/ceiling/."
	exit 3
fi

# ---- 3. safety gates before starting a controller ----------------------------
sleep 30   # let the pipeline's own children reap
if ! ps -p "$IDS_WORKER" >/dev/null 2>&1; then
	log "ABORT: IDS worker $IDS_WORKER is not alive — IDS is priority and this is a \
human call. NOT resuming."
	exit 4
fi
left="$(phased_count)"
if [ "${left:-0}" -ne 0 ]; then
	log "ABORT: $left phased_ga python(s) still alive after the pipeline exited — \
starting the sweep now is the double-run OOM risk. NOT resuming."
	exit 5
fi
if [ "$(driver_count)" -gt 0 ]; then
	log "ABORT: a sweep driver is already running — nothing to resume."
	exit 6
fi

# ---- 4. resume the sweep, then re-arm its supervisor -------------------------
# The per-cell markers ARE the resume state: the driver re-enters the loop and
# skips every cell that already has one. No resume flag needed.
for attempt in 1 2 3; do
	nohup bash "$SCRIPT" >> "$DRIVER_LOG" 2>&1 < /dev/null &
	dpid=$!
	sleep 8
	if ps -p "$dpid" -o command= 2>/dev/null | grep -q "$SCRIPT"; then
		log "SWEEP RESUMED: driver pid=$dpid (skips every marked cell)"
		break
	fi
	log "driver relaunch attempt ${attempt}/3 did not survive 8s"
	dpid=""
	sleep 5
done
if [ -z "${dpid:-}" ]; then
	log "ABORT: driver failed to start 3 times — sweep STALLED, needs attention."
	exit 7
fi

nohup bash "$SUPERVISOR" 300 >> "$SUP_LOG" 2>&1 < /dev/null &
spid=$!
sleep 5
if ps -p "$spid" >/dev/null 2>&1; then
	log "supervisor re-armed pid=$spid — sweep is self-healing again"
else
	log "WARNING: supervisor did not start; the sweep runs but is NOT self-healing"
fi
log "DONE (children reparent to PPID=1 as this exits)"
exit 0
