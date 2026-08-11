#!/usr/bin/env bash
# E1 LADDER RUNG n=7 — runs ONLY if the pre-registered rule says ESCALATE.
#
# WHY A GATE AND NOT JUST TWO MORE SEEDS (10/08/2026). The box runs one controller at
# a time and a chain waits on a PID, so this rung must be ARMED before the n=5 rung
# has finished. Arming it unconditionally would pre-commit to escalation regardless of
# what n=5 says — the same optional-stopping error the CI rule was written to avoid.
# So it asks scripts/refit_ladder_decision.py, which applies the rule verbatim:
#
#   PROMOTE / REFUTE  -> the question is ANSWERED; run nothing.
#   STOP              -> genuine null (CI spans 0, half-width <= 0.15 deg); run nothing.
#   ESCALATE          -> CI spans 0 and is still wide; run the next two seeds.
#
# Deciding in code rather than in a human's head also puts the reasoning in the chain
# log, which is where the next reader will look.
#
# NEW SEEDS ONLY. The pipeline is DETERMINISTIC given (seed, code, flags) — five
# control cells across three independently-launched chains reproduced bit-identically
# on every stage. So run-to-run variance is exactly zero, the whole 0.34 deg SD is
# BETWEEN-seed, and re-running an existing seed adds literally no information. Every
# rung must spend its runs on base seeds never flown before.
#
# Delegates to e1_coverage_2x2_chain.sh with E1_SEEDS: that chain already resolves the
# encoder factor, skips cells whose markers exist, and carries the refit pre-flight
# guard (abort unless "[thr-refit] ... REGRIDDING" appears). No second implementation.
#
# ARMING:  E1N7_WAIT_PID=<dob chain pid> nohup scripts/e1_rung_n7_chain.sh &
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1

LOG="/private/tmp/e1_rung_n7.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
SEEDS="${E1N7_SEEDS:-31337007 31337008}"
WAIT_PID="${E1N7_WAIT_PID:-}"
WAIT_CEIL="${E1N7_WAIT_CEIL:-259200}"

log() { echo "[e1n7] $(date -u +%FT%TZ) $*" >> "$LOG"; }

log "########## ARMED — rung n=7 candidate seeds=[$SEEDS] wait_pid=${WAIT_PID:-none} ##########"

if [ -n "$WAIT_PID" ]; then
	waited=0
	while kill -0 "$WAIT_PID" 2>/dev/null; do
		[ $((waited % 1800)) -eq 0 ] && log "waiting for gate PID $WAIT_PID (${waited}s)"
		sleep 60; waited=$((waited + 60))
		[ "$waited" -ge "$WAIT_CEIL" ] && { log "ABORT: gate alive after ${waited}s"; exit 3; }
	done
	log "gate PID $WAIT_PID exited after ${waited}s"
fi

# Decide AFTER the gate, so the n=5 (or n=6) markers are all on disk.
log "applying the pre-registered ladder rule to experiments/e1_coverage_markers"
VERDICT="$("$VP" scripts/refit_ladder_decision.py experiments/e1_coverage_markers 2>>"$LOG" | tr -d '[:space:]')"
log "verdict: ${VERDICT:-<none>}"

case "$VERDICT" in
	ESCALATE)
		log "ESCALATE — running seeds [$SEEDS] via e1_coverage_2x2_chain.sh"
		E1_SEEDS="$SEEDS" exec "$ROOT/scripts/e1_coverage_2x2_chain.sh"
		;;
	PROMOTE|REFUTE)
		log "$VERDICT — the CI excludes zero, the question is ANSWERED. Running nothing."
		log "  Write it up; do NOT add seeds to a decided result (that is optional stopping)."
		exit 0
		;;
	STOP)
		log "STOP — genuine null (CI spans 0, half-width <= 0.15 deg). Running nothing."
		exit 0
		;;
	*)
		log "ABORT: unusable verdict '${VERDICT:-<empty>}' — running nothing. See $LOG."
		exit 4
		;;
esac
