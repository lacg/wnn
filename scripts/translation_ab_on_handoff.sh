#!/usr/bin/env bash
# TRANSLATION A/B → ON-ONLY HANDOFF (04/09/2026 20:30 EDT, Luiz's call).
#
# WHY. The A/B's OFF arm (no z plant, 5 obs features) measures what the altitude
# regimen COSTS attitude. That number changes no lever: the axis is mandatory, and
# the gap it could explain (~0.3° of the ~0.9° to MPCOF) is the smaller part of
# the distance to the classical bar. One OFF point (seed 31337002, banking as this
# is written) is kept as the reference; the four remaining OFF runs (~24 h) are
# DROPPED and the budget goes to the ON seeds, which double as the replication the
# b32 n256 record (hd 0.1129, n=1) needs before it is a claim.
#
# HOW. The running chain (scripts/translation_ab_chain.sh) writes run 2's marker
# itself, so it must stay alive until that marker exists; a bash script cannot be
# edited while bash executes it (byte offsets). So this supervisor:
#   1. WAITS for TAB_off_..._s31337002.json to exist AND parse (pure wait; fails
#      closed if the chain dies without it — a run needs a human).
#   2. STOPS the chain (SIGKILL the bash loop) and any controller it launched in
#      the seconds between the marker and the kill — that can only be the ON s3
#      run, seconds old, nothing lost. SIGTERM → grace → SIGKILL, because
#      phased_ga HANDLES SIGTERM and does not exit on it.
#   3. PATCHES `for arm in on off` → `for arm in ${TAB_ARMS:-on off}` into the
#      chain in the exit window (grep-guarded, backed up, bash -n checked — the
#      sweep_ladder_gamma2_supervisor.sh precedent), so the recipe is never copied.
#   4. RELAUNCHES the same chain with TAB_ARMS=on for seeds 31337003..31337006.
#      run_point SKIPs any tag whose marker exists, so the relaunch is idempotent.
# The downstream waiters (crn_refly_chain.sh, leak_revisit_chain.sh) were re-gated
# on the ON markers and accept this script's PID as the chain's keeper, so the
# handoff window cannot trip their fail-closed abort.
set -u
ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1

LOG="/private/tmp/translation_ab_handoff.log"
CHAIN="scripts/translation_ab_chain.sh"
MARKDIR="experiments/translationab_markers"
GATE_MARKER="${MARKDIR}/TAB_off_b32n256_cf21_brushless_L4C_s31337002.json"
ON_SEEDS="${TAB_ON_SEEDS:-31337003 31337004 31337005 31337006}"

log() { echo "[tab-handoff] $(date -u +%FT%TZ) $*" >> "$LOG"; }
chain_pids() { pgrep -f "scripts/translation_ab_chain.sh" 2>/dev/null || true; }
# Broad on purpose: the /usr/bin/time wrapper must not be invisible to a kill.
controller_pids() { pgrep -f -- "-m wnn.control.phased_ga" 2>/dev/null || true; }

marker_ok() { [ -f "$GATE_MARKER" ] && /usr/bin/python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$GATE_MARKER" 2>/dev/null; }

# stop_controllers <grace-s>: SIGTERM, then SIGKILL. Fails closed.
stop_controllers() {
	local grace=${1:-60} waited=0 pids
	pids="$(controller_pids)"
	[ -z "$pids" ] && return 0
	log "preempt: SIGTERM $(echo $pids | tr '\n' ' ') (race-window launch, seconds old)"
	kill $pids 2>/dev/null || true
	while [ "$waited" -lt "$grace" ]; do
		sleep 5; waited=$((waited + 5))
		[ -z "$(controller_pids)" ] && { log "preempt: gone after SIGTERM (${waited}s)."; return 0; }
	done
	pids="$(controller_pids)"
	log "preempt: SURVIVED ${grace}s of SIGTERM -> SIGKILL $(echo $pids | tr '\n' ' ')"
	kill -9 $pids 2>/dev/null || true
	sleep 3
	if [ -n "$(controller_pids)" ]; then
		log "preempt: FATAL — survived SIGKILL: $(controller_pids | tr '\n' ' '). Refusing to continue."
		return 1
	fi
	log "preempt: gone after SIGKILL."
	return 0
}

log "########## ARMED — waiting for $(basename "$GATE_MARKER"); then chain -> ON-only seeds [${ON_SEEDS}] ##########"
[ -n "$(chain_pids)" ] || { log "ABORT — no translation_ab_chain.sh running; nothing to hand off from."; exit 1; }

# ---- 1. WAIT for run 2's marker. Pure wait, 2 s poll (the race window is seconds).
beat=0
while ! marker_ok; do
	if [ -z "$(chain_pids)" ]; then
		log "ABORT — chain gone and $(basename "$GATE_MARKER") missing (watchdog kill / crash / no MEMORY triple). A run needs a human. Box left idle."
		exit 1
	fi
	[ $((beat % 900)) = 0 ] && log "waiting — marker not yet written (chain pid $(chain_pids | tr '\n' ' '))"
	beat=$((beat + 1)); sleep 2
done
log "marker present and parses: $(basename "$GATE_MARKER")"

# ---- 2. STOP the chain, then whatever it launched in the race window.
pids="$(chain_pids)"
log "stopping chain pid(s) $(echo $pids | tr '\n' ' ') (SIGKILL — a bash loop, nothing to flush)"
kill -9 $pids 2>/dev/null || true
sleep 2
[ -n "$(chain_pids)" ] && { log "ABORT — chain survived SIGKILL: $(chain_pids | tr '\n' ' ')"; exit 1; }
if [ -n "$(controller_pids)" ]; then
	newest="$(ls -t logs/controller/translation_ab/*.out 2>/dev/null | head -1)"
	log "race-window controller present (newest .out: $(basename "${newest:-none}")) — stopping it"
	stop_controllers 60 || exit 1
else
	log "no controller in the race window."
fi

# ---- 3. PATCH the arm hook in. Safe now: nothing executes the file.
if grep -q 'for arm in ${TAB_ARMS:-on off}' "$CHAIN"; then
	log "arm hook already present — no patch needed."
else
	cp "$CHAIN" "${CHAIN}.pre-onOnly-$(date -u +%Y%m%dT%H%M%SZ).bak"
	/usr/bin/python3 - "$CHAIN" <<'PY'
import sys
p = sys.argv[1]
src = open(p).read()
old = '\tfor arm in on off; do\n'
assert src.count(old) == 1, "arm loop not found exactly once"
new = ('\t# TAB_ARMS hook (04/09/2026, Luiz): OFF arms dropped after one reference point —\n'
       '\t# see scripts/translation_ab_on_handoff.sh for the argument. Default unchanged.\n'
       '\tfor arm in ${TAB_ARMS:-on off}; do\n')
open(p, "w").write(src.replace(old, new))
PY
	if ! bash -n "$CHAIN"; then
		log "ABORT — patched $CHAIN FAILS bash -n. Restoring backup, launching nothing."
		cp "$(ls -t ${CHAIN}.pre-onOnly-*.bak | head -1)" "$CHAIN"
		exit 1
	fi
	log "arm hook patched in and syntax-checked."
fi

# ---- 4. RELAUNCH the same chain, ON-only. Its gates (gamma=2 markers, round-2
# sentinel) are already satisfied; the OFF preflight is seconds and harmless.
log "===== LAUNCHING $CHAIN with TAB_ARMS=on TAB_SEEDS=[${ON_SEEDS}] ====="
TAB_ARMS=on TAB_SEEDS="$ON_SEEDS" nohup bash "$CHAIN" >/dev/null 2>&1 &
log "launched pid $! — tags TAB_on_b32n256_cf21_brushless_L4C_s{SEED}"
sleep 5
[ -n "$(chain_pids)" ] || { log "ABORT — relaunched chain not running after 5 s."; exit 1; }
log "########## HANDOFF DONE — chain pid $(chain_pids | tr '\n' ' ') owns the ON seeds ##########"
