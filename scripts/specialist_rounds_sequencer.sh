#!/usr/bin/env bash
# SPECIALIST ROUNDS SEQUENCER (16/08/2026 late evening). Launched by the
# install guard at the round-1 boundary, AFTER the wheel install + 4 smokes.
#
# ORDER (Luiz 16/08 ~21:20 EDT): round 3 FIRST — he supplied the winners
# explicitly (SP3_BITS=30 SP3_NEURONS=128, arm A's shape) so the window sweep
# does not wait ~30h for round 2's sweeps — then round 2. Both chains are
# idempotent (markers skip completed runs), so re-running this sequencer
# resumes wherever it died.
set -u
ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
LOG="/private/tmp/specialist_sequencer.log"
log() { echo "[seq] $(date -u +%FT%TZ) $*" >> "$LOG"; }

log "########## SEQUENCER START — round 3 (windows @ b30/128n) then round 2 (sweep ladder) ##########"
SP3_BITS=30 SP3_NEURONS=128 bash scripts/specialist_round3_windows.sh
log "round 3 chain exited rc=$? — starting round 2"
bash scripts/specialist_round2_chain.sh
log "########## SEQUENCER COMPLETE — round 2 chain exited rc=$? ##########"
