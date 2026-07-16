#!/usr/bin/env bash
# One-off recovery: QUAD held-out failed (bad mode string "QUAD"; needs QUAD_WEIGHTED).
# Wait for the current controller (PLN held-out) to finish so we NEVER run 2 controllers
# at once, then run QUAD at s2000/p50 with the correct mode string and write its marker.
set -u
ROOT="/Users/lacg/wnn"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
export PYTHONPATH="${ROOT}/src/wnn:${PYTHONPATH:-}"
cd "$ROOT" || exit 1
log() { echo "[quad-after-pln] $* $(date -u +%FT%TZ)"; }

log "waiting for controller to free (PLN held-out to finish)…"
while pgrep -f "gran_grid_winner_holdout" >/dev/null 2>&1; do sleep 60; done
log "controller free — launching QUAD_WEIGHTED held-out"

out="logs/controller/quad_grid_winner_holdout_s2000p50.out"
save="logs/controller/quad_grid_winner_s2000p50.yaml.gz"
"$VP" -u scripts/gran_grid_winner_holdout.py \
	--memory-mode QUAD_WEIGHTED --pop 50 --steps 2000 --save "$save" > "$out" 2>&1
rc=$?
res=$(grep -E "RESULT \(held-out\)" "$out" 2>/dev/null | tail -1)
all=$(grep -E "ALL DONE" "$out" 2>/dev/null | tail -1)
printf '{"arm":"quad","mode":"QUAD_WEIGHTED","rc":%s,"steps":2000,"pop":50,"result":"%s","alldone":"%s","done":"%s"}\n' \
	"$rc" "$(echo "$res" | tr -d '"' | sed 's/  */ /g')" \
	"$(echo "$all" | tr -d '"' | sed 's/  */ /g')" "$(date -u +%FT%TZ)" \
	> /tmp/wnn_gran_holdout_quad_s2000p50_done.json
log "QUAD done rc=$rc (marker written)"
