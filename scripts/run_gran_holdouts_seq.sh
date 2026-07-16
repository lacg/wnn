#!/usr/bin/env bash
# Sequential granularity grid-winner held-outs at ONE config (default s2000/p50) so all
# modes are apples-to-apples. Runs ONE mode at a time (never >1 controller), writes a
# per-mode marker, SKIPS modes whose marker already exists, and NEVER kills a run.
# Detach via scripts/detach_launch.py so it survives /clear and CLI exit (PPID=1).
#
#   MODES="TERNARY QUAD PLN" POP=50 STEPS=2000 bash scripts/run_gran_holdouts_seq.sh
set -u

MODES="${MODES:-TERNARY QUAD PLN}"
POP="${POP:-50}"
STEPS="${STEPS:-2000}"
ROOT="/Users/lacg/wnn"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
export PYTHONPATH="${ROOT}/src/wnn:${PYTHONPATH:-}"
cd "$ROOT" || exit 1

log() { echo "[gran-holdouts] $* $(date -u +%FT%TZ)"; }

for m in $MODES; do
	tag=$(echo "$m" | tr '[:upper:]' '[:lower:]')
	marker="/tmp/wnn_gran_holdout_${tag}_s${STEPS}p${POP}_done.json"
	out="logs/controller/${tag}_grid_winner_holdout_s${STEPS}p${POP}.out"
	save="logs/controller/${tag}_grid_winner_s${STEPS}p${POP}.yaml.gz"
	if [ -f "$marker" ]; then
		log "$tag: marker exists — skipping"
		continue
	fi
	log "===== START $m held-out (pop=$POP steps=$STEPS) -> $out ====="
	"$VP" -u scripts/gran_grid_winner_holdout.py \
		--memory-mode "$m" --pop "$POP" --steps "$STEPS" --save "$save" > "$out" 2>&1
	rc=$?
	# Pull the reported numbers out of the log for the marker (best-effort).
	res_line=$(grep -E "RESULT \(held-out\)" "$out" 2>/dev/null | tail -1)
	all_line=$(grep -E "ALL DONE" "$out" 2>/dev/null | tail -1)
	printf '{"arm":"%s","mode":"%s","rc":%s,"steps":%s,"pop":%s,"result":"%s","alldone":"%s","done":"%s"}\n' \
		"$tag" "$m" "$rc" "$STEPS" "$POP" \
		"$(echo "$res_line" | tr -d '"' | sed 's/  */ /g')" \
		"$(echo "$all_line" | tr -d '"' | sed 's/  */ /g')" \
		"$(date -u +%FT%TZ)" > "$marker"
	log "===== END $m rc=$rc (marker $marker) ====="
done

log "ALL HELD-OUTS DONE: $MODES"
> "/tmp/wnn_gran_holdouts_seq_done.json"
