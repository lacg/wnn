#!/usr/bin/env bash
# Phase 1 of the robust ablation: leak-free top-K held-out (train→pick-on-val→report-on-test)
# over the 5 EXISTING saved grid populations (base seed 31337002 = seed #1 of the M=3 plan).
# One mode at a time (never >1 controller), per-mode marker, skips done, NEVER kills.
#   BASE=31337002 bash scripts/run_gran_topk.sh
set -u
ROOT="/Users/lacg/wnn"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
export PYTHONPATH="${ROOT}/src/wnn:${PYTHONPATH:-}"
cd "$ROOT" || exit 1
BASE="${BASE:-31337002}"
TOPK="${TOPK:-8}"
log() { echo "[gran-topk] $* $(date -u +%FT%TZ)"; }

# tag -> saved population file (existing grids)
FILES="binary:binary_grid_winner qsr:qsr_grid_winner ternary:ternary_grid_winner_s2000p50 quad:quad_grid_winner_s2000p50 pln:pln_grid_winner_s2000p50"

for pair in $FILES; do
	tag="${pair%%:*}"; f="${pair##*:}"
	marker="/tmp/wnn_gran_topk_${tag}_b${BASE}_done.json"
	out="logs/controller/topk_${tag}_b${BASE}.out"
	yaml="logs/controller/${f}.yaml.gz"
	if [ -f "$marker" ]; then log "$tag: marker exists — skipping"; continue; fi
	if [ ! -f "$yaml" ]; then log "$tag: MISSING $yaml — skipping"; continue; fi
	log "===== START top-K $tag (base=$BASE, K=$TOPK) -> $out ====="
	"$VP" -u scripts/gran_topk_holdout.py "$yaml" --top-k "$TOPK" --base-seed "$BASE" > "$out" 2>&1
	rc=$?
	res=$(grep -E "TOPK DONE" "$out" 2>/dev/null | tail -1)
	printf '{"tag":"%s","rc":%s,"base":%s,"topk":%s,"result":"%s","done":"%s"}\n' \
		"$tag" "$rc" "$BASE" "$TOPK" "$(echo "$res" | tr -d '"' | sed 's/  */ /g')" "$(date -u +%FT%TZ)" > "$marker"
	log "===== END top-K $tag rc=$rc (marker $marker) ====="
done
> "/tmp/wnn_gran_topk_b${BASE}_done.json"
log "TOP-K PHASE DONE for base=$BASE"
