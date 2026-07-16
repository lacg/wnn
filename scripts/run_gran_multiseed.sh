#!/usr/bin/env bash
# Phase 2 of the robust ablation: MULTI-SEED. For each new base seed, run the 5 grids
# (fresh independent search) then leak-free top-K on each. Combined with Phase 1 (base
# 31337002) this gives M=3 independent replications per mode. Test seed stays 99990101
# for all → mean±SD is search/model-induced on a fixed held-out.
# One controller at a time, per-step markers, skips done, NEVER kills. ~23h (grids dominate).
#   SEEDS="31337003 31337004" bash scripts/run_gran_multiseed.sh
set -u
ROOT="/Users/lacg/wnn"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
export PYTHONPATH="${ROOT}/src/wnn:${PYTHONPATH:-}"
cd "$ROOT" || exit 1
SEEDS="${SEEDS:-31337003 31337004}"
MODES="${MODES:-BINARY QSR TERNARY QUAD_WEIGHTED PLN}"
log() { echo "[gran-multiseed] $* $(date -u +%FT%TZ)"; }

for BASE in $SEEDS; do
	for M in $MODES; do
		ml=$(echo "$M" | tr '[:upper:]' '[:lower:]')          # file stem (matches grid script save)
		tag=$([ "$M" = "QUAD_WEIGHTED" ] && echo quad || echo "$ml")  # short marker tag
		yaml="logs/controller/${ml}_grid_winner_s2000p50_b${BASE}.yaml.gz"
		gmarker="/tmp/wnn_gran_grid_${tag}_b${BASE}_done.json"
		tkmarker="/tmp/wnn_gran_topk_${tag}_b${BASE}_done.json"

		# 1. GRID (fresh independent search at this base seed) → saves population yaml + #0 held-out
		if [ ! -f "$gmarker" ]; then
			gout="logs/controller/${tag}_grid_holdout_b${BASE}.out"
			log "===== START grid $M base=$BASE -> $gout ====="
			"$VP" -u scripts/gran_grid_winner_holdout.py --memory-mode "$M" --pop 50 --steps 2000 \
				--base-seed "$BASE" > "$gout" 2>&1
			rc=$?
			ad=$(grep -E "ALL DONE" "$gout" 2>/dev/null | tail -1)
			printf '{"tag":"%s","mode":"%s","base":%s,"rc":%s,"alldone":"%s","done":"%s"}\n' \
				"$tag" "$M" "$BASE" "$rc" "$(echo "$ad" | tr -d '"' | sed 's/  */ /g')" "$(date -u +%FT%TZ)" > "$gmarker"
			log "===== END grid $M base=$BASE rc=$rc ====="
		else
			log "grid $tag b$BASE: marker exists — skipping"
		fi

		# 2. TOP-K (leak-free train/val/test) on that population
		if [ ! -f "$tkmarker" ] && [ -f "$yaml" ]; then
			tkout="logs/controller/topk_${tag}_b${BASE}.out"
			log "===== START top-K $tag base=$BASE -> $tkout ====="
			"$VP" -u scripts/gran_topk_holdout.py "$yaml" --top-k 8 --base-seed "$BASE" > "$tkout" 2>&1
			rc2=$?
			res=$(grep -E "TOPK DONE" "$tkout" 2>/dev/null | tail -1)
			printf '{"tag":"%s","base":%s,"rc":%s,"result":"%s","done":"%s"}\n' \
				"$tag" "$BASE" "$rc2" "$(echo "$res" | tr -d '"' | sed 's/  */ /g')" "$(date -u +%FT%TZ)" > "$tkmarker"
			log "===== END top-K $tag base=$BASE rc=$rc2 ====="
		elif [ ! -f "$yaml" ]; then
			log "top-K $tag b$BASE: MISSING $yaml (grid failed?) — skipping"
		fi
	done
done
> "/tmp/wnn_gran_multiseed_done.json"
log "MULTI-SEED PHASE DONE for seeds: $SEEDS"
