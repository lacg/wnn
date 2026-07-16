#!/usr/bin/env bash
# Phase 3 of the robust granularity ablation: FULL-PIPELINE CEILING.
# Runs the REAL production phased_ga (grid -> GA-Neurons -> GA-Memory, the binqsr_v3/C10
# recipe — identical argv to gran_grid_winner_holdout._phased_argv, but through main() so
# ALL stages run) for each mode x seed. Per run captures the 4-axis deployability matrix:
#   quality  — held-out triple (stable/err/steady) printed by phased_ga's report-seed path
#   runtime  — wall duration + peak RSS (/usr/bin/time -l)
#   FPGA     — populated cells x bits/cell via scripts/gran_fpga_count.py on --save-winner
# Modes INTERLEAVED within each seed (feedback_sweeps_always_interleave: round 1 = one
# result per mode). One controller at a time, per-run markers, skips done, NEVER kills.
# GATED: waits for Phase-2's /tmp/wnn_gran_multiseed_done.json before starting (user
# 16/07: let Phase-2 finish; production recipe, ~2 days).
#   bash scripts/run_gran_phase3.sh          # or via detach_launch.py
set -u
ROOT="/Users/lacg/wnn"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
export PYTHONPATH="${ROOT}/src/wnn:${PYTHONPATH:-}"
cd "$ROOT" || exit 1
SEEDS="${SEEDS:-31337002 31337003 31337004}"
MODES="${MODES:-BINARY QSR TERNARY QUAD_WEIGHTED PLN}"
GATE="/tmp/wnn_gran_multiseed_done.json"
log() { echo "[gran-phase3] $* $(date -u +%FT%TZ)"; }

# ---- gate: wait for Phase-2 to finish (poll; runner idles at zero cost) ----
while [ ! -f "$GATE" ]; do
	log "waiting for Phase-2 marker $GATE (sleep 300)"
	sleep 300
done
log "Phase-2 done — starting Phase 3 (seeds: $SEEDS)"

for BASE in $SEEDS; do
	for M in $MODES; do
		ml=$(echo "$M" | tr '[:upper:]' '[:lower:]')
		tag=$([ "$M" = "QUAD_WEIGHTED" ] && echo quad || echo "$ml")
		marker="/tmp/wnn_gran_phase3_${tag}_b${BASE}_done.json"
		if [ -f "$marker" ]; then
			log "phase3 $tag b$BASE: marker exists — skipping"
			continue
		fi
		out="logs/controller/phase3_${tag}_b${BASE}.out"
		winner="logs/controller/phase3_${tag}_b${BASE}_winner.yaml.gz"
		log "===== START full-pipeline $M base=$BASE -> $out ====="
		t0=$SECONDS
		# Recipe argv = gran_grid_winner_holdout._phased_argv VERBATIM (binqsr_v3 chain,
		# STEPS=2000 POP=50) + --save-winner. --skip-stages bits,connections leaves the
		# NEURONS + MEMORY GA stages ACTIVE — this IS the full production pipeline.
		/usr/bin/time -l "$VP" -u -m wnn.control.phased_ga \
			--grid-state-neurons 8 12 16 --grid-bits 24 30 --levels 16 \
			--skip-stages bits,connections --lamarckian --saturation-grow-gain 1.0 \
			--neurons-gens 60 --neurons-patience 3 --memory-gens 120 --memory-patience 2 \
			--pop 50 --num-eval-folds 5 --check-interval 2 --magnitude-aware-patience \
			--eval-episodes 100 --memory-eval-episodes 200 --steps 2000 \
			--max-state-neurons 24 --max-output-neurons 128 --tilt 5.0 \
			--fit-weight-err-sq 0.4 --fit-weight-stable 0.3 --fit-weight-jerk 0.2 --fit-weight-mono 0.1 \
			--report-seed 99990101 --report-episodes 100 --holdout-pop-sample 8 \
			--base-seed "$BASE" --runs 1 --teacher lqr \
			--memory-mode "$M" --save-winner "$winner" > "$out" 2>&1
		rc=$?
		dur=$((SECONDS - t0))
		# peak RSS: macOS time -l reports "maximum resident set size" in BYTES
		rss=$(grep -E "maximum resident set size" "$out" | awk '{print $1}' | tail -1)
		# FPGA sparse size from the saved final (memory-stage) winner
		if [ -f "$winner" ]; then
			"$VP" -u scripts/gran_fpga_count.py "$winner" >> "$out" 2>&1
		else
			log "phase3 $tag b$BASE: NO saved winner ($winner) — fpga count skipped"
		fi
		fpga=$(grep -E "^\[FPGA\]" "$out" | tail -1)
		held=$(grep -E "held-out" "$out" | tail -3 | tr '\n' ';')
		printf '{"tag":"%s","mode":"%s","base":%s,"rc":%s,"dur_s":%s,"peak_rss_bytes":%s,"fpga":"%s","held":"%s","done":"%s"}\n' \
			"$tag" "$M" "$BASE" "$rc" "$dur" "${rss:-null}" \
			"$(echo "$fpga" | tr -d '"' | sed 's/  */ /g')" \
			"$(echo "$held" | tr -d '"' | sed 's/  */ /g')" \
			"$(date -u +%FT%TZ)" > "$marker"
		log "===== END full-pipeline $M base=$BASE rc=$rc dur=${dur}s ====="
	done
done
> "/tmp/wnn_gran_phase3_done.json"
log "PHASE 3 DONE for seeds: $SEEDS"
