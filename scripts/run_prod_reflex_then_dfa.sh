#!/usr/bin/env bash
# Production A/B chain (20/07/2026), SEQUENTIAL — one controller at a time.
#
# PHASE 1 — single-layer ("reflex", sn=0) A/B: DAGGER vs pure BC.
#   Fast by construction (~200x faster gens than the two-layer DFA), so it runs
#   first. BC uses the SAME patience config as DAGGER (Luiz 20/07): each arm
#   stops when IT thinks it converged — no artificial budget handicap either way.
#
# PHASE 2 — two-layer WNN DFA A/B (yaw-blind vs +obs_yaw_err) WITH
#   WNN_STATE_SPLIT=1. This is the apples-to-apples rerun of the 19/07 A/B,
#   which accidentally ran with the state-splitting trainer DISABLED (env never
#   exported; Rust default is false) — so its "state 0/0" result measured a
#   substrate with the state-PLANTING mechanism off, not a GA that declined to
#   use state. Phase 2 answers the real question on a substrate that FLIES
#   (arm A held 91% stable under L2). Check the FPGA state-cell count.
#
# Both phases: C10 weights, folds 5, held-out --report-seed, BINARY, lqr, L2,
# --skip-stages bits,connections (CONNECTIONS is hit-or-miss / local-optima
# prone — Luiz 20/07; both smokes showed it collapsing 80% -> 0% stable).
#   bash scripts/run_prod_reflex_then_dfa.sh    # via detach_launch.py
set -u
ROOT="/Users/lacg/wnn"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
export PYTHONPATH="${ROOT}/src/wnn:${PYTHONPATH:-}"
# The known-good controller env (probe_gran_mode.sh). WNN_STATE_SPLIT is set
# PER-PHASE below — phase 1 (sn=0) has no state layer to split.
# GPU_EVAL=0 keeps the controller off the GPU while the IDS worker owns it
# (IDS is priority). Since 20/07/2026 the CPU scorer computes ALL 13 metrics
# (cpu_score.rs), so this no longer costs the transient/display ones — before
# that fix it silently zeroed steady/rise/settle/ITAE/IAE/ISE.
# RAYON_NUM_THREADS stays 10 for BOTH phases (corrected 20/07/2026).
# An earlier note here claimed the phase-2 OOM was "concurrency-bound, not
# pop-bound" and capped phase 2 to 4 threads. That was WRONG, and the evidence
# against it had already been observed and explained away: capping 10->4 barely
# moved the peak. Thread-scoped buffers (split record sets + trajectory clones)
# are ~8% of the peak. The peak is POPULATION-scoped — it was the per-fold
# export of every genome's cells to Python triples (~2.4 GB at pop=50) plus
# _eval_batch_size falling back to "whole population in one batch" at gen 0,
# when no prior generation had written cells back for it to measure.
# Both are fixed in ABI 15 (commit 3500fbf5): the K-fold chain never leaves
# Rust, and the split-aware floor sizes the chunk to ~7 concurrent genomes.
# Lowering threads here would only cost wall-clock and buy ~nothing.
export WNN_RUST_DAGGER=1 RAYON_NUM_THREADS=10 WNN_CONTROLLER_GPU_EVAL=0
cd "$ROOT" || exit 1
BASE="${BASE:-31337002}"
log() { echo "[prod-ab] $* $(date -u +%FT%TZ)"; }

# Shared production numbers (run_yawab_L2.sh recipe verbatim).
PROD="--levels 16 --skip-stages bits,connections --lamarckian --saturation-grow-gain 1.0 \
--neurons-gens 60 --neurons-patience 3 --memory-gens 120 --memory-patience 2 \
--pop 50 --num-eval-folds 5 --check-interval 2 --magnitude-aware-patience \
--eval-episodes 100 --memory-eval-episodes 200 --steps 2000 --tilt 5.0 \
--fit-weight-err-sq 0.4 --fit-weight-stable 0.3 --fit-weight-jerk 0.2 --fit-weight-mono 0.1 \
--report-seed 99990101 --report-episodes 100 --holdout-pop-sample 8 \
--base-seed $BASE --runs 1 --teacher lqr --disturbance L2 --memory-mode BINARY"

run_arm() {
	tag="$1"; shift
	marker="/tmp/wnn_prodab_${tag}_b${BASE}_done.json"
	if [ -f "$marker" ]; then log "$tag: marker exists — skipping"; return; fi
	out="logs/controller/prodab_${tag}_b${BASE}.out"
	winner="logs/controller/prodab_${tag}_b${BASE}_winner.yaml.gz"
	log "===== START $tag -> $out ====="
	t0=$SECONDS
	/usr/bin/time -l "$VP" -u -m wnn.control.phased_ga $PROD "$@" \
		--save-winner "$winner" > "$out" 2>&1
	rc=$?; dur=$((SECONDS - t0))
	# rc=143 (SIGTERM) = the mem-watchdog gracefully PAUSED this arm; it dumped for
	# resume and expects the chain to HOLD, not advance (20/07/2026: the first run
	# wrote empty done-markers on 143 and marched on, losing both phase-2 arms AND
	# poisoning any rerun, since an existing marker means "skip"). Treat it as
	# paused: write NO marker, wait for memory to recover, then RETRY this arm.
	if [ "$rc" = "143" ] || [ "$rc" = "137" ]; then
		log "$tag: rc=$rc (watchdog pause/kill) — NO marker; waiting for memory, will retry"
		tries="${TRIES:-0}"
		if [ "$tries" -ge 2 ]; then
			log "$tag: retried $tries times and keeps getting killed — ABORTING chain (needs a fix, not a retry)"
			exit 3
		fi
		# Wait until the box is genuinely comfortable (>= 25 GB avail, 3 calm ticks).
		calm=0
		while [ "$calm" -lt 3 ]; do
			sleep 60
			availgb=$(vm_stat | awk '/free|inactive|speculative|purgeable/ {gsub("\\.","");s+=$NF} END {printf "%.0f", s*16384/1073741824}')
			if [ "${availgb:-0}" -ge 25 ]; then calm=$((calm+1)); else calm=0; fi
			log "$tag: waiting to retry (avail=${availgb}GB, calm=$calm/3)"
		done
		TRIES=$((tries+1)) run_arm "$tag" "$@"
		return
	fi
	rss=$(grep -E "maximum resident set size" "$out" | awk '{print $1}' | tail -1)
	[ -f "$winner" ] && "$VP" -u scripts/gran_fpga_count.py "$winner" >> "$out" 2>&1
	fpga=$(grep -E "^\[FPGA\]" "$out" | tail -1)
	held=$(grep -E "RESULT — during-search winner" "$out" | tail -1)
	printf '{"tag":"%s","base":%s,"rc":%s,"dur_s":%s,"peak_rss_bytes":%s,"fpga":"%s","held":"%s","done":"%s"}\n' \
		"$tag" "$BASE" "$rc" "$dur" "${rss:-null}" \
		"$(echo "$fpga" | tr -d '"' | sed 's/  */ /g')" \
		"$(echo "$held" | tr -d '"' | sed 's/  */ /g')" \
		"$(date -u +%FT%TZ)" > "$marker"
	log "===== END $tag rc=$rc dur=${dur}s ====="
}

# ---------------- PHASE 1: single-layer (sn=0), DAGGER vs BC ----------------
# No state layer ⇒ split is irrelevant (split_train_loop no-ops at sn=0 anyway).
log "PHASE 1: single-layer reflex A/B (DAGGER vs BC), split N/A"
SN0="--grid-state-neurons 0 --grid-bits 24 30 --max-state-neurons 0 --max-output-neurons 128"
run_arm reflex_dagger $SN0
run_arm reflex_bc     $SN0 --expert-drives --rg-rounds 1
> "/tmp/wnn_prodab_phase1_done.json"
log "PHASE 1 DONE"

# ---------------- PHASE 2: two-layer DFA A/B, WNN_STATE_SPLIT=1 -------------
# THE fix for the 19/07 confound. Same argv as run_yawab_L2.sh, split ON.
log "PHASE 2: WNN DFA A/B with WNN_STATE_SPLIT=1 (state-cell question)"
export WNN_STATE_SPLIT=1
# No per-phase thread cap: memory is bounded by the ABI-15 split-aware chunker
# (evaluator._split_cell_floor), not by rayon width. See the RAYON note at top.
log "PHASE 2: RAYON_NUM_THREADS=$RAYON_NUM_THREADS (memory bounded by the eval chunker, not threads)"
DFA="--grid-state-neurons 8 12 16 --grid-bits 24 30 --max-state-neurons 24 --max-output-neurons 128"
run_arm dfa_split_blind $DFA
run_arm dfa_split_yaw   $DFA --obs-yaw-err
> "/tmp/wnn_prodab_phase2_done.json"
log "PHASE 2 DONE"

> "/tmp/wnn_prodab_all_done.json"
log "ALL DONE (base $BASE)"
