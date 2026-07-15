#!/bin/bash
# Granularity ablation — FULL 5-arm rerun on the fixed+capped recipe (14/07/2026 PM).
# Extends rerun_gran_all3_capped.sh (QUAD/TERNARY/BINARY) with the two stochastic
# decode arms QSR (mode 4, stochastic QUAD read) and PLN (mode 5, stochastic TERNARY
# read) — the Part 5 decode (commit 3c26d976, controller ABI 12) plus the Python
# wiring (evaluator.MEMORY_MODES + phased_ga --memory-mode choices + ga_memory 4-state
# classification). Each arm is its own fresh `python -m wnn.control.phased_ga`, so the
# QSR/PLN arms pick up the Python wiring at spawn time (no wheel rebuild — Rust is done).
#
# ONE arm at a time (run_arm blocks on the python call). guard() waits for ≥12GB REAL
# free (vm_stat) before each arm. Runs UNDER scripts/controller_mem_watchdog.sh (the
# hard SIGKILL backstop at 5GB real-free). Fresh STAMP so nothing collides with pass-1.
# Marker: /tmp/wnn_gran_5arm_done.json ; per-arm: /tmp/wnn_gran_arm_<tag>_done.json.
set -u

PROJ="/Users/lacg/wnn"
VENV="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv"
# STEPS/POP overridable (env) so we can run a lighter recipe alongside a heavy IDS
# drain — steps=1000 pop=30 roughly halves per-arm footprint + wall vs 2000/50 (we
# know a fitness knee improves toward 2000, so this is a coexistence compromise, not
# the final recipe). STAMP defaults follow the recipe so dirs never collide.
STEPS="${STEPS:-2000}"
POP="${POP:-50}"
# MEM_EVAL_EP sizes the Memory-stage address universe (rollout episodes → distinct
# addresses recorded). It is the dominant Memory-stage RAM driver: at 200 the stage
# hit ~9.7GB RSS and hard-breached the floor alongside IDS. Lower it (e.g. 80) to fit.
MEM_EVAL_EP="${MEM_EVAL_EP:-200}"
STAMP="${STAMP:-20260714_s${STEPS}p${POP}}"
LOGDIR="$PROJ/logs/controller"

export PYTHONPATH="$PROJ/src:${PYTHONPATH:-}"
export WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=10 WNN_CONTROLLER_GPU_EVAL=0
unset CONDA_PREFIX || true
# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$PROJ"

# GUARD_GB: AVAILABLE memory (free+purgeable+speculative+inactive — the same correct
# metric as watchdog v4, NOT strict free which counts ~27GB reclaimable IDS file cache
# as used) needed before (re)launching an arm. Default 15GB of available headroom.
GUARD_GB="${GUARD_GB:-15}"
log() { echo "[gran-5arm] $1 $(date -u +%FT%TZ)"; }
avail_gb() { vm_stat 2>/dev/null | awk '
	/Pages free/{f=$3} /Pages inactive/{i=$3} /Pages speculative/{s=$3} /Pages purgeable/{p=$3}
	END{printf "%.1f",(f+i+s+p)*16384/1073741824}'; }
guard() {   # wait for ≥GUARD_GB AVAILABLE memory (reclaimable headroom), up to 60 min
	local tag="$1"
	for _ in $(seq 1 60); do
		local a; a=$(avail_gb)
		if [ "$(echo "${a:-0} >= $GUARD_GB" | bc 2>/dev/null)" = "1" ]; then log "$tag: available=${a}GB — launching"; return 0; fi
		log "$tag: waiting for memory (available=${a:-?}GB, need ≥${GUARD_GB}GB)…"; sleep 60
	done
	log "$tag: guard timed out (available=$(avail_gb)GB) — proceeding"
}

launch_arm() {  # $1 = mode, $2 = dir, $3 = optional resume args. Blocks on phased_ga.
	local mode="$1" dir="$2" resume="$3"
	# shellcheck disable=SC2086
	python -u -m wnn.control.phased_ga \
		--grid-state-neurons 8 12 16 --grid-bits 24 30 --levels 16 \
		--skip-stages bits,connections --lamarckian --saturation-grow-gain 1.0 \
		--neurons-gens 60 --neurons-patience 3 --memory-gens 120 --memory-patience 2 \
		--pop "$POP" --num-eval-folds 5 --check-interval 2 --magnitude-aware-patience \
		--eval-episodes 100 --memory-eval-episodes "$MEM_EVAL_EP" --steps "$STEPS" --max-state-neurons 24 --max-output-neurons 128 --tilt 5.0 \
		--fit-weight-err-sq 0.4 --fit-weight-stable 0.3 --fit-weight-jerk 0.2 --fit-weight-mono 0.1 \
		--report-seed 99990101 --report-episodes 100 --holdout-pop-sample 8 \
		--base-seed 31337002 --runs 1 --teacher lqr \
		--memory-mode "$mode" \
		--save-winner "$dir/winner.yaml.gz" --save-stage-checkpoints "$dir" \
		$resume \
		>> "$dir/run.out" 2>&1
}

run_arm() {  # $1 = mode, $2 = tag. Resume-looping: the watchdog can SIGTERM-PAUSE this
	         # arm (graceful emergency dump); we then wait for RAM to recover and RESUME
	         # the SAME arm from its dump, so no evolved steps are lost. Only a clean
	         # completion (or exhausting the resume budget) advances to the next arm.
	local mode="$1" tag="$2"
	local dir="$LOGDIR/c10_gran_${tag}_$STAMP/seed_base31337002_SCREENING_p32"
	mkdir -p "$dir"
	# Resumable across relaunches: if this arm already completed cleanly (its run.out
	# has the final banner), skip it — so a relaunch after a mid-chain failure picks up
	# at the first UNfinished arm instead of redoing finished ones (e.g. QUAD done →
	# start at TERNARY). The WNN_RESUME_FIRST_DUMP handoff still overrides for a paused arm.
	if [ -z "${WNN_RESUME_FIRST_DUMP:-}" ] && grep -q "PHASED-GA RESULT" "$dir/run.out" 2>/dev/null; then
		log "$tag: already COMPLETE (PHASED-GA RESULT present) — skipping"
		return 0
	fi
	local tries=0 max_tries=20 rc resume=""
	# One-shot HANDOFF resume: when this chain replaces a running one, the caller can
	# pass WNN_RESUME_FIRST_DUMP (+ WNN_RESUME_FIRST_TAG, default quad) so the first
	# arm continues from the paused run's dump instead of starting fresh — zero lost
	# steps across a chain redeploy. Skips the dump-clear/truncate so that dump survives.
	if [ -n "${WNN_RESUME_FIRST_DUMP:-}" ] && [ "$tag" = "${WNN_RESUME_FIRST_TAG:-quad}" ]; then
		resume="--resume-from-emergency $WNN_RESUME_FIRST_DUMP --resume-mode same"
		log "$tag: HANDOFF-resume from $WNN_RESUME_FIRST_DUMP"
		export WNN_RESUME_FIRST_DUMP=""   # one-shot
	else
		rm -f "$dir"/emergency_stage*        # clear stale dumps so any post-exit dump is THIS arm's
		                                     # (save_checkpoint writes .yaml.gz, not the .pkl the path implies)
		: > "$dir/run.out"                  # truncate once; launch_arm appends across resumes so the
		                                    # "PHASED-GA RESULT" completion check can't match a stale banner
	fi
	while :; do
		tries=$((tries + 1))
		guard "$tag"   # wait for >=12GB real-free before (re)launching
		if [ -n "$resume" ]; then
			log "===== RESUME arm mode=$mode (try $tries) from $resume -> $dir/run.out ====="
		else
			log "===== START arm mode=$mode (try $tries) -> $dir/run.out ====="
		fi
		launch_arm "$mode" "$dir" "$resume"
		rc=$?
		# Clean completion? phased_ga prints the final banner only on a full run.
		if grep -q "PHASED-GA RESULT" "$dir/run.out" 2>/dev/null; then
			log "===== END arm mode=$mode COMPLETE rc=$rc (try $tries) ====="
			break
		fi
		# Not complete. A fresh emergency dump => watchdog PAUSE (or crash-save): resume it.
		local dump; dump=$(ls -t "$dir"/emergency_stage* 2>/dev/null | head -1)
		if [ -n "$dump" ] && [ "$tries" -lt "$max_tries" ]; then
			resume="--resume-from-emergency $dump --resume-mode same"
			log "===== PAUSED arm mode=$mode rc=$rc — dump=$dump; will RESUME when RAM recovers ====="
			continue
		fi
		log "===== END arm mode=$mode rc=$rc — no dump or resume budget spent ($tries/$max_tries); advancing ====="
		break
	done
	echo "{\"done\": \"$(date -u +%FT%TZ)\", \"mode\": \"$mode\", \"tag\": \"$tag\", \"rc\": $rc, \"tries\": $tries, \"stamp\": \"$STAMP\"}" \
		> "/tmp/wnn_gran_arm_${tag}_done.json"
}

run_arm QUAD_WEIGHTED quad
run_arm TERNARY       ternary
run_arm BINARY        binary
run_arm QSR           qsr
run_arm PLN           pln

echo "{\"done\": \"$(date -u +%FT%TZ)\", \"arms\": [\"quad\", \"ternary\", \"binary\", \"qsr\", \"pln\"], \"stamp\": \"$STAMP\", \"recipe\": \"steps${STEPS}_pop${POP}_sn24_on128\", \"teacher\": \"lqr\", \"seed\": 31337002}" \
	> /tmp/wnn_gran_5arm_done.json
log "ALL 5 GRANULARITY ARMS DONE (capped recipe)"
