#!/usr/bin/env bash
# WNN 1-layer vs DFA study — the 2x2x2x5 factorial, production scale, one arm at
# a time behind the IDS worker.
#
#   substrate    : 1layer (sn=0, reflex)   |  dfa (sn=8-16, two-layer)
#   feature      : 9feat (base)            |  10feat (--obs-yaw-err)
#   memory mode  : BINARY                  |  QUAD (QUAD_WEIGHTED)
#   seed         : 5 training base-seeds
# = 40 runs. Each: grid -> GA-NEURONS -> GA-MEMORY (Lamarckian, SPLIT=0 / BPTT),
# disturbance L2D, held-out on the FIXED report-seed so all 40 are comparable on
# the same test set. Trained by DAGGER against the LQR teacher.
#
# SPLIT=0 keeps cells ~3k/genome (the yawab reference), so this whole campaign is
# in the SAFE memory regime — no 63GB blowup. A footprint sampler + IDS guard are
# armed by the LAUNCHER (not here); this script only sequences the runs and, on a
# watchdog SIGTERM/kill (rc 143/137), writes NO marker and retries after memory
# recovers — exactly the run_prod_reflex_then_dfa.sh pause protocol.
#
# Resumable: an existing per-run marker means "done, skip". Delete a marker to
# force a re-run.
set -u
ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
export PYTHONPATH="${ROOT}/src/wnn:${PYTHONPATH:-}"
export WNN_RUST_DAGGER=1 RAYON_NUM_THREADS=10 WNN_CONTROLLER_GPU_EVAL=0
# SPLIT=0 (BPTT): the state layer trains via bptt_train_window, NOT the
# conflict-driven split trainer (parked — see the 21/07 error-gate finding).
unset WNN_STATE_SPLIT
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
[ -x "$VP" ] || VP="python"

OUTDIR="${OUTDIR:-logs/controller/dfa1l}"
MARKDIR="${MARKDIR:-/tmp/wnn_dfa1l}"
mkdir -p "$OUTDIR" "$MARKDIR"

SEEDS=(${SEEDS:-31337002 31337003 31337004 31337005 31337006})
REPORT_SEED="${REPORT_SEED:-99990101}"

# Production recipe (yawab / run_prod verbatim), minus the per-cell factors.
COMMON="--levels 16 --skip-stages bits,connections --lamarckian \
--saturation-grow-gain 1.0 --neurons-gens 60 --neurons-patience 3 \
--memory-gens 120 --memory-patience 2 --pop 50 --num-eval-folds 5 \
--check-interval 2 --magnitude-aware-patience --eval-episodes 100 \
--memory-eval-episodes 200 --steps 2000 --tilt 5.0 \
--fit-weight-err-sq 0.4 --fit-weight-stable 0.3 --fit-weight-jerk 0.2 \
--fit-weight-mono 0.1 --report-seed ${REPORT_SEED} --report-episodes 100 \
--holdout-pop-sample 8 --grid-bits 24 30 --max-output-neurons 128 \
--runs 1 --teacher lqr --disturbance L2D"

log() { echo "[dfa1l] $* $(date -u +%FT%TZ)"; }

# One factorial cell. Args: substrate feature memmode seed
run_cell() {
	local sub="$1" feat="$2" mode="$3" seed="$4"
	local tag="${sub}_${feat}_${mode}_s${seed}"
	local marker="${MARKDIR}/${tag}.json"
	local out="${OUTDIR}/${tag}.out"
	local winner="${OUTDIR}/${tag}_winner.yaml.gz"
	if [ -f "$marker" ]; then log "$tag: marker exists — skip"; return; fi

	# substrate → state-neuron flags
	local sub_flags
	if [ "$sub" = "1layer" ]; then
		sub_flags="--grid-state-neurons 0 --max-state-neurons 0"
	else
		sub_flags="--grid-state-neurons 8 12 16 --max-state-neurons 24"
	fi
	# feature → yaw-err flag
	local feat_flags=""
	[ "$feat" = "10feat" ] && feat_flags="--obs-yaw-err"
	# memory mode
	local mm="QUAD_WEIGHTED"; [ "$mode" = "BINARY" ] && mm="BINARY"

	log "===== START $tag ====="
	local t0=$SECONDS
	/usr/bin/time -l "$VP" -u -m wnn.control.phased_ga $COMMON \
		$sub_flags $feat_flags --memory-mode "$mm" --base-seed "$seed" \
		--save-winner "$winner" > "$out" 2>&1
	local rc=$? dur=$((SECONDS - t0))

	# Watchdog pause/kill → hold the chain (no marker), wait for memory, retry.
	if [ "$rc" = "143" ] || [ "$rc" = "137" ]; then
		log "$tag: rc=$rc (watchdog) — NO marker; waiting for memory, will retry"
		local tries="${TRIES:-0}"
		if [ "$tries" -ge 2 ]; then
			log "$tag: killed ${tries}x — SKIPPING this cell (needs a fix, not a retry)"
			return
		fi
		local calm=0
		while [ "$calm" -lt 3 ]; do
			sleep 60
			local av
			av=$(vm_stat | awk '/free|inactive|speculative|purgeable/ {gsub("\\.","");s+=$NF} END {printf "%.0f", s*16384/1073741824}')
			if [ "${av:-0}" -ge 25 ]; then calm=$((calm+1)); else calm=0; fi
			log "$tag: waiting to retry (avail=${av}GB, calm=${calm}/3)"
		done
		TRIES=$((tries+1)) run_cell "$sub" "$feat" "$mode" "$seed"
		return
	fi

	# Extract the two held-out stage triples (NEURONS + MEMORY) + cells + FPGA.
	local rss held_n held_m fpga
	rss=$(grep -E "maximum resident set size" "$out" | awk '{print $1}' | tail -1)
	held_n=$(grep -E "RESULT — during-search winner" "$out" | sed -n '1p')
	held_m=$(grep -E "RESULT — during-search winner" "$out" | sed -n '2p')
	[ -f "$winner" ] && "$VP" -u scripts/gran_fpga_count.py "$winner" >> "$out" 2>&1
	fpga=$(grep -E "^\[FPGA\]" "$out" | tail -1)
	local cells
	cells=$(grep -oE "cells\[[0-9-]+ Σ[0-9]+k μ[0-9]+k\]" "$out" | tail -1)

	printf '{"tag":"%s","substrate":"%s","feature":"%s","mode":"%s","seed":%s,"rc":%s,"dur_s":%s,"peak_rss_bytes":%s,"cells":"%s","fpga":"%s","held_neurons":"%s","held_memory":"%s","done":"%s"}\n' \
		"$tag" "$sub" "$feat" "$mode" "$seed" "$rc" "$dur" "${rss:-null}" \
		"$cells" \
		"$(echo "$fpga" | tr -d '"' | sed 's/  */ /g')" \
		"$(echo "$held_n" | tr -d '"' | sed 's/  */ /g')" \
		"$(echo "$held_m" | tr -d '"' | sed 's/  */ /g')" \
		"$(date -u +%FT%TZ)" > "$marker"
	log "===== END $tag rc=$rc dur=${dur}s ====="
}

# ---- baselines (once) -------------------------------------------------------
BL="${MARKDIR}/baselines.json"
if [ ! -f "$BL" ]; then
	log "computing classical baselines (once) → $BL"
	"$VP" -u scripts/compute_baselines.py --disturbance L2D --tilt 5.0 \
		--report-seed "$REPORT_SEED" --report-episodes 100 --steps 2000 --out "$BL"
fi

# ---- the 40-run sweep, INTERLEAVED so early culling / partial reads are useful.
# Round r fires one of each (substrate x feature x mode) at seed[r] — so after
# round 1 you already have all 8 cells at one seed, not 5 seeds of one cell.
for seed in "${SEEDS[@]}"; do
	for sub in 1layer dfa; do
		for feat in 9feat 10feat; do
			for mode in BINARY QUAD; do
				run_cell "$sub" "$feat" "$mode" "$seed"
			done
		done
	done
done

> "${MARKDIR}/ALL_DONE.marker"
log "ALL 40 RUNS DONE"
