#!/usr/bin/env bash
# E5 residual-hybrid ABLATION: baseline ∈ {pd, stock_pid} × seed ∈ {20260609, 20260610}.
#
# Extends the single-cell proof (which showed PD-base → 100% @L2 for one seed) to
# 4 cells, testing two claims:
#   (a) robustness across seeds, and
#   (b) lift-scales-with-gap: PD-base (gap 84→99.8) should lift more than
#       stock_pid-base (gap 97→99.8).
# Each cell prints held-out BASE / HYBRID / PID+ stable-rates on disjoint seeds.
#
# Launch DETACHED via scripts/detach_launch.py so it survives CLI exit (PPID=1).
set -euo pipefail

ROOT="/Users/lacg/wnn"
VENV="/Users/lacg/wnn-venv"
LOG="${ROOT}/logs/controller/E5ResidualAblation_20260708.log"
MARKER="/tmp/wnn_e5residual_ablation_done.json"

export PYTHONPATH="${ROOT}/src/wnn"
export RAYON_NUM_THREADS=4            # leave cores for the live IDS worker (10 cores) + dashboard
export PYTHONUNBUFFERED=1

source "${VENV}/bin/activate"

# Memory safety: the live IDS worker's 96b RE-EVAL phase spikes to 25-31 GB and
# jetsam killed a prior concurrent ablation run (detached jobs sit in a lower
# jetsam band). Wait until IDS is past re-eval — i.e. an OFFSPRING gen line
# ("[ArchitectureGA] Gen 0XX/250") appears BEYOND the current log position, at
# which point the batch-sizing fix bounds IDS to ~14 GB and there's headroom.
# Up to ~3h; if IDS isn't running, proceed immediately.
IDS_LOG="/private/tmp/wnn_worker.log"
if pgrep -f "wnn.ram.experiments.flow_runner" >/dev/null 2>&1 && [ -f "$IDS_LOG" ]; then
	START_LN=$(wc -l < "$IDS_LOG")
	echo "[ablation] waiting for IDS re-eval to finish (new Gen line beyond ln $START_LN)..." | tee -a "$LOG"
	for _ in $(seq 1 360); do
		if awk -v s="$START_LN" 'NR>s && /\[ArchitectureGA\] Gen 0[0-9][0-9]\/250/{f=1} END{exit !f}' "$IDS_LOG"; then
			echo "[ablation] IDS offspring gen detected — headroom available, starting." | tee -a "$LOG"
			break
		fi
		sleep 30
	done
fi

BASELINES=(pd stock_pid)
SEEDS=(20260609 20260610)

echo "[ablation] START $(date -u +%Y-%m-%dT%H:%M:%SZ)  cells=$(( ${#BASELINES[@]} * ${#SEEDS[@]} ))  RAYON=${RAYON_NUM_THREADS}" | tee -a "$LOG"

declare -a RESULTS
rc=0
for bl in "${BASELINES[@]}"; do
	for sd in "${SEEDS[@]}"; do
		echo "" | tee -a "$LOG"
		echo "[ablation] ===== CELL baseline=${bl} seed=${sd} =====" | tee -a "$LOG"
		# Capture this cell's VERDICT line into the summary; keep going on failure.
		if python -u "${ROOT}/scripts/e5_residual_proof.py" "${sd}" "${bl}" 2>&1 | tee -a "$LOG"; then
			v=$(grep "VERDICT" -A1 "$LOG" | tail -1)
			RESULTS+=("baseline=${bl} seed=${sd} :: ${v}")
		else
			rc=1
			RESULTS+=("baseline=${bl} seed=${sd} :: FAILED (see log)")
		fi
	done
done

echo "" | tee -a "$LOG"
echo "[ablation] ===== SUMMARY (4 cells) =====" | tee -a "$LOG"
for r in "${RESULTS[@]}"; do echo "[ablation] ${r}" | tee -a "$LOG"; done
echo "[ablation] END $(date -u +%Y-%m-%dT%H:%M:%SZ)  rc=${rc}" | tee -a "$LOG"

# Done-marker for the verdict waiter.
printf '{"done":"%s","rc":%d,"log":"%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${rc}" "${LOG}" > "${MARKER}"
exit "${rc}"
