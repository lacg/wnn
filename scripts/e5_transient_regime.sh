#!/usr/bin/env bash
# METRIC-DRIVEN experiment: "does the hybrid REACT FASTER?" The transient metrics
# (rise/settle) only discriminate when controllers settle inside the 2° band — at
# L2 the ~3.75° floor pins them at the sentinel. So sweep the disturbance regime
# {OFF, L1, L2} on the PD baseline and read rise / settle2° / ITAE for PD vs HYBRID
# vs PID+. Faster rise/settle for HYBRID vs PD (and vs PID+) = the "faster reaction"
# claim, measured natively via the collapsed Rust path.
set -euo pipefail

ROOT="/Users/lacg/wnn"
VENV="/Users/lacg/wnn-venv"
LOG="${ROOT}/logs/controller/E5TransientRegime_20260708.log"
MARKER="/tmp/wnn_e5transient_done.json"

export PYTHONPATH="${ROOT}/src/wnn"
export RAYON_NUM_THREADS=4
export PYTHONUNBUFFERED=1
source "${VENV}/bin/activate"

SEED=20260609
BASELINE=pd
CLAMP=0.4
LEVELS=(OFF L1 L2)

echo "[transient] START $(date -u +%Y-%m-%dT%H:%M:%SZ)  baseline=${BASELINE} seed=${SEED}  levels=${LEVELS[*]}" | tee -a "$LOG"

declare -a ROWS
rc=0
for level in "${LEVELS[@]}"; do
	# Memory guard: don't launch a retrain during an IDS memory spike (a prior
	# concurrent run was jetsam-killed). Wait (≤30 min) for ≥35% free.
	for _ in $(seq 1 60); do
		free=$(memory_pressure 2>/dev/null | grep -i 'free perc' | grep -oE '[0-9]+' | head -1)
		[ "${free:-0}" -ge 35 ] && break
		sleep 30
	done
	echo "" | tee -a "$LOG"
	echo "[transient] ===== dist=${level} (free=${free:-?}%) =====" | tee -a "$LOG"
	if python -u "${ROOT}/scripts/e5_residual_proof.py" "${SEED}" "${BASELINE}" "${CLAMP}" "${level}" 2>&1 | tee -a "$LOG"; then
		# Capture the three GPU rows (rise/settle/ITAE) for this regime.
		rows=$(grep -E "BASE \(gpu|PID\+ \(gpu|HYBRID \(gpu" "$LOG" | tail -3)
		ROWS+=("--- dist=${level} ---"); while IFS= read -r ln; do ROWS+=("$ln"); done <<< "$rows"
	else
		rc=1; ROWS+=("dist=${level} :: FAILED")
	fi
done

echo "" | tee -a "$LOG"
echo "[transient] ===== SUMMARY (rise/settle/ITAE by regime) =====" | tee -a "$LOG"
for r in "${ROWS[@]}"; do echo "[transient] ${r}" | tee -a "$LOG"; done
echo "[transient] END $(date -u +%Y-%m-%dT%H:%M:%SZ)  rc=${rc}" | tee -a "$LOG"
printf '{"done":"%s","rc":%d,"log":"%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${rc}" "${LOG}" > "${MARKER}"
exit "${rc}"
