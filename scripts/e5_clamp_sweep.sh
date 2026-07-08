#!/usr/bin/env bash
# LEARN-THE-CLAMP (scalar sweep). How LITTLE residual authority holds the @L2
# ceiling? The residual_clamp bounds the WNN's per-motor correction AND shapes
# the DAGGER label clamp(PID+ − baseline), so each clamp is a fresh retrain.
# Sweeps clamp ∈ {0.1,0.15,0.2,0.3,0.4} on the PD baseline (biggest gap) and
# reports held-out BASE/HYBRID/PID+ (both Python + collapsed Rust path) per clamp.
# Minimal clamp still clearing BASE → the FPGA-friendliest authority.
set -euo pipefail

ROOT="/Users/lacg/wnn"
VENV="/Users/lacg/wnn-venv"
LOG="${ROOT}/logs/controller/E5ClampSweep_20260708.log"
MARKER="/tmp/wnn_e5clampsweep_done.json"

export PYTHONPATH="${ROOT}/src/wnn"
export RAYON_NUM_THREADS=4            # coexist with the live IDS worker
export PYTHONUNBUFFERED=1
source "${VENV}/bin/activate"

SEED=20260609
BASELINE=pd
# LOW range: the coarse sweep {0.1..0.4} was identical (residual never binds ⇒ the
# WNN needs <0.1 authority). Sweep low to find the BINDING point = minimal authority.
CLAMPS=(0.01 0.02 0.03 0.05 0.08 0.10)

echo "[clampsweep] START $(date -u +%Y-%m-%dT%H:%M:%SZ)  baseline=${BASELINE} seed=${SEED}  clamps=${CLAMPS[*]}" | tee -a "$LOG"

declare -a ROWS
rc=0
for clamp in "${CLAMPS[@]}"; do
	echo "" | tee -a "$LOG"
	echo "[clampsweep] ===== clamp=${clamp} =====" | tee -a "$LOG"
	if python -u "${ROOT}/scripts/e5_residual_proof.py" "${SEED}" "${BASELINE}" "${clamp}" 2>&1 | tee -a "$LOG"; then
		# Pull the collapsed-Rust-path verdict for this clamp.
		rust=$(grep "\[rust\]" "$LOG" | tail -1)
		ROWS+=("clamp=${clamp} :: ${rust}")
	else
		rc=1
		ROWS+=("clamp=${clamp} :: FAILED")
	fi
done

echo "" | tee -a "$LOG"
echo "[clampsweep] ===== SUMMARY (minimal authority holding the ceiling) =====" | tee -a "$LOG"
for r in "${ROWS[@]}"; do echo "[clampsweep] ${r}" | tee -a "$LOG"; done
echo "[clampsweep] END $(date -u +%Y-%m-%dT%H:%M:%SZ)  rc=${rc}" | tee -a "$LOG"
printf '{"done":"%s","rc":%d,"log":"%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${rc}" "${LOG}" > "${MARKER}"
exit "${rc}"
