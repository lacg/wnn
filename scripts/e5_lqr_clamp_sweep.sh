#!/usr/bin/env bash
# LQR-teacher clamp sweep: authority-limited or capacity-limited? The PD+WNN-imit-LQR
# hybrid reached only ~⅓ of the LQR's advantage (hybrid 3.16° vs LQR 1.57° vs PD 4.04°).
# The LQR is far more aggressive than PD, so clamp(LQR−PD) likely exceeds the 0.4 clamp
# → clipped residual. Sweep the clamp HIGH {0.4,0.6,0.8,1.0} on expert=lqr @L2:
#   err drops toward LQR (1.57°) as clamp rises → AUTHORITY-limited (raise the clamp);
#   err plateaus                                 → CAPACITY-limited (need a bigger WNN).
set -euo pipefail

ROOT="/Users/lacg/wnn"
VENV="/Users/lacg/wnn-venv"
LOG="${ROOT}/logs/controller/E5LQRClampSweep_20260708.log"
MARKER="/tmp/wnn_e5lqrclamp_done.json"

export PYTHONPATH="${ROOT}/src/wnn"
export RAYON_NUM_THREADS=3
export PYTHONUNBUFFERED=1
source "${VENV}/bin/activate"

SEED=20260609; BASELINE=pd; LEVEL=L2; EXPERT=lqr
CLAMPS=(0.4 0.6 0.8 1.0)

echo "[lqrclamp] START $(date -u +%Y-%m-%dT%H:%M:%SZ)  expert=${EXPERT} baseline=${BASELINE} ${LEVEL}  clamps=${CLAMPS[*]}" | tee -a "$LOG"

declare -a ROWS; rc=0
for clamp in "${CLAMPS[@]}"; do
	# Memory guard vs the live IDS worker (a prior concurrent run was jetsam-killed).
	for _ in $(seq 1 60); do
		free=$(memory_pressure 2>/dev/null | grep -i 'free perc' | grep -oE '[0-9]+' | head -1)
		[ "${free:-0}" -ge 40 ] && break
		sleep 30
	done
	echo "" | tee -a "$LOG"
	echo "[lqrclamp] ===== clamp=${clamp} (free=${free:-?}%) =====" | tee -a "$LOG"
	if python -u "${ROOT}/scripts/e5_residual_proof.py" "${SEED}" "${BASELINE}" "${clamp}" "${LEVEL}" "${EXPERT}" 2>&1 | tee -a "$LOG"; then
		hyb=$(grep "HYBRID (base+residual)" "$LOG" | tail -1)
		ROWS+=("clamp=${clamp} :: ${hyb}")
	else
		rc=1; ROWS+=("clamp=${clamp} :: FAILED")
	fi
done

echo "" | tee -a "$LOG"
echo "[lqrclamp] ===== SUMMARY (HYBRID err/ITAE by clamp; LQR ruler = 1.57°/0.047) =====" | tee -a "$LOG"
for r in "${ROWS[@]}"; do echo "[lqrclamp] ${r}" | tee -a "$LOG"; done
echo "[lqrclamp] END $(date -u +%Y-%m-%dT%H:%M:%SZ)  rc=${rc}" | tee -a "$LOG"
printf '{"done":"%s","rc":%d,"log":"%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${rc}" "${LOG}" > "${MARKER}"
exit "${rc}"
