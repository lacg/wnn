#!/usr/bin/env bash
# LQR-teacher CAPACITY sweep (Task #1: close the WNN→LQR gap).
# The clamp sweep proved the PD+WNN-imit-LQR gap is CAPACITY-limited, not authority-
# limited (raising the clamp {0.4..1.0} did not move HYBRID err off ~3.16° vs LQR 1.57°).
# Capacity here = the recurrent STATE WIDTH (state_neurons): the fed-back state is
# state_neurons bits wide, so reachable state space ~2^state_neurons. bits_per_neuron
# must be >= state_neurons (forced full-state connectivity); we hold the baseline 2×
# ratio (bits = 2·state_neurons, both state & output) so sensor-sampled bits = sn stays
# proportional — the ONLY variable is state capacity. Sweep sn {16,24,32,40,48} (sn>48
# is a no-go: 2^sn state space explodes / exceeds the 4× bits cap), expert=lqr clamp=0.4 @L2:
#   HYBRID err drops toward LQR (1.57°) as sn rises → CAPACITY confirmed → invest in
#     GA-optimized connectivity next (the real fix);
#   err plateaus at ~3.16°                          → random connectivity is the ceiling,
#     not raw capacity → GA connectivity is REQUIRED, bigger-random won't help.
set -euo pipefail

ROOT="/Users/lacg/wnn"
VENV="/Users/lacg/wnn-venv"
LOG="${ROOT}/logs/controller/E5LQRNeuronSweep_20260708.log"
MARKER="/tmp/wnn_e5lqrneuron_done.json"

export PYTHONPATH="${ROOT}/src/wnn"
export RAYON_NUM_THREADS=3
export PYTHONUNBUFFERED=1
source "${VENV}/bin/activate"

SEED=20260609; BASELINE=pd; CLAMP=0.4; LEVEL=L2; EXPERT=lqr
NEURONS=(16 24 32 40 48)   # bits = 2·n (baseline-matching): 32 48 64 80 96

echo "[lqrneuron] START $(date -u +%Y-%m-%dT%H:%M:%SZ)  expert=${EXPERT} baseline=${BASELINE} ${LEVEL} clamp=${CLAMP} bits=2·n  neurons=${NEURONS[*]}" | tee -a "$LOG"

declare -a ROWS; rc=0
for n in "${NEURONS[@]}"; do
	BITS=$(( 2 * n ))   # hold the baseline 2× ratio; sensor-sampled = n bits
	# Memory guard vs the live IDS worker (a prior concurrent controller run was jetsam-killed).
	for _ in $(seq 1 60); do
		free=$(memory_pressure 2>/dev/null | grep -i 'free perc' | grep -oE '[0-9]+' | head -1)
		[ "${free:-0}" -ge 40 ] && break
		sleep 30
	done
	echo "" | tee -a "$LOG"
	echo "[lqrneuron] ===== neurons=${n} bits=${BITS} (free=${free:-?}%) =====" | tee -a "$LOG"
	if python -u "${ROOT}/scripts/e5_residual_proof.py" "${SEED}" "${BASELINE}" "${CLAMP}" "${LEVEL}" "${EXPERT}" "${n}" "${BITS}" 2>&1 | tee -a "$LOG"; then
		hyb=$(grep "HYBRID (base+residual)" "$LOG" | tail -1)
		ROWS+=("neurons=${n} :: ${hyb}")
	else
		rc=1; ROWS+=("neurons=${n} :: FAILED")
	fi
done

echo "" | tee -a "$LOG"
echo "[lqrneuron] ===== SUMMARY (HYBRID err/ITAE by neuron count; LQR ruler = 1.57°/0.047, PD = 4.04°) =====" | tee -a "$LOG"
for r in "${ROWS[@]}"; do echo "[lqrneuron] ${r}" | tee -a "$LOG"; done
echo "[lqrneuron] END $(date -u +%Y-%m-%dT%H:%M:%SZ)  rc=${rc}" | tee -a "$LOG"
printf '{"done":"%s","rc":%d,"log":"%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${rc}" "${LOG}" > "${MARKER}"
exit "${rc}"
