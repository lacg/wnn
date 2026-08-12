#!/bin/bash
# E5 FROZEN-ARCH L1→L2 (08/07/2026, Luiz's corrected variant). The first "memory-only"
# attempt (--seed-winner-stage memory) was a NO-OP: the MEMORY stage uses score_genomes
# (value-GA, NO DAGGER) so it never adapted the frozen L1 cells to L2 — it just measured
# RAW L1→L2 transfer (s09 48.2 / s10 0.2). The L2 cell-adaptation lives in the Lamarckian
# ARCH phases. Reuse the CONNECTIONS stage (synaptogenesis): it FREEZES neuron-count +
# bit-width and varies only connectivity, and with --lamarckian it DAGGER-retrains the
# cells under L2 → the true "fine-tune the proven L1 shape under the storm" test. Chain:
# CONNECTIONS → MEMORY (grid/NEURONS/BITS skipped via --seed-winner-stage connections).
#
# Cells: each seed from its OWN L1 winner (paired vs E5.2 neurons+memory + raw-transfer):
#   s09 ← W23Weather_20260706/PWM2K_L1_seed20260609/winner.yaml.gz
#   s10 ← W23Weather_20260706/PWM2K_L1_seed20260610/winner.yaml.gz
# Rulers @L2 (FRESH-eval): from-scratch 19.5 · L1-transfer 57.2 · E5.2 neurons+memory
# fresh 46.1 (s09 29 / s10 63.2) · raw-transfer 48.2/0.2 · PD 84 · PID+ 99.8. Q: does
# freezing neurons+bits (only rewire+retrain) beat full neuron search (46.1) AND rescue
# s10 (raw 0.2)? If s10 recovers → cell/wiring retrain suffices; if not → s10 needs sn change.
set -u
cd /Users/lacg/wnn
LOG=logs/controller/E5FrozenArch_20260707.log
exec >>"$LOG" 2>&1

echo "[e5f] $(date '+%Y-%m-%d %H:%M:%S') starting L1→L2 FROZEN-ARCH (connections→memory)"

export PYTHONPATH=/Users/lacg/wnn/src/wnn WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=8
PY=/Users/lacg/wnn-venv/bin/python
ERR=0.25 STEADY=0.35 STABLE=0.20 JERK=0.15 MONO=0.05
ROOT=logs/controller/E5FrozenArch_20260707

run_one() {
	local name="$1" seed="$2" l1win="$3"
	local dir="$ROOT/${name}_seed${seed}"; mkdir -p "$dir"
	if [ -f "$dir/done.json" ]; then echo "[e5f] $(date '+%H:%M:%S') SKIP ${name} seed=${seed}"; return; fi
	if [ ! -f "$l1win" ]; then echo "[e5f] $(date '+%H:%M:%S') MISSING L1 winner $l1win — skip ${name}"; return; fi
	echo "[e5f] $(date '+%Y-%m-%d %H:%M:%S') START ${name} seed=${seed} ← $l1win (CONNECTIONS→MEMORY, neurons+bits frozen)"
	$PY -u -m wnn.control.phased_ga \
		--seed-winner "$l1win" \
		--seed-winner-stage connections \
		--disturbance L2 \
		--no-delta-control --integral-leak 0.99 --integral-scale 1.0 \
		--lamarckian --saturation-grow-gain 1.0 --magnitude-aware-patience \
		--check-interval 5 --conns-gens 15 --conns-patience 6 --memory-gens 15 --memory-patience 8 \
		--pop 24 --num-eval-folds 5 \
		--eval-episodes 100 --steps 2000 --tilt 5.0 --body-rate 0.5 --yaw-rate 0.3 \
		--rg-rounds 3 --rg-episodes-per-round 8 --universe-episodes 5 \
		--immigrants 0.15 --obs-pwm \
		--fit-weight-err-sq "$ERR" --fit-weight-steady "$STEADY" --fit-weight-stable "$STABLE" \
		--fit-weight-jerk "$JERK" --fit-weight-mono "$MONO" \
		--report-episodes 100 --report-seeds 99990001 99990101 12345 67890 --holdout-pop-sample 8 \
		--base-seed "$seed" --runs 1 \
		--save-winner "$dir/winner.yaml.gz" --save-stage-checkpoints "$dir" > "$dir/run.out" 2>&1
	if [ $? -ne 0 ]; then echo "[e5f] $(date '+%H:%M:%S') FAIL ${name} seed=${seed} (continuing)"
	else echo "{\"arm\":\"${name}\",\"seed\":${seed},\"disturbance\":\"L2\",\"mode\":\"frozen_arch_connections\"}" > "$dir/done.json"
		echo "[e5f] $(date '+%Y-%m-%d %H:%M:%S') COMPLETE ${name} seed=${seed}"; fi
}

run_one FROZENARCH_L2 20260609 "logs/controller/W23Weather_20260706/PWM2K_L1_seed20260609/winner.yaml.gz"
run_one FROZENARCH_L2 20260610 "logs/controller/W23Weather_20260706/PWM2K_L1_seed20260610/winner.yaml.gz"

echo "{\"e5frozenarch_done\":true,\"ts\":\"$(date -u '+%Y-%m-%dT%H:%M:%SZ')\"}" > /tmp/wnn_e5frozenarch_done.json
echo "[e5f] $(date '+%Y-%m-%d %H:%M:%S') ALL FROZEN-ARCH RUNS DONE"
