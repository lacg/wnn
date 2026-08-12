#!/bin/bash
# Launch the 46M thermometer encoding LOW-end sweep.
#
# Complements the high-end sweep (16/32/64-bit thermometer) which
# confirmed that wider encoding gives no lift on CIC-IoT-2023.  This
# sweep probes the other direction: does NARROWER encoding (2-bit or
# 4-bit thermometer) still work, or does it break the saturation in a
# different way?
#
# Hypothesis: counter-intuitively, narrower encoding may match or
# slightly beat 8-bit because (a) each neuron's 32-bit address tap
# samples a larger fraction of the total effective information, and
# (b) sparse memory cells see more repeated addresses, giving more
# training signal per run.  Counter-risk: at 2-bit the 3 patterns
# per feature may be too coarse to discriminate attack signatures.
#
# Two architectures × 2 thermometer widths × 2 seeds = 8 flows:
#
#   96n × 32b   (the smaller peak from the high-end sweep — keeps
#                continuity with the upper-bound series)
#     - 2-bit thermometer × 2 seeds
#     - 4-bit thermometer × 2 seeds
#
#   400n × 8b   (the 25 KB FPGA-deployable peak from the Pareto sweep
#                — deployment-interesting architecture)
#     - 2-bit thermometer × 2 seeds
#     - 4-bit thermometer × 2 seeds
#
# Expected wall-clock: ~25-35 minutes total (smaller input vectors
# mean faster training than the wide-thermometer sweep).
#
# Usage:
#     bash scripts/launch_46M_thermometer_low_sweep.sh           # creates pending
#     bash scripts/launch_46M_thermometer_low_sweep.sh --queue   # also queues

set -e

cd "$(dirname "$0")/.."
source wnn/bin/activate
export PYTHONPATH="$(pwd)/src/wnn:$PYTHONPATH"

LAUNCHER=scripts/create_46M_eval_flow.py
DB=db/wnn.db
AUTO_QUEUE=0
SWEEP_TAG="THERMO46MLOW"

for arg in "$@"; do
	case "$arg" in
		--queue) AUTO_QUEUE=1 ;;
		*) echo "Unknown arg: $arg"; exit 1 ;;
	esac
done

# (arch_neurons, arch_bits, thermometer_bits, n_seeds)
declare -a CONFIGS=(
	"96 32 2 2"       # smaller peak @ 2b thermometer (ultra-narrow, ~32 effective bits)
	"96 32 4 2"       # smaller peak @ 4b thermometer (~46 effective bits)
	"400 8 2 2"       # FPGA-deployable peak @ 2b thermometer
	"400 8 4 2"       # FPGA-deployable peak @ 4b thermometer
)

echo "============================================="
echo "46M Thermometer LOW Sweep Launcher"
echo "============================================="
echo "Sweep tag: $SWEEP_TAG"
echo "Auto-queue: $AUTO_QUEUE"
echo

TOTAL_FLOWS=0
FAILED_FLOWS=0
for cfg in "${CONFIGS[@]}"; do
	read -r neurons bits thermo n_seeds <<< "$cfg"
	echo "  ${neurons}n × ${bits}b @ ${thermo}b thermometer: ${n_seeds} seed(s)"
	for seed in $(seq 1 "$n_seeds"); do
		if python "$LAUNCHER" \
			--neurons "$neurons" \
			--bits "$bits" \
			--thermometer-bits "$thermo" \
			--seed "$seed" \
			--name "$SWEEP_TAG" 2>&1 | grep "✓ Created"; then
			TOTAL_FLOWS=$((TOTAL_FLOWS + 1))
		else
			FAILED_FLOWS=$((FAILED_FLOWS + 1))
			echo "    seed=$seed FAILED"
		fi
	done
done

echo
echo "============================================="
echo "Created $TOTAL_FLOWS pending flows ($FAILED_FLOWS failed)."
echo "============================================="
echo

if [ "$AUTO_QUEUE" -eq 1 ]; then
	echo "Auto-queueing all $SWEEP_TAG flows..."
	queued=$(sqlite3 "$DB" "
		UPDATE flows SET status='queued'
		WHERE name LIKE '%${SWEEP_TAG}%' AND status='pending';
		SELECT changes();
	")
	echo "  Queued $queued flows."
	echo
	echo "Worker will pick them up in ID-descending order after the current"
	echo "PUB50 flow finishes (newest IDs first)."
else
	echo "Flows are status='pending'. To queue them:"
	echo "  sqlite3 $DB \"UPDATE flows SET status='queued' WHERE name LIKE '%${SWEEP_TAG}%' AND status='pending';\""
fi

echo
echo "To monitor progress:"
echo "  sqlite3 $DB \"SELECT id, name, status FROM flows WHERE name LIKE '%${SWEEP_TAG}%' ORDER BY id;\""
