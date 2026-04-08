#!/bin/bash
# Launch the 46M thermometer encoding sweep on two fixed architectures.
#
# Tests whether the 32-bit address tap saturation observed in the Pareto
# sweep (yesterday) is bounded by the 8-bit thermometer encoding rather
# than by the address-tap width itself.
#
# Two architectures × 3 thermometer widths × 2 seeds = 12 flows:
#
#   96n × 32b   (the surprise winner from yesterday's sweep)
#     - 16-bit thermometer × 2 seeds
#     - 32-bit thermometer × 2 seeds
#     - 64-bit thermometer × 2 seeds
#
#   500n × 34b  (the maximized peak from PUB50, also yesterday's saturation probe)
#     - 16-bit thermometer × 2 seeds
#     - 32-bit thermometer × 2 seeds
#     - 64-bit thermometer × 2 seeds
#
# Combined with yesterday's 8-bit baseline runs (flows 1185 and 1188),
# this gives a clean 4-point thermometer-width series (8/16/32/64) per
# architecture for the paper.
#
# Hypothesis: if the 32-bit address-tap was the ceiling, all thermometer
# widths give similar F1. If the 8-bit thermometer was the ceiling, F1
# should rise from 8b to 16b and possibly to 32b before plateauing or
# dropping (the existing 1.3M sweep showed dilution at 48-64b).
#
# Expected wall-clock: ~75-90 minutes.
#
# Usage:
#     bash scripts/launch_46M_thermometer_sweep.sh           # creates pending
#     bash scripts/launch_46M_thermometer_sweep.sh --queue   # also queues

set -e

cd "$(dirname "$0")/.."
source wnn/bin/activate
export PYTHONPATH="$(pwd)/src/wnn:$PYTHONPATH"

LAUNCHER=scripts/create_46M_eval_flow.py
DB=db/wnn.db
AUTO_QUEUE=0
SWEEP_TAG="THERMO46M"

for arg in "$@"; do
	case "$arg" in
		--queue) AUTO_QUEUE=1 ;;
		*) echo "Unknown arg: $arg"; exit 1 ;;
	esac
done

# (arch_neurons, arch_bits, thermometer_bits, n_seeds)
declare -a CONFIGS=(
	"96 32 16 2"      # surprise winner @ 16b thermo (existing 1.3M peak region)
	"96 32 32 2"      # surprise winner @ 32b thermo
	"96 32 64 2"      # surprise winner @ 64b thermo (existing 1.3M dilution region)
	"500 34 16 2"     # max architecture @ 16b
	"500 34 32 2"     # max architecture @ 32b
	"500 34 64 2"     # max architecture @ 64b
)

echo "============================================="
echo "46M Thermometer Sweep Launcher"
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
