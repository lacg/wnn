#!/bin/bash
# Launch the 46M Pareto frontier sweep across small WNN configurations.
#
# Creates 18 single-genome evaluation flows on the full CIC-IoT-2023 46M
# benchmark (lacg030175/CIC-IoT-2023-full, random 80/20 config). Each flow
# is one (neurons, bits, seed) point — no GA refinement, just train+eval
# with all 7 threshold modes reported.
#
# Configs sweep memory range: 20 bytes (5n × 4b) up to 25 KB (400n × 8b).
# 5n × 4b runs 10 seeds for multi-seed validation of the "20-byte
# detector" candidate; all others run 1 seed.
#
# Expected wall-clock time: ~2-2.5 hours total (small configs ~3 min each,
# large configs ~22 min). Run AFTER PUB50 completes so it doesn't compete
# for memory (each 46M load peaks at ~30 GB RAM during dataset prep).
#
# Usage:
#     bash scripts/launch_46M_pareto_sweep.sh
#
# All flows are created with status='pending' so they don't auto-start.
# Move them to 'queued' once you're ready by running:
#     sqlite3 db/wnn.db "UPDATE flows SET status='queued' WHERE name LIKE 'SWEEP46M-%';"

set -e

cd "$(dirname "$0")/.."
source wnn/bin/activate
export PYTHONPATH="$(pwd)/src/wnn:$PYTHONPATH"

LAUNCHER=scripts/create_46M_eval_flow.py

# (neurons, bits, n_seeds) — same list as project_46M_small_genome_sweep.md
declare -a CONFIGS=(
	"5 4 10"      # 20 B   — multi-seed validation of "20-byte detector"
	"100 4 1"     # 400 B
	"200 4 1"     # 800 B  — paper headline (already cross-validated)
	"300 4 1"     # 1.2 KB
	"500 4 1"     # 2.0 KB
	"5 12 1"      # 5.0 KB — Lonely Valley Dweller (control)
	"100 8 1"     # 6.2 KB — Pareto-best in 5-6 KB regime
	"300 8 1"     # 18.8 KB
	"400 8 1"     # 25.0 KB — biggest config that fits on iCE40UP5K
)

echo "============================================="
echo "46M Pareto Sweep Launcher"
echo "============================================="
echo
echo "Creating $(echo "${CONFIGS[@]}" | wc -w) configs..."
echo

TOTAL_FLOWS=0
for cfg in "${CONFIGS[@]}"; do
	read -r neurons bits n_seeds <<< "$cfg"
	BYTES=$((neurons * (1 << bits) * 2 / 8))
	echo "  ${neurons}n × ${bits}b (${BYTES} B): ${n_seeds} seed(s)"
	for seed in $(seq 1 "$n_seeds"); do
		python "$LAUNCHER" \
			--neurons "$neurons" \
			--bits "$bits" \
			--seed "$seed" \
			--name "SWEEP46M" 2>&1 | grep "✓ Created" || echo "    seed=$seed FAILED"
		TOTAL_FLOWS=$((TOTAL_FLOWS + 1))
	done
done

echo
echo "============================================="
echo "Created $TOTAL_FLOWS pending flows."
echo "============================================="
echo
echo "All flows are status='pending'. To start the sweep:"
echo "  sqlite3 db/wnn.db \"UPDATE flows SET status='queued' WHERE name LIKE '%SWEEP46M%';\""
echo
echo "To monitor progress:"
echo "  sqlite3 db/wnn.db \"SELECT id, name, status FROM flows WHERE name LIKE '%SWEEP46M%' ORDER BY id;\""
