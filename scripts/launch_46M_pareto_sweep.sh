#!/bin/bash
# Launch the 46M Pareto frontier sweep across small WNN configurations + peak.
#
# Creates ~21 single-genome evaluation flows on the full CIC-IoT-2023 46M
# benchmark (lacg030175/CIC-IoT-2023-full, random 80/20 config). Each flow
# is one (neurons, bits, seed) point — no GA refinement, just train+eval
# with all 7 threshold modes reported.
#
# Memory coverage:
#   Micro   (20 B - 2 KB):  5n×4b through 500n×4b      — ultra-low-power edge
#   Small   (5 - 25 KB):    5n×12b, 100n×8b, 400n×8b   — embedded / FPGA
#   Peak    (~200 MB - 1 GB sparse): 96/198/245n × 32b, 500n × 34b
#           — best PUB50 architectures transferred to 46M
#
# 5n × 4b runs 10 seeds for multi-seed validation of the "20-byte detector".
# All other architectures run 1 seed. The peak tier has four genomes:
#   - 96n × 32b:   smallest PUB50 peak-class (flow 788, F1 82.24% @ fixed_05)
#   - 198n × 32b:  deployable peak <5% FPR (flow 789, F1 81.00% @ train_cal)
#   - 245n × 32b:  literature-competitive <10% FPR (flow 794, F1 82.45% @ fixed_05)
#   - 500n × 34b:  saturation proof (flow 809, F1 82.50% @ train_cal)
#                  — tests if 2 extra bits help with 46M samples (expected: no,
#                  thermometer-encoding collisions cap effective address space)
#
# Expected wall-clock: ~90 minutes total
#   - Micro/small configs:     9 × ~2.5 min = ~23 min
#   - 5n × 4b multi-seed:      9 × ~2.5 min = ~23 min
#   - Peak 96n × 32b:          ~5 min
#   - Peak 198n × 32b:         ~8 min
#   - Peak 245n × 32b:         ~10 min
#   - Peak 500n × 34b:         ~15 min
#
# Run AFTER PUB50 completes so it doesn't compete for memory
# (each 46M load peaks at ~30 GB RAM during dataset prep).
#
# Usage:
#     bash scripts/launch_46M_pareto_sweep.sh           # creates pending flows
#     bash scripts/launch_46M_pareto_sweep.sh --queue   # also queues them

set -e

cd "$(dirname "$0")/.."
source wnn/bin/activate
export PYTHONPATH="$(pwd)/src/wnn:$PYTHONPATH"

LAUNCHER=scripts/create_46M_eval_flow.py
DB=db/wnn.db
AUTO_QUEUE=0
SWEEP_TAG="SWEEP46M"

for arg in "$@"; do
	case "$arg" in
		--queue) AUTO_QUEUE=1 ;;
		*) echo "Unknown arg: $arg"; exit 1 ;;
	esac
done

# (neurons, bits, n_seeds, tier_label) — full Pareto + peak tier
declare -a CONFIGS=(
	# Micro/small Pareto frontier
	"5 4 10 MICRO"        # 20 B   — multi-seed validation of "20-byte detector"
	"100 4 1 MICRO"       # 400 B
	"200 4 1 MICRO"       # 800 B  — paper headline (cross-validated independently)
	"300 4 1 MICRO"       # 1.2 KB
	"500 4 1 MICRO"       # 2.0 KB
	"5 12 1 SMALL"        # 5.0 KB — Lonely Valley Dweller (control)
	"100 8 1 SMALL"       # 6.2 KB — Pareto-best in 5-6 KB regime
	"300 8 1 SMALL"       # 18.8 KB
	"400 8 1 SMALL"       # 25.0 KB — biggest config that fits on iCE40UP5K BRAM
	# Peak tier: best PUB50 architectures transferred to 46M
	"96 32 1 PEAK"        # flow 788: F1 82.24% @ fixed_05 — smallest peak
	"198 32 1 PEAK"       # flow 789: F1 81.00% @ train_cal — deployable <5% FPR
	"245 32 1 PEAK"       # flow 794: F1 82.45% @ fixed_05 — literature-competitive
	"500 34 1 PEAK"       # flow 809: F1 82.50% @ train_cal — 34b saturation proof
)

echo "============================================="
echo "46M Pareto Sweep Launcher"
echo "============================================="
echo "Sweep tag: $SWEEP_TAG"
echo "Auto-queue: $AUTO_QUEUE"
echo

TOTAL_FLOWS=0
FAILED_FLOWS=0
for cfg in "${CONFIGS[@]}"; do
	read -r neurons bits n_seeds tier <<< "$cfg"
	BYTES=$((neurons * (1 << bits) * 2 / 8))
	if [ "$BYTES" -gt 1073741824 ]; then
		HUMAN="$((BYTES / 1024 / 1024 / 1024)) GB dense"
	elif [ "$BYTES" -gt 1048576 ]; then
		HUMAN="$((BYTES / 1024 / 1024)) MB dense"
	elif [ "$BYTES" -gt 1024 ]; then
		HUMAN="$((BYTES / 1024)) KB dense"
	else
		HUMAN="${BYTES} B"
	fi
	echo "  [$tier] ${neurons}n × ${bits}b (${HUMAN}): ${n_seeds} seed(s)"
	for seed in $(seq 1 "$n_seeds"); do
		# Tag includes tier for clarity, all sharing SWEEP46M for bulk ops
		name_suffix="${SWEEP_TAG}-${tier}"
		if python "$LAUNCHER" \
			--neurons "$neurons" \
			--bits "$bits" \
			--seed "$seed" \
			--name "$name_suffix" 2>&1 | grep "✓ Created"; then
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
	echo "Auto-queueing all sweep flows..."
	queued=$(sqlite3 "$DB" "
		UPDATE flows SET status='queued'
		WHERE name LIKE '%${SWEEP_TAG}%' AND status='pending';
		SELECT changes();
	")
	echo "  Queued $queued flows."
	echo
	echo "Worker will pick them up in ID-descending order (newest first)."
else
	echo "Flows are status='pending'. To queue them:"
	echo "  sqlite3 $DB \"UPDATE flows SET status='queued' WHERE name LIKE '%${SWEEP_TAG}%' AND status='pending';\""
fi

echo
echo "To monitor progress:"
echo "  sqlite3 $DB \"SELECT id, name, status FROM flows WHERE name LIKE '%${SWEEP_TAG}%' ORDER BY id;\""
echo "  watch -n 10 \"sqlite3 $DB 'SELECT status, count(*) FROM flows WHERE name LIKE \\\"%${SWEEP_TAG}%\\\" GROUP BY status;'\""
