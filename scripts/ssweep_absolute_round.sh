#!/bin/bash
# Generic ABSOLUTE round driver (mirrors delta's ssweep_round.sh/ssweep_finals.sh):
# run the given combo labels at a given seed, --no-delta-control, sequentially
# (one controller at a time). Used for rounds 2/3 (survivors @ seeds 10/11) and
# finals (top-3+control @ seeds 12/13). Reuses run_absolute_winner.sh per combo
# (writes S{LABEL}_ABS_seed{SEED}). Args: SEED LABEL [LABEL...].
set -u
cd /Users/lacg/wnn
SEED="${1:?usage: ssweep_absolute_round.sh SEED LABEL [LABEL...]}"; shift
LABELS=("$@")
[ ${#LABELS[@]} -ge 1 ] || { echo "need >=1 label"; exit 1; }
LOG=logs/controller/Ssweep_20260622/driver_abs_round_${SEED}.log
mkdir -p logs/controller/Ssweep_20260622; exec >>"$LOG" 2>&1
echo "[abs-round] $(date '+%Y-%m-%d %H:%M:%S') START seed=$SEED labels: ${LABELS[*]}"
for lab in "${LABELS[@]}"; do
  echo "[abs-round] $(date '+%H:%M:%S') -> $lab seed=$SEED"
  bash scripts/run_absolute_winner.sh "$lab" "$SEED"
done
echo "{\"abs_round_done\":$SEED,\"labels\":\"${LABELS[*]}\",\"ts\":\"$(date '+%Y-%m-%dT%H:%M:%S')\"}" > /tmp/wnn_ssweep_abs_round_${SEED}_done.json
echo "[abs-round] $(date '+%Y-%m-%d %H:%M:%S') DONE seed=$SEED"
