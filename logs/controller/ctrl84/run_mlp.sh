#!/bin/zsh
# CTRL-84: train + VAL-select + TEST-score ONE (arm, seed) MLP baseline (docs/ctrl84_mlp_baseline_prereg.md).
#
#   run_mlp.sh <a|b> <seed> [--dry-run]
#
# 1. mlp_baseline.py: DAgger (Rust rollouts, MLP forward in Rust; PyTorch fit) under the paired
#    l2bnm sN run's argv, the 8-config HP grid, selection on the 20 VAL units  -> data/mlp<arm>_s<N>.npz
# 2. ctrl83/score_cell.py mlp<arm>_s<N> nom  -> ctrl83/data/mlp<arm>_s<N>__nom.npz  (the 35 TEST units, ONCE)
#    and, when cells.py has it, the joint-OOD cell `all`  -> ctrl83/data/mlp<arm>_s<N>__all.npz
# 3. one line in ctrl48/runs.log: "N84 MLP <id> ..." (TEST whole vector; report-only).
#
# Env: WNN_PY (python, default the wnn venv), WNN_SRC (src tree, default this checkout's), CTRL84_SITE
# (prepend to PYTHONPATH — e.g. an UNPACKED ABI-71 wheel; never installed by this script). nice 15, CPU scorer.
# Never touches wnn.control.phased_ga as a process (argv); the run is a plain python driver.
set -euo pipefail
ARM=${1:?arm a|b}; SEED=${2:?seed 1..5}; DRY=${3:-}
HERE=${0:A:h}
REPO=${HERE:h:h:h}
PY=${WNN_PY:-/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python}
SRC=${WNN_SRC:-$REPO/src}
export PYTHONPATH="${CTRL84_SITE:+$CTRL84_SITE:}$SRC"
export WNN_CONTROLLER_GPU_EVAL=0
ID="mlp${ARM}_s${SEED}"
LOG="$HERE/data/${ID}.log"
mkdir -p "$HERE/data"
ABI=$($PY -c "import ram_controller as c; print(c.ABI_VERSION)")
if [[ "$ABI" -lt 71 ]]; then echo "[ctrl84] ram_controller ABI $ABI < 71 on PYTHONPATH=$PYTHONPATH — set CTRL84_SITE to an ABI-71 build" >&2; exit 2; fi
if [[ "$DRY" == "--dry-run" ]]; then
	nice -n 15 $PY "$HERE/mlp_baseline.py" --arm "$ARM" --seed "$SEED" --dry-run
	echo "[ctrl84] dry run: would then score $ID on cells nom (+ all if defined) via ctrl83/score_cell.py"
	exit 0
fi
nice -n 15 $PY "$HERE/mlp_baseline.py" --arm "$ARM" --seed "$SEED" 2>&1 | tee "$LOG"
CELLS="nom"
if $PY -c "import sys; sys.path.insert(0, '$REPO/logs/controller/ctrl83'); import cells; sys.exit(0 if 'all' in cells.CELL else 1)"; then CELLS="nom all"; fi
for CELL in ${=CELLS}; do
	OUT="$REPO/logs/controller/ctrl83/data/${ID}__${CELL}.npz"
	LINE=$(nice -n 15 $PY "$REPO/logs/controller/ctrl83/score_cell.py" "$ID" "$CELL" "$OUT" 2>&1 | tail -1)
	echo "$LINE" | tee -a "$LOG"
	echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) N84 MLP $ID cell $CELL (TEST, report-only, ABI $ABI): $LINE" >> /Users/lacg/wnn/logs/controller/ctrl48/runs.log
done
