#!/usr/bin/env bash
# Deploy the ABI-8 worker wheel (desirability_fitness_combine) AT WORKER IDLE.
# Order per CLAUDE.md + feedback memories: wheel and Python facade land
# ATOMICALLY; the worker must be idle (swap takes the shared wheel).
set -euo pipefail
ROOT="/Users/lacg/wnn"
cd "$ROOT"
DB="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db"

RUNNING=$(sqlite3 "file:${DB}?mode=ro" "select count(*) from flows where status='running';")
if [ "$RUNNING" != "0" ]; then
	echo "REFUSED: $RUNNING flow(s) running — the worker is NOT idle. Never swap under a live flow."
	exit 1
fi

echo "worker idle — building + installing ABI-8 wheel..."
unset CONDA_PREFIX || true
source /Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/activate
cd src/wnn/ram/strategies/accelerator
maturin develop --release

# Facade lands WITH the wheel (stage-the-python-with-the-wheel).
cd "$ROOT"
python - <<'PY'
import pathlib
p = pathlib.Path("src/wnn/accel.py")
t = p.read_text()
if "EXPECTED_ABI = 7" in t:
	t = t.replace("EXPECTED_ABI = 7",
		"# 8 (26/08/2026): desirability_fitness_combine (docs/DESIRABILITY_FITNESS_SHAPES.md). Additive.\nEXPECTED_ABI = 8", 1)
	p.write_text(t)
	print("accel.py: EXPECTED_ABI -> 8")
else:
	print("accel.py already expects != 7 — verify manually")
PY
python -c "import ram_accelerator as a; assert a.ABI_VERSION == 8 and hasattr(a,'desirability_fitness_combine'); print('worker wheel ABI', a.ABI_VERSION, '— desirability present')"
python -c "from wnn.accel import desirability_fitness_combine as d; print('facade OK:', d([0.9,8.0],1,[0.5,0.5],['power','exp'],[0.7,8.0]))"
echo "DEPLOYED. Restart the worker, then queue the A/B: python scripts/queue_ids_desir_ab.py"
