#!/usr/bin/env bash
# Deploy the ABI-9 worker wheel (IDSCacheWrapper.desirability_ce_anchor /
# .base_rate_entropy) AT WORKER IDLE.
#
# ram_core is UNTOUCHED by this change — base_rate_entropy lives in the worker
# crate because only IDS has a ce column — so the CONTROLLER wheel is not
# rebuilt and a flying controller chain cannot be disturbed.
#
# accel.py already carries EXPECTED_ABI = 9 (staged with this change), so the
# wheel and the Python land atomically: until the build lands, wnn.accel
# refuses to import rather than silently pairing new Python with an old wheel.
set -euo pipefail
ROOT="/Users/lacg/wnn"
cd "$ROOT"
DB="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db"

RUNNING=$(sqlite3 "file:${DB}?mode=ro" "select count(*) from flows where status='running';")
if [ "$RUNNING" != "0" ]; then
	echo "REFUSED: $RUNNING flow(s) running — the worker is NOT idle."
	exit 1
fi

echo "worker idle — building + installing ABI-9 worker wheel..."
unset CONDA_PREFIX || true
source /Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/activate
cd src/wnn/ram/strategies/accelerator
maturin develop --release

cd "$ROOT"
python -c "import ram_accelerator as a; assert a.ABI_VERSION == 9, a.ABI_VERSION; assert hasattr(a.IDSCacheWrapper,'desirability_ce_anchor'); print('worker wheel ABI', a.ABI_VERSION, '— ce anchor present')"
python -c "from wnn.accel import desirability_fitness_combine as d; print('facade OK (ABI gate passed):', d([0.9,8.0],1,[0.5,0.5],['power','exp'],[0.7,8.0]))"
echo "DEPLOYED — now restart the worker so it picks up the wheel AND the edited Python."
