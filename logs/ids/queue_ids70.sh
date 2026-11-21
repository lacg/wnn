#!/bin/bash
# IDS70 (Plane IDS-70, prereg docs/ids70_cut_method_prereg.md): queue the 6 ids_cut_method flows
# (arm E = info_gain_mc, arm B = xgb_splits; r20421-23; arm-major E then B) on the unswr fixed-80b recipe.
# Run ONLY after the ids-cut-method branch is merged into the worker checkout AND the worker restarted.
# Refuses unless:
#   (a) the worker checkout's KNOWN_PARAMS contains ids_cut_method and the fitter module imports;
#   (b) a worker process is running and STARTED AFTER that code landed (else it still runs the old encoder,
#       and would warn "UNKNOWN PARAM" and encode with quantile cuts = a silent control re-fly);
#   (c) the 6 bodies re-derived from the DB equal docs/ids70_flows.json (the python's --dry-run).
# Then POSTs each body (with its experiments array), verifies experiments + stored params, PATCHes queued.
set -u
REPO=/Users/lacg/wnn
LOG=$REPO/logs/ids/queue_ids70.log
SRC=$REPO/src
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
cd "$REPO" || exit 2
unset CONDA_PREFIX
source wnn/bin/activate

say "IDS70 queue: gate (a) worker checkout supports ids_cut_method"
if ! PYTHONPATH=$SRC python -c '
import wnn.ram.experiments.params as p, wnn.representations.supervised_cut_fitter as f
assert "ids_cut_method" in p.KNOWN_PARAMS, "ids_cut_method missing from KNOWN_PARAMS"
print("  KNOWN_PARAMS ok:", p.__file__, "| fitter:", f.__file__)' >> "$LOG" 2>&1; then
	say "REFUSED: $SRC has no ids_cut_method support (merge the ids-cut-method branch first)"; exit 3
fi

say "IDS70 queue: gate (b) running worker started after the code landed"
WPID=$(pgrep -f "wnn.ram.experiments.worke[r]" | head -1)
if [ -z "$WPID" ]; then say "REFUSED: no worker process running"; exit 4; fi
WSTART=$(date -j -f "%a %b %d %T %Y" "$(ps -o lstart= -p "$WPID" | sed "s/ *$//")" +%s)
CODE=$(stat -f %m "$SRC/wnn/ram/experiments/params.py" "$SRC/wnn/representations/supervised_cut_fitter.py" "$SRC/wnn/representations/thermometer.py" | sort -n | tail -1)
if [ "$WSTART" -le "$CODE" ]; then
	say "REFUSED: worker $WPID started $(date -r "$WSTART") <= code mtime $(date -r "$CODE"); restart the worker at idle first"; exit 5
fi
say "  worker $WPID started $(date -r "$WSTART") > code mtime $(date -r "$CODE")"

say "IDS70 queue: gate (c) bodies == docs/ids70_flows.json"
if ! python scripts/queue_ids70_cut_method.py --dry-run >> "$LOG" 2>&1; then say "REFUSED: dry-run failed (see log)"; exit 6; fi

say "IDS70 queue: POST + verify + queue"
python scripts/queue_ids70_cut_method.py >> "$LOG" 2>&1
RC=$?
say "IDS70 queue rc=$RC"
exit $RC
