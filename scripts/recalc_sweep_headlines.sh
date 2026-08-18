#!/bin/bash
# Re-headline already-flown sweep runs — NO re-fly, NO training, NO search.
#
# WHY (17/08/2026). `_select_headline_stage` was fed a hardcoded stage list
# (`(1,"NEURONS"),(4,"MEMORY")`), so CONNECTIONS and BITS never entered the
# candidate pool. On a sweep run (`--skip-stages neurons,bits`) that left GRID +
# MEMORY = 6 candidates while the header still read as if the pool were complete
# — and at b=10 the CONNECTIONS block was the run's BEST block yet could not be
# selected. Fixed by ControllerOrchestrator.stage_entries() (registry-driven).
#
# The already-banked widths do not need re-flying: every stage checkpoint holds
# its final_population and its spec, so the selection can simply be re-run over
# the full 9-candidate pool. That is what --recalc-headline does.
#
# The phased_ga flags MUST match the original run exactly (they define the
# episodes, the seeds and the scoring), so this script does NOT retype them: it
# evals the chain's own config block. If sweep_ladder_chain.sh changes, this
# follows automatically instead of silently drifting.
#
# Usage:  scripts/recalc_sweep_headlines.sh [tag ...]
#   default = every SL_A_* marker whose run predates the fix.
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
CHAIN="$ROOT/scripts/sweep_ladder_chain.sh"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
OUTDIR="logs/controller/sweep_ladder"
MARKDIR="experiments/sweepladder_markers"
LOG="/private/tmp/sweep_recalc.log"

# The chain's OWN config block (AIRFRAME .. S16_WEIGHTS) — single source of truth.
eval "$(sed -n '/^AIRFRAME=/,/^S16_WEIGHTS=/p' "$CHAIN")"

log() { echo "[recalc] $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

TAGS=("$@")
if [ ${#TAGS[@]} -eq 0 ]; then
	while IFS= read -r m; do
		TAGS+=("$(basename "$m" .json)")
	done < <(ls -1 "$MARKDIR"/SL_A_*.json 2>/dev/null)
fi
[ ${#TAGS[@]} -eq 0 ] && { log "no markers to recalc"; exit 0; }

log "########## RE-HEADLINE ${#TAGS[@]} run(s): ${TAGS[*]} ##########"

for tag in "${TAGS[@]}"; do
	marker="$MARKDIR/${tag}.json"
	ckpt="$OUTDIR/ckpt/$tag"
	out="$OUTDIR/${tag}.recalc.out"
	if [ ! -f "$marker" ]; then log "$tag: no marker — skip"; continue; fi
	if [ ! -d "$ckpt" ]; then log "$tag: no checkpoint dir — skip"; continue; fi
	if grep -q '"headline_recalc_at"' "$marker" 2>/dev/null; then
		log "$tag: already recalculated — skip"; continue
	fi

	# bits / neurons / seed come from the marker, so the flags cannot disagree
	# with the run they describe.
	read -r BITS_R NEUR_R SEED_R < <("$VP" -c "
import json,sys
d=json.load(open('$marker'))
print(d['bits'], d['neurons'], d['seed'])
")
	log "===== $tag (b=$BITS_R n=$NEUR_R s=$SEED_R) ====="
	# shellcheck disable=SC2086
	"$VP" -u -m wnn.control.phased_ga \
		--recalc-headline "$ckpt" \
		--levels 16 \
		--eval-episodes 100 --memory-eval-episodes 200 \
		--steps 2000 --tilt 5.0 \
		$S16_WEIGHTS \
		--report-episodes 100 --holdout-pop-sample 8 \
		--runs 1 --memory-mode BINARY \
		--airframe "$AIRFRAME" --disturbance "$DIST" --teacher mpcof \
		$FEAT_STAGE1 \
		--translation --fit-weight-alt 16 \
		--grid-state-neurons 0 --max-state-neurons 0 \
		--grid-bits "$BITS_R" --grid-output-neurons "$NEUR_R" \
		--max-output-neurons "$NEUR_R" \
		--num-eval-folds 5 \
		--report-seeds $REPORT_SEEDS \
		--base-seed "$SEED_R" > "$out" 2>&1
	rc=$?
	if [ $rc -ne 0 ]; then
		log "$tag: rc=$rc — marker UNCHANGED (see $out)"
		continue
	fi

	# Fold the new headline into the marker, keeping the old one under *_v1 so
	# the superseded number stays auditable next to its replacement.
	"$VP" - "$marker" "$out" <<'PY'
import json, sys, re, time
marker, out = sys.argv[1], sys.argv[2]
txt = open(out, errors="replace").read()
def grab(pat):
	m = [l for l in txt.splitlines() if pat in l]
	return m[-1] if m else None
stage = grab("[stage-select] HEADLINE stage=")
hold  = grab("[stage-select] HEADLINE held-out:")
if stage is None or hold is None:
	print(f"  [recalc] {marker}: no headline in output — marker UNCHANGED")
	sys.exit(1)
d = json.load(open(marker))
d["headline_stage_v1"]   = d.get("headline_stage")
d["headline_holdout_v1"] = d.get("headline_holdout")
d["headline_stage"]      = stage
d["headline_holdout"]    = hold
d["headline_recalc_at"]  = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
d["headline_recalc_why"] = ("re-selected over the FULL candidate pool; the original "
                            "run excluded CONNECTIONS/BITS (hardcoded stage list, "
                            "fixed 17/08/2026). Per-stage held-out blocks unchanged.")
cands = grab("val ") if "<- HEADLINE" in txt else None
json.dump(d, open(marker, "w"), indent=2)
print(f"  [recalc] {marker}: headline updated")
print(f"    was: {d['headline_holdout_v1']}")
print(f"    now: {d['headline_holdout']}")
PY
	log "$tag: done (rc=$rc)"
done

log "########## RE-HEADLINE COMPLETE ##########"
