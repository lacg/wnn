#!/usr/bin/env bash
# SPECIALIST ROUND-2 INSTALL GUARD (16/08/2026; re-armed with smoke 4 the same
# evening). Owns the round-1 -> round-2 chain boundary: waits for round 1 to
# genuinely drain (6 specialist1 markers + 0 controllers), installs the staged
# 16/08 wheel (alt_err row 14 + --target-levels + Rust samplers + scoped
# axonogenesis), verifies the features are really in the installed build,
# smokes FOUR pop-6 launches (legacy / target-levels / framed1 / the P2
# connectivity pipeline; each needs rc=0, zero FELL BACK lines, and its config
# line greppable), then launches specialist_round2_chain.sh detached.
#
# A failed smoke leaves the box IDLE deliberately and launches NOTHING — the
# status tick reports the idleness, a human reads the smoke .out. This pattern
# caught the arm-D trainer panic (twice) before it could burn a 27h round.
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1

LOG="/private/tmp/specialist2_guard.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
WHEEL="$ROOT/dist_staged/ram_controller-2026.212.37-cp313-cp313-macosx_11_0_arm64.whl"
SMOKE_DIR="/private/tmp/specialist2_smoke"

log() { echo "[spec2guard] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controllers() { pgrep -f "MacOS/Python -u -m wnn.control.phased_ga" | wc -l | tr -d ' '; }

mkdir -p "$SMOKE_DIR"
[ -f "$WHEEL" ] || { log "ABORT: staged wheel missing"; exit 1; }
log "########## ARMED — waiting for 6 specialist1 markers + 0 controllers ##########"

# ~5 round-1 arms remain at ~4.5h; 60h ceiling = fires only if the pipeline died.
WAITED=0
while true; do
	NS=$(ls "$ROOT/experiments/specialist1_markers" 2>/dev/null | wc -l | tr -d ' ')
	C=$(controllers)
	if [ "$NS" -ge 6 ] && [ "$C" -eq 0 ]; then
		log "gate open: specialist1=$NS controllers=0 (waited ${WAITED}s)"
		break
	fi
	if [ "$WAITED" -ge 216000 ]; then
		log "ABORT: specialist1=$NS controllers=$C after 60 h — round 1 likely died."
		exit 1
	fi
	sleep 300
	WAITED=$((WAITED + 300))
done

log "installing $(basename "$WHEEL")"
if ! "$VP" -m pip install --force-reinstall --no-deps "$WHEEL" >> "$LOG" 2>&1; then
	log "ABORT: pip install failed"
	exit 1
fi
if ! "$VP" -c "
import ram_controller as c, sys
sig = c.dagger_train_batch_inplace.__text_signature__ or ''
ok = 'target_levels' in sig and hasattr(c.WnnController, 'set_target_levels')
sys.exit(0 if ok else 1)"; then
	log "ABORT: installed wheel does NOT expose target_levels — refusing to continue"
	exit 1
fi
log "install verified: dagger_train_batch_inplace + WnnController carry target_levels"

smoke() {
	local name="$1"; shift
	local out="$SMOKE_DIR/${name}.out"
	log "smoke $name starting"
	timeout 1800 "$VP" -u -m wnn.control.phased_ga \
		--levels 16 --skip-stages bits,connections --lamarckian \
		--max-cells 180000 --max-cells-strict \
		--neurons-gens 1 --neurons-patience 3 --memory-gens 1 --memory-patience 2 \
		--pop 6 --num-eval-folds 5 --check-interval 2 --magnitude-aware-patience \
		--eval-episodes 4 --memory-eval-episodes 4 --steps 100 --tilt 5.0 \
		--fit-weight-err-sq 0.25 --fit-weight-steady 0.35 --fit-weight-stable 0.20 \
		--fit-weight-jerk 0.15 --fit-weight-mono 0.05 \
		--report-episodes 4 --holdout-pop-sample 2 \
		--grid-state-neurons 0 --max-state-neurons 0 \
		--runs 1 --memory-mode BINARY \
		--airframe cf21_brushless --disturbance L4C --teacher mpcof \
		--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i \
		--obs-collective-cmd --obs-alt-err --obs-vz \
		--translation --reward-lambda-alt 16 \
		"$@" \
		--report-seeds 99990101 --base-seed 31337002 > "$out" 2>&1
	local rc=$?
	local fb; fb=$(grep -ac "FELL BACK" "$out")
	if [ "$rc" != "0" ] || [ "$fb" != "0" ]; then
		log "SMOKE FAILED ($name): rc=$rc fallback_lines=$fb — see $out. Box left IDLE; chain NOT launched."
		return 1
	fi
	log "smoke $name PASSED"
	return 0
}

smoke legacy  --grid-bits 24 30 --max-output-neurons 128                                || exit 1
smoke tlevels --grid-bits 15 --max-output-neurons 256 --target-levels 32                || exit 1
smoke framed1 --grid-bits 18 --max-output-neurons 240 --conn-policy framed1 \
              --output-full-window --input-window-k 4 --frame-stride 10           || exit 1
# Smoke 4 (16/08 evening): the P2 CONNECTIVITY pipeline the sweep chain flies —
# grid -> CONNECTIONS(feature-scope) -> MEMORY. The later --skip-stages
# overrides the helper's default (argparse last-wins). This is the exact shape
# WSB-P2 runs; nothing else in the smoke set exercises a skipped-NEURONS stage.
smoke connscope --skip-stages neurons,bits --conn-mutation-scope feature \
                --conns-gens 1 --conns-patience 2 \
                --grid-bits 14 18 --grid-output-neurons 32 --max-output-neurons 32 || exit 1
# target-levels must have ACTUALLY armed (config line printed), not just not-crashed.
if ! grep -aq "\[target-levels\] T=32" "$SMOKE_DIR/tlevels.out"; then
	log "SMOKE FAILED (tlevels): rc=0 but no [target-levels] config line — flag never reached the run. Box left IDLE."
	exit 1
fi
# Same rule for framed1: rc=0 proves nothing armed, only that nothing crashed.
if ! grep -aq "\[conn-policy\] framed1" "$SMOKE_DIR/framed1.out"; then
	log "SMOKE FAILED (framed1): rc=0 but no [conn-policy] framed1 line — policy never reached the run. Box left IDLE."
	exit 1
fi
# And for the connectivity pipeline: the scope must have armed AND the
# CONNECTIONS stage must have actually run (NEURONS genuinely skipped).
if ! grep -aq "\[conn-scope\] feature" "$SMOKE_DIR/connscope.out"; then
	log "SMOKE FAILED (connscope): rc=0 but no [conn-scope] feature line — scope never reached the run. Box left IDLE."
	exit 1
fi
if ! grep -aq "STAGE 3 (CONNECTIONS) done" "$SMOKE_DIR/connscope.out"; then
	log "SMOKE FAILED (connscope): rc=0 but the CONNECTIONS stage never completed — skip-stages neurons broke the pipeline. Box left IDLE."
	exit 1
fi

# 16/08 ~21:20 EDT (Luiz): round 3 flies FIRST at b30/128n, then round 2 —
# the sequencer owns that order; both chains are marker-idempotent.
CH=$("$VP" -c "
import subprocess
p = subprocess.Popen(['bash', 'scripts/specialist_rounds_sequencer.sh'], cwd='$ROOT',
	stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
print(p.pid)
")
log "########## GUARD COMPLETE — specialist_rounds_sequencer (round 3 then round 2) launched as PID $CH ##########"
