#!/usr/bin/env bash
# SWEEP-LADDER BOOT (16/08/2026 ~21:45 EDT). One-shot: the box is IDLE (Luiz
# stopped round 1 mid-arm-E and every armed chain/guard), so the deferred wheel
# install happens NOW, followed by the same 4 smokes the retired install guard
# carried, then the ladder chain launches detached. A failed smoke leaves the
# box IDLE deliberately and launches NOTHING.
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1

LOG="/private/tmp/sweep_ladder_boot.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
WHEEL="$ROOT/dist_staged/ram_controller-2026.212.37-cp313-cp313-macosx_11_0_arm64.whl"
SMOKE_DIR="/private/tmp/sweep_ladder_smoke"

log() { echo "[boot] $(date -u +%FT%TZ) $*" >> "$LOG"; }
mkdir -p "$SMOKE_DIR"

[ -f "$WHEEL" ] || { log "ABORT: staged wheel missing"; exit 1; }
if [ "$(pgrep -f 'MacOS/Python -u -m wnn.control.phased_ga' | wc -l | tr -d ' ')" != "0" ]; then
	log "ABORT: a controller is running — this boot must only run on an idle box."
	exit 1
fi

log "installing $(basename "$WHEEL")"
if ! "$VP" -m pip install --force-reinstall --no-deps "$WHEEL" >> "$LOG" 2>&1; then
	log "ABORT: pip install failed"
	exit 1
fi
if ! "$VP" -c "
import ram_controller as c, sys
sig = c.dagger_train_batch_inplace.__text_signature__ or ''
ok = ('target_levels' in sig and hasattr(c.WnnController, 'set_target_levels')
      and hasattr(c, 'arch_sample_framed1') and hasattr(c, 'arch_resample_suffix_scoped'))
sys.exit(0 if ok else 1)"; then
	log "ABORT: installed wheel is missing target_levels or the 16/08 samplers — refusing to continue"
	exit 1
fi
log "install verified: target_levels + framed1 sampler + scoped axonogenesis present"

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
		--translation --fit-weight-alt 16 \
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
smoke connscope --skip-stages neurons,bits --conn-mutation-scope feature \
                --conns-gens 1 --conns-patience 2 \
                --grid-bits 14 18 --grid-output-neurons 32 --max-output-neurons 32 || exit 1
grep -aq "\[target-levels\] T=32" "$SMOKE_DIR/tlevels.out" \
	|| { log "SMOKE FAILED (tlevels): no [target-levels] config line. Box left IDLE."; exit 1; }
grep -aq "\[conn-policy\] framed1" "$SMOKE_DIR/framed1.out" \
	|| { log "SMOKE FAILED (framed1): no [conn-policy] framed1 line. Box left IDLE."; exit 1; }
grep -aq "\[conn-scope\] feature" "$SMOKE_DIR/connscope.out" \
	|| { log "SMOKE FAILED (connscope): no [conn-scope] feature line. Box left IDLE."; exit 1; }
grep -aq "STAGE 3 (CONNECTIONS) done" "$SMOKE_DIR/connscope.out" \
	|| { log "SMOKE FAILED (connscope): CONNECTIONS stage never completed. Box left IDLE."; exit 1; }

CH=$("$VP" -c "
import subprocess
p = subprocess.Popen(['bash', 'scripts/sweep_ladder_chain.sh'], cwd='$ROOT',
	stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
print(p.pid)
")
log "########## BOOT COMPLETE — sweep_ladder_chain launched as PID $CH ##########"
