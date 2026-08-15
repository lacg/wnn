#!/usr/bin/env bash
# SPECIALIST-PROGRAMME INSTALL GUARD (14/08/2026 overnight, Luiz).
#
# Waits for the queued pipeline to DRAIN (4 calibab + 2 scopecost markers, 0
# controllers), then installs the staged specialist wheel (arm D full-window +
# batch-trainer flag; MIN_PER_CLUSTER is Python-side), verifies the new kwarg is
# genuinely importable, smokes FOUR pop-6 launches (legacy stage-1 recipe,
# min2 policy, full-window arm D, and D at stride 10 — each rc=0 AND zero FELL BACK), and only
# then launches specialist_round1_chain.sh.
#
# Same discipline as chunk_d_install_guard: never install mid-arm (the straddle
# that split the lambda sweep), never launch on a wheel that failed its smoke,
# and a failed smoke leaves the box IDLE deliberately — the status tick reports
# an idle box + pending item loudly.
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1

LOG="/private/tmp/specialist_guard.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
WHEEL="$ROOT/dist_staged/ram_controller-2026.212.37-cp313-cp313-macosx_11_0_arm64.whl"
SMOKE_DIR="/private/tmp/specialist_smoke"

log() { echo "[specguard] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controllers() { ps -axo pid,command 2>/dev/null | grep "wnn.control.phased_ga" \
	| grep -v "/usr/bin/time" | grep -v grep | grep -c python; }

mkdir -p "$SMOKE_DIR"
[ -f "$WHEEL" ] || { log "ABORT: staged wheel missing"; exit 1; }
log "########## ARMED — waiting for 4 calibab + 2 scopecost markers + 0 controllers ##########"

# ~14h calibab + ~7h scopecost remain; 60h ceiling = only fires if the pipeline died.
WAITED=0
while true; do
	NC=$(ls "$ROOT/experiments/calibab_markers" 2>/dev/null | wc -l | tr -d ' ')
	NS=$(ls "$ROOT/experiments/scopecost_markers" 2>/dev/null | wc -l | tr -d ' ')
	C=$(controllers)
	if [ "$NC" -ge 4 ] && [ "$NS" -ge 2 ] && [ "$C" -eq 0 ]; then
		log "gate open: calibab=$NC scopecost=$NS controllers=0 (waited ${WAITED}s)"
		break
	fi
	if [ "$WAITED" -ge 216000 ]; then
		log "ABORT: calibab=$NC scopecost=$NS controllers=$C after 60 h — pipeline likely died."
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
sys.exit(0 if 'output_full_window' in sig else 1)"; then
	log "ABORT: installed wheel does NOT expose output_full_window — refusing to continue"
	exit 1
fi
log "install verified: dagger_train_batch_inplace carries output_full_window"

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

smoke legacy  --grid-bits 24 30 --max-output-neurons 128                          || exit 1
smoke min2    --grid-bits 15 --max-output-neurons 256 --conn-policy min2          || exit 1
smoke fullwin --grid-bits 24 30 --max-output-neurons 128 --output-full-window     || exit 1
smoke stride10 --grid-bits 24 30 --max-output-neurons 128 --output-full-window --frame-stride 10 || exit 1

CH=$("$VP" -c "
import subprocess
p = subprocess.Popen(['bash', 'scripts/specialist_round1_chain.sh'], cwd='$ROOT',
	stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
print(p.pid)")
log "########## GUARD COMPLETE — specialist_round1_chain launched as PID $CH ##########"
