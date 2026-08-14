#!/usr/bin/env bash
# SCOPE C CHUNK D — install the staged controller wheel AT THE CHAIN BOUNDARY.
#
# WHY A GUARD AND NOT A MANUAL INSTALL. The chain launches every run as a NEW
# process, so a wheel installed mid-arm straddles a cohort across two wheels —
# which is exactly what split the lambda_alt sweep on 14/08/2026 and cost a run.
# The only safe moment is: the sweep's markers are ALL in, and nothing is flying.
#
# WHY IT ALSO OWNS calibab. calib_airframe_ab_chain.sh gates on the SAME
# condition this script waits for (10 stage1lambda markers AND 0 controllers), so
# leaving both armed is a race: if calibab wins, it starts a ~3h30m run on the OLD
# wheel and the install slips behind another cohort. The caller therefore KILLS the
# calibab waiter before arming this, and this script RELAUNCHES it after the
# install + smoke pass. calibab is a pure gate-waiter — restarting it loses nothing.
#
# ORDER (each step gates the next; any failure stops before anything flies):
#   1. wait for WAIT_COUNT markers + 0 controllers
#   2. install the staged wheel, verify the new symbol is really there
#   3. remove the _unpack transitional arity shim (git apply — fails safely if the
#      file moved under us, in which case the wheel is in and the shim is harmless)
#   4. smoke pop-6 stage 1 (channel OFF) and pop-6 stage 2 (channel ON), each
#      requiring rc=0 AND zero "FELL BACK" lines
#   5. relaunch calibab so the queued pipeline resumes
#
# A FAILED SMOKE DELIBERATELY LEAVES THE BOX IDLE. Launching a 4-run cohort on a
# wheel that just failed its smoke is worse than an idle box, and the status tick
# reports "box IDLE + pending item" loudly enough that a human sees it.
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1

LOG="/private/tmp/chunk_d_install.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
WHEEL="$ROOT/dist_staged/ram_controller-2026.212.37-cp313-cp313-macosx_11_0_arm64.whl"
PATCH="$ROOT/dist_staged/remove_unpack_shim.patch"
MARKERS="$ROOT/experiments/stage1lambda_markers"
WAIT_COUNT="${CD_WAIT_COUNT:-10}"
SMOKE_DIR="/private/tmp/chunk_d_smoke"

log() { echo "[chunkd] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controllers() { ps -axo pid,command 2>/dev/null | grep "wnn.control.phased_ga" \
	| grep -v "/usr/bin/time" | grep -v grep | grep -c python; }

mkdir -p "$SMOKE_DIR"
log "########## ARMED — waiting for $WAIT_COUNT markers in $(basename "$MARKERS") + 0 controllers ##########"

[ -f "$WHEEL" ] || { log "ABORT: staged wheel missing at $WHEEL"; exit 1; }
[ -f "$PATCH" ] || { log "ABORT: shim patch missing at $PATCH"; exit 1; }

# ---- 1. gate -----------------------------------------------------------------
# 40 h ceiling: 5 remaining runs at ~3h30m is ~18 h, so this only fires if the
# sweep died. Waiting forever would hide that.
WAITED=0
while true; do
	N=$(ls "$MARKERS" 2>/dev/null | wc -l | tr -d ' ')
	C=$(controllers)
	if [ "$N" -ge "$WAIT_COUNT" ] && [ "$C" -eq 0 ]; then
		log "gate open: markers=$N controllers=0 (waited ${WAITED}s)"
		break
	fi
	if [ "$WAITED" -ge 144000 ]; then
		log "ABORT: markers=$N controllers=$C after 40 h — the sweep likely died. Not installing."
		exit 1
	fi
	sleep 300
	WAITED=$((WAITED + 300))
done

# ---- 2. install --------------------------------------------------------------
log "installing $(basename "$WHEEL")"
if ! "$VP" -m pip install --force-reinstall --no-deps "$WHEEL" >> "$LOG" 2>&1; then
	log "ABORT: pip install failed"
	exit 1
fi
# The install is only real if the new signature is actually importable. A wheel
# that installs but does not carry the symbol is the silent-skew failure again.
if ! "$VP" -c "
import ram_controller as c, sys
sig = c.score_classical_baseline.__text_signature__ or ''
sys.exit(0 if 's2_init_x' in sig else 1)"; then
	log "ABORT: installed wheel does NOT expose s2_init_x — refusing to continue"
	exit 1
fi
log "install verified: score_classical_baseline carries s2_init_x"

# ---- 3. drop the transitional shim -------------------------------------------
# git apply --check first so a moved file is a clean skip, not a half-applied
# patch. The shim is harmless if it survives (it just tolerates an arity that no
# longer occurs), so this step must never be fatal.
if git apply --check "$PATCH" 2>/dev/null; then
	git apply "$PATCH" && log "removed the _unpack transitional shim"
else
	log "WARN: shim patch no longer applies (file changed) — leaving the shim in, remove by hand"
fi

# ---- 4. smoke ----------------------------------------------------------------
# Reuses the sweep's own recipe shape at pop-6 so the smoke exercises the real
# stack: grid -> neurons -> memory -> held-out (which is where the rival scorer,
# i.e. the whole point of chunk D, actually runs).
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
		--grid-bits 24 30 --grid-state-neurons 0 --max-state-neurons 0 \
		--max-output-neurons 128 --runs 1 --memory-mode BINARY \
		--airframe cf21_brushless --disturbance L4C --teacher mpcof \
		--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i \
		--obs-collective-cmd --obs-alt-err --obs-vz \
		--translation --fit-weight-alt 16 \
		"$@" \
		--report-seeds 99990101 --base-seed 31337003 > "$out" 2>&1
	local rc=$?
	local fb; fb=$(grep -ac "FELL BACK" "$out")
	if [ "$rc" != "0" ] || [ "$fb" != "0" ]; then
		log "SMOKE FAILED ($name): rc=$rc fallback_lines=$fb — see $out"
		return 1
	fi
	log "smoke $name PASSED (rc=0, 0 fallback lines)"
	return 0
}

if ! smoke stage1; then
	log "ABORT after stage-1 smoke — box left IDLE deliberately; calibab NOT relaunched"
	exit 1
fi
if ! smoke stage2 --obs-pos-err-xy --obs-vel-xy --xy-offset 1.0; then
	log "ABORT after stage-2 smoke — box left IDLE deliberately; calibab NOT relaunched"
	exit 1
fi

# ---- 5. resume the queued pipeline -------------------------------------------
# start_new_session via python: macOS has no setsid(1), and a chain inheriting
# this script's session would die with it.
CAB=$("$VP" -c "
import subprocess
p = subprocess.Popen(['bash', 'scripts/calib_airframe_ab_chain.sh'], cwd='$ROOT',
	stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
print(p.pid)")
log "########## CHUNK D COMPLETE — calibab relaunched as PID $CAB ##########"
