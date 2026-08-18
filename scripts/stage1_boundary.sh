#!/usr/bin/env bash
# CHAIN-BOUNDARY AUTOMATION (13/08/2026) — hand the box from the L1 re-fly to
# the stage-1 λ_alt sweep without an idle overnight, WITHOUT ever breaking the
# "never deploy while a chain is armed; smoke ONE first" rule.
#
# ORDER (the pending list, encoded):
#   1. wait for the live chain to finish  — 4 markers AND no controller process
#   2. build + install the controller wheel (motor lag + estimator + stage 1;
#      ONE install, not three — they all landed since the last one)
#   3. verify the wheel imports and reports the expected surface
#   4. SMOKE: one tiny pop-6 stage-1 launch, ~2 min
#   5. ONLY IF the smoke passes: arm the λ_alt sweep
#
# A failing smoke STOPS here and leaves the box idle. That is the correct
# outcome: three cohorts died in a day the last time a chain was armed on an
# unsmoked wheel (feedback_never_deploy_while_chain_armed), and an idle box
# costs one night while a bad cohort costs the runs AND the trust in them.
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
LOG="/private/tmp/stage1_boundary.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
ACCEL="$ROOT/src/wnn/ram/strategies/accelerator"
WATCH_MARKERS="$ROOT/experiments/l1refly_markers"
EXPECT_MARKERS="${BOUNDARY_EXPECT_MARKERS:-4}"

log() { echo "[boundary] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controllers() { ps -axo pid,command 2>/dev/null | grep "wnn.control.phased_ga" \
	| grep -v "/usr/bin/time" | grep -v grep | grep -c python; }

log "########## ARMED — waiting for the live chain (expect $EXPECT_MARKERS markers) ##########"

# ---- 1. wait for the boundary -------------------------------------------
WAITED=0
while true; do
	N=$(ls "$WATCH_MARKERS" 2>/dev/null | wc -l | tr -d ' ')
	C=$(controllers)
	if [ "$N" -ge "$EXPECT_MARKERS" ] && [ "$C" -eq 0 ]; then
		log "boundary reached: markers=$N controllers=0 (waited ${WAITED}s)"
		break
	fi
	# Safety valve: never wait more than 6 h — if the chain died mid-run the
	# markers never complete, and silently waiting forever hides that.
	if [ "$WAITED" -ge 21600 ]; then
		log "ABORT: still markers=$N controllers=$C after 6 h — the chain likely died. \
Not installing, not arming. Investigate before anything flies."
		exit 1
	fi
	sleep 120
	WAITED=$((WAITED + 120))
done

# ---- 2. install the wheel ------------------------------------------------
log "installing the controller wheel (motor lag + Mahony estimator + scope C stage 1)"
# shellcheck disable=SC1091
if ! ( cd "$ACCEL" && unset CONDA_PREFIX && . /Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/activate \
	&& maturin develop --release -m controller/Cargo.toml ) >> "$LOG" 2>&1; then
	log "ABORT: wheel build/install FAILED — nothing armed."
	exit 1
fi
log "wheel installed"

# ---- 3. verify the surface ----------------------------------------------
if ! "$VP" - >> "$LOG" 2>&1 <<'PYEOF'
import ram_controller as rc
from wnn.control._accel import AttitudeSim
print("[verify] ABI", rc.ABI_VERSION)
sim = AttitudeSim(dt=0.001, arm_length=0.0707, k_thrust=0.2, k_drag=0.0057,
                  inertia=[1.66e-5, 1.66e-5, 2.93e-5], gravity=9.81)
assert hasattr(sim, "set_translation"), "wheel lacks stage-1 translation"
assert hasattr(sim, "set_motor_lag"), "wheel lacks motor lag"
assert hasattr(rc, "MahonyEstimatorRs"), "wheel lacks the Mahony estimator"
sim.set_translation(0.0393)
h = sim.hover_pwm()
assert abs(h - 0.6942) < 1e-3, f"hover_pwm {h} is not the cf21 value"
for _ in range(1000):
    sim.step([0.0, 0.0, 0.0, 0.0])
# NOTE: altitude / vertical_velocity are #[getter]s — ATTRIBUTES, not methods.
# Calling them cost the 13/08 boundary run a false abort.
vz = sim.vertical_velocity
assert abs(vz + 9.81) < 0.02, f"drop test does not fall at g (vz={vz})"
assert abs(sim.altitude + 4.905) < 0.02, f"drop test z wrong ({sim.altitude})"
print("[verify] stage-1 surface OK (hover_pwm %.4f, free fall %.3f m/s, z %.3f m)"
      % (h, vz, sim.altitude))
PYEOF
then
	log "ABORT: wheel surface verification FAILED — nothing armed."
	exit 1
fi
log "surface verified"

# ---- 4. SMOKE: one tiny pop-6 stage-1 launch ----------------------------
log "smoke: pop-6 stage-1 launch (never arm a cohort on an unsmoked wheel)"
SMOKE_LOG="$ROOT/logs/controller/stage1_smoke.out"
mkdir -p "$ROOT/logs/controller"
if ! "$VP" -u -m wnn.control.phased_ga \
	--levels 16 --skip-stages bits,connections \
	--neurons-gens 1 --memory-gens 1 --pop 6 \
	--num-eval-folds 5 --eval-episodes 4 --memory-eval-episodes 4 \
	--steps 200 --tilt 5.0 \
	--grid-bits 24 --grid-state-neurons 0 --max-state-neurons 0 \
	--max-output-neurons 64 --runs 1 --memory-mode BINARY \
	--airframe cf21_brushless --disturbance L4C --teacher mpcof \
	--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw \
	--obs-yaw-err --obs-yaw-err-i \
	--obs-collective-cmd --obs-alt-err --obs-vz \
	--translation --reward-lambda-alt 4.0 \
	--report-episodes 4 --report-seeds 99990101 \
	--base-seed 31337099 > "$SMOKE_LOG" 2>&1; then
	log "ABORT: SMOKE FAILED (rc != 0) — see $SMOKE_LOG. Nothing armed."
	exit 1
fi
# A zero exit is necessary but not sufficient: the run must have produced a
# held-out line, i.e. it actually flew rather than exiting early.
if ! grep -qE "RESULT|held-out" "$SMOKE_LOG"; then
	log "ABORT: smoke exited 0 but produced no held-out result — see $SMOKE_LOG. Nothing armed."
	exit 1
fi
log "smoke PASSED: $(grep -aE 'RESULT|held-out' "$SMOKE_LOG" | tail -1)"

# ---- 5. arm the sweep ---------------------------------------------------
log "arming the stage-1 lambda_alt sweep"
setsid_or_bg() {
	# macOS has no setsid; start_new_session via python is the house idiom
	# (feedback_detach_background_processes) so the chain survives this script.
	"$VP" - "$@" <<'PYEOF'
import os, subprocess, sys
p = subprocess.Popen(sys.argv[1:], start_new_session=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(p.pid)
PYEOF
}
PID=$(setsid_or_bg /bin/bash "$ROOT/scripts/stage1_lambda_alt_sweep.sh")
log "sweep armed, pid=$PID (log /private/tmp/stage1_lambda_sweep.log)"
log "########## BOUNDARY COMPLETE ##########"
