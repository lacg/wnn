#!/bin/bash
# L4 teacher screen — RERUN on the FIXED moment arm (05/08/2026).
#
# WHY THIS RERUN EXISTS. Every cf21_brushless number before 05/08/2026 was flown with
# `Airframe.arm_length` set to the published motor RADIUS where our '+'-config mixer needs
# the PER-AXIS moment arm (L = 2a = radius*sqrt(2)), so roll/pitch authority was 0.7071x
# the real vehicle's. Fixed by `axis_arm_from_radius`; see docs/disturbance_param_sources.md
# "MOTOR GEOMETRY". The defect was plant-level and hit every teacher identically, so it did
# not bias the ranking — but no number measured on it was faithful, so the screen is
# re-flown rather than rescued.
#
# ALSO NEW: the PID teacher is no longer the retired hand-tuned single loop. It is the
# firmware cascade from platform_defaults_cf21bl.h (wnn/control/pid_firmware.py +
# controller/pid_firmware.rs), so all five teachers now derive from the airframe.
#
# ORDER (set by Luiz, 05/08): closed-form arm FIRST, MPC family LAST.
#   - closed form (lqr, lqi, pid) x 2 seeds = 6 runs, NEURONS to natural early-stop
#   - MPC family (mpcof, mpc)     x 2 seeds = 4 runs, NEURONS CAPPED AT 5 GENERATIONS
# The cap makes the MPC arm NOT budget-matched to the closed-form one. Beating the
# closed-form mean is conclusive; losing is ambiguous between teacher quality and search
# budget and must be reported that way.
#
# Sequential by construction: l4_teacher_chain.sh aborts if another controller is already
# running, so the second arm cannot start until the first has exited. One controller at a
# time is a hard project rule (the IDS worker runs alongside and must never be disturbed).

set -u
ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
LOG="/private/tmp/l4_rerun_fixedplant.log"

say() { echo "[l4rerun] $(date -u +%FT%TZ) $*" >> "$LOG"; }

say "########## RERUN ARMED — fixed moment arm + firmware-cascade PID ##########"

say "===== ARM 1/2: closed form (lqr lqi pid), NEURONS to early-stop ====="
L4_TEACHERS="lqr lqi pid" L4_NEURONS_GENS=60 \
	bash scripts/l4_teacher_chain.sh
rc1=$?
say "ARM 1/2 finished rc=$rc1"

if [ "$rc1" -ne 0 ]; then
	say "ABORT: arm 1 returned rc=$rc1 — NOT starting the MPC arm. Investigate first."
	exit "$rc1"
fi

say "===== ARM 2/2: MPC family (mpcof mpc), NEURONS CAPPED AT 5 ====="
L4_TEACHERS="mpcof mpc" L4_NEURONS_GENS=5 \
	bash scripts/l4_teacher_chain.sh
rc2=$?
say "ARM 2/2 finished rc=$rc2"

say "########## RERUN DONE — arm1 rc=$rc1 arm2 rc=$rc2 ##########"
exit "$rc2"
