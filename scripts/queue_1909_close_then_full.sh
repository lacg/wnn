#!/usr/bin/env bash
# QUEUE 19/09/2026 (Luiz, 12:1x EDT): "close something that is in the middle first, before we
# start a new chapter" — CTRL-7 → CTRL-15 → CTRL-10, THEN the full-pipeline arm (CTRL-16).
# Every step is scripts/seed_arm_chain.sh (idle-gated, marker-gated, fails closed, honours
# experiments/HOLD_CONTROLLER through the ladder). Log /private/tmp/queue_1909.log.
#
#   STEP 1  CTRL-7 5th seed, control half: `_hd29` on s31337006 (ARM_NO_CONTROL — no `_hd` s6 exists).
#           Lever: "clean control at a 5th seed so the err CI of stage D can be settled".
#   STEP 2  CTRL-7 5th seed, arm half: `_pipeN` on s31337006 vs `_hd29`; then the n=5 paired verdict
#           (seeds 2-6) — the honest read that the Done comment called "err unresolved at n=4".
#   STEP 3  CTRL-15 `_op` ×4 (s2-s5): --obs-pwm WITHOUT --dagger-label-delta, vs `_hd29`
#           (secondary read vs `_bd` by hand). Lever: "does the pwm OBSERVATION alone carry arm B's
#           err gain, and alone cause its 6-7x key blow-up?"
#   STEP 4  CTRL-10 `_pon` s2 only: plant ON (--translation stays), the three vertical features OFF
#           (--no-obs-collective-cmd --no-obs-alt-err --no-obs-vz), vs `_hd29` s2. Lever: "split the
#           translation regimen's ~0.3° err / ~0.5° steady cost into plant vs features" (direction, n=1).
#   STEP 5  CTRL-16 `_full` ×4 (s2-s5): the NATIVE phased order GRID→NEURONS→BITS→CONNECTIONS→MEMORY —
#           no --skip-stages (passed as ","), BITS budgeted like the other arch stages (5 gens /
#           patience 3 — the ladder never set --bits-gens because it never ran BITS), neuron cap
#           opened like `_pipeN` (512, --max-cells off). vs `_hd29`; second read vs `_pipeN` by hand.
#           Lever: "do the stages COMPOUND, or is it diminishing returns after NEURONS?" Smoke-tested
#           at a tiny budget on 19/09 before arming (scratchpad/full_smoke).
#   STEP 6  STOP — box idle, say so.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG="/private/tmp/queue_1909.log"
MARK="experiments/sweepladder_markers"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
BASE="SL_C_b24n256_cf21_brushless_L4C_g10"
log() { echo "[q1909] $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
have() { [ -f "${MARK}/${BASE}_s$1$2.json" ]; }
count() { ls ${MARK}/${BASE}_s3133700[2-5]$1.json 2>/dev/null | wc -l | tr -d ' '; }
busy() { pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null || pgrep -f "scripts/sweep_ladder_gamm[a].sh" >/dev/null || pgrep -f "scripts/seed_arm_chai[n].sh" >/dev/null; }

log "########## ARMED — CTRL-7 5th seed (_hd29 s6 -> _pipeN s6) -> CTRL-15 _op x4 -> CTRL-10 _pon s2 -> CTRL-16 _full x4 -> STOP ##########"
busy && { log "ABORT — box not idle at arm time. Nothing launched."; exit 1; }

# ---- STEP 1: _hd29 on the 5th seed (no control exists at s6) -------------------------
if have 31337006 _hd29; then log "STEP 1 already banked (_hd29 s31337006) — skipping"; else
	log "STEP 1 — _hd29 on s31337006 (clean control, 5th seed; no paired verdict)"
	ARM_SUFFIX="_hd29" ARM_LABEL="d0-derived-hover-abi29-refly" ARM_SEEDS="31337006" ARM_NO_CONTROL=1 \
		ARM_EXTRA_ARGS="--teacher-hover derived" ARM_REQUIRE_FLAGS="--teacher-hover" \
		ARM_CTRL_SUFFIX="_hd" \
		ARM_MARKER_JSON=",\"refly_of\":\"_hd\",\"purpose\":\"5th-seed control for CTRL-7 (stage D err CI)\"" \
		ARM_LOG="/private/tmp/hd29_s6.log" bash scripts/seed_arm_chain.sh
	have 31337006 _hd29 || { log "ABORT — _hd29 s31337006 marker missing. A run needs a human. Box left idle."; exit 1; }
fi

# ---- STEP 2: _pipeN on the 5th seed, then the n=5 verdict ---------------------------
if have 31337006 _pipeN; then log "STEP 2 already banked (_pipeN s31337006) — skipping"; else
	log "STEP 2 — _pipeN on s31337006 vs _hd29 (CTRL-7 5th seed)"
	ARM_SUFFIX="_pipeN" ARM_LABEL="pipeline-neurons" ARM_SEEDS="31337006" \
		ARM_EXTRA_ARGS="--skip-stages connections,bits --max-output-neurons 512 --max-cells 1000000000" \
		ARM_REQUIRE_FLAGS="--skip-stages --max-output-neurons" ARM_CTRL_SUFFIX="_hd29" \
		ARM_MARKER_JSON=",\"pipeline\":\"grid-neurons-memory\",\"max_output_neurons\":512,\"max_cells\":\"off\"" \
		ARM_LOG="/private/tmp/pipeN_s6.log" bash scripts/seed_arm_chain.sh
	have 31337006 _pipeN || { log "ABORT — _pipeN s31337006 marker missing. A run needs a human. Box left idle."; exit 1; }
fi
log "---------- CTRL-7 VERDICT at n=5: _pipeN − _hd29, seeds 31337002-06 (mean paired delta + 95% CI) ----------"
PYTHONPATH=src/wnn $VP scripts/paired_power.py --arm _pipeN --base "${BASE}_s{seed}" \
	--seed 31337002 --seed 31337003 --seed 31337004 --seed 31337005 --seed 31337006 \
	--control-suffix _hd29 2>&1 | tee -a "$LOG"

# ---- STEP 3: CTRL-15 _op x4 -----------------------------------------------------------
if [ "$(count _op)" -ge 4 ]; then log "STEP 3 already complete (4/4 _op) — skipping"; else
	log "STEP 3 — CTRL-15: _op x4 (--obs-pwm ONLY) vs _hd29"
	ARM_SUFFIX="_op" ARM_LABEL="arm-b-obs-pwm-only" \
		ARM_EXTRA_ARGS="--obs-pwm" ARM_REQUIRE_FLAGS="--obs-pwm" ARM_CTRL_SUFFIX="_hd29" \
		ARM_MARKER_JSON=",\"obs_pwm\":true,\"dagger_label_delta\":false,\"cell\":\"2x2 obs-pwm-only\",\"secondary_control\":\"_bd\"" \
		ARM_LOG="/private/tmp/seed_arm_op.log" bash scripts/seed_arm_chain.sh
	[ "$(count _op)" -ge 4 ] || { log "ABORT — _op chain exited with $(count _op)/4 markers. A run needs a human. Box left idle."; exit 1; }
	log "---------- CTRL-15 secondary read: _op − _bd ----------"
	PYTHONPATH=src/wnn $VP scripts/paired_power.py --arm _op --base "${BASE}_s{seed}" \
		--seed 31337002 --seed 31337003 --seed 31337004 --seed 31337005 --control-suffix _bd 2>&1 | tee -a "$LOG"
fi

# ---- STEP 4: CTRL-10 _pon s2 (plant ON, vertical features OFF) -------------------------
if have 31337002 _pon; then log "STEP 4 already banked (_pon s31337002) — skipping"; else
	log "STEP 4 — CTRL-10: _pon s31337002 (plant ON, --no-obs-collective-cmd --no-obs-alt-err --no-obs-vz) vs _hd29"
	ARM_SUFFIX="_pon" ARM_LABEL="translation-plant-on-features-off" ARM_SEEDS="31337002" \
		ARM_EXTRA_ARGS="--no-obs-collective-cmd --no-obs-alt-err --no-obs-vz" \
		ARM_REQUIRE_FLAGS="--obs-collective-cmd --obs-alt-err --obs-vz" ARM_CTRL_SUFFIX="_hd29" \
		ARM_MARKER_JSON=",\"features\":\"vertical-off\",\"plant\":\"translation-on\",\"n\":1,\"read\":\"direction only\"" \
		ARM_LOG="/private/tmp/seed_arm_pon.log" bash scripts/seed_arm_chain.sh
	have 31337002 _pon || { log "ABORT — _pon s31337002 marker missing. A run needs a human. Box left idle."; exit 1; }
fi

# ---- STEP 5: CTRL-16 _full x4 (the new chapter, last) --------------------------------
if [ "$(count _full)" -ge 4 ]; then log "STEP 5 already complete (4/4 _full) — skipping"; else
	log "STEP 5 — CTRL-16: _full x4 (GRID->NEURONS->BITS->CONNECTIONS->MEMORY, no skip) vs _hd29"
	ARM_SUFFIX="_full" ARM_LABEL="pipeline-full-nbcm" \
		ARM_EXTRA_ARGS="--skip-stages , --bits-gens 5 --bits-patience 3 --max-output-neurons 512 --max-cells 1000000000" \
		ARM_REQUIRE_FLAGS="--skip-stages --bits-gens --max-output-neurons" ARM_CTRL_SUFFIX="_hd29" \
		ARM_MARKER_JSON=",\"pipeline\":\"grid-neurons-bits-connections-memory\",\"max_output_neurons\":512,\"max_cells\":\"off\",\"bits_budget\":\"5/3\"" \
		ARM_LOG="/private/tmp/seed_arm_full.log" bash scripts/seed_arm_chain.sh
	[ "$(count _full)" -ge 4 ] || { log "ABORT — _full chain exited with $(count _full)/4 markers. A run needs a human. Box left idle."; exit 1; }
	log "---------- CTRL-16 second read: _full − _pipeN ----------"
	PYTHONPATH=src/wnn $VP scripts/paired_power.py --arm _full --base "${BASE}_s{seed}" \
		--seed 31337002 --seed 31337003 --seed 31337004 --seed 31337005 --control-suffix _pipeN 2>&1 | tee -a "$LOG"
fi

log "########## QUEUE COMPLETE — CTRL-7 n=5, CTRL-15 4/4, CTRL-10 1/1, CTRL-16 4/4; nothing else queued; box IDLE ##########"
