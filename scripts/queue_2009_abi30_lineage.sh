#!/usr/bin/env bash
# QUEUE 19/09/2026 (Luiz, 14:5x EDT) — THE ABI-30 LINEAGE RESTART.
# WHY a restart: the DAgger trainer never received the airframe (spec §5.1.3a; memory
# project_trainer_airframe_gap) — every cf21 run so far TRAINED on the synthetic plant and SCORED on
# cf21. Fixed 19/09 (evaluator passes ec.airframe_kwargs()), landed with the ABI-30 controller wheel
# (which also carries axis-F actuator lag). The lineage suffix carries the ABI on purpose: a marker's
# tag IS its identity — a same-suffix re-fly would be SKIPPED as "already banked", and paired_power
# pairs by suffix, so two trainer eras under one name would be read as one controller (the STRADDLE
# mistake). Order = Luiz 19/09 ("close what is in the middle before a new chapter"), unchanged.
#
#   STEP 1  `_hd30` x4 (s2-s5): the anchor on the fixed trainer. ARM_NO_CONTROL — paired era A/B vs
#           `_hd29` is printed by hand afterwards (same recipe, only the trainer's plant differs).
#           Lever: "what does training on the REAL cf21 plant change?"
#   STEP 2  `_pipeN30` x4 vs `_hd30` (CTRL-7 at n=4 on the new lineage; the 5th seed later if the CI needs it).
#   STEP 3  `_op30` x4 vs `_hd30` (CTRL-15: --obs-pwm only; secondary read vs `_bd` is CROSS-ERA — say so).
#   STEP 4  `_pon30` s2 vs `_hd30` (CTRL-10: plant ON, vertical features OFF; direction, n=1).
#   STEP 5  `_full30` x4 vs `_hd30`, second read vs `_pipeN30` (CTRL-16: GRID->NEURONS->BITS->CONNECTIONS->MEMORY).
#   STEP 6  LEVELS n=5 (CTRL-18, Luiz 19/09: "we are competing against controllers that are continuous"):
#           `_hd30` s6 (64 levels → n=5), then 96 levels (n384) and 128 levels (n512) at s2-s6, suffix `_L30`.
#           Off-chip deployment tier (16 MB QSPI) is a legitimate row group now — no flash gate on these.
#           Primary read = alt AND stable (the 96-level n=1 point was +2 pp / −7 cm), all four columns.
#           Controls are the `_hd30` seeds (n256 tags differ from n384/n512 tags → ARM_NO_CONTROL, paired by hand).
#   STEP 7  STOP — box idle, say so. Then the multi-axis round 1 (scripts/multi_axis_chain.sh, MA_ANCHOR_SUFFIX=_hd30).
# ~78 h for steps 1-5, ~48 h for step 6. Every step = scripts/seed_arm_chain.sh (idle-gated, marker-gated, fails closed, HOLD-aware).
# PRECONDITIONS (checked by preflight): installed ram_controller ABI == 30 AND the Python facade agrees
# (a half-landed wheel kills every fresh launch); phased_ga --help has --motor-lag-s; a cf21 EpisodeConfig
# packs k_thrust 0.2 (the trainer-airframe dead-plumbing check).
set -u
cd "$(dirname "$0")/.." || exit 1
LOG="/private/tmp/queue_2009.log"
MARK="experiments/sweepladder_markers"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
BASE="SL_C_b24n256_cf21_brushless_L4C_g10"
A="_hd30"
log() { echo "[q2009] $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
have() { [ -f "${MARK}/${BASE}_s$1$2.json" ]; }
count() { ls ${MARK}/${BASE}_s3133700[2-5]$1.json 2>/dev/null | wc -l | tr -d ' '; }
busy() { pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null || pgrep -f "scripts/sweep_ladder_gamm[a].sh" >/dev/null || pgrep -f "scripts/seed_arm_chai[n].sh" >/dev/null; }
pair() { PYTHONPATH=src/wnn $VP scripts/paired_power.py --arm "$1" --base "${BASE}_s{seed}" \
	--seed 31337002 --seed 31337003 --seed 31337004 --seed 31337005 --control-suffix "$2" 2>&1 | tee -a "$LOG"; }

log "########## ARMED — ABI-30 LINEAGE: _hd30 x4 -> _pipeN30 x4 -> _op30 x4 -> _pon30 s2 -> _full30 x4 -> STOP ##########"
busy && { log "ABORT — box not idle at arm time. Nothing launched."; exit 1; }
[ -f experiments/HOLD_CONTROLLER ] && { log "ABORT — experiments/HOLD_CONTROLLER present; rm it to arm."; exit 1; }
PYTHONPATH=src/wnn $VP - <<'PY' || { log "ABORT — preflight: wheel/facade/trainer-airframe check failed (see above). Nothing launched."; exit 1; }
import ram_controller as rc
from wnn.control import _accel
assert rc.ABI_VERSION == 30, f"installed ram_controller ABI {rc.ABI_VERSION} != 30"
from wnn.control import airframe as A
af = A._AIRFRAMES["cf21_brushless"]
from wnn.control.evaluator import _stage1_train_kwargs, _airframe_train_kwargs  # noqa: F401 — fix must exist
from wnn.control.training import EpisodeConfig
ec = EpisodeConfig(airframe=af, translation=True)
kw = {**_stage1_train_kwargs(ec), **_airframe_train_kwargs(ec)}
cfg = rc.RewardGatedConfigPacked(**kw)
assert abs(cfg.af_k_thrust - 0.2) < 1e-6, f"trainer k_thrust {cfg.af_k_thrust} (airframe NOT reaching the trainer)"
print(f"preflight OK: ABI 30, trainer plant k_thrust={cfg.af_k_thrust:.3f} inertia_x={cfg.af_inertia[0]:.2e} mass={cfg.af_mass:.4f}")
PY

if [ "$(count $A)" -ge 4 ]; then log "STEP 1 already complete (4/4 $A) — skipping"; else
	log "STEP 1 — $A x4: the anchor on the fixed trainer (real cf21 plant in the DAgger rollout)"
	ARM_SUFFIX="$A" ARM_LABEL="anchor-abi30-airframe-in-trainer" ARM_NO_CONTROL=1 \
		ARM_EXTRA_ARGS="--teacher-hover derived" ARM_REQUIRE_FLAGS="--teacher-hover --motor-lag-s" ARM_CTRL_SUFFIX="_hd29" \
		ARM_MARKER_JSON=",\"refly_of\":\"_hd29\",\"purpose\":\"ABI-30 anchor: trainer sees the cf21 plant (spec 5.1.3a)\"" \
		ARM_LOG="/private/tmp/seed_arm_hd30.log" bash scripts/seed_arm_chain.sh
	[ "$(count $A)" -ge 4 ] || { log "ABORT — $A $(count $A)/4. A run needs a human. Box left idle."; exit 1; }
	log "---------- ERA A/B (cross-trainer, descriptive): $A − _hd29 ----------"; pair "$A" "_hd29"
fi

if [ "$(count _pipeN30)" -ge 4 ]; then log "STEP 2 already complete (4/4 _pipeN30) — skipping"; else
	log "STEP 2 — CTRL-7: _pipeN30 x4 (GRID->NEURONS->MEMORY) vs $A"
	ARM_SUFFIX="_pipeN30" ARM_LABEL="pipeline-neurons" \
		ARM_EXTRA_ARGS="--skip-stages connections,bits --max-output-neurons 512 --max-cells 1000000000" \
		ARM_REQUIRE_FLAGS="--skip-stages --max-output-neurons" ARM_CTRL_SUFFIX="$A" \
		ARM_MARKER_JSON=",\"pipeline\":\"grid-neurons-memory\",\"max_output_neurons\":512,\"max_cells\":\"off\"" \
		ARM_LOG="/private/tmp/seed_arm_pipeN30.log" bash scripts/seed_arm_chain.sh
	[ "$(count _pipeN30)" -ge 4 ] || { log "ABORT — _pipeN30 $(count _pipeN30)/4. A run needs a human. Box left idle."; exit 1; }
fi

if [ "$(count _op30)" -ge 4 ]; then log "STEP 3 already complete (4/4 _op30) — skipping"; else
	log "STEP 3 — CTRL-15: _op30 x4 (--obs-pwm ONLY) vs $A"
	ARM_SUFFIX="_op30" ARM_LABEL="arm-b-obs-pwm-only" \
		ARM_EXTRA_ARGS="--obs-pwm" ARM_REQUIRE_FLAGS="--obs-pwm" ARM_CTRL_SUFFIX="$A" \
		ARM_MARKER_JSON=",\"obs_pwm\":true,\"dagger_label_delta\":false,\"cell\":\"2x2 obs-pwm-only\",\"secondary_control\":\"_bd (CROSS-ERA)\"" \
		ARM_LOG="/private/tmp/seed_arm_op30.log" bash scripts/seed_arm_chain.sh
	[ "$(count _op30)" -ge 4 ] || { log "ABORT — _op30 $(count _op30)/4. A run needs a human. Box left idle."; exit 1; }
	log "---------- CTRL-15 secondary read (CROSS-ERA — trainer differs): _op30 − _bd ----------"; pair "_op30" "_bd"
fi

if have 31337002 _pon30; then log "STEP 4 already banked (_pon30 s31337002) — skipping"; else
	log "STEP 4 — CTRL-10: _pon30 s31337002 (plant ON, --no-obs-collective-cmd --no-obs-alt-err --no-obs-vz) vs $A"
	ARM_SUFFIX="_pon30" ARM_LABEL="translation-plant-on-features-off" ARM_SEEDS="31337002" \
		ARM_EXTRA_ARGS="--no-obs-collective-cmd --no-obs-alt-err --no-obs-vz" \
		ARM_REQUIRE_FLAGS="--obs-collective-cmd --obs-alt-err --obs-vz" ARM_CTRL_SUFFIX="$A" \
		ARM_MARKER_JSON=",\"features\":\"vertical-off\",\"plant\":\"translation-on\",\"n\":1,\"read\":\"direction only\"" \
		ARM_LOG="/private/tmp/seed_arm_pon30.log" bash scripts/seed_arm_chain.sh
	have 31337002 _pon30 || { log "ABORT — _pon30 s31337002 marker missing. A run needs a human. Box left idle."; exit 1; }
fi

if [ "$(count _full30)" -ge 4 ]; then log "STEP 5 already complete (4/4 _full30) — skipping"; else
	log "STEP 5 — CTRL-16: _full30 x4 (GRID->NEURONS->BITS->CONNECTIONS->MEMORY, no skip) vs $A"
	ARM_SUFFIX="_full30" ARM_LABEL="pipeline-full-nbcm" \
		ARM_EXTRA_ARGS="--skip-stages , --bits-gens 5 --bits-patience 3 --max-output-neurons 512 --max-cells 1000000000" \
		ARM_REQUIRE_FLAGS="--skip-stages --bits-gens --max-output-neurons" ARM_CTRL_SUFFIX="$A" \
		ARM_MARKER_JSON=",\"pipeline\":\"grid-neurons-bits-connections-memory\",\"max_output_neurons\":512,\"max_cells\":\"off\",\"bits_budget\":\"5/3\"" \
		ARM_LOG="/private/tmp/seed_arm_full30.log" bash scripts/seed_arm_chain.sh
	[ "$(count _full30)" -ge 4 ] || { log "ABORT — _full30 $(count _full30)/4. A run needs a human. Box left idle."; exit 1; }
	log "---------- CTRL-16 second read: _full30 − _pipeN30 ----------"; pair "_full30" "_pipeN30"
fi

# ---- STEP 6: LEVELS n=5 — 64 (anchor s6), 96 (n384), 128 (n512), seeds s2-s6 ----------------
SEEDS5="31337002 31337003 31337004 31337005 31337006"
countn() { ls ${MARK}/SL_C_b24n$1_cf21_brushless_L4C_g10_s3133700[2-6]$2.json 2>/dev/null | wc -l | tr -d ' '; }
if have 31337006 $A; then log "STEP 6a already banked ($A s31337006)"; else
	log "STEP 6a — CTRL-18: $A s31337006 (64 levels → n=5)"
	ARM_SUFFIX="$A" ARM_LABEL="anchor-abi30-airframe-in-trainer" ARM_SEEDS="31337006" ARM_NO_CONTROL=1 \
		ARM_EXTRA_ARGS="--teacher-hover derived" ARM_REQUIRE_FLAGS="--teacher-hover --motor-lag-s" ARM_CTRL_SUFFIX="_hd29" \
		ARM_MARKER_JSON=",\"refly_of\":\"_hd29\",\"purpose\":\"5th seed of the 64-level rung (CTRL-18)\"" \
		ARM_LOG="/private/tmp/seed_arm_hd30.log" bash scripts/seed_arm_chain.sh
	have 31337006 $A || { log "ABORT — $A s31337006 missing. A run needs a human. Box left idle."; exit 1; }
fi
for N in 384 512; do
	LV=$((N / 4))
	if [ "$(countn $N _L30)" -ge 5 ]; then log "STEP 6 ${LV} levels already complete (5/5)"; continue; fi
	log "STEP 6 — CTRL-18: ${LV} levels (n${N}) x5 (s2-s6), suffix _L30; off-chip tier allowed, no flash gate"
	ARM_SUFFIX="_L30" ARM_LABEL="levels-${LV}-abi30" ARM_SEEDS="$SEEDS5" ARM_NEURONS="$N" ARM_NO_CONTROL=1 \
		ARM_EXTRA_ARGS="--teacher-hover derived" ARM_REQUIRE_FLAGS="--teacher-hover --motor-lag-s" ARM_CTRL_SUFFIX="$A" \
		ARM_MARKER_JSON=",\"levels_study\":true,\"levels\":${LV},\"control\":\"${A} n256 same seed\",\"deploy_tier\":\"off-chip allowed\"" \
		ARM_LOG="/private/tmp/seed_arm_L30_n${N}.log" bash scripts/seed_arm_chain.sh
	[ "$(countn $N _L30)" -ge 5 ] || { log "ABORT — ${LV} levels $(countn $N _L30)/5. A run needs a human. Box left idle."; exit 1; }
	log "---------- CTRL-18 read: ${LV} levels − 64 levels ($A), same seeds, n=5 ----------"
	PYTHONPATH=src/wnn $VP scripts/paired_power.py --arm "_L30" --base "SL_C_b24n${N}_cf21_brushless_L4C_g10_s{seed}" \
		$(for sd in $SEEDS5; do printf -- "--seed %s --control-override %s=${BASE}_s%s${A} " "$sd" "$sd" "$sd"; done) 2>&1 | tee -a "$LOG"
done

log "########## QUEUE COMPLETE — _hd30 5/5, _pipeN30 4/4, _op30 4/4, _pon30 1/1, _full30 4/4, levels 96/128 5/5; box IDLE. Next: multi_axis_chain.sh --round 1 (MA_ANCHOR_SUFFIX=_hd30) ##########"
