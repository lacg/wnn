#!/usr/bin/env bash
# QUEUE 2009b — the ABI-30 lineage WITH THE PROMOTION GATE (Luiz, 19/09/2026 16:0x EDT).
# Supersedes scripts/queue_2009_abi30_lineage.sh after its STEP 1 chain (`_hd30` x4) — that
# supervisor was killed at the step boundary (its chain kept flying; nothing was interrupted).
#
# WHY A GATE. "Control" and "winner" are different jobs. `_hd` was the one-factor baseline every arm
# was built on; `_pipeN` (GRID->GA-NEURONS->MEMORY) beat it on every headline column at n=4 on the
# ABI-29 trainer. The multi-axis programme should anchor on the best DEPLOYABLE recipe, and the
# remaining arms (`_op`, `_pon`, levels) should be flown on the recipe we would ship — otherwise they
# measure a controller we would not use. But the trainer changed (airframe fix), so the promotion is
# re-decided on the fixed trainer, not assumed.
#
#   STEP 1  (wait) `_hd30` x4 — flying under seed_arm_chain.sh from queue_2009. Then era A/B `_hd30 − _hd29`.
#   STEP 2  `_pipeN30` x4 vs `_hd30` (CTRL-7 at n=4 on the new lineage).
#   GATE    PRE-REGISTERED: PROMOTE the neurons-GA pipeline iff, over the 4 paired seeds (HEADLINE rows),
#           the mean delta `_pipeN30 − _hd30` favours `_pipeN30` on >= 3 of the 4 columns
#           (stable up, err down, steady down, alt down) AND err is not worse by more than 0.10 deg.
#           Promoted  -> RECIPE_EXTRA = the `_pipeN` stage flags, ANCHOR = `_pipeN30`.
#           Otherwise -> RECIPE_EXTRA empty (control pipeline), ANCHOR = `_hd30`.
#           The gate result is written to experiments/promotion_gate_2009.json and logged. It is a
#           RECIPE decision, not a result — the arms' verdicts are read against the chosen anchor.
#   STEP 3  `_op30` x4 (--obs-pwm only) on RECIPE vs ANCHOR (CTRL-15).
#   STEP 4  `_pon30` s2 (plant ON, vertical features OFF) on RECIPE vs ANCHOR (CTRL-10).
#   STEP 5  `_full30` x4 (GRID->NEURONS->BITS->CONNECTIONS->MEMORY) vs `_hd30`, second read vs `_pipeN30` (CTRL-16).
#           (Its pipeline IS the factor, so RECIPE_EXTRA does not apply.)
#   STEP 6  LEVELS n=5 (CTRL-18): ANCHOR s6 (64 levels), then 96 (n384) and 128 (n512) `_L30` x5 on RECIPE;
#           read against the same-seed ANCHOR (n256). Off-chip QSPI tier allowed.
#   STEP 7  STOP. Then multi_axis_chain.sh --round 1 with MA_ANCHOR_SUFFIX=$ANCHOR (and, if promoted, the
#           recipe flags — see the TODO printed at the end).
set -u
cd "$(dirname "$0")/.." || exit 1
LOG="/private/tmp/queue_2009.log"
MARK="experiments/sweepladder_markers"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
BASE="SL_C_b24n256_cf21_brushless_L4C_g10"
GATE_JSON="experiments/promotion_gate_2009.json"
PIPE_FLAGS="--skip-stages connections,bits --max-output-neurons 512 --max-cells 1000000000"
SEEDS4="31337002 31337003 31337004 31337005"
SEEDS5="$SEEDS4 31337006"
log() { echo "[q2009b] $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
have() { [ -f "${MARK}/${BASE}_s$1$2.json" ]; }
count() { ls ${MARK}/${BASE}_s3133700[2-5]$1.json 2>/dev/null | wc -l | tr -d ' '; }
countn() { ls ${MARK}/SL_C_b24n$1_cf21_brushless_L4C_g10_s3133700[2-6]$2.json 2>/dev/null | wc -l | tr -d ' '; }
busy() { pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null || pgrep -f "scripts/sweep_ladder_gamm[a].sh" >/dev/null || pgrep -f "scripts/seed_arm_chai[n].sh" >/dev/null; }
pair() { PYTHONPATH=src/wnn $VP scripts/paired_power.py --arm "$1" --base "${BASE}_s{seed}" \
	$(for sd in $SEEDS4; do printf -- '--seed %s ' "$sd"; done) --control-suffix "$2" 2>&1 | tee -a "$LOG"; }
arm() {  # arm <suffix> <label> <seeds> <extra_args> <require_flags> <ctrl_suffix> <marker_json> [neurons] [no_control]
	ARM_SUFFIX="$1" ARM_LABEL="$2" ARM_SEEDS="$3" ARM_EXTRA_ARGS="$4" ARM_REQUIRE_FLAGS="$5" ARM_CTRL_SUFFIX="$6" \
		ARM_MARKER_JSON="$7" ARM_NEURONS="${8:-256}" ARM_NO_CONTROL="${9:-0}" \
		ARM_LOG="/private/tmp/seed_arm$1${8:+_n$8}.log" bash scripts/seed_arm_chain.sh
}

log "########## ARMED (2009b) — wait _hd30 x4 -> _pipeN30 x4 -> PROMOTION GATE -> _op30 -> _pon30 -> _full30 -> LEVELS n=5 -> STOP ##########"
[ -f experiments/HOLD_CONTROLLER ] && { log "ABORT — experiments/HOLD_CONTROLLER present; rm it to arm."; exit 1; }

# ---- STEP 1: wait for the _hd30 chain launched by queue_2009 ------------------------------------
beat=0
while busy; do sleep 60; beat=$((beat + 1)); [ $((beat % 30)) = 0 ] && log "waiting — _hd30 chain flying ($(count _hd30)/4)"; done
if [ "$(count _hd30)" -lt 4 ]; then
	log "STEP 1 — _hd30 chain exited with $(count _hd30)/4; relaunching the same idempotent chain for the missing seeds"
	arm "_hd30" "anchor-abi30-airframe-in-trainer" "$SEEDS4" "--teacher-hover derived" "--teacher-hover --motor-lag-s" "_hd29" \
		",\"refly_of\":\"_hd29\",\"purpose\":\"ABI-30 anchor: trainer sees the cf21 plant (spec 5.1.3a)\"" 256 1
	[ "$(count _hd30)" -ge 4 ] || { log "ABORT — _hd30 $(count _hd30)/4. A run needs a human. Box left idle."; exit 1; }
fi
log "---------- ERA A/B (cross-trainer, descriptive): _hd30 − _hd29 ----------"; pair "_hd30" "_hd29"

# ---- STEP 2: _pipeN30 x4 vs _hd30 ----------------------------------------------------------------
if [ "$(count _pipeN30)" -ge 4 ]; then log "STEP 2 already complete (4/4 _pipeN30)"; else
	log "STEP 2 — CTRL-7: _pipeN30 x4 (GRID->NEURONS->MEMORY) vs _hd30"
	arm "_pipeN30" "pipeline-neurons" "$SEEDS4" "$PIPE_FLAGS" "--skip-stages --max-output-neurons" "_hd30" \
		",\"pipeline\":\"grid-neurons-memory\",\"max_output_neurons\":512,\"max_cells\":\"off\""
	[ "$(count _pipeN30)" -ge 4 ] || { log "ABORT — _pipeN30 $(count _pipeN30)/4. A run needs a human. Box left idle."; exit 1; }
fi

# ---- GATE: promote the neurons-GA pipeline? (pre-registered, HEADLINE rows, n=4 paired) ---------
log "---------- PROMOTION GATE: _pipeN30 − _hd30 on the HEADLINE rows ----------"
PROMOTE=$($VP - "$MARK" "$BASE" "$GATE_JSON" <<'PY'
import json, re, sys, statistics as st
mark, base, out = sys.argv[1:]
def hl(tag):
	d = json.load(open(f"{mark}/{tag}.json"))
	m = re.search(r"stable=([\d.]+)%\s+err=([\d.]+)°\s+steady=([\d.]+)°.*?alt=([\d.]+)m", d["headline_holdout"])
	return [float(x) for x in m.groups()]
seeds = ["31337002", "31337003", "31337004", "31337005"]
d = [[a - c for a, c in zip(hl(f"{base}_s{s}_pipeN30"), hl(f"{base}_s{s}_hd30"))] for s in seeds]
mean = [st.mean(col) for col in zip(*d)]
# favourable direction: stable UP, err DOWN, steady DOWN, alt DOWN
fav = [mean[0] > 0, mean[1] < 0, mean[2] < 0, mean[3] < 0]
promote = sum(fav) >= 3 and mean[1] <= 0.10
res = dict(rule="promote iff >=3/4 columns favour _pipeN30 (stable up, err down, steady down, alt down) AND err delta <= +0.10 deg",
           mean_delta=dict(stable_pp=round(mean[0], 3), err_deg=round(mean[1], 3), steady_deg=round(mean[2], 3), alt_m=round(mean[3], 4)),
           per_seed=dict(zip(seeds, [[round(x, 3) for x in row] for row in d])), favourable=fav, promote=promote)
json.dump(res, open(out, "w"), indent=1)
print("PROMOTE" if promote else "KEEP", file=sys.stderr)
print("1" if promote else "0")
PY
)
case "$PROMOTE" in 0|1) ;; *) log "ABORT — the gate did not produce a verdict (markers missing or parse error). A human decides. Box left idle."; exit 1;; esac
cat "$GATE_JSON" | tee -a "$LOG"
if [ "$PROMOTE" = "1" ]; then
	RECIPE="$PIPE_FLAGS"; RECIPE_NAME="grid-neurons-memory"; ANCHOR="_pipeN30"; REQ="--skip-stages --max-output-neurons"
	log "GATE: PROMOTED — recipe = neurons-GA pipeline; ANCHOR = _pipeN30 for steps 3, 4, 6 and the multi-axis round 1"
else
	RECIPE=""; RECIPE_NAME="grid-connections-memory"; ANCHOR="_hd30"; REQ=""
	log "GATE: KEPT — control pipeline stays the recipe; ANCHOR = _hd30"
fi
RJ=",\"recipe\":\"${RECIPE_NAME}\",\"anchor\":\"${ANCHOR}\",\"gate\":\"${GATE_JSON}\""

# ---- STEP 3: _op30 x4 on RECIPE vs ANCHOR ---------------------------------------------------------
if [ "$(count _op30)" -ge 4 ]; then log "STEP 3 already complete (4/4 _op30)"; else
	log "STEP 3 — CTRL-15: _op30 x4 (--obs-pwm ONLY) on ${RECIPE_NAME} vs ${ANCHOR}"
	arm "_op30" "arm-b-obs-pwm-only" "$SEEDS4" "--obs-pwm ${RECIPE}" "--obs-pwm ${REQ}" "$ANCHOR" \
		",\"obs_pwm\":true,\"dagger_label_delta\":false,\"cell\":\"2x2 obs-pwm-only\",\"secondary_control\":\"_bd (CROSS-ERA)\"${RJ}"
	[ "$(count _op30)" -ge 4 ] || { log "ABORT — _op30 $(count _op30)/4. A run needs a human. Box left idle."; exit 1; }
	log "---------- CTRL-15 secondary read (CROSS-ERA): _op30 − _bd ----------"; pair "_op30" "_bd"
fi

# ---- STEP 4: _pon30 s2 on RECIPE vs ANCHOR --------------------------------------------------------
if have 31337002 _pon30; then log "STEP 4 already banked (_pon30 s31337002)"; else
	log "STEP 4 — CTRL-10: _pon30 s31337002 (plant ON, vertical features OFF) on ${RECIPE_NAME} vs ${ANCHOR}"
	arm "_pon30" "translation-plant-on-features-off" "31337002" "--no-obs-collective-cmd --no-obs-alt-err --no-obs-vz ${RECIPE}" \
		"--obs-collective-cmd --obs-alt-err --obs-vz ${REQ}" "$ANCHOR" \
		",\"features\":\"vertical-off\",\"plant\":\"translation-on\",\"n\":1,\"read\":\"direction only\"${RJ}"
	have 31337002 _pon30 || { log "ABORT — _pon30 s31337002 missing. A run needs a human. Box left idle."; exit 1; }
fi

# ---- STEP 5: _full30 x4 (its pipeline IS the factor) vs _hd30, second read vs _pipeN30 ---------
if [ "$(count _full30)" -ge 4 ]; then log "STEP 5 already complete (4/4 _full30)"; else
	log "STEP 5 — CTRL-16: _full30 x4 (GRID->NEURONS->BITS->CONNECTIONS->MEMORY) vs _hd30"
	arm "_full30" "pipeline-full-nbcm" "$SEEDS4" \
		"--skip-stages , --bits-gens 5 --bits-patience 3 --max-output-neurons 512 --max-cells 1000000000" \
		"--skip-stages --bits-gens --max-output-neurons" "_hd30" \
		",\"pipeline\":\"grid-neurons-bits-connections-memory\",\"max_output_neurons\":512,\"max_cells\":\"off\",\"bits_budget\":\"5/3\""
	[ "$(count _full30)" -ge 4 ] || { log "ABORT — _full30 $(count _full30)/4. A run needs a human. Box left idle."; exit 1; }
	log "---------- CTRL-16 second read: _full30 − _pipeN30 ----------"; pair "_full30" "_pipeN30"
fi

# ---- STEP 6: LEVELS n=5 — ANCHOR s6 (64), then 96 (n384) and 128 (n512) on RECIPE ---------------
if have 31337006 "$ANCHOR"; then log "STEP 6a already banked (${ANCHOR} s31337006)"; else
	log "STEP 6a — CTRL-18: ${ANCHOR} s31337006 (64 levels -> n=5)"
	if [ "$ANCHOR" = "_pipeN30" ]; then
		arm "_pipeN30" "pipeline-neurons" "31337006" "$PIPE_FLAGS" "--skip-stages --max-output-neurons" "_hd30" \
			",\"pipeline\":\"grid-neurons-memory\",\"max_output_neurons\":512,\"max_cells\":\"off\",\"purpose\":\"5th seed of the 64-level rung (CTRL-18)\"" 256 1
	else
		arm "_hd30" "anchor-abi30-airframe-in-trainer" "31337006" "--teacher-hover derived" "--teacher-hover --motor-lag-s" "_hd29" \
			",\"refly_of\":\"_hd29\",\"purpose\":\"5th seed of the 64-level rung (CTRL-18)\"" 256 1
	fi
	have 31337006 "$ANCHOR" || { log "ABORT — ${ANCHOR} s31337006 missing. A run needs a human. Box left idle."; exit 1; }
fi
for N in 384 512; do
	LV=$((N / 4))
	if [ "$(countn $N _L30)" -ge 5 ]; then log "STEP 6 ${LV} levels already complete (5/5)"; continue; fi
	log "STEP 6 — CTRL-18: ${LV} levels (n${N}) x5 (s2-s6) on ${RECIPE_NAME}; off-chip tier allowed"
	# --max-output-neurons must follow the rung, not the 512 cap: last-wins after RECIPE.
	arm "_L30" "levels-${LV}-abi30" "$SEEDS5" "--teacher-hover derived ${RECIPE} --max-output-neurons ${N}" \
		"--teacher-hover --motor-lag-s ${REQ}" "$ANCHOR" \
		",\"levels_study\":true,\"levels\":${LV},\"control\":\"${ANCHOR} n256 same seed\",\"deploy_tier\":\"off-chip allowed\"${RJ}" "$N" 1
	[ "$(countn $N _L30)" -ge 5 ] || { log "ABORT — ${LV} levels $(countn $N _L30)/5. A run needs a human. Box left idle."; exit 1; }
	log "---------- CTRL-18 read: ${LV} levels − 64 levels (${ANCHOR}), same seeds, n=5 ----------"
	PYTHONPATH=src/wnn $VP scripts/paired_power.py --arm "_L30" --base "SL_C_b24n${N}_cf21_brushless_L4C_g10_s{seed}" \
		$(for sd in $SEEDS5; do printf -- "--seed %s --control-override %s=${BASE}_s%s${ANCHOR} " "$sd" "$sd" "$sd"; done) 2>&1 | tee -a "$LOG"
done

log "########## QUEUE COMPLETE — anchor ${ANCHOR} (${RECIPE_NAME}); box IDLE. ##########"
log "NEXT: MA_ANCHOR_SUFFIX=${ANCHOR} bash scripts/multi_axis_chain.sh --round 1 — if PROMOTED, first add the recipe flags (${PIPE_FLAGS}) to every condition's ARM_EXTRA_ARGS (a MA_RECIPE_EXTRA knob), the anchor extension s7-9 included."
