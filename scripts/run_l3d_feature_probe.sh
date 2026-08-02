#!/usr/bin/env bash
# L3D-training x feature-set probe — a 2x2 the dfa1l study never ran.
#
# WHY. Two levers are unexplored while the ones the study spent 178h on are
# measured-dead (memory-GA +0.0, suffix not significant, more DAgger -15.6pp/doubling):
#
#   1. TRAINING DISTURBANCE. Every dfa1l cell trains at L2D. Scored at L3D they read
#      0-2.8% stable — and so does PID (3.8%), while LQR/MPC/LQI hold 48-71%. That is
#      the signature of out-of-distribution failure, not incapacity: the model-based
#      controllers recompute from the plant, a WNN has no cells written at L3D
#      addresses. Nobody has ever TRAINED one at L3D.
#   2. FEATURE SET. FrameFixVal_20260627 (post-frame-fix, 18/18) ranks
#      pidmix (+obs-peraxis-p/i +obs-yaw-err-i) at +5.5pp over s16, both seeds
#      agreeing — while the study only ever swept 9feat vs 10feat, and 10feat
#      (=anchor) is the WEAKEST of the winning post-fix configs at +1.0.
#
# Substrate fixed 1layer, mode fixed BINARY: the 18-cell rescore settled both
# (1layer ties dfa at 10feat and is 35x cheaper — 0.5h vs 17.3h).
#
# TWO DEPARTURES from run_dfa_1layer_study.sh, both deliberate:
#   --holdout-fixed-thresholds  the study's held-out refits decode thresholds on the
#                               REPORT seed, shifting the address function so trained
#                               cells are read where nothing was written. Rank
#                               correlation of that axis to the aligned one: 0.19.
#                               A screen without this flag ranks threshold luck.
#   --report-seeds (5, not 1)   lands the held-out directly on the baselines' axis,
#                               so no separate rescore pass is needed after.
#
# Usage: run_l3d_feature_probe.sh [seed ...]     (default: 31337002 = round 1)
# Resumable: a cell with a marker is skipped, so re-running continues the sweep.
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
OUTDIR="${OUTDIR:-logs/controller/l3dfeat}"
MARKDIR="${MARKDIR:-experiments/l3dfeat_markers}"
mkdir -p "$OUTDIR" "$MARKDIR"

SEEDS=(${@:-31337002})
REPORT_SEEDS="99990101 99990102 99990103 99990104 99990105"

# Identical to the study's COMMON except --disturbance (per-arm) and the two
# departures above. Keeping the rest bit-identical is what makes A1 a control:
# it must reproduce 1layer_10feat_BINARY_s31337002's 99.6±0.5, or the harness is wrong.
COMMON="--levels 16 --skip-stages bits,connections --lamarckian \
--max-cells 180000 --neurons-gens 60 --neurons-patience 3 \
--memory-gens 120 --memory-patience 2 --pop 50 --num-eval-folds 5 \
--check-interval 2 --magnitude-aware-patience --eval-episodes 100 \
--memory-eval-episodes 200 --steps 2000 --tilt 5.0 \
--fit-weight-err-sq 0.4 --fit-weight-stable 0.3 --fit-weight-jerk 0.2 \
--fit-weight-mono 0.1 --report-episodes 100 \
--holdout-pop-sample 8 --grid-bits 24 30 --max-output-neurons 128 \
--runs 1 --teacher lqr --grid-state-neurons 0 --max-state-neurons 0 \
--memory-mode BINARY --holdout-fixed-thresholds"

FEAT_10FEAT="--obs-yaw-err"
FEAT_PIDMIX="--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i"
# FrameFixVal ranked these 2nd and 7th on err — but it ran DFA-style with state
# neurons, where the forced recurrent-state prefix ate most of each neuron's bits
# ("~40b state-prefix leaves ~24b/10feat = 2.4 b/feat, starved"). Here sn=0, so
# there is NO state prefix and every bit goes to sensors. The budget pressure that
# penalised the wider feature sets there is absent, so their old ranking does not
# transfer — nf=19 and nf=21 deserve a fresh measurement, not an inherited verdict.
FEAT_PIDMIX_PWM="$FEAT_PIDMIX --obs-pwm"
FEAT_PIDMIX_PWM_TILT="$FEAT_PIDMIX --obs-pwm --obs-tilt-p --obs-tilt-i"
# NOT a feature flag — an ACTION-SPACE change: 4 controls [T, tau_roll, tau_pitch,
# tau_yaw] mixed to motors instead of 4 raw PWMs. So the bit-starvation argument does
# NOT motivate it; its nf is unchanged. The 1layer argument does: with sn=0 the output
# layer IS the whole network and must learn the motor mixing itself, with no state
# layer to carry that structure. Weakest-motivated of the additions — FrameFixVal had
# it at 3.550 err, dead level with plain anchor. Paired with pidmix (our winner)
# rather than with 10feat, which is how FrameFixVal tested it.
FEAT_PIDMIX_DECOUPLE="$FEAT_PIDMIX --decouple-outputs"
# nf=17 — the untested gap between the nf=15 winner and the nf=19 failure. tilt has
# only ever been tested BUNDLED with pwm (A6, nf=21) or alone without pidmix
# (FrameFixVal's `tilt`), so "does tilt help pidmix" has never actually been asked.
FEAT_PIDMIX_TILT="$FEAT_PIDMIX --obs-tilt-p --obs-tilt-i"

log() { echo "[l3dfeat] $* $(date -u +%FT%TZ)"; }

# One arm. Args: armname disturbance featureset seed
run_arm() {
	local arm="$1" dist="$2" featset="$3" seed="$4"
	local tag="${arm}_${dist}_s${seed}"
	local marker="${MARKDIR}/${tag}.json"
	local out="${OUTDIR}/${tag}.out"
	local winner="${OUTDIR}/${tag}_winner.yaml.gz"
	if [ -f "$marker" ]; then log "$tag: marker exists — skip"; return; fi

	local feat_flags
	case "$featset" in
		pidmix)          feat_flags="$FEAT_PIDMIX" ;;
		pidmix_pwm)      feat_flags="$FEAT_PIDMIX_PWM" ;;
		pidmix_pwm_tilt) feat_flags="$FEAT_PIDMIX_PWM_TILT" ;;
		pidmix_decouple) feat_flags="$FEAT_PIDMIX_DECOUPLE" ;;
		pidmix_tilt)     feat_flags="$FEAT_PIDMIX_TILT" ;;
		10feat)          feat_flags="$FEAT_10FEAT" ;;
		*) log "$tag: UNKNOWN featset '$featset' — refusing to guess"; return ;;
	esac

	log "===== START $tag (dist=$dist feats=$featset) ====="
	local t0=$SECONDS
	/usr/bin/time -l "$VP" -u -m wnn.control.phased_ga $COMMON \
		--disturbance "$dist" $feat_flags \
		--report-seeds $REPORT_SEEDS \
		--base-seed "$seed" --save-winner "$winner" > "$out" 2>&1
	local rc=$? dur=$((SECONDS - t0))

	# Watchdog stop (SIGTERM/SIGKILL) → no marker, leave for re-run. Unlike the study
	# this does NOT auto-retry: the probe is short and a human should see a kill.
	if [ "$rc" = "143" ] || [ "$rc" = "137" ]; then
		log "$tag: rc=$rc (watchdog stop) — NO marker, leaving for re-run"
		return
	fi
	if [ "$rc" != "0" ]; then
		log "$tag: rc=$rc (crash) — NO marker, leaving for re-run"
		return
	fi

	local rss held_n held_m cells fpga
	rss=$(grep -E "maximum resident set size" "$out" | awk '{print $1}' | tail -1)
	held_n=$(grep -E "RESULT — during-search winner" "$out" | sed -n '1p')
	held_m=$(grep -E "RESULT — during-search winner" "$out" | sed -n '2p')
	[ -f "$winner" ] && "$VP" -u scripts/gran_fpga_count.py "$winner" >> "$out" 2>&1
	fpga=$(grep -E "^\[FPGA\]" "$out" | tail -1)
	cells=$(grep -oE "cells\[[0-9-]+ Σ[0-9]+k μ[0-9]+k\]" "$out" | tail -1)

	# R4: rc=0 with an empty MEMORY triple is a truncated run — no marker, re-run.
	if [ -z "${held_m// /}" ]; then
		log "$tag: rc=0 but no MEMORY-stage held-out (truncated) — NO marker, leaving for re-run"
		return
	fi

	printf '{"tag":"%s","arm":"%s","disturbance":"%s","features":"%s","substrate":"1layer","mode":"BINARY","seed":%s,"rc":%s,"dur_s":%s,"peak_rss_bytes":%s,"cells":"%s","fpga":"%s","held_neurons":"%s","held_memory":"%s","fixed_thresholds":true,"done":"%s"}\n' \
		"$tag" "$arm" "$dist" "$featset" "$seed" "$rc" "$dur" "${rss:-null}" \
		"$cells" \
		"$(echo "$fpga" | tr -d '"' | sed 's/  */ /g')" \
		"$(echo "$held_n" | tr -d '"' | sed 's/  */ /g')" \
		"$(echo "$held_m" | tr -d '"' | sed 's/  */ /g')" \
		"$(date -u +%FT%TZ)" > "$marker"
	log "===== END $tag rc=$rc dur=${dur}s ====="
}

# INTERLEAVED by seed: round r fires all four arms at seed r, so after one round
# every arm has a datapoint and the losing arms can be culled before spending more.
for seed in "${SEEDS[@]}"; do
	log "########## ROUND seed=$seed ##########"
	# ORDER IS A DROP-PRIORITY. The original 2x2 (A1-A4) runs first because it carries
	# the two questions the probe exists to answer; the width extension (A5-A7) is
	# appended, so a night that runs out of time loses extension arms at the LAST seed
	# rather than truncating the core design at every seed.
	run_arm A1 L2D 10feat          "$seed"   # control — must reproduce ~99.6
	run_arm A2 L2D pidmix          "$seed"   # feature lever  (WON round 1: 1.63 deg)
	run_arm A3 L3D 10feat          "$seed"   # THE question: is the L3D collapse OOD?
	run_arm A4 L3D pidmix          "$seed"   # both levers
	run_arm A5 L2D pidmix_pwm      "$seed"   # nf=19 — starved in DFA, not here
	run_arm A6 L2D pidmix_pwm_tilt "$seed"   # nf=21 — most starved there, most to gain
	run_arm A7 L2D pidmix_decouple "$seed"   # action space, not features (weakest prior)
	run_arm A8 L2D pidmix_tilt     "$seed"   # nf=17 — P1: where exactly is the boundary?
done

> "${MARKDIR}/ROUND_DONE.marker"
log "ALL ARMS DONE for seeds: ${SEEDS[*]}"
