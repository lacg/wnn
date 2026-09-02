#!/usr/bin/env bash
# TRANSLATION A/B — does integrating vertical translation cost attitude
# performance? (01/09/2026, Luiz's call.)
#
# THE CLAIM UNDER TEST. Across 174 banked markers, stable=100.0% appears 32 times
# and EVERY ONE is attitude-only; the altitude regimen's ceiling over 86 markers
# is 98.0% (docs/controller_gate_distance_leaderboard.md). But no run has ever
# changed only that flag — the two regimes also differ by chain, date, cohort and
# teacher — so "translation moved the ceiling" is something the archive is merely
# CONSISTENT WITH. This is the paired test that would earn the claim.
#
# ⚠️ IT IS NOT A ONE-FLAG A/B, AND CANNOT BE. phased_ga.py:3049 refuses
# --obs-collective-cmd / --obs-alt-err / --obs-vz without --translation, because
# without a z axis in the sim those three channels are constant zeros — three
# wasted features and a silently different address space. So the smallest
# coherent difference the code permits is a FOUR-FLAG BUNDLE: the plant's
# vertical integration plus the three features that read it. Everything else is
# byte-identical across the arms.
#
# WHAT THIS MEANS FOR THE WRITEUP. The OFF arm is a SMALLER CONTROLLER (5 obs
# features vs 8), so an OFF win reads as "translation costs attitude performance
# AND/OR the vertical features do not pay for their address space." That
# ambiguity is structural, not sloppiness, and must be stated wherever the result
# is quoted. What the A/B DOES settle is whether the 100%-vs-98% gap in the
# archive survives holding everything else fixed.
#
# ALTITUDE IS ALREADY OUT OF THE OBJECTIVE in both arms: --reward-lambda-alt 0 and
# alt rank weight 0.0, exactly as the whole sweep ladder has run. So this tests
# the PLANT and the OBSERVATION, never the fitness. Do not describe it as
# "optimizing for altitude" — nothing here ever has.
#
# DESIGN. 2 arms x 5 seeds = 10 runs, ~37h, at the ladder's best shape
# (b=32, n=64 = 16 levels/motor, the point that banked hd 0.2240). SEED-MAJOR
# INTERLEAVE: both arms fly each seed before any seed finishes, per the standing
# sweep rule, so stopping the chain anywhere leaves a balanced paired set rather
# than five of one arm and none of the other.
#
# WHY 5 SEEDS. One fixed recipe (GWS_S16noJM) spans 90.8-98.0% stable across five
# base seeds — a ~7pp band, wider than any effect the sweeps have measured. n=1
# per arm could not distinguish this hypothesis from seed luck, and n=3 resolves
# only a 2-1 split, which the gamma gate has already shown is close to a coin
# flip. 5 gives a paired majority that can actually carry a verdict.
#
# GATING (re-gated 01/09/2026, Luiz: option A). Runs only after (1) the gamma=2
# arm banks all four of its markers, (2) BITS ROUND 2 (scripts/bits_round2_chain.sh)
# writes its sentinel BITS_ROUND2_DONE.json, and (3) the box is clear. The A/B
# then flies at the SHAPE THE SENTINEL NAMES — the width sweep's winner at
# (n*, gamma*) — so a better-replicated base recipe is what the 37 h are spent
# on. TAB_BITS / TAB_NEURONS / TAB_GAMMA override that read. Both waits are pure. It never preempts anything: both waits are pure. If the gamma
# chain dies without its markers this chain waits, logging a heartbeat, and the
# box stays idle to be inspected — the same fail-closed posture as the gamma=2
# supervisor.
set -u
ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
. "$ROOT/scripts/controller_arm_lib.sh"

LOG="/private/tmp/translation_ab.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
OUTDIR="logs/controller/translation_ab"
MARKDIR="experiments/translationab_markers"
LADDERMARK="experiments/sweepladder_markers"
AIRFRAME="cf21_brushless"; DIST="L4C"
BITS="${TAB_BITS:-32}"; NEURONS="${TAB_NEURONS:-64}"; GAMMA="${TAB_GAMMA:-1.0}"
R2_SENTINEL="${LADDERMARK}/BITS_ROUND2_DONE.json"   # re-read below once round 2 lands
SEEDS="${TAB_SEEDS:-31337002 31337003 31337004 31337005 31337006}"
REPORT_SEEDS="99990101 99990102 99990103 99990104 99990105"
GATE_SEED="${TAB_GATE_SEED:-31337002}"   # the gamma=2 arm's seed, for the wait

mkdir -p "$MARKDIR" "$OUTDIR"

# The attitude half of the observation set — IDENTICAL in both arms.
FEAT_BASE="--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i"
# The vertical bundle — the ONLY thing that differs. It moves as a unit because
# phased_ga refuses the features without the plant.
FEAT_VERT="--obs-collective-cmd --obs-alt-err --obs-vz"
WEIGHTS="--fit-weight-err-sq 0.3125 --fit-weight-stable 0.25 --fit-weight-steady 0.4375"
AGG_GATE="--fit-aggregation zscore --zrank-clamp 3.0 --gate-stable 0.70 --gate-err 8.0"

log() { echo "[translation-ab] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controller_pids() { pgrep -f -- "-m wnn.control.phased_ga" 2>/dev/null || true; }

gamma2_missing() {
	local miss="" b n t
	for n in 64 96; do
		for b in 36 32; do
			t="SL_C_b${b}n${n}_${AIRFRAME}_${DIST}_g20_s${GATE_SEED}.json"
			[ -f "${LADDERMARK}/${t}" ] || miss="${miss} ${t}"
		done
	done
	echo "$miss"
}

# run_point <arm: on|off> <seed>
run_point() {
	local arm="$1" seed="$2"
	local tag="TAB_${arm}_b${BITS}n${NEURONS}_${AIRFRAME}_${DIST}_s${seed}"
	[ -f "${MARKDIR}/${tag}.json" ] && { log "SKIP $tag (marker exists)"; return 0; }

	local vert plant
	if [ "$arm" = "on" ]; then
		vert="$FEAT_VERT"; plant="--translation --reward-lambda-alt 0"
	else
		vert=""; plant="--no-translation"
	fi

	while [ -n "$(controller_pids)" ]; do sleep 20; done
	mkdir -p "$OUTDIR/ckpt/$tag"
	log "===== START $tag (translation=${arm}, $([ "$arm" = on ] && echo 8 || echo 5) obs features) ====="
	# shellcheck disable=SC2086
	run_controller_arm "$tag" "$MARKDIR" "$OUTDIR" "$VP" log \
		"\"study\":\"translation-ab\",\"arm\":\"${arm}\",\"translation\":$([ "$arm" = on ] && echo true || echo false),\"bits\":${BITS},\"neurons\":${NEURONS},\"levels_per_motor\":$((NEURONS / 4)),\"delta_gamma\":${GAMMA},\"seed\":${seed}" \
		-- \
		--levels 16 --lamarckian \
		--skip-stages neurons,bits \
		--max-cells 180000 --max-cells-strict \
		--save-stage-checkpoints "$OUTDIR/ckpt/$tag" \
		--neurons-gens 5 --neurons-patience 3 \
		--conns-gens 5 --conns-patience 3 \
		--memory-gens 120 --memory-patience 2 \
		--pop 50 --num-eval-folds 5 --check-interval 2 --magnitude-aware-patience \
		--eval-episodes 100 --memory-eval-episodes 200 \
		--steps 2000 --tilt 5.0 \
		$WEIGHTS $AGG_GATE \
		--delta-gamma "$GAMMA" \
		--grid-bits "$BITS" --grid-output-neurons "$NEURONS" --max-output-neurons "$NEURONS" \
		--report-episodes 100 --holdout-pop-sample 8 \
		--runs 1 --memory-mode BINARY \
		--airframe "$AIRFRAME" --disturbance "$DIST" --teacher mpcof \
		$FEAT_BASE $vert \
		$plant \
		--grid-state-neurons 0 --max-state-neurons 0 \
		--report-seeds $REPORT_SEEDS \
		--base-seed "$seed"
	log "$tag finished rc=$?"
}

log "########## ARMED — translation A/B, gated on the gamma=2 arm ##########"
log "shape: the bits round-2 winner from ${R2_SENTINEL} unless TAB_BITS/TAB_NEURONS/TAB_GAMMA are set (defaults b=${BITS} n=${NEURONS} gamma=${GAMMA}) · seeds=[$SEEDS] · seed-major interleave"

# ---- GATE: wait for the gamma=2 arm, with a heartbeat so a stall is visible.
beat=0
while :; do
	miss="$(gamma2_missing)"
	[ -z "$miss" ] && break
	[ $((beat % 30)) = 0 ] && log "waiting on the gamma=2 arm — still missing:${miss}"
	beat=$((beat + 1)); sleep 60
done
log "gamma=2 arm complete — all four markers present."

# ---- GATE 2 (01/09/2026): bits round 2 must have banked its sentinel. Its
# chain fails closed — no sentinel means a run needs a human — so this wait
# parks the A/B deliberately rather than flying it at a shape round 2 was
# about to revise.
beat=0
while [ ! -f "$R2_SENTINEL" ]; do
	[ $((beat % 30)) = 0 ] && log "waiting on bits round 2 — no ${R2_SENTINEL} yet"
	beat=$((beat + 1)); sleep 60
done
r2() { "$VP" -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$R2_SENTINEL" "$1"; }
BITS="${TAB_BITS:-$(r2 bits)}"; NEURONS="${TAB_NEURONS:-$(r2 neurons)}"; GAMMA="${TAB_GAMMA:-$(r2 gamma)}"
log "bits round 2 complete — the A/B flies at its winner: b=${BITS} n=${NEURONS} ($((NEURONS / 4)) lvl/motor) gamma=${GAMMA}"
while [ -n "$(controller_pids)" ]; do sleep 30; done

# ---- PREFLIGHT: prove the OFF flag set survives phased_ga's guards.
#
# The OFF arm is the ON recipe minus three obs flags with --translation negated,
# and by reading the guards at phased_ga.py:3049 it is safe: the vertical obs
# flags default False and --reward-lambda-alt defaults 0.0, so nothing it omits
# can trip them. Reading is not proof — --fit-weight-alt was a silent no-op at
# 4 of 5 call sites and every one of those looked right on the page too. The
# guards raise BEFORE any simulation, so a micro-budget invocation settles it in
# seconds. Run here, where the box is idle by construction (the wait above just
# cleared it), so this never becomes a second controller.
preflight_off() {
	local o="$OUTDIR/PREFLIGHT_off.out"
	rm -f "$o"
	# shellcheck disable=SC2086
	"$VP" -u -m wnn.control.phased_ga \
		--levels 16 --skip-stages neurons,bits \
		--neurons-gens 0 --conns-gens 0 --memory-gens 0 \
		--pop 2 --num-eval-folds 5 --eval-episodes 1 --memory-eval-episodes 1 \
		--steps 1 --tilt 5.0 --report-episodes 1 --runs 1 --memory-mode BINARY \
		$WEIGHTS $AGG_GATE --delta-gamma "$GAMMA" \
		--grid-bits 8 --grid-output-neurons 8 --max-output-neurons 8 \
		--airframe "$AIRFRAME" --disturbance "$DIST" --teacher mpcof \
		$FEAT_BASE --no-translation \
		--grid-state-neurons 0 --max-state-neurons 0 \
		--report-seeds $REPORT_SEEDS --base-seed 31337002 > "$o" 2>&1 &
	local pid=$! i=0
	while [ $i -lt 90 ]; do
		grep -qa "Phased-GA controller search:" "$o" 2>/dev/null && { kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; return 0; }
		kill -0 "$pid" 2>/dev/null || break
		sleep 1; i=$((i + 1))
	done
	kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
	grep -qa "Phased-GA controller search:" "$o" 2>/dev/null && return 0
	return 1
}

if preflight_off; then
	log "PREFLIGHT ok — the OFF flag set clears phased_ga's guards."
else
	log "ABORT — the OFF flag set FAILED preflight. Last lines of $OUTDIR/PREFLIGHT_off.out:"
	tail -5 "$OUTDIR/PREFLIGHT_off.out" 2>/dev/null | while read -r l; do log "    $l"; done
	log "Launching nothing: half an A/B is not an A/B. Box left idle."
	exit 1
fi

# ---- THE A/B, SEED-MAJOR: both arms fly a seed before the next seed starts.
for seed in $SEEDS; do
	for arm in on off; do
		run_point "$arm" "$seed"
	done
done

# ---- PAIRED VERDICT on the same gate-distance scale the ladder ranks on.
log "---------- PAIRED VERDICT ----------"
"$VP" - "$MARKDIR" "$BITS" "$NEURONS" "$AIRFRAME" "$DIST" "$SEEDS" >> "$LOG" <<'PY'
import json, math, os, re, sys
markdir, bits, neurons, airframe, dist, seeds = sys.argv[1:7]
K = math.log(0.5) / math.log(0.70)


def read(arm, seed):
	p = f"{markdir}/TAB_{arm}_b{bits}n{neurons}_{airframe}_{dist}_s{seed}.json"
	if not os.path.exists(p):
		return None
	h = json.load(open(p)).get("headline_holdout", "")
	ms = re.search(r"stable=([0-9.]+)%", h)
	me = re.search(r"err=([0-9.]+)", h)
	md = re.search(r"steady=([0-9.]+)", h)
	if not (ms and me):
		return None
	s, e = float(ms.group(1)), float(me.group(1))
	hd = 0.5556 * (e / 8.0) + 0.4444 * min(K * -math.log2(max(s, 1e-4) / 100.0), 20.0)
	return dict(stable=s, err=e, steady=float(md.group(1)) if md else float("nan"), hd=hd)


on_w = off_w = 0
rows = []
for seed in seeds.split():
	a, b = read("on", seed), read("off", seed)
	if not (a and b):
		rows.append(f"  s{seed}  INCOMPLETE (on={'y' if a else 'n'} off={'y' if b else 'n'})")
		continue
	d = a["hd"] - b["hd"]
	if d < 0:
		on_w += 1
	else:
		off_w += 1
	rows.append("  s%s  ON %5.1f%%/%5.2f/%5.2f hd %.4f   OFF %5.1f%%/%5.2f/%5.2f hd %.4f   delta %+.4f -> %s"
	            % (seed, a["stable"], a["err"], a["steady"], a["hd"],
	               b["stable"], b["err"], b["steady"], b["hd"], d,
	               "ON" if d < 0 else "OFF"))
print("\n".join(rows))
print("  PAIRED MAJORITY: translation ON %d - %d OFF over %d complete pairs."
      % (on_w, off_w, on_w + off_w))
print("  Read an OFF win as 'the plant AND/OR the 3 vertical features cost attitude'")
print("  — the arms differ by 8 vs 5 observation features and the code forbids splitting them.")
PY
log "########## TRANSLATION A/B COMPLETE ##########"
