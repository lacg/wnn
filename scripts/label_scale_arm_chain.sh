#!/usr/bin/env bash
# LABEL-SCALE ARM A — does shrinking the DAgger label's dead zone lower the hold floor?
# (08/09/2026, Luiz: "arm A with s=2, 4 and 8 then".)
#
# THE QUESTION. The live trainer (bptt_train_window) labels each output bank with
# the teacher's ABSOLUTE pwm, floored to the 1/levels antagonist grid. At the
# ladder's L=64 (256 output neurons / 4 motors) one grid step is 1/64 = 0.0156
# pwm, so any teacher deviation from neutral smaller than that is written as
# "neutral" — a DEAD ZONE. Measured on mpcof over the recipe's plant (20 eps):
#   motor-labels written "neutral" in the hold window, m0/m1/m2/m3:
#     L=64 today   ±0.0156    1 / 67 / 10 / 71 %   (mean ~37 %)
#     s=2          ±0.0078    0 / 37 /  3 / 40 %   (mean ~20 %)
#     s=4          ±0.0039    0 / 18 /  1 / 21 %   (mean ~10 %)
#     s=8          ±0.0020    0 /  9 /  1 / 10 %   (mean  ~5 %)
# Motors 0/2 carry the L4C trim and are visible; motors 1/3 sit at neutral with a
# hold-window std of ~0.014 pwm, INSIDE the band, so their fine corrections —
# the content of the hold window — are erased two thirds of the time.
#
# THE LEVER. --delta-label-scale s widens the label's deviation from the decode
# neutral BEFORE the grid floors it (decoded = neutral + s·(p − neutral)). Paired
# with --delta-max delta_max/s the loop gain G = 2·delta_max/(1−leak) is
# UNCHANGED (4.0 at the recipe's leak 0.95), so the ONLY thing that moves is the
# dead zone (s× narrower) and the emitted increment (s× finer) — the same effect
# as s× more output neurons, at the same cells, alphabet and H743 footprint.
# Pinned in Rust: label_semantics_tests (196/196, branch label-scale-and-legacy-fix).
#
# DESIGN — paired, same-seed, same-recipe, same-era, ONE lever at three rungs.
#   control = ALREADY BANKED, no re-fly: the four b24 n256 CRN ladder runs
#             (seeds 31337002_crn, 31337003, 31337004, 31337005), s=1.
#   rungs   = s ∈ {2, 4, 8} with delta_max 0.05 / 0.025 / 0.0125, tagged
#             _ls2/_ls4/_ls8 so marker, .out and ckpt never collide.
#   order   = SEED-MAJOR rounds (the sweep rule: interleave EVERY dimension —
#             round 1 = one run of each rung at seed 2), so the 2→4→8 curve
#             exists at n=1 after three runs and a dead arm is culled early.
#   read    = paired per seed on the MEMORY same-rule row, ALL FOUR columns
#             (stable / err / steady / alt) — steady and alt are the primaries,
#             err is ~80 % transient and should not move. A MONOTONE response
#             across 2→4→8 is the strongest evidence for the mechanism.
#             4 seeds → paired majority per rung. n=4 is a DIRECTION.
#
# PRECONDITIONS (fail closed, all checked before the first run):
#   · the box is IDLE — no controller, no ladder, no other chain. This chain
#     must never be launched while another chain is armed.
#   · the installed ram_controller wheel exposes delta_label_scale on the
#     batched trainer, AND the Python tree carries --delta-label-scale, AND the
#     fix commit is an ancestor of HEAD — Python and wheel staged TOGETHER.
#   · the four control markers exist.
#   · STEP 0 smoke: (a) the label pins re-run against the INSTALLED wheel
#     (s=1 byte-identical to the banked behaviour; s=4 resolves 1/64);
#     (b) one tiny end-to-end phased_ga run with the new flags returns rc=0.
#
# GATING. Pure wait for any controller; never preempts. Every point is
# marker-gated and idempotent (the ladder SKIPs an existing marker). FAILS
# CLOSED: a missing marker after a run means a human is needed — the chain stops
# and leaves the box idle rather than stacking work on a crash.
set -u
ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1

LOG="/private/tmp/label_scale_arm.log"
LADDER="scripts/sweep_ladder_gamma.sh"
MARK="experiments/sweepladder_markers"
LSMARK="experiments/labelscale_markers"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
AIRFRAME="cf21_brushless"; DIST="L4C"
BITS=24
NEURONS=256
DMAX0="0.1"                       # the recipe's delta_max at s=1
SEEDS="${LS_SEEDS:-31337002 31337003 31337004 31337005}"
SCALES="${LS_SCALES:-2 4 8}"
FIX_COMMIT="262574a7"             # label-scale-and-legacy-fix; must be merged into HEAD

log() { echo "[label-scale] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controller_pids() { pgrep -f -- "-m wnn.control.phased_ga" 2>/dev/null || true; }
ladder_pids() { pgrep -f "scripts/sweep_ladder_gamma.sh" 2>/dev/null || true; }
other_chain_pids() { pgrep -f "scripts/(mutstep_ab|queue_after_ab|leak_revisit|crn_|translation_ab|window_k)[a-z_]*chain" 2>/dev/null || true; }
wait_box_clear() {
	local beat=0
	while [ -n "$(controller_pids)" ] || [ -n "$(ladder_pids)" ]; do
		[ $((beat % 30)) = 0 ] && log "waiting — box busy (controller/ladder still running)"
		beat=$((beat + 1)); sleep 60
	done
}

# The control must already exist, or the rungs would be an unpaired cohort.
ctrl_tag() { # $1 = seed
	if [ "$1" = "31337002" ]; then echo "SL_C_b${BITS}n${NEURONS}_${AIRFRAME}_${DIST}_g10_s31337002_crn"
	else echo "SL_C_b${BITS}n${NEURONS}_${AIRFRAME}_${DIST}_g10_s$1"; fi
}
dmax_for() { $VP -c "print(${DMAX0}/$1)"; }

mkdir -p "$LSMARK" logs/controller/label_scale
log "########## ARMED — LABEL-SCALE ARM A at b${BITS} n${NEURONS}: s=[${SCALES}] vs banked s=1 ##########"
log "seeds=[${SEEDS}]  G = 2·delta_max/(1−leak) held at 4.0 via delta_max = ${DMAX0}/s"

# ---- PRECONDITIONS -------------------------------------------------------------
if [ -n "$(controller_pids)" ] || [ -n "$(ladder_pids)" ] || [ -n "$(other_chain_pids)" ]; then
	log "ABORT — the box is NOT idle (controller/ladder/another chain is running). Never launch this chain beside another."
	exit 1
fi
if ! git merge-base --is-ancestor "$FIX_COMMIT" HEAD 2>/dev/null; then
	log "ABORT — fix commit ${FIX_COMMIT} (label-scale-and-legacy-fix) is not merged into HEAD. Merge first (Python + wheel land together)."
	exit 1
fi
if ! $VP -c "import ram_controller as c; sig=c.dagger_train_batch_inplace.__text_signature__ or ''; assert 'delta_label_scale' in sig, sig" 2>>"$LOG"; then
	log "ABORT — the INSTALLED ram_controller wheel predates delta_label_scale. Install the staged wheel (pip install wnn-labelfix/wheels/ram_controller-*.whl)."
	exit 1
fi
if ! PYTHONPATH=src/wnn $VP -m wnn.control.phased_ga --help 2>/dev/null | grep -q -- "--delta-label-scale"; then
	log "ABORT — the Python tree has no --delta-label-scale (source/wheel skew)."
	exit 1
fi
MISSING=""
for s in $SEEDS; do
	[ -f "${MARK}/$(ctrl_tag "$s").json" ] || MISSING="${MISSING} $(ctrl_tag "$s")"
done
[ -z "$MISSING" ] || { log "ABORT — control markers missing:${MISSING}. Nothing to pair against."; exit 1; }
log "preflight OK — box idle, fix merged, wheel + Python carry delta_label_scale, 4 controls present."

# ---- STEP 0: SMOKE (marker-gated, fails closed) --------------------------------
SMOKE="${LSMARK}/SMOKE_OK.json"
if [ -f "$SMOKE" ]; then
	log "SKIP smoke (marker exists)."
else
	log "===== STEP 0a: label pins against the INSTALLED wheel ====="
	if ! PYTHONPATH=src/wnn $VP - >> "$LOG" 2>&1 <<'PY'
import numpy as np, ram_controller as rc
levels, bpf, obpn, nf = 16, 3, 8, 9; fb = nf * bpf
def fixture(dmax, scale):
	rng = np.random.default_rng(7)
	thr = [float(x) for x in rng.uniform(-5, 5, fb)]; oc = [int(x) for x in rng.integers(0, fb, 4 * levels * obpn)]
	return rc.WnnController(4, levels, bpf, 1, 0, 0, obpn, thr, [], oc, delta_control=True, delta_max=dmax,
	                        delta_leak=0.95, memory_mode=3, delta_label_scale=scale)
def fly(dmax, scale, p, T=20):
	c = fixture(dmax, scale)
	g = [[0.0] * 3] * T; a = [[0.0, 0.0, 9.81]] * T; tg = [[0.0] * 3] * T
	c.bptt_train_window(g, a, tg, [[p] * 4] * T, 4, True); c.reset(0.0)
	o = [c.step([0.0] * 3, [0.0, 0.0, 9.81], [0.0] * 3)[0] for _ in range(300)]
	return o[0] - 0.5, o[-1]
d, s = fly(0.1, 1.0, 0.5 + 1 / 64); assert d == 0.0, ("s=1 must not see 1/64", d)
d, s = fly(0.1, 1.0, 0.5625); assert abs(d - 0.0125) < 1e-5 and abs(s - 0.75) < 1e-3, ("s=1 legacy", d, s)
d, s = fly(0.025, 4.0, 0.5 + 1 / 64); assert abs(d - 0.003125) < 1e-5 and abs(s - 0.5625) < 1e-3, ("s=4 dead zone", d, s)
d4, s4 = fly(0.025, 4.0, 0.5625); d1, s1 = fly(0.1, 1.0, 0.5625)
assert abs(d4 - d1) < 1e-5 and abs(s4 - s1) < 1e-3, ("G invariance", d4, d1, s4, s1)
print("[smoke] label pins OK on the installed wheel: s=1 legacy byte-identical; s=4 resolves 1/64; G invariant")
PY
	then
		log "ABORT — label pins FAILED against the installed wheel. Nothing runs."; exit 1
	fi
	log "===== STEP 0b: tiny end-to-end phased_ga run with the new flags (rc must be 0) ====="
	PYTHONPATH=src/wnn $VP -u -m wnn.control.phased_ga \
		--levels 16 --lamarckian --skip-stages neurons,bits \
		--max-cells 180000 --max-cells-strict \
		--conns-gens 1 --conns-patience 1 --memory-gens 1 --memory-patience 1 \
		--pop 4 --num-eval-folds 5 --check-interval 1 \
		--eval-episodes 2 --memory-eval-episodes 2 --steps 200 --tilt 5.0 \
		--fit-weight-err-sq 0.3125 --fit-weight-stable 0.25 --fit-weight-steady 0.4375 \
		--fit-aggregation zscore --zrank-clamp 3.0 --gate-stable 0.70 --gate-err 8.0 \
		--delta-gamma 1.0 --grid-bits "$BITS" --grid-output-neurons "$NEURONS" --max-output-neurons "$NEURONS" \
		--report-episodes 2 --holdout-pop-sample 2 --runs 1 --memory-mode BINARY \
		--airframe "$AIRFRAME" --disturbance "$DIST" --teacher mpcof \
		--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i \
		--obs-collective-cmd --obs-alt-err --obs-vz \
		--translation --reward-lambda-alt 0 --grid-state-neurons 0 --max-state-neurons 0 \
		--report-seeds 99990101 --base-seed 31337002 \
		--delta-label-scale 2 --delta-max "$(dmax_for 2)" \
		> logs/controller/label_scale/smoke.out 2>&1
	rc=$?
	log "smoke phased_ga rc=${rc}"
	if [ "$rc" != "0" ]; then
		log "ABORT — smoke run FAILED; nothing else runs. Last lines:"
		tail -8 logs/controller/label_scale/smoke.out | while read -r l; do log "    $l"; done
		exit 1
	fi
	echo "{\"smoke\":\"label_scale\",\"rc\":0,\"done\":\"$(date -u +%FT%TZ)\"}" > "$SMOKE"
	log "smoke OK — banked ${SMOKE}"
fi

# ---- THE RUNS: seed-major rounds, s = 2, 4, 8 at each seed ---------------------
for seed in $SEEDS; do
	for S in $SCALES; do
		DMAX="$(dmax_for "$S")"
		TAG="SL_C_b${BITS}n${NEURONS}_${AIRFRAME}_${DIST}_g10_s${seed}_ls${S}"
		if [ -f "${MARK}/${TAG}.json" ]; then
			log "SKIP — ${TAG} already banked."; continue
		fi
		wait_box_clear
		log "===== START ${TAG} (s=${S}, --delta-label-scale ${S} --delta-max ${DMAX}; control $(ctrl_tag "$seed")) ====="
		SL_SKIP_PHASE1=1 SL_SWEEP_LABEL="label-scale" SL_FORCE_PHASE2_GAMMA="1.0" \
			SL_WIDTHS="$BITS" SL_NEURONS="$NEURONS" SL_SEED="$seed" \
			SL_TAG_SUFFIX="_ls${S}" \
			SL_EXTRA_ARGS="--delta-label-scale ${S} --delta-max ${DMAX}" \
			SL_EXTRA_MARKER_JSON=",\"arm_ls\":${S},\"delta_label_scale\":${S},\"delta_max\":${DMAX},\"control_tag\":\"$(ctrl_tag "$seed")\"" \
			bash "$LADDER"
		log "ladder exited rc=$? for ${TAG}"
		while [ -n "$(controller_pids)" ]; do sleep 30; done
		[ -f "${MARK}/${TAG}.json" ] || { log "ABORT — marker ${TAG}.json MISSING. A run needs a human. Box left idle."; exit 1; }
		log "banked: ${TAG}"
	done
done

# ---- VERDICT: per seed, control vs s=2/4/8 on the MEMORY same-rule row, four columns.
$VP scripts/gate_distance_leaderboard.py > docs/controller_gate_distance_leaderboard.md 2>/dev/null
log "---------- VERDICT: label scale s=2/4/8 vs banked s=1, MEMORY same-rule row (lower = better), ALL FOUR columns ----------"
LS_B="$BITS" LS_N="$NEURONS" LS_SEEDS_LIST="$SEEDS" LS_SCALES_LIST="$SCALES" $VP - >> "$LOG" <<'PY'
import json, math, os, re
K = math.log(0.5) / math.log(0.70)
def gd(s, e): return 0.5556 * (e / 8.0) + 0.4444 * min(K * -math.log2(s / 100), 20.0)
B, N = os.environ["LS_B"], os.environ["LS_N"]
def row(tag):
	p = f"experiments/sweepladder_markers/{tag}.json"
	if not os.path.exists(p): return None
	m = json.load(open(p)).get("held_memory_multiseed", "")
	f = lambda pat: float(re.search(pat, m).group(1))
	try: return (f(r"stable=([0-9.]+)"), f(r"err=([0-9.]+)"), f(r"steady=([0-9.]+)"),
	             f(r"alt=([0-9.]+)"), gd(f(r"stable=([0-9.]+)"), f(r"err=([0-9.]+)")))
	except Exception: return None
fmt = lambda r: f"{r[0]:5.1f}% {r[1]:.2f}° {r[2]:.2f}° {r[3]:.3f}m hd {r[4]:.4f}"
seeds = os.environ["LS_SEEDS_LIST"].split(); scales = os.environ["LS_SCALES_LIST"].split()
tally = {s: [0, 0, 0] for s in scales}   # wins, losses, ties vs control (same-rule)
for sd in seeds:
	ct = f"SL_C_b{B}n{N}_cf21_brushless_L4C_g10_s{sd}" + ("_crn" if sd == "31337002" else "")
	c = row(ct)
	print(f"\n  seed {sd}")
	print(f"    {'s=1 control':12s} {fmt(c) if c else 'control missing'}")
	steady_curve, alt_curve = [], []
	for s in scales:
		a = row(f"SL_C_b{B}n{N}_cf21_brushless_L4C_g10_s{sd}_ls{s}")
		if not a or not c:
			print(f"    {'s='+s:12s} not flown yet"); continue
		# A TIE is neither side's win — the tally refuses to score it, and so does the label.
		w = "arm" if a[4] < c[4] else ("control" if c[4] < a[4] else "TIE")
		tally[s][0] += a[4] < c[4]; tally[s][1] += c[4] < a[4]; tally[s][2] += a[4] == c[4]
		d = [a[i] - c[i] for i in range(4)]
		print(f"    {'s='+s:12s} {fmt(a)}   Δ stable {d[0]:+.1f}pp err {d[1]:+.2f}° steady {d[2]:+.2f}° alt {d[3]:+.3f}m   {w}")
		steady_curve.append(a[2]); alt_curve.append(a[3])
	if len(steady_curve) == len(scales):
		mono = lambda v: "monotone ↓" if all(x > y for x, y in zip(v, v[1:])) else ("monotone ↑" if all(x < y for x, y in zip(v, v[1:])) else "not monotone")
		print(f"    2→4→8  steady {steady_curve}  {mono(steady_curve)}   alt {alt_curve}  {mono(alt_curve)}")
print("\n  PAIRED TALLY vs control (same-rule):  " + "   ".join(f"s={s}: arm {t[0]} - {t[1]} control, {t[2]} tie" for s, t in tally.items()))
print("  Paired majority is the standard; n<=4 is a DIRECTION. Steady and alt are the primaries —")
print("  err is ~80% transient and is NOT expected to move. hd cannot see steady or alt: read all four columns.")
PY
log "########## LABEL-SCALE ARM A COMPLETE ##########"
