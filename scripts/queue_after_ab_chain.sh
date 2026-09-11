#!/usr/bin/env bash
# QUEUE BEHIND THE MUTATION-STEP A/B — (1) the DAgger-ROUND racing rung, (2) the
# leak-0.90 ladder at the committed shape, (3) window-k, FRAMED. (07/09/2026,
# Luiz's order: "the DAgger-round rung first, then the leak ladder, and finally
# arm window-k after the A/B".)
#
# ONE CONTROLLER AT A TIME is the standing rule and the mutation-step A/B was
# already flying, so this chain WAITS — it never preempts. Every point is
# marker-gated and idempotent; it FAILS CLOSED, leaving the box idle rather than
# stacking work on a crash. The controls for steps 2 and 3 are the SAME four
# banked b24 n256 CRN runs, so both steps are paired with no control re-fly.
#
# ---- STEP 1: the DAgger-ROUND rung (~1.5 h) --------------------------------
# The FOLD rung closed 07/09: exact (identical=True) but its rank signal was noise
# (rho ~0.2), because K folds are five DRAWS OF THE SAME DISTRIBUTION. Rounds are
# different in kind: round 3 of 8 is a genuinely less-trained policy. THE CHECK
# ORDER IS REVERSED ON PURPOSE: exactness here is already known FALSE without a
# Rust change (dagger_train.rs round_tilt_rad ramps frac=it/(num_rounds-1), so a
# split call ramps the curriculum twice), so this probe measures PREDICTIVITY ONLY
# and we buy the Rust work only if the signal survives. Smoke first, fails closed.
#
# ---- STEP 2: leak 0.90 at b24 n256 (4 runs, ~18.5 h) -----------------------
# The screen (n=1, CRN) was flown at b32 n64 — 16 levels/motor, a shape the
# programme has left: the bits curve closed 07/09 as a tie and the H743 flash
# constraint made b24 n256 (64 levels/motor) the published family. Leak's whole
# mechanism is the sustainable offset quantum/(1-leak), and the QUANTUM is 4x
# smaller at 64 levels than at 16, so the n64 result does not transfer — the
# leak x alphabet interaction the leak chain itself called "never measured" is
# exactly what this measures. (Luiz 07/09: "we are committing to the n256 family".)
# Control = the banked runs (they flew at the default --delta-leak 0.95,
# phased_ga.py:2268). Half the runs of the b32 n64 version this replaced.
#
# ---- STEP 3: window-k, k = 2/3/4, FRAMED (12 runs, ~55 h) --------------------
# Luiz 07/09: "where are the runs testing the window k=1..4?" — nowhere: the
# specialist round-3 sweep (16/08) designed it and NEVER RAN (no markers, no dir).
# TWO TRAPS this design has to clear, both verified 07/09:
#   (a) at sn=0 the OUTPUT layer samples `sensor_frame` (one frame, 144 bits);
#       `input_window_k` only sizes the STATE layer's pool (genome.py:59-60,121).
#       The banked winners' spec says k=4 and their exported header's highest tap
#       index is 143 — k was INERT. --input-window-k alone would fly 12
#       bit-identical runs. --output-full-window (arm D) is what makes k real.
#   (b) a CONNECTIONS-stage rewire under the default scope=free may land in ANY
#       frame, so a framed initial map would dissolve toward uniform over 5 gens.
#       --conn-mutation-scope window pins each tap to its frame — and at k=1 it
#       "degenerates to free", which is WHY the banked free-scope runs are exactly
#       the k=1 point of this family.
# THE DESIGN (Luiz's framing, 16/08 + 07/09): each output neuron is assigned ONE
# frame, newest frames heaviest — recency weights 2^slot, exact largest-remainder
# counts per 4-motor block (arch_ops::framed1_slot_schedule), so every motor gets
# the same share of every frame. At n=256 (64 levels/motor):
#     k=2  172/84      k=3  148/72/36    k=4  136/68/36/16   newest->oldest (2^slot, slot k-1 = newest)
# Luiz's 07/09 recollection was 3:2:1 at k=3; the code's schedule is the geometric
# one he specified on 16/08, kept as-is ("the numbers can be different") — a
# different ratio is a Rust knob, deferred.
#   flags    --conn-policy framed1 --output-full-window --input-window-k K
#            --frame-stride 10 --conn-mutation-scope window
#   stride   10 -> frames 10 ms apart, k=4 = 40 ms look-back (at stride 1 the
#            thermometer moves ~1 bit/step and the frames are near-copies).
#   order    round-major per the sweep rule: k2 s2, k3 s2, k4 s2, then seed 3...
#   budget   k grows the input POOL, not the address space (2^b): populated cells
#            and the 180k cap are comparable across k.
#   marker   SL_WINDOW_K makes the marker record the real k (it used to hardcode 1).
#   deploy   the exporter's wnn_conn is uint8; a k>=2 pool (288-576) overflows it.
#            A k>=2 WINNER needs conn as uint16 (+~6 KB, still fits). Not a blocker
#            for the experiment; fix only if an arm wins.
set -u
ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1

LOG="/private/tmp/queue_after_ab.log"
RMARK="experiments/racing_markers"
SMARK="experiments/sweepladder_markers"
LADDER="scripts/sweep_ladder_gamma.sh"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
CKPT="logs/controller/translation_ab/ckpt/TAB_on_b32n256_cf21_brushless_L4C_s31337002/stage3_connections.yaml.gz"
FAM_SEEDS="${QAB_SEEDS:-31337002 31337003 31337004 31337005}"

log() { echo "[queue-after-ab] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controller_pids() { pgrep -f -- "-m wnn.control.phased_ga" 2>/dev/null || true; }
probe_pids() { pgrep -f "scripts/racing_(fold|round)_probe.py" 2>/dev/null || true; }
chain_pids() { pgrep -f "scripts/(mutstep_ab|leak_revisit|sweep_ladder_gamma)" 2>/dev/null || true; }
wait_box_clear() {
	local beat=0
	while [ -n "$(controller_pids)" ] || [ -n "$(probe_pids)" ] || [ -n "$(chain_pids)" ]; do
		[ $((beat % 30)) = 0 ] && log "waiting — box busy (controller/probe/chain still running)"
		beat=$((beat + 1)); sleep 60
	done
}
ab_done() { ls "${SMARK}"/SL_C_b24n256_*_mut1tap.json 2>/dev/null | wc -l | tr -d ' '; }
ctrl_tag() { if [ "$1" = "31337002" ]; then echo "SL_C_b24n256_cf21_brushless_L4C_g10_s31337002_crn"; else echo "SL_C_b24n256_cf21_brushless_L4C_g10_s$1"; fi; }

mkdir -p "$RMARK" logs/controller/racing_probe
log "########## ARMED — 1 DAgger-round rung · 2 leak-0.90 @ b24n256 · 3 window-k 2/3/4 FRAMED @ b24n256 ##########"
log "waiting behind the mutation-step A/B (currently $(ab_done)/4 markers). Never preempts."
for s in $FAM_SEEDS; do
	[ -f "${SMARK}/$(ctrl_tag "$s").json" ] || { log "ABORT — control $(ctrl_tag "$s") missing; steps 2/3 would be unpaired."; exit 1; }
done
log "all 4 b24 n256 controls present — steps 2 and 3 are paired."
# Print the exact framed split so the design is on record before a run flies.
PYTHONPATH=src/wnn $VP - >> "$LOG" <<'PY'
from collections import Counter
from wnn.control import _accel as ra
for k in (2, 3, 4):
	c = Counter(int(s) for s in ra.arch_framed1_slot_schedule(256, k, 4, 12345, 0, 0, 0))
	# arch_ops.rs: slot 0 = OLDEST frame, slot k-1 = newest (Luiz's "window0"). Print newest first.
	tot = [c[i] for i in reversed(range(k))]
	print(f"  framed1 split at n=256, k={k}: newest->oldest {tot} (per motor {[t // 4 for t in tot]})")
PY

RA="$($VP -c "import json;print(json.load(open('${RMARK}/PROBE_stage3_s31337002.json'))['recipe_args'])")"
[ -n "$RA" ] || { log "ABORT — cannot read the recipe args from the fold probe's marker."; exit 1; }

# ---- STEP 1a: smoke. 3 candidates, a 2-point grid, tiny episodes. FAILS CLOSED.
SMOKE="${RMARK}/PROBE_rounds_smoke.json"
if [ -f "$SMOKE" ]; then
	log "SKIP round-probe smoke (marker exists)."
else
	wait_box_clear
	log "===== STEP 1a START round-probe smoke (3 candidates, grid 1,2) ====="
	PYTHONPATH=src/wnn $VP -u scripts/racing_round_probe.py --ckpt "$CKPT" \
		--recipe-args "$RA --rg-episodes-per-round 2 --rg-eval-episodes 2 --eval-episodes 2" \
		--candidates 3 --rounds-grid 1,2 --out "$SMOKE" \
		> logs/controller/racing_probe/rounds_smoke.out 2>&1
	rc=$?
	log "round-probe smoke rc=${rc}"
	[ "$rc" = "0" ] && [ -f "$SMOKE" ] || {
		log "ABORT — round-probe smoke FAILED; nothing else runs. Last lines:"
		tail -8 logs/controller/racing_probe/rounds_smoke.out | while read -r l; do log "    $l"; done
		exit 1; }
fi

# ---- STEP 1b: the real round probe at the CONNECTIONS population.
ROUT="${RMARK}/PROBE_rounds_stage3_s31337002.json"
if [ -f "$ROUT" ]; then
	log "SKIP round probe (marker exists)."
else
	wait_box_clear
	log "===== STEP 1b START round probe: 60 candidates, grid 1,2,3,4,5,6,8 ====="
	PYTHONPATH=src/wnn $VP -u scripts/racing_round_probe.py --ckpt "$CKPT" --recipe-args "$RA" \
		--candidates 60 --rounds-grid 1,2,3,4,5,6,8 --out "$ROUT" \
		> logs/controller/racing_probe/rounds_stage3_s31337002.out 2>&1
	rc=$?
	log "round probe rc=${rc}"
	[ "$rc" = "0" ] && [ -f "$ROUT" ] || {
		log "ABORT — round probe failed (rc=${rc}). A run needs a human. Box left idle."
		tail -8 logs/controller/racing_probe/rounds_stage3_s31337002.out | while read -r l; do log "    $l"; done
		exit 1; }
	log "banked: $(basename "$ROUT")"
	log "---------- ROUND-RUNG READ (predictivity only; exactness NOT measured) ----------"
	$VP - "$ROUT" >> "$LOG" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for c in d["cuts"]:
	print("  cut after round %d: spearman %+0.3f  top-third kept %d/%d  regret %.4f  "
	      "true-best survives %s  train-units %.2f"
	      % (c["cut_after_round"], c["spearman"], c["top_kept"], c["top_size"],
	         c["regret"], c["true_best_survives"], c["train_units"]))
print("  The FOLD rung closed at rho ~0.24 / 10-11 of 20 kept. A rung earns Rust work only")
print("  if an EARLY cut holds most of the true top third AND keeps the true best.")
PY
fi

# ---- STEPS 2 + 3 share one ladder arm runner.
run_ladder_arm() { # $1 seed  $2 suffix  $3 extra args  $4 extra marker json  $5 window_k
	local seed="$1" suf="$2" extra="$3" mj="$4" wk="$5"
	local tag="SL_C_b24n256_cf21_brushless_L4C_g10_s${seed}${suf}"
	if [ -f "${SMARK}/${tag}.json" ]; then log "SKIP ${tag} (marker exists)."; return 0; fi
	wait_box_clear
	log "===== START ${tag} (${extra}) ====="
	SL_SKIP_PHASE1=1 SL_FORCE_PHASE2_GAMMA="1.0" SL_WIDTHS="24" SL_NEURONS="256" SL_SEED="$seed" \
		SL_SWEEP_LABEL="${suf#_}" SL_TAG_SUFFIX="$suf" SL_WINDOW_K="$wk" \
		SL_EXTRA_ARGS="$extra --teacher-hover ${TEACHER_HOVER:-derived}" SL_EXTRA_MARKER_JSON="$mj" bash "$LADDER"
	log "ladder exited rc=$? for ${tag}"
	while [ -n "$(controller_pids)" ]; do sleep 30; done
	[ -f "${SMARK}/${tag}.json" ] || { log "ABORT — marker ${tag}.json MISSING. A run needs a human. Box left idle."; exit 1; }
	log "banked: ${tag}"
}

# ---- STEP 2: leak 0.90 at b24 n256, paired against the banked 0.95 controls.
log "===== STEP 2 leak-0.90 @ b24 n256: seeds [${FAM_SEEDS}], control = banked (default leak 0.95) ====="
for seed in $FAM_SEEDS; do
	run_ladder_arm "$seed" "_leak090" "--delta-leak 0.90" \
		",\"arm\":\"leak090\",\"delta_leak\":0.90,\"control_tag\":\"$(ctrl_tag "$seed")\"" 1
done

# ---- STEP 3: window-k FRAMED, round-major so every k has a point after 3 runs.
log "===== STEP 3 window-k FRAMED @ b24 n256: k in [2 3 4] x seeds [${FAM_SEEDS}] ====="
for seed in $FAM_SEEDS; do
	for k in 2 3 4; do
		run_ladder_arm "$seed" "_win${k}" \
			"--conn-policy framed1 --output-full-window --input-window-k ${k} --frame-stride 10 --conn-mutation-scope window" \
			",\"arm\":\"win${k}\",\"conn_policy\":\"framed1\",\"output_full_window\":true,\"frame_stride\":10,\"conn_mutation_scope\":\"window\",\"recency_weights\":\"2^slot\",\"control_tag\":\"$(ctrl_tag "$seed")\"" \
			"$k"
	done
done

# ---- VERDICT: paired per-seed on the MEMORY same-rule row, all four columns.
$VP scripts/gate_distance_leaderboard.py > docs/controller_gate_distance_leaderboard.md
log "---------- VERDICT: each arm paired same-seed vs the banked b24 n256 control (lower same-rule = better) ----------"
QAB_SEEDS_LIST="$FAM_SEEDS" $VP - >> "$LOG" <<'PY'
import json, math, os, re
K = math.log(0.5) / math.log(0.70)
def gd(s, e): return 0.5556 * (e / 8.0) + 0.4444 * min(K * -math.log2(s / 100), 20.0)
def row(tag):
	p = f"experiments/sweepladder_markers/{tag}.json"
	if not os.path.exists(p): return None
	m = json.load(open(p)).get("held_memory_multiseed", "")
	f = lambda pat: float(re.search(pat, m).group(1))
	try:
		return (f(r"stable=([0-9.]+)"), f(r"err=([0-9.]+)"), f(r"steady=([0-9.]+)"),
		        f(r"alt=([0-9.]+)"), gd(f(r"stable=([0-9.]+)"), f(r"err=([0-9.]+)")))
	except Exception:
		return None
fmt = lambda r: f"{r[0]:5.1f}% {r[1]:.2f} {r[2]:.2f} {r[3]:.3f}m hd {r[4]:.4f}"
seeds = os.environ["QAB_SEEDS_LIST"].split()
for arm in ("_leak090", "_win2", "_win3", "_win4"):
	wa = wc = 0
	print(f"\n  ARM {arm[1:]:<8} seed       control (banked, k=1, leak .95)          arm                                    winner")
	for s in seeds:
		ct = f"SL_C_b24n256_cf21_brushless_L4C_g10_s{s}" + ("_crn" if s == "31337002" else "")
		c, a = row(ct), row(f"SL_C_b24n256_cf21_brushless_L4C_g10_s{s}{arm}")
		if not c or not a:
			print(f"               {s}   {'control missing' if not c else 'arm not flown yet':<38}   —"); continue
		# A TIE is neither side's win — the tally already refuses to score it, so
		# the label must refuse too. An `else` here credited every tie to control.
		w = "arm" if a[4] < c[4] else ("control" if c[4] < a[4] else "TIE")
		wa += a[4] < c[4]; wc += c[4] < a[4]
		print(f"               {s}   {fmt(c):<38}   {fmt(a):<38}   {w}")
	print(f"               PAIRED TALLY  arm {wa} - {wc} control")
print("\n  Paired majority is the standard; n<=4 is a DIRECTION. hd cannot see alt or steady —")
print("  read all four columns (the leak screen won hd and LOST 0.178 m of altitude).")
PY
log "########## QUEUE COMPLETE — round rung + leak ladder + window-k ##########"
