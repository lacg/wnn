#!/usr/bin/env bash
# QUEUE BEHIND THE MUTATION-STEP A/B — (1) the DAgger-ROUND racing rung, then
# (2) the leak-0.90 multi-seed ladder. (07/09/2026, Luiz's order: "let's do this
# [the DAgger-round rung] first now, then the leak ladder, and finally let's arm
# window-k next after the A/B".)
#
# ONE CONTROLLER AT A TIME is the standing rule, and the mutation-step A/B was
# already flying when this was written, so this chain WAITS — it never preempts a
# running run. Everything is marker-gated and idempotent; it FAILS CLOSED, leaving
# the box idle rather than stacking work on a crash.
#
# ---- STEP 1: the DAgger-ROUND rung (~1.5 h) --------------------------------
# The FOLD rung closed 07/09: exact (identical=True) but the rank signal was noise
# (rho ~0.2 at CONNECTIONS, ~0.1 at GRID), because K folds are five DRAWS OF THE
# SAME DISTRIBUTION — ranking after fold 1 re-measures luck. Rounds are different in
# kind: round 3 of 8 is a genuinely less-trained policy on a learning curve.
# THE CHECK ORDER IS REVERSED ON PURPOSE. The fold probe paid for exactness first
# and the answer turned out not to matter. Here exactness is already known to be
# FALSE without a Rust change (dagger_train.rs round_tilt_rad ramps
# frac=it/(num_rounds-1), so a split call ramps the curriculum twice), so this
# probe measures PREDICTIVITY ONLY and we buy the Rust work only if the signal
# survives. Smoke first, fails closed.
#
# ---- STEP 2: the leak-0.90 ladder (6 runs, ~20 h) --------------------------
# The screen (n=1, seed 31337002, all CRN): leak 0.90 beat the 0.95 control by
# 1.09 deg err and 7.2 pp stable — past the ~0.4 deg / ~2.5 pp pool noise — while
# PAYING 0.178 m of altitude hold (0.626 vs 0.448 m). n=1 is a direction, so this
# promotes it to a paired multi-seed ladder, which is what the queue block said a
# win would earn.
# WHY BOTH ARMS FLY. There is NO b32 n64 CRN control at seeds 3/4/5 — only seed 2
# has one (SL_C_b32n64_..._s31337002_crn, byte-identical recipe bar --delta-leak).
# Flying only the 0.90 arm at new seeds would produce an UNPAIRED cohort compared
# against a single-seed control: the exact mistake IDSXD paid for. So each new seed
# flies 0.90 AND its own 0.95 control through the SAME script (run_point 0.95 is
# the default leak), giving 3 new paired seeds + the banked seed-2 pair = n=4.
set -u
ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1

LOG="/private/tmp/queue_round_leak.log"
RMARK="experiments/racing_markers"
LMARK="experiments/leakrevisit_markers"
SMARK="experiments/sweepladder_markers"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
CKPT="logs/controller/translation_ab/ckpt/TAB_on_b32n256_cf21_brushless_L4C_s31337002/stage3_connections.yaml.gz"
LEAK_SEEDS="${QRL_LEAK_SEEDS:-31337003 31337004 31337005}"

log() { echo "[queue-round-leak] $(date -u +%FT%TZ) $*" >> "$LOG"; }
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

# The A/B is a 4-marker arm; report its progress in the log so a reader can see WHY
# this chain is asleep, but do NOT gate on it — if Luiz kills the A/B, this should
# take the box, not deadlock waiting for markers that will never arrive.
ab_done() { ls "${SMARK}"/SL_C_b24n256_*_mut1tap.json 2>/dev/null | wc -l | tr -d ' '; }

mkdir -p "$RMARK" logs/controller/racing_probe
log "########## ARMED — step 1 DAgger-round rung, step 2 leak-0.90 ladder ##########"
log "waiting behind the mutation-step A/B (currently $(ab_done)/4 markers). Never preempts."

RA="$($VP -c "import json;print(json.load(open('${RMARK}/PROBE_stage3_s31337002.json'))['recipe_args'])")"
[ -n "$RA" ] || { log "ABORT — cannot read the recipe args from the fold probe's marker."; exit 1; }

# ---- STEP 1a: smoke. 3 candidates, a 2-point grid, tiny episodes. FAILS CLOSED.
SMOKE="${RMARK}/PROBE_rounds_smoke.json"
if [ -f "$SMOKE" ]; then
	log "SKIP round-probe smoke (marker exists)."
else
	wait_box_clear
	log "===== STEP 1a START round-probe smoke (3 candidates, grid 1,2) ====="
	$VP -u scripts/racing_round_probe.py --ckpt "$CKPT" \
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
	$VP -u scripts/racing_round_probe.py --ckpt "$CKPT" --recipe-args "$RA" \
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
print("  Compare against the FOLD rung, which was closed at rho ~0.24 / 10-11 of 20 kept.")
print("  A rung is only worth Rust work if an EARLY cut holds most of the true top third")
print("  AND keeps the true best. Exactness is a SEPARATE, unpaid question (round_tilt_rad).")
PY
fi

# ---- STEP 2: the leak ladder. Both arms per seed, so every seed is paired.
log "===== STEP 2 leak-0.90 ladder: seeds [${LEAK_SEEDS}], arms 0.90 + 0.95 control ====="
[ -f "${SMARK}/SL_C_b32n64_cf21_brushless_L4C_g10_s31337002_crn.json" ] || {
	log "ABORT — the seed-2 CRN control is missing; the ladder would have no anchor."; exit 1; }
for seed in $LEAK_SEEDS; do
	for leak in 0.90 0.95; do
		ltag="l$(echo "$leak" | tr -d '.')"
		tag="LKR_${ltag}_b32n64_cf21_brushless_L4C_g10_s${seed}"
		if [ -f "${LMARK}/${tag}.json" ]; then
			log "SKIP ${tag} (marker exists)."; continue
		fi
		wait_box_clear
		log "===== START ${tag} (seed ${seed}, delta_leak ${leak}) ====="
		LKR_SEED="$seed" LKR_LEAKS="$leak" bash scripts/leak_revisit_chain.sh
		log "leak chain exited rc=$? for seed ${seed} leak ${leak}"
		while [ -n "$(controller_pids)" ]; do sleep 30; done
		[ -f "${LMARK}/${tag}.json" ] || {
			log "ABORT — marker ${tag}.json MISSING. A run needs a human. Box left idle."; exit 1; }
		log "banked: ${tag}"
	done
done

# ---- VERDICT: paired per-seed, 0.90 vs its own 0.95 control, ALL FOUR columns.
$VP scripts/gate_distance_leaderboard.py > docs/controller_gate_distance_leaderboard.md
log "---------- LEAK LADDER VERDICT: paired same-seed, MEMORY same-rule ----------"
QRL_SEEDS="31337002 ${LEAK_SEEDS}" $VP - >> "$LOG" <<'PY'
import json, math, os, re
K = math.log(0.5) / math.log(0.70)
def gd(s, e): return 0.5556 * (e / 8.0) + 0.4444 * min(K * -math.log2(s / 100), 20.0)
def row(path):
	if not os.path.exists(path): return None
	m = json.load(open(path)).get("held_memory_multiseed", "")
	f = lambda p: float(re.search(p, m).group(1))
	try:
		return (f(r"stable=([0-9.]+)"), f(r"err=([0-9.]+)"), f(r"steady=([0-9.]+)"),
		        f(r"alt=([0-9.]+)"), gd(f(r"stable=([0-9.]+)"), f(r"err=([0-9.]+)")))
	except Exception:
		return None
w90 = w95 = 0
print("  seed        leak 0.90 (stable/err/steady/alt/hd)      control 0.95                        winner")
for s in os.environ["QRL_SEEDS"].split():
	a = row(f"experiments/leakrevisit_markers/LKR_l090_b32n64_cf21_brushless_L4C_g10_s{s}.json")
	# Seed 2's control came from the ladder, not the leak script — byte-identical
	# recipe bar --delta-leak, which is why it is a legitimate anchor.
	b = row(f"experiments/leakrevisit_markers/LKR_l095_b32n64_cf21_brushless_L4C_g10_s{s}.json") or \
	    row(f"experiments/sweepladder_markers/SL_C_b32n64_cf21_brushless_L4C_g10_s{s}_crn.json")
	if not a or not b:
		print(f"  {s}    {'0.90 not flown' if not a else 'control not flown':<38}   —"); continue
	win = "0.90" if a[4] < b[4] else "0.95"
	w90 += a[4] < b[4]; w95 += b[4] < a[4]
	fmt = lambda r: f"{r[0]:5.1f}% {r[1]:.2f} {r[2]:.2f} {r[3]:.3f}m hd {r[4]:.4f}"
	print(f"  {s}    {fmt(a):<38}   {fmt(b):<38}   {win}")
print(f"\n  PAIRED TALLY  0.90 {w90} - {w95} 0.95")
print("  ALTITUDE IS THE COLUMN THAT CARRIES THE TRADE: at seed 2 the 0.90 arm won")
print("  stable/err/steady and LOST alt by 0.178 m. gate-distance cannot see alt, so a")
print("  tally win is NOT a free win — read all four columns before promoting leak.")
PY
log "########## QUEUE COMPLETE — round rung + leak ladder ##########"
