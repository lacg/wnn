#!/usr/bin/env bash
# CRN BITS CURVE — does the bits round-2 ordering survive the scorer change?
# (06/09/2026, Luiz: "#1, #2" of the post-A/B queue.)
#
# THE QUESTION. Bits round 2 picked b=32 (n=256, gamma=1) by averaging each width
# over seeds 31337002 and 31337003 — but those two seeds fell in DIFFERENT scorer
# eras (seed 2 rotation, seed 3 CRN), unevenly per width. The b24 CRN re-fly on
# 06/09 came in at gate-dist 0.1029 against a rotation-era 0.1972 at the same
# seed, so the curve that crowned b32 was measured against a handicapped b24.
# Restricting to CRN-era runs only (06/09 08:40 EDT):
#     b24  n=2  mean 0.1150        b28  n=1  0.1349        b32  n=5  mean 0.1325
# Narrower leads. Every run since round 2 — the five-seed replication, the leak
# screen's shape — has flown at a width that may not be the optimum.
#
# DESIGN. Two steps, sequential, ONE controller at a time:
#   #1  b28 n256 seed 31337002 under CRN (the one missing cell for a 3-width x
#       2-seed CRN curve). Via crn_refly_chain.sh, exactly as b24 was re-flown:
#       same ladder invocation, SL_TAG_SUFFIX=_crn, only the scorer differs.
#   #2  b24 n256 seeds 31337004 and 31337005 — NEW seeds, so no rotation control
#       exists and crn_refly_chain.sh would refuse ("nothing to pair against").
#       Flown via the ladder script directly with the same hooks bits round 2
#       used for seed 3 (SL_SKIP_PHASE1, forced gamma=1, one width, one seed).
#       Balances b24 (2 -> 4 seeds) against b32's five.
#   Luiz chose to queue #2 unconditionally; the verdict at the end prints the
#   CRN-era per-width means so the ordering can be read either way.
#
# GATING. Waits for the running 0.95 leak-control re-fly (a crn_refly_chain.sh
# instance) AND any controller to finish — pure wait, never preempts. Each
# point is marker-gated and idempotent (the ladder SKIPs an existing marker).
# FAILS CLOSED: a missing marker after a point means a run needs a human; the
# chain stops and leaves the box idle rather than stacking work on a crash.
set -u
ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1

LOG="/private/tmp/crn_bits_curve.log"
LADDER="scripts/sweep_ladder_gamma.sh"
REFLY="scripts/crn_refly_chain.sh"
MARK="experiments/sweepladder_markers"
AIRFRAME="cf21_brushless"; DIST="L4C"
NEURONS=256
NEW_SEEDS="${CBC_B24_SEEDS:-31337004 31337005}"

log() { echo "[crn-bits-curve] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controller_pids() { pgrep -f -- "-m wnn.control.phased_ga" 2>/dev/null || true; }
refly_pids() { pgrep -f "scripts/crn_refly_chain.sh" 2>/dev/null || true; }
ladder_pids() { pgrep -f "scripts/sweep_ladder_gamma.sh" 2>/dev/null || true; }
wait_box_clear() {
	local beat=0
	while [ -n "$(controller_pids)" ] || [ -n "$(refly_pids)" ] || [ -n "$(ladder_pids)" ]; do
		[ $((beat % 30)) = 0 ] && log "waiting — box busy (controller/refly/ladder still running)"
		beat=$((beat + 1)); sleep 60
	done
}

log "########## ARMED — #1 b28 s31337002 under CRN, then #2 b24 seeds [${NEW_SEEDS}] ##########"
for t in "SL_C_b28n${NEURONS}_${AIRFRAME}_${DIST}_g10_s31337002"; do
	[ -f "${MARK}/${t}.json" ] || { log "ABORT — control marker ${t}.json missing."; exit 1; }
done

# ---- #1: b28 seed 2 under CRN, via the re-fly chain (blocking; it waits on its own ladder).
wait_box_clear
T1="SL_C_b28n${NEURONS}_${AIRFRAME}_${DIST}_g10_s31337002_crn"
if [ -f "${MARK}/${T1}.json" ]; then
	log "#1 SKIP — ${T1} already banked."
else
	log "===== #1 START ${T1} via ${REFLY} ====="
	CRN_BITS=28 CRN_NEURONS="$NEURONS" CRN_SEED=31337002 bash "$REFLY"
	log "#1 refly chain exited rc=$?"
	while [ -n "$(controller_pids)" ] || [ -n "$(ladder_pids)" ]; do sleep 30; done
	[ -f "${MARK}/${T1}.json" ] || { log "ABORT — #1 marker ${T1}.json MISSING after the re-fly (watchdog kill / crash / no MEMORY triple). A run needs a human. Box left idle."; exit 1; }
	log "#1 banked: ${T1}"
fi

# ---- #2: b24 at new seeds, via the ladder directly (no rotation control exists).
for seed in $NEW_SEEDS; do
	T2="SL_C_b24n${NEURONS}_${AIRFRAME}_${DIST}_g10_s${seed}"
	if [ -f "${MARK}/${T2}.json" ]; then
		log "#2 SKIP — ${T2} already banked."; continue
	fi
	wait_box_clear
	log "===== #2 START ${T2} via ${LADDER} (SL_SKIP_PHASE1, gamma=1 forced, CRN default) ====="
	SL_SKIP_PHASE1=1 SL_SWEEP_LABEL="crn-bits-curve" SL_FORCE_PHASE2_GAMMA="1.0" \
		SL_WIDTHS="24" SL_NEURONS="$NEURONS" SL_SEED="$seed" bash "$LADDER"
	log "#2 ladder exited rc=$? for seed ${seed}"
	while [ -n "$(controller_pids)" ]; do sleep 30; done
	[ -f "${MARK}/${T2}.json" ] || { log "ABORT — #2 marker ${T2}.json MISSING. A run needs a human. Box left idle."; exit 1; }
	log "#2 banked: ${T2}"
done

# ---- VERDICT: the leaderboard is the bar; then the CRN-era per-width means.
python3 scripts/gate_distance_leaderboard.py > docs/controller_gate_distance_leaderboard.md
log "---------- VERDICT: CRN-era means per width at n=256 (gate-dist / same-rule), seeds listed ----------"
/usr/bin/python3 - >> "$LOG" <<'PY'
import glob, json, math, os, re, statistics as st
K = math.log(0.5) / math.log(0.70)
def gd(s, e): return 0.5556 * (e / 8.0) + 0.4444 * min(K * -math.log2(s / 100), 20.0)
def era(tag):
	for d in ("sweep_ladder", "translation_ab"):
		p = f"logs/controller/{d}/{tag}.out"
		if os.path.exists(p):
			t = open(p, errors="ignore").read(200000)
			m = re.search(r"fitness_pools=(CRN|rotation)", t)
			return ("CRN" if m.group(1) == "CRN" else "rot") if m else ("rot" if "Phased-GA controller search:" in t else "?")
	return "?"
rows = []
for p in glob.glob("experiments/*_markers/*.json"):
	tag = os.path.basename(p)[:-5]
	m = re.match(r"(?:SL_C|TAB_on)_b(\d+)n256_cf21_brushless_L4C_(?:g10_)?s(\d+)(_crn)?$", tag)
	if not m: continue
	d = json.load(open(p)); h = d.get("headline_holdout", ""); mem = d.get("held_memory_multiseed", "")
	if not h or not mem: continue
	f = lambda t, pat: float(re.search(pat, t).group(1))
	rows.append((int(m.group(1)), m.group(2), era(tag),
	             gd(f(h, r"stable=([0-9.]+)"), f(h, r"err=([0-9.]+)")),
	             gd(f(mem, r"stable=([0-9.]+)"), f(mem, r"err=([0-9.]+)"))))
for b in sorted({r[0] for r in rows}):
	c = [r for r in rows if r[0] == b and r[2] == "CRN"]
	if c:
		print("  b%2d  n=%d  gate-dist %.4f  same-rule %.4f   seeds %s" % (
			b, len(c), st.mean(x[3] for x in c), st.mean(x[4] for x in c), ",".join(sorted(x[1][-1] for x in c))))
	else:
		print("  b%2d  no CRN-era run" % b)
print("  Paired majority is the standard; n<=4 per width is a DIRECTION, not a claim.")
PY
log "########## CRN BITS CURVE COMPLETE ##########"
