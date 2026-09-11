#!/usr/bin/env bash
# 2x2 LEAK x LABEL-SCALE — does the label scale remove the leak arm's altitude cost?
# (09/09/2026, Luiz: "queue the 2x2 leak x label-scale after arm A".)
#
# THE QUESTION. The leak-0.90 ladder (08/09) won attitude 3-1 and lost altitude 4-0
# (+0.10 m, 0.35 -> 0.45 m): in gain terms G = 2·dmax/(1−leak) it is a gain-2 /
# tau-10 ms controller against the recipe's gain-4 / tau-20 ms, and it bleeds the
# rarely-labelled collective correction twice as fast. The label scale s narrows
# the dead zone that makes that correction rare. If s removes the altitude cost,
# leak 0.90's attitude gain becomes free and goes in the recipe; if not, the two
# levers are independent and the leak stays a trade.
#
# THE 2x2, four seeds each, MEMORY same-rule row, ALL FOUR columns:
#   (leak 0.95, s=1)  = the banked b24 n256 CRN controls          (no re-fly)
#   (leak 0.90, s=1)  = the banked leak ladder, _leak090          (no re-fly)
#   (leak 0.95, s=s*) = ARM A's winning rung, _ls{s*}             (no re-fly)
#   (leak 0.90, s=s*) = THIS chain: 4 runs, tagged _l090_ls{s*}   (the missing cell)
# Read = the INTERACTION on altitude: [alt(0.90,s*) − alt(0.95,s*)] vs
# [alt(0.90,1) − alt(0.95,1)], paired per seed; and the same on steady and err.
#
# s* SELECTION (explicit, overridable with LS_STAR=<2|4|8>): among arm A's rungs,
# the one with the most PAIRED SEED WINS vs the s=1 control on STEADY (the primary),
# tie-break paired wins on ALT, then lowest mean same-rule hd. A rung must beat the
# control on steady at >= 3 of 4 seeds to be crossed at all — if none does, arm A
# refuted the mechanism and there is nothing to cross: the chain ABORTS and says so.
#
# PRECONDITIONS (fail closed): arm A COMPLETE (12 _ls markers); box idle; the four
# _leak090 and the four _ls{s*} markers present; wheel + Python carry the flags.
# Every point is marker-gated and idempotent; a missing marker after a run stops
# the chain and leaves the box idle.
set -u
# D0 (11/09): every run after the fix carries --teacher-hover ${TEACHER_HOVER:-derived}.
# The s=2 arm and leak-0.90 inputs were LEGACY-trained; the D0 A/B bounds that gap (~0).
ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1

LOG="/private/tmp/leak_x_labelscale.log"
LADDER="scripts/sweep_ladder_gamma.sh"
MARK="experiments/sweepladder_markers"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
AIRFRAME="cf21_brushless"; DIST="L4C"
BITS=24; NEURONS=256
DMAX0="0.1"; LEAK="0.90"
SEEDS="${LX_SEEDS:-31337002 31337003 31337004 31337005}"
RUNGS="2 4 8"

log() { echo "[leak-x-ls] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controller_pids() { pgrep -f -- "-m wnn.control.phased_ga" 2>/dev/null || true; }
ladder_pids() { pgrep -f "scripts/sweep_ladder_gamma.sh" 2>/dev/null || true; }
other_chain_pids() { pgrep -f "scripts/(mutstep_ab|queue_after_ab|leak_revisit|crn_|translation_ab|window_k|label_scale_arm)[a-z_]*chain" 2>/dev/null || true; }
wait_box_clear() {
	local beat=0
	while [ -n "$(controller_pids)" ] || [ -n "$(ladder_pids)" ]; do
		[ $((beat % 30)) = 0 ] && log "waiting — box busy (controller/ladder still running)"
		beat=$((beat + 1)); sleep 60
	done
}
ctrl_tag() { if [ "$1" = "31337002" ]; then echo "SL_C_b${BITS}n${NEURONS}_${AIRFRAME}_${DIST}_g10_s31337002_crn"
             else echo "SL_C_b${BITS}n${NEURONS}_${AIRFRAME}_${DIST}_g10_s$1"; fi; }
base_tag() { echo "SL_C_b${BITS}n${NEURONS}_${AIRFRAME}_${DIST}_g10_s$1"; }
dmax_for() { $VP -c "print(${DMAX0}/$1)"; }

log "########## ARMED — 2x2 leak x label-scale: the missing cell (leak ${LEAK}, s=s*) at b${BITS} n${NEURONS} ##########"

# ---- PRECONDITIONS -------------------------------------------------------------
if [ -n "$(controller_pids)" ] || [ -n "$(ladder_pids)" ] || [ -n "$(other_chain_pids)" ]; then
	log "ABORT — the box is NOT idle. Never launch this chain beside another."; exit 1
fi
n_ls=$(ls "$MARK" | grep -cE "_ls[248]\.json$")
[ "$n_ls" -ge 12 ] || { log "ABORT — arm A incomplete (${n_ls}/12 _ls markers). This chain crosses arm A's winner; it cannot run first."; exit 1; }
for s in $SEEDS; do
	[ -f "${MARK}/$(base_tag "$s")_leak090.json" ] || { log "ABORT — leak-0.90 control $(base_tag "$s")_leak090 missing."; exit 1; }
	[ -f "${MARK}/$(ctrl_tag "$s").json" ] || { log "ABORT — s=1 control $(ctrl_tag "$s") missing."; exit 1; }
done
$VP -c "import ram_controller as c; sig=c.dagger_train_batch_inplace.__text_signature__ or ''; assert 'delta_label_scale' in sig" 2>>"$LOG" \
	|| { log "ABORT — installed wheel predates delta_label_scale."; exit 1; }

# ---- s* SELECTION ---------------------------------------------------------------
if [ -n "${LS_STAR:-}" ]; then
	STAR="$LS_STAR"; log "s* = ${STAR} (LS_STAR override)"
else
	STAR=$(LX_B="$BITS" LX_N="$NEURONS" LX_SEEDS_LIST="$SEEDS" LX_RUNGS="$RUNGS" $VP - 2>>"$LOG" <<'PY'
import json, math, os, re, sys
K = math.log(0.5) / math.log(0.70)
def gd(s, e): return 0.5556 * (e / 8.0) + 0.4444 * min(K * -math.log2(s / 100), 20.0)
B, N = os.environ["LX_B"], os.environ["LX_N"]
def row(tag):
	p = f"experiments/sweepladder_markers/{tag}.json"
	if not os.path.exists(p): return None
	m = json.load(open(p)).get("held_memory_multiseed", "")
	f = lambda pat: float(re.search(pat, m).group(1))
	try: return (f(r"stable=([0-9.]+)"), f(r"err=([0-9.]+)"), f(r"steady=([0-9.]+)"), f(r"alt=([0-9.]+)"),
	             gd(f(r"stable=([0-9.]+)"), f(r"err=([0-9.]+)")))
	except Exception: return None
seeds = os.environ["LX_SEEDS_LIST"].split(); rungs = os.environ["LX_RUNGS"].split()
best = None
for s in rungs:
	w_steady = w_alt = 0; hds = []
	for sd in seeds:
		c = row(f"SL_C_b{B}n{N}_cf21_brushless_L4C_g10_s{sd}" + ("_crn" if sd == "31337002" else ""))
		a = row(f"SL_C_b{B}n{N}_cf21_brushless_L4C_g10_s{sd}_ls{s}")
		if not (a and c): continue
		w_steady += a[2] < c[2]; w_alt += a[3] < c[3]; hds.append(a[4])
	key = (w_steady, w_alt, -sum(hds) / max(len(hds), 1))
	print(f"  rung s={s}: steady wins {w_steady}/4, alt wins {w_alt}/4, mean hd {sum(hds)/max(len(hds),1):.4f}", file=sys.stderr)
	if best is None or key > best[0]: best = (key, s)
if best is None or best[0][0] < 3:
	print(f"NONE", end=""); print("  no rung beats the s=1 control on steady at >=3 of 4 seeds — nothing to cross", file=sys.stderr)
else:
	print(best[1], end="")
PY
)
	[ "$STAR" != "NONE" ] && [ -n "$STAR" ] || { log "ABORT — arm A did not produce a rung worth crossing (see selection lines above). The 2x2 is moot."; exit 1; }
	log "s* = ${STAR} (rule: most paired steady wins vs s=1, tie-break alt wins, then mean hd; >=3/4 required)"
fi
for s in $SEEDS; do
	[ -f "${MARK}/$(base_tag "$s")_ls${STAR}.json" ] || { log "ABORT — arm-A cell $(base_tag "$s")_ls${STAR} missing."; exit 1; }
done
DMAX="$(dmax_for "$STAR")"
log "preflight OK — cell (leak ${LEAK}, s=${STAR}, dmax ${DMAX}); three other cells banked."

# ---- THE RUNS: the missing cell, one per seed ------------------------------------
for seed in $SEEDS; do
	TAG="$(base_tag "$seed")_l090_ls${STAR}"
	if [ -f "${MARK}/${TAG}.json" ]; then log "SKIP — ${TAG} already banked."; continue; fi
	wait_box_clear
	log "===== START ${TAG} (--delta-leak ${LEAK} --delta-label-scale ${STAR} --delta-max ${DMAX}) ====="
	SL_SKIP_PHASE1=1 SL_SWEEP_LABEL="leak-x-labelscale" SL_FORCE_PHASE2_GAMMA="1.0" \
		SL_WIDTHS="$BITS" SL_NEURONS="$NEURONS" SL_SEED="$seed" \
		SL_TAG_SUFFIX="_l090_ls${STAR}" \
		SL_EXTRA_ARGS="--delta-leak ${LEAK} --delta-label-scale ${STAR} --delta-max ${DMAX} --teacher-hover ${TEACHER_HOVER:-derived}" \
		SL_EXTRA_MARKER_JSON=",\"arm_2x2\":\"leak090_ls${STAR}\",\"delta_leak\":${LEAK},\"delta_label_scale\":${STAR},\"delta_max\":${DMAX},\"control_tag\":\"$(ctrl_tag "$seed")\"" \
		bash "$LADDER"
	log "ladder exited rc=$? for ${TAG}"
	while [ -n "$(controller_pids)" ]; do sleep 30; done
	[ -f "${MARK}/${TAG}.json" ] || { log "ABORT — marker ${TAG}.json MISSING. A run needs a human. Box left idle."; exit 1; }
	log "banked: ${TAG}"
done

# ---- VERDICT: the 2x2 per seed, four columns, and the altitude interaction ------
$VP scripts/gate_distance_leaderboard.py > docs/controller_gate_distance_leaderboard.md 2>/dev/null
log "---------- VERDICT: 2x2 leak x label-scale (s*=${STAR}), MEMORY same-rule row, ALL FOUR columns ----------"
LX_B="$BITS" LX_N="$NEURONS" LX_SEEDS_LIST="$SEEDS" LX_STAR="$STAR" $VP - >> "$LOG" <<'PY'
import json, math, os, re
K = math.log(0.5) / math.log(0.70)
def gd(s, e): return 0.5556 * (e / 8.0) + 0.4444 * min(K * -math.log2(s / 100), 20.0)
B, N, S = os.environ["LX_B"], os.environ["LX_N"], os.environ["LX_STAR"]
def row(tag):
	p = f"experiments/sweepladder_markers/{tag}.json"
	if not os.path.exists(p): return None
	m = json.load(open(p)).get("held_memory_multiseed", "")
	f = lambda pat: float(re.search(pat, m).group(1))
	try: return (f(r"stable=([0-9.]+)"), f(r"err=([0-9.]+)"), f(r"steady=([0-9.]+)"), f(r"alt=([0-9.]+)"),
	             gd(f(r"stable=([0-9.]+)"), f(r"err=([0-9.]+)")))
	except Exception: return None
fmt = lambda r: f"{r[0]:5.1f}% {r[1]:.2f}° {r[2]:.2f}° {r[3]:.3f}m hd {r[4]:.4f}" if r else "missing"
seeds = os.environ["LX_SEEDS_LIST"].split()
inter = {"alt": [], "steady": [], "err": []}
idx = {"err": 1, "steady": 2, "alt": 3}
for sd in seeds:
	base = f"SL_C_b{B}n{N}_cf21_brushless_L4C_g10_s{sd}"
	c11 = row(base + ("_crn" if sd == "31337002" else "")); c01 = row(base + "_leak090")
	c1s = row(base + f"_ls{S}"); c0s = row(base + f"_l090_ls{S}")
	print(f"\n  seed {sd}                       stable   err    steady  alt      hd")
	print(f"    leak .95  s=1  (control)   {fmt(c11)}")
	print(f"    leak .90  s=1  (leak arm)  {fmt(c01)}")
	print(f"    leak .95  s={S}  (arm A)     {fmt(c1s)}")
	print(f"    leak .90  s={S}  (this)      {fmt(c0s)}")
	if all((c11, c01, c1s, c0s)):
		for k, i in idx.items():
			leak_cost_s1 = c01[i] - c11[i]; leak_cost_ss = c0s[i] - c1s[i]
			inter[k].append((leak_cost_s1, leak_cost_ss))
			print(f"    leak cost on {k:6s}: at s=1 {leak_cost_s1:+.3f}   at s={S} {leak_cost_ss:+.3f}   "
			      f"{'REMOVED' if abs(leak_cost_ss) < abs(leak_cost_s1) * 0.5 else 'persists'}")
print("\n  INTERACTION (paired, per seed): the leak's cost at s=1 vs at s=s*.")
for k, v in inter.items():
	if v:
		removed = sum(abs(b) < abs(a) * 0.5 for a, b in v)
		print(f"    {k:6s}: leak cost halved-or-better at {removed}/{len(v)} seeds")
print("  If the ALTITUDE cost is removed at >=3/4 seeds while attitude keeps its gain, leak 0.90 + s* goes in the")
print("  recipe. If altitude persists, the two levers are independent and the leak remains a trade. n=4 is a DIRECTION.")
PY
log "########## 2x2 LEAK x LABEL-SCALE COMPLETE ##########"
