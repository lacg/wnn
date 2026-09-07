#!/usr/bin/env bash
# MUTATION-STEP A/B — is one tap per neuron a better step than "rewire everything"?
# (07/09/2026, Luiz's queue item: "mutation-step A/B, rate 1/32 (one tap per neuron)
# vs the current 0.1 per tap x 32 taps".)
#
# THE QUESTION. `FiniteStateGenome.mutate` resamples EACH connection tap with
# probability `mutation_rate`, canonically 0.1. At b=24 that is 24 independent
# draws per neuron, so P(a neuron is left alone) = 0.9^24 = 8.0% and the expected
# number of taps moved per neuron is 2.4. In other words a "child" rewires
# essentially every neuron it owns: the GA is closer to random restart than to
# local search, which is a candidate explanation for why CONNECTIONS improves the
# elite so rarely (the b24 seeds spent 5 gens moving the incumbent 0-2 times).
# ARM B sets the rate to 1/bits, i.e. ONE expected tap per neuron: P(untouched)
# = (1-1/24)^24 = 35.8%, so most neurons are inherited intact and the child is a
# neighbour of its parent rather than a stranger.
#
# DESIGN — paired, same-seed, same-recipe, same-era, ONE factor.
#   A side  = ALREADY BANKED, no re-fly. The four b24 n256 CRN ladder runs
#             (seeds 31337002_crn, 31337003, 31337004, 31337005) ARE arm A: same
#             ladder invocation, same CRN scorer, canonical rate 0.1.
#   B side  = the same four seeds, the same ladder invocation, plus exactly one
#             flag: --conn-mutation-rate $(1/bits). Tagged _mut1tap so marker,
#             .out and ckpt never collide with their A twin.
#   Read    = paired per-seed on the MEMORY same-rule row (the fixed stage; the
#             headline is a stage-select draw and must not be the comparator).
#             4 seeds -> paired majority, the standard here. n=4 is a DIRECTION.
#
# WHY b24 n256. It is the shape the programme decided to publish: the CRN bits
# curve closed 07/09 as a dead tie on attitude (paired b24-vs-b32 2-2), so the
# H743 flash constraint picks the width, and b24 is the only n=256 width that
# fits internal flash. Its four CRN seeds are also the cheapest banked control we
# have (~4.6 h/run vs b32's 5.8 h).
#
# WHY THE RATE IS 1/24 AND NOT LITERALLY 1/32. The hypothesis is "one tap per
# neuron", and 1/32 was that number at b=32, the shape the queue was written
# against. At b=24 one-tap-per-neuron is 1/24 = 0.0416667. Keeping the literal
# 0.03125 here would test 0.75 taps per neuron and quietly change the hypothesis.
#
# THE FLAG IS SCOPED TO CONNECTIONS. --conn-mutation-rate lands only on the arch
# GA (phased_ga._build_ga_config's arch call site); MEMORY keeps the canonical
# 0.1. Both stages build their config through the same helper, and genome.mutate
# means a different thing per dimension, so a single un-scoped knob would have
# made this a two-factor experiment.
#
# GATING. Pure wait for any controller; never preempts. Every point is
# marker-gated and idempotent (the ladder SKIPs an existing marker). FAILS
# CLOSED: a missing marker after a run means a human is needed — the chain stops
# and leaves the box idle rather than stacking work on a crash.
set -u
ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1

LOG="/private/tmp/mutstep_ab.log"
LADDER="scripts/sweep_ladder_gamma.sh"
MARK="experiments/sweepladder_markers"
AIRFRAME="cf21_brushless"; DIST="L4C"
BITS=24
NEURONS=256
SEEDS="${MUT_SEEDS:-31337002 31337003 31337004 31337005}"
SUFFIX="_mut1tap"
# One expected tap per neuron at this width. bc keeps it exact-ish and visible in
# the log; the marker records the value so the arm names its own flag.
RATE="${MUT_RATE:-$(python3 -c "print(1/${BITS})")}"

log() { echo "[mutstep-ab] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controller_pids() { pgrep -f -- "-m wnn.control.phased_ga" 2>/dev/null || true; }
ladder_pids() { pgrep -f "scripts/sweep_ladder_gamma.sh" 2>/dev/null || true; }
wait_box_clear() {
	local beat=0
	while [ -n "$(controller_pids)" ] || [ -n "$(ladder_pids)" ]; do
		[ $((beat % 30)) = 0 ] && log "waiting — box busy (controller/ladder still running)"
		beat=$((beat + 1)); sleep 60
	done
}

# The A arm must already exist, or there is nothing to pair against and the B
# runs would be an unpaired cohort — the exact mistake IDSXD paid for.
a_tag() { # $1 = seed
	if [ "$1" = "31337002" ]; then echo "SL_C_b${BITS}n${NEURONS}_${AIRFRAME}_${DIST}_g10_s31337002_crn"
	else echo "SL_C_b${BITS}n${NEURONS}_${AIRFRAME}_${DIST}_g10_s$1"; fi
}

log "########## ARMED — mutation-step A/B at b${BITS} n${NEURONS}, rate ${RATE} vs canonical 0.1 ##########"
log "seeds=[${SEEDS}]  A=banked (rate 0.1)  B=${SUFFIX} (--conn-mutation-rate ${RATE})"
MISSING=""
for s in $SEEDS; do
	[ -f "${MARK}/$(a_tag "$s").json" ] || MISSING="${MISSING} $(a_tag "$s")"
done
[ -z "$MISSING" ] || { log "ABORT — control (arm A) markers missing:${MISSING}. Nothing to pair against."; exit 1; }
log "all 4 arm-A controls present — paired comparison is possible."

for seed in $SEEDS; do
	TB="SL_C_b${BITS}n${NEURONS}_${AIRFRAME}_${DIST}_g10_s${seed}${SUFFIX}"
	if [ -f "${MARK}/${TB}.json" ]; then
		log "SKIP — ${TB} already banked."; continue
	fi
	wait_box_clear
	log "===== START ${TB} (arm B, --conn-mutation-rate ${RATE}) ====="
	SL_SKIP_PHASE1=1 SL_SWEEP_LABEL="mutstep-ab" SL_FORCE_PHASE2_GAMMA="1.0" \
		SL_WIDTHS="$BITS" SL_NEURONS="$NEURONS" SL_SEED="$seed" \
		SL_TAG_SUFFIX="$SUFFIX" \
		SL_EXTRA_ARGS="--conn-mutation-rate ${RATE}" \
		SL_EXTRA_MARKER_JSON=",\"arm_b\":\"mut1tap\",\"conn_mutation_rate\":${RATE},\"control_tag\":\"$(a_tag "$seed")\"" \
		bash "$LADDER"
	log "ladder exited rc=$? for seed ${seed}"
	while [ -n "$(controller_pids)" ]; do sleep 30; done
	[ -f "${MARK}/${TB}.json" ] || { log "ABORT — marker ${TB}.json MISSING. A run needs a human. Box left idle."; exit 1; }
	log "banked: ${TB}"
done

# ---- VERDICT: paired per-seed on the MEMORY same-rule row, then the tally.
python3 scripts/gate_distance_leaderboard.py > docs/controller_gate_distance_leaderboard.md
log "---------- VERDICT: paired same-seed, MEMORY same-rule (lower = better) ----------"
MUT_BITS="$BITS" MUT_NEURONS="$NEURONS" MUT_SEEDS="$SEEDS" MUT_SUFFIX="$SUFFIX" \
/usr/bin/python3 - >> "$LOG" <<'PY'
import glob, json, math, os, re
K = math.log(0.5) / math.log(0.70)
def gd(s, e): return 0.5556 * (e / 8.0) + 0.4444 * min(K * -math.log2(s / 100), 20.0)
B, N = os.environ["MUT_BITS"], os.environ["MUT_NEURONS"]
SUF = os.environ["MUT_SUFFIX"]
def row(tag):
	p = f"experiments/sweepladder_markers/{tag}.json"
	if not os.path.exists(p): return None
	m = json.load(open(p)).get("held_memory_multiseed", "")
	f = lambda pat: float(re.search(pat, m).group(1))
	try: return (f(r"stable=([0-9.]+)"), f(r"err=([0-9.]+)"), f(r"steady=([0-9.]+)"),
	             f(r"alt=([0-9.]+)"), gd(f(r"stable=([0-9.]+)"), f(r"err=([0-9.]+)")))
	except Exception: return None
wa = wb = 0
print("  seed       arm A (rate 0.1)                       arm B (one tap/neuron)                 winner")
for s in os.environ["MUT_SEEDS"].split():
	at = f"SL_C_b{B}n{N}_cf21_brushless_L4C_g10_s{s}" + ("_crn" if s == "31337002" else "")
	a, b = row(at), row(f"SL_C_b{B}n{N}_cf21_brushless_L4C_g10_s{s}{SUF}")
	# An arm with no marker is NOT a loss — say so instead of scoring it.
	if not a or not b:
		print(f"  {s}   {'A missing' if not a else 'B not flown yet':<38}   —"); continue
	w = "A" if a[4] < b[4] else "B"
	wa += a[4] < b[4]; wb += b[4] < a[4]
	fmt = lambda r: f"{r[0]:5.1f}% {r[1]:.2f} {r[2]:.2f} {r[3]:.3f}m hd {r[4]:.4f}"
	print(f"  {s}   {fmt(a):<38}   {fmt(b):<38}   {w}")
print(f"\n  PAIRED TALLY  A {wa} - {wb} B   (columns: stable / err / steady / alt / same-rule)")
print("  Paired majority is the standard. n<=4 is a DIRECTION, not a claim, and the")
print("  seed band on this shape (0.1029-0.1615) is WIDER than any step-size effect")
print("  is likely to be — read the tally, not the means.")
PY
log "########## MUTATION-STEP A/B COMPLETE ##########"
