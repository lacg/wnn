#!/usr/bin/env bash
# BITS ROUND 2 — the width sweep re-flown AT THE WINNING ALPHABET, with seeds
# (01/09/2026, Luiz's call: option A — after the gamma=2 arm, before the
# translation A/B).
#
# WHY THIS EXISTS. Round 1 of the bits sweep (the 34 SL_A markers) is ONE seed,
# 31337002, at n=32 = 8 levels/motor — an alphabet-starved regime in which every
# width but b=36 sits OUTSIDE the gate (hd 0.92 / 1.14 / 1.51 at b36 / b32 /
# b34; the b34 dip between its neighbours is what n=1 noise looks like when it
# can be seen). The pre-registered round 2 (cull top-6, then seed 31337003 on
# the survivors) never ran, and it is now DEAD rather than late: the levels
# ladder INVERTED the b36-over-b32 ordering at 16 levels/motor, so a second seed
# at 8 levels would replicate a ranking already known not to transfer. The bits
# axis therefore has NO replication at any alphabet. This chain buys it, at the
# alphabet we actually fly.
#
# THE QUESTION. At (n*, gamma*) — the best point of the stage-C ladder — where is
# the width optimum, and is the b32-over-b36 ordering real or seed luck?
#
# DESIGN.
#   widths  b in {24,28,32,36,40}: Luiz's 4-by-4 band, centred on the round-1
#           plateau and extended one step DOWN because 16 levels/motor moved the
#           optimum narrower.
#   shape   (n*, gamma*) = the n and gamma of the LOWEST gate-distance point
#           among the ten stage-C ladder markers (b{36,32} x n{64,96,256} at
#           gamma=1, b{36,32} x n{64,96} at gamma=2). Env SL_R2_NEURONS /
#           SL_R2_GAMMA override the pick; the table is logged either way.
#   seeds   TWO from the start: 31337002 REUSES the banked b32/b36 ladder points
#           at (n*, gamma*) — run_point skips on marker presence, nothing is
#           re-flown — and 31337003 is new.
#   order   WIDTHS-MAJOR (feedback_sweeps_always_interleave): every width flies
#           seed 1 before any width flies seed 2, so a stall leaves the whole
#           curve at low resolution instead of a perfect answer for b=24.
#   cull    after seed 1, on the ladder's own gate-distance scale: keep a width
#           if it is top-SL_R2_CULL_K (3) by hd OR within SL_R2_CULL_RATIO
#           (1.25x) of the best. Culled widths keep their seed-1 marker — the
#           curve stays complete — they just never get a second seed. A width
#           with no readable seed-1 marker is dropped LOUDLY.
#   verdict the winner is the LOWEST MEAN hd across its seeds (mean, not best —
#           best-of-N inflates). The b32-vs-b36 ordering is claimed ONLY if the
#           paired same-seed comparison agrees on every seed.
#
# NO DUPLICATED RECIPE. This script does NOT re-implement the arm — a copied
# recipe block is exactly how S16 silently substituted C10. Like the gamma=2
# supervisor, it relaunches scripts/sweep_ladder_gamma.sh itself with env
# overrides, so every flag stays byte-identical to the ladder and only b / n /
# gamma / seed vary through the ladder's OWN parametrization. It needs one more
# hook in that file (SL_SKIP_PHASE1 — phase 1 would otherwise fly NEW n=32
# points at the new widths) plus a provenance label, both patched in exactly the
# way the gamma=2 supervisor patches its hook: grep-guarded, `bash -n` checked,
# and ONLY after the chain has exited — bash resumes at a byte offset, so the
# file cannot be edited while any instance is executing it.
#
# THE COMPLETION SIGNAL IS THE MARKERS, NOT THE PROCESS. controller_arm_lib.sh
# withholds a marker on watchdog kill, crash, or a clean exit with no MEMORY
# triple. Every gate here waits on markers and FAILS CLOSED: if a run needs a
# human, this chain stops, writes NO sentinel, and leaves the box idle rather
# than stacking work on a crash. The translation A/B is gated on the sentinel
# this chain writes at the very end (BITS_ROUND2_DONE.json, which also carries
# the winning shape), so a failure here parks the A/B too — deliberately.
set -u
ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1

LOG="/private/tmp/bits_round2.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
CHAIN="scripts/sweep_ladder_gamma.sh"
MARKDIR="experiments/sweepladder_markers"
SENTINEL="${MARKDIR}/BITS_ROUND2_DONE.json"
AIRFRAME="cf21_brushless"; DIST="L4C"
GATE_SEED="${SL_R2_GATE_SEED:-31337002}"          # the ladder's seed, for the wait + the pick
WIDTHS="${SL_R2_WIDTHS:-24 28 32 36 40}"
SEEDS="${SL_R2_SEEDS:-31337002 31337003}"
CULL_K="${SL_R2_CULL_K:-3}"
CULL_RATIO="${SL_R2_CULL_RATIO:-1.25}"
LADDER_WIDTHS="36 32"; LADDER_G1_N="64 96 256"; LADDER_G2_N="64 96"

log() { echo "[bits-round2] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controller_pids() { pgrep -f -- "-m wnn.control.phased_ga" 2>/dev/null || true; }
chain_pids() { pgrep -f "$CHAIN" 2>/dev/null | grep -v "^$$\$" || true; }
gtag_of() { echo "g$(echo "$1" | tr -d '.')"; }
marker_of() { echo "${MARKDIR}/SL_C_b${1}n${2}_${AIRFRAME}_${DIST}_$(gtag_of "$3")_s${4}.json"; }

# Single-instance guard: a second copy would double-launch the ladder.
if [ "$(pgrep -fc "bits_round2_chain.sh" 2>/dev/null || echo 1)" -gt 1 ]; then
	log "ABORT — another instance of this chain is already running."
	exit 1
fi

# Gate-distance of ONE marker on the ladder's scale (lifted verbatim from
# sweep_ladder_gamma.sh so round 2 ranks on exactly the scale the ladder did).
gate_distance() {
	"$VP" - "$1" <<'PY'
import json, math, re, sys
K = math.log(0.5) / math.log(0.70)
try:
	h = json.load(open(sys.argv[1])).get("headline_holdout", "")
except Exception:
	print(""); sys.exit(0)
ms = re.search(r"stable=([0-9.]+)%", h); me = re.search(r"err=([0-9.]+)", h)
if not (ms and me):
	print(""); sys.exit(0)
s = max(float(ms.group(1)) / 100.0, 1e-6); e = float(me.group(1))
print(f"{0.5556 * (e / 8.0) + 0.4444 * min(K * -math.log2(s), 20.0):.4f}")
PY
}

gamma2_missing() {
	local miss="" b n
	for n in $LADDER_G2_N; do
		for b in $LADDER_WIDTHS; do
			[ -f "$(marker_of "$b" "$n" 2.0 "$GATE_SEED")" ] || miss="${miss} $(basename "$(marker_of "$b" "$n" 2.0 "$GATE_SEED")")"
		done
	done
	echo "$miss"
}

# ---------------------------------------------------------------------------
log "########## ARMED — bits round 2 at the winning alphabet, gated on the gamma=2 arm ##########"
log "widths=[$WIDTHS] seeds=[$SEEDS] cull=top-${CULL_K}/${CULL_RATIO}x · shape (n*, gamma*) picked from the ladder markers at launch"

# ---- 1. GATE: the four gamma=2 markers, with a heartbeat so a stall is visible.
beat=0
while :; do
	miss="$(gamma2_missing)"
	[ -z "$miss" ] && break
	[ $((beat % 30)) = 0 ] && log "waiting on the gamma=2 arm — still missing:${miss}"
	beat=$((beat + 1)); sleep 60
done
log "gamma=2 arm complete — all four markers present."
# The gamma=2 arm IS an instance of $CHAIN; let it exit before touching the file.
while [ -n "$(chain_pids)" ]; do sleep 30; done
while [ -n "$(controller_pids)" ]; do sleep 30; done

# ---- 2. PICK (n*, gamma*): the lowest gate-distance among the ladder's points.
pick_table=""; best_hd=""; NSTAR=""; GSTAR=""
for g in 1.0 2.0; do
	ns="$LADDER_G1_N"; [ "$g" = "2.0" ] && ns="$LADDER_G2_N"
	for n in $ns; do
		for b in $LADDER_WIDTHS; do
			m="$(marker_of "$b" "$n" "$g" "$GATE_SEED")"
			hd="$(gate_distance "$m")"
			pick_table="${pick_table}\n    b=${b} n=${n} gamma=${g}  hd=${hd:-MISSING}"
			[ -z "$hd" ] && continue
			if [ -z "$best_hd" ] || [ "$(awk -v a="$hd" -v c="$best_hd" 'BEGIN{print (a<c)?1:0}')" = "1" ]; then
				best_hd="$hd"; NSTAR="$n"; GSTAR="$g"
			fi
		done
	done
done
log "---------- THE PICK (lowest hd over the stage-C ladder) ----------$(printf "$pick_table")"
if [ -n "${SL_R2_NEURONS:-}" ] || [ -n "${SL_R2_GAMMA:-}" ]; then
	NSTAR="${SL_R2_NEURONS:-$NSTAR}"; GSTAR="${SL_R2_GAMMA:-$GSTAR}"
	log "OVERRIDE: SL_R2_NEURONS/SL_R2_GAMMA set — flying n=${NSTAR} gamma=${GSTAR} regardless of the pick."
fi
if [ -z "$NSTAR" ] || [ -z "$GSTAR" ]; then
	log "ABORT — no ladder marker carries a gate-distance; nothing to fly at. Box left idle."
	exit 1
fi
log "PICK: n*=${NSTAR} ($((NSTAR / 4)) levels/motor) gamma*=${GSTAR} (hd ${best_hd})"
[ "$NSTAR" = "256" ] && log "⚠️ n*=256: gamma=2 was never flown there (excluded on footprint) and each run is ~4.5x the n=32 cost — override with SL_R2_NEURONS=96 if that is not wanted."

# ---- 3. PATCH the two hooks in. Safe now: nothing is executing the file.
patch_chain() {
	if grep -q "SL_SKIP_PHASE1" "$CHAIN" && grep -q "SL_SWEEP_LABEL" "$CHAIN"; then
		log "hooks already present — no patch needed."
		return 0
	fi
	cp "$CHAIN" "${CHAIN}.pre-r2-$(date -u +%Y%m%dT%H%M%SZ).bak"
	/usr/bin/python3 - "$CHAIN" <<'PY'
import sys
p = sys.argv[1]
src = open(p).read()
# Hook 1: skip phase 1 when asked by name. Phase 1 flies gamma=GAMMA at n=32 for
# every width in SL_WIDTHS; at round 2's widths those would be NEW points the
# programme never asked for. The gate verdict still runs and logs "not judged"
# for widths with no n=32 pair — harmless, and the override below decides gamma.
old1 = 'for b in $WIDTHS; do\n\trun_point "$b" 32 "$GAMMA"\ndone\n'
new1 = ('for b in $WIDTHS; do\n'
        '\t# bits round 2 (01/09/2026): phase 1 is skipped BY NAME — its n=32 points\n'
        '\t# exist for the ladder widths and would be new, unasked-for runs elsewhere.\n'
        '\t[ -n "${SL_SKIP_PHASE1:-}" ] && { log "SKIP phase 1 for b=${b} (SL_SKIP_PHASE1)"; continue; }\n'
        '\trun_point "$b" 32 "$GAMMA"\n'
        'done\n')
# Hook 2: provenance. The marker's "sweep" field names which programme flew it.
old2 = '\\"sweep\\":\\"gamma-levels\\"'
new2 = '\\"sweep\\":\\"${SL_SWEEP_LABEL:-gamma-levels}\\"'
if 'SL_SKIP_PHASE1' not in src:
	assert src.count(old1) == 1, "phase-1 loop not found exactly once"
	src = src.replace(old1, new1)
if 'SL_SWEEP_LABEL' not in src:
	assert src.count(old2) == 1, "sweep label not found exactly once"
	src = src.replace(old2, new2)
open(p, 'w').write(src)
PY
	if ! bash -n "$CHAIN"; then
		log "ABORT — patched $CHAIN FAILS bash -n. Restoring backup, launching nothing."
		cp "$(ls -t ${CHAIN}.pre-r2-*.bak | head -1)" "$CHAIN"
		return 1
	fi
	log "SL_SKIP_PHASE1 + SL_SWEEP_LABEL hooks patched in and syntax-checked."
}
patch_chain || exit 1
if ! grep -q "SL_FORCE_PHASE2_GAMMA" "$CHAIN"; then
	log "ABORT — $CHAIN has no SL_FORCE_PHASE2_GAMMA hook (the gamma=2 supervisor should have patched it). Launching nothing."
	exit 1
fi

# ---- 4. FLY one seed over a width set, via the ladder's own script. Waits for
# the instance to exit, then FAILS CLOSED unless every width banked a marker.
fly_seed() {
	local seed="$1" widths="$2" b m miss=""
	log "===== LAUNCHING seed ${seed} over widths=[${widths}] at n=${NSTAR} gamma=${GSTAR} ====="
	SL_SKIP_PHASE1=1 SL_SWEEP_LABEL="bits-round2" SL_FORCE_PHASE2_GAMMA="$GSTAR" \
		SL_WIDTHS="$widths" SL_NEURONS="$NSTAR" SL_SEED="$seed" \
		nohup bash "$CHAIN" >/dev/null 2>&1 &
	log "launched pid $! — tags SL_C_b{W}n${NSTAR}_${AIRFRAME}_${DIST}_$(gtag_of "$GSTAR")_s${seed}"
	sleep 30
	while [ -n "$(chain_pids)" ]; do sleep 60; done
	while [ -n "$(controller_pids)" ]; do sleep 30; done
	for b in $widths; do
		m="$(marker_of "$b" "$NSTAR" "$GSTAR" "$seed")"
		[ -f "$m" ] || miss="${miss} $(basename "$m")"
	done
	if [ -n "$miss" ]; then
		log "ABORT — seed ${seed} chain exited but markers MISSING:${miss}"
		log "A run needs a human (watchdog kill / crash / no MEMORY triple). No sentinel written; box left idle."
		return 1
	fi
	log "seed ${seed}: every width banked a marker."
}

set -- $SEEDS
SEED1="$1"; shift; REST_SEEDS="$*"
fly_seed "$SEED1" "$WIDTHS" || exit 1

# ---- 5. CULL on seed 1, on the ladder's scale. Loud about every decision.
log "---------- CULL after seed ${SEED1} (keep top-${CULL_K} by hd OR within ${CULL_RATIO}x of the best) ----------"
SURVIVORS="$("$VP" - "$MARKDIR" "$AIRFRAME" "$DIST" "$NSTAR" "$(gtag_of "$GSTAR")" "$SEED1" "$WIDTHS" "$CULL_K" "$CULL_RATIO" <<'PY'
import json, math, re, sys
markdir, af, dist, n, gtag, seed, widths, k, ratio = sys.argv[1:10]
K = math.log(0.5) / math.log(0.70); k = int(k); ratio = float(ratio)
rows = []
for b in widths.split():
	p = f"{markdir}/SL_C_b{b}n{n}_{af}_{dist}_{gtag}_s{seed}.json"
	try:
		h = json.load(open(p)).get("headline_holdout", "")
	except Exception:
		print(f"  b={b}: NO MARKER — dropped loudly", file=sys.stderr); continue
	ms = re.search(r"stable=([0-9.]+)%", h); me = re.search(r"err=([0-9.]+)", h); md = re.search(r"steady=([0-9.]+)", h)
	if not (ms and me):
		print(f"  b={b}: no headline held-out — dropped loudly", file=sys.stderr); continue
	s, e = float(ms.group(1)), float(me.group(1))
	hd = 0.5556 * (e / 8.0) + 0.4444 * min(K * -math.log2(max(s, 1e-4) / 100.0), 20.0)
	rows.append((hd, b, s, e, float(md.group(1)) if md else float("nan")))
rows.sort()
if not rows:
	sys.exit(0)
best = rows[0][0]
keep = []
for i, (hd, b, s, e, st) in enumerate(rows):
	why = "top-%d" % k if i < k else ("within %.2fx" % ratio if hd <= best * ratio else "CULLED")
	if why != "CULLED":
		keep.append(b)
	print("  b=%s  %5.1f%%/%5.2f/%5.2f  hd %.4f  -> %s" % (b, s, e, st, hd, why), file=sys.stderr)
print(" ".join(keep))
PY
)" 2>>"$LOG"
if [ -z "$SURVIVORS" ]; then
	log "ABORT — the cull kept nothing (no readable seed-1 markers). No sentinel written."
	exit 1
fi
log "SURVIVORS: [${SURVIVORS}]"

# ---- 6. The remaining seeds, survivors only.
for seed in $REST_SEEDS; do
	fly_seed "$seed" "$SURVIVORS" || exit 1
done

# ---- 7. VERDICT: mean hd per width across its seeds; the paired b32-vs-b36 table.
log "---------- VERDICT (winner = lowest MEAN hd across seeds; never best-of-N) ----------"
"$VP" - "$MARKDIR" "$AIRFRAME" "$DIST" "$NSTAR" "$GSTAR" "$(gtag_of "$GSTAR")" "$SEEDS" "$WIDTHS" "$SURVIVORS" "$SENTINEL" >>"$LOG" 2>&1 <<'PY'
import json, math, re, sys, datetime
markdir, af, dist, n, g, gtag, seeds, widths, survivors, sentinel = sys.argv[1:11]
K = math.log(0.5) / math.log(0.70)
seeds = seeds.split(); survivors = survivors.split()

def read(b, seed):
	p = f"{markdir}/SL_C_b{b}n{n}_{af}_{dist}_{gtag}_s{seed}.json"
	try:
		h = json.load(open(p)).get("headline_holdout", "")
	except Exception:
		return None
	ms = re.search(r"stable=([0-9.]+)%", h); me = re.search(r"err=([0-9.]+)", h)
	md = re.search(r"steady=([0-9.]+)", h); ma = re.search(r"alt=([0-9.]+)", h)
	if not (ms and me):
		return None
	s, e = float(ms.group(1)), float(me.group(1))
	return dict(stable=s, err=e, steady=float(md.group(1)) if md else float("nan"),
	            alt=float(ma.group(1)) if ma else float("nan"),
	            hd=0.5556 * (e / 8.0) + 0.4444 * min(K * -math.log2(max(s, 1e-4) / 100.0), 20.0))

means = {}
for b in widths.split():
	pts = [(sd, read(b, sd)) for sd in (seeds if b in survivors else seeds[:1])]
	pts = [(sd, r) for sd, r in pts if r]
	if not pts:
		print(f"  b={b}: no markers"); continue
	m = sum(r["hd"] for _, r in pts) / len(pts)
	means[b] = (m, len(pts))
	print("  b=%s  n=%d  mean hd %.4f   " % (b, len(pts), m)
	      + "  ".join("s%s %5.1f%%/%5.2f/%5.2f alt %.3f hd %.4f" % (sd, r["stable"], r["err"], r["steady"], r["alt"], r["hd"]) for sd, r in pts))
if "32" in means and "36" in means:
	w32 = w36 = 0
	for sd in seeds:
		a, c = read("32", sd), read("36", sd)
		if a and c:
			if a["hd"] < c["hd"]: w32 += 1
			else: w36 += 1
			print("  PAIR s%s  b32 hd %.4f  vs  b36 hd %.4f  -> %s" % (sd, a["hd"], c["hd"], "b32" if a["hd"] < c["hd"] else "b36"))
	print("  b32-over-b36 CLAIMABLE only if unanimous: b32 %d - %d b36" % (w32, w36))
if not means:
	print("  NO VERDICT — nothing readable. No sentinel."); sys.exit(1)
# Fail closed: every survivor must carry every seed.
for b in survivors:
	if means.get(b, (0, 0))[1] != len(seeds):
		print(f"  INCOMPLETE — survivor b={b} has {means.get(b,(0,0))[1]}/{len(seeds)} seeds. No sentinel."); sys.exit(1)
bstar = min(means, key=lambda b: means[b][0])
print("  WINNER: b*=%s (mean hd %.4f over %d seeds) at n*=%s gamma*=%s" % (bstar, means[bstar][0], means[bstar][1], n, g))
json.dump(dict(bits=int(bstar), neurons=int(n), gamma=float(g), mean_hd=means[bstar][0],
               widths=widths.split(), survivors=survivors, seeds=[int(s) for s in seeds],
               means={b: v[0] for b, v in means.items()},
               done=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")),
          open(sentinel, "w"), indent=1)
print("  sentinel written: " + sentinel)
PY
rc=$?
if [ "$rc" != "0" ] || [ ! -f "$SENTINEL" ]; then
	log "########## BITS ROUND 2 ENDED WITHOUT A SENTINEL (rc=${rc}) — the translation A/B stays parked ##########"
	exit 1
fi
log "########## BITS ROUND 2 COMPLETE — $(cat "$SENTINEL" | tr -d '\n' | sed 's/  */ /g' | cut -c1-200) ##########"
