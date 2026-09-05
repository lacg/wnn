#!/usr/bin/env bash
# LEAK REVISIT AT THE NEW ALPHABET (01/09/2026, Luiz's call) — re-test a
# REFUTED lever under the conditions that have changed since it was refuted.
#
# WHAT WAS REFUTED, AND WHY THAT MAY NOT BIND ANY MORE. L3 tested
# --delta-leak 0.8 against the 0.95 default and the whole hold-floor family
# (L1/L1b/L2/L3) went out 4-for-4. But every one of those runs flew the OLD
# 8-level alphabet, where the smallest sustainable offset was
#     quantum/(1-leak) = (delta_max/(levels/2))/(1-leak) = 0.025/0.05 = 0.5 PWM
# — HALF of full authority. Any effect of leak was capped by a quantum 4x
# coarser than today's: the alphabet floor plausibly drowned the lever, and the
# leak x alphabet INTERACTION was never measured. The levels ladder has since
# shown the alphabet was exactly the binding constraint (b=32: hd 1.144 -> 0.224
# from one doubling). At 16 levels/motor the same arithmetic gives
#     leak 0.95 -> 0.0125/0.05 = 0.25 PWM sustainable
#     leak 0.90 -> 0.125 PWM
#     leak 0.80 -> 0.0625 PWM
# so leak is now the larger factor in the sustainable-offset product. If steady
# and alt are set by that product, lowering leak should move them; if L3's
# refutation was fundamental (the accumulator forgetting trim faster than the
# student can re-apply it), it will lose again and the refutation UPGRADES from
# "at the old alphabet" to "at both alphabets".
#
# DESIGN — a SCREEN, not a verdict: 2 leak values x 1 seed against a BANKED
# control, the same shape/gate/seed pattern the gamma phase-1 A/B used. The
# control is SL_C_b32n64_..._g10_s31337002 (hd 0.2240, the ladder's best), which
# ran at the 0.95 default — verified: sweep_ladder_gamma.sh passes no
# --delta-leak and phased_ga.py:2255 defaults it. Everything here is
# byte-identical to that run except --delta-leak; n=1 per value means a WIN sends
# leak into a proper multi-seed ladder, it does not bank a claim (seed spread
# 90.8-98.0% dwarfs most effects).
#
# GATING. Runs after the translation A/B banks all 10 of its markers. Pure
# waits, marker-gated, FAIL CLOSED if the A/B chain died short — same posture as
# the rest of the queue (a marker is withheld on watchdog kill / crash / no
# MEMORY triple, so its absence means a human is needed, and stacking a new
# study on top of that would bury it).
set -u
ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
. "$ROOT/scripts/controller_arm_lib.sh"

LOG="/private/tmp/leak_revisit.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
OUTDIR="logs/controller/leak_revisit"
MARKDIR="experiments/leakrevisit_markers"
TABMARK="experiments/translationab_markers"
CTRLMARK="experiments/sweepladder_markers/SL_C_b32n64_cf21_brushless_L4C_g10_s31337002.json"
AIRFRAME="cf21_brushless"; DIST="L4C"
BITS=32; NEURONS=64; SEED="${LKR_SEED:-31337002}"
LEAKS="${LKR_LEAKS:-0.90 0.80}"
TAB_SEEDS="31337002 31337003 31337004 31337005 31337006"
# 04/09/2026 20:30 EDT (Luiz): the OFF arms were DROPPED after one reference point
# (OFF s31337002). Knowing the regimen's cost changes no lever — the axis is
# mandatory — so the remaining budget goes to the ON seeds, which double as the
# replication of the b32 n256 record. The gate therefore waits on the ON markers
# only (TAB_ARMS override), and the chain's keeper may be the handoff script.
TAB_ARMS="${TAB_ARMS:-on}"
REPORT_SEEDS="99990101 99990102 99990103 99990104 99990105"

mkdir -p "$MARKDIR" "$OUTDIR"

FEAT="--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i \
--obs-collective-cmd --obs-alt-err --obs-vz"
WEIGHTS="--fit-weight-err-sq 0.3125 --fit-weight-stable 0.25 --fit-weight-steady 0.4375"
AGG_GATE="--fit-aggregation zscore --zrank-clamp 3.0 --gate-stable 0.70 --gate-err 8.0"

log() { echo "[leak-revisit] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controller_pids() { pgrep -f -- "-m wnn.control.phased_ga" 2>/dev/null || true; }
tab_pids() { pgrep -f "scripts/translation_ab_(chain|on_handoff)\.sh" 2>/dev/null || true; }
# 04/09/2026 (Luiz): the CRN re-fly of b24 s31337002 runs BETWEEN the A/B and this
# study (scripts/crn_refly_chain.sh). Gating on its marker too keeps the two
# waiters from racing for the box the moment the A/B's last marker lands.
# (b32 dropped 04/09 14:30: the A/B's TAB_on s31337002 IS that re-fly — same recipe, CRN.)
REFLY_MARKS="experiments/sweepladder_markers/SL_C_b24n256_${AIRFRAME}_${DIST}_g10_s31337002_crn.json"
refly_pids() { pgrep -f "scripts/crn_refly_chain.sh" 2>/dev/null || true; }

tab_missing() {
	local miss="" arm seed
	for seed in $TAB_SEEDS; do
		for arm in $TAB_ARMS; do
			# Any shape (01/09/2026): the A/B now flies at the bits round-2 winner,
			# so its b/n are not known here. The seed is what sequences us.
			ls "${TABMARK}"/TAB_${arm}_b*n*_${AIRFRAME}_${DIST}_s${seed}.json >/dev/null 2>&1 \
				|| miss="${miss} TAB_${arm}_s${seed}"
		done
	done
	for m in $REFLY_MARKS; do [ -f "$m" ] || miss="${miss} $(basename "$m" .json)"; done
	echo "$miss"
}

# run_point <leak>  (tag encodes leak with the dot stripped: 0.90 -> l090)
run_point() {
	local leak="$1"
	local ltag; ltag="l$(echo "$leak" | tr -d '.')"
	local tag="LKR_${ltag}_b${BITS}n${NEURONS}_${AIRFRAME}_${DIST}_g10_s${SEED}"
	[ -f "${MARKDIR}/${tag}.json" ] && { log "SKIP $tag (marker exists)"; return 0; }
	while [ -n "$(controller_pids)" ]; do sleep 20; done
	mkdir -p "$OUTDIR/ckpt/$tag"
	log "===== START $tag (delta_leak=${leak}; sustainable offset $(awk -v l="$leak" 'BEGIN{printf "%.4f", 0.0125/(1-l)}') PWM vs control 0.25) ====="
	# shellcheck disable=SC2086
	run_controller_arm "$tag" "$MARKDIR" "$OUTDIR" "$VP" log \
		"\"study\":\"leak-revisit\",\"delta_leak\":${leak},\"bits\":${BITS},\"neurons\":${NEURONS},\"levels_per_motor\":$((NEURONS / 4)),\"delta_gamma\":1.0,\"seed\":${SEED}" \
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
		--delta-gamma 1.0 \
		--delta-leak "$leak" \
		--grid-bits "$BITS" --grid-output-neurons "$NEURONS" --max-output-neurons "$NEURONS" \
		--report-episodes 100 --holdout-pop-sample 8 \
		--runs 1 --memory-mode BINARY \
		--airframe "$AIRFRAME" --disturbance "$DIST" --teacher mpcof \
		$FEAT \
		--translation --reward-lambda-alt 0 \
		--grid-state-neurons 0 --max-state-neurons 0 \
		--report-seeds $REPORT_SEEDS \
		--base-seed "$SEED"
	log "$tag finished rc=$?"
}

log "########## ARMED — leak revisit at 16 levels/motor, gated on the translation A/B ##########"
log "leaks=[$LEAKS] vs banked control leak=0.95 (SL_C_b32n64, hd 0.2240) · shape b=${BITS} n=${NEURONS} · seed ${SEED}"

# ---- GATE on the A/B's 10 markers; fail closed if its chain died short.
beat=0
while :; do
	miss="$(tab_missing)"
	[ -z "$miss" ] && break
	if [ -z "$(tab_pids)" ] && [ -z "$(refly_pids)" ]; then
		log "ABORT — translation A/B + CRN re-fly chains gone with markers MISSING:${miss}"
		log "A run needs a human. Launching nothing; box left idle deliberately."
		exit 1
	fi
	[ $((beat % 30)) = 0 ] && log "waiting on the translation A/B — still missing:${miss}"
	beat=$((beat + 1)); sleep 60
done
log "translation A/B complete — all ON-arm markers present (arms=[$TAB_ARMS])."
while [ -n "$(controller_pids)" ]; do sleep 30; done

for leak in $LEAKS; do
	run_point "$leak"
done

# ---- VERDICT: each leak paired against the banked 0.95 control on gate-distance.
log "---------- VERDICT (paired vs banked control, hd = stable+err ONLY; report all four columns) ----------"
"$VP" - "$MARKDIR" "$CTRLMARK" "$SEED" "$LEAKS" >> "$LOG" <<'PY'
import json, math, os, re, sys
markdir, ctrlmark, seed, leaks = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
K = math.log(0.5) / math.log(0.70)


def read(path):
	if not os.path.exists(path):
		return None
	h = json.load(open(path)).get("headline_holdout", "")
	m = {k: re.search(p, h) for k, p in
	     (("s", r"stable=([0-9.]+)%"), ("e", r"err=([0-9.]+)"),
	      ("d", r"steady=([0-9.]+)"), ("a", r"alt=([0-9.]+)m"))}
	if not (m["s"] and m["e"]):
		return None
	s, e = float(m["s"].group(1)), float(m["e"].group(1))
	return dict(s=s, e=e, d=float(m["d"].group(1)) if m["d"] else float("nan"),
	            a=float(m["a"].group(1)) if m["a"] else float("nan"),
	            hd=0.5556 * (e / 8.0) + 0.4444 * min(K * -math.log2(max(s, 1e-4) / 100.0), 20.0))


ctrl = read(ctrlmark)
if not ctrl:
	print("  control marker unreadable — no verdict.")
	sys.exit(0)
print("  control leak=0.95  %5.1f%%/%5.2f/%5.2f alt %.3fm  hd %.4f"
      % (ctrl["s"], ctrl["e"], ctrl["d"], ctrl["a"], ctrl["hd"]))
for leak in leaks.split():
	ltag = "l" + leak.replace(".", "")
	r = read(f"{markdir}/LKR_{ltag}_b32n64_cf21_brushless_L4C_g10_s{seed}.json")
	if not r:
		print(f"  leak={leak}  MISSING marker — not judged.")
		continue
	d = r["hd"] - ctrl["hd"]
	print("  leak=%s     %5.1f%%/%5.2f/%5.2f alt %.3fm  hd %.4f  delta %+.4f -> %s"
	      % (leak, r["s"], r["e"], r["d"], r["a"], r["hd"], d,
	         "BETTER — promote to a multi-seed ladder" if d < 0 else "worse"))
print("  n=1 per value: a win here EARNS a ladder, it does not bank a claim.")
PY
log "########## LEAK REVISIT COMPLETE ##########"
