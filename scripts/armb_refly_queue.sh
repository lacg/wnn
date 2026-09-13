#!/usr/bin/env bash
# ARM B RE-FLY QUEUE (13/09/2026). Arm B's first flight (`_bd` s31337002) was DEAD ON
# ARRIVAL — 0.0%/57° at every stage — because --obs-pwm was broken at two sites: the
# threshold fitter's pwm ladder was degenerate (constant samples ⇒ every threshold =
# anchor) and the replay trainers never restored the accumulator (train saw hover,
# deploy saw the real stream). Both fixed in e554661b (controller ABI 29). The 4-arm
# smoke (logs/controller/armb_smoke/) showed --obs-pwm alone kills and
# --dagger-label-delta alone flies, so the hypothesis is UNTESTED, not refuted.
#
# This queue waits behind the window-k chain (12/12 `_win[234]` markers), archives the
# void `_bd` marker/out/winner to *_void_abi28 (README inside), then runs the unchanged
# arm_b_delta_label_chain.sh — which re-flies all 4 seeds on the fixed wheel. Marker-
# gated, idempotent, never preempts, fails closed. The HOLD sentinel is honoured by the
# ladder (wait_while_held), so an un-lifted HOLD from the deploy smoke stops it too.
cd /Users/lacg/wnn || exit 1
LOG="/private/tmp/armb_refly_queue.log"
MARK="experiments/sweepladder_markers"
OUTDIR="logs/controller/sweep_ladder"
VOID="experiments/sweepladder_markers_void_abi28"
log() { echo "[armb-refly] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controller_pids() { pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" 2>/dev/null || true; }
wink_done() { ls "$MARK" 2>/dev/null | grep -cE "_win[234]\.json$"; }
abi_ok() { PYTHONPATH=src/wnn /Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python -c "from wnn.control import _accel; import sys; sys.exit(0 if getattr(_accel,'_abi',0) >= 29 else 1)" 2>/dev/null; }

log "########## ARMED — arm B re-fly on the obs_pwm-fixed wheel, behind window-k ($(wink_done)/12) ##########"
b=0
while [ "$(wink_done)" -lt 12 ] || pgrep -f "scripts/queue_after_ab_chai[n]\.sh" >/dev/null; do
	sleep 120; b=$((b + 1)); [ $((b % 30)) = 0 ] && log "waiting — window-k $(wink_done)/12"
done
while [ -n "$(controller_pids)" ]; do sleep 30; done
sleep 120
abi_ok || { log "ABORT — installed controller wheel is not ABI >= 29 (obs_pwm fix not deployed). Nothing flies."; exit 1; }

# Archive the void first flight so the chain's marker gate re-flies seed 2.
T="SL_C_b24n256_cf21_brushless_L4C_g10_s31337002_bd"
marker_abi() { /Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python -c "import json,sys; print(json.load(open(sys.argv[1])).get('provenance',{}).get('abi',0))" "$1" 2>/dev/null || echo 0; }
if [ -f "$MARK/$T.json" ] && [ "$(marker_abi "$MARK/$T.json")" -lt 29 ]; then
	mkdir -p "$VOID"
	mv "$MARK/$T.json" "$VOID/"
	[ -f "$OUTDIR/$T.out" ] && mv "$OUTDIR/$T.out" "$VOID/"
	[ -f "$OUTDIR/${T}_winner.yaml.gz" ] && mv "$OUTDIR/${T}_winner.yaml.gz" "$VOID/"
	cat > "$VOID/README.md" <<'R'
# VOID — arm B first flight on the obs_pwm-broken wheel (ABI 28)

`SL_C_b24n256_cf21_brushless_L4C_g10_s31337002_bd` flew 13/09/2026 00:40-03:01 EDT and
banked 0.0% / 57.15° / 57.15° / 0.238 m at every stage: the student output near-constant
pwm (effort 0.92, mono_viol 10). Cause, proven by a 4-arm smoke (logs/controller/armb_smoke):
`--obs-pwm` alone kills (0.0%/70°), `--dagger-label-delta` alone flies. Two defects on the
pwm feature — a degenerate thermometer ladder (the fitter's untrained feature controller
never left the anchor ⇒ all 8 thresholds = 0.5) and a replay-frozen accumulator (the
documented `OBS_PWM_FIXED=false` gap). Fixed in e554661b, controller ABI 29. This row is
not a result and is never paired; the re-fly under the same tag supersedes it.
R
	log "archived void $T to $VOID/"
fi
log "launching arm_b_delta_label_chain.sh (4 seeds, ~20 h)"
bash scripts/arm_b_delta_label_chain.sh
log "arm B chain exited rc=$? — markers: $(ls "$MARK" | grep -cE "_bd\.json$")/4"
