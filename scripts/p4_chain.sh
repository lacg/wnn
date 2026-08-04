#!/usr/bin/env bash
# P4: is QUAD worth pursuing once it is given the E/I decode BINARY always had?
#
# THE CONFOUND THIS EXISTS TO BREAK. The aligned dfa1l table has BINARY beating
# QUAD 10 paired cells out of 10, several by 40-60pp stable (docs/dfa1l_aligned_study.md
# §4). Read naively that says "the 4-state nudging cell is worse than the 1-bit cell".
# It cannot say that yet, because until 03/08/2026 the decode topology was WELDED to
# the memory mode: BINARY implied antagonist E/I halves, everything else implied
# cumulative. So every one of those 10 pairs moved TWO variables at once.
#
# The weld was never a design choice — it was a necessity for BINARY alone. A 1-bit
# cell reads 0 untrained, so a single thermometer bank can only push up from the
# floor; the E/I split is what buys it a neutral. QUAD does not NEED the split, but
# it has its own asymmetry: QUAD_WEIGHTS[EMPTY]=0.75, so an untrained cumulative bank
# sits at 0.75 and can travel 0.75 down but only 0.25 up — 3:1 around hover. Under
# the antagonist decode an untrained QUAD bank cancels to exactly 0.5.
#
# THE 3-CELL DESIGN (BINARY+cumulative is refused by the CLI — its untrained bank
# would decode to the floor — so the 2x2 has a structurally empty corner):
#
#                 antagonist              cumulative
#   BINARY        P2, MEASURED n=3        n/a — refused
#   QUAD          P4a  <- new             P4c  <- new
#
#   P4a - P4c  = decode topology, cell alphabet held at QUAD  (is E/I the lever?)
#   P4a - P2   = cell alphabet, decode held at antagonist     (is QUAD the lever?)
#
# Without P4c a QUAD+E/I win is real but unattributable, which is the exact mistake
# the dfa1l table made. Hence both arms, not just the one.
#
# ANCHOR. P2 (pidmix_tilt nf=17 + the output-neuron sweep) is the strongest BINARY
# reference we hold at these seeds: 100.0+-0.0 stable / 1.96+-0.24 err / 1.39+-0.24
# steady, n=3 on 31337002/3/4. Every flag below is P2's verbatim EXCEPT --memory-mode
# and --output-decode, so the contrast is genuinely one variable (two for P4a vs P2).
#
# NOT pidmix_pwm_tilt. "pwm" here means the PWM DECODE RESOLUTION axis
# (--grid-output-neurons 64 96 128 = 16/24/32 levels per motor), which is what P2
# swept and what P2's own tag "_on_" abbreviates. It does NOT mean --obs-pwm: the
# arms carrying that feature COLLAPSED at these very seeds (A6 pidmix_pwm_tilt
# 16.3+-26.6 stable / 17.37 err, per-seed 2.0/47.0/0.0; A5 pidmix_pwm 11.0+-19.1).
# Anchoring P4 there would measure that collapse, not the substrate.
#
# TRIGGER: 9 P3 cell markers AND no live p3_chain. P3 publishes no ALL_DONE file, and
# p3_chain.sh CANNOT be edited to add one while it is running — bash reads a script
# incrementally by byte offset, so editing it live corrupts the running process. The
# glob P3_*_L3D_s*.json is cell-specific, so unlike the dfa1l marker dir there are no
# auxiliary files to miscount. If a P3 cell crashes it writes no marker and this parks
# forever by design: better a stalled P4 than a P4 running on top of an unfinished P3.
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
# shellcheck source=controller_arm_lib.sh
. "$ROOT/scripts/controller_arm_lib.sh"

LOG="/private/tmp/p4_chain.log"
P3_MARKDIR="experiments/p3_markers"
P3_WANT="${P3_WANT:-9}"
STUDY="scripts/run_dfa_1layer_study.sh"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
OUTDIR="logs/controller/p4"
MARKDIR="experiments/p4_markers"
SEEDS="${P4_SEEDS:-31337002 31337003 31337004}"
REPORT_SEEDS="99990101 99990102 99990103 99990104 99990105"
P4_CONFIRMED="${P4_CONFIRMED:-0}"

# nf=17 pidmix_tilt — identical flags to p2_chain.sh and to
# run_l3d_feature_probe.sh:FEAT_PIDMIX_TILT.
FEAT_PIDMIX_TILT="--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw \
--obs-yaw-err --obs-yaw-err-i --obs-tilt-p --obs-tilt-i"

log() { echo "[p4] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controllers() { ps -axo pid,command 2>/dev/null | grep "wnn.control.phased_ga" \
	| grep -v "/usr/bin/time" | grep -v grep | grep -c python; }
p3_markers() { ls -1 "$P3_MARKDIR"/P3_*_L3D_s*.json 2>/dev/null | wc -l | tr -d ' '; }

if [ "$P4_CONFIRMED" != "1" ]; then
	log "REFUSING TO ARM — set P4_CONFIRMED=1 once the 3-cell design is signed off"
	exit 2
fi

mkdir -p "$OUTDIR" "$MARKDIR"
log "########## ARMED — waiting for ${P3_WANT} P3 markers + p3_chain exit ##########"

# ---- 1. wait for P3 to finish COMPLETELY ------------------------------------
while :; do
	got="$(p3_markers)"
	if [ "$got" -ge "$P3_WANT" ] && ! pgrep -f "scripts/p3_chain.sh" >/dev/null 2>&1; then
		log "P3 complete: ${got}/${P3_WANT} markers, chain exited"
		break
	fi
	sleep 300
done

# ---- 2. confirm the box is free (never kill; abort instead) -----------------
sleep 60
if [ "$(controllers)" -gt 0 ] || pgrep -f "$STUDY" >/dev/null 2>&1; then
	log "ABORT: P3 says done but controllers=$(controllers) drivers=$(pgrep -fc "$STUDY" 2>/dev/null || echo 0) \
— refusing to run alongside. Needs a human."
	exit 3
fi
log "box clear: controllers=0 drivers=0"

# ---- 3. the two QUAD arms ---------------------------------------------------
# Args: <armsuffix> <output-decode> <seed>. Everything else is P2 verbatim.
run_quad_arm() {
	local name="$1" decode="$2" seed="$3"
	run_controller_arm "P4${name}_L2D_s${seed}" "$MARKDIR" "$OUTDIR" "$VP" log \
		"\"arm\":\"P4${name}\",\"axis\":\"output_decode\",\"output_decode\":\"${decode}\",\"disturbance\":\"L2D\",\"features\":\"pidmix_tilt\",\"substrate\":\"1layer\",\"mode\":\"QUAD_WEIGHTED\",\"seed\":${seed}" \
		-- \
		--levels 16 --skip-stages bits,connections --lamarckian \
		--max-cells 180000 \
		--neurons-gens 60 --neurons-patience 3 \
		--memory-gens 120 --memory-patience 2 \
		--pop 50 --num-eval-folds 5 --check-interval 2 --magnitude-aware-patience \
		--eval-episodes 100 --memory-eval-episodes 200 \
		--steps 2000 --tilt 5.0 \
		--fit-weight-err-sq 0.4 --fit-weight-stable 0.3 \
		--fit-weight-jerk 0.2 --fit-weight-mono 0.1 \
		--report-episodes 100 --holdout-pop-sample 8 \
		--grid-bits 24 30 \
		--grid-output-neurons 64 96 128 \
		--max-output-neurons 128 \
		--grid-state-neurons 0 --max-state-neurons 0 \
		--runs 1 --teacher lqr \
		--memory-mode QUAD_WEIGHTED --output-decode "$decode" \
		--disturbance L2D \
		$FEAT_PIDMIX_TILT \
		--report-seeds $REPORT_SEEDS \
		--base-seed "$seed"
}

# INTERLEAVED BY SEED (the sweep rule): round r fires BOTH arms at seed r. Losing
# the box partway then leaves a complete low-n version of the whole contrast rather
# than three seeds of one arm and nothing of the other — and the contrast is the
# entire point, so a truncated P4a alone would answer nothing.
#
# 'a' (antagonist) runs FIRST each round: it is the treatment. If the box is lost
# mid-round the survivor is the arm that carries the question.
for seed in $SEEDS; do
	log "===== ROUND seed=$seed ====="
	run_quad_arm a antagonist "$seed"   # THE question: does E/I rescue QUAD?
	run_quad_arm c cumulative "$seed"   # control: QUAD's historical decode
done

got=$(ls -1 "$MARKDIR"/P4[ac]_L2D_s*.json 2>/dev/null | wc -l | tr -d ' ')
want=$(( $(echo $SEEDS | wc -w) * 2 ))
if [ "$got" = "$want" ]; then
	: > "${MARKDIR}/P4_ALL_DONE.marker"
	log "########## P4 CHAIN DONE — ${got}/${want} markers, ALL_DONE published ##########"
else
	log "########## P4 CHAIN INCOMPLETE — ${got}/${want} markers, NO ALL_DONE ##########"
	exit 4
fi
