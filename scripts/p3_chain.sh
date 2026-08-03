#!/usr/bin/env bash
# P3: does RECURRENT STATE buy anything at L3D?
#
# THE GAP THIS FILLS. Every L3D arm ever run was STATELESS — not by choice, but
# because run_l3d_feature_probe.sh:54 pins `--grid-state-neurons 0 --max-state-neurons 0`
# in its COMMON block, so A3/A4 inherited sn=0 along with everything else. A DFA has
# never been trained at L3D. That is the open question, and it is the whole of P3.
#
# WHAT COUNTS AS A RESULT. At L3D everything collapses: WNN 1.1-4.2%, PID 3.8%, and
# even LQR/MPC only reach 48-71%. So the comparison is NOT against 100% — it is
# stateful-WNN vs the stateless 1.1-4.2% floor on MATCHED SEEDS. A jump to 20% would
# be a large positive result here; anything near the floor is a null.
#
# WHY A 2x2 AND NOT ONE CELL. input_window_k and recurrent state are two mechanisms
# for the SAME job — carrying history past the current frame. Raise both at once and
# a positive result is unattributable. So K is the control condition, not a second
# treatment:
#
#            K=4              K=8
#   sn=0     (A4, measured)   P3_k8
#   sn=8     P3_state         P3_state_k8
#
# Note K cannot brute-force this problem: the L3D freeze is 40 steps and K=4 sees 4.
# Spanning it would need K~40, i.e. a 40*15*8 = 4800-bit input pool with the address
# space 2^(prefix+suffix) UNCHANGED — coverage would collapse long before K got there.
# State, not window, is the plausible mechanism; the K arm exists so that a null on
# state cannot be waved away as "we never gave it enough history".
#
# sn=8 matches the DFA cells in the dfa1l study (marker: sn=8 sb=30 ob=30), so the
# stateful corner is the same substrate we already hold L2D numbers for.
# Features fixed to nf=15 pidmix — the A2 probe winner (1.87 deg pooled at L2D).
#
# TRIGGER: P2_ALL_DONE.marker, published by p2_chain.sh only when every P2 seed has
# a marker. A partial P2 deliberately withholds it, so P3 stays parked rather than
# starting on top of an unfinished predecessor and losing both.
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
# shellcheck source=controller_arm_lib.sh
. "$ROOT/scripts/controller_arm_lib.sh"

LOG="/private/tmp/p3_chain.log"
DONE_MARKER="experiments/p2_markers/P2_ALL_DONE.marker"
STUDY="scripts/run_dfa_1layer_study.sh"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
OUTDIR="logs/controller/p3"
MARKDIR="experiments/p3_markers"
SEEDS="${P3_SEEDS:-31337002 31337003 31337004}"
REPORT_SEEDS="99990101 99990102 99990103 99990104 99990105"
P3_CONFIRMED="${P3_CONFIRMED:-0}"

# nf=15 pidmix — identical flags to run_l3d_feature_probe.sh:58 (FEAT_PIDMIX).
FEAT_PIDMIX="--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i"

log() { echo "[p3] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controllers() { ps -axo pid,command 2>/dev/null | grep "wnn.control.phased_ga" \
	| grep -v "/usr/bin/time" | grep -v grep | grep -c python; }

if [ "$P3_CONFIRMED" != "1" ]; then
	log "REFUSING TO ARM — set P3_CONFIRMED=1 once the 2x2 design is signed off"
	exit 2
fi

mkdir -p "$OUTDIR" "$MARKDIR"
log "########## ARMED — waiting for $DONE_MARKER ##########"

# ---- 1. wait for P2 to publish a COMPLETE result ----------------------------
while [ ! -f "$DONE_MARKER" ]; do
	sleep 300
done
log "P2_ALL_DONE present — P2 finished all seeds"

# ---- 2. confirm the box is free (never kill; abort instead) -----------------
sleep 60
if [ "$(controllers)" -gt 0 ] || pgrep -f "$STUDY" >/dev/null 2>&1; then
	log "ABORT: P2 says done but controllers=$(controllers) drivers=$(pgrep -fc "$STUDY" 2>/dev/null || echo 0) \
— refusing to run alongside. Needs a human."
	exit 3
fi
log "box clear: controllers=0 drivers=0"

# ---- 3. the three unmeasured corners ----------------------------------------
# INTERLEAVED BY SEED, not by corner (Luiz's sweep rule): round 1 runs one of each
# corner, then round 2, then round 3. If the box is lost partway we hold a complete
# low-n version of the WHOLE 2x2 rather than three seeds of one corner and nothing
# of the others — and an early cull becomes possible.
#
# corner: <name> <state_neurons> <input_window_k>
run_corner() {
	local name="$1" sn="$2" k="$3" seed="$4"
	local maxsn="$sn"
	run_controller_arm "P3_${name}_L3D_s${seed}" "$MARKDIR" "$OUTDIR" "$VP" log \
		"\"arm\":\"P3\",\"corner\":\"${name}\",\"state_neurons\":${sn},\"input_window_k\":${k},\"disturbance\":\"L3D\",\"features\":\"pidmix\",\"mode\":\"BINARY\",\"seed\":${seed}" \
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
		--grid-state-neurons "$sn" --max-state-neurons "$maxsn" \
		--max-output-neurons 128 \
		--input-window-k "$k" \
		--runs 1 --teacher lqr --memory-mode BINARY \
		--holdout-fixed-thresholds --disturbance L3D \
		$FEAT_PIDMIX \
		--report-seeds $REPORT_SEEDS \
		--base-seed "$seed"
}

for seed in $SEEDS; do
	log "===== ROUND seed=$seed ====="
	run_corner state    8 4 "$seed"   # THE question: does state help at L3D?
	run_corner k8       0 8 "$seed"   # control: does a longer window help instead?
	run_corner state_k8 8 8 "$seed"   # both levers
done

got=$(ls -1 "$MARKDIR"/P3_*_L3D_s*.json 2>/dev/null | wc -l | tr -d ' ')
want=$(( $(echo $SEEDS | wc -w) * 3 ))
log "########## P3 CHAIN DONE — ${got}/${want} markers ##########"
[ "$got" = "$want" ] || exit 4
