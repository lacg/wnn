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
# WHY A 2x2 AND NOT ONE RUN. input_window_k and recurrent state are two mechanisms
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
# Which predecessor this run waits on. Default is P2 (the original chain order).
# 04/08/2026: P3's stateful corners were re-queued BEHIND P4 — 4.6h per NEURONS
# GENERATION at sn=8 (not per run) makes each one a multi-day run, while P4's 6 runs
# total ~7h, so P4 first returns an answer the same day instead of in a fortnight.
# Set P3_WAIT_MARKER=experiments/p4_markers/P4_ALL_DONE.marker for that ordering.
DONE_MARKER="${P3_WAIT_MARKER:-experiments/p2_markers/P2_ALL_DONE.marker}"
STUDY="scripts/run_dfa_1layer_study.sh"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
OUTDIR="logs/controller/p3"
MARKDIR="experiments/p3_markers"
SEEDS="${P3_SEEDS:-31337002 31337003 31337004}"
REPORT_SEEDS="99990101 99990102 99990103 99990104 99990105"
P3_CONFIRMED="${P3_CONFIRMED:-0}"
# CORNER SUBSET (04/08/2026). The three corners are NOT equal-cost, and the original
# all-three-per-round order made that expensive: measured on matched seed/features,
# sn=0 runs a whole cell in ~40 min (A4: 2496s, 12 NEURONS gens + 10 MEMORY gens)
# while sn=8 spends 16657s on NEURONS gen 1 ALONE — ~150x per generation. Running
# corners expensive-first therefore buried the cheap CONTROL behind days of the
# treatment, which is backwards: the control is what tells you whether the treatment
# is even the right question. P3_CORNERS lets a run take the cheap corner first and
# decide before committing the box to the stateful ones.
#   P3_CORNERS="control"               the sn=0 stateless CONTROL alone (~40 min)
#   P3_CORNERS="state"                 the K=4 stateful corner alone
#   P3_CORNERS="control state state_k8"  control first, then both treatments (default)
# The k8 corner was CANCELLED 04/08 (void at sn=0), so it is no longer a valid value.
P3_CORNERS="${P3_CORNERS:-control control_l3 state state_k8}"

# TEACHER (04/08/2026). Was hardcoded lqr. Changed to lqi because the L3D baselines
# say lqr is the WRONG teacher for THIS disturbance — measured, 5-seed held-out,
# experiments/dfa1l_markers/baselines_L3D.json:
#           L2D               L3D
#   LQR   100.0% / 1.60°    60.8%+-18.3 / 5.48°
#   LQI   100.0% / 1.36°    70.6%+-15.0 / 4.82°   <- +9.8pp stable, -0.66 deg
# The two tie at L2D and diverge at L3D, and the mechanism is in the run's own header:
# L3D carries tau_bias=0.0540 N.m, a CONSTANT torque offset. Pure state feedback (LQR)
# cannot null a constant disturbance; the integral term (LQI) exists to. A DAgger
# student cannot exceed the policy it clones, so cloning the 60.8% teacher caps the
# student below a ceiling that is not the substrate's fault.
# The teacher was originally picked from an L2D screening (LQR > PID) — a ranking that
# INVERTS at L3D. A screening result is only valid in the regime it was screened in.
P3_TEACHER="${P3_TEACHER:-lqi}"
# Slug for the completion marker, so a subset run publishes something a downstream
# chain can key on WITHOUT claiming the whole 2x2 finished.
CORNERS_SLUG="$(echo $P3_CORNERS | tr ' ' '-')"

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
# corner: <name> <state_neuron_GRID_AXIS> <max_state_neurons> <input_window_k> <seed>
# sn is an AXIS (space-separated), deliberately unquoted at the flag so it expands to
# the multiple values --grid-state-neurons takes.
run_corner() {
	local name="$1" sn="$2" maxsn="$3" k="$4" seed="$5" dist="$6"
	# The teacher is IN THE TAG, not just the marker body. A marker that recorded the
	# teacher only internally would still be skipped by name on a re-run under a
	# different teacher — silently returning the wrong teacher's cell as this one's
	# result. In the filename, that mistake cannot be made.
	run_controller_arm "P3_${name}_${dist}_${P3_TEACHER}_s${seed}" "$MARKDIR" "$OUTDIR" "$VP" log \
		"\"arm\":\"P3\",\"corner\":\"${name}\",\"state_neurons\":\"${sn}\",\"max_state_neurons\":${maxsn},\"input_window_k\":${k},\"disturbance\":\"${dist}\",\"features\":\"pidmix\",\"mode\":\"BINARY\",\"teacher\":\"${P3_TEACHER}\",\"seed\":${seed}" \
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
		--grid-state-neurons $sn --max-state-neurons "$maxsn" \
		--max-output-neurons 128 \
		--input-window-k "$k" \
		--runs 1 --teacher "$P3_TEACHER" --memory-mode BINARY \
		--disturbance "$dist" \
		$FEAT_PIDMIX \
		--report-seeds $REPORT_SEEDS \
		--base-seed "$seed"
}

# corner name -> (state_neuron grid AXIS, max_state_neurons, input_window_k). One
# place, so the loop below and any subset selection cannot disagree about what a
# corner means.
#
# 04/08/2026: the stateful corners use dfa1l's ACTUAL sn recipe — the axis {8,12,16}
# with headroom to 24 (run_dfa_1layer_study.sh:87) — not the pinned sn=8 this chain
# shipped with. Pinning 8 hard-coded dfa1l's OUTCOME back in as an input (6 of its 7
# dfa cells settled on sn=8, one on sn=16), which is the same antipattern as seeding
# a search from a winner-of-one. Building on prior work means reusing the recipe, not
# its result.
#
# The k8 corner is GONE, not merely unselected — see P3_k8_CANCELLED.marker.
# --input-window-k cannot reach a memoryless policy: the output layer samples
# `sensor_frame` (one frame) while only the state layer samples `sensor_window`
# (k frames), so at sn=0 K is inert BY CONSTRUCTION. P3_k8_L3D_s31337002 reproduced
# A4_L3D_s31337002 bit-identically, held-out included. A4 IS that corner, already
# measured, and is the stateless floor the two stateful corners are scored against.
# Asserted by tests/controller_arch_shape_invariants.py [5].
# The `control` corner is the sn=0 STATELESS floor, shape-identical to A4
# (run_l3d_feature_probe.sh COMMON: --grid-state-neurons 0 --max-state-neurons 0,
# --input-window-k default 4). A4 already measured that floor at teacher=lqr
# (4.2% stable / 11.02 deg, 5-seed). It does NOT carry over to teacher=lqi: changing
# the teacher moves the floor, so scoring lqi treatments against an lqr floor would
# confound teacher with state. This corner re-measures the floor under the SAME
# teacher as the treatments — one variable, matched pair. It is also the cheap one
# (~40 min vs multi-day), so it runs FIRST and can veto the treatments.
# DISTURBANCE IS A PER-CORNER AXIS (04/08/2026). L3D is not one notch above L2D: it
# is the L3 ladder rung PLUS the whole D5/D6/D7 extension, so L2D->L3D moves NINE
# parameters at once (tau_bias/gust 2x, gyro 2.67x, asym 1.5x, torque jitter 1.67x,
# obs_delay 2x, and sensor-frozen duty 3.8%->16.7%). A collapse measured across that
# jump cannot be attributed to any one of them. `control_l3` is the same sn=0 control
# at PLAIN L3 — the ladder magnitudes with NONE of the D-fields — so
#   control (L3D) vs control_l3 (L3)  =  the D-extension ALONE.
# It costs ~40 min and it gates the multi-day stateful corners: if plain L3 also
# collapses, the sensor pathology is not the target and state is aimed at the wrong
# variable. Cheap control before expensive treatment, same rule as the corner order.
corner_dist()  { case "$1" in control_l3) echo "L3" ;; *) echo "L3D" ;; esac; }
corner_sn()    { case "$1" in control|control_l3) echo "0" ;; state|state_k8) echo "8 12 16" ;; *) echo "" ;; esac; }
corner_maxsn() { case "$1" in control|control_l3) echo 0 ;; state|state_k8) echo 24 ;; *) echo "" ;; esac; }
corner_k()     { case "$1" in control|control_l3|state) echo 4 ;; state_k8) echo 8 ;; *) echo "" ;; esac; }

for seed in $SEEDS; do
	log "===== ROUND seed=$seed (corners: $P3_CORNERS) ====="
	for corner in $P3_CORNERS; do
		sn="$(corner_sn "$corner")"; maxsn="$(corner_maxsn "$corner")"; k="$(corner_k "$corner")"
		dist="$(corner_dist "$corner")"
		if [ -z "$sn" ] || [ -z "$maxsn" ] || [ -z "$k" ]; then
			log "UNKNOWN corner '$corner' — refusing to guess (k8 was CANCELLED, see P3_k8_CANCELLED.marker)"
			exit 5
		fi
		run_corner "$corner" "$sn" "$maxsn" "$k" "$seed" "$dist"
	done
done

# Count ONLY the corners this run was asked for. Globbing all P3_*_L3D_s*.json would
# make a subset run look incomplete forever (or, worse, look complete because an
# earlier subset's markers are still on disk).
got=0
for seed in $SEEDS; do
	for corner in $P3_CORNERS; do
		[ -f "${MARKDIR}/P3_${corner}_$(corner_dist "$corner")_${P3_TEACHER}_s${seed}.json" ] && got=$((got+1))
	done
done
want=$(( $(echo $SEEDS | wc -w) * $(echo $P3_CORNERS | wc -w) ))
if [ "$got" = "$want" ]; then
	: > "${MARKDIR}/P3_ALL_DONE_${CORNERS_SLUG}.marker"
	log "########## P3 CHAIN DONE — ${got}/${want} markers (${P3_CORNERS}), ALL_DONE published ##########"
else
	log "########## P3 CHAIN INCOMPLETE — ${got}/${want} markers (${P3_CORNERS}), NO ALL_DONE ##########"
	exit 4
fi
