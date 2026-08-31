#!/usr/bin/env bash
# LONG-BUDGET b=48 RUN (30/08/2026, Luiz approved: "yes, queue the long-budget b=48
# run behind the probe").
#
# WHAT THIS TESTS — and it is the ONLY thing that tests it. The wide probe gives every
# width the SAME budget (--conns-gens 5 --conns-patience 3), but b=48 searches a much
# larger space than b=36 (Σ32.6M vs Σ12.3M cells at gen 1). So "performance falls off
# past b=36" is equally consistent with:
#   (H1) b=36 is the REPRESENTATIONAL optimum — wider tuples spread the training data
#        over more addresses, so each address generalizes worse. Width is the cause.
#   (H2) the GA simply RUNS OUT OF BUDGET before wide tuples pay off. Budget is the
#        cause, and the "knee" is an artifact of holding gens fixed across widths.
# Both predict the identical curve shape in the probe, so the probe cannot separate
# them. This run can: same width, same seed, same everything — only the connections
# budget changes (5 gens/patience 3 -> 20 gens/patience 5).
#   b=48 climbs to b=36 territory  => H2. The knee is a budget artifact and the whole
#                                     bits sweep needs re-reading at matched search.
#   b=48 stays put                 => H1. The knee is real and b=36 stands.
# Either way it is ONE run and it resolves a confound that nothing else scheduled will.
#
# ORDERING — why this is safe alongside the handoff supervisor (PID 97139).
# That supervisor waits for the probe's 6 markers, then wait_no_controller, then
# relaunches the ladder chain. wait_no_controller is a PURE WAIT that never escalates.
# So if this run is already flying when it looks, it simply waits for us and relaunches
# the ladder afterwards — we insert into a gap it already leaves, and we never need to
# touch it. That matters because BOTH supervisors are running right now and a .sh must
# never be edited while bash is executing it (bash resumes at a byte offset).
# We poll at 3s against its 60s so we reliably claim the idle box first; if we lose the
# race anyway, the guard below stands down rather than run two controllers.
set -u
ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
. "$ROOT/scripts/controller_arm_lib.sh"

LOG="/private/tmp/b48_longbudget.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
OUTDIR="logs/controller/sweep_ladder"
MARKDIR="experiments/sweepladder_markers"
AIRFRAME="cf21_brushless"; DIST="L4C"; SEED=31337002; B=48; SWEEP_N=32
TAG="SL_A_b${B}n${SWEEP_N}_${AIRFRAME}_${DIST}_g20_s${SEED}"
REPORT_SEEDS="99990101 99990102 99990103 99990104 99990105"

log() { echo "[b48-long] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controller_pids() { pgrep -f -- "-m wnn.control.phased_ga" 2>/dev/null || true; }
chain_pids() { pgrep -f "scripts/sweep_ladder_ab_chain.sh" 2>/dev/null || true; }

mkdir -p "$OUTDIR/ckpt/$TAG" "$MARKDIR"
log "########## ARMED — long-budget b=48, waits for the wide probe's 6 markers ##########"

# ---- 1. wait for the probe to finish (all 6 markers). 3s poll: see header.
while :; do
	missing=0
	for b in 40 48 64; do
		[ -f "${MARKDIR}/SL_A_b${b}n${SWEEP_N}_${AIRFRAME}_${DIST}_s${SEED}.json" ] || missing=1
		[ -f "${MARKDIR}/SL_A_b${b}n${SWEEP_N}_${AIRFRAME}_${DIST}_desir_s${SEED}.json" ] || missing=1
	done
	[ "$missing" = "0" ] && break
	sleep 3
done
log "wide probe COMPLETE (6/6 markers)."

# ---- 2. claim the box. ONE controller at a time, checked twice.
for attempt in 1 2; do
	while [ -n "$(controller_pids)" ] || [ -n "$(chain_pids)" ]; do
		log "waiting for box: controllers=[$(controller_pids | tr '\n' ' ')] chain=[$(chain_pids | tr '\n' ' ')]"
		sleep 20
	done
	sleep 2
done
log "box is clear — claiming it."

# ---- 3. COLLISION GUARD. If the handoff won the race and the ladder chain starts
# alongside us, WE stand down: kill our own run, leave no marker (so it is re-runnable),
# and let the main programme proceed untouched. Never disturb the running programme to
# make room for an inserted run.
(
	sleep 5
	for _ in $(seq 1 36); do   # watch the first ~3 minutes
		if [ -n "$(chain_pids)" ]; then
			log "COLLISION: ladder chain started alongside us — standing down (killing OUR run, no marker)."
			pkill -f -- "-m wnn.control.phased_ga.*_g20_s${SEED}" 2>/dev/null
			pkill -f "b48_longbudget_supervisor.sh" 2>/dev/null
			exit 0
		fi
		sleep 5
	done
) &

# ---- 4. THE RUN. Identical to scripts/sweep_ladder_probe_wide.sh in every respect —
# same S16noJM weights, same GATE aggregation, same features, folds 5, same report
# seeds, same base seed — EXCEPT the connections budget. That single difference is the
# experiment; anything else changing would reintroduce the confound it exists to kill.
#   probe:     --conns-gens 5  --conns-patience 3
#   this run:  --conns-gens 20 --conns-patience 5     <- the only delta
# Patience 5 is the project default (memory: PATIENCE IS 5); the probe's 3 is the
# tighter setting the sweep used to fit 26 runs. 20 gens is a CAP, not a target: if
# b=48 cannot improve, magnitude-aware patience stops it around gen 10-12 (~6h) and
# that null IS the answer. If it keeps improving it runs longer, which is the answer too.
log "===== START $TAG (conns-gens 20, conns-patience 5; probe used 5/3) ====="
run_controller_arm "$TAG" "$MARKDIR" "$OUTDIR" "$VP" log \
	"\"stage\":\"A\",\"sweep\":\"bits-budget\",\"arm\":\"gate\",\"budget\":\"conns20p5\",\"bits\":${B},\"neurons\":${SWEEP_N},\"input_window_k\":1,\"seed\":${SEED}" \
	-- \
	--levels 16 --lamarckian \
	--skip-stages neurons,bits \
	--max-cells 180000 --max-cells-strict \
	--save-stage-checkpoints "$OUTDIR/ckpt/$TAG" \
	--neurons-gens 5 --neurons-patience 3 \
	--conns-gens 20 --conns-patience 5 \
	--memory-gens 120 --memory-patience 2 \
	--pop 50 --num-eval-folds 5 --check-interval 2 --magnitude-aware-patience \
	--eval-episodes 100 --memory-eval-episodes 200 \
	--steps 2000 --tilt 5.0 \
	--fit-weight-err-sq 0.3125 --fit-weight-stable 0.25 --fit-weight-steady 0.4375 \
	--fit-aggregation zscore --zrank-clamp 3.0 --gate-stable 0.70 --gate-err 8.0 \
	--grid-bits "$B" --grid-output-neurons "$SWEEP_N" --max-output-neurons "$SWEEP_N" \
	--report-episodes 100 --holdout-pop-sample 8 \
	--runs 1 --memory-mode BINARY \
	--airframe "$AIRFRAME" --disturbance "$DIST" --teacher mpcof \
	--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i \
	--obs-collective-cmd --obs-alt-err --obs-vz \
	--translation --reward-lambda-alt 0 \
	--grid-state-neurons 0 --max-state-neurons 0 \
	--report-seeds $REPORT_SEEDS \
	--base-seed "$SEED"
log "$TAG finished rc=$? ########## LONG-BUDGET b=48 COMPLETE ##########"
log "READ IT AS: compare headline_holdout vs the probe's b48 (5 gens) and vs b=36's 66.6%/5.94/6.53."
