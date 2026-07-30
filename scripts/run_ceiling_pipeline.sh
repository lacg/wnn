#!/usr/bin/env bash
# Ceiling pipeline — "imitation as initialization, reward as the optimizer".
# Designed 30/07/2026 (docs/motor_fault_experiment.md + the gap decomposition).
#
#   B  data-budget probe : retrain the winner arch at 1x and 16x DAgger budget.
#                          If data alone closes the 77->96 gap, stop redesigning.
#   A  nominal memory-GA : seed-winner MEMORY-only value-GA (closed-loop reward,
#                          NO teacher ceiling), LONG (800 gens, patience 20).
#                          Payoff question: past the imitation plateau?
#   C  motor-fault       : the same memory-GA on the fault=0.3 plant (+ matched
#                          fault baselines). Beat MPCOF's 20.0% = first genuine
#                          WNN-beats-classical result.
#
# Phase-A/C patience is DELIBERATELY loose (user 30/07): tight patience was a
# param-search economy; the long run is the point here. Known failure mode from
# the same history: the memory-GA sometimes finds train-pool optima worth
# -20..-30pp on held-out — the per-stage HELD-OUT REPORT is the tripwire, and
# --save-winner keeps the population either way, so an overfit outcome is still
# a measured, reportable result (not lost work).
#
# ONE-CONTROLLER RULE: this script REFUSES to start while any phased_ga python
# is alive. Run it at a dfa1l cell boundary or after the sweep. Phases run
# sequentially inside the same guard.
#
# Usage: run_ceiling_pipeline.sh [B|A|C|all]   (default: all)
set -u
ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
export PYTHONPATH="${ROOT}/src/wnn:${PYTHONPATH:-}"
export WNN_RUST_DAGGER=1 RAYON_NUM_THREADS=10 WNN_CONTROLLER_GPU_EVAL=0
unset WNN_STATE_SPLIT
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
[ -x "$VP" ] || VP="python"

PHASE="${1:-all}"
OUTDIR="${OUTDIR:-logs/controller/ceiling}"
MARKDIR="${MARKDIR:-experiments/dfa1l_markers}"
WINNER="${WINNER:-logs/controller/dfa1l/dfa_9feat_BINARY_s31337002_winner.yaml.gz}"
BASE_SEED="${BASE_SEED:-31337002}"     # MUST match the winner's _sNNNN
REPORT_SEED="${REPORT_SEED:-99990101}"
FAULT="${FAULT:-1:0.3}"                # motor 2 at 30% effectiveness (the cliff)
mkdir -p "$OUTDIR"

log() { echo "[ceiling] $* $(date -u +%FT%TZ)"; }

# ---- one-controller guard (never a second phased_ga) ------------------------
guard() {
	local n
	n=$(ps -axo command= | grep -c "[w]nn\.control\.phased_ga")
	if [ "$n" -gt 0 ]; then
		log "REFUSING to start: $n phased_ga process(es) alive (one-controller rule)."
		log "Run at a dfa1l cell boundary or after the sweep."
		exit 2
	fi
	if [ ! -f "$WINNER" ]; then
		log "REFUSING: winner not found: $WINNER"; exit 3
	fi
}

# The dfa1l COMMON recipe (run_dfa_1layer_study.sh), minus the stage budgets the
# phases override. Keep in sync with the study or the arms stop being comparable.
COMMON="--levels 16 --pop 50 --num-eval-folds 5 --check-interval 2 \
--magnitude-aware-patience --eval-episodes 100 --memory-eval-episodes 200 \
--steps 2000 --tilt 5.0 --fit-weight-err-sq 0.4 --fit-weight-stable 0.3 \
--fit-weight-jerk 0.2 --fit-weight-mono 0.1 --report-seed ${REPORT_SEED} \
--report-episodes 100 --holdout-pop-sample 8 --grid-bits 24 30 \
--grid-state-neurons 8 12 16 --max-state-neurons 24 --max-output-neurons 128 \
--runs 1 --teacher lqr --disturbance L2D --base-seed ${BASE_SEED}"

phase_B() {
	log "===== PHASE B: data-budget probe (16x) ====="
	"$VP" -u scripts/data_budget_probe.py --winner "$WINNER" --scale 16 \
		--base-seed "$BASE_SEED" --report-seed "$REPORT_SEED" \
		--out "$MARKDIR/data_budget_probe.json" \
		> "$OUTDIR/phaseB_probe.out" 2>&1
	local rc=$?
	log "phase B rc=$rc"; tail -5 "$OUTDIR/phaseB_probe.out"
	return $rc
}

phase_A() {
	log "===== PHASE A: nominal long memory-GA (800 gens, patience 20) ====="
	/usr/bin/time -l "$VP" -u -m wnn.control.phased_ga $COMMON \
		--neurons-gens 0 --bits-gens 0 --conns-gens 0 \
		--memory-gens 800 --memory-patience 20 \
		--seed-winner "$WINNER" --seed-winner-stage memory \
		--save-winner "$OUTDIR/nominal_memga_winner.yaml.gz" \
		> "$OUTDIR/phaseA_nominal_memga.out" 2>&1
	local rc=$?
	log "phase A rc=$rc"
	grep -E "RESULT — during-search winner" "$OUTDIR/phaseA_nominal_memga.out" | tail -2
	return $rc
}

phase_C() {
	log "===== PHASE C: motor-fault (${FAULT}) memory-GA + matched baselines ====="
	# Baselines on the SAME faulted plant first (scoring-only, minutes).
	"$VP" -u scripts/compute_baselines.py --disturbance L2D --motor-fault "$FAULT" \
		--report-seeds 99990101 99990102 99990103 99990104 99990105 \
		--report-episodes 100 --steps 2000 \
		--out "$MARKDIR/baselines_fault_${FAULT//:/x}.json" \
		> "$OUTDIR/phaseC_baselines.out" 2>&1 || { log "fault baselines FAILED"; return 4; }
	tail -7 "$OUTDIR/phaseC_baselines.out"
	/usr/bin/time -l "$VP" -u -m wnn.control.phased_ga $COMMON \
		--motor-fault "$FAULT" \
		--neurons-gens 0 --bits-gens 0 --conns-gens 0 \
		--memory-gens 800 --memory-patience 20 \
		--seed-winner "$WINNER" --seed-winner-stage memory \
		--save-winner "$OUTDIR/fault_memga_winner.yaml.gz" \
		> "$OUTDIR/phaseC_fault_memga.out" 2>&1
	local rc=$?
	log "phase C rc=$rc"
	grep -E "RESULT — during-search winner" "$OUTDIR/phaseC_fault_memga.out" | tail -2
	log "beat-the-classical bar on this plant: MPCOF 20.0% stable (see baselines_fault json)"
	return $rc
}

guard
case "$PHASE" in
	B)   phase_B ;;
	A)   phase_A ;;
	C)   phase_C ;;
	all) phase_B && phase_A && phase_C ;;
	*)   echo "usage: $0 [B|A|C|all]"; exit 1 ;;
esac
log "pipeline ($PHASE) done"
