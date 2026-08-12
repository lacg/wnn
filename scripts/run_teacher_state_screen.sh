#!/usr/bin/env bash
# Teacher-state screening — does a teacher whose advantage lives in INTERNAL
# STATE actually transfer to the student, and does teaching the student to
# reconstruct that state close the gap?
#
# HYPOTHESIS
#   MPCOF beats every classical baseline on steady-state offset (0.2° vs LQR
#   1.3°, 5-seed held-out) purely because of d̂, the input-equivalent disturbance
#   estimate it accumulates:
#       observer:  d̂ ← d̂ + L·( (gyro − gyro_prev)/dt − b·u_applied − d̂ )
#       control :  u = clamp( u_mpc(x) − clamp(d̂/b, ±ff_clamp), ±authority )
#   optimal.rs calls d̂ "the teacher's accumulated memory". That makes the
#   teacher NON-MARKOVIAN in the student's observation: identical observations
#   with different d̂ demand different actions, so a blind student cannot
#   represent the policy and should inherit the very offset MPCOF removes.
#   --state-integral trains the recurrent state as a learned integrator, which
#   is the mechanism by which the student could hold d̂'s analog.
#
#   Why this matters: the WNN stability gap is PRECISION, not robustness —
#   failures are SOFT, ~5.6° steady offset. MPCOF is the offset killer, so the
#   mechanism matches our actual failure mode.
#
# DESIGN (substrate x feature x teacher x state-integral)
#
# The teacher question cannot be answered on dfa alone. Running BOTH substrates
# says what the state neurons actually buy and what they cost:
#   1layer (sn=0) — NO state layer, so --state-integral is INAPPLICABLE. This is
#     the control that shows whether a memoryless student can absorb a stateful
#     teacher at all. If mpcof beats lqr here, the gain is NOT about memory.
#   dfa (sn>0) — has the state layer that could hold d-hat's analog, so both
#     state-integral settings are meaningful.
# Both 9feat and 10feat are run so every cell is directly comparable to the
# 40-cell dfa1l study, which spans the same substrate x feature grid.
#
# The lqr/no-state-int corner is ALREADY RUN: the dfa1l study's BINARY cells at
# each (substrate, feature) are exactly that arm, same seeds and same recipe, so
# they serve as the baseline for free and are not repeated here.
#
# ARMS (4 per feature, 8 total):
#   T1 1layer mpcof            can a memoryless student use a stateful teacher?
#   T2 dfa    mpcof            same teacher, state layer available but untargeted
#   T3 dfa    mpcof + state-int  the full hypothesis: teach d-hat's analog
#   T4 dfa    lqr   + state-int  isolates the state-integral effect FROM the
#                                teacher effect (memoryless teacher, integral target)
# T3 vs T2 = what targeting the state buys. T2 vs T1 = what having a state layer
# buys. T4 vs the study's dfa-lqr baseline = the integral target alone.
#
# READOUT: held-out steady° is PRIMARY (that is the offset the hypothesis is
#   about), with stable%/err° alongside. NOT conflict rate — scan_conflicts
#   lives in the split trainer and this screen runs SPLIT=0 (BPTT), so enabling
#   it to measure conflicts would change the trainer and confound the result.
#
# READ BEFORE LAUNCHING
#   * ONE controller at a time, always behind the IDS worker. This script
#     REFUSES to start if a phased_ga python is already alive.
#   * INTERLEAVED: round r runs one of each (arm x feature) at seed[r], so after
#     round 1 you have all 8 combinations at one seed and can cull before
#     spending the rest.
#   * n=1 RANKS NOTHING. A single round is a smoke test, not a result. Ranking
#     arms needs >=3 seeds; anything read before that is provisional.
#   * COST IS THE REAL CONSTRAINT: ~17-30h per run, 4 arms x 2 features x 3 seeds
#     = 24 runs ~= 3 weeks of box time. Stage it with LIMIT and cull hard after
#     round 1; do NOT queue the whole thing blind.
#   * CAVEAT: --state-integral's help says "use small --grid-state-neurons", but
#     this screen uses the study's dfa range (8 12 16 / max 24) so cells stay
#     comparable with the 40-cell study. If T3/T4 underperform, a small-state
#     re-probe (SINT_STATE="--grid-state-neurons 4 6 8 --max-state-neurons 12")
#     is the follow-up before concluding the integral target does not work.
#
# Usage:
#   LIMIT=8 bash scripts/run_teacher_state_screen.sh          # round 1 (8 runs)
#   FEATURES=9feat LIMIT=4 bash scripts/...sh                 # 9feat arms only
#   SEEDS="31337002" LIMIT=1 bash scripts/...sh               # single smoke cell
#   bash scripts/run_teacher_state_screen.sh                  # full 24-cell screen
set -u
ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
export PYTHONPATH="${ROOT}/src/wnn:${PYTHONPATH:-}"
export WNN_RUST_DAGGER=1 RAYON_NUM_THREADS=10 WNN_CONTROLLER_GPU_EVAL=0
# SPLIT=0 (BPTT) — same trainer as the dfa1l study, so cells stay comparable.
unset WNN_STATE_SPLIT
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
[ -x "$VP" ] || VP="python"

OUTDIR="${OUTDIR:-logs/controller/teacher_screen}"
MARKDIR="${MARKDIR:-/tmp/wnn_teacher_screen}"
mkdir -p "$OUTDIR" "$MARKDIR"

SEEDS=(${SEEDS:-31337002 31337003 31337004})
FEATURES=(${FEATURES:-9feat 10feat})
REPORT_SEED="${REPORT_SEED:-99990101}"
LIMIT="${LIMIT:-0}"
_cells_done=0
_cell_ran=0

log() { echo "[tscreen] $* $(date -u +%FT%TZ)"; }

# ---- hard gate: never a second controller --------------------------------
_live=$(ps -axo pid,command 2>/dev/null | grep "wnn.control.phased_ga" \
	| grep -v "/usr/bin/time" | grep -v grep | grep -c python)
if [ "${_live:-0}" -gt 0 ]; then
	log "REFUSING to start: ${_live} phased_ga python already running."
	log "One controller at a time — wait for the current sweep to finish."
	exit 1
fi
if ! pgrep -f "wnn.ram.experiments" >/dev/null 2>&1; then
	log "NOTE: no IDS worker detected. That is allowed, but IDS is priority —"
	log "      confirm this box is meant to be running controller work alone."
fi

# Recipe is the dfa1l study's COMMON *verbatim* (gens 60/120, not trimmed) so
# every cell here is directly comparable with the 40 study cells. Trimming the
# gens would have saved nothing real anyway: those cells early-stop near gen 10,
# so the cap is not what ends them — patience is.
# BINARY: the study covers BINARY and QUAD, but BINARY is the best substrate so
# far (dfa 9feat BINARY, 92.0%/3.1°) AND its study cells give us the lqr baseline
# for free. It also avoids QUAD's cell-growth risk on a long screen.
# L2D keeps the persistent torque bias that offset-free MPC is designed to
# cancel — without it the hypothesis is untestable.
COMMON="--levels 16 --skip-stages bits,connections --lamarckian \
--max-cells 180000 --neurons-gens 60 --neurons-patience 3 \
--memory-gens 120 --memory-patience 2 --pop 50 --num-eval-folds 5 \
--check-interval 2 --magnitude-aware-patience --eval-episodes 100 \
--memory-eval-episodes 200 --steps 2000 --tilt 5.0 \
--fit-weight-err-sq 0.4 --fit-weight-stable 0.3 --fit-weight-jerk 0.2 \
--fit-weight-mono 0.1 --report-seed ${REPORT_SEED} --report-episodes 100 \
--holdout-pop-sample 8 --grid-bits 24 30 --max-output-neurons 128 \
--memory-mode BINARY --runs 1 --disturbance L2D"

# One cell. Args: arm-key substrate feature teacher state-integral(0|1) seed
run_cell() {
	local arm="$1" sub="$2" feat="$3" teacher="$4" sint="$5" seed="$6"
	local tag="${arm}_${sub}_${feat}_${teacher}$([ "$sint" = 1 ] && echo "_sint")_s${seed}"
	local marker="${MARKDIR}/${tag}.json"
	local out="${OUTDIR}/${tag}.out"
	local winner="${OUTDIR}/${tag}_winner.yaml.gz"
	_cell_ran=0
	if [ -f "$marker" ]; then log "$tag: marker exists — skip"; return; fi
	_cell_ran=1

	local sint_flag=""
	[ "$sint" = "1" ] && sint_flag="--state-integral"
	# substrate -> state-neuron flags. 1layer has NO state layer (sn=0), which is
	# why --state-integral is never combined with it: there is nothing to train as
	# an integrator. The dfa range matches the study's so cells stay comparable.
	local sub_flags
	if [ "$sub" = "1layer" ]; then
		sub_flags="--grid-state-neurons 0 --max-state-neurons 0"
	else
		sub_flags="${SINT_STATE:---grid-state-neurons 8 12 16 --max-state-neurons 24}"
	fi
	local feat_flags=""
	[ "$feat" = "10feat" ] && feat_flags="--obs-yaw-err"

	log "===== START $tag ====="
	local t0=$SECONDS
	/usr/bin/time -l "$VP" -u -m wnn.control.phased_ga $COMMON \
		$sub_flags $feat_flags --teacher "$teacher" $sint_flag --base-seed "$seed" \
		--save-winner "$winner" > "$out" 2>&1
	local rc=$? dur=$((SECONDS - t0))

	# Watchdog stop -> hold the chain, wait for memory, retry (same protocol as
	# the dfa1l driver: no marker on 143/137, calm-gated retry, attempt-3 limit).
	if [ "$rc" = "143" ] || [ "$rc" = "137" ]; then
		log "$tag: rc=$rc (watchdog) — NO marker; waiting for memory, will retry"
		local tries="${TRIES:-0}"
		if [ "$tries" -ge 2 ]; then
			log "$tag: killed ${tries}x — SKIPPING (attempt-3 limit: needs a fix, not a retry)"
			return
		fi
		local calm=0
		while [ "$calm" -lt 3 ]; do
			sleep 60
			local av
			av=$(vm_stat | awk '/free|inactive|speculative|purgeable/ {gsub("\\.","");s+=$NF} END {printf "%.0f", s*16384/1073741824}')
			if [ "${av:-0}" -ge 25 ]; then calm=$((calm+1)); else calm=0; fi
			log "$tag: waiting to retry (avail=${av}GB, calm=${calm}/3)"
		done
		TRIES=$((tries+1)) run_cell "$arm" "$sub" "$feat" "$teacher" "$sint" "$seed"
		return
	fi

	local rss held_n held_m cells
	rss=$(grep -E "maximum resident set size" "$out" | awk '{print $1}' | tail -1)
	held_n=$(grep -E "RESULT — during-search winner" "$out" | sed -n '1p')
	held_m=$(grep -E "RESULT — during-search winner" "$out" | sed -n '2p')
	cells=$(grep -oE "cells\[[0-9-]+ Σ[0-9]+k μ[0-9]+k\]" "$out" | tail -1)

	# Only a genuine completion writes a marker (R4): rc!=0 is a crash, and rc=0
	# with an empty MEMORY triple is a truncated run. Either way leave it unmarked
	# so a later pass re-runs it rather than permanently skipping a wrong number.
	if [ "$rc" != "0" ]; then
		log "$tag: rc=$rc (crash/abnormal) — NO marker; leaving for re-run"; return
	fi
	if [ -z "${held_m// /}" ]; then
		log "$tag: rc=0 but no MEMORY-stage held-out (truncated) — NO marker"; return
	fi

	printf '{"tag":"%s","arm":"%s","substrate":"%s","feature":"%s","teacher":"%s","state_integral":%s,"seed":%s,"rc":%s,"dur_s":%s,"peak_rss_bytes":%s,"cells":"%s","held_neurons":"%s","held_memory":"%s","done":"%s"}\n' \
		"$tag" "$arm" "$sub" "$feat" "$teacher" "$sint" "$seed" "$rc" "$dur" "${rss:-null}" \
		"$cells" \
		"$(echo "$held_n" | tr -d '"' | sed 's/  */ /g')" \
		"$(echo "$held_m" | tr -d '"' | sed 's/  */ /g')" \
		"$(date -u +%FT%TZ)" > "$marker"
	log "===== END $tag rc=$rc dur=${dur}s ====="
}

# arm-key : substrate : teacher : state-integral
# The lqr/no-sint corner is deliberately absent on BOTH substrates — the dfa1l
# study's BINARY cells already ARE that arm at the same seeds and recipe.
ARMS=(
	"T1:1layer:mpcof:0"
	"T2:dfa:mpcof:0"
	"T3:dfa:mpcof:1"
	"T4:dfa:lqr:1"
)

# INTERLEAVED: seeds outermost, then features, then arms — round r gives one of
# every (arm x feature) combination at seed[r], so a partial run is still a
# balanced design rather than all of one arm.
for seed in "${SEEDS[@]}"; do
	for feat in "${FEATURES[@]}"; do
		for spec in "${ARMS[@]}"; do
			IFS=':' read -r arm sub teacher sint <<< "$spec"
			run_cell "$arm" "$sub" "$feat" "$teacher" "$sint" "$seed"
			if [ "$_cell_ran" = "1" ]; then _cells_done=$((_cells_done + 1)); fi
			if [ "$LIMIT" -gt 0 ] && [ "$_cells_done" -ge "$LIMIT" ]; then
				log "LIMIT=$LIMIT reached — stopping after $_cells_done launched cell(s)"
				exit 0
			fi
		done
	done
done

> "${MARKDIR}/ALL_DONE.marker"
log "TEACHER SCREEN COMPLETE (${#ARMS[@]} arms x ${#FEATURES[@]} features x ${#SEEDS[@]} seeds)"
