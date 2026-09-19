#!/usr/bin/env bash
# MULTI-AXIS CHAIN — the OFAT programme (Plane CTRL-5; docs/multi_axis_programme_spec.md),
# ROUND-MAJOR over seed_arm_chain.sh. Written 19/09/2026 against Luiz's D1-D9 answers.
#
# WHAT IT IS. One script, conditions x seeds, marker-gated, idempotent, fails closed. Round r
# flies seed r of EVERY condition before any condition gets seed r+1 (D3 CONFIRMED; the
# standing sweep rule — a condition's SD is known after its 2nd seed, which is when "the n it
# would need" is actionable). Every launch is ONE call of scripts/seed_arm_chain.sh with
# ARM_SEEDS = that one seed, so the ladder recipe (sweep_ladder_gamma.sh) is never copied and
# ZERO ladder edits are needed (controller agent, 19/09): every condition flag is a plain
# argparse store, so ARM_EXTRA_ARGS appended AFTER the ladder's own flags wins (last-wins) for
# --disturbance / --teacher / --airframe / --grid-state-neurons / --max-state-neurons /
# --motor-lag-s / --report-seeds. The ladder's TAG still reads cf21_brushless_L4C — the
# ARM_SUFFIX (_axA_L4A, ...) is what says which condition a marker is. Read the suffix.
#
# DECISIONS HONOURED (Luiz, 19/09/2026):
#   D1 axis D = cf2x_firmware only        D2 axis E dropped        D3 round-major CONFIRMED
#   D4 anchor extended to 8 seeds: s31337006 flies in queue_1909 (CTRL-7); this chain adds
#      s31337007..09 in round 1, first (every Welch primary shares them)
#   D5 report seeds 99990201..05 on EVERY programme run (interim ticks keep reading 99990101..05)
#   D6 axis F actuator lag: --motor-lag-s 0.0375 (tau, sourced) and 0.075 (2 tau, stress)
#   D7 axis G optional tail after round 2 — NOT in this table (add a line when decided)
#   D9 programme anchor = _hd29 (ABI-29 era control), the paired control for seeds 2-5
#
# ROUND 1 ORDER (spec §5.1.5), 12 launches, ~46 h at 3.2 h/run (C budgeted 6-7 h):
#   1 anchor _hd29 s7,s8,s9   2 A L4A,L4B   3 F tau,2tau   4 B pid,lqr   5 D cf2x_firmware   6 C sn4,sn8
#   (C flies by default — its memory budget is DECIDED, spec §5.1.4a, see the table)
# Rounds 2-4: the 9 conditions at seeds 31337003/4/5 (no anchor; D4's 8 seeds are complete
# after round 1). 12 + 27 = 39 runs. §4 escalation (+2 seeds where the primary CI straddles 0
# AND includes the MEI after 4 seeds) is a HUMAN decision after round 4 — not automated.
#
# ============================ POWER STATEMENT PER AXIS (R2) ============================
# Paired-t MDE (80% power, alpha .05 two-sided) at n=4 from the _hd29 anchor SDs, re-derived
# 19/09 (paired_power.py on the _hd29 markers):  err ~0.44 deg   steady ~0.31 deg
# alt ~0.13 m   stable ~1.7 pp.  The R2 RANGES from the older arms still bracket these
# (err 0.61-0.69, steady 0.69-1.00, stable 1.9-2.0, alt 0.08-0.16); a 4-pair SD is itself
# uncertain (95% CI [0.57x, 3.7x]) so QUOTE RANGES, never a point. The PRIMARY analysis is
# Welch two-sample, anchor n=8 (s2-s9) vs condition n=4, SE ~0.61 SD — the pairing is cosmetic
# (R4: pair correlation ~0), paired-4 is the robustness line (SE 0.79 SD).
#   ROUND 1 (n=1 per condition) RESOLVES NOTHING by itself: it buys the first seed of every
#   condition so round 2 yields a per-condition SD (the re-size point). Any "verdict" printed at
#   the end of round 1 is a DIRECTION line, and the script labels it so.
#   A  DISTURBANCE  primary ERR, MEI 0.6 deg.  Resolvable at n=4 (MDE 0.44-0.69 < MEI): the plant-
#      jitter step is expected well above it (PID err 0.58 -> 1.08 -> 1.79 deg L4A/L4B/L4C).
#      CANNOT resolve V2 (gap to PID) as a test — PID moves with the rung; V2 is the READ.
#   F  ACTUATOR LAG primary ERR. Expected LARGE (every controller degrades) — resolvable at n=4;
#      the question is whether the WNN degrades MORE than the classicals (V2 read). CANNOT
#      say anything until the ABI-30 s=1 bit-identity pin at lag 0 is banked (§5.1.2 step 3).
#   B  TEACHER      primary ERR. Resolves FULL tracking (student err moves ~the 1.1 deg teacher
#      gap) vs NONE. CANNOT resolve PARTIAL tracking (0.3-0.6 deg) at n=4; the null is an
#      EQUIVALENCE claim: margin 0.55 deg, TOST 90%, decidable only if |d| < 0.2 AND SD <= 0.3.
#   D  AIRFRAME     primary ALT. n=4 at the anchor's alt SD resolves 0.08-0.16 m of the 0.27 m
#      gap. CANNOT be sized honestly today: the cf21 SDs give NO basis for the SD at another
#      airframe; D's statement is a guess until its 2nd seed lands (round 2).
#   C  STATE NEURONS primary ALT, MEI 0.16 m (the n=4 MDE at the pessimistic SD 0.077). A state
#      layer that closes less than that is indistinguishable from nothing at n=4 — said up
#      front. Reports the CONNECTIONS row too (R7): sn>0 adds a stage.
#      MEMORY BUDGET (Luiz 19/09, spec §5.1.4a): the ladder's --max-cells 180000 --max-cells-strict
#      would clamp sn>0 to a 5-50x SMALLER memory than the prior sn runs, so axis C OPENS the cap:
#      sn=4 --max-cells 1000000000 (open; ~18 GB peak fits the box), sn=8 --max-cells 4000000 (the
#      4 M cap — uncapped sn=8 peaks 35-38 GB and trips the watchdog HOG_GB 28). Each C marker
#      records "max_cells". Escape hatch: MA_REFUSE_AXIS_C=1 refuses both C runs (exit 2).
#   anchor extension  Welch 8-vs-4 is the payoff; n=8 alone also tightens the anchor's own
#      err CI (currently [0.086, 0.167] hd at n=4 spans MPC..worse-than-PID).
#   stable is DESCRIPTIVE everywhere (R11: 2-5 failures in 500; t-CI meaningless; counts only).
# ==========================================================================================
#
# KNOWN TENSION THE SCRIPT CANNOT RESOLVE (flagged, not hidden): the banked _hd29 s2-s6 were
# READ on report seeds 99990101..05; every run here (anchor s7-9 included) flies D5 = 99990201..05.
# An 8-vs-4 Welch that mixes the two sets is NOT clean. Before the FINAL table, re-score the
# s2-s6 _hd29 winners on the D5 set (rescore_winners.py, §5 #12 — minutes) and recompute the
# classical baselines on D5 (R6, §5 #6). The marker records report_seed_set so a reader can tell.
#
# PRE-CONDITIONS (human, at the idle window — §5.1.2; this script checks only what it can):
#   F wheel (ABI 30) installed + Python patch applied ATOMICALLY; s=1 pin bit-identical;
#   per-condition 60 s smokes rc=0 (B: the label-saturation print must NOT fire);
#   D5 baselines banked. The axis-C budget (§5 #8) is DECIDED (§5.1.4a) and encoded in the table —
#   no env is needed to fly it. Skew guard: the chain aborts BEFORE the first launch if phased_ga
#   --help lacks any flag a launch in the requested round(s) depends on (e.g. --motor-lag-s).
#
# USAGE
#   bash scripts/multi_axis_chain.sh --dry-run [--round N]     print the plan, touch nothing, exit 0
#   bash scripts/multi_axis_chain.sh --plan-only-json [--round N]   same plan as JSON, exit 0
#   bash scripts/multi_axis_chain.sh [--round N]               fly (default: rounds 1..4 in order)
# ARM (detached, PPID=1; macOS has no setsid — detach_launch.py uses start_new_session):
#   cd /Users/lacg/wnn && /Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python \
#     scripts/detach_launch.py /private/tmp/multi_axis_chain.launch.log /Users/lacg/wnn -- \
#     bash scripts/multi_axis_chain.sh --round 1
# VERIFY ARMED:  ps -o pid,ppid,etime,command -p <printed pid>   -> PPID must be 1
#                tail -3 /private/tmp/multi_axis_chain.log         -> "ARMED" + "preflight OK" lines
#                cat /private/tmp/multi_axis_chain.lock/pid         -> the same pid
# LOGS  chain: /private/tmp/multi_axis_chain.log   per launch: /private/tmp/multi_axis<suffix>_s<seed>.log
#       ladder: /private/tmp/sweep_ladder_gamma.log  run .out: logs/controller/sweep_ladder/<tag>.out
# EXIT CODES  0 every planned launch in the requested round(s) is banked
#             1 ABORT — lock held / box not idle / flag skew / control marker missing / a run
#               finished without a marker (R1-R3 in controller_arm_lib.sh). Box left idle. Human.
#             2 round(s) done but >=1 launch REFUSED by an escape hatch (MA_REFUSE_AXIS_C=1).
#               Relaunch the same round without it — idempotent, banked runs are skipped.
#             3 usage error
# HOLD  touch experiments/HOLD_CONTROLLER -> the flying run banks, then this chain waits (logged
#       here AND inside the ladder); rm resumes. Never kill the chain to get an idle window.
# IT WILL NOT: launch beside another controller/ladder/chain; re-fly a banked marker; retry a
#   crash (R2); write a marker (only controller_arm_lib.sh does); edit any script; fly axis C
#   when MA_REFUSE_AXIS_C=1; fly axis G/E; decide escalation; pkill anything (no kill path at all).
# A RUNNING COPY HOLDS THIS FILE IN MEMORY — editing it changes nothing until relaunch, and
#   NEVER edit it while a copy is running (bash resumes at a byte offset).
set -u
cd "$(dirname "$0")/.." || exit 1

# =============================== THE TABLE — edit here ===============================
# One line per condition:  axis|value|suffix|primary|flags|require-flags|refuse-env|extra-json
#   suffix     = ARM_SUFFIX (marker/tag suffix; the ONLY thing that names the condition)
#   primary    = paired_power.py --primary column (R1: ONE pre-registered column per axis)
#   flags      = the condition's phased_ga flags (appended last -> override the ladder's)
#   require    = flags that must exist in phased_ga --help before launch (skew guard)
#   refuse-env = escape hatch: if this env var is =1 the launch is REFUSED (empty = none)
#   extra-json = extra marker fields (raw JSON, no leading comma; empty = none)
# Order = §5.1.5 round-1 order. Seed r of each line flies in round r.
CONDITIONS=(
	"A|L4A|_axA_L4A|err|--disturbance L4A|--disturbance||"
	"A|L4B|_axA_L4B|err|--disturbance L4B|--disturbance||"
	"F|lag0.0375|_axF_lag0375|err|--motor-lag-s 0.0375|--motor-lag-s||"
	"F|lag0.075|_axF_lag075|err|--motor-lag-s 0.075|--motor-lag-s||"
	"B|pid|_axB_pid|err|--teacher pid|--teacher||"
	"B|lqr|_axB_lqr|err|--teacher lqr|--teacher||"
	"D|cf2x_firmware|_axD_cf2xfw|alt|--airframe cf2x_firmware|--airframe||"
	"C|sn4|_axC_sn4|alt|--grid-state-neurons 4 --max-state-neurons 4 --max-cells 1000000000|--grid-state-neurons --max-state-neurons --max-cells|MA_REFUSE_AXIS_C|\"max_cells\":1000000000,\"max_cells_note\":\"open; ~18 GB peak (spec 5.1.4a)\""
	"C|sn8|_axC_sn8|alt|--grid-state-neurons 8 --max-state-neurons 8 --max-cells 4000000|--grid-state-neurons --max-state-neurons --max-cells|MA_REFUSE_AXIS_C|\"max_cells\":4000000,\"max_cells_note\":\"4M cap; uncapped 35-38 GB trips HOG_GB 28 (spec 5.1.4a)\""
)
COND_SEEDS="31337002 31337003 31337004 31337005"          # seed r flies in round r (R4)
ANCHOR_SUFFIX="_hd29"                                     # D9
ANCHOR_LABEL="d0-derived-hover-abi29-refly"               # same ARM_LABEL as the banked _hd29
ANCHOR_FLAGS="--teacher-hover derived"
ANCHOR_REQUIRE="--teacher-hover"
ANCHOR_ROUND1_SEEDS="31337007 31337008 31337009"          # D4: + s31337006 from queue_1909 = 8
REPORT_SEEDS_D5="99990201 99990202 99990203 99990204 99990205"   # D5, on EVERY run
# MA_REFUSE_AXIS_C=1: escape hatch that REFUSES both C runs (loud, exit 2). Default = fly them with
# the §5.1.4a caps above. The ladder's --max-cells-strict stays; with the cap opened it is inert.
# =====================================================================================

BASE="SL_C_b24n256_cf21_brushless_L4C_g10"
MARK="experiments/sweepladder_markers"
# MA_PYTHON / MA_LOG / MA_LOCK are TEST HOOKS for tests/multi_axis_chain_dryrun.sh (stub python,
# scratch log/lock) — never set them on a production launch.
VP="${MA_PYTHON:-/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python}"
LOG="${MA_LOG:-/private/tmp/multi_axis_chain.log}"
LOCK="${MA_LOCK:-/private/tmp/multi_axis_chain.lock}"
LIB="scripts/controller_arm_lib.sh"
MA_WELCH_ARGS="${MA_WELCH_ARGS:---welch}"   # set to the exact flags once paired_power.py grows Welch
ROUNDS="1 2 3 4"
MODE="run"
REFUSED=0
BANKED_NOW=0

usage() { echo "usage: $0 [--dry-run | --plan-only-json] [--round N]" >&2; exit 3; }
while [ $# -gt 0 ]; do
	case "$1" in
		--dry-run) MODE="dry" ;;
		--plan-only-json) MODE="json" ;;
		--round) shift; [ $# -gt 0 ] || usage; case "$1" in 1|2|3|4) ROUNDS="$1" ;; *) usage ;; esac ;;
		-h|--help) sed -n '2,80p' "$0"; exit 0 ;;
		*) usage ;;
	esac
	shift
done

log() {
	if [ "$MODE" = "run" ]; then echo "[multi-axis] $(date -u +%FT%TZ) $*" | tee -a "$LOG"
	else echo "[multi-axis] $*"; fi
}
tag_of() { echo "${BASE}_s$1$2"; }
banked() { [ -f "${MARK}/$(tag_of "$1" "$2").json" ]; }
nth_seed() { echo "$COND_SEEDS" | awk -v n="$1" '{print $n}'; }
busy() {
	pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null \
	|| pgrep -f "scripts/sweep_ladder_gamm[a].sh" >/dev/null \
	|| pgrep -f "scripts/seed_arm_chai[n].sh" >/dev/null \
	|| pgrep -f "scripts/queue_[0-9]" >/dev/null
}
wait_box_clear() {
	local beat=0
	while busy; do
		sleep 30; beat=$((beat + 1))
		[ $((beat % 30)) = 0 ] && log "waiting — box busy (another controller/ladder/chain)"
	done
}

# ---- plan --------------------------------------------------------------------------
# A plan record: r|idx|axis|value|suffix|seed|primary|flags|require|refuse|label|control|extra
# control = _hd29 (paired) or NONE (anchor extension, ARM_NO_CONTROL=1); refuse = escape-hatch env.
PLAN=()
build_plan() {
	local r="$1" idx=0 seed rec axis value suffix primary flags require gate extra
	PLAN=()
	if [ "$r" = "1" ]; then
		for seed in $ANCHOR_ROUND1_SEEDS; do
			idx=$((idx + 1))
			PLAN[${#PLAN[@]}]="$r|$idx|anchor|hd29-ext|${ANCHOR_SUFFIX}|${seed}|-|${ANCHOR_FLAGS}|${ANCHOR_REQUIRE}||${ANCHOR_LABEL}|NONE|"
		done
	fi
	seed="$(nth_seed "$r")"
	for rec in "${CONDITIONS[@]}"; do
		IFS='|' read -r axis value suffix primary flags require gate extra <<< "$rec"
		idx=$((idx + 1))
		PLAN[${#PLAN[@]}]="$r|$idx|${axis}|${value}|${suffix}|${seed}|${primary}|${flags}|${require}|${gate}|multi-axis-${axis}-${value}|${ANCHOR_SUFFIX}|${extra}"
	done
}
status_of() {   # <suffix> <seed> <refuse-env>
	if banked "$2" "$1"; then echo banked
	elif [ -n "$3" ] && [ "$(eval "echo \${$3:-0}")" = "1" ]; then echo "REFUSED(${3}=1)"
	else echo todo; fi
}
print_plan_line() {
	local r idx axis value suffix seed primary flags require gate label control extra
	IFS='|' read -r r idx axis value suffix seed primary flags require gate label control extra <<< "$1"
	printf 'PLAN r%s #%02d axis=%s cond=%s seed=%s tag=%s control=%s primary=%s status=%s flags="%s --report-seeds %s"\n' \
		"$r" "$idx" "$axis" "$suffix" "$seed" "$(tag_of "$seed" "$suffix")" "$control" "$primary" \
		"$(status_of "$suffix" "$seed" "$gate")" "$flags" "$REPORT_SEEDS_D5"
}
print_plan_json() {
	local first=1 r idx axis value suffix seed primary flags require gate label control extra
	echo "["
	for rr in $ROUNDS; do
		build_plan "$rr"
		for rec in "${PLAN[@]}"; do
			IFS='|' read -r r idx axis value suffix seed primary flags require gate label control extra <<< "$rec"
			[ "$first" = 1 ] || echo ","
			first=0
			printf '  {"round":%s,"idx":%s,"axis":"%s","condition":"%s","suffix":"%s","seed":%s,"tag":"%s","control":"%s","primary":"%s","status":"%s","flags":"%s --report-seeds %s","require_flags":"%s --report-seeds","refuse_env":"%s"}' \
				"$r" "$idx" "$axis" "$value" "$suffix" "$seed" "$(tag_of "$seed" "$suffix")" "$control" "$primary" \
				"$(status_of "$suffix" "$seed" "$gate")" "$flags" "$REPORT_SEEDS_D5" "$require" "$gate"
		done
	done
	echo; echo "]"
}

# ---- preflight (run mode only) -------------------------------------------------------
take_lock() {
	if mkdir "$LOCK" 2>/dev/null; then echo $$ > "$LOCK/pid"; return 0; fi
	local old; old="$(cat "$LOCK/pid" 2>/dev/null || echo)"
	if [ -n "$old" ] && kill -0 "$old" 2>/dev/null; then
		log "ABORT — another multi_axis_chain (pid $old) holds $LOCK. Nothing launched."; exit 1
	fi
	log "stale lock (pid ${old:-?} dead) — reclaiming"
	rm -rf "$LOCK"; mkdir "$LOCK" || exit 1; echo $$ > "$LOCK/pid"
}
release_lock() { [ "$(cat "$LOCK/pid" 2>/dev/null)" = "$$" ] && rm -rf "$LOCK"; }

preflight() {
	busy && { log "ABORT — the box is NOT idle. Never launch this chain beside another."; exit 1; }
	[ -f "$LIB" ] || { log "ABORT — $LIB missing."; exit 1; }
	[ -x "$VP" ] || { log "ABORT — venv python $VP missing."; exit 1; }
	# Skew guard over EVERY launch in the requested rounds that will actually fly — abort
	# BEFORE the first launch, not 20 h in when seed_arm_chain reaches the condition.
	local help need="--report-seeds" rec r idx axis value suffix seed primary flags require gate label control extra f missing=""
	help="$(PYTHONPATH=src/wnn "$VP" -m wnn.control.phased_ga --help 2>/dev/null)"
	[ -n "$help" ] || { log "ABORT — phased_ga --help printed nothing (tree broken?)."; exit 1; }
	for r in $ROUNDS; do
		build_plan "$r"
		for rec in "${PLAN[@]}"; do
			IFS='|' read -r r idx axis value suffix seed primary flags require gate label control extra <<< "$rec"
			[ "$(status_of "$suffix" "$seed" "$gate")" = "todo" ] || continue
			need="$need $require"
			if [ "$control" != "NONE" ] && ! banked "$seed" "$control"; then
				missing="$missing $(tag_of "$seed" "$control")"
			fi
		done
	done
	for f in $(echo "$need" | tr ' ' '\n' | sort -u); do
		echo "$help" | grep -q -- "$f" || { log "ABORT — phased_ga has no $f (source/wheel skew, or the flag has not landed). Nothing launched."; exit 1; }
	done
	[ -z "$missing" ] || { log "ABORT — paired control markers missing:${missing}. Nothing to pair against."; exit 1; }
	log "preflight OK — box idle, $(echo "$need" | tr ' ' '\n' | sort -u | grep -c .) flags present, controls (${ANCHOR_SUFFIX}) banked for every paired launch in rounds [${ROUNDS}]."
}

# ---- one launch ------------------------------------------------------------------------
launch_one() {
	local r idx axis value suffix seed primary flags require gate label control extra tag st
	IFS='|' read -r r idx axis value suffix seed primary flags require gate label control extra <<< "$1"
	tag="$(tag_of "$seed" "$suffix")"
	st="$(status_of "$suffix" "$seed" "$gate")"
	case "$st" in
		banked) log "r$r #$idx SKIP — $tag already banked."; return 0 ;;
		REFUSED*) log "r$r #$idx REFUSED — $tag: escape hatch ${gate}=1. Unset it and relaunch --round $r to fly it."; REFUSED=$((REFUSED + 1)); return 0 ;;
	esac
	wait_while_held log "$tag"
	wait_box_clear
	log "===== r$r #$idx LAUNCH $tag (axis $axis = $value; control $control; primary $primary) ====="
	local marker_json=",\"programme\":\"multi-axis\",\"axis\":\"${axis}\",\"condition\":\"${value}\",\"round\":${r},\"primary\":\"${primary}\",\"report_seed_set\":\"D5\",\"report_seeds\":\"${REPORT_SEEDS_D5}\""
	[ -n "$extra" ] && marker_json="${marker_json},${extra}"
	if [ "$control" = "NONE" ]; then
		ARM_SUFFIX="$suffix" ARM_LABEL="$label" ARM_SEEDS="$seed" ARM_NO_CONTROL=1 ARM_CTRL_SUFFIX="_hd" \
			ARM_EXTRA_ARGS="${flags} --report-seeds ${REPORT_SEEDS_D5}" \
			ARM_REQUIRE_FLAGS="${require} --report-seeds" \
			ARM_MARKER_JSON="${marker_json},\"refly_of\":\"_hd\",\"purpose\":\"anchor extension (D4) for the multi-axis Welch primary\"" \
			ARM_LOG="/private/tmp/multi_axis${suffix}_s${seed}.log" bash scripts/seed_arm_chain.sh
	else
		ARM_SUFFIX="$suffix" ARM_LABEL="$label" ARM_SEEDS="$seed" ARM_CTRL_SUFFIX="$control" \
			ARM_EXTRA_ARGS="${flags} --report-seeds ${REPORT_SEEDS_D5}" \
			ARM_REQUIRE_FLAGS="${require} --report-seeds" \
			ARM_MARKER_JSON="$marker_json" \
			ARM_LOG="/private/tmp/multi_axis${suffix}_s${seed}.log" bash scripts/seed_arm_chain.sh
	fi
	log "seed_arm_chain exited rc=$? for $tag"
	banked "$seed" "$suffix" || { log "ABORT — marker ${tag}.json MISSING after the run (R1-R3: killed, crashed or truncated). A run needs a human. Box left idle."; exit 1; }
	BANKED_NOW=$((BANKED_NOW + 1))
	log "r$r #$idx banked: $tag"
}

# ---- end-of-round verdict --------------------------------------------------------------
anchor_inventory() {
	local s n=0 list=""
	for s in 31337002 31337003 31337004 31337005 31337006 31337007 31337008 31337009; do
		banked "$s" "$ANCHOR_SUFFIX" && { n=$((n + 1)); list="$list $s"; }
	done
	echo "$n:$list"
}
round_verdict() {
	local r="$1" rec axis value suffix primary flags require gate extra s seeds inv nn
	inv="$(anchor_inventory)"; nn="${inv%%:*}"
	log "---------- ROUND $r VERDICT — anchor ${ANCHOR_SUFFIX} banked n=${nn} (seeds${inv#*:}) ----------"
	[ "$r" = "1" ] && log "ROUND 1 = n=1 per condition: DIRECTION only. Nothing below is a verdict (R2)."
	for rec in "${CONDITIONS[@]}"; do
		IFS='|' read -r axis value suffix primary flags require gate extra <<< "$rec"
		seeds=""
		for s in $(echo "$COND_SEEDS" | awk -v n="$r" '{for(i=1;i<=n;i++) print $i}'); do
			banked "$s" "$suffix" && seeds="$seeds $s"
		done
		if [ -z "$seeds" ]; then log "$suffix: no markers yet — no read."; continue; fi
		log "--- $suffix (axis $axis, primary $primary) paired vs ${ANCHOR_SUFFIX}, seeds [${seeds# }] — ROBUSTNESS line (R4) ---"
		# shellcheck disable=SC2046
		PYTHONPATH=src/wnn "$VP" scripts/paired_power.py --arm "$suffix" --base "${BASE}_s{seed}" \
			$(for s in $seeds; do printf -- '--seed %s ' "$s"; done) \
			--control-suffix "$ANCHOR_SUFFIX" --primary "$primary" 2>&1 | tee -a "$LOG"
		if PYTHONPATH=src/wnn "$VP" scripts/paired_power.py --help 2>/dev/null | grep -q -- "--welch"; then
			log "--- $suffix WELCH (PRIMARY, R4): condition n=$(echo $seeds | wc -w | tr -d ' ') vs anchor n=${nn} ---"
			# shellcheck disable=SC2046,SC2086
			PYTHONPATH=src/wnn "$VP" scripts/paired_power.py $MA_WELCH_ARGS --arm "$suffix" --base "${BASE}_s{seed}" \
				$(for s in $seeds; do printf -- '--seed %s ' "$s"; done) \
				--control-suffix "$ANCHOR_SUFFIX" --primary "$primary" 2>&1 | tee -a "$LOG" \
				|| log "TODO — paired_power.py $MA_WELCH_ARGS exited non-zero; check MA_WELCH_ARGS against its interface. Chain continues."
		else
			log "TODO — paired_power.py has no --welch yet (parallel worktree): run the Welch primary for $suffix by hand once it lands. Chain continues."
		fi
	done
	log "CAVEAT: _hd29 s2-s6 were read on 99990101..05; programme runs fly D5 99990201..05 — re-score s2-s6 on D5 before the FINAL table."
	log "CAVEAT: stable is descriptive (R11); alt is the well-resolved channel (R3); hd is descriptive."
}

# ============================== MAIN ==============================
case "$MODE" in
	json) print_plan_json; exit 0 ;;
	dry)
		log "DRY RUN — nothing launched, nothing written. Rounds [${ROUNDS}]. Anchor ${ANCHOR_SUFFIX}; report seeds D5 = ${REPORT_SEEDS_D5}."
		for r in $ROUNDS; do
			build_plan "$r"
			log "ROUND $r — ${#PLAN[@]} launches, round-major (seed $( [ "$r" = 1 ] && echo "$(nth_seed 1) + anchor ${ANCHOR_ROUND1_SEEDS}" || nth_seed "$r"))"
			for rec in "${PLAN[@]}"; do print_plan_line "$rec"; done
		done
		if [ "${MA_REFUSE_AXIS_C:-0}" = "1" ]; then log "MA_REFUSE_AXIS_C=1 — axis C is REFUSED (exit 2 in run mode)."
		else log "axis C flies with the spec 5.1.4a caps (sn4 --max-cells 1000000000, sn8 --max-cells 4000000); MA_REFUSE_AXIS_C=1 is the escape hatch."; fi
		exit 0 ;;
esac

. "$LIB"          # read-only: wait_while_held (the ladder sources the same file; never edited here)
take_lock
trap release_lock EXIT
log "########## ARMED — multi-axis chain, rounds [${ROUNDS}], anchor ${ANCHOR_SUFFIX}, D5 report seeds, ${#CONDITIONS[@]} conditions + anchor ext [${ANCHOR_ROUND1_SEEDS}] ##########"
preflight
for r in $ROUNDS; do
	build_plan "$r"
	log "========== ROUND $r — ${#PLAN[@]} planned launches =========="
	for rec in "${PLAN[@]}"; do launch_one "$rec"; done
	round_verdict "$r"
done
if [ "$REFUSED" -gt 0 ]; then
	log "########## ROUNDS [${ROUNDS}] DONE WITH ${REFUSED} REFUSED launch(es) (gate) — ${BANKED_NOW} banked this pass; box IDLE; exit 2 ##########"
	exit 2
fi
log "########## ROUNDS [${ROUNDS}] COMPLETE — ${BANKED_NOW} banked this pass, every planned launch banked; box IDLE ##########"
exit 0
