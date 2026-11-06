#!/usr/bin/env bash
# MULTI-AXIS CHAIN — the CTRL-5 OFAT programme (docs/multi_axis_programme_spec.md, DRAFT 4 + the
# Decisions 05/11/2026 block). REWRITTEN 05/11/2026: the 19/09 version wrapped seed_arm_chain.sh →
# sweep_ladder_gamma.sh, i.e. the `_hd` stage-1 ladder recipe; it would have flown the wrong recipe.
#
# WHAT IT IS. One script, conditions x seeds, ROUND-MAJOR (round r flies seed r of every condition),
# winner-gated + marker-banked, idempotent, fails closed, ONE controller at a time. It never kills
# anything and never edits a script. NOT LAUNCHED by its author — arm it by hand (see ARM below).
#
# RECIPE = THE CTRL-8 WINNER, VERBATIM. The anchor's own `[argv]` line is read from its s3 .out
# (logs/controller/ctrl8_ab_pop50/<arm|twin>_s3.out) with --base-seed / --save-* stripped; the
# condition flags are appended AFTER it (argparse last-wins). No recipe is copied into this file.
#   MA_ANCHOR=arm|twin   REQUIRED in run mode — the CTRL-8 rule_v2 verdict, set by a human.
# THE ANCHOR FLIES IN THIS CHAIN (ABI 35 item I, Luiz 05/11/2026). The recipe carries no
#   --motor-lag-s, so under --translation it now flies the AIRFRAME-BOUND lag (cf21_brushless
#   T=0.20 s, Gräfe 2026 — phased_ga AIRFRAME_MOTOR_LAG_S). The CTRL-8 seeds were flown LAG-FREE
#   and are therefore NO LONGER the anchor: the ANCHOR condition (`ma_anchor_s<seed>`, no condition
#   flags) flies FIRST in every round r at seed r, so every condition seed has its same-seed,
#   same-plant anchor before the round's verdict. (Supersedes decision 2's "CTRL-8 seeds ARE the
#   anchor".)
# REPORT SEEDS: the anchor argv's own 99990101..05 (interim set, so every headline compares to the
#   banked anchor headlines directly). The FINAL table re-scores every crown on D5 99990201..05 with
#   `phased_ga <argv> --report-seeds <D5> --recalc-headline <ckpt>` (spec §5 #12) — not this chain.
#
# DECISIONS 05/11/2026 (Luiz) ENCODED:
#   (1) n=3 interim stop per condition under rule_v2: after a condition's 3rd seed banks, the round
#       verdict runs scripts/headline_compare.py (paired vs the anchor's same seeds); if the interim-
#       stop condition is MET (all three attitude CIs clear their margins, same direction) the
#       condition is STOPPED (experiments/multi_axis_markers/<suffix>.STOPPED) and seeds 4-5 never
#       fly; otherwise it continues to n=5.
#       sn=8 flies ONLY if sn=4 showed an effect — a HUMAN reading: touch
#       experiments/multi_axis_markers/SN8_APPROVED after reading sn4's verdict. Until then sn8 is
#       REFUSED (exit 2) and the chain moves on.
#   (3) cf2x_firmware flies at --rotor-drag 0.25 (Förster 2015, rotor-only — disclosed) and
#       INHERITS its airframe's lag T=0.26 s (Förster 2015 H2 pole, item I) — no lag flag here.
#   (4) axis C has NO pre-set cell cap: the smoke re-measures at the current anchor size. Axis C is
#       REFUSED until MA_AXIS_C_MAX_CELLS is set by a human from that measurement (a number appended
#       as --max-cells; 1000000000 = the anchor's own "uncapped").
#   (6) ITEM I (05/11/2026): axis F = lag OFF (--motor-lag-s 0, the lag-free arm vs the lagged
#       anchor) and the stress T=0.30. The F030 approval gate is GONE: the teachers are lag-aware
#       since ABI 34. A banked run's `[provenance]` must carry the resolved `motor_lag=` token.
#   (5) wheel FROZEN at ABI 35 sha 7e08dd27843f19a1 (re-frozen 06/11, Luiz; post-2a27ce01 canonical export order): preflight refuses any other installed wheel, and
#       a run whose `[provenance]` line differs is NOT banked (abort).
#
# ATTEMPT-3 RULE. A launch that fails (rc!=0, no winner, no headline, wrong provenance) aborts the
# chain and counts an attempt in <tag>.attempts; at 3 attempts the launch is DEFERRED (REFUSED) on
# every later pass. Retry = a human relaunching the same round after reading the failure.
#
# USAGE
#   bash scripts/multi_axis_chain.sh --dry-run [--round N]   plan + status, touches nothing
#   bash scripts/multi_axis_chain.sh --smoke-list            the per-condition pop-6 smoke argv
#   MA_ANCHOR=arm bash scripts/multi_axis_chain.sh [--round N]   fly (default rounds 1..5)
# ARM (detached; macOS has no setsid):
#   cd /Users/lacg/wnn && MA_ANCHOR=arm /Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python \
#     scripts/detach_launch.py /private/tmp/multi_axis_chain.launch.log /Users/lacg/wnn -- \
#     bash scripts/multi_axis_chain.sh --round 1
# GATES  waits while experiments/HOLD_CONTROLLER exists (wait_while_held) and while ANY controller /
#   ladder / chain / CTRL-8 script runs (bracketed pgrep patterns — never self-match).
# LOGS   chain /private/tmp/multi_axis_chain.log · runs logs/controller/multi_axis/<tag>.out
# MARKERS experiments/multi_axis_markers/<tag>.json (written ONLY after every success check)
# EXIT   0 all planned launches banked/stopped · 1 ABORT (human) · 2 done but >=1 REFUSED · 3 usage
# NEVER edit this file while a copy runs (bash resumes at a byte offset).
set -u
cd "$(dirname "$0")/.." || exit 1

# =============================== THE TABLE — edit here ===============================
# axis|value|suffix|flags|gate     gate: ""=none · AXIS_C=needs MA_AXIS_C_MAX_CELLS · SN8=AXIS_C + SN8_APPROVED
# Order = the ANCHOR first (item I), then round-1 order of spec draft 4 §4 (A, F, B, D, C).
# The ANCHOR row MUST stay first and keep suffix `anchor`: the verdict pairs every condition
# against its banked outs (anchor_out) and skips it as an arm.
CONDITIONS=(
	"ANC|airframe-lag|anchor||"
	"A|L4A|axA_L4A|--disturbance L4A|"
	"A|L4B|axA_L4B|--disturbance L4B|"
	"F|off|axF_lagoff|--motor-lag-s 0|"
	"F|T0.30|axF_T030|--motor-lag-s 0.30|"
	"B|pid|axB_pid|--teacher pid|"
	"B|lqr|axB_lqr|--teacher lqr|"
	"D|cf2x_firmware|axD_cf2xfw_kd025|--airframe cf2x_firmware --rotor-drag 0.25|"
	"C|sn4|axC_sn4|--grid-state-neurons 4 --max-state-neurons 4|AXIS_C"
	"C|sn8|axC_sn8|--grid-state-neurons 8 --max-state-neurons 8|SN8"
)
SEEDS="31337002 31337003 31337004 31337005 31337006"   # seed r flies in round r
FROZEN_ABI="35"
FROZEN_SHA="7e08dd27843f19a1"
# =====================================================================================

# MA_PYTHON / MA_LOG / MA_LOCK / MA_TIME / MA_CTRL8_DIR are TEST HOOKS for
# tests/multi_axis_chain_dryrun.sh — never set them on a production launch.
VP="${MA_PYTHON:-/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python}"
LOG="${MA_LOG:-/private/tmp/multi_axis_chain.log}"
LOCK="${MA_LOCK:-/private/tmp/multi_axis_chain.lock}"
TIME_CMD="${MA_TIME-/usr/bin/time -l}"
CTRL8_DIR="${MA_CTRL8_DIR:-logs/controller/ctrl8_ab_pop50}"
OUT="logs/controller/multi_axis"
MARK="experiments/multi_axis_markers"
LIB="scripts/controller_arm_lib.sh"
ANCHOR="${MA_ANCHOR:-}"
ROUNDS="1 2 3 4 5"
MODE="run"
REFUSED=0
BANKED_NOW=0

usage() { echo "usage: $0 [--dry-run | --smoke-list] [--round N]" >&2; exit 3; }
while [ $# -gt 0 ]; do
	case "$1" in
		--dry-run) MODE="dry" ;;
		--smoke-list) MODE="smoke" ;;
		--round) shift; [ $# -gt 0 ] || usage; case "$1" in 1|2|3|4|5) ROUNDS="$1" ;; *) usage ;; esac ;;
		-h|--help) sed -n '2,60p' "$0"; exit 0 ;;
		*) usage ;;
	esac
	shift
done
case "$ANCHOR" in ""|arm|twin) ;; *) echo "MA_ANCHOR must be arm or twin (got '$ANCHOR')" >&2; exit 3 ;; esac

log() {
	if [ "$MODE" = "run" ]; then echo "[multi-axis] $(date -u +%FT%TZ) $*" | tee -a "$LOG"
	else echo "[multi-axis] $*"; fi
}
nth_seed() { echo "$SEEDS" | awk -v n="$1" '{print $n}'; }
seed_idx() { echo "$SEEDS" | awk -v s="$1" '{for(i=1;i<=NF;i++) if($i==s) print i}'; }
has_headline() { [ -f "$1" ] && grep -a -q '\[stage-select\] HEADLINE held-out:' "$1"; }

# ---- the anchor --------------------------------------------------------------------------
anchor_out() {   # <seed> -> this chain's own ANCHOR-condition .out for that seed (item I)
	echo "${OUT}/$(tag_of anchor "$1").out"
}
anchor_argv() {   # the winner's recipe, verbatim, minus the per-run fields
	local src="${CTRL8_DIR}/${ANCHOR:-arm}_s3.out"
	[ -f "$src" ] || { echo ""; return; }
	grep -a -m1 '^\[argv\]' "$src" | sed -E 's/^\[argv\] //; s/--base-seed [0-9]+//g; s/--save-stage-checkpoints [^ ]+//g; s/--save-winner [^ ]+//g; s/  +/ /g; s/ +$//'
}

# ---- status ------------------------------------------------------------------------------
tag_of() { echo "ma_$1_s$2"; }
banked() { [ -f "${MARK}/$(tag_of "$1" "$2").json" ]; }
attempts() { cat "${MARK}/$(tag_of "$1" "$2").attempts" 2>/dev/null || echo 0; }
stopped() { [ -f "${MARK}/$1.STOPPED" ]; }
gate_state() {   # <gate> -> "" (open) or the refusal reason
	case "$1" in
		"") echo "" ;;
		AXIS_C) [ -n "${MA_AXIS_C_MAX_CELLS:-}" ] && echo "" || echo "MA_AXIS_C_MAX_CELLS unset (decision 4: set it from the smoke's measurement)" ;;
		SN8) if [ -z "${MA_AXIS_C_MAX_CELLS:-}" ]; then echo "MA_AXIS_C_MAX_CELLS unset"
		     elif [ ! -f "${MARK}/SN8_APPROVED" ]; then echo "sn=8 waits for ${MARK}/SN8_APPROVED (decision 1: only if sn=4 shows an effect)"
		     else echo ""; fi ;;
	esac
}
status_of() {   # <suffix> <seed> <gate>
	local n; n="$(seed_idx "$2")"
	if banked "$1" "$2"; then echo banked
	elif [ "$n" -ge 4 ] && stopped "$1"; then echo "STOPPED(n=3 interim)"
	elif [ "$(attempts "$1" "$2")" -ge 3 ]; then echo "DEFERRED(attempt-3)"
	elif [ -n "$(gate_state "$3")" ]; then echo "REFUSED"
	else echo todo; fi
}
cond_flags() {   # <flags> <gate> -> flags + the axis-C cap when gated
	case "$2" in AXIS_C|SN8) echo "$1 --max-cells ${MA_AXIS_C_MAX_CELLS:-UNSET}" ;; *) echo "$1" ;; esac
}

# ---- plan: r|axis|value|suffix|seed|flags|gate ------------------------------------------
PLAN=()
build_plan() {
	local r="$1" seed rec axis value suffix flags gate
	PLAN=()
	seed="$(nth_seed "$r")"
	for rec in "${CONDITIONS[@]}"; do
		IFS='|' read -r axis value suffix flags gate <<< "$rec"
		PLAN[${#PLAN[@]}]="$r|${axis}|${value}|${suffix}|${seed}|${flags}|${gate}"
	done
}
print_plan_line() {
	local r axis value suffix seed flags gate st why
	IFS='|' read -r r axis value suffix seed flags gate <<< "$1"
	st="$(status_of "$suffix" "$seed" "$gate")"
	why="$(gate_state "$gate")"
	printf 'PLAN r%s axis=%-6s cond=%-16s seed=%s tag=%s status=%s flags="%s"%s\n' "$r" "$axis" "$suffix" "$seed" \
		"$(tag_of "$suffix" "$seed")" "$st" "$(cond_flags "$flags" "$gate")" "${why:+ [$why]}"
}

# ---- preflight / lock (run mode) -----------------------------------------------------------
busy() {
	pgrep -f "wnn.control.phased_g[a]" >/dev/null \
	|| pgrep -f "scripts/sweep_ladder_gamm[a].sh" >/dev/null \
	|| pgrep -f "scripts/seed_arm_chai[n].sh" >/dev/null \
	|| pgrep -f "scripts/queue_[0-9]" >/dev/null \
	|| pgrep -f "scripts/ctrl8_[a-z0-9_]*\.s[h]" >/dev/null
}
wait_box_clear() {
	local beat=0
	while busy; do
		sleep 60; beat=$((beat + 1))
		[ $((beat % 30)) = 0 ] && log "waiting — another controller / ladder / chain is running"
	done
}
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
wheel_identity() {
	PYTHONPATH=src/wnn "$VP" -c "from wnn.control.provenance import collect_provenance as c; p=c(''); print(p.abi, p.wheel_sha256)" 2>/dev/null
}
preflight() {
	[ -n "$ANCHOR" ] || { log "ABORT — MA_ANCHOR unset: the CTRL-8 verdict names the anchor (arm|twin)."; exit 1; }
	[ -f "$LIB" ] || { log "ABORT — $LIB missing."; exit 1; }
	[ -x "$VP" ] || { log "ABORT — venv python $VP missing."; exit 1; }
	[ -n "$(anchor_argv)" ] || { log "ABORT — no [argv] in ${CTRL8_DIR}/${ANCHOR}_s3.out."; exit 1; }
	# Item I: the anchor must fly the AIRFRAME-BOUND lag — a recipe carrying its own lag flag would
	# silently override it (and the F conditions' last-wins flags would no longer be the only lag).
	if echo " $(anchor_argv) " | grep -qE -- " --(motor-lag-s|no-motor-lag)( |=)"; then
		log "ABORT — the anchor argv carries a motor-lag flag; item I needs it to resolve the airframe lag. Nothing launched."; exit 1
	fi
	local id; id="$(wheel_identity)"
	[ "$id" = "${FROZEN_ABI} ${FROZEN_SHA}" ] || { log "ABORT — installed wheel is '${id:-unknown}', the programme is FROZEN at ABI ${FROZEN_ABI} sha ${FROZEN_SHA} (decision 5). Nothing launched."; exit 1; }
	local help f
	help="$(PYTHONPATH=src/wnn "$VP" -m wnn.control.phased_ga --help 2>/dev/null)"
	[ -n "$help" ] || { log "ABORT — phased_ga --help printed nothing (tree broken?)."; exit 1; }
	for f in --disturbance --motor-lag-s --no-motor-lag --teacher --airframe --rotor-drag --grid-state-neurons --max-state-neurons --max-cells; do
		echo "$help" | grep -q -- "$f" || { log "ABORT — phased_ga has no $f (source/wheel skew). Nothing launched."; exit 1; }
	done
	log "preflight OK — anchor ${ANCHOR}, wheel ABI ${FROZEN_ABI} sha ${FROZEN_SHA}, flags present."
}

# ---- one launch ------------------------------------------------------------------------------
check_run() {   # <rc> <out> <winner> -> "" if bankable, else the reason
	local rc="$1" out="$2" win="$3" prov
	[ "$rc" = "0" ] || { echo "rc=$rc"; return; }
	[ -f "$win" ] || { echo "no winner file"; return; }
	has_headline "$out" || { echo "no HEADLINE held-out line"; return; }
	prov="$(grep -a -m1 '^\[provenance\]' "$out")"
	echo "$prov" | grep -q "abi=${FROZEN_ABI} " && echo "$prov" | grep -q "wheel_sha256=${FROZEN_SHA}" \
		|| { echo "provenance mismatch: ${prov:-none}"; return; }
	# Item I: the run must state the lag it RESOLVED (a pre-item-I tree would fly lag-free silently).
	echo "$prov" | grep -q " motor_lag=" || { echo "provenance has no motor_lag= (pre-item-I tree): ${prov}"; return; }
	echo ""
}
bank_marker() {   # <tag> <axis> <value> <suffix> <seed> <out> <dur_s> <flags>
	local head prov rss
	head="$(grep -a '\[stage-select\] HEADLINE held-out:' "$6" | tail -1 | sed -E 's/^ +//; s/"//g')"
	prov="$(grep -a -m1 '^\[provenance\]' "$6" | sed 's/"//g')"
	rss="$(grep -a 'maximum resident set size' "$6" | tail -1 | awk '{print $1}')"
	mkdir -p "$MARK"
	printf '{"tag":"%s","programme":"CTRL-5 multi-axis","spec":"draft 4","axis":"%s","condition":"%s","suffix":"%s","seed":%s,"anchor":"%s","flags":"%s","report_seed_set":"interim 99990101-05","dur_s":%s,"peak_rss_bytes":%s,"headline":"%s","provenance":"%s","out":"%s"}\n' \
		"$1" "$2" "$3" "$4" "$5" "$ANCHOR" "$8" "$7" "${rss:-null}" "$head" "$prov" "$6" > "${MARK}/$1.json.tmp" \
		&& mv "${MARK}/$1.json.tmp" "${MARK}/$1.json"
}
launch_one() {
	local r axis value suffix seed flags gate tag st full out win rc t0 why n
	IFS='|' read -r r axis value suffix seed flags gate <<< "$1"
	tag="$(tag_of "$suffix" "$seed")"
	st="$(status_of "$suffix" "$seed" "$gate")"
	case "$st" in
		banked) log "r$r SKIP $tag — banked."; return 0 ;;
		STOPPED*) log "r$r SKIP $tag — condition stopped at the n=3 interim look."; return 0 ;;
		DEFERRED*) log "r$r DEFERRED $tag — 3 failed attempts (attempt-3 rule)."; REFUSED=$((REFUSED + 1)); return 0 ;;
		REFUSED) log "r$r REFUSED $tag — $(gate_state "$gate")."; REFUSED=$((REFUSED + 1)); return 0 ;;
	esac
	full="$(cond_flags "$flags" "$gate")"
	wait_while_held log "$tag"
	wait_box_clear
	mkdir -p "$OUT/ckpt"
	out="$OUT/${tag}.out"; win="$OUT/${tag}_winner.yaml.gz"
	[ -f "$out" ] && mv "$out" "${out%.out}.attempt$(attempts "$suffix" "$seed").out"
	log "===== r$r LAUNCH $tag (axis $axis = $value; anchor $ANCHOR) flags: ${full:-<none>} ====="
	t0=$SECONDS
	# shellcheck disable=SC2086
	PYTHONPATH=src/wnn nice -n 5 $TIME_CMD "$VP" -u -m wnn.control.phased_ga $(anchor_argv) $full \
		--base-seed "$seed" --save-stage-checkpoints "$OUT/ckpt/$tag" --save-winner "$win" > "$out" 2>&1
	rc=$?
	why="$(check_run "$rc" "$out" "$win")"
	if [ -n "$why" ]; then
		n=$(( $(attempts "$suffix" "$seed") + 1 )); mkdir -p "$MARK"; echo "$n" > "${MARK}/${tag}.attempts"
		log "ABORT — $tag NOT banked ($why); attempt $n/3. A human reads $out before relaunching. Box left idle."
		exit 1
	fi
	bank_marker "$tag" "$axis" "$value" "$suffix" "$seed" "$out" "$((SECONDS - t0))" "$full"
	BANKED_NOW=$((BANKED_NOW + 1))
	log "r$r banked $tag | $(grep -a '\[stage-select\] HEADLINE held-out' "$out" | tail -1 | cut -c1-220)"
}

# ---- round verdict (rule_v2 via headline_compare.py) ---------------------------------------
cond_outs() {   # <suffix> -> the banked .outs of a condition, one per line
	local s
	for s in $SEEDS; do banked "$1" "$s" && echo "$OUT/$(tag_of "$1" "$s").out"; done
}
round_verdict() {
	local r="$1" rec axis value suffix flags gate outs n args s js
	log "---------- ROUND $r VERDICT (paired vs anchor ${ANCHOR}, rule_v2; REPORT headline) ----------"
	for rec in "${CONDITIONS[@]}"; do
		IFS='|' read -r axis value suffix flags gate <<< "$rec"
		[ "$suffix" = "anchor" ] && continue   # the control group, not an arm
		outs="$(cond_outs "$suffix")"; n="$(echo "$outs" | grep -c .)"
		[ "$n" -ge 2 ] || { log "$suffix: n=$n — direction only, no read."; continue; }
		args=""
		for s in $outs; do args="$args --arm $s"; done
		for s in $SEEDS; do banked anchor "$s" && args="$args --ctl $(anchor_out "$s")"; done
		js="/private/tmp/multi_axis_verdict_${suffix}.json"
		log "--- $suffix (axis $axis = $value), n=$n ---"
		# shellcheck disable=SC2086
		PYTHONPATH=src/wnn "$VP" scripts/headline_compare.py --mode paired $args --json "$js" 2>&1 | tee -a "$LOG"
		# shellcheck disable=SC2086
		PYTHONPATH=src/wnn "$VP" scripts/headline_compare.py --mode welch $args 2>&1 | tail -9 | tee -a "$LOG"
		if [ "$n" = "3" ] && ! stopped "$suffix" \
			&& [ "$("$VP" -c 'import json,sys; print(json.load(open(sys.argv[1]))["interim_stop"])' "$js" 2>/dev/null)" = "True" ]; then
			mkdir -p "$MARK"
			echo "STOPPED $(date -u +%FT%TZ) at n=3: rule_v2 interim-stop MET ($js)" > "${MARK}/${suffix}.STOPPED"
			log "$suffix: n=3 INTERIM STOP MET — seeds 4-5 will not fly (decision 1)."
		fi
	done
}

# ---- smoke list (decision: smokes fly after the CTRL-8 verdict; NOT run by this script) ---------
print_smokes() {
	local rec axis value suffix flags gate argv
	argv="$(anchor_argv)"
	[ -n "$argv" ] || { echo "no anchor argv (${CTRL8_DIR}/${ANCHOR:-arm}_s3.out)" >&2; exit 1; }
	echo "# pop-6 plumbing smokes (anchor ${ANCHOR:-arm}; numbers are NOT read; decisions at pop 50)."
	echo "# assert rc=0 + [PLANT] motor_lag=… + [provenance] abi=${FROZEN_ABI} sha ${FROZEN_SHA} + GRID + HEADLINE; /usr/bin/time -l gives peak RSS."
	echo "# axis C: no cap (decision 4) — read peak RSS + cells, then set MA_AXIS_C_MAX_CELLS."
	for rec in "${CONDITIONS[@]}"; do
		IFS='|' read -r axis value suffix flags gate <<< "$rec"
		echo "cd /Users/lacg/wnn && PYTHONPATH=src/wnn nice -n 5 /usr/bin/time -l $VP -u -m wnn.control.phased_ga $argv" \
			"--pop 6 --steps 200 --neurons-gens 1 --memory-gens 1 --eval-episodes 2 --memory-eval-episodes 2 --report-episodes 2" \
			"--base-seed 1 ${flags:+$flags }--save-stage-checkpoints /private/tmp/ma_smoke/$suffix > /private/tmp/ma_smoke_$suffix.out 2>&1"
	done
}

# ============================== MAIN ==============================
case "$MODE" in
	smoke) print_smokes; exit 0 ;;
	dry)
		log "DRY RUN — nothing launched, nothing written. Rounds [${ROUNDS}]. Anchor ${ANCHOR:-UNSET (showing arm)}."
		log "recipe: $(anchor_argv | cut -c1-160)…"
		for r in $ROUNDS; do
			build_plan "$r"
			log "ROUND $r — seed $(nth_seed "$r"), anchor out $(anchor_out "$(nth_seed "$r")")"
			for rec in "${PLAN[@]}"; do print_plan_line "$rec"; done
		done
		exit 0 ;;
esac

. "$LIB"          # read-only: wait_while_held
take_lock
trap release_lock EXIT
log "########## ARMED — multi-axis chain (draft 4), rounds [${ROUNDS}], anchor ${ANCHOR} ##########"
preflight
for r in $ROUNDS; do
	build_plan "$r"
	log "========== ROUND $r — ${#PLAN[@]} planned launches (seed $(nth_seed "$r")) =========="
	for rec in "${PLAN[@]}"; do launch_one "$rec"; done
	round_verdict "$r"
done
if [ "$REFUSED" -gt 0 ]; then
	log "########## ROUNDS [${ROUNDS}] DONE WITH ${REFUSED} REFUSED/DEFERRED — ${BANKED_NOW} banked this pass; exit 2 ##########"
	exit 2
fi
log "########## ROUNDS [${ROUNDS}] COMPLETE — ${BANKED_NOW} banked this pass ##########"
exit 0
