#!/usr/bin/env bash
# multi_axis_chain.sh — proof harness. NOTHING REAL IS LAUNCHED: the chain runs inside a scratch
# root with a STUB seed_arm_chain.sh (records its ARM_* env, optionally writes the marker), a STUB
# python (canned --help, no-op paired_power), a fake pgrep (box idle) and scratch log/lock paths.
# The real tree is reached only through a read-only symlink to experiments/ for the banked _hd29
# controls. Production PIDs are counted before and after with the REAL pgrep — this chain has no
# kill path at all, and the harness asserts that stays true.
#
# What is proven:
#   PLAN   --dry-run --round 1 = exactly 12 PLAN lines, §5.1.5 order, D5 report seeds on every
#          line, anchor control=NONE, conditions control=_hd29, axis C carries its §5.1.4a caps
#          (sn4 --max-cells 1000000000, sn8 --max-cells 4000000) and is REFUSED only under the
#          MA_REFUSE_AXIS_C=1 escape hatch; banked markers show status=banked; rounds 2-4 = 9 lines
#          each, correct seed, no anchor;
#          all rounds = 39; --plan-only-json is valid JSON with 12 entries; dry-run writes nothing.
#   RUN    skew guard aborts BEFORE any launch; missing paired control aborts before any launch;
#          happy round 1 launches all 12 in plan order with the right ARM_* contract (C markers carry
#          max_cells) and exits 0; MA_REFUSE_AXIS_C=1 refuses the 2 C runs -> exit 2; relaunch skips
#          everything banked (idempotent); a run that banks no marker aborts the chain at THAT launch;
#          lock held by a live pid aborts; busy box aborts; HOLD sentinel delays the launch.
#
# Run: bash tests/multi_axis_chain_dryrun.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REAL_PGREP="$(command -v pgrep)"
prod_pids() { "$REAL_PGREP" -f "wnn.control.phased_g[a]|sweep_ladder_gamm[a]|seed_arm_chai[n]|queue_[0-9]" | sort | tr '\n' ' '; }
PROD_BEFORE="$(prod_pids)"

TD=$(mktemp -d)
trap 'rm -rf "$TD"' EXIT
FAILS=0
check() {
	local label="$1" got="$2" want="$3"
	if [ "$got" = "$want" ]; then printf '  ok   %-64s -> %s\n' "$label" "$got"
	else printf '  FAIL %-64s -> %s (expected %s)\n' "$label" "$got" "$want"; FAILS=$((FAILS + 1)); fi
}

# ---- scratch root: the chain cds to $(dirname $0)/.. so give it a root of its own ---------
SR="$TD/root"; mkdir -p "$SR/scripts" "$SR/experiments/sweepladder_markers" "$SR/bin"
cp "$ROOT/scripts/multi_axis_chain.sh" "$SR/scripts/"
ln -s "$ROOT/scripts/controller_arm_lib.sh" "$SR/scripts/controller_arm_lib.sh"
touch "$SR/scripts/paired_power.py"
# banked controls: the four _hd29 anchors, copied (never the real dir — the stub writes markers)
for s in 31337002 31337003 31337004 31337005; do
	cp "$ROOT/experiments/sweepladder_markers/SL_C_b24n256_cf21_brushless_L4C_g10_s${s}_hd29.json" "$SR/experiments/sweepladder_markers/"
done
CALLS="$TD/calls.log"
# stub seed_arm_chain.sh: record the ARM_* contract; write the marker unless STUB_NO_MARKER=1
cat > "$SR/scripts/seed_arm_chain.sh" <<'EOS'
#!/usr/bin/env bash
printf '%s|%s|%s|%s|%s|%s\n' "$ARM_SUFFIX" "$ARM_SEEDS" "${ARM_NO_CONTROL:-0}" "${ARM_CTRL_SUFFIX:-}" "$ARM_EXTRA_ARGS" "${ARM_MARKER_JSON:-}" >> "$STUB_CALLS"
[ "${STUB_NO_MARKER:-0}" = "1" ] && exit 0
echo "{\"tag\":\"SL_C_b24n256_cf21_brushless_L4C_g10_s${ARM_SEEDS}${ARM_SUFFIX}\"${ARM_MARKER_JSON:-},\"stub\":true}" \
	> "experiments/sweepladder_markers/SL_C_b24n256_cf21_brushless_L4C_g10_s${ARM_SEEDS}${ARM_SUFFIX}.json"
exit 0
EOS
chmod +x "$SR/scripts/seed_arm_chain.sh"
# stub python: `--help` prints $STUB_HELP (the flag inventory); anything else is a no-op
cat > "$SR/bin/python" <<'EOS'
#!/usr/bin/env bash
case " $* " in *" --help "*) printf '%s\n' "$STUB_HELP"; exit 0 ;; esac
echo "[stub-python] $*"; exit 0
EOS
chmod +x "$SR/bin/python"
# fake pgrep: exit 1 = nothing running (box idle) unless FAKE_BUSY=1
cat > "$SR/bin/pgrep" <<'EOS'
#!/usr/bin/env bash
[ "${FAKE_BUSY:-0}" = "1" ] && { echo 4242; exit 0; }
exit 1
EOS
chmod +x "$SR/bin/pgrep"
FULL_HELP="--disturbance --motor-lag-s --teacher --airframe --grid-state-neurons --max-state-neurons --max-cells --report-seeds --teacher-hover --welch"
D5="--report-seeds 99990201 99990202 99990203 99990204 99990205"
WANT_ORDER="_hd29 _hd29 _hd29 _axA_L4A _axA_L4B _axF_T015 _axF_T030 _axB_pid _axB_lqr _axD_cf2xfw _axC_sn4 _axC_sn8"

chain() {   # run the scratch copy with the harness env; args pass through
	( cd "$SR" && PATH="$SR/bin:$PATH" MA_PYTHON="$SR/bin/python" MA_LOG="$TD/chain.log" MA_LOCK="$TD/lock" \
		STUB_CALLS="$CALLS" STUB_HELP="${STUB_HELP:-$FULL_HELP}" WNN_HOLD_FILE="${HOLD_FILE:-$TD/no-hold}" WNN_HOLD_POLL_S=1 \
		bash scripts/multi_axis_chain.sh "$@" )
}
plan_field() { sed -nE "s/.* $1=([^ ]+).*/\1/p"; }

echo
echo "=== PLAN: --dry-run --round 1 ==="
OUT="$(chain --dry-run --round 1)"; rc=$?
check "exit code" "$rc" "0"
PLAN1="$(echo "$OUT" | grep '^PLAN r1 ')"
check "12 PLAN lines" "$(echo "$PLAN1" | grep -c .)" "12"
check "§5.1.5 order (anchor x3, A, F, B, D, C)" "$(echo "$PLAN1" | plan_field cond | tr '\n' ' ' | sed 's/ $//')" "$WANT_ORDER"
check "every line carries the D5 report seeds" "$(echo "$PLAN1" | grep -c -- "$D5")" "12"
check "no line carries the interim seeds" "$(echo "$PLAN1" | grep -c "99990101")" "0"
check "anchor lines: seeds 7,8,9 and control=NONE" "$(echo "$PLAN1" | grep 'cond=_hd29' | plan_field seed | tr '\n' ' ')$(echo "$PLAN1" | grep 'cond=_hd29' | grep -c 'control=NONE')" "31337007 31337008 31337009 3"
check "condition lines: seed 31337002, control=_hd29" "$(echo "$PLAN1" | grep -v 'cond=_hd29' | grep -c 'seed=31337002 .*control=_hd29')" "9"
check "primary: err for A/F/B, alt for D/C" "$(echo "$PLAN1" | grep -v 'cond=_hd29' | plan_field primary | tr '\n' ' ' | sed 's/ $//')" "err err err err err err alt alt alt"
check "all 12 are todo (axis C flies by default — §5.1.4a)" "$(echo "$PLAN1" | grep -c 'status=todo')" "12"
check "axis C carries its caps: sn4 open, sn8 4 M" "$(echo "$PLAN1" | grep 'axis=C' | sed -E 's/.*(--max-cells [0-9]+).*/\1/' | tr '\n' ';')" "--max-cells 1000000000;--max-cells 4000000;"
check "condition flags are the plain last-wins stores" \
	"$(echo "$PLAN1" | grep -v 'cond=_hd29' | sed -E 's/.*flags="([^"]*) --report-seeds.*/\1/' | tr '\n' ';')" \
	"--disturbance L4A;--disturbance L4B;--motor-lag-s 0.15;--motor-lag-s 0.30;--teacher pid;--teacher lqr;--airframe cf2x_firmware;--grid-state-neurons 4 --max-state-neurons 4 --max-cells 1000000000;--grid-state-neurons 8 --max-state-neurons 8 --max-cells 4000000;"
check "axis F flags are SETTLING TIMES (T=0.15/0.30), never the tau values" "$(echo "$PLAN1" | grep 'axis=F' | grep -c -- '--motor-lag-s 0.\(15\|30\) ')/$(echo "$PLAN1" | grep -c -- '--motor-lag-s 0.0\(375\|75\)')" "2/0"
check "MA_ANCHOR_SUFFIX=_hd30 switches anchor + control in one edit" "$(MA_ANCHOR_SUFFIX=_hd30 chain --dry-run --round 1 | grep '^PLAN r1 ' | grep -c 'cond=_hd30 .*control=NONE\|control=_hd30 ')" "12"
check "MA_REFUSE_AXIS_C=1 -> axis C REFUSED, others todo" "$(MA_REFUSE_AXIS_C=1 chain --dry-run --round 1 | grep -c 'axis=C .*status=REFUSED(MA_REFUSE_AXIS_C=1)')/$(MA_REFUSE_AXIS_C=1 chain --dry-run --round 1 | grep -c 'status=todo')" "2/10"
check "dry-run wrote no log, no lock, no marker" "$([ ! -e "$TD/chain.log" ] && [ ! -e "$TD/lock" ] && [ "$(ls "$SR/experiments/sweepladder_markers" | wc -l | tr -d ' ')" = 4 ] && echo clean || echo DIRTY)" "clean"
cp "$SR/experiments/sweepladder_markers/SL_C_b24n256_cf21_brushless_L4C_g10_s31337002_hd29.json" "$SR/experiments/sweepladder_markers/SL_C_b24n256_cf21_brushless_L4C_g10_s31337002_axA_L4A.json"
check "a banked marker shows status=banked (idempotent plan)" "$(chain --dry-run --round 1 | grep 'cond=_axA_L4A' | plan_field status)" "banked"
rm "$SR/experiments/sweepladder_markers/SL_C_b24n256_cf21_brushless_L4C_g10_s31337002_axA_L4A.json"

echo
echo "=== PLAN: rounds 2-4 and the whole programme ==="
for r in 2 3 4; do
	P="$(chain --dry-run --round $r | grep "^PLAN r$r ")"
	check "round $r: 9 lines, no anchor, seed $((31337001 + r))" "$(echo "$P" | grep -c .)/$(echo "$P" | grep -c 'axis=anchor')/$(echo "$P" | plan_field seed | sort -u | tr -d '\n')" "9/0/$((31337001 + r))"
done
check "all rounds: 39 PLAN lines (12 + 3 x 9)" "$(chain --dry-run | grep -c '^PLAN r')" "39"
check "--plan-only-json: valid JSON, 12 entries, D5 in every flags" \
	"$(chain --plan-only-json --round 1 | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d), sum(1 for x in d if '$D5' in x['flags']), d[0]['control'], d[3]['control'])")" "12 12 NONE _hd29"
check "--round 5 is a usage error (exit 3)" "$(chain --dry-run --round 5 >/dev/null 2>&1; echo $?)" "3"

echo
echo "=== RUN: skew guard aborts before the first launch ==="
: > "$CALLS"
OUT="$(STUB_HELP="--disturbance --teacher --airframe --grid-state-neurons --max-state-neurons --max-cells --report-seeds --teacher-hover" chain --round 1 2>&1)"; rc=$?
check "exit 1 naming the missing flag" "$rc/$(echo "$OUT" | grep -c 'ABORT — phased_ga has no --motor-lag-s')" "1/1"
check "zero seed_arm_chain calls" "$(grep -c . "$CALLS")" "0"
check "lock released after abort" "$([ -e "$TD/lock" ] && echo held || echo released)" "released"

echo
echo "=== RUN: a missing paired control aborts before the first launch ==="
mv "$SR/experiments/sweepladder_markers/SL_C_b24n256_cf21_brushless_L4C_g10_s31337002_hd29.json" "$TD/keep.json"
OUT="$(chain --round 1 2>&1)"; rc=$?
check "exit 1 naming the control" "$rc/$(echo "$OUT" | grep -c 'control markers missing: SL_C_b24n256_cf21_brushless_L4C_g10_s31337002_hd29')" "1/1"
check "zero seed_arm_chain calls" "$(grep -c . "$CALLS")" "0"
mv "$TD/keep.json" "$SR/experiments/sweepladder_markers/SL_C_b24n256_cf21_brushless_L4C_g10_s31337002_hd29.json"
: > "$CALLS"
OUT="$(MA_ANCHOR_SUFFIX=_hd30 chain --round 1 2>&1)"; rc=$?
check "MA_ANCHOR_SUFFIX=_hd30 with no _hd30 markers -> abort, 0 calls" "$rc/$(echo "$OUT" | grep -c 'control markers missing: .*_s31337002_hd30')/$(grep -c . "$CALLS")" "1/1/0"

echo
echo "=== RUN: happy round 1 — all 12 launches in plan order, ARM_* contract, exit 0 ==="
: > "$CALLS"; rm -f "$TD/chain.log"
OUT="$(chain --round 1 2>&1)"; rc=$?
check "exit 0 (every planned launch banked)" "$rc" "0"
check "12 seed_arm_chain calls" "$(grep -c . "$CALLS")" "12"
check "call order = plan order" "$(cut -d'|' -f1 "$CALLS" | tr '\n' ' ' | sed 's/ $//')" "$WANT_ORDER"
check "one seed per call; anchors 7,8,9 then 31337002 x9" "$(cut -d'|' -f2 "$CALLS" | tr '\n' ' ' | sed 's/ $//')" "31337007 31337008 31337009 31337002 31337002 31337002 31337002 31337002 31337002 31337002 31337002 31337002"
check "anchors ARM_NO_CONTROL=1, conditions ARM_CTRL_SUFFIX=_hd29" "$(grep -c '^_hd29|[0-9]*|1|' "$CALLS")/$(grep -v '^_hd29' "$CALLS" | grep -c '|0|_hd29|')" "3/9"
check "every ARM_EXTRA_ARGS carries the D5 report seeds" "$(grep -c -- "$D5" "$CALLS")" "12"
check "F markers record motor_lag_settling_s and derived tau_s" "$(grep '^_axF_T015|' "$CALLS" | grep -c '"motor_lag_settling_s":0.15,"tau_s":0.0375')/$(grep '^_axF_T030|' "$CALLS" | grep -c '"motor_lag_settling_s":0.30,"tau_s":0.075')" "1/1"
check "F flags reach ARM_EXTRA_ARGS as settling times" "$(grep '^_axF_T015|' "$CALLS" | grep -c -- '|--motor-lag-s 0.15 --report-seeds')/$(grep '^_axF_T030|' "$CALLS" | grep -c -- '|--motor-lag-s 0.30 --report-seeds')" "1/1"
check "C flags + caps reach ARM_EXTRA_ARGS" "$(grep '^_axC_sn4|' "$CALLS" | grep -c -- '|--grid-state-neurons 4 --max-state-neurons 4 --max-cells 1000000000 --report-seeds')/$(grep '^_axC_sn8|' "$CALLS" | grep -c -- '|--grid-state-neurons 8 --max-state-neurons 8 --max-cells 4000000 --report-seeds')" "1/1"
check "C markers record max_cells" "$(grep '^_axC_sn4|' "$CALLS" | grep -c '"max_cells":1000000000')/$(grep '^_axC_sn8|' "$CALLS" | grep -c '"max_cells":4000000')" "1/1"
check "non-C markers carry no max_cells field" "$(grep -v '^_axC_' "$CALLS" | grep -c 'max_cells')" "0"
check "condition flags reach ARM_EXTRA_ARGS (e.g. --teacher lqr)" "$(grep '^_axB_lqr|' "$CALLS" | grep -c -- '|--teacher lqr --report-seeds')" "1"
check "ARM_MARKER_JSON records axis/condition/round/report_seed_set" "$(grep '^_axD_cf2xfw|' "$CALLS" | grep -c '"programme":"multi-axis","axis":"D","condition":"cf2x_firmware","round":1,"primary":"alt","report_seed_set":"D5"')" "1"
check "no REFUSED line" "$(grep -c 'REFUSED — ' "$TD/chain.log")" "0"
check "12 markers banked (stub) + 4 controls" "$(ls "$SR/experiments/sweepladder_markers" | wc -l | tr -d ' ')" "16"
check "banked C marker is valid JSON with max_cells" "$(python3 -c "import json;print(json.load(open('$SR/experiments/sweepladder_markers/SL_C_b24n256_cf21_brushless_L4C_g10_s31337002_axC_sn8.json'))['max_cells'])" 2>/dev/null)" "4000000"
check "round verdict ran the paired robustness line per banked condition" "$(grep -c 'ROBUSTNESS line' "$TD/chain.log")" "9"
check "round 1 labelled DIRECTION only" "$(grep -c 'ROUND 1 = n=1 per condition: DIRECTION only' "$TD/chain.log")" "1"
check "welch path taken when --help advertises --welch" "$(grep -c 'WELCH (PRIMARY, R4)' "$TD/chain.log")" "9"
check "lock released on exit" "$([ -e "$TD/lock" ] && echo held || echo released)" "released"

echo
echo "=== RUN: relaunch is idempotent; MA_REFUSE_AXIS_C=1 refuses the 2 C runs -> exit 2 ==="
: > "$CALLS"
chain --round 1 >/dev/null 2>&1; rc=$?
check "fully banked round: 0 calls, exit 0" "$(grep -c . "$CALLS")/$rc" "0/0"
rm -f "$SR/experiments/sweepladder_markers/"*_axC_sn[48].json
: > "$CALLS"; rm -f "$TD/chain.log"
MA_REFUSE_AXIS_C=1 chain --round 1 >/dev/null 2>&1; rc=$?
check "escape hatch: 0 calls (others banked), 2 REFUSED lines, exit 2" "$(grep -c . "$CALLS")/$(grep -c 'REFUSED — .*escape hatch MA_REFUSE_AXIS_C=1' "$TD/chain.log")/$rc" "0/2/2"
: > "$CALLS"
chain --round 1 >/dev/null 2>&1; rc=$?
check "hatch released: exactly the 2 C calls, exit 0" "$(cut -d'|' -f1 "$CALLS" | tr '\n' ' ' | sed 's/ $//')/$rc" "_axC_sn4 _axC_sn8/0"

echo
echo "=== RUN: no --welch in --help -> TODO line, chain continues ==="
STUB_HELP="--disturbance --motor-lag-s --teacher --airframe --grid-state-neurons --max-state-neurons --max-cells --report-seeds --teacher-hover" \
	chain --round 1 > "$TD/nowelch.log" 2>&1; rc=$?
check "exit 0 and a TODO per condition" "$rc/$(grep -c 'TODO — paired_power.py has no --welch yet' "$TD/nowelch.log")" "0/9"

echo
echo "=== RUN: fail closed — a run that banks no marker stops the chain at THAT launch ==="
: > "$CALLS"
OUT="$(STUB_NO_MARKER=1 chain --round 2 2>&1)"; rc=$?
check "exit 1 with the MISSING-marker abort" "$rc/$(echo "$OUT" | grep -c 'ABORT — marker SL_C_b24n256_cf21_brushless_L4C_g10_s31337003_axA_L4A.json MISSING')" "1/1"
check "exactly one launch attempted (the first of round 2)" "$(grep -c . "$CALLS")" "1"
check "no round-2 marker was written" "$(ls "$SR/experiments/sweepladder_markers" | grep -c 's31337003_ax')" "0"

echo
echo "=== RUN: lock held by a live pid / busy box -> abort, nothing launched ==="
: > "$CALLS"
mkdir -p "$TD/lock"; echo $$ > "$TD/lock/pid"
OUT="$(chain --round 2 2>&1)"; rc=$?
check "live lock -> exit 1, 0 calls" "$rc/$(echo "$OUT" | grep -c 'holds')/$(grep -c . "$CALLS")" "1/1/0"
rm -rf "$TD/lock"
mkdir -p "$TD/lock"; echo 2147483000 > "$TD/lock/pid"
OUT="$(FAKE_BUSY=1 chain --round 2 2>&1)"; rc=$?
check "stale lock reclaimed, then busy box -> exit 1, 0 calls" "$rc/$(echo "$OUT" | grep -c 'stale lock')/$(echo "$OUT" | grep -c 'NOT idle')/$(grep -c . "$CALLS")" "1/1/1/0"
check "lock released after the busy abort" "$([ -e "$TD/lock" ] && echo held || echo released)" "released"

echo
echo "=== RUN: HOLD sentinel delays the launch at the chain level ==="
: > "$CALLS"; HOLD="$TD/HOLD"; touch "$HOLD"
( sleep 3; rm -f "$HOLD" ) &
t0=$SECONDS
HOLD_FILE="$HOLD" chain --round 2 > "$TD/hold.log" 2>&1; rc=$?
wait
check "waited >= 2 s, logged HOLD, then flew 9 and exited 0" "$([ $((SECONDS - t0)) -ge 2 ] && echo waited || echo no-wait)/$(grep -c 'HOLD — ' "$TD/hold.log" | awk '{print ($1>0)?"logged":"silent"}')/$(grep -c . "$CALLS")/$rc" "waited/logged/9/0"

echo
echo "=== production PIDs untouched ==="
check "same live controller/ladder/chain PIDs before and after" "$([ "$PROD_BEFORE" = "$(prod_pids)" ] && echo same || echo CHANGED)" "same"
check "real /private/tmp/multi_axis_chain.log not written by the harness" "$([ -e /private/tmp/multi_axis_chain.log ] && [ "$(find /private/tmp/multi_axis_chain.log -newer "$SR/bin/pgrep" | wc -l | tr -d ' ')" = 1 ] && echo TOUCHED || echo untouched)" "untouched"

echo
if [ "$FAILS" -gt 0 ]; then echo "FAILED ($FAILS)"; exit 1; fi
echo "ALL PASS — the multi-axis chain plans round-major in §5.1.5 order, carries D5 everywhere, and fails closed"
