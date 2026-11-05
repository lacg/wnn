#!/usr/bin/env bash
# multi_axis_chain.sh (draft-4 rewrite, 05/11/2026) — proof harness. NOTHING REAL IS LAUNCHED: the
# chain runs in a scratch root with a STUB python (fake phased_ga that writes a canned .out + winner,
# canned --help, canned wheel identity; the REAL headline_compare.py/paired_power.py run under the
# system python3), a fake pgrep (box idle) and scratch log/lock/CTRL-8 dirs. Production PIDs are
# counted before and after with the REAL pgrep — the chain has no kill path, and that is asserted.
#
# Proven:
#   PLAN   --dry-run round 1 = 9 lines in A,F,B,D,C order; C REFUSED until MA_AXIS_C_MAX_CELLS;
#          sn8 REFUSED until SN8_APPROVED; dry-run writes nothing; --smoke-list = anchor + 9 lines.
#   RUN    no MA_ANCHOR -> abort; wrong wheel -> abort before any launch; happy round 1 banks 8
#          (sn8 refused -> exit 2) and passes the recipe + condition flags + the C cap; relaunch is
#          idempotent; a failed run aborts at THAT launch and counts an attempt; 3 attempts DEFER it;
#          a provenance mismatch is not banked; HOLD delays a launch; the n=3 interim stop fires on a
#          clearly-better condition (STOPPED, seeds 4-5 skipped) while the missing anchor s5 is flown.
# Run: bash tests/multi_axis_chain_dryrun.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REAL_PGREP="$(command -v pgrep)"
prod_pids() { "$REAL_PGREP" -f "wnn.control.phased_g[a]|sweep_ladder_gamm[a]|seed_arm_chai[n]|queue_[0-9]|ctrl8_[a-z0-9_]*\.s[h]" | sort | tr '\n' ' '; }
PROD_BEFORE="$(prod_pids)"
TD=$(mktemp -d)
trap 'rm -rf "$TD"' EXIT
FAILS=0
check() {
	if [ "$2" = "$3" ]; then printf '  ok   %-66s -> %s\n' "$1" "$2"
	else printf '  FAIL %-66s -> %s (expected %s)\n' "$1" "$2" "$3"; FAILS=$((FAILS + 1)); fi
}

SR="$TD/root"; mkdir -p "$SR/scripts" "$SR/experiments" "$SR/bin" "$SR/c8" "$SR/sw"
cp "$ROOT/scripts/multi_axis_chain.sh" "$ROOT/scripts/controller_arm_lib.sh" \
   "$ROOT/scripts/headline_compare.py" "$ROOT/scripts/paired_power.py" "$SR/scripts/"
printf '#!/bin/sh\nexit 1\n' > "$SR/bin/pgrep"; chmod +x "$SR/bin/pgrep"
ARGV0='--levels 16 --pop 50 --teacher mpcof --report-seeds 99990101 99990102 99990103 99990104 99990105'
headline() {   # <stable> <err> <steady>
	printf '  [stage-select] HEADLINE held-out: stable=%s%% err=%s°  steady=%s°  jerk=0.02  mono_viol=1  alt=0.180m  pos=0.420m  dx=0.2m  dy=0.2m  effort=1.9\n' "$1" "$2" "$3"
}
fake_anchor() {   # <file> <seed> <jitter>
	{ echo "[argv] $ARGV0 --base-seed $2 --save-stage-checkpoints x --save-winner y"
	  echo "[provenance] wheel=w abi=33 wheel_sha256=a6398538ce04588c git=g"
	  headline "90.$3" "3.0$3" "3.0$3"; } > "$1"
}
fake_anchor "$SR/sw/arm_kd078.out" 31337002 1
fake_anchor "$SR/c8/arm_s3.out" 31337003 5
fake_anchor "$SR/c8/arm_s4.out" 31337004 3

# STUB python: provenance / --help / phased_ga run / json / real python3 for the comparison tools.
cat > "$SR/bin/stubpy" <<'EOF'
#!/usr/bin/env bash
case "$1" in
	-c)
		case "$2" in
			*collect_provenance*) echo "${STUB_WHEEL:-33 a6398538ce04588c}"; exit 0 ;;
			*) exec python3 "$@" ;;
		esac ;;
	scripts/*) exec python3 "$@" ;;
esac
[ "$1" = "-u" ] && shift
[ "$1 $2" = "-m wnn.control.phased_ga" ] || { echo "stub: unexpected $*" >&2; exit 9; }
shift 2
if [ "${1:-}" = "--help" ]; then echo "--disturbance --motor-lag-s --teacher --airframe --rotor-drag --grid-state-neurons --max-state-neurons --max-cells"; exit 0; fi
seed=""; win=""; prev=""
for w in "$@"; do
	[ "$prev" = "--base-seed" ] && seed="$w"
	[ "$prev" = "--save-winner" ] && win="$w"
	prev="$w"
done
tag="$(basename "$win" _winner.yaml.gz)"
echo "$tag $*" >> "$STUB_CALLS"
echo "[argv] $*"
if [ "${STUB_BAD_PROV:-}" = "$tag" ]; then echo "[provenance] wheel=w abi=32 wheel_sha256=deadbeef git=g"
else echo "[provenance] wheel=w abi=33 wheel_sha256=a6398538ce04588c git=g"; fi
[ "${STUB_FAIL:-}" = "$tag" ] && { echo "boom"; exit 7; }
j=$(( ${seed: -1} % 3 ))
case "$tag" in
	ma_anchor_*) printf '  [stage-select] HEADLINE held-out: stable=90.%s%% err=3.0%s°  steady=3.0%s°  alt=0.180m  pos=0.420m\n' $j $j $j ;;
	*) printf '  [stage-select] HEADLINE held-out: stable=99.%s%% err=1.0%s°  steady=0.5%s°  alt=0.170m  pos=0.400m\n' $j $j $j ;;
esac
echo w > "$win"
exit 0
EOF
chmod +x "$SR/bin/stubpy"
export PATH="$SR/bin:$PATH" MA_PYTHON="$SR/bin/stubpy" MA_LOG="$TD/chain.log" MA_LOCK="$TD/lock" MA_TIME="" \
       MA_CTRL8_DIR="$SR/c8" MA_SWEEP_DIR="$SR/sw" STUB_CALLS="$TD/calls" WNN_HOLD_POLL_S=1
: > "$STUB_CALLS"
C="$SR/scripts/multi_axis_chain.sh"
launches() { wc -l < "$STUB_CALLS" | tr -d ' '; }

echo "== PLAN"
DRY="$(bash "$C" --dry-run --round 1)"
check "round-1 plan lines" "$(echo "$DRY" | grep -c '^PLAN r1')" 9
check "order A,A,F,F,B,B,D,C,C" "$(echo "$DRY" | grep '^PLAN' | sed -E 's/.*axis=([A-Z]+).*/\1/' | tr -d '\n')" "AAFFBBDCC"
check "C refused without MA_AXIS_C_MAX_CELLS" "$(echo "$DRY" | grep 'axC_' | grep -c 'status=REFUSED')" 2
check "cf2x flies at k_d 0.25" "$(echo "$DRY" | grep -c 'cf2x_firmware --rotor-drag 0.25')" 1
DRY2="$(MA_AXIS_C_MAX_CELLS=123 bash "$C" --dry-run --round 1)"
check "sn4 todo + cap when MA_AXIS_C_MAX_CELLS set" "$(echo "$DRY2" | grep axC_sn4 | grep -c 'status=todo.*--max-cells 123')" 1
check "sn8 still refused (no SN8_APPROVED)" "$(echo "$DRY2" | grep axC_sn8 | grep -c 'SN8_APPROVED')" 1
check "dry-run wrote nothing" "$(ls "$SR/experiments" | wc -l | tr -d ' ')$(launches)" "00"
check "smoke list = anchor + 9" "$(bash "$C" --smoke-list | grep -c '^cd ')" 10

echo "== RUN guards"
bash "$C" --round 1 >/dev/null 2>&1; check "no MA_ANCHOR -> exit 1" "$?" 1
STUB_WHEEL="32 deadbeefdeadbeef" MA_ANCHOR=arm bash "$C" --round 1 >/dev/null 2>&1
check "wrong wheel -> exit 1, nothing launched" "$?/$(launches)" "1/0"

echo "== RUN round 1"
MA_ANCHOR=arm MA_AXIS_C_MAX_CELLS=1000000000 bash "$C" --round 1 >/dev/null 2>&1
check "round 1 exit 2 (sn8 refused)" "$?" 2
check "8 launches" "$(launches)" 8
check "8 markers" "$(ls "$SR/experiments/multi_axis_markers"/*.json | wc -l | tr -d ' ')" 8
check "recipe passed verbatim (base-seed stripped)" "$(grep -c -- "--levels 16 --pop 50 --teacher mpcof" "$STUB_CALLS")" 8
check "condition flag last + seed" "$(grep '^ma_axB_pid_s31337002' "$STUB_CALLS" | grep -c -- '--teacher pid --base-seed 31337002')" 1
check "sn4 carries the cap" "$(grep '^ma_axC_sn4' "$STUB_CALLS" | grep -c -- '--max-cells 1000000000')" 1
MA_ANCHOR=arm MA_AXIS_C_MAX_CELLS=1000000000 bash "$C" --round 1 >/dev/null 2>&1
check "relaunch idempotent (no new launch)" "$(launches)" 8

echo "== RUN failure / attempt-3 / provenance"
for i in 1 2 3; do STUB_FAIL=ma_axA_L4A_s31337003 MA_ANCHOR=arm MA_AXIS_C_MAX_CELLS=1 bash "$C" --round 2 >/dev/null 2>&1; done
check "failed run aborts at that launch (exit 1)" "$?" 1
check "attempt counter = 3" "$(cat "$SR/experiments/multi_axis_markers/ma_axA_L4A_s31337003.attempts")" 3
check "not banked" "$(ls "$SR/experiments/multi_axis_markers/ma_axA_L4A_s31337003.json" 2>/dev/null | wc -l | tr -d ' ')" 0
check "now DEFERRED in the plan" "$(MA_ANCHOR=arm bash "$C" --dry-run --round 2 | grep axA_L4A | grep -c 'DEFERRED')" 1
rm "$SR/experiments/multi_axis_markers/ma_axA_L4A_s31337003.attempts"
STUB_BAD_PROV=ma_axA_L4A_s31337003 MA_ANCHOR=arm MA_AXIS_C_MAX_CELLS=1 bash "$C" --round 2 >/dev/null 2>&1
check "provenance mismatch -> exit 1, not banked" "$?/$(ls "$SR/experiments/multi_axis_markers/ma_axA_L4A_s31337003.json" 2>/dev/null | wc -l | tr -d ' ')" "1/0"
rm "$SR/experiments/multi_axis_markers/ma_axA_L4A_s31337003.attempts"

echo "== HOLD"
touch "$SR/experiments/HOLD_CONTROLLER"
( sleep 3; rm -f "$SR/experiments/HOLD_CONTROLLER" ) &
MA_ANCHOR=arm MA_AXIS_C_MAX_CELLS=1 bash "$C" --round 2 >/dev/null 2>&1
check "round 2 completes after HOLD lifts (exit 2)" "$?" 2
check "HOLD logged" "$(grep -c 'HOLD released' "$MA_LOG")" 1

echo "== interim stop (n=3) + missing anchor s5"
MA_ANCHOR=arm MA_AXIS_C_MAX_CELLS=1 bash "$C" --round 3 >/dev/null 2>&1
check "round 3 exit 2" "$?" 2
check "L4A STOPPED at n=3 (clearly better on all three)" "$(ls "$SR/experiments/multi_axis_markers/axA_L4A.STOPPED" 2>/dev/null | wc -l | tr -d ' ')" 1
N0="$(launches)"
MA_ANCHOR=arm MA_AXIS_C_MAX_CELLS=1 bash "$C" --round 4 >/dev/null 2>&1
check "round 4 flies ONLY the missing anchor s5" "$(( $(launches) - N0 ))/$(tail -1 "$STUB_CALLS" | cut -d' ' -f1)" "1/ma_anchor_arm_s31337005"

check "production PIDs untouched" "$(prod_pids)" "$PROD_BEFORE"
echo "FAILS=$FAILS"
[ "$FAILS" = 0 ]
