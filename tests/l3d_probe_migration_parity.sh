#!/usr/bin/env bash
# run_l3d_feature_probe.sh migration parity — the marker it writes must not change.
#
# The probe carried its own copy of run_arm; it now calls the shared
# run_controller_arm. 21 l3dfeat markers already exist on disk from the old code,
# and any analysis over that directory reads old and new markers side by side. So
# the migration is only safe if the JSON is IDENTICAL, field for field.
#
# This proves it without launching a controller: the OLD run_arm is lifted verbatim
# out of git history, the NEW one out of the working tree, and both are driven by
# the same stub standing in for python. The two markers are then compared with the
# `done` timestamp masked (it is wall-clock and differs by construction).
#
# Run: bash tests/l3d_probe_migration_parity.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

PROBE="scripts/run_l3d_feature_probe.sh"
# The commit that introduced the shared helper — its parent holds the pre-migration probe.
OLD_REF="${OLD_REF:-4a6b483d}"

TD=$(mktemp -d)
trap 'rm -rf "$TD"' EXIT

git show "${OLD_REF}:${PROBE}" > "$TD/probe_old.sh" 2>/dev/null || {
	echo "FAIL: cannot read ${OLD_REF}:${PROBE} from git"; exit 1; }

# --- the stub python: canned phased_ga output, rc=0 --------------------------
STUB="$TD/fakepy"
cat > "$STUB" <<'EOS'
#!/usr/bin/env bash
cat <<'BODY'
  RESULT — during-search winner (held-out): stable=99.0% err=2.27° steady=1.78° mono_viol=3
  RESULT — during-search winner (held-out): stable=100.0% err=2.33° steady=1.91° mono_viol=3
cells[80160-277005 Σ9653k μ193k]
BODY
exit 0
EOS
chmod +x "$STUB"

# gran_fpga_count.py is invoked on the winner file; stub it so both paths behave alike.
mkdir -p "$TD/scripts"
cat > "$TD/scripts/gran_fpga_count.py" <<'EOS'
print("[FPGA] mode=BINARY sn=0 sb=0 ob=30 populated=7556 x 1b/cell = 7556 bits (945 bytes)")
EOS

# --- harness: extract one run_arm, run it with a fixed environment -----------
# `which` selects old|new. Everything the function closes over is defined here so
# both sides see byte-identical inputs.
run_side() {
	local which="$1" src="$2" outroot="$TD/$1"
	mkdir -p "$outroot/markers" "$outroot/out"
	(
		cd "$TD" || exit 1
		MARKDIR="$outroot/markers"; OUTDIR="$outroot/out"
		VP="$STUB"
		COMMON="--levels 16"
		REPORT_SEEDS="99990101"
		FEAT_PIDMIX="--obs-peraxis-p"; FEAT_PIDMIX_PWM="x"; FEAT_PIDMIX_PWM_TILT="x"
		FEAT_PIDMIX_DECOUPLE="x"; FEAT_PIDMIX_TILT="x"; FEAT_10FEAT="x"
		log() { :; }
		export MARKDIR OUTDIR VP COMMON REPORT_SEEDS
		# the new probe sources the helper; provide it relative to the real root
		. "$ROOT/scripts/controller_arm_lib.sh"
		# lift ONLY the run_arm function out of the script (never execute its main loop)
		eval "$(awk '/^run_arm\(\) \{/,/^\}/' "$src")"
		run_arm A8 L2D pidmix 31337002
	)
	echo "$outroot/markers/A8_L2D_s31337002.json"
}

OLD_MARKER=$(run_side old "$TD/probe_old.sh")
NEW_MARKER=$(run_side new "$ROOT/$PROBE")

FAILS=0
say() { printf '  %-5s %s\n' "$1" "$2"; }

echo
echo "=== both sides produced a marker ==="
for m in "$OLD_MARKER" "$NEW_MARKER"; do
	if [ -f "$m" ]; then say ok "$(basename "$(dirname "$(dirname "$m")")") wrote a marker"
	else say FAIL "missing marker: $m"; FAILS=$((FAILS+1)); fi
done
[ "$FAILS" -gt 0 ] && { echo; echo "FAILED ($FAILS)"; exit 1; }

# `done` is wall-clock and `dur_s` can differ by a second — mask both, compare the rest.
mask() { sed -E 's/"done":"[^"]*"/"done":"MASKED"/; s/"dur_s":[0-9]+/"dur_s":MASKED/' "$1"; }

echo
echo "=== marker JSON is byte-identical (done/dur_s masked) ==="
if diff <(mask "$OLD_MARKER") <(mask "$NEW_MARKER") > "$TD/diff.txt"; then
	say ok "old and new markers match exactly"
else
	say FAIL "markers differ:"; sed 's/^/       /' "$TD/diff.txt"; FAILS=$((FAILS+1))
fi

echo
echo "=== every field the old schema promised is still present ==="
for f in tag arm disturbance features substrate mode seed rc dur_s peak_rss_bytes \
         cells fpga held_neurons held_memory fixed_thresholds done; do
	got=$(python3 -c "import json;print('$f' in json.load(open('$NEW_MARKER')))" 2>/dev/null)
	if [ "$got" = "True" ]; then say ok "field $f"
	else say FAIL "field $f MISSING"; FAILS=$((FAILS+1)); fi
done

echo
echo "=== field ORDER is unchanged too (real markers on disk are line-diffed) ==="
old_keys=$(python3 -c "import json,collections;print(','.join(json.load(open('$OLD_MARKER'),object_pairs_hook=collections.OrderedDict).keys()))")
new_keys=$(python3 -c "import json,collections;print(','.join(json.load(open('$NEW_MARKER'),object_pairs_hook=collections.OrderedDict).keys()))")
if [ "$old_keys" = "$new_keys" ]; then say ok "key order identical"
else say FAIL "key order changed: $old_keys -> $new_keys"; FAILS=$((FAILS+1)); fi

echo
if [ "$FAILS" -gt 0 ]; then echo "FAILED ($FAILS)"; exit 1; fi
echo "ALL PASS — the migrated probe writes the same marker as the code it replaced"
