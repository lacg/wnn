#!/usr/bin/env bash
# run_controller_arm's marker rules — the invariant that protects every study table.
#
# A marker is a CLAIM THAT A CELL GENUINELY FINISHED. build_dfa_1layer_table.py and
# the P2/P3 chains both treat marker-absence as "re-run me" and marker-presence as
# "this is a measurement". Write one for a run that did not finish and a hole enters
# the paper's table looking exactly like data.
#
# So the three refusal rules are tested here directly, with a stub standing in for
# the python so no controller is launched:
#   R1  rc=143/137 (watchdog took the box) -> no marker
#   R2  rc!=0 (crash)                      -> no marker, no retry
#   R3  rc=0 but empty MEMORY triple       -> no marker (truncated run)
# plus: the happy path writes valid JSON, and an existing marker is never rewritten.
#
# Run: bash tests/controller_arm_marker_rules.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
. "$ROOT/scripts/controller_arm_lib.sh"

TD=$(mktemp -d)
trap 'rm -rf "$TD"' EXIT
MARKDIR="$TD/markers"; OUTDIR="$TD/out"
mkdir -p "$MARKDIR" "$OUTDIR"
FAILS=0
quiet() { :; }

# Stub "python": emits the canned body in $STUB_BODY, exits with $STUB_RC.
STUB="$TD/fakepy"
cat > "$STUB" <<'EOS'
#!/usr/bin/env bash
printf '%s\n' "$STUB_BODY"
exit "$STUB_RC"
EOS
chmod +x "$STUB"

GOOD_BODY='  RESULT — during-search winner (held-out): stable=99.0% err=2.27° steady=1.78°
  RESULT — during-search winner (held-out): stable=100.0% err=2.33° steady=1.91°
cells[80160-277005 Σ9653k μ193k]'

check() {
	local label="$1" got="$2" want="$3"
	if [ "$got" = "$want" ]; then
		printf '  ok   %-52s -> %s\n' "$label" "$got"
	else
		printf '  FAIL %-52s -> %s (expected %s)\n' "$label" "$got" "$want"
		FAILS=$((FAILS + 1))
	fi
}
has_marker() { [ -f "$MARKDIR/$1.json" ] && echo yes || echo no; }

echo
echo "=== R1: watchdog stop must not claim a finished cell ==="
for rc in 143 137; do
	STUB_BODY="$GOOD_BODY" STUB_RC=$rc \
		run_controller_arm "wd$rc" "$MARKDIR" "$OUTDIR" "$STUB" quiet "" -- --x >/dev/null 2>&1
	check "rc=$rc (watchdog) writes no marker" "$(has_marker "wd$rc")" "no"
done

echo
echo "=== R2: a crash must not claim a finished cell ==="
STUB_BODY="$GOOD_BODY" STUB_RC=1 \
	run_controller_arm "crash" "$MARKDIR" "$OUTDIR" "$STUB" quiet "" -- --x >/dev/null 2>&1
check "rc=1 (crash) writes no marker" "$(has_marker crash)" "no"

echo
echo "=== R3: clean exit with no MEMORY triple is a truncated run ==="
STUB_BODY="  RESULT — during-search winner (held-out): stable=99.0% err=2.27° steady=1.78°" STUB_RC=0 \
	run_controller_arm "trunc" "$MARKDIR" "$OUTDIR" "$STUB" quiet "" -- --x >/dev/null 2>&1
check "rc=0, only a NEURONS triple -> no marker" "$(has_marker trunc)" "no"
STUB_BODY="no results at all" STUB_RC=0 \
	run_controller_arm "empty" "$MARKDIR" "$OUTDIR" "$STUB" quiet "" -- --x >/dev/null 2>&1
check "rc=0, no triples at all -> no marker" "$(has_marker empty)" "no"

echo
echo "=== happy path writes a valid, complete marker ==="
STUB_BODY="$GOOD_BODY" STUB_RC=0 \
	run_controller_arm "good" "$MARKDIR" "$OUTDIR" "$STUB" quiet '"arm":"P3","corner":"state","state_neurons":8' -- --x >/dev/null 2>&1
check "rc=0 with a MEMORY triple -> marker" "$(has_marker good)" "yes"
check "marker is valid JSON" \
	"$(python3 -c "import json;json.load(open('$MARKDIR/good.json'));print('yes')" 2>/dev/null || echo no)" "yes"
check "extra fields spliced in" \
	"$(python3 -c "import json;d=json.load(open('$MARKDIR/good.json'));print(d.get('corner'))" 2>/dev/null)" "state"
check "held_memory is the SECOND triple" \
	"$(python3 -c "import json;d=json.load(open('$MARKDIR/good.json'));print('100.0%' in d['held_memory'])" 2>/dev/null)" "True"
check "cells captured" \
	"$(python3 -c "import json;d=json.load(open('$MARKDIR/good.json'));print(d['cells'].startswith('cells[80160'))" 2>/dev/null)" "True"

echo
echo "=== an existing marker is never rewritten (idempotent resume) ==="
before=$(cat "$MARKDIR/good.json")
STUB_BODY="$GOOD_BODY" STUB_RC=0 \
	run_controller_arm "good" "$MARKDIR" "$OUTDIR" "$STUB" quiet "" -- --x >/dev/null 2>&1
check "second call leaves the marker byte-identical" \
	"$([ "$before" = "$(cat "$MARKDIR/good.json")" ] && echo same || echo CHANGED)" "same"

echo
if [ "$FAILS" -gt 0 ]; then
	echo "FAILED ($FAILS)"
	exit 1
fi
echo "ALL PASS — a marker is written only for a run that genuinely finished"
