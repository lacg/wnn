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

# The body carries the STAGE headers the helper anchors on (04/08/2026 fix): the
# NEURONS triple follows "STAGE 1 (NEURONS) done", the MEMORY triple follows
# "STAGE 4 (MEMORY) done". Before 11/09/2026 this body had no headers, so every
# happy-path check here had been failing silently since that fix landed.
GOOD_BODY='  STAGE 1 (NEURONS) done: gen 5/5
  RESULT — during-search winner (held-out): stable=99.0% err=2.27° steady=1.78°
  STAGE 4 (MEMORY) done: gen 6/120
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
# The retry's memory-calm wait is zeroed (it is 3x60 s at >=25 GB in production,
# and used to make this section hang on a busy box).
export WNN_RETRY_POLL_S=0 WNN_RETRY_CALM_GB=0
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
STUB_BODY="  STAGE 1 (NEURONS) done: gen 5/5
  RESULT — during-search winner (held-out): stable=99.0% err=2.27° steady=1.78°" STUB_RC=0 \
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
echo "=== R9: provenance is copied from the [provenance] line, null when absent ==="
check "a .out WITHOUT the line banks provenance=null (pre-R9 run, visible absence)" \
	"$(python3 -c "import json;d=json.load(open('$MARKDIR/good.json'));print(d['provenance'])" 2>/dev/null)" "None"
PROV_BODY="Pop=50 fitness_pools=CRN(all 5 pools/gen)
[provenance] wheel=ram_controller-2026.212.37 abi=27 wheel_sha256=08cdd0462eac0d15 git=3b5fc37d+dirty fitness_pools=CRN(all 5 pools/gen)
$GOOD_BODY"
STUB_BODY="$PROV_BODY" STUB_RC=0 \
	run_controller_arm "prov" "$MARKDIR" "$OUTDIR" "$STUB" quiet "" -- --x >/dev/null 2>&1
check "marker with the line is valid JSON" \
	"$(python3 -c "import json;json.load(open('$MARKDIR/prov.json'));print('yes')" 2>/dev/null || echo no)" "yes"
check "provenance.wheel" \
	"$(python3 -c "import json;print(json.load(open('$MARKDIR/prov.json'))['provenance']['wheel'])" 2>/dev/null)" "ram_controller-2026.212.37"
check "provenance.abi is an INTEGER" \
	"$(python3 -c "import json;print(repr(json.load(open('$MARKDIR/prov.json'))['provenance']['abi']))" 2>/dev/null)" "27"
check "provenance.wheel_sha256" \
	"$(python3 -c "import json;print(json.load(open('$MARKDIR/prov.json'))['provenance']['wheel_sha256'])" 2>/dev/null)" "08cdd0462eac0d15"
check "provenance.git keeps the +dirty suffix" \
	"$(python3 -c "import json;print(json.load(open('$MARKDIR/prov.json'))['provenance']['git'])" 2>/dev/null)" "3b5fc37d+dirty"
check "provenance.fitness_pools keeps its spaces and parens" \
	"$(python3 -c "import json;print(json.load(open('$MARKDIR/prov.json'))['provenance']['fitness_pools'])" 2>/dev/null)" "CRN(all 5 pools/gen)"

echo
echo "=== HOLD sentinel: no launch while the file exists, resume when it is removed ==="
HOLD="$TD/HOLD"; touch "$HOLD"
( sleep 3; rm -f "$HOLD" ) &
t0=$SECONDS
WNN_HOLD_FILE="$HOLD" WNN_HOLD_POLL_S=1 STUB_BODY="$GOOD_BODY" STUB_RC=0 \
	run_controller_arm "held" "$MARKDIR" "$OUTDIR" "$STUB" quiet "" -- --x >/dev/null 2>&1
wait
check "launch waited for the sentinel (>=2 s)" "$([ $((SECONDS - t0)) -ge 2 ] && echo waited || echo did-not-wait)" "waited"
check "then banked the marker" "$(has_marker held)" "yes"
check "no sentinel -> no wait" "$(WNN_HOLD_FILE="$TD/absent" wait_while_held quiet x; echo rc=$?)" "rc=0"

echo
echo "=== watchdog retry RESUMES from the right point (stage, not mtime) ==="
# A stub that is killed (143) on its first flight and records the retry's argv.
# The resume picker is real python: the stub hands it straight through.
RSTUB="$TD/resumepy"
cat > "$RSTUB" <<'EOS'
#!/usr/bin/env bash
if [ "${2:-}" = "scripts/pick_resume_point.py" ]; then exec "$PICK_PY" "$2" "$3"; fi
n=$(cat "$FLIGHTS" 2>/dev/null || echo 0); echo $((n + 1)) > "$FLIGHTS"
if [ "$n" = "0" ]; then exit 143; fi
printf '%s\n' "$*" > "$ARGV_OUT"
printf '%s\n' "$STUB_BODY"
EOS
chmod +x "$RSTUB"
retry_argv() {   # retry_argv <case> -> the retry's "--resume-from-emergency X --resume-mode M"
	local ck="$TD/ck_$1"
	FLIGHTS="$TD/fl_$1" ARGV_OUT="$TD/argv_$1" STUB_BODY="$GOOD_BODY" \
		run_controller_arm "rs_$1" "$MARKDIR" "$OUTDIR" "$RSTUB" quiet "" -- --save-stage-checkpoints "$ck" >/dev/null 2>&1
	sed -nE 's/.*--resume-from-emergency ([^ ]+) --resume-mode ([a-z]+).*/\1 \2/p' "$TD/argv_$1" \
		| sed "s|$ck/||"
}
# The picker reads checkpoint HEADS (completion marker / metrics), so it needs a
# python with PyYAML — the project venv when present.
PICK_PY=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python
[ -x "$PICK_PY" ] || PICK_PY=python3
export FLIGHTS ARGV_OUT PICK_PY
# mk <case> <file> <touch-time> [extra-mapping]: a gz whose head carries `extra`.
# Default: a FINISHED final (stage_complete: true) or an emergency dump, by name.
mk() {
	local extra="${4:-}"
	if [ -z "$extra" ]; then
		case "$2" in
			emergency_*) extra="{emergency_dump: true}" ;;
			*) extra="{stage_complete: true}" ;;
		esac
	fi
	mkdir -p "$TD/ck_$1"
	printf '{schema: 2, phase_key: x, extra: %s, saved_at: 1.0, best_genome: {type: x}, final_population: []}\n' \
		"$extra" | gzip > "$TD/ck_$1/$2"
	touch -t "$3" "$TD/ck_$1/$2"
}
# Case A — the stale-dump trap: stage 1 finished AFTER its dump, stage 4 never dumped.
mk A emergency_stage1_neurons.yaml.gz 202610130449
mk A stage0_grid.yaml.gz 202610122125
mk A stage1_neurons.yaml.gz 202610130651
check "stale stage-1 dump ignored -> stage-1 FINAL, mode next" "$(retry_argv A)" "stage1_neurons.yaml.gz next"
# Case B — the _tw20 layout: stage 4 has a live dump (newer than any final).
mk B emergency_stage1_neurons.yaml.gz 202610130449
mk B stage0_grid.yaml.gz 202610122125
mk B stage1_neurons.yaml.gz 202610130651
mk B emergency_stage4_memory.yaml.gz 202610131744
check "live stage-4 dump -> resume it, mode same" "$(retry_argv B)" "emergency_stage4_memory.yaml.gz same"
# Case C — killed after MEMORY's final save: publish-only resume from it.
mk C stage1_neurons.yaml.gz 202610130651
mk C emergency_stage4_memory.yaml.gz 202610131744
mk C stage4_memory.yaml.gz 202610131753
check "stage-4 final newer than its dump -> stage-4 FINAL, mode next" "$(retry_argv C)" "stage4_memory.yaml.gz next"
# Case D — the dump's stage is newer than a stale earlier dump in mtime order too.
mk D stage0_grid.yaml.gz 202610122125
mk D emergency_stage1_neurons.yaml.gz 202610130100
check "grid final + live stage-1 dump -> the dump, mode same" "$(retry_argv D)" "emergency_stage1_neurons.yaml.gz same"
# Case E — the 15/10 _tw515 s3 layout: MEMORY SIGTERMed at gen 2, its dump (00:18)
# followed by a pre-fix "final" (00:19) with all-zero metrics and no stage_complete.
# That is NOT a finished stage: resume the dump, mode same — never `next`.
ZERO="{metrics: {stable_rate: 0.0, mean_attitude_error_deg: 0.0, mean_steady_error_deg: 0.0}}"
mk E stage0_grid.yaml.gz 202610141806
mk E emergency_stage1_neurons.yaml.gz 202610142235
mk E stage1_neurons.yaml.gz 202610142339
mk E emergency_stage4_memory.yaml.gz 202610150018
mk E stage4_memory.yaml.gz 202610150019 "$ZERO"
check "_tw515 s3: zero-metric stage-4 'final' ignored -> dump, mode same" "$(retry_argv E)" "emergency_stage4_memory.yaml.gz same"
# Case F — same, but the final says so explicitly (stage_complete: false).
mk F stage1_neurons.yaml.gz 202610142339
mk F emergency_stage4_memory.yaml.gz 202610150018
mk F stage4_memory.yaml.gz 202610150019 "{stage_complete: false}"
check "stage_complete: false final ignored -> dump, mode same" "$(retry_argv F)" "emergency_stage4_memory.yaml.gz same"

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
