#!/bin/bash
# CTRL-69 arm R (Luiz 22/11/2026 ~21:15 EST: prereg + land ABI 62 + fly R). Prereg: docs/ctrl69_rate_err_prereg.md.
#   1. waits for 'MF61 DONE' written to runs.log after this script's arm line (stands down on MF61 ABORT / STAND DOWN),
#      then for ab_mf61.sh and every phased_ga child to exit;
#   2. LANDS ABI 62: merge-tree check, branch pinned at aa2854d0, wheel sha256 pinned, git merge --no-ff ctrl62-rate-err,
#      pip install the wheel, assert ram_controller.ABI_VERSION == 62 and the facade's EXPECTED_ABI == 62, push.
#      Log 'R62: ABI 62 LANDED' or 'R62: REFUSED: why' (box left on ABI 61 when it refuses before/at install);
#   3. rival re-score on the installed lineage (MPCOF rows A/B/C via the cargo probe test + PID from the run-printed
#      rows) -> docs/ctrl61_rival_rows.md, log 'R62: RIVALS ...'; capped at 1 h, else deferred until after the flights;
#   4. flies R s1..s5 = A sN's own flown argv (rebuild_argv.py fly, A = ..._a61_..._L1_yld_n0_full_sN) + 5 flags,
#      tags ..._a62_..._L1_yld_n0rerr_full_sN, ABI 62 asserted before each flight. Ends 'R62 DONE' / 'R62 ABORT: why'.
# Every argv is checked by fixed text (R argv minus the 5 flags, with the tag swapped back, must EQUAL A's [argv] line)
# and through phased_ga's own resolver (_resolve_ctbr). Dry run: R62_DRY=1 prints the diff vs A sN and flies nothing.
cd /Users/lacg/wnn || exit 1
ROOT=/Users/lacg/wnn
L=logs/controller/ctrl48
D=logs/controller/ctrl62
RUNS=$L/runs.log
VENV=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv
ABI=62
SEEDS=${R62_SEEDS:-1 2 3 4 5}
DRY=${R62_DRY:-0}
PREREG=docs/ctrl69_rate_err_prereg.md
BRANCH=ctrl62-rate-err
BRANCH_SHA=aa2854d09fa23b54823e178c694144ada2704f39
WHEEL=$D/wheel_abi62/ram_controller-2026.212.37-cp311-abi3-macosx_11_0_arm64.whl
WHEEL_SHA=8bd004fe974e48aa335d54b7c5387101c4ac91c2bfcc5a7919d4023d5ff707af
PRE_SITE=$ROOT/$D/site_abi62:$ROOT/.claude/worktrees/ctrl62-rate-err/src   # pre-landing validation only
RFLAGS="--obs-rate-err --obs-rate-err-tau 0.05 --obs-rate-err-axes roll,pitch --obs-rate-i --obs-rate-i-axes yaw"
export CARGO_TARGET_DIR=/Volumes/20260401-WDBlack-SN850X-2TB/cargo-target
export PYO3_PYTHON=$VENV/bin/python
export PATH=/Users/lacg/.cargo/bin:/opt/homebrew/bin:$PATH
log() { echo "$(date -u +%FT%TZ) $*" >> $RUNS; }
abort() { [ "$DRY" = 1 ] || log "R62 ABORT: $1"; echo "R62 ABORT: $1" >&2; exit 1; }
refuse() { log "R62: REFUSED: $1"; echo "R62: REFUSED: $1" >&2; exit 1; }
row()
{
	local g r
	g=$(grep -a 'stage-select\] HEADLINE stage' $1 | grep -oE 'genome=[^ ]+' | cut -d= -f2)
	r=$(grep -a "HEADLINE-$g MULTI-SEED" $1 | tail -1)
	[ -z "$r" ] && r=$(grep -aE "\[(test|report)-seeds\] ${g%%#*} MULTI-SEED" $1 | tail -1)
	echo "genome=$g $(echo "$r" | grep -oE 'stable=[^ ]+  err=[^ ]+  steady=[^ ]+|alt=[^ ]+|pos=[^ ]+|pos_hold=[^ ]+|dz_hold=[^ ]+|stable_fail=[^ ]+' | tr '\n' ' ')"
}
src_of() { echo C-S2-d_wposx2_a61_rf_c200_dc_biasin_ms01_P_both_L1_yld_n0_full_s$1; }
tag_of() { echo C-S2-d_wposx2_a${ABI}_rf_c200_dc_biasin_ms01_P_both_L1_yld_n0rerr_full_s$1; }
count() { echo "$1" | grep -o -- "$2" | wc -l | tr -d ' '; }
ctrl_busy() { pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null; }
# validate(argv, pythonpath): phased_ga's own resolution chain (parse -> episode_config_from_args -> _resolve_ctbr)
validate()
{
	(cd /tmp && PYTHONPATH=$2 $VENV/bin/python -c "
import shlex, sys
from wnn.control import phased_ga as p
a = p.build_arg_parser().parse_args(shlex.split(sys.argv[1]))
ec = p.episode_config_from_args(a); p._resolve_ctbr(a, ec); c = a._ctbr
print('OK', a.delta_max_yaw, c.get('obs_rate_i'), c.get('obs_rate_i_axes'), c.get('obs_rate_err_tau'), c.get('obs_rate_err_axes'))" "$1" 2>&1 | tail -1)
}
# build(s, pythonpath) -> sets ARGV, or aborts with the reason
build()
{
	local s=$1 src tag aline back
	src=$(src_of $s); tag=$(tag_of $s)
	[ -f $L/$src.done ] || abort "missing finished A (ABI-61 N0) run $src"
	aline=$(grep -a '^\[argv\] ' $L/$src.out | cut -c8-)
	[ "$(echo "$aline" | grep -c .)" = 1 ] || abort "$src: expected exactly one [argv] line"
	ARGV=$($VENV/bin/python $L/rebuild_argv.py $L/$src.out fly $L/ckpt_$tag $L/${tag}_winner.yaml.gz $RFLAGS) \
		|| abort "argv rebuild failed for $tag"
	# fixed text: R minus the 5 flags, tag swapped back, must be A's flown argv byte for byte
	[ "$(count "$ARGV" " $RFLAGS ")" = 1 ] || abort "$tag: the 5 R flags are not present exactly once, in order"
	back=${ARGV/ $RFLAGS / }
	back=${back//$tag/$src}
	[ "$back" = "$aline" ] || abort "$tag: R argv minus the 5 flags is not A's [argv] (diff: $(diff <(echo "$back" | tr ' ' '\n') <(echo "$aline" | tr ' ' '\n') | tr '\n' ' '))"
	echo "$aline" | grep -q -- "--obs-rate-" && abort "$src: the A argv already carries a rate-loop input"
	echo "$ARGV" | grep -q -- "--obs-cascade-dhat" && abort "$tag: an MPCOF observer flag is in the argv"
	echo "$ARGV" | grep -q -- "--yaw-trim-ff" && abort "$tag: --yaw-trim-ff (F) is in the argv"
	echo "$ARGV" | grep -q -- "--refine-fitness" && abort "$tag: refine fitness must be off"
	echo "$ARGV" | grep -q -- "--output-mode ctbr " || abort "$tag: not --output-mode ctbr"
	[ "$(count "$ARGV" "--ctbr-delta-law ")" = 1 ] || abort "$tag: lacks the ctbr delta law"
	echo "$ARGV" | grep -q -- "--yaw-centred-ladder dense" || abort "$tag: dense ladder (L) missing"
	echo "$ARGV" | grep -q -- "--base-seed $((31337000 + s)) " || abort "$tag: wrong base seed"
	echo "$ARGV" | grep -q -- "--select-tail p99" || abort "$tag: rule P missing"
	echo "$ARGV" | grep -q -- "--num-eval-folds 5 " || abort "$tag: folds must be 5"
	[ "$(count "$ARGV" "--delta-max-yaw 0.5 ")" = 1 ] || abort "$tag: --delta-max-yaw is not N0's 0.5"
	local v; v=$(validate "$ARGV" "$2")
	[ "$v" = "OK 0.5 True 4 0.05 3" ] || abort "$tag: phased_ga refuses / mis-resolves the R stack: $v"
}

# --- rival re-score (MPCOF A/B/C via the cargo probe + PID from the run-printed rows) ---
rivals()
{
	local cap=$1 fx=$D/rival_fixture plog=$D/rival_rescore.log olds news line s
	rm -rf $fx
	( PYTHONPATH=src/wnn timeout $cap nice -n 5 $VENV/bin/python scripts/ctrl_indi_val_fixture.py \
		--run-log $L/$(src_of 1).out --out $fx --units test ) > $D/rival_fixture.log 2>&1 || { echo "fixture failed (see $D/rival_fixture.log)"; return 1; }
	[ "$(ls $fx | grep -c '^unit_')" = 35 ] || { echo "fixture has $(ls $fx | grep -c '^unit_') units, not 35"; return 1; }
	( cd src/wnn/ram/strategies/accelerator && WNN_RIVAL_RESCORE_FIXTURE=$ROOT/$fx timeout $cap nice -n 5 cargo test --release \
		-p ram_controller --lib --no-default-features -- --ignored --nocapture --exact \
		rival_probe_tests::test_units_rival_rows_rescore_f32_lineage ) > $plog 2>&1 \
		|| { echo "cargo probe failed or hit the ${cap}s cap (see $plog)"; return 1; }
	olds=""; news=""
	for s in 1 2 3 4 5; do
		[ -f $L/C-S2-d_wposx2_a60_rf_c200_dc_biasin_ms01_P_both_L1_yld_n0_full_s$s.done ] && olds="$olds $L/C-S2-d_wposx2_a60_rf_c200_dc_biasin_ms01_P_both_L1_yld_n0_full_s$s.out"
		[ -f $L/$(src_of $s).done ] && news="$news $L/$(src_of $s).out"
	done
	line=$($VENV/bin/python scripts/ctrl61_rival_rows.py --probe-log $plog --old-outs $olds --new-outs $news \
		--md docs/ctrl61_rival_rows.md 2>&1 | tail -1) || { echo "summary failed: $line"; return 1; }
	echo "$line" | grep -q "^RIVALS " || { echo "summary failed: $line"; return 1; }
	git add docs/ctrl61_rival_rows.md && git add -f $plog $D/rival_fixture.log \
		&& git commit -q -m "CTRL-69: MPCOF A/B/C + PID rival rows re-scored on the f32 (ABI 61/62) lineage, 35 TEST units

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Exc8QwHhxZqBnX4jPhZ3aZ" -- docs/ctrl61_rival_rows.md $plog $D/rival_fixture.log >> $RUNS 2>&1 \
		&& git push -q origin HEAD >> $RUNS 2>&1
	echo "$line"
}

# --- preflight: prereg committed + every available argv built and validated (ABI-62 code in a scratch site) ---
git ls-files --error-unmatch $PREREG >/dev/null 2>&1 || abort "prereg $PREREG is not committed"
git diff --quiet HEAD -- $PREREG || abort "prereg $PREREG has uncommitted edits"
grep -q "PRE-REGISTERED 22/11/2026, before any flight of arm R" $PREREG || abort "prereg header missing"
[ "$(git rev-parse $BRANCH)" = "$BRANCH_SHA" ] || abort "$BRANCH moved off $BRANCH_SHA (the wheel was built from it)"
built=0
for s in $SEEDS; do
	[ -f $L/$(src_of $s).done ] || continue
	build $s "$PRE_SITE"; built=$((built + 1))
	if [ "$DRY" = 1 ]; then
		echo "DRY $(tag_of $s)  (diff vs the flown A row $(src_of $s))"
		diff <(grep -a '^\[argv\] ' $L/$(src_of $s).out | cut -c8- | tr ' ' '\n') <(echo "$ARGV" | tr ' ' '\n')
		echo "validate: $(validate "$ARGV" "$PRE_SITE")"
	fi
done
[ "$DRY" = 1 ] && exit 0

# --- 1. wait for the MF61 block to end (only lines written after arm time count) ---
START_LINE=$(wc -l < $RUNS | tr -d ' ')
log "R62 armed (pid $$): CTRL-69 arm R = N0 + rate-loop error (roll,pitch, tau 50 ms) + rate-I (yaw), seeds $SEEDS; $built argv validated now (rest at flight time); waiting for the MF61 block end after line $START_LINE, then land ABI 62, re-score rivals, fly"
while :; do
	NEW=$(tail -n +"$((START_LINE + 1))" $RUNS | grep -v "R62")
	echo "$NEW" | grep -qE "Z MF61 DONE" && break
	hit=$(echo "$NEW" | grep -E "Z MF61 (ABORT|STAND DOWN)" | head -1)
	if [ -n "$hit" ]; then log "R62 STAND DOWN: the MF61 block did not finish, nothing landed or flown ($hit)"; exit 1; fi
	sleep 60
done
while pgrep -f "ab_mf61.s[h]" >/dev/null || ctrl_busy; do sleep 30; done

# --- 2. land ABI 62 ---
[ "$(git rev-parse $BRANCH)" = "$BRANCH_SHA" ] || refuse "$BRANCH moved off $BRANCH_SHA — ABI 61 left installed"
git merge-tree --write-tree HEAD $BRANCH >/dev/null 2>&1 || refuse "merge-tree reports conflicts — ABI 61 left installed"
[ "$(shasum -a 256 $WHEEL | cut -d' ' -f1)" = "$WHEEL_SHA" ] || refuse "wheel sha256 mismatch — ABI 61 left installed"
if ! git merge --no-ff $BRANCH -m "Merge ctrl62-rate-err: ABI 62 model-free rate-loop inputs (CTRL-69)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Exc8QwHhxZqBnX4jPhZ3aZ" >> $RUNS 2>&1; then
	git merge --abort 2>/dev/null
	refuse "merge failed (aborted) — ABI 61 left installed"
fi
log "R62: merged $(git rev-parse --short HEAD)"
if ! $VENV/bin/python -m pip install --force-reinstall --no-deps -q $WHEEL >> $RUNS 2>&1; then
	git reset -q --merge ORIG_HEAD
	refuse "pip install failed — merge undone (local, not pushed), ABI 61 left installed"
fi
got=$(cd /tmp && $VENV/bin/python -c "import ram_controller as c; print(c.ABI_VERSION)" 2>&1 | tail -1)
[ "$got" = 62 ] || refuse "installed ram_controller reads ABI '$got', expected 62 (tree merged, fix by hand)"
fac=$(cd /tmp && PYTHONPATH=$ROOT/src $VENV/bin/python -c "import wnn.control._accel as a; print(a.EXPECTED_ABI, a.ABI_VERSION)" 2>&1 | tail -1)
[ "$fac" = "62 62" ] || refuse "facade (EXPECTED_ABI ABI_VERSION) reads '$fac', expected '62 62' (tree merged, fix by hand)"
git push -q origin HEAD >> $RUNS 2>&1 && pushed=pushed || pushed="push FAILED (merge local only)"
log "R62: ABI 62 LANDED ($(git rev-parse --short HEAD), wheel ${WHEEL_SHA:0:8}, facade 62 62, $pushed)"

# --- 3. rival re-score, capped at 1 h before the flights ---
RIV_DONE=0
if out=$(rivals 3600); then log "R62: $out"; RIV_DONE=1
else log "R62: RIVALS DEFERRED to after the flights: $out"; fi

# --- 4. fly R ---
log "R62 START R = N0 + --obs-rate-err (roll,pitch, tau 0.05) + --obs-rate-i (yaw), seeds $SEEDS, ABI $ABI, paired vs A ..._a61_..._yld_n0_full_sN (prereg $PREREG)"
for s in $SEEDS; do
	tag=$(tag_of $s)
	[ -f $L/$tag.done ] && continue
	if [ ! -f $L/$(src_of $s).done ]; then log "R62 SKIP s$s: no finished A run $(src_of $s) (no pair, not flown)"; continue; fi
	build $s "$ROOT/src"
	while ctrl_busy; do sleep 30; done
	(cd /tmp && PYTHONPATH=$ROOT/src $VENV/bin/python -c "import ram_controller as c, wnn.control._accel as f; assert c.ABI_VERSION == $ABI and f.EXPECTED_ABI == $ABI") \
		|| abort "ram_controller / facade is not ABI $ABI (before $tag)"
	log "START $tag [R n0rerr = N0 + rate-loop error roll,pitch tau 0.05 + rate-I yaw, P, floor] (ABI $ABI, seed $((31337000 + s)), 35 units)"
	eval "set -- $ARGV"
	PYTHONPATH=src/wnn nice -n 5 /usr/bin/time -l $VENV/bin/python -u -m wnn.control.phased_ga "$@" > $L/$tag.out 2>&1
	rc=$?
	log "END $tag rc=$rc $(row $L/$tag.out)"
	[ $rc -eq 0 ] && touch $L/$tag.done
done
if [ $RIV_DONE = 0 ]; then
	if out=$(rivals 7200); then log "R62: $out (after the flights)"; else log "R62: RIVALS FAILED after the flights: $out"; fi
fi
log "R62 DONE"
