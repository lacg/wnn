#!/bin/bash
# CTRL-67/68 n=10 extension (Luiz 23/11/2026 ~07:30 EST: n=5 too close — B steady 0.240 vs A 0.308, err 1.100 vs 1.088).
# Flies the SAME two arms as ab_mf61.sh on seeds s6-s10, seed-interleaved A s6, B s6, ..., A s10, B s10:
#   A (CTRL-67) = N0, the model-free control:   source argv = the flown A s1 row ..._a61_..._L1_yld_n0_full_s1
#   B (CTRL-68) = N0 + --delta-max-yaw 0.3:     source argv = the flown B s1 row ..._a61_..._L1_yld_n0dmy03_full_s1
# Each sN argv = the source s1 [argv] line with ONLY the seed-dependent fields changed: --base-seed 31337001 ->
# 31337000+N and the two save paths (tag). Flown s1-s5 argv differ in exactly those fields (checked 23/11). The 35
# TEST units 99990101..135 are the same for every seed and are asserted unchanged.
# Wheel: ABI 62 (installed by ab_r62.sh). ABI 62 with the rate flags OFF is byte-identical to ABI 61
# (tests/controller_ctrl62_abi61_pin.py) -> s6-s10 pair validly with s1-s5. Tags carry a62 = the wheel used.
# Waits for 'R62 DONE' written after this script's arm line (stands down on R62 ABORT / STAND DOWN / REFUSED), then
# for ab_r62.sh and every phased_ga child to exit; asserts ram_controller + facade == 62 before every flight.
# Every argv is checked by fixed text (swap seed + tag back -> must EQUAL the source [argv] line; A sN and B sN must
# differ only by the --delta-max-yaw value + tag) and through phased_ga's own resolver (_resolve_ctbr).
# Prereg: docs/ctrl_modelfree_prereg.md, n=10 addendum (must be committed, no uncommitted edits).
# START/END lines in runs.log; ends 'MF10 DONE' (abort 'MF10 ABORT: why', stand-down 'MF10 STAND DOWN').
# Dry run: MF10_DRY=1 prints each argv's diff vs its s1 source + the resolver line, flies nothing.
cd /Users/lacg/wnn || exit 1
ROOT=/Users/lacg/wnn
L=logs/controller/ctrl48
RUNS=$L/runs.log
VENV=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv
ABI=62
SEEDS=${MF10_SEEDS:-6 7 8 9 10}
DRY=${MF10_DRY:-0}
PREREG=docs/ctrl_modelfree_prereg.md
PREREG_HDR="^## Addendum (23/11/2026): n=10 extension"
UNITS="--test-seeds $(seq -f %.0f 99990101 99990135 | tr "\n" " ")"
log() { echo "$(date -u +%FT%TZ) $*" >> $RUNS; }
abort() { [ "$DRY" = 1 ] || log "MF10 ABORT: $1"; echo "MF10 ABORT: $1" >&2; exit 1; }
row()
{
	local g r
	g=$(grep -a 'stage-select\] HEADLINE stage' $1 | grep -oE 'genome=[^ ]+' | cut -d= -f2)
	r=$(grep -a "HEADLINE-$g MULTI-SEED" $1 | tail -1)
	[ -z "$r" ] && r=$(grep -aE "\[(test|report)-seeds\] ${g%%#*} MULTI-SEED" $1 | tail -1)
	echo "genome=$g $(echo "$r" | grep -oE 'stable=[^ ]+  err=[^ ]+  steady=[^ ]+|alt=[^ ]+|pos=[^ ]+|pos_hold=[^ ]+|dz_hold=[^ ]+|stable_fail=[^ ]+' | tr '\n' ' ')"
}
suffix() { [ $1 = a ] && echo n0 || echo n0dmy03; }
src_of() { echo C-S2-d_wposx2_a61_rf_c200_dc_biasin_ms01_P_both_L1_yld_$(suffix $1)_full_s1; }
tag_of() { echo C-S2-d_wposx2_a${ABI}_rf_c200_dc_biasin_ms01_P_both_L1_yld_$(suffix $2)_full_s$1; }
dmy_of() { [ $1 = a ] && echo 0.5 || echo 0.3; }
count() { echo "$1" | grep -o -- "$2" | wc -l | tr -d ' '; }
ctrl_busy() { pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null; }
# validate(argv): phased_ga's own resolution chain (parse -> episode_config_from_args -> _resolve_ctbr)
validate()
{
	(cd /tmp && PYTHONPATH=$ROOT/src $VENV/bin/python -c "
import shlex, sys
from wnn.control import phased_ga as p
a = p.build_arg_parser().parse_args(shlex.split(sys.argv[1]))
ec = p.episode_config_from_args(a); p._resolve_ctbr(a, ec); c = a._ctbr
print('OK', a.base_seed, a.delta_max_yaw, c.get('obs_rate_i'), c.get('obs_rate_err'), c.get('obs_rate_i_axes'), c.get('obs_rate_err_tau'), c.get('obs_rate_err_axes'), a.num_eval_folds, len(a.test_seeds), a.test_seeds[0], a.test_seeds[-1])" "$1" 2>&1 | tail -1)
}
# build(s, arm) -> sets ARGV, or aborts with the reason
build()
{
	local s=$1 arm=$2 src tag sline back v
	src=$(src_of $arm); tag=$(tag_of $s $arm)
	[ -f $L/$src.done ] || abort "missing finished s1 source run $src"
	sline=$(grep -a '^\[argv\] ' $L/$src.out | cut -c8-)
	[ "$(echo "$sline" | grep -c .)" = 1 ] || abort "$src: expected exactly one [argv] line"
	[ "$(count "$sline" "--base-seed 31337001 ")" = 1 ] || abort "$src: source does not carry --base-seed 31337001 once"
	ARGV=$($VENV/bin/python $L/rebuild_argv.py $L/$src.out fly $L/ckpt_$tag $L/${tag}_winner.yaml.gz) \
		|| abort "argv rebuild failed for $tag"
	ARGV=${ARGV/--base-seed 31337001 /--base-seed $((31337000 + s)) }
	# fixed text: seed + tag swapped back must be the source's flown argv byte for byte
	back=${ARGV/--base-seed $((31337000 + s)) /--base-seed 31337001 }
	back=${back//$tag/$src}
	[ "$back" = "$sline" ] || abort "$tag: argv minus seed+tag is not $src's [argv] (diff: $(diff <(echo "$back" | tr ' ' '\n') <(echo "$sline" | tr ' ' '\n') | tr '\n' ' '))"
	[ "$(count "$ARGV" "--base-seed $((31337000 + s)) ")" = 1 ] || abort "$tag: base seed not applied once"
	echo "$ARGV" | grep -q -- "$UNITS" || abort "$tag: the 35 TEST units 99990101..135 changed"
	echo "$ARGV" | grep -q -- "--obs-rate-" && abort "$tag: a rate-loop input flag is in the argv"
	echo "$ARGV" | grep -q -- "--obs-cascade-dhat" && abort "$tag: an MPCOF observer flag is in the argv"
	echo "$ARGV" | grep -q -- "--yaw-trim-ff" && abort "$tag: --yaw-trim-ff (F) is in the argv"
	echo "$ARGV" | grep -q -- "--refine-fitness" && abort "$tag: refine fitness must be off"
	echo "$ARGV" | grep -q -- "--output-mode ctbr " || abort "$tag: not --output-mode ctbr"
	[ "$(count "$ARGV" "--ctbr-delta-law ")" = 1 ] || abort "$tag: lacks the ctbr delta law"
	echo "$ARGV" | grep -q -- "--yaw-centred-ladder dense" || abort "$tag: dense ladder (L) missing"
	echo "$ARGV" | grep -q -- "--select-tail p99" || abort "$tag: rule P missing"
	echo "$ARGV" | grep -q -- "--num-eval-folds 5 " || abort "$tag: folds must be 5"
	[ "$(count "$ARGV" "--delta-max-yaw ")" = 1 ] || abort "$tag: must carry exactly one --delta-max-yaw"
	[ "$(count "$ARGV" "--delta-max-yaw $(dmy_of $arm) ")" = 1 ] || abort "$tag: --delta-max-yaw is not $(dmy_of $arm)"
	v=$(validate "$ARGV")
	[ "$v" = "OK $((31337000 + s)) $(dmy_of $arm) None None None None None 5 35 99990101 99990135" ] \
		|| abort "$tag: phased_ga refuses / mis-resolves the stack: $v"
}
# pair_check(s): A sN and B sN differ only by the --delta-max-yaw value and the tag
pair_check()
{
	local s=$1 a b
	build $s a; a=$ARGV
	build $s b; b=$ARGV
	b=${b/--delta-max-yaw 0.3 /--delta-max-yaw 0.5 }
	b=${b//$(tag_of $s b)/$(tag_of $s a)}
	[ "$a" = "$b" ] || abort "s$s: A and B differ beyond --delta-max-yaw + tag"
}

# --- preflight: prereg committed + every argv built, cross-checked and resolved BEFORE arming ---
git ls-files --error-unmatch $PREREG >/dev/null 2>&1 || abort "prereg $PREREG is not committed"
git diff --quiet HEAD -- $PREREG || abort "prereg $PREREG has uncommitted edits"
grep -q "$PREREG_HDR" $PREREG || abort "prereg n=10 addendum missing in $PREREG"
for s in $SEEDS; do
	pair_check $s
	for arm in a b; do
		[ -e $L/$(tag_of $s $arm).out ] && [ "$DRY" != 1 ] && abort "$(tag_of $s $arm).out already exists (never repeat a seed)"
		build $s $arm
		if [ "$DRY" = 1 ]; then
			echo "DRY $(tag_of $s $arm)  (diff vs the flown s1 row $(src_of $arm))"
			diff <(grep -a '^\[argv\] ' $L/$(src_of $arm).out | cut -c8- | tr ' ' '\n') <(echo "$ARGV" | tr ' ' '\n')
			echo "validate: $(validate "$ARGV")"
		fi
	done
done
[ "$DRY" = 1 ] && exit 0

# --- wait for the R62 block to end (only lines written after arm time count) ---
START_LINE=$(wc -l < $RUNS | tr -d ' ')
log "MF10 armed (pid $$): CTRL-67/68 n=10 extension, A = N0 (model-free control), B = N0 + --delta-max-yaw 0.3, seeds $SEEDS on ABI 62 flags-off; all argv validated; waiting for 'R62 DONE' after line $START_LINE"
while :; do
	NEW=$(tail -n +"$((START_LINE + 1))" $RUNS | grep -v "MF10")
	echo "$NEW" | grep -qE "Z R62 DONE" && break
	hit=$(echo "$NEW" | grep -E "Z R62 (ABORT|STAND DOWN)|Z R62: REFUSED" | head -1)
	if [ -n "$hit" ]; then log "MF10 STAND DOWN: the R62 block did not finish cleanly, nothing flown ($hit)"; exit 1; fi
	sleep 60
done
while pgrep -f "ab_r62.s[h]" >/dev/null || ctrl_busy; do sleep 30; done
log "MF10 START A = N0 (model-free control), B = N0 + --delta-max-yaw 0.3, seed-interleaved, seeds $SEEDS, ABI $ABI flags-off (prereg $PREREG, n=10 addendum)"
for s in $SEEDS; do
	for arm in a b; do
		tag=$(tag_of $s $arm)
		[ -f $L/$tag.done ] && continue
		pair_check $s
		build $s $arm
		while ctrl_busy; do sleep 30; done
		(cd /tmp && PYTHONPATH=$ROOT/src $VENV/bin/python -c "import ram_controller as c, wnn.control._accel as f; assert c.ABI_VERSION == $ABI and f.EXPECTED_ABI == $ABI") \
			|| abort "ram_controller / facade is not ABI $ABI (before $tag)"
		[ $arm = a ] && what="A n0 = N0 model-free control, n=10 ext, P, floor" || what="B n0dmy03 = N0 + --delta-max-yaw 0.3, n=10 ext, P, floor"
		log "START $tag [$what] (ABI $ABI, seed $((31337000 + s)), 35 units)"
		eval "set -- $ARGV"
		PYTHONPATH=src/wnn nice -n 5 /usr/bin/time -l $VENV/bin/python -u -m wnn.control.phased_ga "$@" > $L/$tag.out 2>&1
		rc=$?
		log "END $tag rc=$rc $(row $L/$tag.out)"
		[ $rc -eq 0 ] && touch $L/$tag.done
	done
done
log "MF10 DONE"
