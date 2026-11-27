#!/bin/bash
# CTRL-79 (Luiz 25/11/2026): yaw-integral ladder (yil = --yaw-integral-ladder dense) on the G+T base, prereg
# docs/ctrl79_yaw_integral_ladder_prereg.md. Derived from the CTRL-77 factorial driver (same checks); the arm carries ONLY
# --yaw-integral-ladder dense (no delta swap). Needs branch ctrl79-yaw-integral-ladder MERGED into this tree first.
# Y79_ROOT=<worktree> is allowed ONLY with Y79_DRY=1 (validate against the unmerged branch). Factorial comment block follows.
# 11 arms (every non-empty combo of L1{off,on} x L2{off,a,b} x L3{off,on}), staged n=3 -> 5 -> 7 -> 10.
#   L1  = --delta-leak-yaw 0.95 + the in-place swap --delta-max-yaw 0.5 -> 0.08333 (gain-matched: 0.08333/0.05 = 1.667
#         = 0.5/0.3 to 3 decimals)
#   L2a = --gyro-xy-ladder dense       L2b = --gyro-xy-ladder dense_hold       L3 = --yaw-err-outer-cut 4.5
# Control of every pair = the FLOWN G+T sN run (..._a62_..._L1_yld_n0g3gz_full_sN; ABI 62; never re-flown). Each arm argv
# = that run's [argv] line + ONLY the arm's flags (+ L1's swap) + the two save paths (tag
# ..._a63_..._L1_yld_gt_<arm>_full_sN). Fixed-text check: arm argv minus its flags, L1 swap undone, tag -> control tag
# == the control's [argv] byte for byte. Resolver check (phased_ga's own parser + _resolve_ctbr + resolve_block10_args +
# calibration token): the control resolves to G+T; each arm moves ONLY its fields; digest of everything else equal.
# Stage-aware: Y79_SEEDS (default "1 2 3") and Y79_ARMS (default all 11) — later stages pass the survivors only.
# Order: seed-major (every listed arm on the first seed, then the next seed). One run at a time; waits for any other
# controller run. Wheel ram_controller + facade ABI 63 asserted before every flight; src/wnn/control must be clean;
# the prereg must be committed. A tag whose .out exists is never reused (no recipe repeated on a seed).
# START/END in runs.log; block lines 'Y79 ...'; ends 'Y79 DONE' (abort 'Y79 ABORT: why').
# Dry run: Y79_DRY=1 prints, per arm, the token diff vs its G+T control argv + both resolver lines; flies nothing.
cd /Users/lacg/wnn || exit 1
ROOT=${Y79_ROOT:-/Users/lacg/wnn}
L=logs/controller/ctrl48
RUNS=$L/runs.log
VENV=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv
ABI=63
ALL_ARMS="yil l2byil"
SEEDS=${Y79_SEEDS:-1 2 3}
ARMS=${Y79_ARMS:-$ALL_ARMS}
DRY=${Y79_DRY:-0}
PREREG=docs/ctrl79_yaw_integral_ladder_prereg.md
PREREG_HDR="^## Adoption rule"
PINS="tests/controller_ctrl79_yaw_integral_ladder.py"
UNITS="--test-seeds $(seq -f %.0f 99990101 99990135 | tr "\n" " ")"
GT_GAMMA="(1.0, 2.0, 2.0, 3.0)"
GT_CALIB="eps:200,regime:flown,ladder:quantile,yaw:dense,tilt:dense,gyroz:dense"
DMAX0=0.5
PFX=C-S2-d_wposx2
MID=rf_c200_dc_biasin_ms01_P_both_L1_yld
log() { echo "$(date -u +%FT%TZ) $*" >> $RUNS; }
abort() { [ "$DRY" = 1 ] || log "Y79 ABORT: $1"; echo "Y79 ABORT: $1" >&2; exit 1; }
row()
{
	local g r
	g=$(grep -a 'stage-select\] HEADLINE stage' $1 | grep -oE 'genome=[^ ]+' | cut -d= -f2)
	r=$(grep -a "HEADLINE-$g MULTI-SEED" $1 | tail -1)
	[ -z "$r" ] && r=$(grep -aE "\[(test|report)-seeds\] ${g%%#*} MULTI-SEED" $1 | tail -1)
	echo "genome=$g $(echo "$r" | grep -oE 'stable=[^ ]+  err=[^ ]+  steady=[^ ]+|alt=[^ ]+|pos=[^ ]+|pos_hold=[^ ]+|dz_hold=[^ ]+|stable_fail=[^ ]+' | tr '\n' ' ')"
}
has() { case $1 in *$2*) return 0;; esac; return 1; }   # has <arm> <lever>: l1 / l2a / l2b / l3
l2_of() { has $1 l2a && echo dense || { has $1 l2b && echo dense_hold; }; }
flags_of()
{
	local f=""
	[ "$1" = yil ] && f="--yaw-integral-ladder dense"
	[ "$1" = l2byil ] && f="--gyro-xy-ladder dense_hold --yaw-integral-ladder dense"
	echo "$f"
}
calib_of() { [ "$1" = l2byil ] && { echo "$GT_CALIB,gyroxy:dense_hold,yawi:dense"; return; }; echo "$GT_CALIB,yawi:dense"; }
dmax_of() { echo $DMAX0; }
leak_of() { echo None; }
what_of()
{
	local w=""
	w="yaw-integral ladder dense (yaw_err_i cuts +-1.5/3.5/6.5/12): read the yaw trim from the integral input"
	[ "$1" = l2byil ] && w="L2-b gyro_x/y dense_hold ladder + $w (stacking screen, Luiz 26/11)"
	echo "$w"
}
tag_of() { echo ${PFX}_a${ABI}_${MID}_gt_$2_full_s$1; }
ctl_of() { echo ${PFX}_a62_${MID}_n0g3gz_full_s$1; }
count() { echo "$1" | grep -o -- "$2" | wc -l | tr -d ' '; }
ctrl_busy() { pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null; }
validate()
{
	(cd /tmp && PYTHONPATH=$ROOT/src $VENV/bin/python -c "
import hashlib, shlex, sys
from wnn.control import phased_ga as p
a = p.build_arg_parser().parse_args(shlex.split(sys.argv[1]))
ec = p.episode_config_from_args(a); p._resolve_ctbr(a, ec); p.resolve_block10_args(a); c = a._ctbr
skip = ('yaw_integral_ladder', 'gyro_xy_ladder', 'save_stage_checkpoints', 'save_winner')
rest = repr(sorted((k, repr(v)) for k, v in c.items())) + repr(sorted((k, repr(v)) for k, v in vars(a).items() if not k.startswith('_') and k not in skip))
lk = a.delta_leak_yaw if a.delta_leak_yaw is not None else a.delta_leak
gain = round(a.delta_max_yaw / (1.0 - lk), 3)
print('OK', a.base_seed, a.delta_max_yaw, a.delta_leak_yaw, tuple(float(g) for g in c['gamma']), 'gain', gain, '|', p.calibration_from_args(a).token(), '|', a.num_eval_folds, len(a.test_seeds), a.test_seeds[0], a.test_seeds[-1], '|', hashlib.sha256(rest.encode()).hexdigest()[:16])" "$1" 2>&1 | tail -1)
}
# build(s, arm) -> sets ARGV, CLINE, VAL_CTL, VAL_ARM, or aborts
build()
{
	local s=$1 arm=$2 tag ctl flags back v vc want
	tag=$(tag_of $s $arm); ctl=$(ctl_of $s); flags=$(flags_of $arm)
	[ -n "$flags" ] || abort "$arm: no flags (unknown arm)"
	[ -f $L/$ctl.done ] || abort "s$s: the G+T control $ctl has not finished — never fly an arm without its control"
	CLINE=$(grep -a '^\[argv\] ' $L/$ctl.out | cut -c8-)
	[ "$(echo "$CLINE" | grep -c .)" = 1 ] || abort "$ctl: expected exactly one [argv] line"
	[ "$(count "$CLINE" "--base-seed $((31337000 + s)) ")" = 1 ] || abort "$ctl: does not carry --base-seed $((31337000 + s)) once"
	[ "$(count "$CLINE" "--delta-max-yaw $DMAX0 ")" = 1 ] || abort "$ctl: does not carry --delta-max-yaw $DMAX0 exactly once"
	echo "$CLINE" | grep -qE -- "--delta-leak-yaw|--gyro-xy-ladder|--yaw-err-outer-cut|--yaw-integral-ladder" && abort "$ctl: the control already carries a lever flag"
	ARGV=$($VENV/bin/python $L/rebuild_argv.py $L/$ctl.out fly $L/ckpt_$tag $L/${tag}_winner.yaml.gz $flags) \
		|| abort "argv rebuild failed for $tag"
	# fixed text: flags removed, L1 swap undone, tag -> control tag == the flown G+T sN [argv]
	[ "$(count "$ARGV" " $flags ")" = 1 ] || abort "$tag: the arm flags '$flags' not in the argv exactly once"
	back=${ARGV/ $flags / }
	back=${back//$tag/$ctl}
	[ "$back" = "$CLINE" ] || abort "$tag: argv minus flags+swap+tag is not the G+T s$s control ($ctl) argv (diff: $(diff <(echo "$back" | tr ' ' '\n') <(echo "$CLINE" | tr ' ' '\n') | tr '\n' ' '))"
	[ "$(count "$ARGV" "--base-seed $((31337000 + s)) ")" = 1 ] || abort "$tag: base seed not $((31337000 + s)) once"
	echo "$ARGV" | grep -q -- "$UNITS" || abort "$tag: the 35 TEST units 99990101..135 changed"
	echo "$ARGV" | grep -q -- "--obs-rate-" && abort "$tag: a rate-loop input flag is in the argv"
	echo "$ARGV" | grep -q -- "--obs-cascade-dhat" && abort "$tag: an MPCOF observer flag is in the argv (model-free rule)"
	echo "$ARGV" | grep -q -- "--yaw-trim-ff" && abort "$tag: --yaw-trim-ff is in the argv (model-free rule)"
	echo "$ARGV" | grep -q -- "--refine-fitness" && abort "$tag: refine fitness must be off"
	echo "$ARGV" | grep -q -- "--output-mode ctbr " || abort "$tag: not --output-mode ctbr"
	[ "$(count "$ARGV" "--ctbr-delta-law ")" = 1 ] || abort "$tag: lacks the ctbr delta law"
	[ "$(count "$ARGV" "--ctbr-gamma 1 2 2 3 ")" = 1 ] || abort "$tag: G+T yaw gamma 1 2 2 3 missing"
	[ "$(count "$ARGV" "--gyro-z-ladder dense ")" = 1 ] || abort "$tag: G+T gyro_z ladder missing"
	echo "$ARGV" | grep -q -- "--yaw-centred-ladder dense" || abort "$tag: dense yaw-error ladder missing"
	echo "$ARGV" | grep -q -- "--select-tail p99" || abort "$tag: rule P missing"
	echo "$ARGV" | grep -q -- "--num-eval-folds 5 " || abort "$tag: folds must be 5"
	[ "$(count "$ARGV" "--delta-max-yaw ")" = 1 ] || abort "$tag: --delta-max-yaw not exactly once"
	[ "$(count "$ARGV" "--delta-max-yaw $(dmax_of $arm) ")" = 1 ] || abort "$tag: --delta-max-yaw is not $(dmax_of $arm)"
	vc=$(validate "$CLINE"); v=$(validate "$ARGV")
	VAL_CTL=$vc; VAL_ARM=$v
	[ "${vc% | *}" = "OK $((31337000 + s)) $DMAX0 None $GT_GAMMA gain 1.667 | $GT_CALIB | 5 35 99990101 99990135" ] \
		|| abort "$ctl: the G+T s$s control does not resolve to G+T: $vc"
	want="OK $((31337000 + s)) $(dmax_of $arm) $(leak_of $arm) $GT_GAMMA gain 1.667 | $(calib_of $arm) | 5 35 99990101 99990135 | ${vc##* | }"
	[ "$v" = "$want" ] || abort "$tag: phased_ga refuses / mis-resolves the arm (want '$want'): $v"
}

# --- preflight ---
[ "$ROOT" = /Users/lacg/wnn ] || [ "$DRY" = 1 ] || abort "Y79_ROOT may only be overridden with Y79_DRY=1"
for a in $ARMS; do case " $ALL_ARMS " in *" $a "*) ;; *) abort "unknown arm '$a' in Y79_ARMS";; esac; done
git ls-files --error-unmatch $PREREG >/dev/null 2>&1 || abort "prereg $PREREG is not committed"
git diff --quiet HEAD -- $PREREG || abort "prereg $PREREG has uncommitted edits"
grep -q "$PREREG_HDR" $PREREG || abort "prereg winnowing section missing in $PREREG"
for p in $PINS; do git -C $ROOT ls-files --error-unmatch $p >/dev/null 2>&1 || abort "the flag-off pin $p is not committed in $ROOT (merge ctrl79-yaw-integral-ladder first)"; done
git diff --quiet HEAD -- src/wnn/control || abort "src/wnn/control has uncommitted edits"
(cd /tmp && PYTHONPATH=$ROOT/src $VENV/bin/python -c "import ram_controller as c, wnn.control._accel as f; assert c.ABI_VERSION == $ABI and f.EXPECTED_ABI == $ABI") \
	|| abort "ram_controller / facade is not ABI $ABI"
N=0
for s in $SEEDS; do
	for arm in $ARMS; do
		tag=$(tag_of $s $arm)
		[ -f $L/$tag.done ] && continue
		[ -e $L/$tag.out ] && [ "$DRY" != 1 ] && abort "$tag.out already exists (never repeat a recipe on a seed / reuse a tag)"
		build $s $arm; N=$((N + 1))
		if [ "$DRY" = 1 ]; then
			echo "DRY $tag  (control $(ctl_of $s))"
			diff <(echo "$CLINE" | tr ' ' '\n') <(echo "$ARGV" | tr ' ' '\n') | grep '^[<>]' | grep -v -- "_full_s$s" | tr '\n' ' '; echo
			echo "  validate control: $VAL_CTL"
			echo "  validate arm:     $VAL_ARM"
		fi
	done
done
[ "$DRY" = 1 ] && { echo "DRY OK: $N runs validated (seeds: $SEEDS; arms: $ARMS)"; exit 0; }

LOCK=/tmp/ab_yil79.pid
[ -f $LOCK ] && kill -0 "$(cat $LOCK)" 2>/dev/null && abort "another Y79 driver is already running (pid $(cat $LOCK))"
echo $$ > $LOCK
log "Y79 armed (pid $$, ab_yil79.sh): CTRL-79 yaw-integral ladder, seeds [$SEEDS] x arms [$ARMS] = $N runs, seed-major, ABI $ABI; every argv validated vs its flown G+T control (prereg $PREREG)"
while ctrl_busy; do sleep 30; done
log "Y79 START seeds [$SEEDS] arms [$ARMS]"
for s in $SEEDS; do
	for arm in $ARMS; do
		tag=$(tag_of $s $arm)
		[ -f $L/$tag.done ] && continue
		[ -e $L/$tag.out ] && abort "$tag.out exists without .done (void flight) — never reuse a tag"
		git diff --quiet HEAD -- src/wnn/control || abort "src/wnn/control changed under the block (before $tag)"
		git diff --quiet HEAD -- $PREREG || abort "prereg edited under the block (before $tag)"
		build $s $arm
		while ctrl_busy; do sleep 30; done
		(cd /tmp && PYTHONPATH=$ROOT/src $VENV/bin/python -c "import ram_controller as c, wnn.control._accel as f; assert c.ABI_VERSION == $ABI and f.EXPECTED_ABI == $ABI") \
			|| abort "ram_controller / facade is not ABI $ABI (before $tag)"
		log "START $tag [Y79 $arm = G+T + $(what_of $arm), P, floor] (ABI $ABI, seed $((31337000 + s)), 35 units; control $(ctl_of $s))"
		eval "set -- $ARGV"
		PYTHONPATH=src/wnn nice -n 5 /usr/bin/time -l $VENV/bin/python -u -m wnn.control.phased_ga "$@" > $L/$tag.out 2>&1
		rc=$?
		log "END $tag rc=$rc $(row $L/$tag.out)"
		[ $rc -eq 0 ] && touch $L/$tag.done
	done
done
rm -f $LOCK
log "Y79 DONE seeds [$SEEDS] arms [$ARMS]"
