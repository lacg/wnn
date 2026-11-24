#!/bin/bash
# CTRL-71/72 n=10 extension (Luiz 24/11/2026 ~11:15 EST): G, T, G+T on seeds s6-s10 + the missing N0 controls s9, s10.
# Per seed: N0 sN (s9, s10 only), G sN, T sN, G+T sN -> 17 runs.
#   N0  = the CTRL-67 model-free control          tags ..._yld_n0_full_s10, ..._yld_n0_full_s9r (the s9 tag is VOID)
#   G   = N0 + --ctbr-gamma 1 2 2 3                tags ..._yld_n0g3_full_sN
#   T   = N0 + --gyro-z-ladder dense               tags ..._yld_n0gz_full_sN
#   G+T = N0 + both flags                          tags ..._yld_n0g3gz_full_sN
# Every argv = the flown N0 s1 [argv] line (..._a61_..._L1_yld_n0_full_s1) with ONLY --base-seed 31337001 -> 31337000+N,
# the two save paths (tag) and the arm's flag(s). Fixed-text checks: (1) flag(s) + seed + tag swapped back == the s1
# source line; (2) an arm minus its flag(s) + tag == the N0 sN argv — the FLOWN control's [argv] line when it exists
# (s6-s8 always; s9r/s10 once flown), else the N0 sN argv built by the same rule (preflight only). Resolver: the
# control resolves to the N0 gamma + calib token; each arm moves only its own field(s); digest of everything else equal.
# The void tag ..._a62_..._L1_yld_n0_full_s9 is refused anywhere in any argv. A tag whose .out exists is never reused.
# Wheel ABI 62 (rate-loop flags off), asserted before every flight. Prereg docs/ctrl71_72_levers_prereg.md (n=10
# addendum, committed); src/wnn/control must be clean (the CTRL-71 pinned tree).
# START/END in runs.log; ends 'GT10 DONE' (abort 'GT10 ABORT: why'; also if an N0 control does not finish).
# Dry run: GT10_DRY=1 prints each argv's diff vs the s1 source + the resolver lines, flies nothing.
cd /Users/lacg/wnn || exit 1
ROOT=/Users/lacg/wnn
L=logs/controller/ctrl48
RUNS=$L/runs.log
VENV=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv
ABI=62
SEEDS=${GT10_SEEDS:-6 7 8 9 10}
DRY=${GT10_DRY:-0}
PREREG=docs/ctrl71_72_levers_prereg.md
PREREG_HDR="^## Addendum (24/11/2026, before any s6-s10 flight): n=10 extension"
PIN=tests/controller_ctrl71_gyro_z_ladder.py
UNITS="--test-seeds $(seq -f %.0f 99990101 99990135 | tr "\n" " ")"
N0_GAMMA="(1.0, 2.0, 2.0, 2.0)"
N0_CALIB="eps:200,regime:flown,ladder:quantile,yaw:dense,tilt:dense"
PFX=C-S2-d_wposx2
MID=rf_c200_dc_biasin_ms01_P_both_L1_yld
SRC=${PFX}_a61_${MID}_n0_full_s1
VOID=${PFX}_a62_${MID}_n0_full_s9
log() { echo "$(date -u +%FT%TZ) $*" >> $RUNS; }
abort() { [ "$DRY" = 1 ] || log "GT10 ABORT: $1"; echo "GT10 ABORT: $1" >&2; exit 1; }
row()
{
	local g r
	g=$(grep -a 'stage-select\] HEADLINE stage' $1 | grep -oE 'genome=[^ ]+' | cut -d= -f2)
	r=$(grep -a "HEADLINE-$g MULTI-SEED" $1 | tail -1)
	[ -z "$r" ] && r=$(grep -aE "\[(test|report)-seeds\] ${g%%#*} MULTI-SEED" $1 | tail -1)
	echo "genome=$g $(echo "$r" | grep -oE 'stable=[^ ]+  err=[^ ]+  steady=[^ ]+|alt=[^ ]+|pos=[^ ]+|pos_hold=[^ ]+|dz_hold=[^ ]+|stable_fail=[^ ]+' | tr '\n' ' ')"
}
arms_of() { [ $1 -ge 9 ] && echo "n0 g t gt" || echo "g t gt"; }
suffix() { case $1 in n0) echo n0;; g) echo n0g3;; t) echo n0gz;; gt) echo n0g3gz;; esac; }
flag_of() { case $1 in n0) echo "";; g) echo "--ctbr-gamma 1 2 2 3";; t) echo "--gyro-z-ladder dense";; gt) echo "--ctbr-gamma 1 2 2 3 --gyro-z-ladder dense";; esac; }
gamma_of() { case $1 in g|gt) echo "(1.0, 2.0, 2.0, 3.0)";; *) echo "$N0_GAMMA";; esac; }
calib_of() { case $1 in t|gt) echo "$N0_CALIB,gyroz:dense";; *) echo "$N0_CALIB";; esac; }
tag_of() { [ $2 = n0 ] && [ $1 = 9 ] && echo ${VOID}r || echo ${PFX}_a${ABI}_${MID}_$(suffix $2)_full_s$1; }
# the N0 control tag of seed s: s6-s8 = the one finished a61/a62 row; s9/s10 = the extension's own N0 tag
ctl_of()
{
	local s=$1 t n=0 hit=
	[ $s -ge 9 ] && { tag_of $s n0; return; }
	for a in 61 62; do
		t=${PFX}_a${a}_${MID}_n0_full_s$s
		[ -f $L/$t.done ] && { hit=$t; n=$((n + 1)); }
	done
	[ $n = 1 ] && echo $hit
}
count() { echo "$1" | grep -o -- "$2" | wc -l | tr -d ' '; }
ctrl_busy() { pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null; }
validate()
{
	(cd /tmp && PYTHONPATH=$ROOT/src $VENV/bin/python -c "
import hashlib, shlex, sys
from wnn.control import phased_ga as p
a = p.build_arg_parser().parse_args(shlex.split(sys.argv[1]))
ec = p.episode_config_from_args(a); p._resolve_ctbr(a, ec); c = a._ctbr
skip = ('ctbr_gamma', 'gyro_z_ladder', 'save_stage_checkpoints', 'save_winner')
rest = repr(sorted((k, v) for k, v in c.items() if k != 'gamma')) + repr(sorted((k, repr(v)) for k, v in vars(a).items() if not k.startswith('_') and k not in skip))
print('OK', a.base_seed, a.delta_max_yaw, tuple(float(g) for g in c['gamma']), '|', p.calibration_from_args(a).token(), '|', a.num_eval_folds, len(a.test_seeds), a.test_seeds[0], a.test_seeds[-1], '|', hashlib.sha256(rest.encode()).hexdigest()[:16])" "$1" 2>&1 | tail -1)
}
# rule(s, arm) -> the argv by the rule (s1 source + seed + tag + flags), sets RULE
rule()
{
	local s=$1 arm=$2 tag flag
	tag=$(tag_of $s $arm); flag=$(flag_of $arm)
	RULE=$($VENV/bin/python $L/rebuild_argv.py $L/$SRC.out fly $L/ckpt_$tag $L/${tag}_winner.yaml.gz $flag) \
		|| abort "argv rebuild failed for $tag"
	RULE=${RULE/--base-seed 31337001 /--base-seed $((31337000 + s)) }
}
# build(s, arm) -> sets ARGV and VAL_CTL / VAL_ARM, or aborts
build()
{
	local s=$1 arm=$2 tag sline back ctl cline flag v vc want
	tag=$(tag_of $s $arm); flag=$(flag_of $arm)
	[ -f $L/$SRC.done ] || abort "missing finished s1 source run $SRC"
	sline=$(grep -a '^\[argv\] ' $L/$SRC.out | cut -c8-)
	[ "$(echo "$sline" | grep -c .)" = 1 ] || abort "$SRC: expected exactly one [argv] line"
	[ "$(count "$sline" "--base-seed 31337001 ")" = 1 ] || abort "$SRC: source does not carry --base-seed 31337001 once"
	echo "$sline" | grep -qE -- "--ctbr-gamma|--gyro-z-ladder" && abort "$SRC: the control already carries an arm flag"
	rule $s $arm; ARGV=$RULE
	echo "$ARGV" | grep -qE "${VOID}( |_winner)" && abort "$tag: the VOID tag $VOID is in the argv"
	# fixed text 1: flag(s) removed, seed + tag swapped back == the flown s1 source argv
	back=$ARGV
	[ -n "$flag" ] && { [ "$(count "$ARGV" " $flag ")" = 1 ] || abort "$tag: the arm flag(s) '$flag' not in the argv exactly once"; back=${back/ $flag / }; }
	back=${back/--base-seed $((31337000 + s)) /--base-seed 31337001 }
	back=${back//$tag/$SRC}
	[ "$back" = "$sline" ] || abort "$tag: argv minus flag+seed+tag is not $SRC's [argv] (diff: $(diff <(echo "$back" | tr ' ' '\n') <(echo "$sline" | tr ' ' '\n') | tr '\n' ' '))"
	# fixed text 2 (arms): flag(s) removed, tag swapped == the N0 sN control argv (flown if it exists, else by rule)
	ctl=$(ctl_of $s); [ -n "$ctl" ] || abort "s$s: no single finished N0 control row"
	if [ -f $L/$ctl.done ]; then
		cline=$(grep -a '^\[argv\] ' $L/$ctl.out | cut -c8-)
	elif [ $arm != n0 ] && [ "$DRY" != 1 ] && [ "${PREFLIGHT:-0}" != 1 ]; then
		abort "$tag: its N0 control $ctl has not finished — never fly an arm without its control"
	else
		rule $s n0; cline=$RULE
	fi
	back=$ARGV
	[ -n "$flag" ] && back=${back/ $flag / }
	back=${back//$tag/$ctl}
	[ "$back" = "$cline" ] || abort "$tag: argv minus flag+tag is not the N0 s$s control ($ctl) argv"
	[ "$(count "$ARGV" "--base-seed $((31337000 + s)) ")" = 1 ] || abort "$tag: base seed not applied once"
	echo "$ARGV" | grep -q -- "$UNITS" || abort "$tag: the 35 TEST units 99990101..135 changed"
	echo "$ARGV" | grep -q -- "--obs-rate-" && abort "$tag: a rate-loop input flag is in the argv"
	echo "$ARGV" | grep -q -- "--obs-cascade-dhat" && abort "$tag: an MPCOF observer flag is in the argv"
	echo "$ARGV" | grep -q -- "--yaw-trim-ff" && abort "$tag: --yaw-trim-ff (F) is in the argv"
	echo "$ARGV" | grep -q -- "--refine-fitness" && abort "$tag: refine fitness must be off"
	echo "$ARGV" | grep -q -- "--output-mode ctbr " || abort "$tag: not --output-mode ctbr"
	[ "$(count "$ARGV" "--ctbr-delta-law ")" = 1 ] || abort "$tag: lacks the ctbr delta law"
	echo "$ARGV" | grep -q -- "--yaw-centred-ladder dense" || abort "$tag: dense yaw-error ladder (L) missing"
	echo "$ARGV" | grep -q -- "--select-tail p99" || abort "$tag: rule P missing"
	echo "$ARGV" | grep -q -- "--num-eval-folds 5 " || abort "$tag: folds must be 5"
	[ "$(count "$ARGV" "--delta-max-yaw 0.5 ")" = 1 ] || abort "$tag: must carry --delta-max-yaw 0.5 exactly once"
	[ "$(count "$ARGV" "--delta-max-yaw ")" = 1 ] || abort "$tag: more than one --delta-max-yaw"
	[ "$(count "$ARGV" "--ctbr-gamma")" = "$(count "$flag" "--ctbr-gamma")" ] || abort "$tag: --ctbr-gamma count wrong for arm $arm"
	[ "$(count "$ARGV" "--gyro-z-ladder")" = "$(count "$flag" "--gyro-z-ladder")" ] || abort "$tag: --gyro-z-ladder count wrong for arm $arm"
	vc=$(validate "$cline"); v=$(validate "$ARGV")
	VAL_CTL=$vc; VAL_ARM=$v
	[ "${vc% | *}" = "OK $((31337000 + s)) 0.5 $N0_GAMMA | $N0_CALIB | 5 35 99990101 99990135" ] \
		|| abort "$ctl: the N0 s$s control does not resolve to N0: $vc"
	want="OK $((31337000 + s)) 0.5 $(gamma_of $arm) | $(calib_of $arm) | 5 35 99990101 99990135 | ${vc##* | }"
	[ "$v" = "$want" ] || abort "$tag: phased_ga refuses / mis-resolves the arm (want '$want'): $v"
}

# --- preflight ---
git ls-files --error-unmatch $PREREG >/dev/null 2>&1 || abort "prereg $PREREG is not committed"
git diff --quiet HEAD -- $PREREG || abort "prereg $PREREG has uncommitted edits"
grep -q "$PREREG_HDR" $PREREG || abort "prereg n=10 addendum missing in $PREREG"
git ls-files --error-unmatch $PIN >/dev/null 2>&1 || abort "the CTRL-71 flag-off pin $PIN is not committed"
git diff --quiet HEAD -- src/wnn/control || abort "src/wnn/control has uncommitted edits"
[ -f $L/$VOID.done ] && abort "the void tag $VOID has a .done — the void premise is wrong, stop"
PREFLIGHT=1
for s in $SEEDS; do
	for arm in $(arms_of $s); do
		tag=$(tag_of $s $arm)
		[ "$tag" = "$VOID" ] && abort "refusing the VOID tag $VOID"
		[ -e $L/$tag.out ] && [ "$DRY" != 1 ] && abort "$tag.out already exists (never repeat a recipe on a seed / reuse a tag)"
		build $s $arm
		if [ "$DRY" = 1 ]; then
			echo "DRY $tag  (diff vs the flown s1 row $SRC; control of the pair = $(ctl_of $s))"
			diff <(grep -a '^\[argv\] ' $L/$SRC.out | cut -c8- | tr ' ' '\n') <(echo "$ARGV" | tr ' ' '\n')
			echo "validate control: $VAL_CTL"
			echo "validate run:     $VAL_ARM"
		fi
	done
done
PREFLIGHT=0
[ "$DRY" = 1 ] && exit 0

LOCK=/tmp/ab_gt7172_n10.pid
[ -f $LOCK ] && kill -0 "$(cat $LOCK)" 2>/dev/null && abort "another ab_gt7172_n10.sh is already running (pid $(cat $LOCK))"
echo $$ > $LOCK
log "GT10 armed (pid $$): CTRL-71/72 n=10 extension, per seed N0 sN (s9r, s10), G sN, T sN, G+T sN, seeds $SEEDS (17 runs) on ABI $ABI flags-off; all argv validated vs the s1 row + N0 sN"
while ctrl_busy; do sleep 30; done
log "GT10 START n=10 extension: N0 (s9r, s10) + G / T / G+T s6-s10, ABI $ABI flags-off (prereg $PREREG, n=10 addendum)"
for s in $SEEDS; do
	for arm in $(arms_of $s); do
		tag=$(tag_of $s $arm)
		[ -f $L/$tag.done ] && continue
		[ -e $L/$tag.out ] && abort "$tag.out exists without .done (void flight) — never reuse a tag"
		git diff --quiet HEAD -- src/wnn/control || abort "src/wnn/control changed under the block (before $tag)"
		build $s $arm
		while ctrl_busy; do sleep 30; done
		(cd /tmp && PYTHONPATH=$ROOT/src $VENV/bin/python -c "import ram_controller as c, wnn.control._accel as f; assert c.ABI_VERSION == $ABI and f.EXPECTED_ABI == $ABI") \
			|| abort "ram_controller / facade is not ABI $ABI (before $tag)"
		case $arm in
			n0) what="N0 n0 = N0 model-free control, n=10 extension (first completed N0 flight on s$s), P, floor";;
			g) what="G n0g3 = N0 + yaw bank gamma 3, n=10 ext, P, floor";;
			t) what="T n0gz = N0 + fixed gyro_z rate ladder, n=10 ext, P, floor";;
			gt) what="G+T n0g3gz = N0 + yaw bank gamma 3 + fixed gyro_z rate ladder, n=10 ext, P, floor";;
		esac
		log "START $tag [$what] (ABI $ABI, seed $((31337000 + s)), 35 units)"
		eval "set -- $ARGV"
		PYTHONPATH=src/wnn nice -n 5 /usr/bin/time -l $VENV/bin/python -u -m wnn.control.phased_ga "$@" > $L/$tag.out 2>&1
		rc=$?
		log "END $tag rc=$rc $(row $L/$tag.out)"
		[ $rc -eq 0 ] && touch $L/$tag.done
		[ $arm = n0 ] && [ $rc -ne 0 ] && abort "the N0 s$s control $tag ended rc=$rc — arms not flown without their control; escalate"
	done
done
log "GT10 DONE"
