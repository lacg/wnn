#!/bin/bash
# CTRL-72 (G) + CTRL-71 (T) + the combination G+T: levers 2 and 1 of the CTRL-70 gap analysis, flown in TANDEM
# (Luiz 23/11/2026; G+T added ~18:00 EST), per seed G sN, T sN, G+T sN, all vs the flown N0 control (CTRL-67):
#   G+T = N0 + --ctbr-gamma 1 2 2 3 + --gyro-z-ladder dense                                          tags ..._yld_n0g3gz_full_sN
# Takes over from ab_gt7172.sh (GT_OLD_PID) BETWEEN two flights: the old driver logs the END + .done of the run it
# is flying, then is stopped before it launches a run that would break the order (its G s1, T s1 are this order
# too; it must not reach G s2). No flight is interrupted or launched twice.
#   G = N0 + --ctbr-gamma 1 2 2 3    (yaw bank alphabet exponent 2 -> 3; --delta-max-yaw 0.5 kept)   tags ..._yld_n0g3_full_sN
#   T = N0 + --gyro-z-ladder dense   (fixed gyro_z rate ladder +-0.011/0.05/0.15/0.45 rad/s)         tags ..._yld_n0gz_full_sN
# Each sN argv = the flown N0 s1 [argv] line (..._a61_..._L1_yld_n0_full_s1) with ONLY --base-seed 31337001 ->
# 31337000+N, the two save paths (tag) and the arm's ONE flag. Checked by fixed text twice: arm flag + seed + tag
# swapped back must EQUAL the s1 source line, and arm flag + tag swapped back must EQUAL the flown N0 sN [argv]
# line (the control of the pair). Checked through phased_ga's own resolver: vs the control's resolution, G moves
# only the ctbr gamma (1,2,2,2 -> 1,2,2,3), T only the calibration token (+ ,gyroz:dense), G+T both.
# Wheel: ABI 62, rate-loop flags OFF (the lineage of N0 s1-s8). T is a Python-side threshold change, no rebuild;
# flag off == N0 is pinned by tests/controller_ctrl71_gyro_z_ladder.py (must be committed, src/wnn/control clean).
# Prereg: docs/ctrl71_72_levers_prereg.md (must be committed, no uncommitted edits).
# START/END lines in runs.log; ends 'GT DONE' (abort 'GT ABORT: why'). Never reuses a tag whose .out exists.
# Dry run: GT_DRY=1 prints each argv's diff vs the flown s1 row + the resolver lines, flies nothing.
cd /Users/lacg/wnn || exit 1
ROOT=/Users/lacg/wnn
L=logs/controller/ctrl48
RUNS=$L/runs.log
VENV=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv
ABI=62
SEEDS=${GT_SEEDS:-1 2 3 4 5}
DRY=${GT_DRY:-0}
PREREG=docs/ctrl71_72_levers_prereg.md
PREREG_HDR="^## Addendum (23/11/2026, before any G+T flight): third arm G+T"
OLD_PID=${GT_OLD_PID:-26648}
ARMS="g t gt"
PIN=tests/controller_ctrl71_gyro_z_ladder.py
UNITS="--test-seeds $(seq -f %.0f 99990101 99990135 | tr "\n" " ")"
N0_GAMMA="(1.0, 2.0, 2.0, 2.0)"
N0_CALIB="eps:200,regime:flown,ladder:quantile,yaw:dense,tilt:dense"
log() { echo "$(date -u +%FT%TZ) $*" >> $RUNS; }
abort() { [ "$DRY" = 1 ] || log "GT ABORT: $1"; echo "GT ABORT: $1" >&2; exit 1; }
row()
{
	local g r
	g=$(grep -a 'stage-select\] HEADLINE stage' $1 | grep -oE 'genome=[^ ]+' | cut -d= -f2)
	r=$(grep -a "HEADLINE-$g MULTI-SEED" $1 | tail -1)
	[ -z "$r" ] && r=$(grep -aE "\[(test|report)-seeds\] ${g%%#*} MULTI-SEED" $1 | tail -1)
	echo "genome=$g $(echo "$r" | grep -oE 'stable=[^ ]+  err=[^ ]+  steady=[^ ]+|alt=[^ ]+|pos=[^ ]+|pos_hold=[^ ]+|dz_hold=[^ ]+|stable_fail=[^ ]+' | tr '\n' ' ')"
}
suffix() { case $1 in g) echo n0g3;; t) echo n0gz;; gt) echo n0g3gz;; esac; }
flag_of() { case $1 in g) echo "--ctbr-gamma 1 2 2 3";; t) echo "--gyro-z-ladder dense";; gt) echo "--ctbr-gamma 1 2 2 3 --gyro-z-ladder dense";; esac; }
gamma_of() { [ $1 = t ] && echo "$N0_GAMMA" || echo "(1.0, 2.0, 2.0, 3.0)"; }
calib_of() { [ $1 = g ] && echo "$N0_CALIB" || echo "$N0_CALIB,gyroz:dense"; }
old_alive() { kill -0 $OLD_PID 2>/dev/null && ps -o command= -p $OLD_PID | grep -q "ab_gt7172.s[h]"; }
flying() { pgrep -fl "MacOS/Python -u -m wnn.control.phased_g[a]" | grep -q -- "$1_winner.yaml.gz"; }
SRC=C-S2-d_wposx2_a61_rf_c200_dc_biasin_ms01_P_both_L1_yld_n0_full_s1
tag_of() { echo C-S2-d_wposx2_a${ABI}_rf_c200_dc_biasin_ms01_P_both_L1_yld_$(suffix $2)_full_s$1; }
# the flown N0 control of seed s (s1-s5 on ABI 61, s6-s8 on ABI 62; exactly one must be finished)
ctl_of()
{
	local t n=0 hit=
	for a in 61 62; do
		t=C-S2-d_wposx2_a${a}_rf_c200_dc_biasin_ms01_P_both_L1_yld_n0_full_s$1
		[ -f $L/$t.done ] && { hit=$t; n=$((n + 1)); }
	done
	[ $n = 1 ] && echo $hit
}
count() { echo "$1" | grep -o -- "$2" | wc -l | tr -d ' '; }
ctrl_busy() { pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null; }
# validate(argv): phased_ga's own resolution chain (parse -> episode_config_from_args -> _resolve_ctbr -> calibration)
# prints: OK seed delta_max_yaw gamma | calib token | folds n_test first last | digest of every OTHER resolved setting
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
# build(s, arm) -> sets ARGV (the arm's flight) and CTL_ARGV (the flown control's [argv] line), or aborts
build()
{
	local s=$1 arm=$2 tag sline back ctl cline flag v vc want
	tag=$(tag_of $s $arm); flag=$(flag_of $arm)
	[ -f $L/$SRC.done ] || abort "missing finished s1 source run $SRC"
	sline=$(grep -a '^\[argv\] ' $L/$SRC.out | cut -c8-)
	[ "$(echo "$sline" | grep -c .)" = 1 ] || abort "$SRC: expected exactly one [argv] line"
	[ "$(count "$sline" "--base-seed 31337001 ")" = 1 ] || abort "$SRC: source does not carry --base-seed 31337001 once"
	echo "$sline" | grep -qE -- "--ctbr-gamma|--gyro-z-ladder" && abort "$SRC: the control already carries an arm flag"
	ARGV=$($VENV/bin/python $L/rebuild_argv.py $L/$SRC.out fly $L/ckpt_$tag $L/${tag}_winner.yaml.gz $flag) \
		|| abort "argv rebuild failed for $tag"
	ARGV=${ARGV/--base-seed 31337001 /--base-seed $((31337000 + s)) }
	[ "$(count "$ARGV" " $flag ")" = 1 ] || abort "$tag: the arm flag '$flag' is not in the argv exactly once"
	# fixed text 1: arm flag removed, seed + tag swapped back == the flown s1 source argv, byte for byte
	back=${ARGV/ $flag / }
	back=${back/--base-seed $((31337000 + s)) /--base-seed 31337001 }
	back=${back//$tag/$SRC}
	[ "$back" = "$sline" ] || abort "$tag: argv minus arm flag+seed+tag is not $SRC's [argv] (diff: $(diff <(echo "$back" | tr ' ' '\n') <(echo "$sline" | tr ' ' '\n') | tr '\n' ' '))"
	# fixed text 2: arm flag removed, tag swapped back == the flown N0 sN control's argv, byte for byte
	ctl=$(ctl_of $s); [ -n "$ctl" ] || abort "s$s: no single finished N0 control row (a61/a62 ..._yld_n0_full_s$s)"
	cline=$(grep -a '^\[argv\] ' $L/$ctl.out | cut -c8-)
	back=${ARGV/ $flag / }
	back=${back//$tag/$ctl}
	[ "$back" = "$cline" ] || abort "$tag: argv minus arm flag+tag is not the flown control $ctl's [argv]"
	CTL_ARGV=$cline
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
	[ $arm = g ] && [ "$(count "$ARGV" "--gyro-z-ladder")" != 0 ] && abort "$tag: G carries the T flag"
	[ $arm = t ] && [ "$(count "$ARGV" "--ctbr-gamma")" != 0 ] && abort "$tag: T carries the G flag"
	[ $arm = gt ] && [ "$(count "$ARGV" "--gyro-z-ladder dense ")$(count "$ARGV" "--ctbr-gamma 1 2 2 3 ")" != 11 ] && abort "$tag: G+T must carry both arm flags once"
	# resolver: the control resolves to the N0 gamma + calib token; the arm moves ONLY its own field
	vc=$(validate "$CTL_ARGV"); v=$(validate "$ARGV")
	VAL_CTL=$vc; VAL_ARM=$v
	[ "${vc% | *}" = "OK $((31337000 + s)) 0.5 $N0_GAMMA | $N0_CALIB | 5 35 99990101 99990135" ] \
		|| abort "$ctl: the control does not resolve to N0: $vc"
	want="OK $((31337000 + s)) 0.5 $(gamma_of $arm) | $(calib_of $arm) | 5 35 99990101 99990135 | ${vc##* | }"
	[ "$v" = "$want" ] || abort "$tag: phased_ga refuses / mis-resolves the arm (want '$want'): $v"
}

# --- preflight: prereg + lever-1 code committed, every argv built, cross-checked and resolved BEFORE arming ---
git ls-files --error-unmatch $PREREG >/dev/null 2>&1 || abort "prereg $PREREG is not committed"
git diff --quiet HEAD -- $PREREG || abort "prereg $PREREG has uncommitted edits"
grep -q "$PREREG_HDR" $PREREG || abort "prereg lever-1 addendum (pin + VAL pre-check) missing in $PREREG"
git ls-files --error-unmatch $PIN >/dev/null 2>&1 || abort "the CTRL-71 flag-off pin $PIN is not committed"
git diff --quiet HEAD -- src/wnn/control || abort "src/wnn/control has uncommitted edits (the pinned tree is not the flown tree)"
for s in $SEEDS; do
	for arm in $ARMS; do
		tag=$(tag_of $s $arm)
		# a tag with an .out is acceptable only if finished (.done: skipped below) or in the air under the old driver
		if [ -e $L/$tag.out ] && [ ! -f $L/$tag.done ] && [ "$DRY" != 1 ]; then
			old_alive && flying $tag || abort "$tag.out already exists and is neither finished nor the old driver's flight (never reuse a tag)"
		fi
		build $s $arm
		if [ "$DRY" = 1 ]; then
			echo "DRY $(tag_of $s $arm)  (diff vs the flown s1 row $SRC; control of the pair = $(ctl_of $s))"
			diff <(grep -a '^\[argv\] ' $L/$SRC.out | cut -c8- | tr ' ' '\n') <(echo "$ARGV" | tr ' ' '\n')
			echo "validate control: $VAL_CTL"
			echo "validate arm:     $VAL_ARM"
		fi
	done
done
[ "$DRY" = 1 ] && exit 0

LOCK=/tmp/ab_gt7172b.pid
[ -f $LOCK ] && kill -0 "$(cat $LOCK)" 2>/dev/null && abort "another ab_gt7172b.sh is already running (pid $(cat $LOCK))"
echo $$ > $LOCK
log "GT armed (pid $$): 3 arms per seed G sN, T sN, G+T sN (G+T = N0 + --ctbr-gamma 1 2 2 3 + --gyro-z-ladder dense, tags ..._yld_n0g3gz_full_sN), seeds $SEEDS on ABI $ABI flags-off; all argv validated vs the flown N0 rows; taking over from ab_gt7172.sh pid $OLD_PID between flights"
# --- handover: stop the old driver right after it has logged an END + .done, before its next launch ---
# Its G s1 and T s1 are this driver's order too, so either .done is a valid handover point; G s2 is not.
for arm in g t; do
	old_alive || break
	tag=$(tag_of 1 $arm)
	while old_alive && [ ! -f $L/$tag.done ]; do sleep 0.2; done
	old_alive || break
	kill -STOP $OLD_PID; sleep 2
	if ! ctrl_busy; then
		kill -TERM $OLD_PID; kill -CONT $OLD_PID
		while kill -0 $OLD_PID 2>/dev/null; do sleep 0.2; done
		log "GT HANDOVER: ab_gt7172.sh pid $OLD_PID stopped after the END of $tag, before its next launch; ab_gt7172b.sh pid $$ continues"
		break
	fi
	kill -CONT $OLD_PID   # it had already launched its next run: let it fly, hand over at that run's END
	[ $arm = t ] && abort "handover missed twice: the old driver pid $OLD_PID launched a run after T s1 — left running, order broken, escalate"
done
old_alive && abort "the old driver pid $OLD_PID is still alive after the handover"
while ctrl_busy; do sleep 30; done
for arm in g t; do   # a run the old driver left without .done (it died mid-flight) is void: stop, never relaunch a tag
	tag=$(tag_of 1 $arm)
	[ -e $L/$tag.out ] && [ ! -f $L/$tag.done ] && abort "$tag.out has no .done after the handover (the old driver did not log its END)"
done
log "GT START (3 arms) G = N0 + yaw gamma 3, T = N0 + fixed gyro_z rate ladder, G+T = both, per seed G, T, G+T, seeds $SEEDS, ABI $ABI flags-off (prereg $PREREG)"
for s in $SEEDS; do
	for arm in $ARMS; do
		tag=$(tag_of $s $arm)
		[ -f $L/$tag.done ] && continue
		[ -e $L/$tag.out ] && abort "$tag.out exists without .done (void flight) — never reuse a tag"
		git diff --quiet HEAD -- src/wnn/control || abort "src/wnn/control changed under the block (before $tag)"
		build $s $arm
		while ctrl_busy; do sleep 30; done
		(cd /tmp && PYTHONPATH=$ROOT/src $VENV/bin/python -c "import ram_controller as c, wnn.control._accel as f; assert c.ABI_VERSION == $ABI and f.EXPECTED_ABI == $ABI") \
			|| abort "ram_controller / facade is not ABI $ABI (before $tag)"
		case $arm in
			g) what="G n0g3 = N0 + yaw bank gamma 3 (finer small yaw steps, same largest step), P, floor";;
			t) what="T n0gz = N0 + fixed gyro_z rate ladder (yaw rate no longer saturated), P, floor";;
			gt) what="G+T n0g3gz = N0 + yaw bank gamma 3 + fixed gyro_z rate ladder (do the two levers add up?), P, floor";;
		esac
		log "START $tag [$what] (ABI $ABI, seed $((31337000 + s)), 35 units)"
		eval "set -- $ARGV"
		PYTHONPATH=src/wnn nice -n 5 /usr/bin/time -l $VENV/bin/python -u -m wnn.control.phased_ga "$@" > $L/$tag.out 2>&1
		rc=$?
		log "END $tag rc=$rc $(row $L/$tag.out)"
		[ $rc -eq 0 ] && touch $L/$tag.done
	done
done
log "GT DONE"
