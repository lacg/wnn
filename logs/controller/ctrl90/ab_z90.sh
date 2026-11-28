#!/bin/bash
# CTRL-90 (Luiz 28/11/2026): l2bnmwz = l2bnmw TRAINED ON the Zhang golden cell's ranges (ctrl83 prereg Amendment 3),
# prereg docs/ctrl90_zhang_trained_prereg.md. Derived from ctrl87/ab_nm87.sh (same checks).
#   l2bnmwz = l2bnmw TRAINED ON THE FULL golden cell `all` (ABI 70, training-only flags: TEST / VAL keep l2bnmw's):
#             --train-motor-asym-mag 0.30 -> --train-motor-asym-range 0.64 1.44   (in place; the cell's exact factor)
#             --train-motor-lag-range 0.15 0.30 -> 0.16 0.24   (in place; exactly the cell's T_e range, NARROWER)
#             + --train-mass-jitter 0.38 --train-noise-scale 1.2 --train-inertia-jitter 0.20 --train-latency-range 8 12
#             (appended: the reference carries none of them; rebuild_argv.py refuses any it does)
# Reference of every argv = the FLOWN l2bnmw sN run (..._a68_..._gt_l2bnmw_full_sN). Arm argv = that run's DEDUPED
# [argv] with the 2 flags replaced in place + the 4 appended + the two save paths (tag ..._a70_..._gt_l2bnmwz_full_sN).
# Fixed-text check: arm argv with the 2 flags reverted, the 4 dropped, tag -> l2bnmw tag == the l2bnmw DEDUPED [argv]
# byte for byte. Resolver check (phased_ga's own parser + episode_config + _resolve_ctbr + resolve_block10_args +
# calibration token): l2bnmw resolves to anchor nominal, train per_episode:0.3 uniform:0.15,0.3 - -; the arm moves
# ONLY the train tokens (wear/lag/plant/jitter); calibration token + digest of everything else equal.
# Pairing: l2bnmwz sN vs l2bnmw sN, rule A-D read on the golden cell `all` (prereg). Seed-major, one run at a time;
# waits for any other controller run. Wheel ram_controller + facade ABI $ABI (70) asserted before every flight;
# src/wnn/control must be clean; the prereg must be committed. A tag whose .out exists is never reused.
# START/END in runs.log; block lines 'Z90 ...'; ends 'Z90 DONE' (abort 'Z90 ABORT: why').
# After each END (CPU, nice 15, report-only): ctrl83 cells nom + all on the new winner (nom = gate G4 vs the logged
# END), then the SECONDARY cell allz90 (Zhang on the arm's OWN training ranges) on l2bnmwz sN AND l2bnmw sN.
# A readout failure is logged ('Z90 STRESS FAIL') and does NOT stop the block.
# Dry run: Z90_DRY=1 prints per seed the token diff vs its l2bnmw argv + both resolver lines; flies nothing.
# Z90_ROOT / Z90_PP / Z90_L (dry runs only) default to the live checkout; Z90_PP may prepend an unpacked wheel dir.
ROOT=${Z90_ROOT:-/Users/lacg/wnn}
cd $ROOT || exit 1
SAVE=logs/controller/ctrl48  # the save paths phased_ga writes (relative to ROOT, as every flown argv carries)
L=${Z90_L:-$SAVE}
RUNS=$L/runs.log
VENV=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv
ABI=70
ALL_ARMS="l2bnmwz"
SEEDS=${Z90_SEEDS:-1 2 3}
ARMS=${Z90_ARMS:-l2bnmwz}
DRY=${Z90_DRY:-0}
PREREG=docs/ctrl90_zhang_trained_prereg.md
PREREG_HDR="^## Rule"
PINS="tests/controller_ctrl83_flag_off.py tests/controller_ctrl87_flag_off.py"
UNITS="--test-seeds $(seq -f %.0f 99990101 99990135 | tr "\n" " ")"
CALIB="eps:200,regime:flown,ladder:quantile,yaw:dense,tilt:dense,gyroz:dense,gyroxy:dense_hold"
# The arm's 2 in-place replacements: reference (l2bnmw) -> arm, at the position l2bnmw carries them; then the 4
# appended training-only flags (the full golden cell `all`, cells.ALL_LEVELS: mass 0.38, noise x1.2, inertia 0.20,
# latency 8-12 ms).
LAG_FLAG=--train-motor-lag-range
REF_WEAR="--train-motor-asym-mag 0.30"
ARM_WEAR="--train-motor-asym-range 0.64 1.44"
REF_LAG="$LAG_FLAG 0.15 0.30"
ARM_LAG="$LAG_FLAG 0.16 0.24"
EXTRA="--train-mass-jitter 0.38 --train-noise-scale 1.2 --train-inertia-jitter 0.20 --train-latency-range 8 12"
# Resolved train tokens: asym, lag, plant (mass/noise/asym-range), jitter (inertia/latency); '-' = off.
REF_TRAIN="per_episode:0.3 uniform:0.15,0.3 - -"
ARM_TRAIN="per_episode:0.384615 uniform:0.16,0.24 mass:0.38+noise:x1.2+asym:0.64,1.44 inertia:0.2+latency:8,12ms"
# Reported cells (ctrl83 Amendment 3): nom (in-range + G4) and `all` (the Zhang golden cell; the rule reads here).
# Secondary (report-only): allz90 = Zhang on the arm's OWN training ranges, scored on the arm AND its l2bnmw pair.
PRIMARY=nom,all
SECONDARY=allz90
PFX=C-S2-d_wposx2
MID=rf_c200_dc_biasin_ms01_P_both_L1_yld
PP=${Z90_PP:-$ROOT/src}
log() { echo "$(date -u +%FT%TZ) $*" >> $RUNS; }
abort() { [ "$DRY" = 1 ] || log "Z90 ABORT: $1"; echo "Z90 ABORT: $1" >&2; exit 1; }
row()
{
	local g r
	g=$(grep -a 'stage-select\] HEADLINE stage' $1 | grep -oE 'genome=[^ ]+' | cut -d= -f2)
	r=$(grep -a "HEADLINE-$g MULTI-SEED" $1 | tail -1)
	[ -z "$r" ] && r=$(grep -aE "\[(test|report)-seeds\] ${g%%#*} MULTI-SEED" $1 | tail -1)
	echo "genome=$g $(echo "$r" | grep -oE 'stable=[^ ]+  err=[^ ]+  steady=[^ ]+|alt=[^ ]+|pos=[^ ]+|pos_hold=[^ ]+|dz_hold=[^ ]+|stable_fail=[^ ]+' | tr '\n' ' ')"
}
what_of() { echo "l2bnmw trained ON the full Zhang golden cell (wear U(0.64,1.44) + mass ±0.38 + noise x1.2 + lag U(0.16,0.24) s + inertia ±0.20 + latency 8-12 ms): covered-range robustness on \`all\` vs l2bnmw (rule A-D on the golden cell)"; }
tag_of() { echo ${PFX}_a70_${MID}_gt_$2_full_s$1; }
ref_of() { echo ${PFX}_a68_${MID}_gt_l2bnmw_full_s$1; }
count() { echo "$1" | grep -o -- "$2" | wc -l | tr -d ' '; }
ctrl_busy() { pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null; }
abi_ok()
{
	(cd /tmp && PYTHONPATH=$PP $VENV/bin/python -c "import ram_controller as c, wnn.control._accel as f; assert c.ABI_VERSION == $ABI and f.EXPECTED_ABI == $ABI")
}
validate()
{
	(cd /tmp && PYTHONPATH=$PP $VENV/bin/python -c "
import hashlib, shlex, sys
from wnn.control import phased_ga as p
from wnn.control.evaluator import train_asym_token, train_lag_token, train_plant_token, train_jitter_token
a = p.build_arg_parser().parse_args(shlex.split(sys.argv[1]))
ec = p.episode_config_from_args(a); p._resolve_ctbr(a, ec); p.resolve_block10_args(a); c = a._ctbr
skip = ('anchor_mass', 'train_motor_asym_mag', 'train_motor_lag_range', 'train_motor_asym_range', 'train_mass_jitter', 'train_noise_scale', 'train_inertia_jitter', 'train_latency_range', 'save_stage_checkpoints', 'save_winner')
rest = repr(sorted((k, repr(v)) for k, v in c.items())) + repr(sorted((k, repr(v)) for k, v in vars(a).items() if not k.startswith('_') and k not in skip))
print('OK', a.base_seed, a.delta_max_yaw, a.delta_leak_yaw, a.gyro_xy_ladder, 'lag', ec.motor_lag_s, ec.model_motor_lag_s, 'test-wear', ec.disturbance.motor_asym_mag, '| anchor', a.anchor_mass, ec.anchor_mass, '| train', train_asym_token(ec), train_lag_token(ec) or 'fixed', train_plant_token(ec) or '-', train_jitter_token(ec) or '-', '|', p.calibration_from_args(a).token(), '|', a.num_eval_folds, len(a.test_seeds), a.test_seeds[0], a.test_seeds[-1], '|', hashlib.sha256(rest.encode()).hexdigest()[:16])" "$1" 2>&1 | tail -1)
}
# subst <line> <old> <new>: replace " old " by " new " exactly once, or abort
subst()
{
	[ "$(count "$1 " " $2 ")" = 1 ] || abort "'$2' not in the argv exactly once"
	local x="$1 "; x=${x/ $2 / $3 }; echo "${x% }"
}
# drop <line> <old>: remove " old" exactly once, or abort
drop()
{
	[ "$(count "$1 " " $2 ")" = 1 ] || abort "'$2' not in the argv exactly once"
	local x="$1 "; x=${x/ $2 / }; echo "${x% }"
}
# build(s, arm) -> sets ARGV, CLINE, VAL_REF, VAL_ARM, or aborts
build()
{
	local s=$1 arm=$2 tag ref back v vr want fly
	tag=$(tag_of $s $arm); ref=$(ref_of $s)
	[ -f $L/$ref.done ] || abort "s$s: the l2bnmw reference $ref has not finished"
	RAWLINE=$(grep -a '^\[argv\] ' $L/$ref.out | cut -c8-)
	[ "$(echo "$RAWLINE" | grep -c .)" = 1 ] || abort "$ref: expected exactly one [argv] line"
	CLINE=$($VENV/bin/python $L/rebuild_argv.py $L/$ref.out argv 2>/dev/null) || abort "$ref: argv dedupe failed"
	[ "$(validate "$RAWLINE")" = "$(validate "$CLINE")" ] || abort "$ref: the deduped l2bnmw argv does not resolve like the flown one"
	[ "$(count "$CLINE" "--base-seed $((31337000 + s)) ")" = 1 ] || abort "$ref: does not carry --base-seed $((31337000 + s)) once"
	[ "$(count "$CLINE" "--anchor-mass nominal ")" = 1 ] || abort "$ref: not the l2bnmw base (--anchor-mass nominal)"
	echo "$CLINE" | grep -qE -- "--train-inertia-jitter|--train-latency-range|--train-mass-jitter|--train-noise-scale|--train-motor-asym-range|--obs-yaw-lead|--motor-lag-down-s|--model-motor-lag" && abort "$ref: the reference carries an unexpected flag"
	fly=$($VENV/bin/python $L/rebuild_argv.py $L/$ref.out fly $SAVE/ckpt_$tag $SAVE/${tag}_winner.yaml.gz $EXTRA 2>/dev/null) \
		|| abort "argv rebuild failed for $tag"
	ARGV=$(subst "$fly" "$REF_WEAR" "$ARM_WEAR") || exit 1
	ARGV=$(subst "$ARGV" "$REF_LAG" "$ARM_LAG") || exit 1
	back=$(subst "$ARGV" "$ARM_WEAR" "$REF_WEAR") || exit 1
	back=$(subst "$back" "$ARM_LAG" "$REF_LAG") || exit 1
	back=$(drop "$back" "$EXTRA") || exit 1
	back=${back//$tag/$ref}
	[ "$back" = "$CLINE" ] || abort "$tag: argv with the 2 flags reverted, the 4 dropped + tag is not the l2bnmw s$s argv (diff: $(diff <(echo "$back" | tr ' ' '\n') <(echo "$CLINE" | tr ' ' '\n') | tr '\n' ' '))"
	echo "$ARGV" | grep -q -- "$UNITS" || abort "$tag: the 35 TEST units 99990101..135 changed"
	echo "$ARGV" | grep -q -- "--obs-cascade-dhat" && abort "$tag: an MPCOF observer flag is in the argv (model-free rule)"
	echo "$ARGV" | grep -q -- "--yaw-trim-ff" && abort "$tag: --yaw-trim-ff is in the argv (model-free rule)"
	echo "$ARGV" | grep -q -- "--refine-fitness" && abort "$tag: refine fitness must be off"
	echo "$ARGV" | grep -q -- "--output-mode ctbr " || abort "$tag: not --output-mode ctbr"
	[ "$(count "$ARGV" "--gyro-xy-ladder dense_hold ")" = 1 ] || abort "$tag: the l2b gyro_x/y dense_hold ladder missing"
	[ "$(count "$ARGV" "--disturbance L4C ")" = 1 ] || abort "$tag: not --disturbance L4C"
	[ "$(count "$ARGV" "--motor-lag-s 0.20 ")" = 1 ] || abort "$tag: not --motor-lag-s 0.20"
	[ "$(count "$ARGV" " --mass-jitter ")" = 0 ] || abort "$tag: --mass-jitter set (it would move the TEST units too)"
	[ "$(count "$ARGV" "--train-motor-asym-mag ")" = 0 ] || abort "$tag: --train-motor-asym-mag left in (exclusive with the range)"
	echo "$ARGV" | grep -q -- "--select-tail p99" || abort "$tag: rule P missing"
	echo "$ARGV" | grep -q -- "--num-eval-folds 5 " || abort "$tag: folds must be 5"
	vr=$(validate "$CLINE"); v=$(validate "$ARGV")
	VAL_REF=$vr; VAL_ARM=$v
	case "$vr" in "OK $((31337000 + s)) "*" dense_hold lag 0.2 None test-wear 0.2 | anchor nominal nominal | train $REF_TRAIN | $CALIB | 5 35 99990101 99990135 | "*) ;;
		*) abort "$ref: l2bnmw s$s does not resolve to the l2bnmw base: $vr";; esac
	want=${vr/| train $REF_TRAIN |/| train $ARM_TRAIN |}
	[ "$want" != "$vr" ] || abort "$ref: the reference train tokens are not '$REF_TRAIN'"
	[ "$v" = "$want" ] || abort "$tag: phased_ga refuses / mis-resolves the arm (want '$want'): $v"
}
grid()
{
	local tag=$1 k=$2 cells=$3 out
	out=$(cd /Users/lacg/wnn && PYTHONPATH=src/wnn nice -n 15 $VENV/bin/python logs/controller/ctrl83/run_grid.py \
		--controllers $k --cells $cells --no-row-b 2>&1)
	if [ $? -eq 0 ]; then
		log "Z90 STRESS $tag ($k, cells $cells): $(echo "$out" | grep '^\[ctrl83\]' | grep -v pending | tr '\n' ' ' | cut -c1-1500)"
	else
		log "Z90 STRESS FAIL $tag ($k, cells $cells): $(echo "$out" | tail -3 | tr '\n' ' ' | cut -c1-600)"
	fi
}
stress() { grid $1 ${2}_s$3 $PRIMARY; grid $1 ${2}_s$3,l2bnmw_s$3 $SECONDARY; }
# --- preflight ---
for a in $ARMS; do case " $ALL_ARMS " in *" $a "*) ;; *) abort "unknown arm '$a' in Z90_ARMS";; esac; done
git -C $ROOT ls-files --error-unmatch $PREREG >/dev/null 2>&1 || abort "prereg $PREREG is not committed"
git -C $ROOT diff --quiet HEAD -- $PREREG || abort "prereg $PREREG has uncommitted edits"
grep -q "$PREREG_HDR" $ROOT/$PREREG || abort "prereg rule section missing in $PREREG"
for p in $PINS; do git -C $ROOT ls-files --error-unmatch $p >/dev/null 2>&1 || abort "the flag-off pin $p is not committed"; done
git -C $ROOT diff --quiet HEAD -- src/wnn/control || abort "src/wnn/control has uncommitted edits"
abi_ok || abort "ram_controller / facade is not ABI $ABI"
N=0
for s in $SEEDS; do
	for arm in $ARMS; do
		tag=$(tag_of $s $arm)
		[ -f $L/$tag.done ] && continue
		[ -e $L/$tag.out ] && [ "$DRY" != 1 ] && abort "$tag.out already exists (never repeat a recipe on a seed / reuse a tag)"
		build $s $arm; N=$((N + 1))
		if [ "$DRY" = 1 ]; then
			echo "DRY $tag  (reference $(ref_of $s))"
			diff <(echo "$CLINE" | tr ' ' '\n') <(echo "$ARGV" | tr ' ' '\n') | grep '^[<>]' | grep -v -- "_full_s$s" | tr '\n' ' '; echo
			echo "  validate ref (l2bnmw): $VAL_REF"
			echo "  validate arm: $VAL_ARM"
		fi
	done
done
[ "$DRY" = 1 ] && { echo "DRY OK: $N runs validated (seeds: $SEEDS; arms: $ARMS)"; exit 0; }

LOCK=/tmp/ab_z90.pid
[ -f $LOCK ] && kill -0 "$(cat $LOCK)" 2>/dev/null && abort "another Z90 driver is already running (pid $(cat $LOCK))"
echo $$ > $LOCK
log "Z90 armed (pid $$, ab_z90.sh): CTRL-90 l2bnmw trained on the full Zhang golden cell, seeds [$SEEDS] x arms [$ARMS] = $N runs, seed-major, ABI $ABI; every argv validated vs its flown l2bnmw sN (prereg $PREREG)"
while ctrl_busy; do sleep 30; done
log "Z90 START seeds [$SEEDS] arms [$ARMS]"
for s in $SEEDS; do
	for arm in $ARMS; do
		tag=$(tag_of $s $arm)
		[ -f $L/$tag.done ] && continue
		[ -e $L/$tag.out ] && abort "$tag.out exists without .done (void flight) — never reuse a tag"
		git diff --quiet HEAD -- src/wnn/control || abort "src/wnn/control changed under the block (before $tag)"
		git diff --quiet HEAD -- $PREREG || abort "prereg edited under the block (before $tag)"
		build $s $arm
		while ctrl_busy; do sleep 30; done
		abi_ok || abort "ram_controller / facade is not ABI $ABI (before $tag)"
		log "START $tag [Z90 $arm = $(what_of $arm)] (ABI $ABI, seed $((31337000 + s)), 35 units; reference $(ref_of $s))"
		eval "set -- $ARGV"
		PYTHONPATH=src/wnn nice -n 5 /usr/bin/time -l $VENV/bin/python -u -m wnn.control.phased_ga "$@" > $L/$tag.out 2>&1
		rc=$?
		log "END $tag rc=$rc $(row $L/$tag.out)"
		[ $rc -eq 0 ] && { touch $L/$tag.done; stress $tag $arm $s; }
	done
done
rm -f $LOCK
log "Z90 DONE seeds [$SEEDS] arms [$ARMS]"
