#!/bin/bash
# CTRL-87 (Luiz 28/11/2026): nominal-mass anchor (D2 option a) + wider training spread on the l2b base, prereg
# docs/ctrl87_nominal_mass_wide_spread_prereg.md. Derived from the CTRL-80 driver (ctrl80/ab_ylead80.sh, same checks).
#   l2bnm  = l2b + --anchor-mass nominal                          (the FAIRNESS base: the WNN loses the true-mass anchor)
#   l2bnmw = l2bnm + the wider TRAINING spread (prereg Amendment 1, ABI 68): wear +-30% (--train-motor-asym-mag 0.30)
#            AND a per-episode plant lag T ~ U(0.15, 0.30) s (--train-motor-lag-range 0.15 0.30), coherent across every
#            memory write (DAgger + per-round eval, calibration sampler, MEMORY recorder) and the GA-search scoring pools;
#            the 35 TEST units, the VAL stage-select and every held-out row stay at L4C +-20% / lag 0.20, and the
#            teachers + the WNN's actuator twin stay at the nominal 0.20 s.
# Reference of every argv = the FLOWN l2b sN run (..._a63_..._gt_l2b_full_sN; ABI 63; never re-flown). Each arm argv
# = that run's DEDUPED [argv] + ONLY the arm's flags + the two save paths (tag ..._a66_..._gt_<arm>_full_sN).
# Fixed-text check: arm argv minus its flags, tag -> l2b tag == the l2b DEDUPED [argv] byte for byte (raw vs deduped
# l2b must resolve identically). Resolver check (phased_ga's own parser + episode_config + _resolve_ctbr +
# resolve_block10_args + calibration token): l2b resolves to anchor=true, train wear per_episode:0.2, lag 0.2; the arm
# moves ONLY anchor_mass (and, for l2bnmw, the training wear token); calibration token + digest of everything else equal.
# Pairing: l2bnm sN vs l2b sN (cost/fairness readout, no adopt/drop); l2bnmw sN vs l2bnm sN (rule A-D) — so l2bnmw sN
# never flies before l2bnm sN is .done.
# Stage-aware: N87_SEEDS (default "1 2 3") and N87_ARMS (default "l2bnm"; "l2bnm l2bnmw" once the flag lands).
# Order: seed-major. One run at a time; waits for any other controller run. Wheel ram_controller + facade ABI $ABI (68)
# asserted before every flight (ABI 68 flag-off == ABI 66, pinned: tests/controller_ctrl87_flag_off.py +
# ctrl87_train_spread_tests.rs; the l2bnm tags keep their _a66_ lineage name, l2bnmw is _a68_); src/wnn/control must be clean; the prereg must be committed. A tag whose .out exists is never
# reused. START/END in runs.log; block lines 'N87 ...'; ends 'N87 DONE' (abort 'N87 ABORT: why').
# After each END (CPU, nice 15, report-only): the CTRL-83 stress grid's 6 PRIMARY cells on the new winner
# (ctrl83/run_grid.py --controllers <arm>_sN --no-row-b); its nom cell re-checks the logged END (gate G4). A readout
# failure is logged ('N87 STRESS FAIL') and does NOT stop the block.
# Dry run: N87_DRY=1 prints, per arm, the token diff vs its l2b argv + both resolver lines; flies nothing, writes nothing.
# N87_ROOT / N87_PP (dry runs only: a worktree + an unpacked, never-installed wheel) default to the live checkout.
ROOT=${N87_ROOT:-/Users/lacg/wnn}
cd $ROOT || exit 1
L=logs/controller/ctrl48
RUNS=$L/runs.log
VENV=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv
ABI=68
ALL_ARMS="l2bnm l2bnmw"
SEEDS=${N87_SEEDS:-1 2 3}
ARMS=${N87_ARMS:-l2bnm l2bnmw}
DRY=${N87_DRY:-0}
PREREG=docs/ctrl87_nominal_mass_wide_spread_prereg.md
PREREG_HDR="^## Rule"
PINS="tests/controller_ctrl83_flag_off.py tests/controller_ctrl87_flag_off.py"
UNITS="--test-seeds $(seq -f %.0f 99990101 99990135 | tr "\n" " ")"
CALIB="eps:200,regime:flown,ladder:quantile,yaw:dense,tilt:dense,gyroz:dense,gyroxy:dense_hold"
WEAR_W=0.30
WEAR_FLAG=--train-motor-asym-mag
# The training lag range (s): the slow side reaches the lag30 stress cell (the WNN's failure mode: lag30 51 / lag40
# 448 fails, lag10 0) and the fast side Molchanov's nominal 0.15; lag40 stays out of range (an honest extrapolation
# cell) and the TEST units' 0.20 sits inside.
LAG_LO=0.15
LAG_HI=0.30
LAG_FLAG=--train-motor-lag-range
PRIMARY=nom,wear30,mass25,lag30,lag10,noise2
PFX=C-S2-d_wposx2
MID=rf_c200_dc_biasin_ms01_P_both_L1_yld
PP=${N87_PP:-$ROOT/src}
log() { echo "$(date -u +%FT%TZ) $*" >> $RUNS; }
abort() { [ "$DRY" = 1 ] || log "N87 ABORT: $1"; echo "N87 ABORT: $1" >&2; exit 1; }
row()
{
	local g r
	g=$(grep -a 'stage-select\] HEADLINE stage' $1 | grep -oE 'genome=[^ ]+' | cut -d= -f2)
	r=$(grep -a "HEADLINE-$g MULTI-SEED" $1 | tail -1)
	[ -z "$r" ] && r=$(grep -aE "\[(test|report)-seeds\] ${g%%#*} MULTI-SEED" $1 | tail -1)
	echo "genome=$g $(echo "$r" | grep -oE 'stable=[^ ]+  err=[^ ]+  steady=[^ ]+|alt=[^ ]+|pos=[^ ]+|pos_hold=[^ ]+|dz_hold=[^ ]+|stable_fail=[^ ]+' | tr '\n' ' ')"
}
flags_of()
{
	local f=""
	[ "$1" = l2bnm ] && f="--anchor-mass nominal"
	[ "$1" = l2bnmw ] && f="--anchor-mass nominal $WEAR_FLAG $WEAR_W $LAG_FLAG $LAG_LO $LAG_HI"
	echo "$f"
}
asym_of() { [ "$1" = l2bnmw ] && { echo "per_episode:0.3 uniform:0.15,0.3"; return; }; echo "per_episode:0.2 fixed"; }
what_of()
{
	[ "$1" = l2bnm ] && echo "l2b + collective anchor/coll_cmd at NOMINAL mass (fairness: the rivals' information; cost readout vs l2b)"
	[ "$1" = l2bnmw ] && echo "l2bnm + training/search spread wear +-30% and lag U(0.15,0.30) s (was +-20% / 0.20; TEST, VAL, twin, teachers unchanged): does a wider training spread cut the wear30 fails at no in-range cost (rule A')?"
}
# The tag's ABI is the arm's LINEAGE: l2bnm flew on ABI 66 (its seeds keep _a66_; 68 flag-off == 66, pinned), l2bnmw is 68.
tag_abi_of() { [ "$1" = l2bnmw ] && echo 68 || echo 66; }
tag_of() { echo ${PFX}_a$(tag_abi_of $2)_${MID}_gt_$2_full_s$1; }
ref_of() { echo ${PFX}_a63_${MID}_gt_l2b_full_s$1; }
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
from wnn.control.evaluator import train_asym_token, train_lag_token
a = p.build_arg_parser().parse_args(shlex.split(sys.argv[1]))
ec = p.episode_config_from_args(a); p._resolve_ctbr(a, ec); p.resolve_block10_args(a); c = a._ctbr
skip = ('anchor_mass', 'train_motor_asym_mag', 'train_motor_lag_range', 'save_stage_checkpoints', 'save_winner')
rest = repr(sorted((k, repr(v)) for k, v in c.items())) + repr(sorted((k, repr(v)) for k, v in vars(a).items() if not k.startswith('_') and k not in skip))
print('OK', a.base_seed, a.delta_max_yaw, a.delta_leak_yaw, a.gyro_xy_ladder, 'lag', ec.motor_lag_s, ec.model_motor_lag_s, 'test-wear', ec.disturbance.motor_asym_mag, '| anchor', a.anchor_mass, ec.anchor_mass, '| train', train_asym_token(ec), train_lag_token(ec) or 'fixed', '|', p.calibration_from_args(a).token(), '|', a.num_eval_folds, len(a.test_seeds), a.test_seeds[0], a.test_seeds[-1], '|', hashlib.sha256(rest.encode()).hexdigest()[:16])" "$1" 2>&1 | tail -1)
}
# build(s, arm) -> sets ARGV, CLINE, VAL_REF, VAL_ARM, or aborts
build()
{
	local s=$1 arm=$2 tag ref flags back v vr want
	tag=$(tag_of $s $arm); ref=$(ref_of $s); flags=$(flags_of $arm)
	[ -n "$flags" ] || abort "$arm: no flags (unknown arm)"
	[ -f $L/$ref.done ] || abort "s$s: the l2b reference $ref has not finished"
	RAWLINE=$(grep -a '^\[argv\] ' $L/$ref.out | cut -c8-)
	[ "$(echo "$RAWLINE" | grep -c .)" = 1 ] || abort "$ref: expected exactly one [argv] line"
	CLINE=$($VENV/bin/python $L/rebuild_argv.py $L/$ref.out argv 2>/dev/null) || abort "$ref: argv dedupe failed"
	[ "$(validate "$RAWLINE")" = "$(validate "$CLINE")" ] || abort "$ref: the deduped l2b argv does not resolve like the flown one"
	[ "$(count "$CLINE" "--base-seed $((31337000 + s)) ")" = 1 ] || abort "$ref: does not carry --base-seed $((31337000 + s)) once"
	echo "$CLINE" | grep -qE -- "--anchor-mass|$WEAR_FLAG|$LAG_FLAG|--obs-yaw-lead|--motor-lag-down-s|--model-motor-lag" && abort "$ref: the reference already carries an arm flag"
	ARGV=$($VENV/bin/python $L/rebuild_argv.py $L/$ref.out fly $L/ckpt_$tag $L/${tag}_winner.yaml.gz $flags 2>/dev/null) \
		|| abort "argv rebuild failed for $tag"
	[ "$(count "$ARGV" " $flags ")" = 1 ] || abort "$tag: the arm flags '$flags' not in the argv exactly once"
	back=${ARGV/ $flags / }
	back=${back//$tag/$ref}
	[ "$back" = "$CLINE" ] || abort "$tag: argv minus flags+tag is not the l2b s$s argv (diff: $(diff <(echo "$back" | tr ' ' '\n') <(echo "$CLINE" | tr ' ' '\n') | tr '\n' ' '))"
	echo "$ARGV" | grep -q -- "$UNITS" || abort "$tag: the 35 TEST units 99990101..135 changed"
	echo "$ARGV" | grep -q -- "--obs-cascade-dhat" && abort "$tag: an MPCOF observer flag is in the argv (model-free rule)"
	echo "$ARGV" | grep -q -- "--yaw-trim-ff" && abort "$tag: --yaw-trim-ff is in the argv (model-free rule)"
	echo "$ARGV" | grep -q -- "--refine-fitness" && abort "$tag: refine fitness must be off"
	echo "$ARGV" | grep -q -- "--output-mode ctbr " || abort "$tag: not --output-mode ctbr"
	[ "$(count "$ARGV" "--gyro-xy-ladder dense_hold ")" = 1 ] || abort "$tag: the l2b gyro_x/y dense_hold ladder missing"
	[ "$(count "$ARGV" "--disturbance L4C ")" = 1 ] || abort "$tag: not --disturbance L4C"
	[ "$(count "$ARGV" "--motor-lag-s 0.20 ")" = 1 ] || abort "$tag: not --motor-lag-s 0.20"
	echo "$ARGV" | grep -q -- "--select-tail p99" || abort "$tag: rule P missing"
	echo "$ARGV" | grep -q -- "--num-eval-folds 5 " || abort "$tag: folds must be 5"
	vr=$(validate "$CLINE"); v=$(validate "$ARGV")
	VAL_REF=$vr; VAL_ARM=$v
	case "$vr" in "OK $((31337000 + s)) "*" dense_hold lag 0.2 None test-wear 0.2 | anchor true true | train per_episode:0.2 fixed | $CALIB | 5 35 99990101 99990135 | "*) ;;
		*) abort "$ref: l2b s$s does not resolve to the l2b base: $vr";; esac
	want=${vr/| anchor true true | train per_episode:0.2 fixed |/| anchor nominal nominal | train $(asym_of $arm) |}
	[ "$v" = "$want" ] || abort "$tag: phased_ga refuses / mis-resolves the arm (want '$want'): $v"
}
stress()
{
	local tag=$1 k=$2 out
	out=$(cd /Users/lacg/wnn && PYTHONPATH=src/wnn nice -n 15 $VENV/bin/python logs/controller/ctrl83/run_grid.py \
		--controllers $k --cells $PRIMARY --no-row-b 2>&1)
	if [ $? -eq 0 ]; then
		log "N87 STRESS $tag ($k, primary cells $PRIMARY): $(echo "$out" | grep '^\[ctrl83\]' | grep -v pending | tr '\n' ' ' | cut -c1-1500)"
	else
		log "N87 STRESS FAIL $tag ($k): $(echo "$out" | tail -3 | tr '\n' ' ' | cut -c1-600)"
	fi
}

# --- preflight ---
for a in $ARMS; do case " $ALL_ARMS " in *" $a "*) ;; *) abort "unknown arm '$a' in N87_ARMS";; esac; done
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
			echo "  validate l2b: $VAL_REF"
			echo "  validate arm: $VAL_ARM"
		fi
	done
done
[ "$DRY" = 1 ] && { echo "DRY OK: $N runs validated (seeds: $SEEDS; arms: $ARMS)"; exit 0; }

LOCK=/tmp/ab_nm87.pid
[ -f $LOCK ] && kill -0 "$(cat $LOCK)" 2>/dev/null && abort "another N87 driver is already running (pid $(cat $LOCK))"
echo $$ > $LOCK
log "N87 armed (pid $$, ab_nm87.sh): CTRL-87 nominal-mass anchor + wide spread, seeds [$SEEDS] x arms [$ARMS] = $N runs, seed-major, ABI $ABI; every argv validated vs its flown l2b sN (prereg $PREREG)"
while ctrl_busy; do sleep 30; done
log "N87 START seeds [$SEEDS] arms [$ARMS]"
for s in $SEEDS; do
	for arm in $ARMS; do
		tag=$(tag_of $s $arm)
		[ -f $L/$tag.done ] && continue
		[ -e $L/$tag.out ] && abort "$tag.out exists without .done (void flight) — never reuse a tag"
		[ "$arm" = l2bnmw ] && { [ -f $L/$(tag_of $s l2bnm).done ] || abort "s$s: l2bnmw needs its primary reference l2bnm s$s finished first"; }
		git diff --quiet HEAD -- src/wnn/control || abort "src/wnn/control changed under the block (before $tag)"
		git diff --quiet HEAD -- $PREREG || abort "prereg edited under the block (before $tag)"
		build $s $arm
		while ctrl_busy; do sleep 30; done
		abi_ok || abort "ram_controller / facade is not ABI $ABI (before $tag)"
		log "START $tag [N87 $arm = $(what_of $arm)] (ABI $ABI, seed $((31337000 + s)), 35 units; reference $(ref_of $s))"
		eval "set -- $ARGV"
		PYTHONPATH=src/wnn nice -n 5 /usr/bin/time -l $VENV/bin/python -u -m wnn.control.phased_ga "$@" > $L/$tag.out 2>&1
		rc=$?
		log "END $tag rc=$rc $(row $L/$tag.out)"
		[ $rc -eq 0 ] && { touch $L/$tag.done; stress $tag ${arm}_s$s; }
	done
done
rm -f $LOCK
log "N87 DONE seeds [$SEEDS] arms [$ARMS]"
