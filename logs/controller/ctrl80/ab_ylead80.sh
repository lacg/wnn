#!/bin/bash
# CTRL-80 (Luiz 26/11/2026): yaw LEAD input (ylead = --obs-yaw-lead --obs-yaw-lead-tau 0.10) on the G+T base, prereg
# docs/ctrl80_yaw_lead_prereg.md. Derived from the CTRL-79 driver (ctrl63/ab_yil79.sh, same checks). The arm carries ONLY
# the two lead flags. Needs branch ctrl80-yaw-lead MERGED into /Users/lacg/wnn AND the ABI-64 ram_controller wheel
# INSTALLED first: it refuses to run otherwise (pin + prereg must be committed in the tree, wheel + facade must be ABI 64).
# Control of every pair = the FLOWN G+T sN run (..._a62_..._L1_yld_n0g3gz_full_sN; ABI 62; never re-flown). Each arm argv
# = that run's [argv] line + ONLY the arm's flags + the two save paths (tag ..._a64_..._L1_yld_gt_ylead_full_sN).
# Fixed-text check: arm argv minus its flags, tag -> control tag == the control's DEDUPED [argv] byte for byte (27/11:
# rebuild_argv.py keeps each repeated flag's last occurrence; raw vs deduped control must resolve identically). Resolver check
# (phased_ga's own parser + _resolve_ctbr + resolve_block10_args + calibration token): the control resolves to G+T with
# the lead OFF; the arm moves ONLY obs_yaw_lead (True, tau 0.1); the calibration token and the digest of everything else
# are equal (the lead has its own fixed ladder and steals no cut: tests/controller_ctrl80_yaw_lead.py).
# Stage-aware: Y80_SEEDS (default "1 2 3") and Y80_ARMS (default ylead) — later stages pass the survivors only.
# Order: seed-major. One run at a time; waits for any other controller run. Wheel ram_controller + facade ABI 64 asserted
# before every flight; src/wnn/control must be clean; the prereg must be committed. A tag whose .out exists is never
# reused (no recipe repeated on a seed). START/END in runs.log; block lines 'Y80 ...'; ends 'Y80 DONE' (abort 'Y80 ABORT: why').
# Dry run: Y80_DRY=1 prints, per arm, the token diff vs its G+T control argv + both resolver lines; flies nothing, writes
# nothing (no runs.log line, no lock). ONLY with Y80_DRY=1: Y80_ROOT=<worktree> validates against the unmerged branch and
# Y80_SITE=<dir> puts an UNPACKED (never installed) ABI-64 wheel first on PYTHONPATH for the resolver.
# After each END (CPU, nice 15): logs/controller/ctrl80/rescore_end.sh <arm_tag> <control_tag> (pre-hold / whole / hold yaw + panel).
cd /Users/lacg/wnn || exit 1
ROOT=${Y80_ROOT:-/Users/lacg/wnn}
SITE=${Y80_SITE:-}
L=logs/controller/ctrl48
RUNS=$L/runs.log
VENV=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv
ABI=64
ALL_ARMS="ylead l2bylead l2byleadngz"
SEEDS=${Y80_SEEDS:-1 2 3}
ARMS=${Y80_ARMS:-l2byleadngz}   # Luiz 27/11 (amendment 2): the lead replaces gyro_z in the yaw bank
DRY=${Y80_DRY:-0}
PREREG=docs/ctrl80_yaw_lead_prereg.md
PREREG_HDR="^## Adoption rule"
PINS="tests/controller_ctrl80_yaw_lead.py"
UNITS="--test-seeds $(seq -f %.0f 99990101 99990135 | tr "\n" " ")"
GT_GAMMA="(1.0, 2.0, 2.0, 3.0)"
GT_CALIB="eps:200,regime:flown,ladder:quantile,yaw:dense,tilt:dense,gyroz:dense"
DMAX0=0.5
TAU=0.10
PFX=C-S2-d_wposx2
MID=rf_c200_dc_biasin_ms01_P_both_L1_yld
PP=$ROOT/src
[ -n "$SITE" ] && PP=$SITE:$ROOT/src
log() { echo "$(date -u +%FT%TZ) $*" >> $RUNS; }
abort() { [ "$DRY" = 1 ] || log "Y80 ABORT: $1"; echo "Y80 ABORT: $1" >&2; exit 1; }
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
	[ "$1" = ylead ] && f="--obs-yaw-lead --obs-yaw-lead-tau $TAU"
	[ "$1" = l2bylead ] && f="--gyro-xy-ladder dense_hold --obs-yaw-lead --obs-yaw-lead-tau $TAU"
	[ "$1" = l2byleadngz ] && f="--gyro-xy-ladder dense_hold --obs-yaw-lead --obs-yaw-lead-tau $TAU --drop-features gyro_z"
	echo "$f"
}
calib_of() { [ "$1" = l2bylead ] && { echo "$GT_CALIB,gyroxy:dense_hold"; return; }; [ "$1" = l2byleadngz ] && { echo "$GT_CALIB,gyroxy:dense_hold"; return; }; echo "$GT_CALIB"; }
l2b_of() { echo ${PFX}_a63_${MID}_gt_l2b_full_s$1; }
what_of() { [ "$1" = l2byleadngz ] && printf "L2-b dense_hold (base) + gyro_z DROPPED from the yaw bank (the lead replaces it: integral gets its bits back) + "; [ "$1" = l2bylead ] && printf "L2-b gyro_x/y dense_hold (the adopted base) + "; echo "yaw lead input yaw_err - ${TAU}s*gyro_z (own fixed ladder +-0.40/0.80/1.35/2.25 deg): can the student see when to brake the yaw transient?"; }
tag_of() { echo ${PFX}_a${ABI}_${MID}_gt_$2_full_s$1; }
ctl_of() { echo ${PFX}_a62_${MID}_n0g3gz_full_s$1; }
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
a = p.build_arg_parser().parse_args(shlex.split(sys.argv[1]))
ec = p.episode_config_from_args(a); p._resolve_ctbr(a, ec); p.resolve_block10_args(a); c = a._ctbr
skip = ('obs_yaw_lead', 'gyro_xy_ladder', 'drop_features', 'obs_yaw_lead_tau', 'save_stage_checkpoints', 'save_winner')
rest = repr(sorted((k, repr(v)) for k, v in c.items())) + repr(sorted((k, repr(v)) for k, v in vars(a).items() if not k.startswith('_') and k not in skip))
lk = a.delta_leak_yaw if a.delta_leak_yaw is not None else a.delta_leak
gain = round(a.delta_max_yaw / (1.0 - lk), 3)
print('OK', a.base_seed, a.delta_max_yaw, a.delta_leak_yaw, tuple(float(g) for g in c['gamma']), 'gain', gain, 'lead', a.obs_yaw_lead, a.obs_yaw_lead_tau, '|', p.calibration_from_args(a).token(), '|', a.num_eval_folds, len(a.test_seeds), a.test_seeds[0], a.test_seeds[-1], '|', hashlib.sha256(rest.encode()).hexdigest()[:16])" "$1" 2>&1 | tail -1)
}
# build(s, arm) -> sets ARGV, CLINE, VAL_CTL, VAL_ARM, or aborts
build()
{
	local s=$1 arm=$2 tag ctl flags back v vc want
	tag=$(tag_of $s $arm); ctl=$(ctl_of $s); flags=$(flags_of $arm)
	[ -n "$flags" ] || abort "$arm: no flags (unknown arm)"
	[ -f $L/$ctl.done ] || abort "s$s: the G+T control $ctl has not finished — never fly an arm without its control"
	case $arm in l2bylead|l2byleadngz) true;; *) false;; esac && { [ -f $L/$(l2b_of $s).done ] || abort "s$s: the l2b base run $(l2b_of $s) (the pairing reference) has not finished"; }
	RAWLINE=$(grep -a '^\[argv\] ' $L/$ctl.out | cut -c8-)
	[ "$(echo "$RAWLINE" | grep -c .)" = 1 ] || abort "$ctl: expected exactly one [argv] line"
	# 27/11/2026 (Luiz "leave the last one", docs/recipe_duplicate_flags_2711.md): the flown [argv] repeats flags
	# (--max-cells 180000 ... 1000000000, --skip-stages neurons,bits ... connections,bits, ...); argparse keeps the last.
	# rebuild_argv.py now DEDUPES keeping the last occurrence, so the arm is compared against the DEDUPED control argv,
	# and the raw and deduped control must resolve to the same args (full resolver line, digest included).
	CLINE=$($VENV/bin/python $L/rebuild_argv.py $L/$ctl.out argv) || abort "$ctl: argv dedupe failed"
	[ "$(validate "$RAWLINE")" = "$(validate "$CLINE")" ] || abort "$ctl: the deduped control argv does not resolve like the flown one"
	[ "$(count "$CLINE" "--base-seed $((31337000 + s)) ")" = 1 ] || abort "$ctl: does not carry --base-seed $((31337000 + s)) once"
	[ "$(count "$CLINE" "--delta-max-yaw $DMAX0 ")" = 1 ] || abort "$ctl: does not carry --delta-max-yaw $DMAX0 exactly once"
	echo "$CLINE" | grep -qE -- "--delta-leak-yaw|--gyro-xy-ladder|--yaw-err-outer-cut|--yaw-integral-ladder|--obs-yaw-lead" && abort "$ctl: the control already carries a lever flag"
	ARGV=$($VENV/bin/python $L/rebuild_argv.py $L/$ctl.out fly $L/ckpt_$tag $L/${tag}_winner.yaml.gz $flags) \
		|| abort "argv rebuild failed for $tag"
	# fixed text: flags removed, tag -> control tag == the flown G+T sN [argv]
	[ "$(count "$ARGV" " $flags ")" = 1 ] || abort "$tag: the arm flags '$flags' not in the argv exactly once"
	back=${ARGV/ $flags / }
	back=${back//$tag/$ctl}
	[ "$back" = "$CLINE" ] || abort "$tag: argv minus flags+tag is not the G+T s$s control ($ctl) argv (diff: $(diff <(echo "$back" | tr ' ' '\n') <(echo "$CLINE" | tr ' ' '\n') | tr '\n' ' '))"
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
	[ "$(count "$ARGV" "--obs-yaw-err ")" = 1 ] || abort "$tag: --obs-yaw-err (the input the lead is built from) not exactly once"
	echo "$ARGV" | grep -q -- "--select-tail p99" || abort "$tag: rule P missing"
	echo "$ARGV" | grep -q -- "--num-eval-folds 5 " || abort "$tag: folds must be 5"
	[ "$(count "$ARGV" "--delta-max-yaw $DMAX0 ")" = 1 ] || abort "$tag: --delta-max-yaw is not $DMAX0 exactly once"
	[ "$(count "$ARGV" "--obs-yaw-lead-tau $TAU ")" = 1 ] || abort "$tag: --obs-yaw-lead-tau $TAU not exactly once"
	vc=$(validate "$CLINE"); v=$(validate "$ARGV")
	VAL_CTL=$vc; VAL_ARM=$v
	[ "${vc% | *}" = "OK $((31337000 + s)) $DMAX0 None $GT_GAMMA gain 1.667 lead False 0.1 | $GT_CALIB | 5 35 99990101 99990135" ] \
		|| abort "$ctl: the G+T s$s control does not resolve to G+T: $vc"
	want="OK $((31337000 + s)) $DMAX0 None $GT_GAMMA gain 1.667 lead True 0.1 | $(calib_of $arm) | 5 35 99990101 99990135 | ${vc##* | }"
	[ "$v" = "$want" ] || abort "$tag: phased_ga refuses / mis-resolves the arm (want '$want'): $v"
}

# --- preflight ---
[ "$ROOT" = /Users/lacg/wnn ] || [ "$DRY" = 1 ] || abort "Y80_ROOT may only be overridden with Y80_DRY=1"
[ -z "$SITE" ] || [ "$DRY" = 1 ] || abort "Y80_SITE may only be set with Y80_DRY=1 (a flight uses the INSTALLED wheel)"
for a in $ARMS; do case " $ALL_ARMS " in *" $a "*) ;; *) abort "unknown arm '$a' in Y80_ARMS";; esac; done
git -C $ROOT ls-files --error-unmatch $PREREG >/dev/null 2>&1 || abort "prereg $PREREG is not committed in $ROOT (merge ctrl80-yaw-lead first)"
git -C $ROOT diff --quiet HEAD -- $PREREG || abort "prereg $PREREG has uncommitted edits"
grep -q "$PREREG_HDR" $ROOT/$PREREG || abort "prereg adoption-rule section missing in $PREREG"
for p in $PINS; do git -C $ROOT ls-files --error-unmatch $p >/dev/null 2>&1 || abort "the flag-off pin $p is not committed in $ROOT (merge ctrl80-yaw-lead first)"; done
git -C $ROOT diff --quiet HEAD -- src/wnn/control || abort "src/wnn/control has uncommitted edits"
abi_ok || abort "ram_controller / facade is not ABI $ABI (merge ctrl80-yaw-lead, then install the controller wheel)"
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

LOCK=/tmp/ab_ylead80.pid
[ -f $LOCK ] && kill -0 "$(cat $LOCK)" 2>/dev/null && abort "another Y80 driver is already running (pid $(cat $LOCK))"
echo $$ > $LOCK
log "Y80 armed (pid $$, ab_ylead80.sh): CTRL-80 yaw lead input, seeds [$SEEDS] x arms [$ARMS] = $N runs, seed-major, ABI $ABI; every argv validated vs its flown G+T control (prereg $PREREG)"
while ctrl_busy; do sleep 30; done
log "Y80 START seeds [$SEEDS] arms [$ARMS]"
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
		log "START $tag [Y80 $arm = G+T + $(what_of $arm), P, floor] (ABI $ABI, seed $((31337000 + s)), 35 units; control $(ctl_of $s))"
		eval "set -- $ARGV"
		PYTHONPATH=src/wnn nice -n 5 /usr/bin/time -l $VENV/bin/python -u -m wnn.control.phased_ga "$@" > $L/$tag.out 2>&1
		rc=$?
		log "END $tag rc=$rc $(row $L/$tag.out)"
		[ $rc -eq 0 ] && touch $L/$tag.done
	done
done
rm -f $LOCK
log "Y80 DONE seeds [$SEEDS] arms [$ARMS]"
