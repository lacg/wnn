#!/bin/bash
# ABI-61 lineage, model-free control + one yaw arm (Luiz 22/11/2026). Waits for land_abi61_after_mf.sh to log the
# ABI-61 landing line in runs.log (stands down on its REFUSED / stand down lines), asserts the installed
# ram_controller (and the facade) is ABI 61, then flies seed-interleaved s1-s5:
#   A (CTRL-67) = N0 on ABI 61: the exact argv of the flown ABI-60 N0 row ..._a60_..._yld_n0_full_sN (rebuild_argv.py
#     fly); only the save paths (tag) differ.   tags ..._a61_..._L1_yld_n0_full_sN
#   B (CTRL-68) = A with --delta-max-yaw 0.5 -> 0.3 (N0 already carries Y = 0.5; one value changes).
#     tags ..._a61_..._L1_yld_n0dmy03_full_sN
# Every argv is checked statically AND through phased_ga's own validator (episode_config_from_args) at launch,
# before the wait, and again before each flight. Prereg: docs/ctrl_modelfree_prereg.md (ABI-61 addendum).
# START/END lines in runs.log; ends 'MF61 DONE' (abort 'MF61 ABORT', stand-down 'MF61 STAND DOWN').
# Dry run: MF61_DRY=1 prints each argv's diff vs the ABI-60 N0 row of the same seed and flies nothing.
cd /Users/lacg/wnn
L=logs/controller/ctrl48
RUNS=$L/runs.log
VENV=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv
ABI=61
SEEDS=${MF61_SEEDS:-1 2 3 4 5}
DRY=${MF61_DRY:-0}
PREREG=docs/ctrl_modelfree_prereg.md
log() { echo "$(date -u +%FT%TZ) $*" >> $RUNS; }
abort() { [ "$DRY" = 1 ] || log "MF61 ABORT: $1"; echo "MF61 ABORT: $1" >&2; exit 1; }
row()
{
	local g r
	g=$(grep -a 'stage-select\] HEADLINE stage' $1 | grep -oE 'genome=[^ ]+' | cut -d= -f2)
	r=$(grep -a "HEADLINE-$g MULTI-SEED" $1 | tail -1)
	[ -z "$r" ] && r=$(grep -aE "\[(test|report)-seeds\] ${g%%#*} MULTI-SEED" $1 | tail -1)
	echo "genome=$g $(echo "$r" | grep -oE 'stable=[^ ]+  err=[^ ]+  steady=[^ ]+|alt=[^ ]+|pos=[^ ]+|pos_hold=[^ ]+|dz_hold=[^ ]+|stable_fail=[^ ]+' | tr '\n' ' ')"
}
src_of() { echo C-S2-d_wposx2_a60_rf_c200_dc_biasin_ms01_P_both_L1_yld_n0_full_s$1; }
tag_of() { [ $2 = a ] && echo C-S2-d_wposx2_a${ABI}_rf_c200_dc_biasin_ms01_P_both_L1_yld_n0_full_s$1 \
	|| echo C-S2-d_wposx2_a${ABI}_rf_c200_dc_biasin_ms01_P_both_L1_yld_n0dmy03_full_s$1; }
count() { echo "$1" | grep -o -- "$2" | wc -l | tr -d ' '; }
# validate(argv): phased_ga's own flag resolution (pure; the same chain main() runs) -> refuses an incompatible stack
validate()
{
	PYTHONPATH=src/wnn $VENV/bin/python -c "
import shlex, sys
from wnn.control import phased_ga as p
a = p.build_arg_parser().parse_args(shlex.split(sys.argv[1]))
p.episode_config_from_args(a)
print(a.delta_max_yaw)" "$1" 2>&1 | tail -1
}
# build(s, arm) -> sets ARGV, or aborts with the reason
build()
{
	local s=$1 arm=$2 src tag
	src=$(src_of $s); tag=$(tag_of $s $arm)
	[ -f $L/$src.done ] || abort "missing finished ABI-60 N0 source run $src"
	ARGV=$($VENV/bin/python $L/rebuild_argv.py $L/$src.out fly $L/ckpt_$tag $L/${tag}_winner.yaml.gz) \
		|| abort "argv rebuild failed for $tag"
	echo "$ARGV" | grep -q -- "--obs-cascade-dhat" && abort "$tag: an MPCOF observer flag is in the N0 argv"
	echo "$ARGV" | grep -q -- "--yaw-trim-ff" && abort "$tag: --yaw-trim-ff (F) is in the N0 argv"
	echo "$ARGV" | grep -q -- "--obs-rate-i" && abort "$tag: --obs-rate-i is in the N0 argv"
	echo "$ARGV" | grep -q -- "--refine-fitness" && abort "$tag: refine fitness must be off"
	echo "$ARGV" | grep -q -- "--output-mode ctbr " || abort "$tag: N0 is not --output-mode ctbr"
	[ "$(count "$ARGV" "--ctbr-delta-law ")" = 1 ] || abort "$tag: N0 lacks the ctbr delta law"
	echo "$ARGV" | grep -q -- "--yaw-centred-ladder dense" || abort "$tag: dense ladder (L) missing"
	echo "$ARGV" | grep -q -- "--base-seed $((31337000 + s)) " || abort "$tag: wrong base seed"
	echo "$ARGV" | grep -q -- "--select-tail p99" || abort "$tag: rule P missing"
	[ "$(count "$ARGV" "--delta-max-yaw ")" = 1 ] || abort "$tag: N0 must carry exactly one --delta-max-yaw"
	[ "$(count "$ARGV" "--delta-max-yaw 0.5 ")" = 1 ] || abort "$tag: N0's --delta-max-yaw is not 0.5"
	if [ $arm = b ]; then
		ARGV=${ARGV/--delta-max-yaw 0.5 /--delta-max-yaw 0.3 }
		[ "$(count "$ARGV" "--delta-max-yaw 0.3 ")" = 1 ] || abort "$tag: --delta-max-yaw 0.3 not applied once"
		echo "$ARGV" | grep -q -- "--delta-max-yaw 0.5" && abort "$tag: --delta-max-yaw 0.5 survived"
		[ "$(validate "$ARGV")" = 0.3 ] || abort "$tag: phased_ga refuses the B stack: $(validate "$ARGV")"
	else
		[ "$(validate "$ARGV")" = 0.5 ] || abort "$tag: phased_ga refuses the N0 argv: $(validate "$ARGV")"
	fi
}

# --- preflight: prereg addendum + every argv built and validated BEFORE arming ---
grep -q "^## Addendum (22/11/2026): ABI-61 lineage" $PREREG || abort "prereg addendum (ABI-61 lineage) missing in $PREREG"
for s in $SEEDS; do for arm in a b; do
	build $s $arm
	if [ "$DRY" = 1 ]; then
		echo "DRY $(tag_of $s $arm)  (diff vs the flown ABI-60 N0 row $(src_of $s))"
		diff <(grep -a '^\[argv\] ' $L/$(src_of $s).out | cut -c8- | tr ' ' '\n') <(echo "$ARGV" | tr ' ' '\n')
	fi
done; done
[ "$DRY" = 1 ] && exit 0

# --- wait for the ABI-61 landing (only lines written after arm time count) ---
START_LINE=$(wc -l < $RUNS | tr -d ' ')
log "MF61 armed (pid $$): A = N0 on ABI 61 (CTRL-67), B = N0 + dmy 0.3 (CTRL-68), seeds $SEEDS; all 10 argv validated; waiting for the F32 landing line after line $START_LINE"
while :; do
	NEW=$(tail -n +"$((START_LINE + 1))" $RUNS)
	echo "$NEW" | grep -qE "Z F32: ABI 61 LANDED" && break
	hit=$(echo "$NEW" | grep -E "Z F32: (REFUSED|stand down)" | head -1)
	if [ -n "$hit" ]; then log "MF61 STAND DOWN: the landing did not land, nothing flown ($hit)"; exit 1; fi
	sleep 60
done
# the landing script, the model-free block and any controller child must all be gone
while pgrep -f "land_abi61_after_m[f]" >/dev/null || pgrep -f "ab_n0_nr.s[h]" >/dev/null \
	|| pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null; do sleep 30; done
PYTHONPATH=src/wnn $VENV/bin/python -c "import ram_controller as c, wnn.control._accel as f; assert c.ABI_VERSION == $ABI and f.EXPECTED_ABI == $ABI, (c.ABI_VERSION, f.EXPECTED_ABI)" \
	|| abort "ram_controller / facade is not ABI $ABI after the landing"
log "MF61 START A = N0 on ABI 61 (lineage control), B = N0 + --delta-max-yaw 0.3, seed-interleaved, seeds $SEEDS, ABI $ABI (prereg $PREREG, ABI-61 addendum)"
for s in $SEEDS; do
	for arm in a b; do
		tag=$(tag_of $s $arm)
		[ -f $L/$tag.done ] && continue
		build $s $arm
		while pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null; do sleep 30; done
		PYTHONPATH=src/wnn $VENV/bin/python -c "import ram_controller as c, wnn.control._accel; assert c.ABI_VERSION == $ABI" \
			|| abort "ram_controller is not ABI $ABI (before $tag)"
		[ $arm = a ] && what="A n0 = N0 on ABI 61, P, floor" || what="B n0dmy03 = N0 + --delta-max-yaw 0.3, P, floor"
		log "START $tag [$what] (ABI $ABI, seed $((31337000 + s)), 35 units)"
		eval "set -- $ARGV"
		PYTHONPATH=src/wnn nice -n 5 /usr/bin/time -l $VENV/bin/python -u -m wnn.control.phased_ga "$@" > $L/$tag.out 2>&1
		rc=$?
		log "END $tag rc=$rc $(row $L/$tag.out)"
		[ $rc -eq 0 ] && touch $L/$tag.done
	done
done
log "MF61 DONE"
