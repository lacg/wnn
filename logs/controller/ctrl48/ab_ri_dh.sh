#!/bin/bash
# CTRL-38 re-test (Luiz 22/11/2026): rate-I (R = Y+F+L + --obs-rate-i) and d-hat (H = Y+F+L + --obs-cascade-dhat),
# one setting each, s1-s5 seed-major interleaved (R sN, H sN). Both were judged on the broken encoding on 08/11.
# No code change; installed ABI 60. Base rows = Y+F+L ..._a59_..._ffF_yld_full_sN (not re-flown).
# Prereg docs/ctrl38_ri_dh_retest_prereg.md. END lines in runs.log; ends RIDH DONE (abort 'RIDH ABORT').
cd /Users/lacg/wnn
L=logs/controller/ctrl48
VENV=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv
ABI=${ABI:-60}
SEEDS=${RIDH_SEEDS:-1 2 3 4 5}
DRY=${RIDH_DRY:-0}
abort() { echo "$(date -u +%FT%TZ) RIDH ABORT: $1" >> $L/runs.log; echo "RIDH ABORT: $1" >&2; exit 1; }
row()
{
	local g r
	g=$(grep -a 'stage-select\] HEADLINE stage' $1 | grep -oE 'genome=[^ ]+' | cut -d= -f2)
	r=$(grep -a "HEADLINE-$g MULTI-SEED" $1 | tail -1)
	[ -z "$r" ] && r=$(grep -aE "\[(test|report)-seeds\] ${g%%#*} MULTI-SEED" $1 | tail -1)
	echo "genome=$g $(echo "$r" | grep -oE 'stable=[^ ]+  err=[^ ]+  steady=[^ ]+|alt=[^ ]+|pos=[^ ]+|pos_hold=[^ ]+|dz_hold=[^ ]+|stable_fail=[^ ]+' | tr '\n' ' ')"
}
[ -f docs/ctrl38_ri_dh_retest_prereg.md ] || abort "prereg docs/ctrl38_ri_dh_retest_prereg.md missing"
[ "$DRY" = 1 ] || echo "$(date -u +%FT%TZ) RIDH START R (--obs-rate-i) and H (--obs-cascade-dhat) on base Y+F+L, seeds $SEEDS, ABI $ABI (prereg docs/ctrl38_ri_dh_retest_prereg.md)" >> $L/runs.log
for s in $SEEDS; do
	src=C-S2-d_wposx2_a59_rf_c200_dc_biasin_ms01_P_both_L1_ffF_yld_full_s$s
	[ -f $L/$src.done ] || abort "missing finished Y+F+L source run $src"
	for arm in ri dh; do
		[ $arm = ri ] && flag="--obs-rate-i" || flag="--obs-cascade-dhat"
		tag=C-S2-d_wposx2_a${ABI}_rf_c200_dc_biasin_ms01_P_both_L1_ffF_yld_${arm}_full_s$s
		[ -f $L/$tag.done ] && continue
		argv=$($VENV/bin/python $L/rebuild_argv.py $L/$src.out fly $L/ckpt_$tag $L/${tag}_winner.yaml.gz $flag) \
			|| abort "argv rebuild failed for $tag"
		[ "$(echo "$argv" | grep -o -- "$flag " | wc -l | tr -d ' ')" = 1 ] || abort "$tag: $flag not applied exactly once"
		for other in --obs-rate-i --obs-cascade-dhat; do
			[ "$other" = "$flag" ] && continue
			echo "$argv" | grep -q -- "$other " && abort "$tag: unexpected $other"
		done
		echo "$argv" | grep -q -- "--obs-cascade-dhat-residual" || abort "$tag: base residual input missing"
		echo "$argv" | grep -q -- "--yaw-centred-ladder dense" || abort "$tag: dense ladder missing"
		echo "$argv" | grep -q -- "--yaw-trim-ff" || abort "$tag: --yaw-trim-ff missing"
		echo "$argv" | grep -q -- "--refine-fitness" && abort "$tag: refine fitness must be off"
		echo "$argv" | grep -q -- "--base-seed $((31337000 + s)) " || abort "$tag: wrong base seed"
		echo "$argv" | grep -q -- "--select-tail p99" || abort "$tag: rule P missing"
		if [ "$DRY" = 1 ]; then
			echo "DRY $tag"; diff <(grep -a '^\[argv\] ' $L/$src.out | cut -c8- | tr ' ' '\n') <(echo "$argv" | tr ' ' '\n'); continue
		fi
		while pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null; do sleep 30; done
		PYTHONPATH=src/wnn $VENV/bin/python -c "import ram_controller as c, wnn.control._accel; assert c.ABI_VERSION == $ABI" \
			|| abort "ram_controller is not ABI $ABI (before $tag)"
		echo "$(date -u +%FT%TZ) START $tag [$arm = Y+F+L + $flag, P, floor] (ABI $ABI, seed $((31337000 + s)), 35 units)" >> $L/runs.log
		eval "set -- $argv"
		PYTHONPATH=src/wnn nice -n 5 /usr/bin/time -l $VENV/bin/python -u -m wnn.control.phased_ga "$@" > $L/$tag.out 2>&1
		rc=$?
		echo "$(date -u +%FT%TZ) END $tag rc=$rc $(row $L/$tag.out)" >> $L/runs.log
		[ $rc -eq 0 ] && touch $L/$tag.done
	done
done
[ "$DRY" = 1 ] || echo "$(date -u +%FT%TZ) RIDH DONE" >> $L/runs.log
