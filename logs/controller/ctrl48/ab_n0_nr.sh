#!/bin/bash
# MODEL-FREE block (Luiz 22/11/2026: "no MPCOF running as part of the WNN"): remove MPCOF's disturbance observer
# from the student. N0 = Y+F+L minus --obs-cascade-dhat-residual minus --yaw-trim-ff (F needs the observer residual);
# NR = N0 + --obs-rate-i (the firmware rate loop's integrator: model-free disturbance memory). s1-s5 seed-major
# interleaved (N0 sN, NR sN). No code change; ABI 60. Paired vs Y+F+L ..._a59_..._ffF_yld_full_sN (not re-flown).
# Prereg docs/ctrl_modelfree_prereg.md. END lines in runs.log; ends MF DONE (abort 'MF ABORT').
cd /Users/lacg/wnn
L=logs/controller/ctrl48
VENV=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv
ABI=${ABI:-60}
SEEDS=${MF_SEEDS:-1 2 3 4 5}
DRY=${MF_DRY:-0}
abort() { echo "$(date -u +%FT%TZ) MF ABORT: $1" >> $L/runs.log; echo "MF ABORT: $1" >&2; exit 1; }
row()
{
	local g r
	g=$(grep -a 'stage-select\] HEADLINE stage' $1 | grep -oE 'genome=[^ ]+' | cut -d= -f2)
	r=$(grep -a "HEADLINE-$g MULTI-SEED" $1 | tail -1)
	[ -z "$r" ] && r=$(grep -aE "\[(test|report)-seeds\] ${g%%#*} MULTI-SEED" $1 | tail -1)
	echo "genome=$g $(echo "$r" | grep -oE 'stable=[^ ]+  err=[^ ]+  steady=[^ ]+|alt=[^ ]+|pos=[^ ]+|pos_hold=[^ ]+|dz_hold=[^ ]+|stable_fail=[^ ]+' | tr '\n' ' ')"
}
[ -f docs/ctrl_modelfree_prereg.md ] || abort "prereg docs/ctrl_modelfree_prereg.md missing"
[ "$DRY" = 1 ] || echo "$(date -u +%FT%TZ) MF START N0 (no observer, no F) and NR (N0 + --obs-rate-i) vs base Y+F+L, seeds $SEEDS, ABI $ABI (prereg docs/ctrl_modelfree_prereg.md)" >> $L/runs.log
for s in $SEEDS; do
	src=C-S2-d_wposx2_a59_rf_c200_dc_biasin_ms01_P_both_L1_ffF_yld_full_s$s
	[ -f $L/$src.done ] || abort "missing finished Y+F+L source run $src"
	for arm in n0 nr; do
		[ $arm = nr ] && extra="--obs-rate-i" || extra=""
		tag=C-S2-d_wposx2_a${ABI}_rf_c200_dc_biasin_ms01_P_both_L1_yld_${arm}_full_s$s
		[ -f $L/$tag.done ] && continue
		argv=$($VENV/bin/python $L/rebuild_argv.py $L/$src.out fly $L/ckpt_$tag $L/${tag}_winner.yaml.gz $extra) \
			|| abort "argv rebuild failed for $tag"
		[ "$(echo "$argv" | grep -o -- "--obs-cascade-dhat-residual " | wc -l | tr -d ' ')" = 1 ] || abort "$tag: base lacks exactly one residual flag"
		[ "$(echo "$argv" | grep -o -- "--yaw-trim-ff " | wc -l | tr -d ' ')" = 1 ] || abort "$tag: base lacks exactly one --yaw-trim-ff"
		argv=${argv/--obs-cascade-dhat-residual /}
		argv=${argv/--yaw-trim-ff /}
		echo "$argv" | grep -q -- "--obs-cascade-dhat" && abort "$tag: an MPCOF observer flag survived"
		echo "$argv" | grep -q -- "--yaw-trim-ff" && abort "$tag: --yaw-trim-ff survived"
		if [ $arm = nr ]; then
			[ "$(echo "$argv" | grep -o -- "--obs-rate-i " | wc -l | tr -d ' ')" = 1 ] || abort "$tag: --obs-rate-i not applied once"
		else
			echo "$argv" | grep -q -- "--obs-rate-i" && abort "$tag: unexpected --obs-rate-i"
		fi
		echo "$argv" | grep -q -- "--yaw-centred-ladder dense" || abort "$tag: dense ladder missing"
		echo "$argv" | grep -q -- "--delta-max-yaw 0.5" || abort "$tag: Y (delta-max-yaw 0.5) missing"
		echo "$argv" | grep -q -- "--refine-fitness" && abort "$tag: refine fitness must be off"
		echo "$argv" | grep -q -- "--base-seed $((31337000 + s)) " || abort "$tag: wrong base seed"
		echo "$argv" | grep -q -- "--select-tail p99" || abort "$tag: rule P missing"
		if [ "$DRY" = 1 ]; then
			echo "DRY $tag"; diff <(grep -a '^\[argv\] ' $L/$src.out | cut -c8- | tr ' ' '\n') <(echo "$argv" | tr ' ' '\n'); continue
		fi
		while pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null; do sleep 30; done
		PYTHONPATH=src/wnn $VENV/bin/python -c "import ram_controller as c, wnn.control._accel; assert c.ABI_VERSION == $ABI" \
			|| abort "ram_controller is not ABI $ABI (before $tag)"
		echo "$(date -u +%FT%TZ) START $tag [$arm = Y+F+L - observer - F ${extra:+ + $extra}, P, floor] (ABI $ABI, seed $((31337000 + s)), 35 units)" >> $L/runs.log
		eval "set -- $argv"
		PYTHONPATH=src/wnn nice -n 5 /usr/bin/time -l $VENV/bin/python -u -m wnn.control.phased_ga "$@" > $L/$tag.out 2>&1
		rc=$?
		echo "$(date -u +%FT%TZ) END $tag rc=$rc $(row $L/$tag.out)" >> $L/runs.log
		[ $rc -eq 0 ] && touch $L/$tag.done
	done
done
[ "$DRY" = 1 ] || echo "$(date -u +%FT%TZ) MF DONE" >> $L/runs.log
