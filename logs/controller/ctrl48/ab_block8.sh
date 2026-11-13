#!/bin/bash
# CTRL-48 block 8 (Luiz 12/11 + 13/11): base dc_biasin_ms01 re-flown as the control + 4 yaw arms, s1-s3 SEED-MAJOR = 15 runs,
# controller ABI 50, ALL runs with rule P (--val-units 20 --select-tail p99; adopted 13/11, docs/ctrl_select_tail_result.md).
#   ctl    = base (dc + --obs-att-bias-pred --write-rule vote --vote-share 0.1) + P
#   ileak  = ctl + --integral-leak 0.999                         (does a leaky rate integral cut the yaw offset?)
#   ycl    = ctl + --yaw-centred-ladder                          (does a yaw ladder centred on the yaw setpoint cut the offset?)
#   yileak = ctl + --yaw-integral-leak 0.9995                    (yaw-only leak)
#   both   = ctl + --yaw-centred-ladder --yaw-integral-leak 0.9995 (pre-check: the yaw leak pays mainly with the ladder)
# Arms are paired vs the SAME seed's ctl from THIS block (same ABI, same P selection).
# Recipe = each seed's block-7 dc_biasin_ms01 [argv] (rebuild_argv.py fly) + P + the arm flags.
# Code: branch unify-write-rules-core via the worktree on PYTHONPATH until the full deploy merges it (Luiz 12/11 23:20 EST:
# controller-only install, swap-free). Hardening (Luiz 12/11): ABI 50 + facade import checked before EVERY run; mismatch or
# import error -> 'BLOCK8 ABORT'. END lines in runs.log; ends 'BLOCK8 DONE'.
cd /Users/lacg/wnn
L=logs/controller/ctrl48
VENV=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv
CODE=/Users/lacg/wnn/.claude/worktrees/agent-a162d4e33a11c36bb/src
P="--val-units 20 --select-tail p99"
abort() { echo "$(date -u +%FT%TZ) BLOCK8 ABORT: $1" >> $L/runs.log; exit 1; }
check_abi()
{
	PYTHONPATH=$CODE $VENV/bin/python -c "import ram_controller as c, wnn.control._accel as a; assert c.ABI_VERSION == 50, c.ABI_VERSION; assert a.__file__.startswith('$CODE'), a.__file__" \
		|| abort "ram_controller is not ABI 50 or the branch facade failed to import (before $1)"
}
row() # headline 35-unit row: HEADLINE-<g> MULTI-SEED, else the #0 winner's stage row
{
	local g r
	g=$(grep -a 'stage-select\] HEADLINE stage' $1 | grep -oE 'genome=[^ ]+' | cut -d= -f2)
	r=$(grep -a "HEADLINE-$g MULTI-SEED" $1 | tail -1)
	[ -z "$r" ] && r=$(grep -aE "\[(test|report)-seeds\] ${g%%#*} MULTI-SEED" $1 | tail -1)
	echo "genome=$g $(echo "$r" | grep -oE 'stable=[^ ]+  err=[^ ]+  steady=[^ ]+|alt=[^ ]+|pos=[^ ]+|stable_fail=[^ ]+' | tr '\n' ' ')"
}
arm_flags()
{
	case $1 in
		ctl)    echo "" ;;
		ileak)  echo "--integral-leak 0.999" ;;
		ycl)    echo "--yaw-centred-ladder" ;;
		yileak) echo "--yaw-integral-leak 0.9995" ;;
		both)   echo "--yaw-centred-ladder --yaw-integral-leak 0.9995" ;;
	esac
}
echo "$(date -u +%FT%TZ) BLOCK8 START 5 arms x s1-s3 seed-major = 15 runs, ABI 50, rule P, code $CODE" >> $L/runs.log
for s in 1 2 3; do
	for arm in ctl ileak ycl yileak both; do
		tag=C-S2-d_wposx2_a50_rf_c200_dc_biasin_ms01_P_${arm}_full_s$s
		[ -f $L/$tag.done ] && continue
		EXTRA="$P $(arm_flags $arm)"
		argv=$($VENV/bin/python $L/rebuild_argv.py $L/C-S2-d_wposx2_a49_rf_c200_dc_biasin_ms01_full_s$s.out fly \
			$L/ckpt_$tag $L/${tag}_winner.yaml.gz $EXTRA) || abort "argv rebuild failed for $tag"
		echo "$argv" | grep -q -- "--base-seed 3133700$s " || abort "$tag: base argv is not seed 3133700$s"
		while pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null; do sleep 30; done
		check_abi $tag
		echo "$(date -u +%FT%TZ) START $tag [$EXTRA] on the dc_biasin_ms01 recipe (posx2, ABI 50, seed 3133700$s, 35 units)" >> $L/runs.log
		eval "set -- $argv"
		PYTHONPATH=$CODE nice -n 5 /usr/bin/time -l $VENV/bin/python -u -m wnn.control.phased_ga "$@" > $L/$tag.out 2>&1
		rc=$?
		echo "$(date -u +%FT%TZ) END $tag rc=$rc $(row $L/$tag.out)" >> $L/runs.log
		[ $rc -eq 0 ] && touch $L/$tag.done
	done
done
echo "$(date -u +%FT%TZ) BLOCK8 DONE" >> $L/runs.log
