#!/bin/bash
# CTRL-48 block 9 (Luiz 13/11): HEAD-TO-HEAD dc_ms01 vs both on NEW seeds s4-s6 (SEED-MAJOR = 6 runs), controller ABI 50,
# BOTH arms with rule P. Prereg docs/ctrl48_block9_prereg.md.
#   dcms01 = block-6 dc_ms01 recipe (its s1 [argv]) + P                          (0/10500 fails on s1-s3, steady 0.94)
#   both   = block-8 both recipe (its s1 [argv], already carries P)             (2/10500 fails on s1-s3, steady 0.77)
# Seed s = the s1 argv with --base-seed 31337001 -> 3133700s (the ONLY seed-dependent token; verified: s1 vs s2 argv differ
# only in --base-seed and the save paths). Runs AFTER 'S5P DONE'. ABI 50 checked before EVERY run ('BLOCK9 ABORT').
# END lines in runs.log; ends 'BLOCK9 DONE'.
cd /Users/lacg/wnn
L=logs/controller/ctrl48
VENV=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv
abort() { echo "$(date -u +%FT%TZ) BLOCK9 ABORT: $1" >> $L/runs.log; exit 1; }
row() # headline 35-unit row: HEADLINE-<g> MULTI-SEED, else the #0 winner's stage row
{
	local g r
	g=$(grep -a 'stage-select\] HEADLINE stage' $1 | grep -oE 'genome=[^ ]+' | cut -d= -f2)
	r=$(grep -a "HEADLINE-$g MULTI-SEED" $1 | tail -1)
	[ -z "$r" ] && r=$(grep -aE "\[(test|report)-seeds\] ${g%%#*} MULTI-SEED" $1 | tail -1)
	echo "genome=$g $(echo "$r" | grep -oE 'stable=[^ ]+  err=[^ ]+  steady=[^ ]+|alt=[^ ]+|pos=[^ ]+|stable_fail=[^ ]+' | tr '\n' ' ')"
}
src()  { case $1 in dcms01) echo C-S2-d_wposx2_a49_rf_c200_dc_ms01_full_s1 ;; both) echo C-S2-d_wposx2_a50_rf_c200_dc_biasin_ms01_P_both_full_s1 ;; esac; }
tagf() { case $1 in dcms01) echo C-S2-d_wposx2_a50_rf_c200_dc_ms01_P_full_s$2 ;; both) echo C-S2-d_wposx2_a50_rf_c200_dc_biasin_ms01_P_both_full_s$2 ;; esac; }
xtra() { case $1 in dcms01) echo "--val-units 20 --select-tail p99" ;; both) echo "" ;; esac; }
until grep -qE "S5P (DONE|ABORT)" $L/runs.log; do sleep 60; done
echo "$(date -u +%FT%TZ) BLOCK9 START dc_ms01 vs both, s4-s6 seed-major = 6 runs, ABI 50, rule P (prereg docs/ctrl48_block9_prereg.md)" >> $L/runs.log
for s in 4 5 6; do
	for arm in dcms01 both; do
		tag=$(tagf $arm $s)
		[ -f $L/$tag.done ] && continue
		argv=$($VENV/bin/python $L/rebuild_argv.py $L/$(src $arm).out fly $L/ckpt_$tag $L/${tag}_winner.yaml.gz $(xtra $arm)) \
			|| abort "argv rebuild failed for $tag"
		echo "$argv" | grep -q -- "--base-seed 31337001 " || abort "$tag: source argv is not seed 31337001"
		argv=${argv/--base-seed 31337001 /--base-seed 3133700$s }
		echo "$argv" | grep -q -- "--base-seed 3133700$s " || abort "$tag: seed substitution failed"
		echo "$argv" | grep -q -- "--select-tail p99" || abort "$tag: rule P missing"
		while pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null; do sleep 30; done
		PYTHONPATH=src/wnn $VENV/bin/python -c "import ram_controller as c, wnn.control._accel; assert c.ABI_VERSION == 50" \
			|| abort "ram_controller is not ABI 50 or the facade failed to import (before $tag)"
		echo "$(date -u +%FT%TZ) START $tag [$arm, P] (ABI 50, seed 3133700$s, 35 units)" >> $L/runs.log
		eval "set -- $argv"
		PYTHONPATH=src/wnn nice -n 5 /usr/bin/time -l $VENV/bin/python -u -m wnn.control.phased_ga "$@" > $L/$tag.out 2>&1
		rc=$?
		echo "$(date -u +%FT%TZ) END $tag rc=$rc $(row $L/$tag.out)" >> $L/runs.log
		[ $rc -eq 0 ] && touch $L/$tag.done
	done
done
echo "$(date -u +%FT%TZ) BLOCK9 DONE" >> $L/runs.log
