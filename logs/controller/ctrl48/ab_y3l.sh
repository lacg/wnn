#!/bin/bash
# Y3+L (Luiz 20/11, pre-approved conditional on the per-axis split): Y3 (--delta-max-yaw 0.3, no F) with the base quantile ladder REPLACED by --yaw-centred-ladder dense. Plane CTRL-65.
# base's `--yaw-centred-ladder` (quantile) REPLACED by `--yaw-centred-ladder dense`, s1-s5, ABI 59.
# B rows = the existing Y+F runs ..._a59_..._ffF_full_sN (not re-flown: argv-identical apart from the ladder value,
# same ABI). Prereg docs/ctrl48_y3l_prereg.md (must exist before any run). END lines in runs.log; ends Y3L DONE.
cd /Users/lacg/wnn
L=logs/controller/ctrl48
VENV=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv
ABI=${ABI:-59}
SEEDS=${Y3L_SEEDS:-1 2 3 4 5}
DRY=${Y3L_DRY:-0}
abort() { echo "$(date -u +%FT%TZ) Y3L ABORT: $1" >> $L/runs.log; echo "Y3L ABORT: $1" >&2; exit 1; }
row()
{
	local g r
	g=$(grep -a 'stage-select\] HEADLINE stage' $1 | grep -oE 'genome=[^ ]+' | cut -d= -f2)
	r=$(grep -a "HEADLINE-$g MULTI-SEED" $1 | tail -1)
	[ -z "$r" ] && r=$(grep -aE "\[(test|report)-seeds\] ${g%%#*} MULTI-SEED" $1 | tail -1)
	echo "genome=$g $(echo "$r" | grep -oE 'stable=[^ ]+  err=[^ ]+  steady=[^ ]+|alt=[^ ]+|pos=[^ ]+|pos_hold=[^ ]+|dz_hold=[^ ]+|stable_fail=[^ ]+' | tr '\n' ' ')"
}
[ -f docs/ctrl48_y3l_prereg.md ] || abort "prereg docs/ctrl48_y3l_prereg.md missing"
[ "$DRY" = 1 ] || echo "$(date -u +%FT%TZ) Y3L START Y3+L block (Y3 + dense yaw ladder, no F), seeds $SEEDS, ABI $ABI (prereg docs/ctrl48_y3l_prereg.md)" >> $L/runs.log
for s in $SEEDS; do
	src=C-S2-d_wposx2_a${ABI}_rf_c200_dc_biasin_ms01_P_both_L1_dmy03_full_s$s
	[ -f $L/$src.done ] || abort "missing finished Y+F source run $src"
	tag=C-S2-d_wposx2_a${ABI}_rf_c200_dc_biasin_ms01_P_both_L1_dmy03_yld_full_s$s
	[ -f $L/$tag.done ] && continue
	argv=$($VENV/bin/python $L/rebuild_argv.py $L/$src.out fly $L/ckpt_$tag $L/${tag}_winner.yaml.gz) \
		|| abort "argv rebuild failed for $tag"
	[ "$(echo "$argv" | grep -o -- "--yaw-centred-ladder --" | wc -l | tr -d ' ')" = 1 ] || abort "$tag: base argv lacks exactly one bare --yaw-centred-ladder"
	argv=${argv/--yaw-centred-ladder --/--yaw-centred-ladder dense --}
	echo "$argv" | grep -q -- "--yaw-centred-ladder dense --" || abort "$tag: dense ladder not applied"
	echo "$argv" | grep -q -- "--yaw-trim-ff" && abort "$tag: Y3+L must not carry --yaw-trim-ff"
	echo "$argv" | grep -q -- "--delta-max-yaw 0.3" || abort "$tag: --delta-max-yaw 0.3 missing"
	echo "$argv" | grep -q -- "--delta-max 0.6 " || abort "$tag: --delta-max 0.6 missing"
	echo "$argv" | grep -q -- "--obs-yaw-from-att-est" || abort "$tag: --obs-yaw-from-att-est missing"
	echo "$argv" | grep -q -- "--base-seed $((31337000 + s)) " || abort "$tag: wrong base seed"
	echo "$argv" | grep -q -- "--select-tail p99" || abort "$tag: rule P missing"
	if [ "$DRY" = 1 ]; then
		echo "DRY $tag"; diff <(grep -a '^\[argv\] ' $L/$src.out | cut -c8- | tr ' ' '\n') <(echo "$argv" | tr ' ' '\n'); continue
	fi
	while pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null; do sleep 30; done
	PYTHONPATH=src/wnn $VENV/bin/python -c "import ram_controller as c, wnn.control._accel; assert c.ABI_VERSION == $ABI" \
		|| abort "ram_controller is not ABI $ABI (before $tag)"
	echo "$(date -u +%FT%TZ) START $tag [y3l = Y3 with dense yaw ladder, P, floor] (ABI $ABI, seed $((31337000 + s)), 35 units)" >> $L/runs.log
	eval "set -- $argv"
	PYTHONPATH=src/wnn nice -n 5 /usr/bin/time -l $VENV/bin/python -u -m wnn.control.phased_ga "$@" > $L/$tag.out 2>&1
	rc=$?
	echo "$(date -u +%FT%TZ) END $tag rc=$rc $(row $L/$tag.out)" >> $L/runs.log
	[ $rc -eq 0 ] && touch $L/$tag.done
done
[ "$DRY" = 1 ] || echo "$(date -u +%FT%TZ) Y3L DONE" >> $L/runs.log
