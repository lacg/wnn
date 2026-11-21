#!/bin/bash
# CTRL-48 yaw decode-gain-1 block (Luiz 19/11 ~18:40 EST: "is F needed at all?"): arm Y3 = adopted base (both_L1 + floor +
# --obs-yaw-from-att-est, --delta-max 0.6) + `--delta-max-yaw 0.3` (yaw DC gain G = 0.3/(1-0.7) = 1), WITHOUT --yaw-trim-ff,
# s1-s5, ABI 59. Rebuilt from each seed's a51 both_L1 .out exactly as the Y rows were (ab_yawff.sh base Y, arm B) with 0.3
# in place of 0.5. Compared against the existing Y rows (dmy Y s1-5) and Y+F rows (..._a59_..._ffF_full_s1-5), not re-flown.
# Prereg docs/ctrl48_dmaxyaw03_prereg.md (must exist). END lines in runs.log; ends DY3 DONE.
cd /Users/lacg/wnn
L=logs/controller/ctrl48
VENV=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv
ABI=${ABI:-59}
SEEDS=${DY3_SEEDS:-1 2 3 4 5}
DRY=${DY3_DRY:-0}
FLAGS="--zrank-pos-floor auto --obs-yaw-from-att-est --delta-max-yaw 0.3"
abort() { echo "$(date -u +%FT%TZ) DY3 ABORT: $1" >> $L/runs.log; echo "DY3 ABORT: $1" >&2; exit 1; }
row()
{
	local g r
	g=$(grep -a 'stage-select\] HEADLINE stage' $1 | grep -oE 'genome=[^ ]+' | cut -d= -f2)
	r=$(grep -a "HEADLINE-$g MULTI-SEED" $1 | tail -1)
	[ -z "$r" ] && r=$(grep -aE "\[(test|report)-seeds\] ${g%%#*} MULTI-SEED" $1 | tail -1)
	echo "genome=$g $(echo "$r" | grep -oE 'stable=[^ ]+  err=[^ ]+  steady=[^ ]+|alt=[^ ]+|pos=[^ ]+|pos_hold=[^ ]+|dz_hold=[^ ]+|stable_fail=[^ ]+' | tr '\n' ' ')"
}
[ "$DRY" = 1 ] || [ -f docs/ctrl48_dmaxyaw03_prereg.md ] || abort "prereg docs/ctrl48_dmaxyaw03_prereg.md missing"
[ "$DRY" = 1 ] || echo "$(date -u +%FT%TZ) DY3 START yaw decode-gain-1 block, arm Y3 (--delta-max-yaw 0.3, no F), seeds $SEEDS, ABI $ABI (prereg docs/ctrl48_dmaxyaw03_prereg.md)" >> $L/runs.log
for s in $SEEDS; do
	src=C-S2-d_wposx2_a51_rf_c200_dc_biasin_ms01_P_both_L1_full_s$s
	[ -f $L/$src.out ] || abort "missing source .out $src"
	tag=C-S2-d_wposx2_a${ABI}_rf_c200_dc_biasin_ms01_P_both_L1_dmy03_full_s$s
	[ -f $L/$tag.done ] && continue
	argv=$($VENV/bin/python $L/rebuild_argv.py $L/$src.out fly $L/ckpt_$tag $L/${tag}_winner.yaml.gz $FLAGS) \
		|| abort "argv rebuild failed for $tag"
	echo "$argv" | grep -q -- "--delta-max 0.6 " || abort "$tag: base argv lacks --delta-max 0.6"
	echo "$argv" | grep -q -- "--delta-max-yaw 0.3" || abort "$tag: --delta-max-yaw 0.3 missing"
	echo "$argv" | grep -q -- "--yaw-trim-ff" && abort "$tag: Y3 must not carry --yaw-trim-ff"
	echo "$argv" | grep -q -- "--yaw-centred-ladder dense" && abort "$tag: Y3 must keep the quantile yaw ladder"
	echo "$argv" | grep -q -- "--obs-yaw-from-att-est" || abort "$tag: --obs-yaw-from-att-est missing"
	echo "$argv" | grep -q -- "--zrank-pos-floor auto" || abort "$tag: floor missing"
	echo "$argv" | grep -q -- "--base-seed $((31337000 + s)) " || abort "$tag: wrong base seed"
	echo "$argv" | grep -q -- "--select-tail p99" || abort "$tag: rule P missing"
	echo "$argv" | grep -q -- "--obs-tilt-ref-err" || abort "$tag: L1 missing"
	if [ "$DRY" = 1 ]; then
		y=$L/C-S2-d_wposx2_a57_rf_c200_dc_biasin_ms01_P_both_L1_dmyY_full_s$s.out
		echo "DRY $tag  (diff vs the flown Y row's argv)"; diff <(grep -a '^\[argv\] ' $y | cut -c8- | tr ' ' '\n') <(echo "$argv" | tr ' ' '\n'); continue
	fi
	while pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null; do sleep 30; done
	PYTHONPATH=src/wnn $VENV/bin/python -c "import ram_controller as c, wnn.control._accel; assert c.ABI_VERSION == $ABI" \
		|| abort "ram_controller is not ABI $ABI (before $tag)"
	echo "$(date -u +%FT%TZ) START $tag [dmy03 Y3 = base + --delta-max-yaw 0.3, no F, P, floor] (ABI $ABI, seed $((31337000 + s)), 35 units)" >> $L/runs.log
	eval "set -- $argv"
	PYTHONPATH=src/wnn nice -n 5 /usr/bin/time -l $VENV/bin/python -u -m wnn.control.phased_ga "$@" > $L/$tag.out 2>&1
	rc=$?
	echo "$(date -u +%FT%TZ) END $tag rc=$rc $(row $L/$tag.out)" >> $L/runs.log
	[ $rc -eq 0 ] && touch $L/$tag.done
done
[ "$DRY" = 1 ] || echo "$(date -u +%FT%TZ) DY3 DONE" >> $L/runs.log
