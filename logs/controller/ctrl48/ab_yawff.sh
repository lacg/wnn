#!/bin/bash
# CTRL-48 yaw trim feed-forward block (Luiz 17/11 ~23:10 EST): arms B (base re-flown on ABI 58) vs F (base +
# --yaw-trim-ff), s1-s5 SEED-MAJOR. Prereg docs/ctrl48_yawff_prereg.md. Launched by chain_abi58_to_ff.sh.
# FF_BASE = M (both_L1 + floor + --obs-yaw-from-att-est) | Y (M + --delta-max-yaw 0.5), from the verdict file.
# ABI checked before EVERY run (FF ABORT); END lines in runs.log; ends FF DONE.
cd /Users/lacg/wnn
L=logs/controller/ctrl48
VENV=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv
ABI=${ABI:-58}
BASE=${FF_BASE:-M}
abort() { echo "$(date -u +%FT%TZ) FF ABORT: $1" >> $L/runs.log; exit 1; }
row()
{
	local g r
	g=$(grep -a 'stage-select\] HEADLINE stage' $1 | grep -oE 'genome=[^ ]+' | cut -d= -f2)
	r=$(grep -a "HEADLINE-$g MULTI-SEED" $1 | tail -1)
	[ -z "$r" ] && r=$(grep -aE "\[(test|report)-seeds\] ${g%%#*} MULTI-SEED" $1 | tail -1)
	echo "genome=$g $(echo "$r" | grep -oE 'stable=[^ ]+  err=[^ ]+  steady=[^ ]+|alt=[^ ]+|pos=[^ ]+|pos_hold=[^ ]+|dz_hold=[^ ]+|stable_fail=[^ ]+' | tr '\n' ' ')"
}
base_flags()
{
	case $BASE in
		M) echo "--zrank-pos-floor auto --obs-yaw-from-att-est" ;;
		Y) echo "--zrank-pos-floor auto --obs-yaw-from-att-est --delta-max-yaw 0.5" ;;
		*) abort "FF_BASE must be M or Y (got '$BASE')" ;;
	esac
}
flags()
{
	case $1 in
		B) base_flags ;;
		F) echo "$(base_flags) --yaw-trim-ff" ;;
	esac
}
echo "$(date -u +%FT%TZ) FF START yaw trim FF block arms B F (base $BASE), s1-s5 seed-major = 10 runs, ABI $ABI, rule P, floor on (prereg docs/ctrl48_yawff_prereg.md)" >> $L/runs.log
for s in 1 2 3 4 5; do
	src=C-S2-d_wposx2_a51_rf_c200_dc_biasin_ms01_P_both_L1_full_s$s
	[ -f $L/$src.out ] || abort "missing source .out $src"
	for arm in B F; do
		tag=C-S2-d_wposx2_a${ABI}_rf_c200_dc_biasin_ms01_P_both_L1_ff${arm}_full_s$s
		[ -f $L/$tag.done ] && continue
		argv=$($VENV/bin/python $L/rebuild_argv.py $L/$src.out fly $L/ckpt_$tag $L/${tag}_winner.yaml.gz $(flags $arm)) \
			|| abort "argv rebuild failed for $tag"
		echo "$argv" | grep -q -- "--delta-max 0.6 " || abort "$tag: base argv lacks --delta-max 0.6"
		[ "$arm" = B ] && { echo "$argv" | grep -q -- "--yaw-trim-ff" && abort "$tag: B must not carry --yaw-trim-ff"; }
		[ "$BASE" = M ] && { echo "$argv" | grep -q -- "--delta-max-yaw" && abort "$tag: base M must not carry --delta-max-yaw"; }
		echo "$argv" | grep -q -- "--base-seed $((31337000 + s)) " || abort "$tag: wrong base seed"
		echo "$argv" | grep -q -- "--select-tail p99" || abort "$tag: rule P missing"
		echo "$argv" | grep -q -- "--obs-tilt-ref-err" || abort "$tag: L1 missing"
		echo "$argv" | grep -q -- "--obs-cascade-dhat-residual" || abort "$tag: the residual block (FF prerequisite) missing"
		for f in $(flags $arm); do [[ $f == --* ]] && { echo "$argv" | grep -q -- "$f" || abort "$tag: flag $f missing"; }; done
		while pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null; do sleep 30; done
		PYTHONPATH=src/wnn $VENV/bin/python -c "import ram_controller as c, wnn.control._accel; assert c.ABI_VERSION == $ABI" \
			|| abort "ram_controller is not ABI $ABI (before $tag)"
		echo "$(date -u +%FT%TZ) START $tag [ff $arm, base $BASE, P, floor] (ABI $ABI, seed $((31337000 + s)), 35 units)" >> $L/runs.log
		eval "set -- $argv"
		PYTHONPATH=src/wnn nice -n 5 /usr/bin/time -l $VENV/bin/python -u -m wnn.control.phased_ga "$@" > $L/$tag.out 2>&1
		rc=$?
		echo "$(date -u +%FT%TZ) END $tag rc=$rc $(row $L/$tag.out)" >> $L/runs.log
		[ $rc -eq 0 ] && touch $L/$tag.done
	done
done
echo "$(date -u +%FT%TZ) FF DONE" >> $L/runs.log
