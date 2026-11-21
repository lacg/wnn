#!/bin/bash
# CTRL-58 A2 block (Luiz 20/11 ~19:20 EST: "close Y3+L at n=5 and go to A2"): arm A2 = base Y+F+L +
# --refine-fitness --refine-w-steady 0.40 --refine-w-mid 0.25 --refine-w-reward 0.10 (tail 0.15 / pos 0.10 = defaults),
# s1-s5 seed-major, ABI 60. B rows = the Y+F+L runs ..._a59_..._ffF_yld_full_sN (not re-flown: argv-identical apart from
# the refine flags; valid because the flag-off determinism gate ABI 60 vs 59 PASSED before this driver may start).
# Prereg docs/ctrl58_a2_prereg.md (must exist). END lines in runs.log; ends A2 DONE (abort 'A2 ABORT').
cd /Users/lacg/wnn
L=logs/controller/ctrl48
VENV=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv
ABI=${ABI:-60}
SEEDS=${A2_SEEDS:-1 2 3 4 5}
DRY=${A2_DRY:-0}
EXTRA="--refine-fitness --refine-w-steady 0.40 --refine-w-mid 0.25 --refine-w-reward 0.10"
abort() { echo "$(date -u +%FT%TZ) A2 ABORT: $1" >> $L/runs.log; echo "A2 ABORT: $1" >&2; exit 1; }
row()
{
	local g r
	g=$(grep -a 'stage-select\] HEADLINE stage' $1 | grep -oE 'genome=[^ ]+' | cut -d= -f2)
	r=$(grep -a "HEADLINE-$g MULTI-SEED" $1 | tail -1)
	[ -z "$r" ] && r=$(grep -aE "\[(test|report)-seeds\] ${g%%#*} MULTI-SEED" $1 | tail -1)
	echo "genome=$g $(echo "$r" | grep -oE 'stable=[^ ]+  err=[^ ]+  steady=[^ ]+|alt=[^ ]+|pos=[^ ]+|pos_hold=[^ ]+|dz_hold=[^ ]+|stable_fail=[^ ]+' | tr '\n' ' ')"
}
[ -f docs/ctrl58_a2_prereg.md ] || abort "prereg docs/ctrl58_a2_prereg.md missing"
grep -q "Launch (20/11/2026" docs/ctrl58_a2_prereg.md || abort "prereg has no launch section"
[ "$DRY" = 1 ] || echo "$(date -u +%FT%TZ) A2 START refine fitness (steady .40 / mid .25 / tail .15 / reward .10 / pos .10) on base Y+F+L, seeds $SEEDS, ABI $ABI (prereg docs/ctrl58_a2_prereg.md)" >> $L/runs.log
for s in $SEEDS; do
	src=C-S2-d_wposx2_a59_rf_c200_dc_biasin_ms01_P_both_L1_ffF_yld_full_s$s
	[ -f $L/$src.done ] || abort "missing finished Y+F+L source run $src"
	tag=C-S2-d_wposx2_a${ABI}_rf_c200_dc_biasin_ms01_P_both_L1_ffF_yld_a2_full_s$s
	[ -f $L/$tag.done ] && continue
	argv=$($VENV/bin/python $L/rebuild_argv.py $L/$src.out fly $L/ckpt_$tag $L/${tag}_winner.yaml.gz $EXTRA) \
		|| abort "argv rebuild failed for $tag"
	echo "$argv" | grep -q -- "--yaw-centred-ladder dense" || abort "$tag: dense ladder missing"
	echo "$argv" | grep -q -- "--yaw-trim-ff" || abort "$tag: --yaw-trim-ff missing"
	echo "$argv" | grep -q -- "--delta-max-yaw 0.5" || abort "$tag: --delta-max-yaw 0.5 missing"
	echo "$argv" | grep -q -- "--obs-yaw-from-att-est" || abort "$tag: --obs-yaw-from-att-est missing"
	echo "$argv" | grep -q -- "--base-seed $((31337000 + s)) " || abort "$tag: wrong base seed"
	echo "$argv" | grep -q -- "--select-tail p99" || abort "$tag: rule P missing"
	echo "$argv" | grep -q -- "--refine-w-mid 0.25" || abort "$tag: refine flags missing"
	if [ "$DRY" = 1 ]; then
		echo "DRY $tag"; diff <(grep -a '^\[argv\] ' $L/$src.out | cut -c8- | tr ' ' '\n') <(echo "$argv" | tr ' ' '\n'); continue
	fi
	while pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null; do sleep 30; done
	PYTHONPATH=src/wnn $VENV/bin/python -c "import ram_controller as c, wnn.control._accel; assert c.ABI_VERSION == $ABI" \
		|| abort "ram_controller is not ABI $ABI (before $tag)"
	echo "$(date -u +%FT%TZ) START $tag [a2 = Y+F+L + refine fitness steady/mid, P, floor] (ABI $ABI, seed $((31337000 + s)), 35 units)" >> $L/runs.log
	eval "set -- $argv"
	PYTHONPATH=src/wnn nice -n 5 /usr/bin/time -l $VENV/bin/python -u -m wnn.control.phased_ga "$@" > $L/$tag.out 2>&1
	rc=$?
	echo "$(date -u +%FT%TZ) END $tag rc=$rc $(row $L/$tag.out)" >> $L/runs.log
	[ $rc -eq 0 ] && touch $L/$tag.done
done
[ "$DRY" = 1 ] || echo "$(date -u +%FT%TZ) A2 DONE" >> $L/runs.log
