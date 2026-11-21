#!/bin/bash
# CTRL-57 A1 block (Luiz 20/11 ~20:00 EST: "queue A1 soon after A2, adopted or not"): arm A1 = base Y+F+L with the
# stage stopper loosened: --neurons-patience 3 -> 6 (both occurrences in the base argv) and --memory-patience 2 -> 10.
# Question: does the search keep improving steady when allowed to run? No code change. s1-s5 seed-major on the
# installed ABI (60 after the A2 chain; the flags exist on 59 too). B rows = the Y+F+L runs ..._a59_..._ffF_yld_full_sN
# (not re-flown; valid via the flag-off determinism gate ABI 60 vs 59). Prereg docs/ctrl57_a1_prereg.md.
# END lines in runs.log; ends A1 DONE (abort 'A1 ABORT').
cd /Users/lacg/wnn
L=logs/controller/ctrl48
VENV=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv
ABI=${ABI:-60}
SEEDS=${A1_SEEDS:-1 2 3 4 5}
DRY=${A1_DRY:-0}
abort() { echo "$(date -u +%FT%TZ) A1 ABORT: $1" >> $L/runs.log; echo "A1 ABORT: $1" >&2; exit 1; }
row()
{
	local g r
	g=$(grep -a 'stage-select\] HEADLINE stage' $1 | grep -oE 'genome=[^ ]+' | cut -d= -f2)
	r=$(grep -a "HEADLINE-$g MULTI-SEED" $1 | tail -1)
	[ -z "$r" ] && r=$(grep -aE "\[(test|report)-seeds\] ${g%%#*} MULTI-SEED" $1 | tail -1)
	echo "genome=$g $(echo "$r" | grep -oE 'stable=[^ ]+  err=[^ ]+  steady=[^ ]+|alt=[^ ]+|pos=[^ ]+|pos_hold=[^ ]+|dz_hold=[^ ]+|stable_fail=[^ ]+' | tr '\n' ' ')"
}
[ -f docs/ctrl57_a1_prereg.md ] || abort "prereg docs/ctrl57_a1_prereg.md missing"
[ "$DRY" = 1 ] || echo "$(date -u +%FT%TZ) A1 START looser stopper (neurons-patience 6, memory-patience 10) on base Y+F+L, seeds $SEEDS, ABI $ABI (prereg docs/ctrl57_a1_prereg.md)" >> $L/runs.log
for s in $SEEDS; do
	src=C-S2-d_wposx2_a59_rf_c200_dc_biasin_ms01_P_both_L1_ffF_yld_full_s$s
	[ -f $L/$src.done ] || abort "missing finished Y+F+L source run $src"
	tag=C-S2-d_wposx2_a${ABI}_rf_c200_dc_biasin_ms01_P_both_L1_ffF_yld_a1_full_s$s
	[ -f $L/$tag.done ] && continue
	argv=$($VENV/bin/python $L/rebuild_argv.py $L/$src.out fly $L/ckpt_$tag $L/${tag}_winner.yaml.gz) \
		|| abort "argv rebuild failed for $tag"
	[ "$(echo "$argv" | grep -o -- "--neurons-patience 3 " | wc -l | tr -d ' ')" -ge 1 ] || abort "$tag: base argv lacks --neurons-patience 3"
	[ "$(echo "$argv" | grep -o -- "--memory-patience 2 " | wc -l | tr -d ' ')" = 1 ] || abort "$tag: base argv lacks exactly one --memory-patience 2"
	argv=${argv//--neurons-patience 3 /--neurons-patience 6 }
	argv=${argv/--memory-patience 2 /--memory-patience 10 }
	echo "$argv" | grep -q -- "--neurons-patience 3 " && abort "$tag: a --neurons-patience 3 survived"
	echo "$argv" | grep -q -- "--memory-patience 10 " || abort "$tag: memory patience not applied"
	echo "$argv" | grep -q -- "--yaw-centred-ladder dense" || abort "$tag: dense ladder missing"
	echo "$argv" | grep -q -- "--yaw-trim-ff" || abort "$tag: --yaw-trim-ff missing"
	echo "$argv" | grep -q -- "--refine-fitness" && abort "$tag: refine flags must NOT be in A1"
	echo "$argv" | grep -q -- "--base-seed $((31337000 + s)) " || abort "$tag: wrong base seed"
	echo "$argv" | grep -q -- "--select-tail p99" || abort "$tag: rule P missing"
	if [ "$DRY" = 1 ]; then
		echo "DRY $tag"; diff <(grep -a '^\[argv\] ' $L/$src.out | cut -c8- | tr ' ' '\n') <(echo "$argv" | tr ' ' '\n'); continue
	fi
	while pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null; do sleep 30; done
	PYTHONPATH=src/wnn $VENV/bin/python -c "import ram_controller as c, wnn.control._accel; assert c.ABI_VERSION == $ABI" \
		|| abort "ram_controller is not ABI $ABI (before $tag)"
	echo "$(date -u +%FT%TZ) START $tag [a1 = Y+F+L + looser stopper (np 6, mp 10), P, floor] (ABI $ABI, seed $((31337000 + s)), 35 units)" >> $L/runs.log
	eval "set -- $argv"
	PYTHONPATH=src/wnn nice -n 5 /usr/bin/time -l $VENV/bin/python -u -m wnn.control.phased_ga "$@" > $L/$tag.out 2>&1
	rc=$?
	echo "$(date -u +%FT%TZ) END $tag rc=$rc $(row $L/$tag.out)" >> $L/runs.log
	[ $rc -eq 0 ] && touch $L/$tag.done
done
[ "$DRY" = 1 ] || echo "$(date -u +%FT%TZ) A1 DONE" >> $L/runs.log
