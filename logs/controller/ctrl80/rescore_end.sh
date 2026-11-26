#!/bin/bash
# CTRL-80 per-END re-score: rescore_end.sh <arm_tag> <control_tag>
# CPU only, nice 15, ~20-60 s per uncached winner. Read-only on the runs: it loads the published winner
# (ctrl48/<tag>_winner.yaml.gz), re-scores it on the 35 TEST units with the production scorer
# (ctrl73/wnn_units.py), caches the npz under ctrl80/data/, gates on the logged END headline, prints the table.
# Exit: 0 table printed | 2 a re-score does not reproduce its logged END | 3 usage / missing input | 4 re-score crashed.
set -u
PY=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python
ROOT=/Users/lacg/wnn/logs/controller
HERE=$ROOT/ctrl80
[ $# -eq 2 ] || { echo "usage: rescore_end.sh <arm_tag> <control_tag>" >&2; exit 3; }
mkdir -p "$HERE/data"

npz_for() {  # prints the npz path of a tag, re-scoring it when no cached copy exists
	local tag=$1 s out old
	s=${tag##*_s}
	out=$HERE/data/$tag.npz
	old=$ROOT/ctrl73/data/n0g3gz_s$s.npz   # the G+T controls re-scored for CTRL-73 (same tool)
	case "$tag" in *_yld_n0g3gz_full_s$s) [ -f "$old" ] && { echo "$old"; return 0; } ;; esac
	if [ ! -f "$out" ]; then
		[ -f "$ROOT/ctrl48/${tag}_winner.yaml.gz" ] || { echo "RESCORE FAIL: no published winner for $tag" >&2; return 3; }
		local tmp=$HERE/data/.tmp.$$.npz
		nice -n 15 "$PY" "$ROOT/ctrl73/wnn_units.py" "$tag" "$tmp" > "$HERE/data/$tag.log" 2>&1 \
			|| { echo "RESCORE FAIL: re-score crashed for $tag (see $HERE/data/$tag.log)" >&2; rm -f "$tmp"; return 4; }
		mv "$tmp" "$out"
	fi
	echo "$out"
}

A=$(npz_for "$1") || exit $?
C=$(npz_for "$2") || exit $?
exec nice -n 15 "$PY" "$HERE/rescore_end.py" "$1" "$A" "$2" "$C"
