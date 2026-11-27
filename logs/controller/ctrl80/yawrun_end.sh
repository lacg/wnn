#!/bin/bash
# CTRL-80 yaw-runaway count: yawrun_end.sh <arm_tag> <control_tag>
# TEST REPORT METRIC ONLY (35 CF2.1 TEST units x 100 eps): NEVER use it to select, tune or design anything.
# CPU only, nice 15, ~45 s per uncached tag (WORKERS=n to fan units out over n processes; default 1).
# Read-only on the runs: loads the published winner (ctrl48/<tag>_winner.yaml.gz), replays every TEST episode per
# tick on the production scorer's captured call (yawrun_end.py, gated per episode vs the Rust scorer), caches
# ctrl80/data/yawrun_<tag>.npz, then asserts the per-episode err against the TEST re-score npz (the same
# resolution as rescore_end.sh; re-scored on demand) and prints ONE table, control | arm.
# Exit: 0 table printed | 2 the replay is not the scored rollout | 3 usage / missing input | 4 a step crashed.
set -u
PY=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python
ROOT=/Users/lacg/wnn/logs/controller
HERE=$ROOT/ctrl80
[ $# -eq 2 ] || { echo "usage: yawrun_end.sh <arm_tag> <control_tag>" >&2; exit 3; }
mkdir -p "$HERE/data"

rescore_for() {  # rescore_end.sh's npz_for: the TEST re-score npz of a tag (re-scored when absent)
	local tag=$1 s out old
	s=${tag##*_s}
	out=$HERE/data/$tag.npz
	old=$ROOT/ctrl73/data/n0g3gz_s$s.npz
	case "$tag" in *_yld_n0g3gz_full_s$s) [ -f "$old" ] && { echo "$old"; return 0; } ;; esac
	if [ ! -f "$out" ]; then
		[ -f "$ROOT/ctrl48/${tag}_winner.yaml.gz" ] || { echo "YAWRUN FAIL: no published winner for $tag" >&2; return 3; }
		local tmp=$HERE/data/.tmp.$$.npz
		nice -n 15 "$PY" "$ROOT/ctrl73/wnn_units.py" "$tag" "$tmp" > "$HERE/data/$tag.log" 2>&1 \
			|| { echo "YAWRUN FAIL: re-score crashed for $tag (see $HERE/data/$tag.log)" >&2; rm -f "$tmp"; return 4; }
		mv "$tmp" "$out"
	fi
	echo "$out"
}

yawrun_for() {  # the per-episode yaw npz of a tag (flown when absent)
	local tag=$1 out rc
	out=$HERE/data/yawrun_$tag.npz
	if [ ! -f "$out" ]; then
		[ -f "$ROOT/ctrl48/${tag}_winner.yaml.gz" ] || { echo "YAWRUN FAIL: no published winner for $tag" >&2; return 3; }
		local tmp=$HERE/data/.tmp.yaw.$$.npz
		nice -n 15 "$PY" "$HERE/yawrun_end.py" fly "$tag" "$tmp" test > "$HERE/data/yawrun_$tag.log" 2>&1
		rc=$?
		if [ $rc -ne 0 ]; then
			rm -f "$tmp"; cat "$HERE/data/yawrun_$tag.log" >&2
			[ $rc -eq 2 ] && return 2
			echo "YAWRUN FAIL: replay crashed for $tag (see $HERE/data/yawrun_$tag.log)" >&2; return 4
		fi
		mv "$tmp" "$out"
	fi
	echo "$out"
}

AR=$(rescore_for "$1") || exit $?
CR=$(rescore_for "$2") || exit $?
AY=$(yawrun_for "$1") || exit $?
CY=$(yawrun_for "$2") || exit $?
exec nice -n 15 "$PY" "$HERE/yawrun_end.py" table "$1" "$AY" "$AR" "$2" "$CY" "$CR"
