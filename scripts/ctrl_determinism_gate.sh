#!/usr/bin/env bash
# Controller DETERMINISM GATE (Luiz 17/11/2026). Run it for every default-off controller ABI bump:
# the same small phased_ga flown on the OLD and NEW build must produce identical gen lines,
# stage-select lines and checkpoint fingerprints (and two NEW-build runs identical [gen-digest]s).
#
#   scripts/ctrl_determinism_gate.sh fly <run_dir> <python_src_root> [base.out]
#       one gate run under the INSTALLED ram_controller wheel and the Python tree at
#       <python_src_root> (the dir that CONTAINS the `wnn` package, e.g. <repo>/src).
#       base.out defaults to the adopted base (yfeM s1). Refuses while any phased_ga runs.
#   scripts/ctrl_determinism_gate.sh compare <run_dir> <run_dir> [...]
#       exit 0 = PASS. The first run_dir is the reference.
#
# Typical (wheel swaps between legs are the caller's: maturin develop / restore a kept copy):
#   fly gate/old  <old_tree>/src   (old wheel installed)
#   fly gate/new1 <new_tree>/src   (new wheel installed)
#   fly gate/new2 <new_tree>/src
#   compare gate/new1 gate/new2 gate/old
set -u
VENV=${VENV:-/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv}
HERE=$(cd "$(dirname "$0")" && pwd)
BASE_DEFAULT=/Users/lacg/wnn/logs/controller/ctrl48/C-S2-d_wposx2_a55_rf_c200_dc_biasin_ms01_P_both_L1_yfeM_full_s1.out
cmd=${1:-}; shift || true
case "$cmd" in
	fly)
		dir=$1; src=$2; base=${3:-$BASE_DEFAULT}
		if pgrep -f "wnn.control.phased_g[a]" >/dev/null; then echo "[gate] a phased_ga is running — refusing (the gate needs the box quiet)"; exit 2; fi
		mkdir -p "$dir"
		abi=$(PYTHONPATH="$src" "$VENV/bin/python" -c "import ram_controller as c; print(c.ABI_VERSION)") || exit 2
		argv=$(PYTHONPATH="$src" "$VENV/bin/python" "$HERE/ctrl_determinism_gate.py" argv "$base" "$dir") || exit 2
		echo "$argv" > "$dir/argv"; echo "abi=$abi src=$src base=$base" > "$dir/meta"
		echo "[gate] fly $dir (ABI $abi, $src) $(TZ=America/New_York date '+%d/%m/%Y %H:%M %Z')"
		eval "set -- $argv"
		PYTHONPATH="$src" "$VENV/bin/python" -u -m wnn.control.phased_ga "$@" > "$dir/run.out" 2>&1
		rc=$?
		echo "rc=$rc" >> "$dir/meta"
		[ $rc -eq 0 ] || { echo "[gate] run failed rc=$rc ($dir/run.out)"; exit 2; }
		PYTHONPATH="$src" "$VENV/bin/python" "$HERE/ctrl_determinism_gate.py" fp "$dir" || exit 2
		echo "[gate] done $dir $(TZ=America/New_York date '+%H:%M %Z')"
		;;
	compare)
		PYTHONPATH="${GATE_SRC:-/Users/lacg/wnn/src}" "$VENV/bin/python" "$HERE/ctrl_determinism_gate.py" compare "$@"
		;;
	*)
		sed -n '2,18p' "$0"; exit 2
		;;
esac
