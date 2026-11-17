#!/usr/bin/env bash
# ABI 57 install + determinism gate — run ONLY after `DMX DONE` is in logs/controller/ctrl48/runs.log
# (the dm block's driver asserts ABI 56 before every run). Refuses otherwise. Merging is a separate,
# explicit step after PASS (the main tree's facade moves to 57 with the merge).
set -u
R=/Users/lacg/wnn; L=$R/logs/controller/ctrl48; WT=/Users/lacg/wnn-abi57; G=$R/logs/controller/detgate
VENV=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv
grep -q "DMX DONE" $L/runs.log || { echo "DMX DONE not in runs.log — refusing"; exit 2; }
pgrep -f "wnn.control.phased_g[a]" >/dev/null && { echo "a phased_ga is running — refusing"; exit 2; }
[ "$($VENV/bin/python -c 'import ram_controller as c; print(c.ABI_VERSION)')" = 56 ] || { echo "installed wheel is not ABI 56"; exit 2; }
$WT/scripts/ctrl_determinism_gate.sh fly $G/dmy_abi56 $R/src || exit 2
( cd $R && unset CONDA_PREFIX && source wnn/bin/activate && cd $WT/src/wnn/ram/strategies/accelerator && maturin develop --release -m controller/Cargo.toml ) || exit 2
[ "$($VENV/bin/python -c 'import ram_controller as c; print(c.ABI_VERSION)')" = 57 ] || { echo "ABI 57 did not install"; exit 2; }
for r in dmy_abi57a dmy_abi57b; do $WT/scripts/ctrl_determinism_gate.sh fly $G/$r $WT/src || exit 2; done
GATE_SRC=$WT/src $WT/scripts/ctrl_determinism_gate.sh compare $G/dmy_abi57a $G/dmy_abi57b $G/dmy_abi56
