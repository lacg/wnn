#!/bin/bash
# CTBR imitation probe (06/11/2026): diag_imitation_probe_recipe on a cascade fixture, release build, nice 10.
# Usage: run.sh FIXTURE_TAG OUT_TAG [ENV=VAL ...]
cd /Users/lacg/wnn/src/wnn/ram/strategies/accelerator
fx=$1; out=$2; shift 2
L=/Users/lacg/wnn/logs/controller/ctbr_probe
env "$@" WNN_IMIT_FIXTURE=$L/$fx.txt PYO3_PYTHON=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python \
  nice -n 10 cargo test --release -p ram_controller --lib --no-default-features diag_imitation_probe_recipe -- --ignored --nocapture > $L/$out.out 2>&1
