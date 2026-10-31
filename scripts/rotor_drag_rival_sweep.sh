cd /Users/lacg/wnn
for K in 0.0 0.1 0.2 0.3 0.5 0.8; do
  echo "##### k_d $K"
  PYTHONPATH=src/wnn nice -n 10 /Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python -u scripts/ctrl22_stage2_rivals.py \
    --run-args-file experiments/ctrl22/ARGS_op31_abi32_tilt_kd$K.txt --steps 2000 6000 \
    --out experiments/ctrl22/rivals_abi32_tilt_kd$K.json 2>&1 | grep -vE "^\s*$"
done
