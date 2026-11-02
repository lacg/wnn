cd /Users/lacg/wnn
for K in 0.78 0.5 0.25 0.0; do
  echo "##### k_d $K $(date -u +%FT%TZ)"
  PYTHONPATH=src/wnn nice -n 10 /Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python -u scripts/ctrl22_stage2_rivals.py \
    --run-args-file experiments/ctrl22/ARGS_ctrl8p50_abi33_kd$K.txt --steps 2000 6000 \
    --out experiments/ctrl22/rivals_abi33_tilt_kd$K.json 2>&1 | grep -vE "^\s*$"
done
echo "##### DONE $(date -u +%FT%TZ)"
