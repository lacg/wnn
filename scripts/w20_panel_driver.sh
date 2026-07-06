#!/bin/bash
# W2.0 + w1_h4000_s09 panels driver (06/07/2026), sequential:
#   1) W2.0 disturbance-ladder calibration (scripts/w2_calibrate.py — PID vs PD
#      x {OFF,L1,L2,L3} x {500,2000,5000}, fresh seeds)
#   2) Committee panels including w1_h4000_s09 (the 93.8-fresh single, absent
#      from every E4 panel) @ {2000,5000,10000} via e4_best_of_k.py:
#        C7_w1s09      = production core6 + w1_h4000_s09
#        C6_w1s09      = core6 with stateint_A_ctrl_s09 -> w1_h4000_s09 (swap)
#        C8_w1s09_pwm2k= core6 + pwm2k_s09 + w1_h4000_s09
# Markers: /tmp/wnn_w20_cal_done.json, /tmp/wnn_w20_done.json (final).
# ONE controller job at a time — this is the only controller compute.
set -u
cd /Users/lacg/wnn
PY=/Users/lacg/wnn-venv/bin/python
export PYTHONPATH=/Users/lacg/wnn/src/wnn
OUT=logs/controller/W20Panels_20260706
LOG=logs/controller/W20Panels_20260706.log
mkdir -p "$OUT"

log() { echo "[w20] $(date '+%F %T') $*" >> "$LOG"; }

log "W2.0 calibration start"
$PY scripts/w2_calibrate.py > "$OUT/w20_calibration.out" 2>&1
log "W2.0 calibration done (exit $?)"
echo "{\"w20_cal\":true,\"ts\":\"$(date -u +%FT%TZ)\"}" > /tmp/wnn_w20_cal_done.json

CORE6="pidmix_s10_R1,s16_s10_R1,pwm_s10_R2,lowedge_s16_in4_s09,stateint_A_ctrl_s09,e2_long_s09"
CORE5="pidmix_s10_R1,s16_s10_R1,pwm_s10_R2,lowedge_s16_in4_s09,e2_long_s09"
PANELS="
C7_w1s09:$CORE6,w1_h4000_s09
C6_w1s09:$CORE5,w1_h4000_s09
C8_w1s09_pwm2k:$CORE6,pwm2k_s09,w1_h4000_s09
"
log "panels start"
for entry in $PANELS; do
	name="${entry%%:*}"
	members="${entry#*:}"
	n=$(echo "$members" | awk -F, '{print NF}')
	for S in 2000 5000 10000; do
		E4_SKIP_SOLO=1 E4_STEPS=$S E4_ONLY="$members" E4_ENSEMBLE_TOP=$n \
			$PY scripts/e4_best_of_k.py > "$OUT/panel_${name}_${S}.out" 2>&1
		log "panel ${name} @${S} done (exit $?, ${n} members)"
	done
done
echo "{\"w20_done\":true,\"ts\":\"$(date -u +%FT%TZ)\"}" > /tmp/wnn_w20_done.json
log "DRIVER COMPLETE"
