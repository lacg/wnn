#!/bin/bash
# W2.2 — brittleness audit (06/07/2026): clean-trained WNN winners + committees
# re-scored under calibrated weather. NO training — pure eval.
#   solos:  {w1_h4000_s09, pwm2k_s09, pwm2k_s10, e2_long_s09} x {L1, L2} @2000
#   panels: {C6_prod, C7_anch} x {L1, L2} @2000
# Rulers: clean fresh @2000 = w1s09 93.8 / pwm2k 89.0/88.0 / long_s09 88.2;
#         C6_prod 95.2 / C7_anch 96.0. PID+ anchors: L1 100 / L2 99.8; PD L2 84.0.
# Marker /tmp/wnn_w22_done.json. ONE controller job at a time (this is it).
set -u
cd /Users/lacg/wnn
PY=/Users/lacg/wnn-venv/bin/python
export PYTHONPATH=/Users/lacg/wnn/src/wnn
OUT=logs/controller/W22Brittleness_20260706
LOG=logs/controller/W22Brittleness_20260706.log
mkdir -p "$OUT"

log() { echo "[w22] $(date '+%F %T') $*" >> "$LOG"; }

SOLOS="w1_h4000_s09,pwm2k_s09,pwm2k_s10,e2_long_s09"
CORE6="pidmix_s10_R1,s16_s10_R1,pwm_s10_R2,lowedge_s16_in4_s09,stateint_A_ctrl_s09,e2_long_s09"
C7ANCH="$CORE6,e2_anch_s09"

for LV in L1 L2; do
	log "solos @$LV start"
	E4_STEPS=2000 E4_DIST=$LV E4_ONLY="$SOLOS" \
		$PY scripts/e4_best_of_k.py > "$OUT/solos_${LV}.out" 2>&1
	log "solos @$LV done (exit $?)"
done
for LV in L1 L2; do
	for entry in "C6_prod:$CORE6" "C7_anch:$C7ANCH"; do
		name="${entry%%:*}"
		members="${entry#*:}"
		n=$(echo "$members" | awk -F, '{print NF}')
		log "panel $name @$LV start"
		E4_SKIP_SOLO=1 E4_STEPS=2000 E4_DIST=$LV E4_ONLY="$members" E4_ENSEMBLE_TOP=$n \
			$PY scripts/e4_best_of_k.py > "$OUT/panel_${name}_${LV}.out" 2>&1
		log "panel $name @$LV done (exit $?, ${n} members)"
	done
done
echo "{\"w22_done\":true,\"ts\":\"$(date -u +%FT%TZ)\"}" > /tmp/wnn_w22_done.json
log "DRIVER COMPLETE"
