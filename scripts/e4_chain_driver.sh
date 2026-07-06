#!/bin/bash
# E4 chain driver (06/07/2026) — post-W1 controller assembly, 3 sequential legs:
#   A) common-ruler decay curves for the 8 horizon-surface winners
#      (scripts/w1_common_ruler.py — {0.5,1,2.5,5,10,20}x own H, fresh seeds)
#   B) fresh-seed truth serum @2000 + @5000 over the C2K pool + W1 winners
#      (e4_best_of_k.py E4_ONLY=<12 @2000-era winners>)
#   C) committee panels of 6-8 (mean-PWM) @2000/5000/10000 — production core
#      vs PWM2K upgrades vs the ANCH-@500 audition
# Markers: /tmp/wnn_e4chain_lega.json, _legb.json, done marker _done.json.
# Monitor: scripts/e4_chain_status.py. ONE controller job at a time (this IS it).
set -u
cd /Users/lacg/wnn
PY=/Users/lacg/wnn-venv/bin/python
export PYTHONPATH=/Users/lacg/wnn/src/wnn
OUT=logs/controller/E4Chain_20260706
LOG=logs/controller/E4Chain_20260706.log
mkdir -p "$OUT"

log() { echo "[e4chain] $(date '+%F %T') $*" >> "$LOG"; }

log "LEG A start — common-ruler decay curves"
$PY scripts/w1_common_ruler.py > "$OUT/leg_a_common_ruler.out" 2>&1
log "LEG A done (exit $?)"
echo "{\"leg\":\"A\",\"ts\":\"$(date -u +%FT%TZ)\"}" > /tmp/wnn_e4chain_lega.json

TS_LABELS="pwm2k_s09,pwm2k_s10,lean2k_s09,lean2k_s10,tilt2k_s09,tilt2k_s10,anch2k_s09,anch2k_s10,w1_h1000_s09,w1_h1000_s10,w1_h4000_s09,w1_h4000_s10"
log "LEG B start — truth serum over C2K pool + W1 winners"
for S in 2000 5000; do
	E4_STEPS=$S E4_ONLY="$TS_LABELS" \
		$PY scripts/e4_best_of_k.py > "$OUT/leg_b_truth_serum_${S}.out" 2>&1
	log "LEG B @${S} done (exit $?)"
done
echo "{\"leg\":\"B\",\"ts\":\"$(date -u +%FT%TZ)\"}" > /tmp/wnn_e4chain_legb.json

# Committee panels. Core6 = the production 6-member (5-member + LONG_s09).
CORE6="pidmix_s10_R1,s16_s10_R1,pwm_s10_R2,lowedge_s16_in4_s09,stateint_A_ctrl_s09,e2_long_s09"
PANELS="
C6_prod:$CORE6
C7_long:$CORE6,e2_long_s10
C7_pwm2k:$CORE6,pwm2k_s09
C7_anch:$CORE6,e2_anch_s09
C8_pwm2k_w1:$CORE6,pwm2k_s09,w1_h4000_s10
C8_2xpwm2k:$CORE6,pwm2k_s09,pwm2k_s10
"
log "LEG C start — committee panels"
for entry in $PANELS; do
	name="${entry%%:*}"
	members="${entry#*:}"
	n=$(echo "$members" | awk -F, '{print NF}')
	for S in 2000 5000 10000; do
		E4_SKIP_SOLO=1 E4_STEPS=$S E4_ONLY="$members" E4_ENSEMBLE_TOP=$n \
			$PY scripts/e4_best_of_k.py > "$OUT/leg_c_${name}_${S}.out" 2>&1
		log "LEG C ${name} @${S} done (exit $?, ${n} members)"
	done
done
echo "{\"e4chain_done\":true,\"ts\":\"$(date -u +%FT%TZ)\"}" > /tmp/wnn_e4chain_done.json
log "CHAIN COMPLETE"
