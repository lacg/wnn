#!/usr/bin/env bash
# Yaw-state A/B under L2 (Phase-4 follow-up, 19/07/2026).
# Motivated by scripts/sensor_degradation_counter.py: the yaw-blind 9-feature
# student faces a 12.8% conflict rate under L2 (yaw unobservability — gust
# wanders yaw, gravity can't see it), collapsing to 0.7% with yaw in the obs.
# Two arms, SEQUENTIAL (one controller at a time), both the Phase-3 BINARY
# production recipe (run_gran_phase3.sh argv VERBATIM) + --disturbance L2:
#   A = baseline 9-feature student (yaw-blind). Question: does the GA/split
#       trainer NOW recruit state cells (Phase-3 clean sim: 0 state cells)?
#   B = + --obs-yaw-err (gyro-z dead-reckoned yaw error, t=0 anchored — the
#       hand-built version of the state the counter says is missing).
# Predictions: B held-out stable% > A; B leaves state EMPTY (pressure gone);
# A either recruits state (DFA story) or eats the 12.8% as instability.
# Compare: held-out triple (err/stable/steady) + state cells via gran_fpga_count.
#   bash scripts/run_yawab_L2.sh   # or via detach_launch.py
set -u
ROOT="/Users/lacg/wnn"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
export PYTHONPATH="${ROOT}/src/wnn:${PYTHONPATH:-}"
cd "$ROOT" || exit 1
BASE="${BASE:-31337002}"
log() { echo "[yawab-L2] $* $(date -u +%FT%TZ)"; }

for ARM in A B; do
	EXTRA=""
	[ "$ARM" = "B" ] && EXTRA="--obs-yaw-err"
	marker="/tmp/wnn_yawab_L2_${ARM}_b${BASE}_done.json"
	if [ -f "$marker" ]; then
		log "arm $ARM: marker exists — skipping"
		continue
	fi
	out="logs/controller/yawab_L2_${ARM}_b${BASE}.out"
	winner="logs/controller/yawab_L2_${ARM}_b${BASE}_winner.yaml.gz"
	log "===== START arm $ARM (extra: '${EXTRA:-none}') -> $out ====="
	t0=$SECONDS
	# run_gran_phase3.sh recipe VERBATIM (BINARY, binqsr_v3/C10 chain) with the
	# single intended delta: --disturbance L2 (+ arm B's --obs-yaw-err).
	/usr/bin/time -l "$VP" -u -m wnn.control.phased_ga \
		--grid-state-neurons 8 12 16 --grid-bits 24 30 --levels 16 \
		--skip-stages bits,connections --lamarckian --saturation-grow-gain 1.0 \
		--neurons-gens 60 --neurons-patience 3 --memory-gens 120 --memory-patience 2 \
		--pop 50 --num-eval-folds 5 --check-interval 2 --magnitude-aware-patience \
		--eval-episodes 100 --memory-eval-episodes 200 --steps 2000 \
		--max-state-neurons 24 --max-output-neurons 128 --tilt 5.0 \
		--fit-weight-err-sq 0.4 --fit-weight-stable 0.3 --fit-weight-jerk 0.2 --fit-weight-mono 0.1 \
		--report-seed 99990101 --report-episodes 100 --holdout-pop-sample 8 \
		--base-seed "$BASE" --runs 1 --teacher lqr \
		--disturbance L2 $EXTRA \
		--memory-mode BINARY --save-winner "$winner" > "$out" 2>&1
	rc=$?
	dur=$((SECONDS - t0))
	rss=$(grep -E "maximum resident set size" "$out" | awk '{print $1}' | tail -1)
	if [ -f "$winner" ]; then
		"$VP" -u scripts/gran_fpga_count.py "$winner" >> "$out" 2>&1
	else
		log "arm $ARM: NO saved winner ($winner) — fpga count skipped"
	fi
	fpga=$(grep -E "^\[FPGA\]" "$out" | tail -1)
	held=$(grep -E "held-out" "$out" | tail -3 | tr '\n' ';')
	printf '{"arm":"%s","base":%s,"rc":%s,"dur_s":%s,"peak_rss_bytes":%s,"fpga":"%s","held":"%s","done":"%s"}\n' \
		"$ARM" "$BASE" "$rc" "$dur" "${rss:-null}" \
		"$(echo "$fpga" | tr -d '"' | sed 's/  */ /g')" \
		"$(echo "$held" | tr -d '"' | sed 's/  */ /g')" \
		"$(date -u +%FT%TZ)" > "$marker"
	log "===== END arm $ARM rc=$rc dur=${dur}s ====="
done
> "/tmp/wnn_yawab_L2_done.json"
log "YAW A/B DONE (base $BASE)"
