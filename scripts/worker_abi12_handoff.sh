#!/usr/bin/env bash
# WORKER ABI-12 HANDOFF (29/08/2026, Luiz: "let's do it asap").
#
# Deploys the address-hash wheel (ram_accelerator ABI 12; see
# .claude/plans/address_hash_above_64.md) to the IDS worker at its natural idle,
# then smokes ONE affected flow before releasing the paused 96b cohort. Every
# step fails closed. Sequence:
#
#   1. scripts/worker_swap.py (already armed, --no-restart) waits for the running
#      flow to end, stops the worker cleanly, re-queues anything it interrupted,
#      and writes $SWAP_MARKER. Nothing here starts before that marker exists.
#   2. pip install the ABI-12 wheel AND bump wnn/accel.py EXPECTED_ABI 11 -> 12
#      in the same breath (the facade asserts EQUALITY; the two must land together).
#   3. verify: `import wnn.accel` sees ABI 12. If not -> STOP, worker stays down.
#   4. relaunch the worker exactly as scripts/worker_swap.py::relaunch_worker does
#      (same venv, PYTHONPATH, RAYON_NUM_THREADS, --url/--no-ssl-verify).
#   5. SMOKE: queue ONE paused 96b flow ($SMOKE_FLOW). Wait for it to reach a
#      terminal state. completed -> release the rest; anything else -> STOP loud.
#   6. release: PATCH the remaining paused 96b flows -> queued, and restart the two
#      pending -w64fix reruns (POST /api/flows/:id/restart), all via the API.
set -u
ROOT="/Users/lacg/wnn"; cd "$ROOT" || exit 1
LOG="/private/tmp/worker_abi12_handoff.log"
SWAP_MARKER="/private/tmp/worker_swap_abi12.json"
WHEEL="/Volumes/20260401-WDBlack-SN850X-2TB/cargo-target/wheels/ram_accelerator-2026.212.37-cp313-cp313-macosx_11_0_arm64.whl"
VENV="$ROOT/wnn"
DB="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db"
API="https://127.0.0.1:3000"
WORKER_LOG="/tmp/wnn_worker.log"
RAYON=13
SMOKE_FLOW=5895
REST_FLOWS="5896 5897 5898 5899 5900 5901 5902 5903 5904 5905 5906 5907 5908 5909"
RERUN_FLOWS="6050 6051"

log() { echo "[abi12] $(date -u +%FT%TZ) $*" >> "$LOG"; }
status_of() { sqlite3 "file:${DB}?mode=ro" "select status from flows where id=$1;"; }
patch_status() {   # id status  — retry: the dashboard drops connections under load
	local id="$1" st="$2" code=""
	for _ in 1 2 3 4 5 6 7 8 9 10; do
		code=$(curl -sk -o /dev/null -w "%{http_code}" -X PATCH -H "Content-Type: application/json" \
			-d "{\"status\":\"$st\"}" "$API/api/flows/$id")
		[ "$code" = "200" ] && { log "flow $id -> $st"; return 0; }
		sleep 3
	done
	log "flow $id -> $st FAILED (last http $code)"; return 1
}

log "########## ARMED — waiting for $SWAP_MARKER (worker_swap.py stops the worker at flow end) ##########"
while [ ! -f "$SWAP_MARKER" ]; do sleep 15; done
log "swap marker present: $(tr -d '\n' < "$SWAP_MARKER" | cut -c1-300)"
# belt and braces: the worker must really be gone
for _ in $(seq 1 40); do
	pgrep -f "wnn.ram.experiments.worker" >/dev/null || break; sleep 3
done
if pgrep -f "wnn.ram.experiments.worker" >/dev/null; then
	log "worker STILL RUNNING after the swap marker — refusing to install over a live worker. STOP."; exit 1
fi

# ---- 2. wheel + facade, together
log "installing $WHEEL"
( unset CONDA_PREFIX; source "$VENV/bin/activate"; pip install --force-reinstall --no-deps "$WHEEL" ) >> "$LOG" 2>&1 \
	|| { log "pip install FAILED — worker stays DOWN. STOP."; exit 1; }
sed -i '' 's/^EXPECTED_ABI = 11$/EXPECTED_ABI = 12/' src/wnn/accel.py
grep -q '^EXPECTED_ABI = 12$' src/wnn/accel.py || { log "accel.py EXPECTED_ABI bump did not apply — STOP."; exit 1; }
log "accel.py EXPECTED_ABI -> 12"

# ---- 3. verify
ABI=$( unset CONDA_PREFIX; source "$VENV/bin/activate"; export PYTHONPATH="$ROOT/src/wnn:${PYTHONPATH:-}"; \
	python -c "import wnn.accel as a; print(a.require_accel().ABI_VERSION)" 2>>"$LOG" )
if [ "$ABI" != "12" ]; then log "VERIFY FAILED: wnn.accel reports '$ABI', want 12 — worker stays DOWN. STOP."; exit 1; fi
log "verify OK: ram_accelerator ABI 12 through wnn.accel"

# ---- 4. relaunch worker (mirror of worker_swap.py::relaunch_worker)
zsh -c "cd $ROOT && unset CONDA_PREFIX && source $VENV/bin/activate && export PYTHONPATH=\"\$(pwd)/src/wnn:\$PYTHONPATH\" && export RAYON_NUM_THREADS=$RAYON && nohup python -u -B -m wnn.ram.experiments.worker --url https://localhost:3000 --no-ssl-verify </dev/null >>$WORKER_LOG 2>&1 &" 
sleep 5
WPID=$(pgrep -f "wnn.ram.experiments.worker" | head -1)
[ -n "$WPID" ] || { log "worker did not come up — STOP."; exit 1; }
log "worker relaunched pid=$WPID rayon=$RAYON"

# ---- 4b. worker_swap.py --no-restart does NOT requeue flows the stop interrupted
# (it only does so on its own relaunch). Any id in the marker's "interrupted" list
# is a flow the OLD worker had admitted after $WATCHED ended and was killed mid-gen:
# restart it from its checkpoint via the API (never by flipping rows).
for fid in $(python3 -c "import json;print(' '.join(str(i) for i in json.load(open('$SWAP_MARKER')).get('interrupted',[])))"); do
	code=$(curl -sk -o /dev/null -w "%{http_code}" -X POST -H "Content-Type: application/json" -d '{"from_beginning":false}' "$API/api/flows/$fid/restart")
	log "interrupted flow $fid requeued from checkpoint -> http $code (status now: $(status_of "$fid"))"
done

# ---- 5. smoke ONE
patch_status "$SMOKE_FLOW" queued || exit 1
log "SMOKE: flow $SMOKE_FLOW queued (FIFO min-id; it runs when the worker reaches it). Waiting for a terminal state."
while :; do
	st=$(status_of "$SMOKE_FLOW")
	case "$st" in
		completed) log "SMOKE flow $SMOKE_FLOW COMPLETED."; break ;;
		failed|cancelled) log "SMOKE flow $SMOKE_FLOW ended '$st' — NOT releasing the cohort. Inspect $WORKER_LOG. STOP."; exit 1 ;;
	esac
	pgrep -f "wnn.ram.experiments.worker" >/dev/null || { log "worker died during the smoke — NOT releasing. STOP."; exit 1; }
	sleep 60
done
# what did the smoke's winner look like (bits)? informational
sqlite3 "file:${DB}?mode=ro" "select f.name, max(j.value) from genomes g join best_genomes bg on bg.genome_id=g.id join flows f on f.id=bg.flow_id, json_each(json_extract(g.tiers_json,'\$.bits_per_neuron')) j where json_valid(g.tiers_json) and f.id=$SMOKE_FLOW;" >> "$LOG" 2>&1

# ---- 6. release
for id in $REST_FLOWS; do patch_status "$id" queued; sleep 1; done
for id in $RERUN_FLOWS; do
	code=$(curl -sk -o /dev/null -w "%{http_code}" -X POST -H "Content-Type: application/json" -d '{"from_beginning":true}' "$API/api/flows/$id/restart")
	log "rerun flow $id restart -> http $code (status now: $(status_of "$id"))"
	sleep 1
done
log "########## HANDOFF COMPLETE — 96b cohort released on ABI 12 ##########"
