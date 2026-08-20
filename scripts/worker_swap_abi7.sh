#!/usr/bin/env bash
# ABI-7 worker swap at the next flow boundary (19/08/2026, Luiz: "it needs to
# happen soon, not when idle in a month" — 357 queued SP100 flows are compute
# on the LOSING fitness if the IDS Z_RANK ablation says zscore wins, so the
# answer is wanted at run ~20, not ~2660).
#
# Why not worker_swap.py alone: the ABI-7 wheel ships WITH a facade change
# (wnn/accel.py EXPECTED_ABI 6→7), and the flip must land BETWEEN the wheel
# install and the worker relaunch. Earlier is measured-fatal: the CONTROLLER
# evaluator imports wnn.accel too, and the 19/08 zscore smoke died at the
# assert when the facade moved ahead of the install. Later bricks the worker:
# it would import the ABI-7 wheel through a facade still expecting 6. And
# worker_swap.py --no-restart skips its own requeue step, so the interrupted
# flow is re-queued here, from the stage-1 marker, via the API (Rule 2 —
# never flip flow rows by hand).
#
# Sequence: boundary stop + install (worker_swap.py --no-restart) → facade
# flip → relaunch (detached, NO --rayon-threads: the budget must read 13
# cores; 7 means nothing admits) → requeue interrupted → verify ABI + budget.
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
WHL="/Volumes/20260401-WDBlack-SN850X-2TB/cargo-target/wheels/ram_accelerator-2026.212.37-cp313-cp313-macosx_11_0_arm64.whl"
LOG="/private/tmp/worker_swap_abi7.log"
M1="/tmp/wnn_swap_abi7_stage1.json"
DONE="/tmp/wnn_swap_abi7_done.json"

log() { echo "[swap-abi7] $(date -u +%FT%TZ) $*" >> "$LOG"; }

log "armed: waiting for the running flow to finish, wheel=$WHL"

# 1. Boundary stop + wheel install, NO relaunch, NO requeue (both happen here).
"$VP" scripts/worker_swap.py --auto-detect-running \
	--install-wheel "$WHL" --no-restart --marker "$M1" >> "$LOG" 2>&1
if [ ! -f "$M1" ]; then
	log "FATAL: worker_swap stage 1 wrote no marker — worker state unknown, STOPPING. Investigate before relaunch."
	exit 1
fi

# 2. Facade flip 6→7, exactly between install and relaunch (see header).
"$VP" - <<'PY' >> "$LOG" 2>&1
import re
path = "src/wnn/accel.py"
src = open(path).read()
new, n = re.subn(r"^EXPECTED_ABI = 6$", "EXPECTED_ABI = 7", src, flags=re.M)
if n != 1:
	raise SystemExit(f"[swap-abi7] expected exactly one 'EXPECTED_ABI = 6' line, found {n} — NOT flipping")
open(path, "w").write(new)
print("[swap-abi7] facade flipped: EXPECTED_ABI = 7")
PY
if ! grep -q "^EXPECTED_ABI = 7$" src/wnn/accel.py; then
	log "FATAL: facade flip failed — worker left DOWN on purpose (relaunching on a mismatched pair is the 12/08 failure)."
	exit 1
fi

# 3. Relaunch detached (PPID=1). No --rayon-threads, per the 13-core budget rule.
"$VP" scripts/detach_launch.py -- "$VP" -u -B -m wnn.ram.experiments.worker \
	--url https://localhost:3000 --no-ssl-verify >> "$LOG" 2>&1
sleep 20

# 4. Requeue whatever the stop interrupted (stage-1 marker), via the API.
"$VP" - <<'PY' >> "$LOG" 2>&1
import json, ssl, urllib.request
m = json.load(open("/tmp/wnn_swap_abi7_stage1.json"))
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
for fid in m.get("interrupted", []):
	req = urllib.request.Request(f"https://localhost:3000/api/flows/{fid}/restart",
		data=json.dumps({"from_beginning": False}).encode(),
		headers={"Content-Type": "application/json"}, method="POST")
	try:
		urllib.request.urlopen(req, context=ctx, timeout=30)
		print(f"[swap-abi7] requeued flow {fid}")
	except Exception as e:
		print(f"[swap-abi7] ⚠️ requeue of flow {fid} FAILED: {e} — requeue by hand")
PY

# 5. Verify: ABI 7 importable through the facade + the 13-core budget line.
sleep 40
ABI=$("$VP" -c "import wnn.accel as a; print(a.require_accel().ABI_VERSION)" 2>&1 | tail -1)
BUDGET=$(grep "CPU budget" /tmp/wnn_worker.log | tail -1)
WPID=$(ps -axo pid,command | grep "wnn.ram.experiments.worker" | grep -v grep | awk '{print $1}' | head -1)
log "verify: ABI=$ABI budget='$BUDGET' worker_pid=${WPID:-NONE}"
printf '{"abi":"%s","worker_pid":"%s","budget":"%s","done":"%s"}\n' \
	"$ABI" "${WPID:-NONE}" "$(echo "$BUDGET" | tr -d '"')" "$(date -u +%FT%TZ)" > "$DONE"
if [ "$ABI" != "7" ] || [ -z "${WPID:-}" ]; then
	log "⚠️ VERIFY FAILED (abi=$ABI pid=${WPID:-NONE}) — escalate in the morning tick."
	exit 1
fi
log "########## SWAP COMPLETE — ABI 7, worker $WPID ##########"
