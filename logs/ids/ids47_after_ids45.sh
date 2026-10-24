#!/bin/bash
# Luiz 24/10/2026 ~16:00 EDT: "choose for me and in the morning I will validate". REPLACES
# unpause_ids47_quad_after_ids45.sh (which released only the 25 QUAD flows and left the 90 BINARY for him).
# When every IDS-45 flow is terminal: run the PRE-REGISTERED rule (ids45_decide.py), then
#   max250 -> un-pause ALL paused IDS-47 flows (QUAD + BINARY, original seed-major interleave);
#   tot250/tot500 -> cancel every PAUSED IDS-47 flow and re-clone all 115 at the chosen cap, same interleave.
# PATCH/POST via the dashboard API only. Never touches a completed/running flow.
cd /Users/lacg/wnn || exit 1
PY=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python
DB="file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro"
echo "$(date -u +%FT%TZ) armed; waiting for all IDS45-% flows terminal"
while :; do
	left=$(sqlite3 "$DB" "select count(*) from flows where name like 'IDS45-%' and status in ('queued','running','pending','paused')")
	[ "$left" = "0" ] && break
	sleep 300
done
echo "$(date -u +%FT%TZ) IDS-45 done; running the pre-registered cap rule"
$PY logs/ids/ids45_decide.py
cap=$($PY -c "import json;print(json.load(open('logs/ids/ids45_decision.json'))['cap'])" 2>/dev/null)
[ -z "$cap" ] && cap=max250 && echo "decision file unreadable -> FAIL-SAFE max250"
echo "$(date -u +%FT%TZ) cap=$cap"
if [ "$cap" = "max250" ]; then
	ok=0; bad=0
	for id in $(sqlite3 "$DB" "select id from flows where name like 'IDS47-%' and status='paused' order by id"); do
		code=000
		for t in 1 2 3 4 5; do
			code=$(curl -sk -o /dev/null -w "%{http_code}" -X PATCH -H "Content-Type: application/json" -d '{"status":"queued"}' "https://127.0.0.1:3000/api/flows/$id")
			[ "$code" = "200" ] && break; sleep 2
		done
		if [ "$code" = "200" ]; then ok=$((ok+1)); else bad=$((bad+1)); echo "FAIL $id http=$code"; fi
		sleep 0.2
	done
	echo "$(date -u +%FT%TZ) DONE max250: un-paused=$ok failed=$bad"
else
	$PY logs/ids/queue_ids47_xds.py --cap "$cap" --cancel-paused
	echo "$(date -u +%FT%TZ) DONE $cap re-clone rc=$?"
fi
