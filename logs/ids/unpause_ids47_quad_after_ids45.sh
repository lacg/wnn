#!/bin/bash
# Luiz 24/10/2026: run the IDS-45 budget arms (tot500 7208-7217 + tot250 7333-7342) BEFORE IDS-47, then decide
# IDS-47's BINARY cap. IDS-47 was paused whole so the lower-id flows could not jump the FIFO. When every IDS-45
# flow is terminal, un-pause ONLY the 25 IDS-47 QUAD 1m flows (independent of the cap) so the worker keeps going.
# The 90 IDS-47 BINARY flows stay PAUSED for Luiz's cap decision. PATCH via the dashboard API only.
DB="file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro"
echo "$(date -u +%FT%TZ) armed; waiting for all IDS45-% flows terminal"
while :; do
	left=$(sqlite3 "$DB" "select count(*) from flows where name like 'IDS45-%' and status in ('queued','running','pending','paused')")
	[ "$left" = "0" ] && break
	sleep 300
done
echo "$(date -u +%FT%TZ) IDS-45 done; un-pausing the IDS-47 QUAD 1m flows"
ok=0; bad=0
for id in $(sqlite3 "$DB" "select id from flows where name like 'IDS47-%-QUAD-1m-%' and status='paused' order by id"); do
	code=000
	for t in 1 2 3 4 5; do
		code=$(curl -sk -o /dev/null -w "%{http_code}" -X PATCH -H "Content-Type: application/json" -d '{"status":"queued"}' "https://127.0.0.1:3000/api/flows/$id")
		[ "$code" = "200" ] && break; sleep 2
	done
	if [ "$code" = "200" ]; then ok=$((ok+1)); else bad=$((bad+1)); echo "FAIL $id http=$code"; fi
	sleep 0.2
done
echo "$(date -u +%FT%TZ) DONE un-paused=$ok failed=$bad (90 BINARY stay paused for Luiz)"
