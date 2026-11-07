#!/bin/bash
# Luiz 07/11: IDS-49 (QUAD crossover, 12 flows 7368-7379) runs BEFORE IDS-48. When every IDS49 flow is terminal,
# re-queue the 18 held IDS48 flows (logs/ids/ids48_held_for_ids49.txt). Read-only DB; PATCH via the dashboard API.
D="file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro"
cd /Users/lacg/wnn
echo "$(date -u +%FT%TZ) armed; waiting for all IDS49 terminal"
until [ "$(sqlite3 "$D" "select count(*) from flows where name like 'IDS49-%' and status in ('queued','running','pending','paused')")" = 0 ]; do sleep 120; done
for id in $(cat logs/ids/ids48_held_for_ids49.txt); do
  for t in 1 2 3 4 5 6; do c=$(curl -sk -o /dev/null -w "%{http_code}" -X PATCH https://localhost:3000/api/flows/$id -H "Content-Type: application/json" -d '{"status":"queued"}'); [ "$c" = 200 ] && break; sleep 5; done
  echo "$(date -u +%FT%TZ) requeue $id -> $c"
done
echo "$(date -u +%FT%TZ) DONE"
