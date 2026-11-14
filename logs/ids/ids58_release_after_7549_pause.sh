#!/bin/bash
# Luiz 14/11: run IDS58 (7567-7569) now. Release them the moment 7549's requested pause lands (worker free).
D="file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro"; API=https://127.0.0.1:3000/api/flows
LOG=/Users/lacg/wnn/logs/ids/ids_sequencer.log
until [ "$(sqlite3 "$D" "select status from flows where id=7549")" != "running" ]; do sleep 20; done
echo "$(date -u +%FT%TZ) [ids58] 7549 now $(sqlite3 "$D" "select status from flows where id=7549"); releasing 7567-7569" >> $LOG
for id in 7567 7568 7569; do c=$(curl -sk -o /dev/null -w "%{http_code}" -X POST $API/$id/resume); echo "$(date -u +%FT%TZ) [ids58] resume $id http $c" >> $LOG; done
