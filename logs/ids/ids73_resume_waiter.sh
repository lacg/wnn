#!/bin/bash
# IDS-73 order waiter (Luiz 21/11 ~16:40 EST: "queue E [on unswt] first"). 7593-7599 (IDS-70 E s3 + B, IDS-71) were
# POST /pause'd so the unswt flows (id > 7599) run first. When NO flow with id > 7599 is still 'queued' (all have
# been admitted), resume 7593-7599 with POST /resume in id order (never PATCH status — leaves pause_requested=1).
DB="file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro"
LOG=/Users/lacg/wnn/logs/ids/ids73_resume_waiter.log
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
say "armed (pid $$): waits until no flow id>7599 is queued, then resumes 7593-7599"
while :; do
	n=$(sqlite3 "$DB" "select count(*) from flows where id>7599 and status='queued';")
	[ "$n" = 0 ] && break
	sleep 60
done
sleep 30
for f in 7593 7594 7595 7596 7597 7598 7599; do
	r=$(curl -sk -X POST "https://localhost:3000/api/flows/$f/resume")
	say "resume $f -> $r"
done
say "done: $(sqlite3 "$DB" "select group_concat(id||':'||status||':'||pause_requested) from flows where id between 7593 and 7599;")"
