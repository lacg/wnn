#!/bin/bash
# IDS-81 (Luiz 27/11/2026): restart paused flows 6433 6438 7535 7549 FROM ZERO on the current worker ABI, AFTER the
# queued 7638-7641 have all started (the worker starts the lowest queued id first, so restarting earlier would jump
# them ahead — never reorder IDS). Fix #1 (checkpoint ABI guard, merged) is live. Before restarting, the stale
# checkpoint folders are archived (the dashboard's restart misses them: it maps '.' to '_' in folder names).
# Pure waits; log lines go to logs/ids/ids81_restart.log. Fails closed.
cd /Users/lacg/wnn || exit 1
LOG=logs/ids/ids81_restart.log
DB="file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro"
API=https://localhost:3000/api/flows
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
st() { sqlite3 "$DB" "select status from flows where id=$1;"; }
log "armed (pid $$): waiting for flow 7641 to start"
while :; do
	s=$(st 7641)
	case "$s" in running|completed) break;; queued) ;; *) log "7641 is '$s' — standing down, nothing restarted"; exit 1;; esac
	sleep 300
done
q=$(sqlite3 "$DB" "select count(*) from flows where status='queued';")
[ "$q" = 0 ] || { log "7641 started but $q other flows are queued — standing down (would reorder IDS)"; exit 1; }
for f in 6433 6438 7535 7549; do [ "$(st $f)" = paused ] || { log "flow $f is '$(st $f)', not paused — standing down"; exit 1; }; done
A=checkpoints/_archive_ids81_$(date -u +%Y%m%d)
mkdir -p $A
for d in ids56c-cicids-ce20-zs-binary-2m-max250-xo_blend_neuron-rate0.25-lattice-encon-r20421 \
		ids56uc-cicids-ce20-zs-binary-2m-max250-xo_blend_neuron-rate0.25-nb02-encon-r20419 \
		ids6-ciciot46m-b15ac-desir-r20419 ids6-ciciot46m-b15ac-desir-r20420; do
	[ -d checkpoints/$d ] && mv checkpoints/$d $A/ && log "archived checkpoints/$d -> $A/"
done
for f in 6433 6438 7535 7549; do
	r1=$(curl -sk -X POST $API/$f/restart -H 'Content-Type: application/json' -d '{"from_beginning":true}' -w ' http=%{http_code}')
	r2=$(curl -sk -X PATCH $API/$f -H 'Content-Type: application/json' -d '{"status":"queued"}' -w ' http=%{http_code}')
	log "flow $f restart: ${r1: -8} | queued: ${r2: -8} | status now $(st $f)"
done
log "done: 6433 6438 7535 7549 restarted from zero, queued behind 7641"
