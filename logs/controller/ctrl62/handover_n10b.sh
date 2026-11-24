#!/bin/bash
# Handover ab_gt7172_n10.sh (PID OLD) -> ab_gt7172_n10b.sh at the next run boundary (Luiz 24/11: N0 s9 under the
# plain s9 tag). Never interrupts a flight. Mechanism: a BLOCKER process whose argv carries the literal the old
# driver's ctrl_busy pgrep matches, so after the current flight ENDs (END line + .done written as usual) the old
# driver parks in its 'while ctrl_busy; do sleep 30; done' loop instead of launching. We then kill the parked
# driver, drop the blocker and start the new driver detached. Log lines: 'GT10 HANDOVER ...' in ctrl48/runs.log.
cd /Users/lacg/wnn || exit 1
L=logs/controller/ctrl48
RUNS=$L/runs.log
OLD=${1:?old driver pid}
log() { echo "$(date -u +%FT%TZ) $*" >> $RUNS; }
real_ctrl() { ps -axo pid=,command= | awk '$2 !~ /\/usr\/bin\/time/ && tolower($2) ~ /python/ && /[w]nn\.control\.phased_ga/ {print $1}'; }
[ "$(cat /tmp/ab_gt7172_n10.pid 2>/dev/null)" = "$OLD" ] && kill -0 $OLD 2>/dev/null \
	|| { log "GT10 HANDOVER REFUSED: old driver $OLD is not the running ab_gt7172_n10.sh"; exit 1; }
(exec -a "HANDOVER-BLOCKER MacOS/Python -u -m wnn.control.phased_ga" sleep 14400) &
B=$!
log "GT10 HANDOVER armed (pid $$, blocker $B): ab_gt7172_n10.sh ($OLD) -> ab_gt7172_n10b.sh at the next run boundary (N0 s9 under the plain s9 tag)"
while :; do
	if ! kill -0 $OLD 2>/dev/null; then
		kill $B 2>/dev/null; log "GT10 HANDOVER FAILED: old driver $OLD exited on its own (see its last line) — new driver NOT started"; exit 1
	fi
	pgrep -P $OLD -x sleep >/dev/null && break
	sleep 2
done
last=$(grep -aE " (START|END) C-S2-d" $RUNS | tail -1)
case "$last" in
	*" END "*) ;;
	*) kill $B 2>/dev/null; log "GT10 HANDOVER FAILED: last line is not an END ($last) — old driver left running"; exit 1;;
esac
tag=$(echo "$last" | awk '{print $3}')
if [ ! -f $L/$tag.done ]; then
	kill $B 2>/dev/null; log "GT10 HANDOVER DEFERRED: $tag ended without .done — old driver left running (escalate)"; exit 1
fi
[ -z "$(real_ctrl)" ] || { kill $B 2>/dev/null; log "GT10 HANDOVER FAILED: a real phased_ga is running — old driver left running"; exit 1; }
pkill -9 -P $OLD sleep 2>/dev/null; kill -9 $OLD
sleep 1
kill -0 $OLD 2>/dev/null && { kill $B; log "GT10 HANDOVER FAILED: could not stop $OLD"; exit 1; }
rm -f /tmp/ab_gt7172_n10.pid
kill $B 2>/dev/null
log "GT10 HANDOVER: old driver $OLD stopped parked after END $tag (no flight interrupted); starting ab_gt7172_n10b.sh"
nohup bash logs/controller/ctrl62/ab_gt7172_n10b.sh > logs/controller/ctrl62/ab_gt7172_n10b.nohup 2>&1 &
NEW=$!
sleep 90
if kill -0 $NEW 2>/dev/null; then
	log "GT10 HANDOVER DONE: ab_gt7172_n10b.sh pid $NEW running"
else
	log "GT10 HANDOVER FAILED: ab_gt7172_n10b.sh exited early (see logs/controller/ctrl62/ab_gt7172_n10b.nohup) — escalate"
fi
