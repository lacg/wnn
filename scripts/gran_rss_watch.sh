#!/bin/bash
# Lightweight RSS + memory sampler for the 5-arm gran ablation (14/07/2026 PM).
# Logs one CSV-ish line every 60s: the phased_ga RSS, macOS real-free RAM, the
# current arm (from the newest c10_gran_*_20260714b run.out), and IDS-worker aliveness.
# Detached (PPID=1) alongside run_gran_5arm_capped.sh. Pure observability — the HARD
# kill is controller_mem_watchdog.sh (floor 5GB). Ends when the 5-arm marker appears.
set -u
OUT="/private/tmp/wnn_gran_rss_watch.log"
echo "ts_utc,phased_pid,phased_rss_gb,real_free_gb,phys_used,cur_arm,ids_worker" >> "$OUT"
while [ ! -f /tmp/wnn_gran_5arm_done.json ]; do
	ts=$(date -u +%FT%TZ)
	pid=$(pgrep -f "wnn.control.phased_ga" | head -1)
	rss=$(ps -o rss= -p "${pid:-0}" 2>/dev/null | awk '{printf "%.2f",$1/1048576}')
	free=$(vm_stat 2>/dev/null | awk '/Pages free/{printf "%.1f",$3*16384/1073741824}')
	used=$(top -l 1 -n 0 2>/dev/null | awk '/PhysMem/{print $2}')
	arm=$(ls -dt /Users/lacg/wnn/logs/controller/c10_gran_*_2026071* 2>/dev/null | head -1 | sed -E 's#.*/c10_gran_([a-z]+)_2026.*#\1#')
	pgrep -f wnn.ram.experiments.worker >/dev/null && ids=up || ids=DOWN
	echo "${ts},${pid:-none},${rss:-0},${free:-?},${used:-?},${arm:-?},${ids}" >> "$OUT"
	sleep 60
done
echo "$(date -u +%FT%TZ),DONE,-,-,-,-,-" >> "$OUT"
