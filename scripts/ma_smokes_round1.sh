#!/usr/bin/env bash
# CTRL-5 round-1 pop-6 PLUMBING smokes (Luiz 05/11/2026: go after the CTRL-8 verdict; option A = no T=0.30 lag;
# sn=8 not smoked — flies only if sn=4 shows an effect). Commands = `multi_axis_chain.sh --smoke-list` minus
# axF_T030/axC_sn8, frozen in logs/controller/ma_smoke/smoke_cmds.txt. Numbers are NOT read (pop 6); checks are
# rc=0, [provenance] abi=33, a HEADLINE line, and peak RSS from /usr/bin/time -l.
set -u
cd "$(dirname "$0")/.." || exit 1
OUT=logs/controller/ma_smoke; LOG=$OUT/smokes.log
log() { echo "[masmoke] $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
log "armed — waiting for controller idle"
while pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null; do sleep 60; done
while IFS= read -r cmd; do
	[ -f experiments/HOLD_CONTROLLER ] && { log "HOLD — stopping"; exit 0; }
	tag=$(echo "$cmd" | grep -oE "ma_smoke/[A-Za-z0-9_]+" | head -1 | cut -d/ -f2)
	o=/private/tmp/ma_smoke_$tag.out
	log "START $tag"
	bash -c "$cmd"; rc=$?
	cp "$o" "$OUT/$tag.out" 2>/dev/null
	abi=$(grep -aoE "abi=[0-9]+" "$o" | head -1); head=$(grep -ac "stage-select\] HEADLINE held-out" "$o")
	rss=$(grep -aE "maximum resident set size" "$o" | awk '{printf "%.1f GiB", $1/1073741824}')
	cells=$(grep -aoE "cells\[[^]]*\]" "$o" | tail -1)
	log "END $tag rc=$rc $abi headline=$head peakRSS=$rss $cells"
done < $OUT/smoke_cmds.txt
log "ROUND-1 SMOKES COMPLETE"
