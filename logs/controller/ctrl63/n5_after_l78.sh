#!/bin/bash
# CTRL-77 n=5 promotion (Luiz 26/11/2026, prereg amendment): after the CTRL-78 leak block finishes, fly s4-s5 for every
# CTRL-77 arm that, over s1-s3, has stable 100.0 on every seed, fails total 0, mean err < 1.05, mean steady < 0.30 (bar lowered from 1.00 by Luiz 26/11 16:10 EST).
# The bar is re-applied to the runs.log END lines at launch time. Pure waits; 'N5Q:' lines. Fails closed.
cd /Users/lacg/wnn || exit 1
RUNS=logs/controller/ctrl48/runs.log
log() { echo "$(date -u +%FT%TZ) $*" >> $RUNS; }
L_DONE='^[0-9TZ:-]+ L78 DONE seeds \[1 2 3\]'
STOP='^[0-9TZ:-]+ (L78 ABORT|Y79 ABORT|F77 ABORT|Q79: .*standing down)'
grep -qE "$L_DONE" $RUNS && { log "N5Q: refused — CTRL-78 already done; launch the n=5 stage by hand"; exit 1; }
log "N5Q: armed (pid $$) — after CTRL-78: CTRL-77 n=5 (s4 s5) for arms meeting stable 100/fails 0/err<1.05/steady<0.30 on s1-s3"
while :; do
	grep -qE "$L_DONE" $RUNS && break
	if grep -qE "$STOP" $RUNS; then log "N5Q: an earlier block aborted or stood down — standing down, n=5 NOT started"; exit 1; fi
	sleep 60
done
ARMS=$(python3 - "$RUNS" <<'PY'
import re, sys
txt = open(sys.argv[1]).read().splitlines()
arms = "l1 l2a l2b l3 l1l2a l1l2b l1l3 l2al3 l2bl3 l1l2al3 l1l2bl3".split()
out = []
for a in arms:
	rows = []
	for s in (1, 2, 3):
		tag = f"C-S2-d_wposx2_a63_rf_c200_dc_biasin_ms01_P_both_L1_yld_gt_{a}_full_s{s} rc=0"
		ln = [l for l in txt if re.match(r"^[0-9TZ:-]+ END ", l) and tag in l]
		if not ln: break
		l = ln[-1]
		st = float(re.search(r"stable=([\d.]+)", l).group(1)); er = float(re.search(r"err=([\d.]+)", l).group(1))
		sd = float(re.search(r"steady=([\d.]+)", l).group(1)); fl = int(re.search(r"stable_fail=(\d+)/", l).group(1))
		rows.append((st, er, sd, fl))
	if len(rows) == 3 and all(r[0] == 100.0 for r in rows) and sum(r[3] for r in rows) == 0 \
			and sum(r[1] for r in rows) / 3 < 1.05 and sum(r[2] for r in rows) / 3 < 0.30:
		out.append(a)
print(" ".join(out))
PY
)
[ -n "$ARMS" ] || { log "N5Q: CTRL-78 done, but no CTRL-77 arm meets the n=5 bar — nothing launched"; exit 0; }
log "N5Q: CTRL-78 done — n=5 arms by the bar: [$ARMS]; starting ab_fact77.sh F77_SEEDS='4 5'"
F77_SEEDS="4 5" F77_ARMS="$ARMS" exec logs/controller/ctrl63/ab_fact77.sh
