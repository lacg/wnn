#!/usr/bin/env python3
"""Quick one-screen status check for an OI-v2 cohort: completed/queued counts + ETA.

Usage:
  python3 cohort_status.py                  # auto-detect cohort
  python3 cohort_status.py --cohort PREFIX
  python3 cohort_status.py --list
"""
import argparse
import json
import sqlite3
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB = Path("/Users/lacg/wnn/db/wnn.db")
HSR_VALUES = [1, 2, 3, 5, 7, 8, 10]


def discover_cohorts(cur):
    cur.execute(
        """SELECT
            SUBSTR(name, 1, INSTR(name, '-FIXED-OLD-')-1) AS prefix,
            COUNT(*) AS cnt
        FROM flows WHERE name LIKE '%-FIXED-OLD-%' AND status='completed'
        GROUP BY prefix ORDER BY cnt DESC"""
    )
    return [(r[0], r[1]) for r in cur.fetchall() if r[0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", type=str, default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--target", type=int, default=112, help="Expected NEW cohort size")
    ap.add_argument("--db", type=str, default=str(DB))
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    available = discover_cohorts(cur)
    if args.list:
        for prefix, cnt in available:
            print(f"  {prefix:<40}  OLD={cnt}")
        sys.exit(0)

    if args.cohort:
        prefix = args.cohort
    elif len(available) == 1:
        prefix = available[0][0]
    elif available:
        print("Multiple cohorts; specify --cohort PREFIX:")
        for p, c in available:
            print(f"  {p:<40}  OLD={c}")
        sys.exit(1)
    else:
        print("No cohorts found.", file=sys.stderr)
        sys.exit(2)

    # NEW cohort status (anything with -OI- excluding -OI-OLD-)
    cur.execute(
        """SELECT status, COUNT(*) FROM flows
        WHERE name LIKE ? AND name NOT LIKE '%OLD%'
        GROUP BY status""",
        (f"{prefix}-OI%-r%",),
    )
    by_status = {r[0]: r[1] for r in cur.fetchall()}
    total = sum(by_status.values())
    completed = by_status.get("completed", 0)
    running = by_status.get("running", 0)
    queued = by_status.get("queued", 0)
    cancelled = by_status.get("cancelled", 0)
    failed = by_status.get("failed", 0)
    pending = by_status.get("pending", 0)

    # HSR-tagged subset
    cur.execute(
        """SELECT
            json_extract(config_json, '$.params.wnn_hybrid_speed_ratio') AS hsr,
            status, COUNT(*)
        FROM flows WHERE name LIKE ? AND name NOT LIKE '%OLD%'
        GROUP BY hsr, status""",
        (f"{prefix}-OI%-r%",),
    )
    hsr_table = {}
    for r in cur.fetchall():
        h = r[0]
        if h is None: continue
        try: h = int(h)
        except Exception: pass
        hsr_table.setdefault(h, {"completed":0, "running":0, "queued":0})
        s = r[1]
        if s in ("completed","running","queued"):
            hsr_table[h][s] = r[2]

    # Durations for ETA
    cur.execute(
        """SELECT (julianday(completed_at)-julianday(started_at))*1440 AS m, completed_at
        FROM flows WHERE name LIKE ? AND name NOT LIKE '%OLD%' AND status='completed'
        ORDER BY completed_at""",
        (f"{prefix}-OI%-r%",),
    )
    rows = list(cur)
    durs = [r[0] for r in rows if r[0] is not None]
    avg = statistics.mean(durs) if durs else 0
    latest_done = rows[-1][1] if rows else None
    eta_str = "n/a"
    if latest_done and avg and completed < args.target:
        latest_dt = datetime.fromisoformat(latest_done.replace("Z","+00:00"))
        eta_dt = latest_dt + timedelta(minutes=(args.target - completed) * avg)
        eta_str = f"{eta_dt.strftime('%d/%m/%Y %H:%M UTC')}  |  {(eta_dt - timedelta(hours=4)).strftime('%d/%m/%Y %H:%M ET')}"

    now = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")
    print(f"Cohort: {prefix}  ({now})")
    print()
    print(f"  completed : {completed}/{args.target}")
    print(f"  running   : {running}")
    print(f"  queued    : {queued}")
    if pending:   print(f"  pending   : {pending}")
    if cancelled: print(f"  cancelled : {cancelled}")
    if failed:    print(f"  failed    : {failed}")
    print(f"  avg/run   : {avg:.0f} min")
    print(f"  ETA       : {eta_str}")
    print()
    if hsr_table:
        print(f"  HSR breakdown (completed / total assigned):")
        for h in HSR_VALUES:
            entry = hsr_table.get(h, {})
            comp = entry.get("completed", 0)
            run = entry.get("running", 0)
            qd = entry.get("queued", 0)
            assigned = comp + run + qd
            star = " <- running" if run else ""
            print(f"    HSR={h:>2}: completed={comp}/{assigned}{star}")


if __name__ == "__main__":
    main()
