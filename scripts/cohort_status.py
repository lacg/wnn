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


# Inline copy of predict_hsr / estimate_samples_per_fold from
# src/wnn/ram/experiments/hsr_function.py — duplicated so this script
# runs under system python without the wnn package being importable.
# Keep in sync when the function thresholds change.
_DATASET_TRAIN_SIZE = {
    "ciciot2023_neto_subsample": 914_000,
    "ciciot2023_neto_full":      37_300_000,
    "cicids2017":                2_300_000,
    "unsw-nb15":                 1_270_000,
}
_DATASET_TRAIN_SIZE_BY_SPLIT = {
    ("unsw-nb15", "temporal"): 175_000,
}


def _estimate_samples_per_fold(params):
    dataset = params.get("ids_dataset", "ciciot2023_neto_subsample")
    split = params.get("ids_split", "random")
    k_folds = params.get("ids_k_folds", 5)
    train_size = _DATASET_TRAIN_SIZE_BY_SPLIT.get((dataset, split))
    if train_size is None:
        train_size = _DATASET_TRAIN_SIZE.get(dataset, 1_000_000)
    return max(1, train_size // k_folds)


def _predict_hsr(params):
    neurons = params.get("max_neurons", 100)
    bits = params.get("max_bits", 32)
    thermo = params.get("ids_n_bits", 96)
    samples = _estimate_samples_per_fold(params)
    workload = samples * thermo * 20 + samples * neurons * bits
    if workload < 10_000_000:
        return 1
    if workload >= 5_000_000_000:
        return 10
    return 8


def discover_cohorts(cur):
    cur.execute(
        """SELECT
            SUBSTR(name, 1, INSTR(name, '-FIXED-OLD-')-1) AS prefix,
            COUNT(*) AS cnt
        FROM flows WHERE name LIKE '%-FIXED-OLD-%' AND status='completed'
        GROUP BY prefix ORDER BY cnt DESC"""
    )
    return [(r[0], r[1]) for r in cur.fetchall() if r[0]]


# XDS cohort aliases (added 30/05/2026 for cross-dataset OI cohorts).
# Each maps a friendly --cohort name to (display_label, LIKE_pattern, default_target).
# These don't have an OLD/NEW split (XDS uses a different rename
# convention, -PREEMP-OLD-, not -FIXED-OLD-) so they bypass discover_cohorts().
XDS_ALIASES = {
    # NOTE 31/05/2026: switched architecture suffix from literal `500n34b` to
    # `%n%b` so cohorts can be re-sized without breaking this matcher (e.g.
    # UNSW-random changed to 50n34b on 31/05 because 250n×100b was 10× too slow).
    "xds-temporal":  ("XDS-unsw-temporal", "XDS-unsw-temporal-%b-W_-C35-%n%b-OI-r%", 42),
    "xds-random":    ("XDS-unsw-random",   "XDS-unsw-random-%b-W_-C35-%n%b-OI-r%",   42),
    "xds-cicids":    ("XDS-cicids",        "XDS-cicids-%b-W_-C35-%n%b-OI-r%",        42),
}


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
        print()
        print("XDS aliases (cross-dataset OI cohorts):")
        for alias, (label, _, target) in XDS_ALIASES.items():
            print(f"  --cohort {alias:<14}  {label:<22}  (default target={target})")
        sys.exit(0)

    # XDS alias dispatch: if --cohort matches an XDS friendly name, override
    # the default `{prefix}-OI%-r%` pattern with the XDS-specific LIKE.
    name_pattern = None
    if args.cohort and args.cohort in XDS_ALIASES:
        label, name_pattern, default_target = XDS_ALIASES[args.cohort]
        prefix = label
        if args.target == 112:  # legacy default — override with XDS-specific
            args.target = default_target
    elif args.cohort:
        prefix = args.cohort
        name_pattern = f"{prefix}-OI%-r%"
    elif len(available) == 1:
        prefix = available[0][0]
        name_pattern = f"{prefix}-OI%-r%"
    elif available:
        print("Multiple cohorts; specify --cohort PREFIX:")
        for p, c in available:
            print(f"  {p:<40}  OLD={c}")
        print()
        print("Or use an XDS alias:")
        for alias in XDS_ALIASES:
            print(f"  --cohort {alias}")
        sys.exit(1)
    else:
        print("No cohorts found.", file=sys.stderr)
        sys.exit(2)

    # NEW cohort status (anything with -OI- excluding -OI-OLD- / PREEMP-OLD)
    cur.execute(
        """SELECT status, COUNT(*) FROM flows
        WHERE name LIKE ? AND name NOT LIKE '%OLD%'
        GROUP BY status""",
        (name_pattern,),
    )
    by_status = {r[0]: r[1] for r in cur.fetchall()}
    total = sum(by_status.values())
    completed = by_status.get("completed", 0)
    running = by_status.get("running", 0)
    queued = by_status.get("queued", 0)
    cancelled = by_status.get("cancelled", 0)
    failed = by_status.get("failed", 0)
    pending = by_status.get("pending", 0)

    # Sample the next pickup flow's config to show what HSR the worker's
    # predict_hsr_from_params() will set when it picks the next flow up.
    cur.execute(
        """SELECT config_json FROM flows
        WHERE name LIKE ? AND name NOT LIKE '%OLD%' AND status='queued'
        ORDER BY id DESC LIMIT 1""",
        (name_pattern,),
    )
    next_row = cur.fetchone()
    next_hsr_preview = None
    if next_row:
        try:
            import json
            params = json.loads(next_row[0])["params"]
            explicit = params.get("wnn_hybrid_speed_ratio")
            if explicit is not None:
                next_hsr_preview = f"HSR={explicit} (explicit override from config)"
            else:
                hsr_pred = _predict_hsr(params)
                next_hsr_preview = f"HSR={hsr_pred} (predicted from workload)"
        except Exception as e:
            next_hsr_preview = f"unknown ({type(e).__name__}: {e})"

    # Durations for ETA
    cur.execute(
        """SELECT (julianday(completed_at)-julianday(started_at))*1440 AS m, completed_at
        FROM flows WHERE name LIKE ? AND name NOT LIKE '%OLD%' AND status='completed'
        ORDER BY completed_at""",
        (name_pattern,),
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
    if next_hsr_preview:
        print(f"  Next pickup: {next_hsr_preview}")


if __name__ == "__main__":
    main()
