"""Build the CICIoT-neto-subsample (1.43M, 46 features) report section for
docs/ids_results.md following the format from CLAUDE.md Rule 7 — five tables,
Grid Search vs GA Neurons side-by-side, mean±std per threshold mode.
"""
import json
import re
import sqlite3
import statistics
from pathlib import Path

# Legacy tiers_json string format: "ClusterGenome(clusters=1, neurons=300, bits=[34-34], memory=...)"
# Modern format: '[{"tier":0,"clusters":1,"neurons":300,"bits":34}]'
_LEGACY_RE = re.compile(r"neurons=(\d+).*?bits=\[(\d+)-(\d+)\]")


def parse_tiers(tiers_str):
    """Return (neurons, avg_bits) from any of the three storage formats:

    1. New JSON (post per-class integration):
       {"bits_per_neuron": [32,32,...], "neurons_per_cluster": [91], "threshold": ...}
    2. Tiered JSON (pre-IDS):
       [{"tier":0,"clusters":1,"neurons":300,"bits":34}]
    3. Legacy stringified ClusterGenome:
       "ClusterGenome(clusters=1, neurons=300, bits=[32-34], memory=...)"
    """
    if not tiers_str:
        return None, None
    s = tiers_str.strip()
    # Format 1 — single dict with bits_per_neuron + neurons_per_cluster
    if s.startswith("{"):
        try:
            data = json.loads(s)
            bpn = data.get("bits_per_neuron", [])
            npc = data.get("neurons_per_cluster", [])
            if bpn and npc:
                neurons = sum(npc)
                avg_bits = sum(bpn) / len(bpn)
                return neurons, avg_bits
        except Exception:
            pass
    # Format 2 — list of tier dicts
    if s.startswith("["):
        try:
            data = json.loads(s)
            if data and isinstance(data, list):
                t = data[0]
                return t.get("neurons"), t.get("bits")
        except Exception:
            pass
    # Format 3 — legacy ClusterGenome string
    m = _LEGACY_RE.search(s)
    if m:
        n = int(m.group(1))
        bits_lo, bits_hi = int(m.group(2)), int(m.group(3))
        return n, (bits_lo + bits_hi) / 2
    return None, None

DB = "/Users/lacg/wnn/db/wnn.db"
NAME_PATTERN = "PUB50-neto-sub-ciciot-random%"
TARGET_TOTAL = 112  # final batch size when complete

THRESHOLD_MODES = [
    "train_cal",
    "fixed_05",
    "platt",
    "beta",
    "empirical",
    "empirical_cumulative",
    "val_cal",
]
GENOME_TYPES = ["best_fitness", "best_f1", "best_fpr", "best_acc", "best_ce"]
PHASE_ORDER = ["grid_search", "ga_neurons"]


def fmt_pair(values):
    """mean±std as percent string, or '—' if no data."""
    if not values:
        return "—"
    if len(values) == 1:
        return f"{values[0]*100:.2f}±0.00"
    return f"{statistics.mean(values)*100:.2f}±{statistics.stdev(values)*100:.2f}"


def fmt_int_pair(values):
    if not values:
        return "—"
    if len(values) == 1:
        return f"{values[0]:.0f}±0"
    return f"{statistics.mean(values):.0f}±{statistics.stdev(values):.0f}"


def load():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Completed flows + duration
    cur.execute(
        """
        SELECT id, name, started_at, completed_at,
               (julianday(completed_at) - julianday(started_at)) * 24 * 3600 AS dur_sec
        FROM flows
        WHERE name LIKE ? AND status = 'completed'
        ORDER BY id
        """,
        (NAME_PATTERN,),
    )
    flows = [dict(r) for r in cur.fetchall()]

    # Pull all validation summaries + matching genome architecture
    cur.execute(
        """
        SELECT
            f.id            AS flow_id,
            f.name          AS flow_name,
            e.phase_type    AS phase_type,
            vs.genome_type  AS genome_type,
            vs.f1_macro     AS f1_macro,
            vs.fpr          AS fpr,
            vs.accuracy     AS accuracy,
            vs.ce           AS ce,
            vs.threshold_metadata AS tm,
            g.tiers_json    AS tiers_json
        FROM validation_summaries vs
        JOIN flows       f ON f.id = vs.flow_id
        JOIN experiments e ON e.id = vs.experiment_id
        LEFT JOIN genomes g
               ON g.experiment_id = vs.experiment_id
              AND g.genome_hash   = vs.genome_hash
        WHERE f.name LIKE ?
          AND f.status = 'completed'
          AND vs.validation_point = 'final'
        """,
        (NAME_PATTERN,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    # Parse threshold_metadata JSON + tiers
    for r in rows:
        try:
            r["tm"] = json.loads(r["tm"]) if r["tm"] else {}
        except Exception:
            r["tm"] = {}
        n, b = parse_tiers(r.get("tiers_json"))
        r["neurons"] = n
        r["bits"] = b
    return flows, rows


def build_section(flows, rows):
    out = []
    completed = len(flows)

    # Header block
    if flows:
        durations_sec = [f["dur_sec"] for f in flows if f["dur_sec"] and f["dur_sec"] > 0]
        avg_dur_sec = statistics.mean(durations_sec) if durations_sec else 0
        total_dur_sec = sum(durations_sec)
        latest_done = flows[-1]["completed_at"]
        # ETA = latest_done + remaining * avg_dur
        from datetime import datetime, timedelta, timezone
        try:
            latest_dt = datetime.fromisoformat(latest_done.replace("Z", "+00:00"))
        except Exception:
            latest_dt = None
        remaining = TARGET_TOTAL - completed
        eta_dt = latest_dt + timedelta(seconds=avg_dur_sec * remaining) if latest_dt else None

        def fmt_dt_utc(dt):
            return dt.strftime("%d/%m/%Y %H:%M") if dt else "—"

        def fmt_dt_et(dt):
            if not dt:
                return "—"
            # Approximate ET = UTC - 4h (EDT in April)
            return (dt - timedelta(hours=4)).strftime("%d/%m/%Y %H:%M")

        avg_min = avg_dur_sec / 60
        total_h = total_dur_sec / 3600

        out.append(f"# CIC-IoT-2023 Neto-Subsample (1.43M, 46 features) Random — Multi-Threshold Run")
        out.append("")
        out.append(f"    Completed : {completed}/{TARGET_TOTAL}  |  Total wall: {total_h:.1f}h  |  Avg/run: {avg_min:.0f}m")
        if latest_dt and eta_dt:
            out.append(
                f"    Latest done: {fmt_dt_utc(latest_dt)} UTC ({fmt_dt_et(latest_dt)} ET)  "
                f"|  ETA(112): {fmt_dt_utc(eta_dt)} UTC ({fmt_dt_et(eta_dt)} ET)"
            )
        out.append("")
        out.append("    Note: validation uses the new single-pass evaluate_at_thresholds path")
        out.append("    (commit f04da00f) — 1 training pass instead of 9 per validation point.")
        out.append("")

    # Per-flow architecture overview — exposes outliers (e.g. degenerate
    # tiny genomes from GA) before they get hidden in mean±std.
    flow_arch = {}  # flow_arch[flow_id] = { (genome_type, phase): (neurons, bits) }
    for r in rows:
        if r["neurons"] is None:
            continue
        flow_arch.setdefault(r["flow_id"], {})[(r["genome_type"], r["phase_type"])] = (r["neurons"], r["bits"])

    if flow_arch:
        out.append("### Per-flow architecture (exposes outliers)")
        out.append("")
        out.append("    Flow  Phase         best_f1     best_fpr    best_acc    best_ce     best_fitness")
        out.append("    ----  ------------- ----------- ----------- ----------- ----------- -----------")
        # Sort by flow id ascending → readable chronological order
        for f in flows:
            arch_for_flow = flow_arch.get(f["id"], {})
            for phase in PHASE_ORDER:
                row_parts = [f"r{f['name'].split('-r')[-1]}", phase]
                for gt in ("best_f1", "best_fpr", "best_acc", "best_ce", "best_fitness"):
                    nb = arch_for_flow.get((gt, phase))
                    if nb is None:
                        row_parts.append("—".rjust(11))
                    else:
                        n, b = nb
                        row_parts.append(f"{n:>3}n×{b:>2.0f}b".rjust(11))
                out.append(f"    {row_parts[0]:<5} {row_parts[1]:<13} {row_parts[2]} {row_parts[3]} {row_parts[4]} {row_parts[5]} {row_parts[6]}")
        out.append("")

    # Build per (genome_type, phase, threshold_mode) buckets
    # bucket[(g, p, m)] = list of metric dicts {f1, fpr, acc}
    buckets = {}
    arch = {}  # arch[(g, p)] = list of (neurons, bits) tuples

    for r in rows:
        g = r["genome_type"]
        p = r["phase_type"]
        if r.get("neurons") is not None and r.get("bits") is not None:
            arch.setdefault((g, p), []).append((r["neurons"], r["bits"]))
        tm = r["tm"] or {}
        for mode in THRESHOLD_MODES:
            md = tm.get(mode)
            if not isinstance(md, dict):
                continue
            f1 = md.get("f1")
            fpr = md.get("fpr")
            acc = md.get("acc")
            if f1 is None or fpr is None or acc is None:
                continue
            buckets.setdefault((g, p, mode), []).append({"f1": f1, "fpr": fpr, "acc": acc})

    # Render 5 tables
    for genome_type in GENOME_TYPES:
        # Per-phase architecture summary
        gs_arch = arch.get((genome_type, "grid_search"), [])
        ga_arch = arch.get((genome_type, "ga_neurons"), [])
        gs_n = [a[0] for a in gs_arch]
        gs_b = [a[1] for a in gs_arch]
        ga_n = [a[0] for a in ga_arch]
        ga_b = [a[1] for a in ga_arch]
        n_runs_gs = len({r["flow_id"] for r in rows if r["genome_type"] == genome_type and r["phase_type"] == "grid_search"})
        n_runs_ga = len({r["flow_id"] for r in rows if r["genome_type"] == genome_type and r["phase_type"] == "ga_neurons"})

        out.append(f"## {genome_type}  (GS: {n_runs_gs} runs | GA: {n_runs_ga} runs)")
        out.append(f"    Grid Search : {fmt_int_pair(gs_n)} neurons | {fmt_int_pair(gs_b)} bits")
        out.append(f"    GA Neurons  : {fmt_int_pair(ga_n)} neurons | {fmt_int_pair(ga_b)} bits")
        out.append("")
        out.append("    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA")
        out.append("    ---------------------+---------------------+---------------------+--------------------")

        for mode in THRESHOLD_MODES:
            gs_data = buckets.get((genome_type, "grid_search", mode), [])
            ga_data = buckets.get((genome_type, "ga_neurons", mode), [])
            f1_gs = fmt_pair([d["f1"] for d in gs_data])
            f1_ga = fmt_pair([d["f1"] for d in ga_data])
            fpr_gs = fmt_pair([d["fpr"] for d in gs_data])
            fpr_ga = fmt_pair([d["fpr"] for d in ga_data])
            acc_gs = fmt_pair([d["acc"] for d in gs_data])
            acc_ga = fmt_pair([d["acc"] for d in ga_data])
            out.append(
                f"    {mode:<20} |{f1_gs:>10} {f1_ga:>9} | {fpr_gs:>10} {fpr_ga:>10} |{acc_gs:>10} {acc_ga:>10}"
            )
        out.append("")

    return "\n".join(out)


def main():
    flows, rows = load()
    section = build_section(flows, rows)
    print(section)


if __name__ == "__main__":
    main()
