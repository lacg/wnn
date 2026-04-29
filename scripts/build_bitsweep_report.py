"""Build the BITSWEEP-neto-sub multi-bit-width report — ids_results.md format.

For each bit-width (16, 32, 64, 72, auto128, ...), produces:
  - Header (n flows completed, total wall, avg/run, durations)
  - "Best individual genomes" table — highest F1/FPR/Acc across all
    (flow × phase × genome × threshold) combinations
  - Per-flow architecture overview (neurons × bits per genome)
  - 5 tables (best_fitness, best_f1, best_fpr, best_acc, best_ce) with
    Grid Search vs GA Neurons side-by-side, mean±std per threshold mode

Mirrors scripts/build_neto_sub_report.py per CLAUDE.md Rule 7.
"""
import json
import re
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

DB = "/Users/lacg/wnn/db/wnn.db"
NAME_PREFIX = "BITSWEEP-neto-sub-"

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

_LEGACY_RE = re.compile(r"neurons=(\d+).*?bits=\[(\d+)-(\d+)\]")


def parse_tiers(s):
    if not s:
        return None, None
    s = s.strip()
    if s.startswith("{"):
        try:
            d = json.loads(s)
            bpn = d.get("bits_per_neuron", [])
            npc = d.get("neurons_per_cluster", [])
            if bpn and npc:
                return sum(npc), sum(bpn) / len(bpn)
        except Exception:
            pass
    if s.startswith("["):
        try:
            d = json.loads(s)
            if d and isinstance(d, list):
                t = d[0]
                return t.get("neurons"), t.get("bits")
        except Exception:
            pass
    m = _LEGACY_RE.search(s)
    if m:
        return int(m.group(1)), (int(m.group(2)) + int(m.group(3))) / 2
    return None, None


def fmt_pair(values):
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


def load_per_bit():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        f"""
        SELECT id, name, started_at, completed_at,
               (julianday(completed_at) - julianday(started_at)) * 24 * 3600 AS dur_sec
        FROM flows
        WHERE name LIKE ? AND status = 'completed'
        ORDER BY id
        """,
        (NAME_PREFIX + "%",),
    )
    flows = [dict(r) for r in cur.fetchall()]

    cur.execute(
        f"""
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
        (NAME_PREFIX + "%",),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    for r in rows:
        try:
            r["tm"] = json.loads(r["tm"]) if r["tm"] else {}
        except Exception:
            r["tm"] = {}
        n, b = parse_tiers(r.get("tiers_json"))
        r["neurons"] = n
        r["bits"] = b

    # Group by bit-width (extracted from name: BITSWEEP-neto-sub-{NNb|autoNNN}-r...)
    flows_by_bit = defaultdict(list)
    rows_by_bit = defaultdict(list)
    for f in flows:
        parts = f["name"].split("-")
        bit_label = parts[3]
        flows_by_bit[bit_label].append(f)
    for r in rows:
        parts = r["flow_name"].split("-")
        bit_label = parts[3]
        rows_by_bit[bit_label].append(r)
    return flows_by_bit, rows_by_bit


def build_section_for_bit(bit_label, flows, rows):
    out = []
    completed = len(flows)
    out.append(f"# BITSWEEP {bit_label} — neto-sub (1.43M, 46f), K-fold=5")
    out.append("")

    if flows:
        durs = [f["dur_sec"] for f in flows if f["dur_sec"] and f["dur_sec"] > 0]
        avg_min = (statistics.mean(durs) / 60) if durs else 0
        total_h = (sum(durs) / 3600) if durs else 0
        out.append(
            f"    Completed : {completed} runs  |  Total wall: {total_h:.1f}h  |  Avg/run: {avg_min:.0f}m"
        )
        out.append("")

    # Best individual genomes
    candidates = []
    for r in rows:
        seed_str = r["flow_name"].rsplit("-r", 1)[-1] if "-r" in r["flow_name"] else "?"
        flow_label = f"r{seed_str}"
        phase_short = "GS" if r["phase_type"] == "grid_search" else "GA"
        for mode in THRESHOLD_MODES:
            md = (r["tm"] or {}).get(mode)
            if not isinstance(md, dict):
                continue
            f1, fpr, acc = md.get("f1"), md.get("fpr"), md.get("acc")
            if f1 is None or fpr is None or acc is None:
                continue
            candidates.append({
                "flow": flow_label,
                "phase": phase_short,
                "genome": r["genome_type"],
                "mode": mode,
                "f1": f1, "fpr": fpr, "acc": acc,
                "neurons": r.get("neurons"),
                "bits": r.get("bits"),
            })

    if candidates:
        criteria = [
            ("Best F1 (any FPR)",   None,                 "f1"),
            ("Best F1 (FPR<14%)",   ("fpr_le", 0.14),     "f1"),
            ("Best F1 (FPR<10%)",   ("fpr_le", 0.10),     "f1"),
            ("Best F1 (FPR<5%)",    ("fpr_le", 0.05),     "f1"),
            ("Best FPR (F1>70%)",   ("f1_ge",  0.70),     "fpr_min"),
            ("Best FPR (F1>80%)",   ("f1_ge",  0.80),     "fpr_min"),
            ("Best Acc (any FPR)",  None,                 "acc"),
        ]
        out.append("### Best individual genomes")
        out.append("")
        out.append("    Metric                   |     F1 |    FPR |    Acc | Flow Phase Genome        Threshold             | arch")
        out.append("    -------------------------+--------+--------+--------+-------------------------------------------------+-----------")
        for label, constraint, sort_key in criteria:
            pool = candidates
            if constraint:
                kind, val = constraint
                if kind == "fpr_le":
                    pool = [c for c in pool if c["fpr"] <= val]
                elif kind == "f1_ge":
                    pool = [c for c in pool if c["f1"] >= val]
            if not pool:
                continue
            if sort_key == "f1":
                best = max(pool, key=lambda c: c["f1"])
            elif sort_key == "acc":
                best = max(pool, key=lambda c: c["acc"])
            elif sort_key == "fpr_min":
                best = min(pool, key=lambda c: c["fpr"])
            else:
                continue
            arch_str = f"{best['neurons']:>3}n×{best['bits']:>2.0f}b" if best.get("neurons") else "—"
            out.append(
                f"    {label:<24} | {best['f1']*100:5.2f}% | {best['fpr']*100:5.2f}% | {best['acc']*100:5.2f}% | "
                f"{best['flow']:<5} {best['phase']:<3} {best['genome']:<14} {best['mode']:<19}| {arch_str}"
            )
        out.append("")

    # Per-flow architecture
    flow_arch = {}
    for r in rows:
        if r["neurons"] is None:
            continue
        flow_arch.setdefault(r["flow_id"], {})[(r["genome_type"], r["phase_type"])] = (r["neurons"], r["bits"])

    if flow_arch:
        out.append("### Per-flow architecture")
        out.append("")
        out.append("    Flow  Phase         best_f1     best_fpr    best_acc    best_ce     best_fitness")
        out.append("    ----  ------------- ----------- ----------- ----------- ----------- -----------")
        for f in flows:
            arch_for_flow = flow_arch.get(f["id"], {})
            seed = f["name"].rsplit("-r", 1)[-1]
            for phase in ["grid_search", "ga_neurons"]:
                cells = []
                for gt in ("best_f1", "best_fpr", "best_acc", "best_ce", "best_fitness"):
                    nb = arch_for_flow.get((gt, phase))
                    if nb is None:
                        cells.append("—".rjust(11))
                    else:
                        n, b = nb
                        cells.append(f"{n:>3}n×{b:>2.0f}b".rjust(11))
                out.append(f"    r{seed:<3} {phase:<13} {cells[0]} {cells[1]} {cells[2]} {cells[3]} {cells[4]}")
        out.append("")

    # 5 tables, one per genome type, GS vs GA side-by-side per threshold mode
    buckets = {}
    arch = {}
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
            f1 = md.get("f1"); fpr = md.get("fpr"); acc = md.get("acc")
            if f1 is None or fpr is None or acc is None:
                continue
            buckets.setdefault((g, p, mode), []).append({"f1": f1, "fpr": fpr, "acc": acc})

    for genome_type in GENOME_TYPES:
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
    flows_by_bit, rows_by_bit = load_per_bit()

    # Sort: numeric bit-widths ascending, "auto" entries last
    def sort_key(s):
        if s.startswith("auto"):
            tail = s[4:]
            return (1, int(tail) if tail.isdigit() else 999)
        return (0, int(s.rstrip("b")))

    sections = []
    for bit_label in sorted(flows_by_bit.keys(), key=sort_key):
        sec = build_section_for_bit(bit_label, flows_by_bit[bit_label], rows_by_bit[bit_label])
        sections.append(sec)
    print("\n".join(sections))


if __name__ == "__main__":
    main()
