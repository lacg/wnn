"""Update docs/ids_results.md with 5-table reports for both cohorts + delta comparison."""
import json, sqlite3, statistics, sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB = Path("/Users/lacg/wnn/db/wnn.db")
GENOMES = ["best_fitness", "best_f1", "best_fpr", "best_acc", "best_ce"]
MODES = ["train_cal", "fixed_05", "platt", "beta", "empirical", "empirical_cumulative", "val_cal"]

COHORTS = {
    "OLD": {
        "pattern": "WSWEEP-T20-96b-C35-250n100b-FIXED-OLD-r%",
        "title": "OLD cohort — FIXED, pre-fixes (paper baseline)",
        "target": 63,
    },
    "NEW": {
        "pattern": "WSWEEP-T20-96b-C35-250n100b-OI%-r%",
        "exclude": "%OLD%",
        "title": "NEW cohort — OI-v2 (val_evaluator + _oi cache key + empirical_cumulative fix)",
        "target": 112,
    },
}

def fmt_pair(values):
    if not values:
        return "  —  "
    if len(values) == 1:
        return f"{values[0]:>5.2f}±0.00"
    return f"{statistics.mean(values):>5.2f}±{statistics.stdev(values):.2f}"

def fmt_int_pair(values):
    if not values:
        return "—"
    if len(values) == 1:
        return f"{int(values[0])}±0"
    return f"{statistics.mean(values):.0f}±{statistics.stdev(values):.0f}"

def parse_arch_from_tiers(tiers_json):
    """Returns (neurons, bits) from tiers_json. Handles two known formats:
    - Dict: {bits_per_neuron:[...], neurons_per_cluster:[...]}  (FIXED-OLD era)
    - List: [{neurons:N, bits:B, ...}]                          (older format)
    Returns (None, None) if unparseable."""
    try:
        tj = json.loads(tiers_json) if isinstance(tiers_json, str) else tiers_json
    except Exception:
        return None, None
    if isinstance(tj, dict):
        bpn = tj.get("bits_per_neuron") or []
        npc = tj.get("neurons_per_cluster") or []
        if bpn and npc:
            return sum(npc), max(bpn)
        return None, None
    if isinstance(tj, list) and tj:
        try:
            n = sum(t.get("neurons", 0) for t in tj)
            b = max(t.get("bits", 0) for t in tj)
            return (n or None), (b or None)
        except Exception:
            return None, None
    return None, None

def pull_cohort(cur, pattern, exclude=None):
    if exclude:
        where = "f.name LIKE ? AND f.name NOT LIKE ? AND f.status='completed'"
        params = (pattern, exclude)
    else:
        where = "f.name LIKE ? AND f.status='completed'"
        params = (pattern,)
    cur.execute(f"""
        SELECT f.id AS flow_id, f.name, f.completed_at, f.started_at,
               e.phase_type, vs.genome_type, vs.threshold_metadata,
               g.total_neurons, g.tiers_json
        FROM flows f
        JOIN experiments e ON e.flow_id = f.id
        LEFT JOIN validation_summaries vs ON vs.experiment_id = e.id AND vs.validation_point = 'final'
        LEFT JOIN genomes g ON g.genome_hash = vs.genome_hash AND g.experiment_id = e.id
        WHERE {where}
    """, params)
    by_cell = defaultdict(lambda: {"f1": [], "fpr": [], "acc": []})
    by_arch = defaultdict(lambda: {"neurons": [], "bits": []})
    flow_ids = set()
    durations = []
    completed_ats = []
    seen_dur = set()
    for r in cur:
        flow_ids.add(r["flow_id"])
        if r["started_at"] and r["completed_at"] and r["flow_id"] not in seen_dur:
            sa = datetime.fromisoformat(r["started_at"].replace("Z","+00:00"))
            ca = datetime.fromisoformat(r["completed_at"].replace("Z","+00:00"))
            durations.append((ca - sa).total_seconds()/60)
            completed_ats.append(ca)
            seen_dur.add(r["flow_id"])
        if r["threshold_metadata"] is None:
            continue
        gt = r["genome_type"]
        phase = r["phase_type"]
        if gt not in GENOMES or phase not in ("grid_search", "ga_neurons"):
            continue
        try:
            tm = json.loads(r["threshold_metadata"])
        except Exception:
            continue
        # Architecture
        if r["tiers_json"]:
            n, b = parse_arch_from_tiers(r["tiers_json"])
            if n is not None:
                by_arch[(gt, phase)]["neurons"].append(n)
            elif r["total_neurons"] is not None:
                by_arch[(gt, phase)]["neurons"].append(r["total_neurons"])
            if b is not None:
                by_arch[(gt, phase)]["bits"].append(b)
        for mode in MODES:
            entry = tm.get(mode)
            if isinstance(entry, dict) and entry.get("f1") is not None:
                by_cell[(gt, phase, mode)]["f1"].append(entry["f1"] * 100)
                by_cell[(gt, phase, mode)]["fpr"].append(entry["fpr"] * 100)
                by_cell[(gt, phase, mode)]["acc"].append(entry["acc"] * 100)
    return {
        "by_cell": by_cell, "by_arch": by_arch,
        "flow_count": len(flow_ids), "durations": durations,
        "completed_ats": sorted(completed_ats),
    }

def render_cohort_section(label, cohort, target):
    by_cell = cohort["by_cell"]
    by_arch = cohort["by_arch"]
    n_flows = cohort["flow_count"]
    durs = cohort["durations"]
    cas = cohort["completed_ats"]
    avg = statistics.mean(durs) if durs else 0
    total_h = sum(durs)/60 if durs else 0
    latest = cas[-1] if cas else None
    eta_utc = "n/a"
    if latest and avg and n_flows < target:
        eta_dt = latest + timedelta(minutes=(target-n_flows)*avg)
        eta_utc = f"{eta_dt.strftime('%d/%m/%Y %H:%M UTC')}  |  {(eta_dt - timedelta(hours=4)).strftime('%d/%m/%Y %H:%M ET')}"

    lines = [f"## {label}", ""]
    lines.append(f"    Completed : {n_flows}/{target}")
    lines.append(f"    Total wall: {total_h:.1f}h")
    lines.append(f"    Avg/run   : {avg:.0f} min")
    if latest:
        lines.append(f"    Latest    : {latest.strftime('%d/%m/%Y %H:%M UTC')}")
        if n_flows < target:
            lines.append(f"    ETA       : {eta_utc}")
    lines.append("")
    for gt in GENOMES:
        gs_arch = by_arch.get((gt, "grid_search"), {"neurons":[], "bits":[]})
        ga_arch = by_arch.get((gt, "ga_neurons"), {"neurons":[], "bits":[]})
        gs_runs = len(by_cell[(gt, "grid_search", "train_cal")]["f1"])
        ga_runs = len(by_cell[(gt, "ga_neurons", "train_cal")]["f1"])
        lines.append(f"### {gt}  (GS: {gs_runs} runs | GA: {ga_runs} runs)")
        lines.append(f"    Grid Search : {fmt_int_pair(gs_arch['neurons'])} neurons | {fmt_int_pair(gs_arch['bits'])} bits")
        lines.append(f"    GA Neurons  : {fmt_int_pair(ga_arch['neurons'])} neurons | {fmt_int_pair(ga_arch['bits'])} bits")
        lines.append("")
        lines.append(f"    Threshold            | F1 Grid    F1 GA    | FPR Grid   FPR GA   | Acc Grid   Acc GA")
        lines.append(f"    ---------------------+---------------------+---------------------+--------------------")
        for mode in MODES:
            gs = by_cell.get((gt, "grid_search", mode), {"f1":[], "fpr":[], "acc":[]})
            ga = by_cell.get((gt, "ga_neurons", mode), {"f1":[], "fpr":[], "acc":[]})
            lines.append(f"    {mode:<20} |{fmt_pair(gs['f1'])} {fmt_pair(ga['f1'])} |{fmt_pair(gs['fpr'])} {fmt_pair(ga['fpr'])} |{fmt_pair(gs['acc'])} {fmt_pair(ga['acc'])}")
        lines.append("")
    return "\n".join(lines)

def render_delta_section(old, new):
    lines = ["## Delta — NEW vs OLD (GA Neurons phase only)", ""]
    lines.append("Positive ΔF1/ΔAcc = NEW better; negative ΔFPR = NEW better.")
    lines.append("")
    for gt in GENOMES:
        lines.append(f"### {gt}")
        lines.append("")
        lines.append(f"    Threshold            |   F1 OLD    F1 NEW   ΔF1   |  FPR OLD   FPR NEW  ΔFPR  |  Acc OLD   Acc NEW  ΔAcc")
        lines.append(f"    ---------------------+-----------------------------+----------------------------+----------------------------")
        for mode in MODES:
            o = old["by_cell"].get((gt, "ga_neurons", mode), {"f1":[], "fpr":[], "acc":[]})
            n = new["by_cell"].get((gt, "ga_neurons", mode), {"f1":[], "fpr":[], "acc":[]})
            def mean(xs): return statistics.mean(xs) if xs else None
            def vstr(x): return f"{x:6.2f}" if x is not None else "  —   "
            def dstr(o, n):
                if o is None or n is None: return "  —   "
                return f"{n-o:+6.2f}"
            of1, nf1 = mean(o["f1"]), mean(n["f1"])
            ofpr, nfpr = mean(o["fpr"]), mean(n["fpr"])
            oacc, nacc = mean(o["acc"]), mean(n["acc"])
            lines.append(f"    {mode:<20} |   {vstr(of1)}   {vstr(nf1)} {dstr(of1,nf1)}  |   {vstr(ofpr)}  {vstr(nfpr)} {dstr(ofpr,nfpr)}  |  {vstr(oacc)}   {vstr(nacc)} {dstr(oacc,nacc)}")
        lines.append("")
    return "\n".join(lines)

def main():
    con = sqlite3.connect(str(DB))
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    old = pull_cohort(cur, COHORTS["OLD"]["pattern"])
    new = pull_cohort(cur, COHORTS["NEW"]["pattern"], COHORTS["NEW"]["exclude"])

    now_utc = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")
    out = [
        f"# T20 96b C35 250n×100b — OI-v2 vs OLD baseline ({now_utc})",
        "",
        "**OLD cohort** = pre-fix FIXED flows (paper baseline). Pre-April-28 (or pre-OI-v2) semantics.",
        "**NEW cohort** = OI-v2 (post-fixes). Same architecture, same dataset. Differences:",
        "  - OI training (WNN_ORDER_INDEPENDENT_TRAIN=1) — order-independent QUAD vote accumulation",
        "  - Validation cache key includes `_oi<0|1>` suffix — no cross-cohort contamination",
        "  - `empirical_cumulative` threshold uses flow's actual fitness weights (was hard-coded F1)",
        "",
        "---",
        "",
        render_cohort_section(COHORTS["OLD"]["title"], old, COHORTS["OLD"]["target"]),
        "---",
        "",
        render_cohort_section(COHORTS["NEW"]["title"], new, COHORTS["NEW"]["target"]),
        "---",
        "",
        render_delta_section(old, new),
    ]
    return "\n".join(out)

if __name__ == "__main__":
    print(main())
