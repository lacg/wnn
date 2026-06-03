"""Generate docs/controller_results.md from a curriculum-GA sweep log.

Parses the per-generation + stage-summary lines that run_curriculum_ga.py emits
and produces a table over ALL 18 weight combos (W1-W4 + C1-C14) with:
  - weights (err²/stable/jerk/mono)
  - search-time stable/err (last generation — the GA's own optimistic fitness)
  - held-out stable/err/reward (the stage-summary re-eval — the honest number)
  - generations run, total wall, per-generation wall
  - status (done / running / pending) + ranking-so-far (held-out stable, then err)

Re-run any time as combos complete:
  python scripts/build_controller_results.py [LOG] [OUT]
Defaults: LOG = $(cat /tmp/curric_log.txt), OUT = docs/controller_results.md
"""

import importlib.util
import re
import sys
import time
from pathlib import Path

LOG = sys.argv[1] if len(sys.argv) > 1 else Path("/tmp/curric_log.txt").read_text().strip()
OUT = sys.argv[2] if len(sys.argv) > 2 else "docs/controller_results.md"

# Full 18-combo list (names + weights) straight from the source of truth.
_spec = importlib.util.spec_from_file_location("rc", "tests/run_curriculum_ga.py")
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
COMBOS = _m.SWEEP_COMBOS

_combo_re = re.compile(r"# COMBO (\w+):")
_gen_re = re.compile(
	r"Gen (\d+)/(\d+): best=\S+ .*?stable=([\d.]+)%, err=([\d.]+)°.*?\| ([\d.]+)s \(offspring")
_best_re = re.compile(
	r"best: err=([\d.]+)°\s+stable=([\d.]+)%\s+reward=(-?[\d.]+|-?inf)\s+iters=(\d+)\s+wall=([\d.]+)s")

def parse_log(path: str) -> dict:
	"""Parse one sweep log → {combo_name: record}. Records carry search-time and
	held-out (stage-summary) metrics plus done/running status."""
	out: dict = {}
	cur = None
	for ln in Path(path).read_text().splitlines():
		cm = _combo_re.search(ln)
		if cm:
			cur = cm.group(1)
			out.setdefault(cur, {"gen_secs": []})
			continue
		if cur is None:
			continue
		gm = _gen_re.search(ln)
		if gm:
			d = out[cur]
			d["last_gen"] = int(gm.group(1))
			d["total_gens"] = int(gm.group(2))
			d["search_stable"] = float(gm.group(3))
			d["search_err"] = float(gm.group(4))
			d["gen_secs"].append(float(gm.group(5)))
			continue
		bm = _best_re.search(ln)
		if bm:
			d = out[cur]
			d["ho_err"] = float(bm.group(1))
			d["ho_stable"] = float(bm.group(2))
			d["reward"] = float(bm.group(3))
			d["iters"] = int(bm.group(4))
			d["wall"] = float(bm.group(5))
			d["done"] = True
	return out


data = parse_log(LOG)

# Round 2+ multi-seed logs: argv[3:], else auto-discover known /tmp pointers.
_extra_args = sys.argv[3:]
if not _extra_args:
	for _p in ("/tmp/curric_r2_log.txt", "/tmp/curric_r3_log.txt"):
		try:
			_lp = Path(_p).read_text().strip()
			if _lp and Path(_lp).exists():
				_extra_args.append(_lp)
		except FileNotFoundError:
			pass
round_logs = [data] + [parse_log(p) for p in _extra_args]


def status(name):
	d = data.get(name)
	if not d:
		return "pending"
	return "done" if d.get("done") else "running"


def per_gen(d):
	if d.get("done") and d.get("iters"):
		return d["wall"] / d["iters"]
	gs = d.get("gen_secs") or []
	return sum(gs) / len(gs) if gs else None


def fnum(v, suf="", nd=2):
	return f"{v:.{nd}f}{suf}" if v is not None else "—"


rows = []
for c in COMBOS:
	nm = c["name"]; d = data.get(nm, {})
	st = status(nm)
	pg = per_gen(d)
	rows.append({
		"name": nm,
		"w": f"{c['err']:.2f}/{c['stable']:.2f}/{c['jerk']:.2f}/{c['mono']:.2f}",
		"s_stab": d.get("search_stable"), "s_err": d.get("search_err"),
		"h_stab": d.get("ho_stable"), "h_err": d.get("ho_err"), "rew": d.get("reward"),
		"gens": (f"{d.get('last_gen','?')}/{d.get('total_gens','?')}" if d else "—"),
		"wall": d.get("wall"), "pg": pg, "status": st,
	})

# Ranking: completed combos by held-out stable desc, then held-out err asc.
done_rows = [r for r in rows if r["status"] == "done" and r["h_stab"] is not None]
done_rows.sort(key=lambda r: (-r["h_stab"], r["h_err"]))

gen_at = time.strftime("%d/%m/%Y %H:%M:%S %Z")
out = []
out.append("# Controller curriculum — weight-sweep results\n")
out.append(f"_Generated {gen_at} from `{LOG}`._\n")
out.append("Sweep config: Stage A only (250-step / 5° tilt / body-rate 0.5), pop=50, "
           "gens=30, patience=3, kfold-eval=5, Rust DAGGER (jerk/mono active). The "
           "auto-full winner then runs the 5-stage IC curriculum at pop=200 / 500 steps.\n")
out.append("**search** = GA's own last-gen fitness (optimistic); **held-out** = "
           "stage-summary re-eval on a fresh draw (honest). W1 dropped 71%→54% on re-eval.\n")

out.append("\n## All 18 combos (weights err²/stable/jerk/mono)\n")
hdr = ("| combo | weights | search stable | search err | held-out stable | held-out err | "
       "reward | gens | total | per-gen | status |")
sep = "|---|---|---|---|---|---|---|---|---|---|---|"
out.append(hdr); out.append(sep)
for r in rows:
	out.append("| {name} | {w} | {ss} | {se} | {hs} | {he} | {rw} | {gens} | {wall} | {pg} | {st} |".format(
		name=r["name"], w=r["w"],
		ss=fnum(r["s_stab"], "%", 1), se=fnum(r["s_err"], "°"),
		hs=fnum(r["h_stab"], "%", 1), he=fnum(r["h_err"], "°"),
		rw=fnum(r["rew"], "", 2),
		gens=r["gens"],
		wall=(f"{r['wall']/60:.0f}m" if r["wall"] else "—"),
		pg=(f"{r['pg']:.0f}s" if r["pg"] else "—"),
		st=r["status"]))

out.append("\n## Ranking so far (completed combos — by held-out stable, then err)\n")
if done_rows:
	out.append("| # | combo | weights | held-out stable | held-out err | reward | per-gen | total |")
	out.append("|---|---|---|---|---|---|---|---|")
	for i, r in enumerate(done_rows, 1):
		out.append(f"| {i} | {r['name']} | {r['w']} | {fnum(r['h_stab'],'%',1)} | "
		           f"{fnum(r['h_err'],'°')} | {fnum(r['rew'],'',2)} | "
		           f"{r['pg']:.0f}s | {r['wall']/60:.0f}m |")
else:
	out.append("_No combos completed yet._")

n_done = len(done_rows)

# Rounds 2-3 confirmation set: stable=0.50 family ∪ top-8 by held-out.
# Single-seed held-out proved noisy (C7 = stable=0.50/jerk=0.10, best search-time
# 79%, crashed to 47% held-out — same recipe as C2 which held 63%). So we widen
# the multi-seed confirmation beyond a naive top-6: always include the stable=0.50
# family (it's the historically-strong recipe), plus the top-8 held-out.
family = [c["name"] for c in COMBOS if abs(c["stable"] - 0.50) < 1e-9]
top8 = [r["name"] for r in done_rows[:8]]
conf = []
for nm in family + top8:
	if nm not in conf:
		conf.append(nm)
out.append("\n## Rounds 2-3 confirmation set (stable=0.50 family ∪ top-8 held-out)\n")
out.append(f"- stable=0.50 family (fixed): **{', '.join(family)}**")
out.append(f"- top-8 by held-out: {', '.join(top8) if top8 else '(none yet)'}")
out.append(f"- **`--combos {','.join(conf)}`** → {len(conf)} combos × 2 fresh seeds (rounds 2-3)")
if n_done < 18:
	out.append(f"- _provisional — top-8 firms up once round 1 completes ({n_done}/18 done)_")

# Multi-seed rounds tally: per-combo held-out stable across rounds + mean±std.
if len(round_logs) > 1:
	import statistics as _st
	n_rounds = len(round_logs)
	# combos that appear (done) in any round, ordered by the confirmation set then rest.
	order = conf + [c["name"] for c in COMBOS if c["name"] not in conf]
	out.append(f"\n## Multi-seed rounds — held-out stable % across {n_rounds} seeds\n")
	out.append("Round 1 = base seed 42; rounds 2-3 = fresh seeds (confirmation set only). "
	           "Mean±std over completed rounds. Watch for combos that crash at a fresh "
	           "seed (overfit) vs hold steady (robust).\n")
	rhdr = "| combo | weights | " + " | ".join(f"R{i+1}" for i in range(n_rounds)) + " | mean±std | rounds |"
	out.append(rhdr)
	out.append("|" + "---|" * (n_rounds + 4))
	wmap = {c["name"]: f"{c['err']:.2f}/{c['stable']:.2f}/{c['jerk']:.2f}/{c['mono']:.2f}" for c in COMBOS}
	tally = []
	for nm in order:
		vals = []
		cells = []
		for rl in round_logs:
			d = rl.get(nm, {})
			if d.get("done") and d.get("ho_stable") is not None:
				vals.append(d["ho_stable"]); cells.append(f"{d['ho_stable']:.0f}%")
			else:
				cells.append("—" if nm in rl or rl is data else "·")
		if not vals:
			continue
		mean = _st.mean(vals)
		ms = f"{mean:.1f}±{_st.pstdev(vals):.1f}" if len(vals) > 1 else f"{mean:.1f}"
		tally.append((mean, len(vals), nm, wmap[nm], cells, ms))
	# sort by mean desc, then more rounds first
	tally.sort(key=lambda t: (-t[0], -t[1]))
	for mean, nseed, nm, w, cells, ms in tally:
		out.append(f"| {nm} | {w} | " + " | ".join(cells) + f" | {ms} | {nseed} |")
	if not tally:
		out.append("| _(no multi-seed combos complete yet)_ |" + " |" * (n_rounds + 3))

out.append(f"\n_{n_done}/18 combos complete (round 1)._\n")

Path(OUT).parent.mkdir(parents=True, exist_ok=True)
Path(OUT).write_text("\n".join(out) + "\n")
print(f"wrote {OUT} ({n_done}/18 combos done)")
