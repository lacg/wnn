#!/usr/bin/env python
"""Write docs/ctrl_select_tail_result.md from the recalc logs (pre-registration docs/ctrl_select_tail_prereg.md).

Reads experiments/ctrl_select_tail/<tag>.<rule>.log `[recalc] PICK` lines (rules B, P, S20, S5p) and each
run's original .out headline. Evaluates exactly the three pre-registered pass criteria on P vs B; no other
inference. Also writes experiments/ctrl_select_tail/picks.json. Usage: select_tail_result.py <run-list>"""
import datetime as dt
import json
import re
import sys

L, O, DOC = "logs/controller/ctrl48", "experiments/ctrl_select_tail", "docs/ctrl_select_tail_result.md"
RULES = ("B", "P", "S20", "S5p")
PICK = re.compile(r"^\[recalc\] PICK \S+ val_units=(\d+) select=(\S+) headline=(\S+) stage=(\S+) "
                  r"report\((\d+) units\): stable=([\d.]+)% err=([\d.]+)°.*?steady=([\d.]+)°.*? fails=(\d+)/(\d+)")


def pick(tag, rule):
	try:
		with open(f"{O}/{tag}.{rule}.log", "rb") as f:
			for raw in f:
				m = PICK.match(raw.decode("utf-8", "replace"))
				if m:
					g = m.groups()
					return dict(val_units=int(g[0]), select=g[1], genome=g[2], units=int(g[4]), stable=float(g[5]),
					            err=float(g[6]), steady=float(g[7]), fails=int(g[8]), episodes=int(g[9]))
	except FileNotFoundError:
		pass
	return None


def original(tag):
	with open(f"{L}/{tag}.out", "rb") as f:
		txt = f.read().decode("utf-8", "replace")
	m = re.search(r"stage-select\] HEADLINE stage=\S+ genome=(\S+)", txt)
	return m.group(1) if m else None


def cell(p):
	return "—" if p is None else f"{p['genome']} | {p['err']:.3f} | {p['steady']:.3f} | {p['fails']}"


def refuse_partial(tags, n):
	"""Hardening (Luiz 12/11/2026): the pre-registered readout is written ONLY when every run in the list has
	both its B and P pick. A missing pick (failed run, wheel/ABI import error) exits 3, so the driver's
	`&& git commit` never publishes a partial readout."""
	if n == len(tags):
		return
	missing = len(tags) - n
	print(f"select_tail_result: REFUSED — {missing}/{len(tags)} runs lack a B or P pick; no readout written",
	      file=sys.stderr)
	sys.exit(3)


def main(listfile):
	tags = [t.strip() for t in open(listfile) if t.strip() and not t.startswith("#")]
	rows = {t: {r: pick(t, r) for r in RULES} | {"orig": original(t)} for t in tags}
	json.dump(rows, open(f"{O}/picks.json", "w"), indent=1)
	pairs = [(t, v["B"], v["P"]) for t, v in rows.items() if v["B"] and v["P"]]
	n = len(pairs)
	refuse_partial(tags, n)
	L_ = ["# Tail-aware selection — pre-registered readout (CTRL-48)", "",
	      f"Generated {dt.datetime.now(dt.timezone.utc):%d/%m/%Y %H:%M} UTC by `{L}/select_tail_result.py` from "
	      f"`{O}/`. Pre-registration: `docs/ctrl_select_tail_prereg.md` (unchanged). Rule P = `--val-units 20 "
	      f"--select-tail p99`; baseline B = 5 val units, no tail term. Every pick scored on the 35 test units x 100 "
	      f"episodes (fixed scorer, controller ABI 49). Runs in the list: {len(tags)}; runs with both B and P: {n}.", "",
	      "## Per-run picks (genome | test err° | test steady° | test fails/3500)", "",
	      "| run | original headline | B | P | S20 (descr.) | S5p (descr.) | B = original? |",
	      "|---|---|---|---|---|---|---|"]
	for t, v in rows.items():
		same = "—" if v["B"] is None else ("yes" if v["B"]["genome"] == v["orig"] else "**NO**")
		L_.append(f"| {t.replace('C-S2-d_wposx2_a49_', '')} | {v['orig']} | {cell(v['B'])} | {cell(v['P'])} | "
		          f"{cell(v['S20'])} | {cell(v['S5p'])} | {same} |")
	L_ += ["", "## Paired P vs B (test units)", "",
	       "| run | B genome | P genome | differs | Δerr° (P−B) | Δsteady° (P−B) | fails B | fails P |",
	       "|---|---|---|---|---|---|---|---|"]
	for t, b, p in pairs:
		L_.append(f"| {t.replace('C-S2-d_wposx2_a49_', '')} | {b['genome']} | {p['genome']} | "
		          f"{'yes' if b['genome'] != p['genome'] else 'no'} | {p['err'] - b['err']:+.3f} | "
		          f"{p['steady'] - b['steady']:+.3f} | {b['fails']} | {p['fails']} |")
	if n:
		fb, fp = sum(b["fails"] for _, b, _ in pairs), sum(p["fails"] for _, _, p in pairs)
		de = sum(p["err"] for *_, p in pairs) / n - sum(b["err"] for _, b, _ in pairs) / n
		ds = sum(p["steady"] for *_, p in pairs) / n - sum(b["steady"] for _, b, _ in pairs) / n
		nd = sum(b["genome"] != p["genome"] for _, b, p in pairs)
		c1, c2, c3 = fp < fb, de <= 0.05 and ds <= 0.05, nd >= 3
		L_ += ["", "## Pass criteria (pre-registered; P adopted only if ALL hold)", "",
		       f"1. Total test fails P < B: P {fp} vs B {fb} (episodes {n * 3500} each) -> **{'PASS' if c1 else 'FAIL'}**",
		       f"2. Mean test err and steady, P not worse than B by > 0.05°: Δerr {de:+.3f}°, Δsteady {ds:+.3f}° -> "
		       f"**{'PASS' if c2 else 'FAIL'}**",
		       f"3. P picks a different genome than B on >= 3 runs: {nd}/{n} -> **{'PASS' if c3 else 'FAIL'}**"
		       + ("" if c3 else " (no evidence either way)"), "",
		       f"**Verdict under the pre-registered rule: {'ADOPTED' if c1 and c2 and c3 else 'NOT ADOPTED'}.**"]
	missing = [t for t, v in rows.items() if not (v["B"] and v["P"])]
	if missing:
		L_ += ["", f"Runs without both B and P picks (see their logs): {', '.join(missing)}"]
	open(DOC, "w").write("\n".join(L_) + "\n")
	print(f"wrote {DOC}: {n} paired runs")


if __name__ == "__main__":
	main(sys.argv[1])
