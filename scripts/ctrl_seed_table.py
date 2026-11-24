#!/usr/bin/env python3
"""Per-seed controller comparison table from runs.log END lines (Luiz 19/11/2026) — ONE implementation for every
tick / report table: the M | Y | D box table and the B | F pair table.

TEST over the 35 CF2.1 units. stable % is FLOORED from the fails count (Luiz: 1 fail can never print 100%):
per seed floor to 0.1, means floor to 0.01. A cell is "flying" when its run has a START and no END yet, "–" otherwise.

Usage:
  ctrl_seed_table.py --arm 'B=_dmD_full_s{s}$|_n10D_full_s{s}$' --arm 'F=_a59_.*_ffD_F_full_s{s}$' --seeds 1-10 --style pair
  ctrl_seed_table.py --arm 'M=...' --arm 'Y=...' --arm 'D=...' --seeds 1-10 --style box \
      --ref 'MPCOF eq-info=100.00,0.691,0.165,0.357,0'
Arm patterns are regexes on the run TAG with {s} = the seed number; the LAST matching END wins.
--style pair adds Δ columns (last arm − first arm) for err / steady / pos. --delta 'X:Y' (repeatable) instead adds one
column per pair, "ΔX−Y err/steady/pos", for any two arms (e.g. --delta 'D+F:D' --delta 'Y+F:Y'); works with either style.
Label arms by their recipe (D, D+F, Y, Y+F), never a bare "B" that means different bases in different tables.

TIE-BREAKER columns (Luiz 23/11): `cells M (tb)` = TRUE keys of the published winner (experiments/h743_keys.json, a
missing finished winner is counted once via scripts/count_true_keys.py) and `min (tb)` = START→END wall minutes from
runs.log. They help decide between arms that TIE on the main metrics; they never outvote stable/err/steady/pos/fails.
"""
import argparse
import json
import math
import re
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

RUNS_LOG = Path("/Users/lacg/wnn/logs/controller/ctrl48/runs.log")
UNITS_EPS = 3500
END_RE = re.compile(r"END (\S+) rc=0 .*?stable=[\d.]+±.*?err=([\d.]+).*?steady=([\d.]+).*? pos=([\d.]+).*stable_fail=(\d+)")
START_RE = re.compile(r"START (\S+) ")
TS_RE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ) ")
REPO = Path("/Users/lacg/wnn")
KEYS_CACHE = REPO / "experiments/h743_keys.json"
VENV_PY = REPO / "wnn/bin/python"
METRICS = ("stable", "err", "steady", "pos", "fails")
TIEBREAK = ("cells", "min")
UNIT = {"stable": "%", "err": "°", "steady": "°", "pos": "m", "fails": f"/{UNITS_EPS}", "cells": "M (tb)", "min": "(tb)"}


def floor_to(x: float, digits: int) -> float:
	f = 10 ** digits
	return math.floor(x * f + 1e-9) / f


def stable_pct(fails: int, episodes: int, digits: int) -> float:
	return floor_to(100.0 * (1 - fails / episodes), digits)


def stamp(line: str):
	m = TS_RE.match(line)
	return datetime.fromisoformat(m.group(1).replace("Z", "+00:00")) if m else None


def parse_log(lines):
	"""{tag: (err, steady, pos, fails)} for finished runs, {tag: start time} for started runs, {tag: wall minutes}."""
	ends, starts, minutes = {}, {}, {}
	for line in lines:
		m = END_RE.search(line)
		if m:
			tag = m.group(1)
			ends[tag] = (float(m.group(2)), float(m.group(3)), float(m.group(4)), int(m.group(5)))
			t0, t1 = starts.get(tag), stamp(line)
			if t0 and t1:
				minutes[tag] = (t1 - t0).total_seconds() / 60
			continue
		m = START_RE.search(line)
		if m:
			starts[m.group(1)] = stamp(line)
	return ends, starts, minutes


def count_missing(tags, cache: dict) -> None:
	"""Count TRUE keys once for finished winners not yet in the cache (count_true_keys.py appends to it)."""
	for tag in tags:
		if tag in cache:
			continue
		winners = sorted(REPO.glob(f"logs/controller/*/{tag}_winner.yaml.gz"))
		if winners:
			subprocess.run([str(VENV_PY), str(REPO / "scripts/count_true_keys.py"), "--winner", str(winners[-1])],
				cwd=REPO, env={"PYTHONPATH": str(REPO / "src/wnn"), "PATH": "/usr/bin:/bin"}, capture_output=True, timeout=900)


def load_cells(tags) -> dict:
	"""{tag: TRUE keys} for the published winners of the given finished runs."""
	read = lambda: json.loads(KEYS_CACHE.read_text()) if KEYS_CACHE.exists() else {}  # noqa: E731
	cache = read()
	if any(t not in cache for t in tags):
		count_missing(tags, cache)
		cache = read()
	return {t: cache[t]["true_keys"] for t in tags if t in cache}


def lookup(pattern: str, seed: int, ends: dict, starts: dict):
	"""(tag or None, state) for one arm at one seed."""
	rx = re.compile(pattern.replace("{s}", str(seed)))
	done = [t for t in ends if rx.search(t)]
	if done:
		return done[-1], "done"
	return None, "flying" if any(rx.search(t) for t in starts) else "–"


def fmt_tb(cells, mins) -> dict:
	"""Formatted tie-breaker cells: '?' when the value is unknown."""
	return {"tb": (cells, mins), "cells": "?" if cells is None else "%.3f" % (cells / 1e6),
		"min": "?" if mins is None else "%.0f" % mins}


def seed_values(arms, seed, log):
	"""Per arm: dict of formatted metric strings + the raw row (or the state word)."""
	ends, starts, minutes, cells = log
	out = []
	for _, pat in arms:
		tag, state = lookup(pat, seed, ends, starts)
		if tag is None:
			out.append({"state": state})
			continue
		err, steady, pos, fails = row = ends[tag]
		out.append({"raw": row, "stable": "%.1f" % stable_pct(fails, UNITS_EPS, 1), "err": "%.2f" % err,
			"steady": "%.2f" % steady, "pos": "%.3f" % pos, "fails": "%d" % fails, **fmt_tb(cells.get(tag), minutes.get(tag))})
	return out


def mean_tb(cols) -> dict:
	"""Mean tie-breakers over the given finished cells; '?' if any is unknown."""
	vals = list(zip(*(c["tb"] for c in cols)))
	avg = lambda v: None if any(x is None for x in v) else statistics.mean(v)  # noqa: E731
	return fmt_tb(avg(vals[0]), avg(vals[1]))


def mean_values(per_seed):
	"""PAIRED means over the seeds where every arm WITH RESULTS finished (arms with none show '–');
	stable floored from pooled fails."""
	cols = list(zip(*per_seed.values()))
	active = [a for a, col in enumerate(cols) if any("raw" in c for c in col)]
	common = [cells for cells in per_seed.values() if all("raw" in cells[a] for a in active)]
	if not common:
		return None, 0
	n = len(common)
	out = []
	for a in range(len(cols)):
		if a not in active:
			out.append({"state": "–"})
			continue
		rows = [cells[a]["raw"] for cells in common]
		tb = mean_tb([cells[a] for cells in common])
		fails = sum(r[3] for r in rows)
		out.append({"raw": tuple(statistics.mean(r[i] for r in rows) for i in range(3)) + (fails,),
			"stable": "%.2f" % stable_pct(fails, UNITS_EPS * n, 2), "err": "%.3f" % statistics.mean(r[0] for r in rows),
			"steady": "%.3f" % statistics.mean(r[1] for r in rows), "pos": "%.3f" % statistics.mean(r[2] for r in rows),
			"fails": "%d" % fails, **tb})
	return out, n


def total_values(per_seed):
	"""UNPAIRED totals: each arm over ALL its own finished seeds (Luiz 19/11: the paired mean hid fails on seeds
	another arm has not reached yet). fails = count per arm (denominator = n_arm x 3500, n in the row label); err/steady/pos = that arm's own mean."""
	out = []
	for col in zip(*per_seed.values()):
		done = [c for c in col if "raw" in c]
		rows = [c["raw"] for c in done]
		if not rows:
			out.append({"state": "–"})
			continue
		n, fails = len(rows), sum(r[3] for r in rows)
		out.append({"stable": "%.2f" % stable_pct(fails, UNITS_EPS * n, 2), "err": "%.3f" % statistics.mean(r[0] for r in rows),
			"steady": "%.3f" % statistics.mean(r[1] for r in rows), "pos": "%.3f" % statistics.mean(r[2] for r in rows),
			"fails": "%d" % fails, "n": n, **mean_tb(done)})
	return out


def joined(cells, key):
	return "|".join(c.get(key, c.get("state", "–")) for c in cells)


def deltas(cells):
	"""Δ last − first for err / steady / pos (pair style)."""
	a, b = cells[0], cells[-1]
	if "raw" not in a or "raw" not in b:
		return ["", "", ""]
	return ["%+.3f" % (b["raw"][i] - a["raw"][i]) for i in range(3)]


def pair_deltas(cells, pairs):
	"""One "err/steady/pos" Δ cell per (x, y) arm-index pair; blank until both arms finished."""
	out = []
	for x, y in pairs:
		a, b = cells[y], cells[x]
		out.append("/".join("%+.3f" % (b["raw"][i] - a["raw"][i]) for i in range(3)) if "raw" in a and "raw" in b else "")
	return out


def extra(cells, style, pairs):
	if pairs:
		return pair_deltas(cells, pairs)
	return deltas(cells) if style == "pair" else []


COLUMNS = METRICS + TIEBREAK


def build_rows(arms, seeds, log, style, pairs=()):
	per_seed = {s: seed_values(arms, s, log) for s in seeds}
	rows = []
	for s, cells in per_seed.items():
		rows.append(["s%d" % s] + [joined(cells, k) for k in COLUMNS] + extra(cells, style, pairs))
	mean, n = mean_values(per_seed)
	if mean:
		row = ["mean (n=%d)" % n] + [joined(mean, k) for k in COLUMNS]
		rows.append(row + extra(mean, style, pairs))
	total = total_values(per_seed)
	if any("n" in t for t in total) and any("n" in t and t["n"] != n for t in total):
		nn = "/".join(str(t.get("n", 0)) for t in total)
		rows.append(["all (n=%s)" % nn] + [joined(total, k) for k in COLUMNS] + [""] * len(extra(total, style, pairs)))
	return rows


def header(labels, style, pairs=()):
	h = ["seed"] + [f"{k} {UNIT[k]}" for k in COLUMNS]  # arm order printed ONCE in the legend line (Luiz 20/11: width)
	if pairs:
		h += [f"Δ{labels[x]}−{labels[y]} err/steady/pos" for x, y in pairs]
	elif style == "pair":
		h += [f"Δ{labels[-1]}−{labels[0]} err", "steady", "pos"]
	return h


def render_pair(h, rows, refs):
	allr = [h] + refs + rows
	w = [max(len(r[i]) for r in allr) for i in range(len(h))]
	fmt = lambda r: " | ".join(r[i].ljust(w[i]) for i in range(len(h))).rstrip()  # noqa: E731
	return "\n".join([fmt(h)] + [fmt(r) for r in refs] + [fmt(r) for r in rows])


def render_box(h, rows, refs):
	allr = [h] + refs + rows
	w = [max(len(r[i]) for r in allr) + 2 for i in range(len(h))]
	rule = lambda l, m, r, ch="─": l + m.join(ch * x for x in w) + r  # noqa: E731
	line = lambda r: "│" + "│".join((" " + r[i]).ljust(w[i]) if i == 0 else r[i].center(w[i]) for i in range(len(h))) + "│"  # noqa: E731
	out = [rule("┌", "┬", "┐"), line(h), rule("├", "┼", "┤")]
	if refs:
		out += [line(r) for r in refs] + [rule("╞", "╪", "╡", "═")]
	foot = [r for r in rows if r[0].startswith(("mean", "all ("))]
	body = [r for r in rows if r not in foot]
	out += [line(r) for r in body]
	if foot:
		out += [rule("├", "┼", "┤")] + [line(r) for r in foot]
	out.append(rule("└", "┴", "┘"))
	return "\n".join(out)


def parse_seeds(text: str) -> list[int]:
	out = []
	for part in text.split(","):
		lo, _, hi = part.partition("-")
		out += list(range(int(lo), int(hi or lo) + 1))
	return out


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--arm", action="append", required=True, help="LABEL=regex-on-tag with {s}")
	ap.add_argument("--seeds", required=True, help="e.g. 1-10 or 1,3,5")
	ap.add_argument("--style", choices=("box", "pair"), default="box")
	ap.add_argument("--ref", action="append", default=[], help="LABEL=stable,err,steady,pos,fails (one value, all arms)")
	ap.add_argument("--delta", action="append", default=[], help="X:Y arm labels -> a ΔX−Y err/steady/pos column")
	ap.add_argument("--log", default=str(RUNS_LOG))
	a = ap.parse_args()
	arms = [tuple(x.split("=", 1)) for x in a.arm]
	ends, starts, minutes = parse_log(Path(a.log).read_text(errors="replace").splitlines())
	seeds = parse_seeds(a.seeds)
	shown = {lookup(p, s, ends, starts)[0] for _, p in arms for s in seeds} - {None}
	log = (ends, starts, minutes, load_cells(shown))
	labels = [lab for lab, _ in arms]
	pairs = [tuple(labels.index(x) for x in d.split(":")) for d in a.delta]
	h = header(labels, a.style, pairs)
	rows = build_rows(arms, seeds, log, a.style, pairs)
	pad = len(h) - 6
	refs = [[lab] + v.split(",") + [""] * pad for lab, v in (r.split("=", 1) for r in a.ref)]
	print("arms: " + " | ".join(labels) + "   (every cell lists the arms in this order; fails = count, /3500 per seed;"
		" (tb) = tie-breakers only: cells = TRUE keys of the published winner, min = wall minutes)")
	print((render_pair if a.style == "pair" else render_box)(h, rows, refs))


if __name__ == "__main__":
	main()
