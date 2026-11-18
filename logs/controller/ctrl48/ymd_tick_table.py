#!/usr/bin/env python3
"""CTRL-48 M / Y / D per-seed table for the hourly tick (Luiz 18/11/2026).

TEST rows over the 35 CF2.1 units, read from the END lines in runs.log:
  s1-s5 = dm M / dmy Y / dm D (ABI 56/57; determinism gates showed identical
  results across ABIs), s6+ = n10 M / Y / D (ABI 58).
Reference row = MPCOF eq-info, the COMPARATOR (docs/ctrl_rival_row_ruling.md row B).
No alt / pos_hold / dz_hold columns (Luiz).
"""
import re
import statistics
from pathlib import Path

RUNS_LOG = Path(__file__).with_name("runs.log")
ARMS = "MYD"
TAG_TO_ARM = {"dmM": "M", "dmyY": "Y", "dmD": "D", "n10M": "M", "n10Y": "Y", "n10D": "D"}
COMPARATOR = ("MPCOF eq-info", "100.00", "0.691", "0.165", "0.357", "0")
END_RE = re.compile(
	r"END .*_(dmM|dmD|dmyY|n10[MYD])_full_s(\d+) rc=0 .*?stable=([\d.]+)±.*?err=([\d.]+)"
	r".*?steady=([\d.]+).*? pos=([\d.]+).*stable_fail=(\d+)"
)
START_RE = re.compile(r"START .*_(n10[MYD])_full_s(\d+) ")
FMT = ["%.1f", "%.2f", "%.2f", "%.3f", "%d"]
HEADER = ("seed", "stable % (M | Y | D)", "err ° (M | Y | D)", "steady ° (M | Y | D)",
	"pos m (M | Y | D)", "fails /3500 (M | Y | D)")


def parse_runs(lines):
	"""Return {(arm, seed): (stable, err, steady, pos, fails)} and the flying (arm, seed)."""
	rows, flying = {}, None
	for line in lines:
		m = END_RE.search(line)
		if m:
			key = (TAG_TO_ARM[m.group(1)], int(m.group(2)))
			rows[key] = tuple(float(m.group(i)) for i in range(3, 7)) + (int(m.group(7)),)
			continue
		m = START_RE.search(line)
		if m:
			flying = (TAG_TO_ARM[m.group(1)], int(m.group(2)))
	if flying in rows:
		flying = None
	return rows, flying


def seed_row(rows, flying, seed):
	def value(arm, i):
		if (arm, seed) in rows:
			return FMT[i] % rows[(arm, seed)][i]
		return "flying" if flying == (arm, seed) else "–"
	return ("s%d" % seed,) + tuple(" | ".join(value(a, i) for a in ARMS) for i in range(5))


def mean_row(rows, seeds):
	"""Means over seeds where all three arms ended; stable from total fails, fails summed."""
	cols = []
	for i in range(5):
		vals = []
		for arm in ARMS:
			series = [rows[(arm, s)] for s in seeds]
			fails = sum(r[4] for r in series)
			if i == 0:
				vals.append("%.2f" % (100.0 - 100.0 * fails / (3500 * len(series))))
			elif i == 4:
				vals.append("%d" % fails)
			else:
				vals.append("%.3f" % statistics.mean(r[i] for r in series))
		cols.append(" | ".join(vals))
	cols[4] += "  (/%d)" % (3500 * len(seeds))
	return ("mean s%d-s%d" % (seeds[0], seeds[-1]),) + tuple(cols)


def render(rows_out):
	widths = [max(len(r[i]) for r in rows_out + [HEADER]) + 2 for i in range(6)]

	def rule(left, mid, right, ch="─"):
		return left + mid.join(ch * w for w in widths) + right

	def line(r):
		return "│" + "│".join((" " + r[i]).ljust(widths[i]) if i == 0 else r[i].center(widths[i])
			for i in range(6)) + "│"

	out = [rule("┌", "┬", "┐"), line(HEADER), rule("├", "┼", "┤"), line(rows_out[0]),
		rule("╞", "╪", "╡", "═")]
	out += [line(r) for r in rows_out[1:-1]]
	out += [rule("├", "┼", "┤"), line(rows_out[-1]), rule("└", "┴", "┘")]
	return "\n".join(out)


def main():
	rows, flying = parse_runs(RUNS_LOG.read_text().splitlines())
	seeds = sorted({s for _, s in rows} | ({flying[1]} if flying else set()))
	complete = [s for s in seeds if all((a, s) in rows for a in ARMS)]
	table = [COMPARATOR] + [seed_row(rows, flying, s) for s in seeds] + [mean_row(rows, complete)]
	print(render(table))


if __name__ == "__main__":
	main()
