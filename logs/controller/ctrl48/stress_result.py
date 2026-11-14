#!/usr/bin/env python
"""Read the stress-test logs (experiments/ctrl_stress/<tag>.u<N>.log) and print, per controller:
the WNN fails / episodes and the MPCOF cascade fails / episodes on the SAME test units.

  stress_result.py <run-list> <n_units>

Refuses a run whose re-picked genome differs from its published headline (runs.log END line, or the
P re-pick log for runs flown before rule P)."""
import re
import sys
from pathlib import Path

L = Path("logs/controller/ctrl48")
O = Path("experiments/ctrl_stress")
EPISODES = 100


def published_headline(tag):
	for line in open(L / "runs.log"):
		m = re.search(rf"END {re.escape(tag)} rc=0 genome=(\S+)", line)
		if m and "_P_" in tag:
			return m.group(1)
	repick = Path("experiments/ctrl_select_tail") / f"{tag}.P.log"
	picks = [l for l in open(repick, errors="ignore") if l.startswith("[recalc] PICK")]
	return re.search(r"headline=(\S+)", picks[-1]).group(1)


def read_run(tag, units):
	text = open(O / f"{tag}.u{units}.log", errors="ignore").read()
	pick = re.findall(r"^\[recalc\] PICK .*headline=(\S+) .*report\((\d+) units\): stable=([\d.]+)% err=([\d.]+)°\s+steady=([\d.]+)°.*pos=([\d.]+)m.*fails=(\d+)/(\d+)", text, re.M)[-1]
	blocks = re.split(r"HELD-OUT REPORT \[HEADLINE-", text)[1:]
	mp = {}
	for b in blocks:
		seed = int(re.search(r"FRESH seed (\d+)", b).group(1))
		row = re.search(r"vs MPCOF>rate\(h\d+\)\[est\].*?stable=([\d.]+)%\s+err=([\d.]+)°\s+steady=([\d.]+)°.*?pos=([\d.]+)m", b)
		mp[seed] = tuple(float(x) for x in row.groups())
	return pick, mp


def main(list_path, units):
	tags = [l.strip() for l in open(list_path) if l.strip() and not l.startswith("#")]
	tot_f = tot_n = 0
	mp_all = {}
	print(f"{'controller':52} pick       stable%   err    steady pos    fails")
	for tag in tags:
		(head, n_units, stable, err, steady, pos, fails, eps), mp = read_run(tag, units)
		want = published_headline(tag)
		if head != want:
			raise SystemExit(f"{tag}: stress re-pick {head} != published {want} — not the published controller")
		if int(n_units) != int(units) or len(mp) != int(units):
			raise SystemExit(f"{tag}: {n_units} units scored, {len(mp)} MPCOF rows, expected {units}")
		tot_f += int(fails); tot_n += int(eps)
		mp_all.setdefault(tuple(sorted(mp.items())), []).append(tag)
		short = tag.replace("C-S2-d_wposx2_", "").replace("_full", "")
		print(f"{short:52} {head:10} {float(stable):7.3f}  {float(err):.3f}  {float(steady):.2f}  {float(pos):.3f}  {fails}/{eps}")
	print(f"\nWNN pooled: {tot_f}/{tot_n} fails = {100 * tot_f / tot_n:.4f}%")
	for rows, who in mp_all.items():
		mf = sum(round((100 - r[0]) * EPISODES / 100) for _s, r in rows)
		n = len(rows) * EPISODES
		me = sum(r[1] for _s, r in rows) / len(rows); ms = sum(r[2] for _s, r in rows) / len(rows); mpos = sum(r[3] for _s, r in rows) / len(rows)
		print(f"MPCOF cascade on the same {len(rows)} units ({len(who)} runs agree): {mf}/{n} fails = {100 * mf / n:.4f}%  "
		      f"err {me:.3f}  steady {ms:.3f}  pos {mpos:.3f}")


if __name__ == "__main__":
	main(sys.argv[1], sys.argv[2])
