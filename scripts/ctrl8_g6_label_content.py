"""CTRL-17 G6 readout: lateral content of the training labels, xy 0 vs 0.5 m (30/10/2026).

Input: WNN_LABEL_TRACE CSVs (ep_key, t, e_x, e_y, v_x, v_y, label[4], base[4]).
A tilt command is a motor DIFFERENTIAL, so per step we take each motor's label minus
the 4-motor mean (the collective) and ask how often it reaches one level of the
1/16 label grid (project_live_dagger_label_dead_zone). Split by time window and by
the episode's start offset |(e_x, e_y)| at t=0.
"""
import csv, math, sys, collections

def load(path):
	eps = collections.defaultdict(list)
	for r in csv.reader(open(path)):
		eps[r[0]].append([float(x) for x in r[1:]])
	return eps

def stats(eps, t_lo, t_hi, r_lo=0.0, r_hi=9.0):
	n = hit16 = hit32 = 0; dmax = []
	for rows in eps.values():
		r0 = math.hypot(rows[0][1], rows[0][2])
		if not (r_lo <= r0 < r_hi):
			continue
		for row in rows:
			t = int(row[0])
			if not (t_lo <= t < t_hi):
				continue
			lab = row[5:9]; m = sum(lab) / 4
			for x in lab:
				d = abs(x - m); n += 1
				hit16 += d >= 1 / 16; hit32 += d >= 1 / 32
	return n, (100 * hit16 / n if n else float('nan')), (100 * hit32 / n if n else float('nan'))

for path in sys.argv[1:]:
	eps = load(path)
	r0 = [math.hypot(v[0][1], v[0][2]) for v in eps.values()]
	print(f"\n{path}: {len(eps)} episodes, start |xy| mean {sum(r0)/len(r0):.3f} m")
	print("  window            motor-steps  |diff|>=1/16  |diff|>=1/32")
	for lo, hi, lab in ((0, 200, "0-0.2 s"), (200, 1000, "0.2-1 s"), (1000, 2000, "1-2 s"), (0, 2000, "all")):
		n, a, b = stats(eps, lo, hi)
		print(f"  {lab:16s} {n:12d}  {a:10.2f}%  {b:10.2f}%")
	if max(r0) > 0.05:
		print("  by start offset (all steps):")
		for lo, hi in ((0, 0.18), (0.18, 0.35), (0.35, 0.8)):
			n, a, b = stats(eps, 0, 2000, lo, hi)
			print(f"    |xy0| {lo:.2f}-{hi:.2f} m  {n:10d}  {a:8.2f}%  {b:8.2f}%")
