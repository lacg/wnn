"""CTRL-17 G6 readout: lateral content of the training labels, xy 0 vs 0.5 m (30/10/2026).

Input: WNN_LABEL_TRACE CSVs (ep_key, t, e_x, e_y, v_x, v_y, label[4], base[4]).
A tilt command is a motor DIFFERENTIAL, so per step we take each motor's label minus
the 4-motor mean (the collective) and ask how often it reaches one level of the
1/16 label grid (project_live_dagger_label_dead_zone). Split by time window and by
the episode's start offset |(e_x, e_y)| at t=0. One streaming pass (traces are GBs);
--max-rows caps the read.
"""
import argparse, csv, math, itertools

WINDOWS = ((0, 200, "0-0.2 s"), (200, 1000, "0.2-1 s"), (1000, 2000, "1-2 s"), (0, 10**9, "all"))
OFFSETS = ((0.0, 0.18), (0.18, 0.35), (0.35, 0.8))


def run(path, max_rows):
	win = {w[2]: [0, 0, 0] for w in WINDOWS}
	off = {o: [0, 0, 0] for o in OFFSETS}
	r0_of, n_rows = {}, 0
	for r in itertools.islice(csv.reader(open(path)), max_rows):
		n_rows += 1
		ep, t = r[0], int(r[1])
		if t == 0:
			r0_of[ep] = math.hypot(float(r[2]), float(r[3]))
		lab = [float(x) for x in r[6:10]]; m = sum(lab) / 4
		h16 = sum(abs(x - m) >= 1 / 16 for x in lab); h32 = sum(abs(x - m) >= 1 / 32 for x in lab)
		for lo, hi, name in WINDOWS:
			if lo <= t < hi:
				c = win[name]; c[0] += 4; c[1] += h16; c[2] += h32
		r0 = r0_of.get(ep)
		if r0 is not None:
			for o in OFFSETS:
				if o[0] <= r0 < o[1]:
					c = off[o]; c[0] += 4; c[1] += h16; c[2] += h32
	r0s = list(r0_of.values())
	print(f"\n{path}: {n_rows} rows, {len(r0s)} episodes, start |xy| mean {sum(r0s)/max(len(r0s),1):.3f} m")
	print("  window          motor-steps  |diff|>=1/16  |diff|>=1/32")
	for _, _, name in WINDOWS:
		n, a, b = win[name]
		print(f"  {name:14s} {n:12d}  {100*a/max(n,1):10.2f}%  {100*b/max(n,1):10.2f}%")
	if max(r0s, default=0) > 0.05:
		print("  by start offset (all steps):")
		for o in OFFSETS:
			n, a, b = off[o]
			print(f"    |xy0| {o[0]:.2f}-{o[1]:.2f} m  {n:10d}  {100*a/max(n,1):8.2f}%  {100*b/max(n,1):8.2f}%")


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("paths", nargs="+")
	ap.add_argument("--max-rows", type=int, default=4_000_000)
	a = ap.parse_args()
	for p in a.paths:
		run(p, a.max_rows)
