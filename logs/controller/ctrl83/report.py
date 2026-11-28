"""CTRL-83 stress panel readout from the cache (data/*.npz, data/mpcofB__*.json). Prints, per cell in
cells.CELLS order, ONE plain-text table: stable / err / steady / pos / fails / p99.9 / p99 / worst unit, WNN as
mean±std (N seeds cached) with fails pooled, rivals as single values. Incomplete cells are printed with their N.
Per-episode errors are capped at 8 deg on BOTH sides here (prereg §4). No classification is computed here —
the prereg §5 read is applied by a human on the full grid.
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import cells as C

DATA, CAP = os.path.join(HERE, "data"), 8.0


def row(U, E):
	st = U[:, 7].sum(); n = len(U) * 100
	q = lambda p: np.percentile(np.minimum(E[:, 2], CAP), p, method="inverted_cdf") if len(E) else float("nan")
	stable = np.floor(10000 * (1 - st / n)) / 100
	return [stable, U[:, 2].mean(), U[:, 3].mean(), U[:, 4].mean(), int(st), q(99.9), q(99), U[:, 2].max()]


def main():
	for cid, axis, lab, prim, _ in C.CELLS:
		wnn = [np.load(os.path.join(DATA, f"wnn_s{s}__{cid}.npz")) for s in C.WNN_SEEDS
		       if os.path.exists(os.path.join(DATA, f"wnn_s{s}__{cid}.npz"))]
		riv = {k: np.load(os.path.join(DATA, f"{k}__{cid}.npz")) for k, *_ in C.RIVALS if os.path.exists(os.path.join(DATA, f"{k}__{cid}.npz"))}
		bj = os.path.join(DATA, f"mpcofB__{cid}.json")
		if not wnn and not riv and not os.path.exists(bj):
			continue
		print(f"\n### {cid} — {lab} ({'primary' if prim else 'secondary'}) — TEST, 35 units x 100 eps")
		print(f"    {'controller':16s} | {'stable %':>9s} | {'err °':>12s} | {'steady °':>12s} | {'pos m':>13s} | {'fails':>11s} | {'p99.9 °':>8s} | {'p99 °':>6s} | {'worst unit °':>12s}")
		print("    " + "-" * 16 + "-+" + "-+".join("-" * w for w in (10, 13, 13, 14, 12, 9, 7, 13)))
		if wnn:
			R = np.array([row(z["units"], z["eps"]) for z in wnn])
			m, s = R.mean(0), R.std(0)
			f = int(R[:, 4].sum())
			print(f"    {'WNN l2b (N=%d)' % len(R):16s} | {m[0]:9.2f} | {m[1]:6.3f}±{s[1]:5.3f} | {m[2]:6.3f}±{s[2]:5.3f} | {m[3]:6.4f}±{s[3]:6.4f}"
			      f" | {f:5d}/{len(R)*3500:<5d} | {m[5]:8.3f} | {m[6]:6.3f} | {m[7]:12.3f}")
		if os.path.exists(bj):
			b = json.load(open(bj))["rows"]["B"]
			print(f"    {'MPCOF B eq-info':16s} | {b['stable']:9.2f} | {b['err']:12.3f} | {b['steady']:12.3f} | {b['pos']:13.4f} | {b['fails']:5d}/3500  | {'n/a':>8s} | {'n/a':>6s} | {'n/a':>12s}")
		for k, r in riv.items():
			v = row(r["units"], r["eps"])
			print(f"    {k:16s} | {v[0]:9.2f} | {v[1]:12.3f} | {v[2]:12.3f} | {v[3]:13.4f} | {v[4]:5d}/3500  | {v[5]:8.3f} | {v[6]:6.3f} | {v[7]:12.3f}")


if __name__ == "__main__":
	main()
