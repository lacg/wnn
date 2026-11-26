"""CTRL-80 per-END descriptive readout (TEST, 35 CF2.1 units; CPU re-score npz from ctrl73/wnn_units.py).
Usage: rescore_end.py <arm_tag> <arm_npz> <control_tag> <control_npz>
Gate: each npz must reproduce its run's logged END headline in ctrl48/runs.log (stable / err / steady / pos / fails);
exit 2 with a message otherwise. Then ONE table of VALUES, control | arm, and the fixed hard-episode panel.
Column layout of npz["units"] (wnn_units.py): 0 unit, 1 stable, 2 err, 3 steady, 4 mid, 5 pos, 6 alt, 7 pos_hold,
12-15 whole r/p/y/|v| (rad), 16-19 hold r/p/y/|v| (rad), 22 signed hold yaw (rad), 26-29 motor asym."""
import re, sys
import numpy as np

RUNS_LOG = "/Users/lacg/wnn/logs/controller/ctrl48/runs.log"
PANEL = ((123, 16), (123, 33), (112, 86), (118, 1), (118, 39), (118, 60), (118, 94))
# The logged END is the in-flight GPU scorer printed at 1 / 2 / 2 / 3 decimals; the CPU re-score reads
# +0.0008..+0.0039 deg lower on err / steady (CTRL-70/73 side finding, same sign 40/40). Tolerance = print
# rounding (half a last digit) + that known offset.
TOL = {"stable": 0.06, "err": 0.010, "steady": 0.010, "pos": 0.0015}
D = np.degrees


def logged_end(tag):
	"""The run's last END line in runs.log -> dict(stable, err, steady, pos, fails, n_eps); None if absent."""
	hit = None
	for line in open(RUNS_LOG, errors="replace"):
		if f" END {tag} " in line:
			hit = line
	if hit is None:
		return None
	if " rc=0 " not in hit:
		sys.exit(f"RESCORE FAIL: {tag}: END line is not rc=0: {hit.strip()[:160]}")
	g = lambda k: float(re.search(k + r"=([\d.]+)", hit).group(1))
	f = re.search(r"stable_fail=(\d+)/(\d+)", hit)
	return dict(stable=g(" stable"), err=g(" err"), steady=g(" steady"), pos=g(" pos"), fails=int(f.group(1)), n_eps=int(f.group(2)))


def values(npz):
	u = npz["units"]; m = u.mean(0)
	n_eps = len(npz["eps"]) // len(u)
	c1 = u[:, 26:30] @ [1, -1, 1, -1]
	v = dict(stable=100 * m[1], err=m[2], steady=m[3], pos=m[5], fails=int(round(((1 - u[:, 1]) * n_eps).sum())))
	for i, x in enumerate("rpy"):
		v["w" + x] = D(m[12 + i]); v["h" + x] = D(m[16 + i]); v["p" + x] = (v["w" + x] - 0.2 * v["h" + x]) / 0.8
	v["slope"] = np.polyfit(c1, D(u[:, 22]), 1)[0]
	v["off"] = D(np.abs(u[:, 22]).mean())
	return v


def gate(tag, npz, v):
	if str(npz["tag"]) != tag:
		sys.exit(f"RESCORE FAIL: npz holds tag {npz['tag']} but {tag} was asked for")
	end = logged_end(tag)
	if end is None:
		sys.exit(f"RESCORE FAIL: no END line for {tag} in {RUNS_LOG}")
	bad = [f"{k} re-score {v[k]:.4f} vs logged {end[k]}" for k in TOL if abs(v[k] - end[k]) > TOL[k]]
	if v["fails"] != end["fails"]:
		bad.append(f"fails re-score {v['fails']} vs logged {end['fails']}/{end['n_eps']}")
	if bad:
		print(f"RESCORE FAIL: {tag} does NOT reproduce its logged END headline: " + "; ".join(bad), file=sys.stderr)
		sys.exit(2)
	return end


def ep_err(npz, unit, ep):
	e = npz["eps"]; i = np.where((e[:, 0] % 1000 == unit) & (e[:, 1] == ep))[0]
	return float(e[i[0], 2]) if len(i) else float("nan")


def main():
	atag, anpz, ctag, cnpz = sys.argv[1:5]
	A, C = np.load(anpz), np.load(cnpz)
	va, vc = values(A), values(C)
	ea, ec = gate(atag, A, va), gate(ctag, C, vc)
	short = lambda t: re.sub(r"^.*_yld_", "", t)
	print(f"RESCORE {short(atag)} vs control {short(ctag)}  (TEST, 35 units, CPU re-score; both reproduce their logged END)")
	print(f"  logged END  control: stable {ec['stable']} err {ec['err']} steady {ec['steady']} pos {ec['pos']} fails {ec['fails']}/{ec['n_eps']}"
	      f" | arm: stable {ea['stable']} err {ea['err']} steady {ea['steady']} pos {ea['pos']} fails {ea['fails']}/{ea['n_eps']}")
	rows = [("err deg", "err", 3), ("  whole roll", "wr", 3), ("  whole pitch", "wp", 3), ("  whole yaw", "wy", 3),
	        ("  pre-hold roll", "pr", 3), ("  pre-hold pitch", "pp", 3), ("  pre-hold yaw", "py", 3),
	        ("steady deg", "steady", 3), ("  hold roll", "hr", 3), ("  hold pitch", "hp", 3), ("  hold yaw", "hy", 3),
	        ("signed hold yaw per c1", "slope", 3), ("mean |hold yaw offset|", "off", 3), ("pos m (3-D)", "pos", 4), ("fails", "fails", 0)]
	print(f"{'row':26s}{'control':>10s}{'arm':>10s}")
	for lab, k, nd in rows:
		print(f"{lab:26s}{vc[k]:>10.{nd}f}{va[k]:>10.{nd}f}")
	print(f"{'hard episodes (err deg)':26s}{'control':>10s}{'arm':>10s}")
	for u, k in PANEL:
		print(f"{'  u%d ep%d' % (u, k):26s}{ep_err(C, u, k):>10.2f}{ep_err(A, u, k):>10.2f}")


if __name__ == "__main__":
	main()
