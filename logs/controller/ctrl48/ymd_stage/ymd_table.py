"""18/11 (Luiz): Y vs D (vs M) with the stage-select draw removed. Every stage-select candidate (top-3 of
GRID / NEURONS / MEMORY; GRID arch-only entries from the CTRL-56 repaired copy) scored through the CPU
scorer with axis_out, on the run's 20 VAL units (tuning) and 35 TEST units (report). Rule inputs (VAL fit,
p99 tail, union rank R) are the runs' own stage-select tables (.out). deg / m; fails = TEST episodes (35x100)."""
import re
import sys
import numpy as np
D = "/Users/lacg/wnn/logs/controller/ctrl48/"
S = D + "ymd_stage/scores/"
ARMS = {"M": ("dmyM", "a57"), "Y": ("dmyY", "a57"), "D": ("dmD", "a56")}
CANDS = [f"{st}#{i}" for st in ("GRID", "NEURONS", "MEMORY") for i in range(3)]
# s1-s5: the dm/dmy blocks (ABI 56/57); s6-s10: the n=10 extension (ABI 58, tags n10{M,Y,D}). Usage: ymd_table.py [seeds...]
N10 = {"M": ("n10M", "a58"), "Y": ("n10Y", "a58"), "D": ("n10D", "a58")}
SEEDS = tuple(int(x) for x in sys.argv[1:]) or (1, 2, 3, 4, 5)


def arm_id(arm, s):
	return N10[arm] if s >= 6 else ARMS[arm]
dg = np.degrees


def tag(arm, s):
	a, abi = arm_id(arm, s)
	return f"C-S2-d_wposx2_{abi}_rf_c200_dc_biasin_ms01_P_both_L1_{a}_full_s{s}"


def stage_table(arm, s):
	"""{cand: (fit, p99, R)} + headline label, from the run's .out."""
	txt = open(D + tag(arm, s) + ".out", errors="replace").read()
	out = {}
	for m in re.finditer(r"^\s+((?:GRID|NEURONS|MEMORY)#\d)\s+val .*?fit=([-\d.]+) p99=([\d.]+)° R=([\d.]+)", txt, re.M):
		out[m.group(1)] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
	head = re.findall(r"\[stage-select\] HEADLINE stage=\S+ genome=(\S+)", txt)[-1]
	return out, head


def vec(arm, s, cand, part):
	z = np.load(f"{S}{arm_id(arm, s)[0]}_s{s}_{cand.replace('#', '-')}_{part}.npz")["rows"]
	m = z.mean(0)
	n_eps = 100 * len(z)
	return dict(stable=100 * m[1], err=m[2], steady=m[3], alt=m[4], pos=m[5], dzh=m[6],
	            sy=dg(m[13]), fails=int(round((1 - z[:, 1]).sum() * 100)), n=n_eps)


KEYS = ("steady", "err", "pos", "fails", "sy", "alt")


def fmt(v):
	return f"st {v['steady']:.3f} err {v['err']:.3f} pos {v['pos']:.3f} fails {v['fails']:>2} yaw {v['sy']:.3f} alt {v['alt']:.3f}"


def dif(a, b):
	return {k: a[k] - b[k] for k in KEYS}


def dfmt(d):
	return f"st {d['steady']:+.3f} err {d['err']:+.3f} pos {d['pos']:+.4f} fails {d['fails']:+3d} yaw {d['sy']:+.3f}"


def mean_v(vs):
	return {k: (sum(v[k] for v in vs) if k == "fails" else float(np.mean([v[k] for v in vs]))) for k in KEYS}


tables = {(a, s): stage_table(a, s) for a in ARMS for s in SEEDS}

# ---------------------------------------------------------------- 1. every candidate
for part in ("val", "test"):
	print(f"\n=== 1. every candidate, {part.upper()} ({'TUNING' if part == 'val' else 'REPORT'}) — VAL fit / p99 from the run; * = rule-P headline ===")
	for s in SEEDS:
		for a in ARMS:
			t, head = tables[(a, s)]
			cells = []
			for c in CANDS:
				v = vec(a, s, c, part)
				cells.append(f"{c[0]}{c[-1]}{'*' if c == head else ' '}{v['steady']:.3f}/{v['sy']:.3f}")
			print(f" s{s} {a}  " + "  ".join(cells) + "   (steady/yaw)")

# ---------------------------------------------------------------- 2. stage for stage
def stage_pick(a, s, st, part, how):
	t, _ = tables[(a, s)]
	cs = [c for c in CANDS if c.startswith(st)]
	if how == "best":
		return vec(a, s, min(cs, key=lambda c: t[c][0]), part)
	return mean_v([vec(a, s, c, part) for c in cs])


for part in ("val", "test"):
	print(f"\n=== 2. stage for stage, {part.upper()} ({'TUNING' if part == 'val' else 'REPORT'}): differences per seed and pooled ===")
	for how in ("best", "mean3"):
		for st in ("GRID", "NEURONS", "MEMORY"):
			for x, y in (("Y", "M"), ("D", "M"), ("D", "Y")):
				ds = [dif(stage_pick(x, s, st, part, how), stage_pick(y, s, st, part, how)) for s in SEEDS]
				pooled = {k: (sum(d[k] for d in ds) if k == "fails" else float(np.mean([d[k] for d in ds]))) for k in KEYS}
				ps = " ".join(f"{d['steady']:+.3f}" for d in ds)
				print(f"  {how:<5} {st:<7} {x}-{y}: pooled {dfmt(pooled)} | steady per seed {ps}")

# ---------------------------------------------------------------- 3. headline rules
def rule_pick(a, s, rule):
	t, head = tables[(a, s)]
	if rule == "P (as flown)":
		return head
	if rule == "fitness-only":
		return min(CANDS, key=lambda c: t[c][0])
	return min([c for c in CANDS if c.startswith("MEMORY")], key=lambda c: t[c][0])


print("\n=== 3. headline rules applied retrospectively — TEST (REPORT) of the picked genome; picks chosen on VAL only ===")
summary = {}
for rule in ("P (as flown)", "fitness-only", "best-of-MEMORY"):
	print(f" rule {rule}")
	for a in ARMS:
		picks = [rule_pick(a, s, rule) for s in SEEDS]
		vs = [vec(a, s, c, "test") for s, c in zip(SEEDS, picks)]
		m = mean_v(vs); summary[(rule, a)] = (m, vs)
		print(f"   {a}: {fmt(m)}  picks {' '.join(picks)}")
	for x, y in (("Y", "M"), ("D", "M"), ("D", "Y")):
		ds = [dif(p, q) for p, q in zip(summary[(rule, x)][1], summary[(rule, y)][1])]
		print(f"   {x}-{y}: {dfmt(mean_v(ds))} | steady per seed {' '.join(f'{d[chr(115)+chr(116)+chr(101)+chr(97)+chr(100)+chr(121)]:+.3f}' for d in ds)}")

# ---------------------------------------------------------------- 4. noise
print("\n=== 4. stage-select noise — TEST steady across the 9 candidates (duplicates included, e.g. NEURONS#0 = MEMORY#0) vs between-arm differences ===")
spread = []
for a in ARMS:
	for s in SEEDS:
		v = [vec(a, s, c, "test")["steady"] for c in CANDS]
		spread.append((a, s, np.std(v, ddof=1), max(v) - min(v)))
		print(f"  {a} s{s}: SD {np.std(v, ddof=1):.3f}  range {max(v) - min(v):.3f}  (min {min(v):.3f} max {max(v):.3f})")
print(f"  mean within-run candidate SD {np.mean([x[2] for x in spread]):.3f}, mean range {np.mean([x[3] for x in spread]):.3f}")
for rule in ("P (as flown)", "fitness-only", "best-of-MEMORY"):
	ds = np.array([p["steady"] - q["steady"] for p, q in zip(summary[(rule, "D")][1], summary[(rule, "Y")][1])])
	sd = ds.std(ddof=1); mu = ds.mean()
	n80 = int(np.ceil(((1.96 + 0.84) * sd / max(abs(mu), 1e-9)) ** 2)) if abs(mu) > 0 else 999
	n30 = int(np.ceil(((1.96 + 0.84) * sd / 0.03) ** 2))
	print(f"  D-Y steady under rule {rule:<15}: mean {mu:+.3f} SD(seed) {sd:.3f} -> seeds for 80% power at the observed diff ~{n80}, at a 0.03 deg diff ~{n30}")

# ---------------------------------------------------------------- 5. prereg verdict (docs/ctrl48_n10_ymd_prereg.md)
def verdict(x, y, rule="fitness-only"):
	"""x vs y under the prereg whole-vector thresholds, on the re-picked TEST headlines."""
	d = mean_v([dif(p, q) for p, q in zip(summary[(rule, x)][1], summary[(rule, y)][1])])
	ok = {"steady < -0.03": d["steady"] < -0.03, "err <= +0.03": d["err"] <= 0.03,
	      "pos <= +0.010": d["pos"] <= 0.010}
	return d, ok


print(f"\n=== 5. PREREG VERDICT, fitness-only re-pick, TEST, n={len(SEEDS)} (fails: pooled counts, judged by Luiz vs the gate) ===")
for x, y in (("Y", "M"), ("D", "M"), ("D", "Y")):
	d, ok = verdict(x, y)
	print(f"  {x}-{y}: {dfmt(d)}  " + "  ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in ok.items()))
