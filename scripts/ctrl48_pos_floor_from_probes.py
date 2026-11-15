"""CTRL-48 round 3 / block 10b: the pos-column ZRank floors from crn_noise_probe.py JSONs (median over
populations x frame pairs of the robust 1.4826*MAD across genomes of D_g = X_g(F1) - X_g(F2)), for
mean_position_error_m and, when the probe carried it, mean_position_hold_m.

  ctrl48_pos_floor_from_probes.py "/tmp/ctrl_poshold/PROBE_*.json"
"""
import glob
import json
import sys

import numpy as np

for key in ("mean_position_error_m", "mean_position_hold_m"):
	est = []
	for f in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ctrl_poshold/PROBE_*.json")):
		pg = json.load(open(f))["per_genome"]
		if key not in pg["A"][0]:
			continue
		tag = f.rsplit("_both_", 1)[-1][:-5]
		for a, b in (("A", "B"), ("A", "C"), ("B", "C")):
			x = np.array([g[key] for g in pg[a]], float); y = np.array([g[key] for g in pg[b]], float)
			ok = np.isfinite(x) & np.isfinite(y); D = (x - y)[ok]
			rob = 1.4826 * np.median(np.abs(D - np.median(D))); est.append(rob)
			sp = 1.4826 * np.median(np.abs(x[ok] - np.median(x[ok])))
			print(f"{key:22s} {tag:14s} {a}-{b}: n {ok.sum()} robust {rob * 1000:.2f} mm  SD {D.std(ddof=1) * 1000:.2f} mm  population spread {sp * 1000:.2f} mm")
	if est:
		print(f"  {key}: median of {len(est)} = {np.median(est) * 1000:.2f} mm")
