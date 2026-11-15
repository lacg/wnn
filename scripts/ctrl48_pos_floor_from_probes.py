"""CTRL-48 round 3: the pos-only ZRank floor from crn_noise_probe.py JSONs (median of the robust per-pair 1.4826*MAD(D))."""
import json, glob, numpy as np
est=[]
for f in sorted(glob.glob("/tmp/ctrl_posfloor/PROBE_*.json")):
	d=json.load(open(f)); pg=d['per_genome']; tag=f.split('PROBE_')[1][:-5].replace("C-S2-d_wposx2_a51_rf_c200_dc_biasin_ms01_P_both_","")
	for a,b in (("A","B"),("A","C"),("B","C")):
		x=np.array([g["mean_position_error_m"] for g in pg[a]]); y=np.array([g["mean_position_error_m"] for g in pg[b]]); D=x-y
		rob=1.4826*np.median(np.abs(D-np.median(D))); est.append(rob)
		spread=1.4826*np.median(np.abs(x-np.median(x)))
		print(f"{tag:12s} {a}-{b}: SD_g(D) {D.std(ddof=1)*1000:.2f} mm  robust 1.4826*MAD(D) {rob*1000:.2f} mm  | population pos spread 1.4826*MAD {spread*1000:.2f} mm")
print(f"median of {len(est)} robust estimates: {np.median(est)*1000:.2f} mm (GA MEMORY frame: 5 pools x 200 eps = 1000 episodes)")
print(f"stage-select frame (20 units x 100 = 2000 episodes) by 1/sqrt(2): {np.median(est)/np.sqrt(2)*1000:.2f} mm")
