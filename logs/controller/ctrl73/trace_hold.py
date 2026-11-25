"""CTRL-73 VAL diagnosis: the hold-window yaw loop (t >= 1.6 s) — setpoint quantisation, limit cycle, yaw_err code occupancy."""
import numpy as np
def yaw(q): w,x,y,z=q[:,0],q[:,1],q[:,2],q[:,3]; return np.degrees(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
for lab, f in {"N0 s1": "/tmp/ctrl70/trace_n0.npz", "G+T s1": "/tmp/ctrl73/trace_gt.npz"}.items():
	T = np.load(f)["trace"].astype(float); t = T[:,2]; h = t >= 1600; key = (T[:,0]*1000+T[:,1])
	yw = yaw(T[:,6:10]); spr = T[:,19]; gz = T[:,5]
	print(f"== {lab} hold")
	vals, cnt = np.unique(np.round(spr[h], 4), return_counts=True)
	top = np.argsort(cnt)[::-1][:8]
	print("  sp_r most common values (rad/s, % ticks): " + " ".join(f"{vals[i]:+.4f}:{100*cnt[i]/h.sum():.0f}" for i in top) + f" | distinct {len(vals)}")
	em = np.array([spr[(key==k)&h].mean() for k in np.unique(key)]); es = np.array([spr[(key==k)&h].std() for k in np.unique(key)])
	print(f"  per-episode hold sp_r: |mean| {abs(em).mean():.4f} (bias needed vs wear), within-ep std {es.mean():.4f}")
	print(f"  corr(sp_r, -yaw) {np.corrcoef(spr[h], -yw[h])[0,1]:+.2f}; corr(sp_r, -gyro_z) {np.corrcoef(spr[h], -gz[h])[0,1]:+.2f}")
	b = [0, .10, .25, 1.23, 2.62, 99]
	print("  hold |yaw| occupancy of yaw_err bins (0-.10/.10-.25/.25-1.23/1.23-2.62/>2.62 deg): " + " ".join(f"{100*((abs(yw[h])>=a)&(abs(yw[h])<c)).mean():.0f}" for a,c in zip(b[:-1],b[1:])))
	# dominant hold period from sign changes of yaw minus its episode mean
	zc = np.mean([(np.diff(np.sign(yw[(key==k)&h]-yw[(key==k)&h].mean()))!=0).sum()/0.4/2 for k in np.unique(key)])
	print(f"  hold yaw oscillation ~{zc:.1f} Hz (zero crossings about the episode mean / 2)")
	# setpoint step size: hold |delta sp_r| per 10-tick action
	d = np.diff(spr[h & (t % 10 == 0)]); d = d[abs(d) > 1e-6]
	print(f"  nonzero setpoint changes in hold: median |d| {np.median(abs(d)):.4f} rad/s, p10 {np.percentile(abs(d),10):.4f}, p90 {np.percentile(abs(d),90):.4f}")
