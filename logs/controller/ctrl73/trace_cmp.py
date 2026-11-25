"""CTRL-73 VAL diagnosis (never a result): N0 s1 vs G+T s1 per-tick traces on the same 3 val units x 100 episodes.
Trace cols: 0 unit_idx,1 ep,2 t,3-5 gyro,6-9 q_true,10 x,11 y,12 vx,13 vy,14 alt,15 vz,16-19 sp[T,p,q,r],20-23 pwm."""
import numpy as np
def yaw(q): w,x,y,z=q[:,0],q[:,1],q[:,2],q[:,3]; return np.degrees(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
def roll(q): w,x,y,z=q[:,0],q[:,1],q[:,2],q[:,3]; return np.degrees(np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y)))
W = [(0,150),(150,300),(300,600),(600,1000),(1000,1600),(1600,2000)]
TR = {"N0 s1": "/tmp/ctrl70/trace_n0.npz", "G+T s1": "/tmp/ctrl73/trace_gt.npz"}
GX, GY, GZ = 0.0693, 0.0930, 0.45  # outer cuts (G+T s1 fit; gyro_x/y are the same quantile fit on both)
for lab, f in TR.items():
	T = np.load(f)["trace"].astype(float); t = T[:,2]; yw = yaw(T[:,6:10]); key = T[:,0]*1000+T[:,1]
	spr = T[:,19]; gz = T[:,5]; gx = T[:,3]; gy = T[:,4]
	sel = lambda a,b: (t>=a)&(t<b)
	print(f"== {lab}")
	print("  mean |yaw| deg   : " + " | ".join(f"{abs(yw[sel(a,b)]).mean():.3f}" for a,b in W) + f" | whole {abs(yw).mean():.3f}")
	print("  |yaw|>2.62 deg % : " + " | ".join(f"{100*(abs(yw[sel(a,b)])>2.616).mean():.1f}" for a,b in W) + "   (yaw_err outer cut)")
	print("  |gyro_z|>outer % : " + " | ".join(f"{100*(abs(gz[sel(a,b)])>(0.0305 if lab=='N0 s1' else GZ)).mean():.1f}" for a,b in W))
	print("  |gyro_x|>0.069 % : " + " | ".join(f"{100*(abs(gx[sel(a,b)])>GX).mean():.1f}" for a,b in W))
	print("  |gyro_y|>0.093 % : " + " | ".join(f"{100*(abs(gy[sel(a,b)])>GY).mean():.1f}" for a,b in W))
	ov = []; tc = []; flips = []; y0s = []
	for k in np.unique(key):
		s = key==k; y = yw[s]; tt = t[s]
		y0s.append(abs(y[0])); ov.append(abs(y[(tt>=300)&(tt<1600)]).max())
		i = np.where(abs(y) < 0.5)[0]; tc.append(i[0]/1000 if len(i) else 2.0)
		flips.append((np.diff(np.sign(spr[s][tt>=1600]))!=0).sum()/0.4)
	ov, tc, y0s = map(np.array, (ov, tc, y0s))
	print(f"  max |yaw| 0.3-1.6 s: median {np.median(ov):.2f} p90 {np.percentile(ov,90):.2f}; time to |yaw|<0.5: mean {tc.mean():.3f} s; hold yaw-sp sign flips/s {np.mean(flips):.2f}")
	print("  time to |yaw|<0.5 by |yaw0|: " + " | ".join(f"[{a},{b}) {tc[(y0s>=a)&(y0s<b)].mean():.2f}" for a,b in ((0.5,1.5),(1.5,2.5),(2.5,3.5),(3.5,5.1))))
	print("  overshoot (max|yaw| 0.3-1.6) by |yaw0|: " + " | ".join(f"[{a},{b}) {ov[(y0s>=a)&(y0s<b)].mean():.2f}" for a,b in ((0,1.5),(1.5,2.5),(2.5,3.5),(3.5,5.1))))
	h = t>=1600
	print(f"  hold: |yaw| {abs(yw[h]).mean():.3f}, signed yaw mean {yw[h].mean():+.3f}, std of per-episode hold-mean yaw {np.std([yw[(key==k)&h].mean() for k in np.unique(key)]):.3f}, within-episode hold yaw std {np.mean([yw[(key==k)&h].std() for k in np.unique(key)]):.3f}; |sp_r| {abs(spr[h]).mean():.4f}")
	print(f"  hold |gyro_z| median {np.median(abs(gz[h])):.4f}; bins occupancy (.011/.05/.15/.45): " + " ".join(f"{100*((abs(gz[h])>=a)&(abs(gz[h])<b)).mean():.0f}" for a,b in ((0,.011),(.011,.05),(.05,.15),(.15,.45),(.45,9))))
	print("  mean |roll| deg by window: " + " | ".join(f"{abs(roll(T[sel(a,b)][:,6:10])).mean():.3f}" for a,b in W))
