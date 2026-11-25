"""CTRL-73 VAL diagnosis: corrective yaw-rate setpoint vs |yaw| (t < 0.6 s) — does the response flatten above the yaw_err outer cut (2.62 deg)?"""
import numpy as np
def yaw(q): w,x,y,z=q[:,0],q[:,1],q[:,2],q[:,3]; return np.degrees(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
bins = [0, .25, .5, 1, 1.5, 2, 2.6, 3.5, 5, 9]
for lab, f in {"N0 s1": "/tmp/ctrl70/trace_n0.npz", "G+T s1": "/tmp/ctrl73/trace_gt.npz"}.items():
	T = np.load(f)["trace"].astype(float); t = T[:,2]; yw = yaw(T[:,6:10]); spr = T[:,19]; e = t < 600
	c = -np.sign(yw) * spr
	print(f"{lab}: corrective sp_r (rad/s) by |yaw| deg: " + " | ".join(f"[{a},{b}) {c[e&(abs(yw)>=a)&(abs(yw)<b)].mean():+.3f} (n{(e&(abs(yw)>=a)&(abs(yw)<b)).sum()//1000}k)" for a,b in zip(bins[:-1], bins[1:]) if (e&(abs(yw)>=a)&(abs(yw)<b)).sum() > 200))
