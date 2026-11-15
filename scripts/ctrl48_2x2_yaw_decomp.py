"""CTRL-48 2x2 s1 diagnosis (docs/ctrl48_block10_2x2_s1_diagnosis.md): decompose the VAL hold attitude error of a
replayed headline genome (scripts/ctrl_gap_student_trace.py --trace-episodes 0 output) into roll / pitch / yaw and
relate the yaw offset to the trim. VAL units only."""
import numpy as np
D=np.degrees
for name in ("L1_","L2y_"):
	z=np.load(f"/tmp/ctrl_s1diag/{name}trace.npz"); R=z["rows"].astype(np.float64); ex=list(z["extra"])
	c=lambda n: R[:,19+ex.index(n)]
	ek=R[:,0]*1000+R[:,1]; _,inv=np.unique(ek,return_inverse=True); em=lambda q: np.bincount(inv,q)/np.bincount(inv)
	err=R[:,3]; r,p,y=R[:,4],R[:,5],R[:,6]
	off=lambda q: D(np.abs(em(q)).mean())
	print(f"{name:5s} hold err {D(err.mean()):.3f}  | mean|.| roll {D(np.abs(r).mean()):.3f} pitch {D(np.abs(p).mean()):.3f} yaw {D(np.abs(y).mean()):.3f}"
	      f" | per-episode offset |mean| roll {off(r):.3f} pitch {off(p):.3f} yaw {off(y):.3f}"
	      f" | within-ep sd roll {D(np.sqrt(((r-em(r)[inv])**2).mean())):.3f} pitch {D(np.sqrt(((p-em(p)[inv])**2).mean())):.3f} yaw {D(np.sqrt(((y-em(y)[inv])**2).mean())):.3f}")
	share=lambda q: (q**2).mean()/((r**2+p**2+y**2).mean())
	print(f"      share of hold err² : roll {share(r):.2f} pitch {share(p):.2f} yaw {share(y):.2f}")
	ye=c("yaw_err"); 
	print(f"      yaw_err input: hold sd {D(ye.std()):.3f} deg, within-ep sd {D(np.sqrt(((ye-em(ye)[inv])**2).mean())):.3f}; corr(ep-mean yaw_err, -ep-mean true yaw) {np.corrcoef(em(ye),-em(y))[0,1]:.3f}; dead-reckon drift |heading-est| {D(np.abs(em(-c('yaw_err')) - em(R[:,9])).mean()) if name=='L1_' else float('nan'):.3f}")
print()
K=0.04165
for name in ("L1_","L2y_"):
	z=np.load(f"/tmp/ctrl_s1diag/{name}trace.npz"); R=z["rows"].astype(np.float64); ex=list(z["extra"])
	c=lambda n: R[:,19+ex.index(n)]
	ek=R[:,0]*1000+R[:,1]; _,inv=np.unique(ek,return_inverse=True); em=lambda q: np.bincount(inv,q)/np.bincount(inv)
	y=em(R[:,6]); cz=em(c("cresid_z")); ye=em(c("yaw_err")); gz=em(c("gyro_z")); yi=em(c("yaw_err_i"))
	b,a=np.polyfit(cz,y,1); r2=np.corrcoef(cz,y)[0,1]**2
	print(f"{name:5s} ep-mean TRUE yaw (deg) ~ cresid_z: slope {D(b):+.3f} deg/unit (kappa = {D(K):.3f}) R² {r2:.2f}; intercept {D(a):+.3f}")
	print(f"      ep-mean yaw_err input |mean| {D(np.abs(ye).mean()):.3f} deg; ep-mean yaw_err_i |mean| {np.abs(yi).mean():.2f}; corr(true yaw, yaw_err_i) {np.corrcoef(y,yi)[0,1]:+.2f}")
	if name=="L2y_":
		yh = ye + K*cz   # the dead-reckoned yaw error the L2y student no longer sees directly
		print(f"      reconstructed raw yaw_err (= input + kappa·cresid): corr with -true yaw {np.corrcoef(yh,-y)[0,1]:.3f}")
print()
for name in ("L1_","L2y_"):
	z=np.load(f"/tmp/ctrl_s1diag/{name}trace.npz"); R=z["rows"].astype(np.float64); ex=list(z["extra"])
	ye=D(R[:,19+ex.index("yaw_err")])
	print(f"{name:5s} hold yaw_err input: |x|<0.25° {np.mean(np.abs(ye)<0.25):.2f}  |x|<0.42° {np.mean(np.abs(ye)<0.42):.2f}  quantiles |x| 50/90 {np.quantile(np.abs(ye),[.5,.9]).round(3)}")
