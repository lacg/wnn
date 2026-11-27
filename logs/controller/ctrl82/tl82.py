"""CTRL-82 VAL: timelines of matched roll-excursion episodes + estimator error (student's roll/pitch_err feature vs truth)."""
exec(open("/Users/lacg/wnn/logs/controller/ctrl82/an82.py").read().split("t = np.arange(2000)")[0])
t = np.arange(2000)[None, :]
hk = np.load("/tmp/ctrl82/hk.npy")
for k, d in D.items():
	fr = np.degrees(d["F"][..., d["fnames"].index("roll_err")]); fp = np.degrees(d["F"][..., d["fnames"].index("pitch_err")])
	cr = np.corrcoef(fr.ravel(), d["re"].ravel())[0, 1]; sg = np.sign(cr)
	d["est_r"] = sg * fr - d["re"]; d["est_p"] = np.sign(np.corrcoef(fp.ravel(), d["pe"].ravel())[0, 1]) * fp - d["pe"]
	ex = hk[:, None] & (t < 1600) & ((abs(d["re"]) > 3) | (abs(d["pe"]) > 3))
	print(f"{k}: corr(feature roll_err, true) {cr:+.3f}; |est err| roll/pitch deg: excursion {abs(d['est_r'][ex]).mean():.3f}/{abs(d['est_p'][ex]).mean():.3f}, "
	      f"pre-hold all {abs(d['est_r'][:, :1600]).mean():.3f}/{abs(d['est_p'][:, :1600]).mean():.3f}, hold {abs(d['est_r'][:, 1600:]).mean():.3f}")
key = D["G+T"]["key"]
for want in [(163226145, 54), (163226092, 16), (163226108, 80), (163226100, 41), (163226145, 18)]:
	i = key.index(want)
	print(f"\n== VAL u{want[0]%1000} ep{want[1]}  every 25 ms: roll e | est-err | gyro_x | sp_p | sp_r | gyro_z | yaw deg | |tref| deg")
	for k, d in D.items():
		print(f"  {k}")
		for tt in range(0, 600, 25):
			print(f"   {tt:4d} {d['re'][i,tt]:+6.2f} {d['est_r'][i,tt]:+5.2f} {d['T'][i,tt,3]:+6.2f} {d['T'][i,tt,17]:+6.2f} {d['T'][i,tt,19]:+6.2f} {d['T'][i,tt,5]:+6.2f} {d['yw'][i,tt]:+6.2f} {np.degrees(np.hypot(*d['tref'][i,tt])):5.2f}")
