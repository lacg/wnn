"""CTRL-82 VAL: can an observed signal separate 'hard transient' ticks from hold / normal ticks? 167 traced VAL eps
(47 hard + 120 random) of the l2b s1 winner (the base the lever would ride on); signals = the student's OWN features."""
exec(open("/Users/lacg/wnn/logs/controller/ctrl82/an82.py").read().split("t = np.arange(2000)")[0])
hk = np.load("/tmp/ctrl82/hk.npy"); t = np.arange(2000)[None, :]
for k in ("l2b", "G+T"):
	d = D[k]; F = d["F"]; nm = d["fnames"]
	sig = {"|tilt_err| (est) deg": np.degrees(np.hypot(F[..., nm.index("roll_err")], F[..., nm.index("pitch_err")])),
	       "|gyro_xy| rad/s": np.hypot(F[..., nm.index("gyro_x")], F[..., nm.index("gyro_y")]),
	       "max(|tilt|/3deg, |gyro_xy|/0.5)": np.maximum(np.degrees(np.hypot(F[..., nm.index("roll_err")], F[..., nm.index("pitch_err")])) / 3, np.hypot(F[..., nm.index("gyro_x")], F[..., nm.index("gyro_y")]) / 0.5)}
	tilt = np.hypot(d["re"], d["pe"])
	hard = (t < 1600) & (np.maximum(abs(d["re"]), abs(d["pe"])) > 3)
	late = hard & (t >= 150)                                       # excursions after the initial-condition kick
	hold = np.broadcast_to(t >= 1600, tilt.shape)
	norm = (~hk)[:, None] & (t < 1600) & ~hard
	print(f"== {k} (VAL, tick level): hard-transient ticks {hard.sum()} (of which t>=150 ms: {late.sum()}), hold ticks {hold.sum()}, normal pre-hold ticks {norm.sum()}")
	for sn, v in sig.items():
		ths = {"|tilt_err| (est) deg": (1.0, 1.5, 2.0, 3.0), "|gyro_xy| rad/s": (0.15, 0.3, 0.5, 1.0), "max(|tilt|/3deg, |gyro_xy|/0.5)": (0.5, 0.75, 1.0)}[sn]
		print(f"   {sn}: hold p99 {np.percentile(v[hold], 99):.3f}, p99.99 {np.percentile(v[hold], 99.99):.3f}, max {v[hold].max():.3f}")
		for th in ths:
			f = v > th
			# random episodes: share that EVER fire after 0.3 s (a schedule that fires there changes normal flight)
			ever = (f & (t >= 300))[~hk].any(1).mean() * 100
			print(f"     > {th:<5}: recall hard {100*f[hard].mean():5.1f}% (t>=150ms {100*f[late].mean():5.1f}%) | fires hold {100*f[hold].mean():.4f}% | normal pre-hold {100*f[norm].mean():5.2f}% | random eps firing after 0.3 s {ever:5.1f}%")
