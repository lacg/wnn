"""CTRL-82 VAL: the yaw-runaway hard episodes (whole err dominated by yaw): timeline of yaw, sp_r, its per-decision delta,
gyro_z, yaw_err feature; every 50 ms."""
exec(open("/Users/lacg/wnn/logs/controller/ctrl82/an82.py").read().split("t = np.arange(2000)")[0])
key = D["G+T"]["key"]
for want in [(163226136, 27), (163226136, 71)]:
	i = key.index(want)
	print(f"\n== VAL u{want[0]%1000} ep{want[1]}: t ms | yaw deg | sp_r | delta_r (sp - leak*prev) | gyro_z | yaw force mN")
	for k, d in D.items():
		lk = 0.9 if k == "lk90" else 0.7
		print(f"  {k}: " + " ".join(f"{tt}:{d['yw'][i,tt]:+.1f}/{d['T'][i,tt,19]:+.2f}/{d['T'][i,tt,19]-lk*d['T'][i,max(tt-10,0),19] if tt else d['T'][i,0,19]:+.2f}/{d['T'][i,tt,5]:+.2f}/{1e3*d['I']['cmd'][i,tt,2]:+.0f}" for tt in range(0, 1000, 50)))
