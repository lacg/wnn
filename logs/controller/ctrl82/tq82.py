"""CTRL-82 VAL: torque book-keeping in the stalled-roll window of matched episodes (roll axis): rate-loop roll force command,
physical roll torque (asym+lag), its yaw-induced part, the implied external torque (gust/jitter) = Ix*dp/dt - tau."""
exec(open("/Users/lacg/wnn/logs/controller/ctrl82/an82.py").read().split("t = np.arange(2000)")[0])
key = D["G+T"]["key"]
for want in [(163226145, 54), (163226092, 16), (163226108, 80)]:
	i = key.index(want); u = want[0]
	print(f"\n== VAL u{u%1000} ep{want[1]} asym {np.round(P[u]['dist_motor_asym'],3)}  every 25 ms (mN·m for torques, mN for forces): roll e | sp_p | gyro_x | roll F cmd | yaw F cmd | tau_roll | yaw->roll | ext (25 ms mean)")
	Ix = P[u]["inertia"][0]
	for k, d in D.items():
		gx = d["T"][i, :, 3]; dp = np.gradient(gx) / 0.001
		ext = Ix * dp - d["tau"][i, :, 0]
		print(f"  {k}")
		for tt in range(0, 600, 50):
			s = slice(tt, tt + 25)
			print(f"   {tt:4d} {d['re'][i,tt]:+6.2f} {d['T'][i,tt,17]:+6.2f} {gx[tt]:+6.2f} | {1e3*d['I']['cmd'][i,s,0].mean():+7.2f} {1e3*d['I']['cmd'][i,s,2].mean():+7.2f} | {1e3*d['tau'][i,s,0].mean():+7.3f} {1e3*d['xt'][i,s,0].mean():+7.3f} {1e3*ext[s].mean():+7.3f}")
