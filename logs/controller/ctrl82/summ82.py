"""CTRL-82 VAL (10,000 eps = 100 VAL units x 100, per arm): is the roll-tail benefit of the yaw leak real under NEUTRAL
selection (not selected on G+T's own bad luck)? Paired per-episode."""
import numpy as np
S = {k: np.load(f"/tmp/ctrl82/summ_{v}.npz")["summ"] for k, v in (("G+T", "gt"), ("lk90", "lk90"), ("l2b", "l2b"))}
assert all((S[k][:, :2] == S["G+T"][:, :2]).all() for k in S)
for k in S: print(k, "gate mismatches > 0.01 deg:", int((abs(S[k][:, 2] - S[k][:, 3]) > 0.01).sum()), "of", len(S[k]))
C = dict(err=3, roll_pre=8, pitch_pre=9, yaw_pre=10, roll_hold=11, pitch_hold=12, yaw_hold=13, roll_max=14, pitch_max=15)
print("\n== all 10,000 VAL eps: mean per arm")
print("   " + " ".join(f"{c:>10s}" for c in C))
for k, s in S.items(): print(f"{k:5s}" + " ".join(f"{s[:, i].mean():10.3f}" for i in C.values()))
rm = {k: s[:, 14] for k, s in S.items()}
sels = {"top1% by G+T roll_max": np.argsort(rm["G+T"])[::-1][:100], "top1% by lk90 roll_max": np.argsort(rm["lk90"])[::-1][:100],
        "top1% by l2b roll_max": np.argsort(rm["l2b"])[::-1][:100], "top1% by 3-arm MEAN roll_max": np.argsort(sum(rm.values()))[::-1][:100],
        "top1% by |roll0| (initial condition)": np.argsort(abs(S["G+T"][:, 4]))[::-1][:100]}
print("\n== roll-hard subsets (100 eps each): mean roll_max / roll pre-hold / pitch pre-hold / yaw pre-hold / whole err, per arm; paired lk90-G+T and l2b-G+T roll_max (wins = eps where the arm is lower)")
for lab, ix in sels.items():
	print(f"  {lab}")
	for k, s in S.items():
		print(f"     {k:5s} {s[ix,14].mean():6.2f} / {s[ix,8].mean():5.2f} / {s[ix,9].mean():5.2f} / {s[ix,10].mean():5.2f} / {s[ix,3].mean():5.2f}")
	for k in ("lk90", "l2b"):
		dd = S[k][ix, 14] - S["G+T"][ix, 14]
		print(f"     {k}-G+T roll_max: mean {dd.mean():+.2f} deg, median {np.median(dd):+.2f}, {k} lower in {(dd<0).sum()}/100")
