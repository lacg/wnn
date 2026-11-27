"""CTRL-82 VAL: per hard episode, where/when the excursion is, per arm (matched unit/episode)."""
exec(open("/Users/lacg/wnn/logs/controller/ctrl82/explore82.py").read())
G = D["G+T"]; hk = [i for i, k in enumerate(G["key"]) if k in H]
E = {k: np.load(f"/tmp/ctrl82/eps_{v}.npz")["eps"] for k, v in (("G+T", "gt"), ("lk90", "lk90"), ("l2b", "l2b"))}
em = {k: {(int(u), int(e)): x for u, e, x in E[k]} for k in E}
print("VAL unit ep | whole err G/l90/l2b | per arm: max|roll e|@ms  max|pitch e|@ms  max|yaw|  mixer-sat ticks")
for i in sorted(hk, key=lambda i: -em["G+T"][G["key"][i]]):
	s = f"u{G['key'][i][0]%1000} ep{G['key'][i][1]:<3d}| " + "/".join(f"{em[k][G['key'][i]]:.2f}" for k in D) + " |"
	for k, d in D.items():
		re, pe, yw = d["re"][i], d["pe"][i], d["yw"][i]
		s += f" {abs(re).max():5.1f}@{abs(re).argmax():4d} {abs(pe).max():5.1f}@{abs(pe).argmax():4d} y{abs(yw).max():4.1f} s{d['I']['sat'][i].sum():3d} |"
	print(s)
