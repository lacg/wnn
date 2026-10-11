"""IDS-25 stratified subset (compute cap ~2 h): per dataset, the first K SP100 runs by flow id
plus one SWEEP run per arm cycling arms (lowest seed first) up to K. Prints flow ids.
K: cicids 5+5, ciciot 5+5, unswr 8+8, unswt 8+8 (cost/flow measured: ~4 / 2.5 / 1 / 0.4 min)."""
import sys, os, re
sys.path.insert(0, os.path.dirname(__file__))
from common import connect, select_flows
K = {"cicids": 5, "ciciot": 5, "unswr": 8, "unswt": 8}
fl = select_flows(connect())
out = []
for ds, k in K.items():
	sp = sorted(f for f, x in fl.items() if x["ds"] == ds and x["fam"] == "SP100")[:k]
	sw = sorted((x["seed"], re.sub(r"-r\d+$", "", x["name"]), f) for f, x in fl.items() if x["ds"] == ds and x["fam"] == "SWEEP")
	arms, pick = {}, []
	for s, arm, f in sw: arms.setdefault(arm, []).append(f)
	i = 0
	while len(pick) < k and any(arms.values()):
		for arm in sorted(arms):
			if arms[arm] and len(pick) < k and (i == 0 or True):
				pick.append(arms[arm].pop(0))
		i += 1
	out += sp + pick
print(" ".join(map(str, out)))
