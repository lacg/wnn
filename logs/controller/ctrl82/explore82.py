import numpy as np, sys
sys.path.insert(0, "/Users/lacg/wnn/logs/controller/ctrl82"); from lib82 import *
H = set(map(tuple, np.load(HARD)))
D = {k: load(v) for k, v in TAGS.items()}
for k, d in D.items():
	I = inner_replay(d); print(k, "inner-replay gate max|pwm diff|", I["gate"]); d["I"] = I
	r, p, yw = euler(d["T"][..., 6:10]); d["re"] = np.degrees(r - d["tref"][..., 0]); d["pe"] = np.degrees(p - d["tref"][..., 1]); d["yw"] = np.degrees(yw)
