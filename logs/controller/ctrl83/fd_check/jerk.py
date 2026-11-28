import sys, dataclasses as dc, numpy as np
sys.path.insert(0, "/Users/lacg/wnn/logs/controller/ctrl83"); sys.path.insert(0, "/Users/lacg/wnn/scripts"); sys.path.insert(0, "/Users/lacg/wnn/logs/controller/ctrl80")
import os; os.environ["WNN_CONTROLLER_GPU_EVAL"]="0"
import cells as C, score_cell as S
_,_,a,ec0 = S.load_run("wnn_s1")
rv = [r for r in C.RIVALS if r[0]=="mpcofC"][0]
for Tp,Td in ((0.3,0.2),(0.3,0.3),(0.2,0.1),(0.2,0.2)):
	ec = dc.replace(ec0, motor_lag_s=Tp, model_motor_lag_s=Td)
	rows=[S.rival_call(a, ec, u, rv)[0] for u in S.UNITS[:10]]
	ks=[k for k,v in rows[0].items() if isinstance(v,(int,float))]
	print(Tp,Td, {k: round(float(np.mean([r[k] for r in rows])),4) for k in ks})
