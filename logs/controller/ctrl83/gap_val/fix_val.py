"""Row-B fixtures on the VAL units of one l2bnmw seed, cell `all`: one dir per unit. usage: fix_val.py <cid> <outdir>"""
import os, sys
sys.path.insert(0, "/Users/lacg/wnn/logs/controller/ctrl83"); sys.path.insert(0, "/Users/lacg/wnn/scripts")
import cells as C, score_cell as S
import ctrl_indi_val_fixture as F
from wnn.control._accel import score_classical_baseline
from wnn.control.val_units import checked_val_unit_seeds
from wnn.seeds import resolve_seed_set
cid, out = sys.argv[1], sys.argv[2]
CELL = os.environ.get("CELL", "all")
_, _, a, ec0 = S.load_run(cid)
s = resolve_seed_set(base=a.base_seed, run_index=0, train=a.train_seed, test=a.test_seed, val=a.val_seed)
ec = C.apply(CELL, ec0)
for unit in checked_val_unit_seeds(a, s):
	assert not (99990101 <= unit <= 99990400)
	f = F._with_defaults(F._fields(a, ec, unit), score_classical_baseline)
	lines = [F._line("unit_seed", unit)] + [F._line(k, v) for k, v in f.items()]
	call = {k: v for k, v in f.items() if k not in ("init_qs", "init_omegas", "steps", "stable_deg")}
	r = score_classical_baseline(F.MPCOF, f["init_qs"], f["init_omegas"], f["steps"], f["stable_deg"], **call)
	lines.append(F._line("expect_est_row", [r[0], r[1], r[2], r[4]]))
	d = os.path.join(out, str(unit)); os.makedirs(d, exist_ok=True)
	open(os.path.join(d, f"unit_{unit}.txt"), "w").write("\n".join(lines) + "\n")
	print(unit, "est-fed A", r[0], r[1], flush=True)
