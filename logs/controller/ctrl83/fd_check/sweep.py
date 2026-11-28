import sys, os, subprocess, re, dataclasses as dc
sys.path.insert(0, "/Users/lacg/wnn/logs/controller/ctrl83"); sys.path.insert(0, "/Users/lacg/wnn/scripts")
import cells as C, rowb as R
plant, design = float(sys.argv[1]), float(sys.argv[2])
out = f"/tmp/ctrl83_fd/fix_p{plant}_d{design}"
import ctrl_indi_val_fixture as F
from score_cell import load_run
from wnn.control._accel import score_classical_baseline
_, _, a, ec0 = load_run("wnn_s1")
ec = dc.replace(ec0, motor_lag_s=plant, model_motor_lag_s=design)
os.makedirs(out, exist_ok=True)
for i, unit in enumerate(int(u) for u in a.test_seeds):
	f = F._with_defaults(F._fields(a, ec, unit), score_classical_baseline)
	lines = [F._line("unit_seed", unit)] + [F._line(k, v) for k, v in f.items()]
	if i == 0:
		call = {k: v for k, v in f.items() if k not in ("init_qs", "init_omegas", "steps", "stable_deg")}
		r = score_classical_baseline(F.MPCOF, f["init_qs"], f["init_omegas"], f["steps"], f["stable_deg"], **call)
		lines.append(F._line("expect_est_row", [r[0], r[1], r[2], r[4]]))
	open(os.path.join(out, f"unit_{unit}.txt"), "w").write("\n".join(lines) + "\n")
env = {**os.environ, "PYO3_PYTHON": R.PY, "WNN_RIVAL_RESCORE_FIXTURE": out, "RAYON_NUM_THREADS": "6"}
cmd = ["nice", "-n", "15", "cargo", "test", "--release", "-p", "ram_controller", "--lib", "--no-default-features", R.TEST, "--", "--ignored", "--nocapture", "--test-threads", "1"]
p = subprocess.run(cmd, cwd=R.ACC, env=env, capture_output=True, text=True)
log = p.stdout + p.stderr
open(out + ".log", "w").write(log)
for m in R.RX.finditer(log):
	print(f"plant {plant} design {design} row {m.group(1)} stable {m.group(2)} err {m.group(3)} steady {m.group(4)} pos {m.group(5)} fails {m.group(6)}")
if p.returncode: print("RC", p.returncode, log[-1500:])
