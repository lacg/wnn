"""CTRL-80 VAL per-tick trace (diagnosis only, never a result): ctrl73/trace_val.py plus the student's own
raw input features on every tick (get_last_feature_vector through a recording proxy; the wheel is untouched).
Usage: trace80.py <tag> <out.npz> <n_val_units>
  trace  ctrl_gap_student_trace layout: 0 unit_idx,1 ep,2 t,3-5 gyro,6-9 q_true,10 x,11 y,12 vx,13 vy,14 alt,15 vz,
         16-19 sp[T,p,q,r] (held CTBR setpoint),20-23 pwm,24 coll,25 talt
  feat   one row per trace row: the raw (pre-threshold) features named in `fnames`
  thr    the fitted thresholds of those features (8 cuts each), `meta` = [levels_per_bank, delta_max_yaw, leak, r_range]"""
import math, os, sys
sys.path.insert(0, "/Users/lacg/wnn/scripts")
import numpy as np
import ctrl_gap_student_trace as T
tag, out, nunits = sys.argv[1], sys.argv[2], int(sys.argv[3])
import wnn.control._accel as A
import ram_controller as RC
import wnn.control.phased_ga as PG
from wnn.control.evaluator import ControllerEvaluator
from wnn.control.feature_layout import feature_names
from wnn.seeds import resolve_seed_set
WANT = ("gyro_x", "gyro_y", "gyro_z", "roll_err", "pitch_err", "yaw_err", "yaw_err_i")


class Rec:
	"""Delegating proxy: records the wanted raw features after every step()."""
	def __init__(self, c, idx, nfeat):
		self._c, self._idx, self._nfeat, self.buf = c, idx, nfeat, []

	def __getattr__(self, name):
		return getattr(self._c, name)

	def step(self, gy, ac, tgt):
		pwm = self._c.step(gy, ac, tgt)
		fv = self._c.get_last_feature_vector()
		assert len(fv) == self._nfeat, (len(fv), self._nfeat)
		self.buf.append([fv[i] for i in self._idx])
		return pwm


txt, a, ec = T.run_args(tag)
seeds = resolve_seed_set(base=a.base_seed, run_index=0, train=a.train_seed, test=a.test_seed, val=a.val_seed)
pay = PG._ctl_load(T.D + tag + "_winner.yaml.gz"); spec, genome = pay["spec"], pay["best_genome"]
units = T.val_units(tag)[:nunits]
names = list(feature_names(spec)); idx = [names.index(n) for n in WANT]
cap = {}; real = A.score_controllers_cpu
def wrap(controllers, *aa, **kk):
	cap.setdefault("calls", []).append((controllers[0], aa, kk)); return real(controllers, *aa, **kk)
A.score_controllers_cpu = wrap
thr = PG._test_thresholds(a, ec, spec, units[0], seeds.train, use_score=True)
bpf = len(thr) // len(names); TH = np.array([thr[i * bpf:(i + 1) * bpf] for i in idx], dtype=float)
Tr, E, F, meta = [], [], [], None
for ui, unit in enumerate(units):
	ev = ControllerEvaluator(spec, num_eval_episodes=a.test_episodes, seed=unit, episode_config=ec, thresholds=thr,
	                         rg_config=PG._rg_config(a, ec, unit), max_train_workers=1, num_eval_folds=a.num_eval_folds, max_eval_workers_gpu=False)
	cap["calls"] = []
	m = ev.score_genomes([genome])[0]
	c, aa, k = cap["calls"][0]; k = {**{"dt": 0.001, "dist_seed": 0}, **k}
	rec = Rec(c, idx, len(names))
	rows, eps, tr = T.replay(RC, rec, aa[0], aa[1], aa[2], aa[3], k, ui, aa[2])
	rs = math.degrees(float(np.nanmean(eps[:, 2])))
	print(f"unit {unit}: scorer err {m.mean_attitude_error_deg:.3f} steady {m.mean_steady_error_deg:.4f} | replay steady {rs:.4f} err {math.degrees(np.mean(eps[:,3])):.3f}", flush=True)
	assert abs(rs - m.mean_steady_error_deg) < 0.005
	f = np.array(rec.buf, dtype=np.float32); n = len(f)
	assert n <= len(tr), (n, len(tr))
	Tr.append(tr[:n]); E.append(eps); F.append(f)
	sp = c.ctbr_spec
	meta = [c.levels_per_motor, c.delta_max_yaw or float("nan"), a.delta_leak, float(sp.range[3]) if hasattr(sp, "range") else float("nan")]
np.savez(out, trace=np.vstack(Tr), feat=np.vstack(F), eps=np.vstack(E), units=np.array(units), fnames=np.array(WANT), thr=TH, meta=np.array(meta, dtype=float))
print("saved", out, np.vstack(Tr).shape, meta, flush=True)
