"""D0 probe: does the teacher's hover anchor change the DAgger LABEL it emits?

Solo, oracle-fed teacher rollouts on the airframe plant, same episodes across variants.
Three variants per teacher (spec docs/multi_axis_programme_spec.md §0.9(e)):
  legacy   teacher hover 0.5, its pwm applied as-is           (attitude-only history)
  today    teacher hover 0.5, applied = pwm + (h_nom - 0.5)   (what DAgger does under
           --translation: the student's accumulator sits at the true hover while the
           teacher was linearized and mixed at 0.5; observe() sees the applied pwm)
  fixed    teacher hover h_nom, applied as-is                 (the D0 fix)
The label the trainer would build is the teacher-relative deviation p - h_teacher on
the 1/L grid; we report its dead-zone share, mean magnitude, saturation share at
s in {1,2,4,8}, and each variant's own attitude error, so the today/fixed ratio of
mean |deviation| is the number the spec calls MATERIAL if outside [0.8, 1.25].

Usage:
  PYTHONPATH=src/wnn nice -n 19 python scripts/hover_anchor_probe.py --episodes 20
"""

import argparse
import math
import sys

import numpy as np

sys.path.insert(0, 'scripts')
from teacher_step_histogram import Recipe, build_episode_config  # noqa: E402


def parse_args():
	ap = argparse.ArgumentParser(description='D0 hover-anchor label probe')
	ap.add_argument('--episodes', type=int, default=20)
	ap.add_argument('--seed', type=int, default=911)
	ap.add_argument('--levels', type=int, default=64, help='label grid per motor (anchor L=64)')
	ap.add_argument('--teacher', action='append', default=None, help='mpcof|lqr; repeatable')
	return ap.parse_args()


def hover_nominal(af) -> float:
	"""pwm at which four motors carry the weight: 4*k*h^2 = m*g."""
	return math.sqrt(af.mass * af.gravity / (4.0 * af.k_thrust))


def build_teacher(name: str, ec, hover: float):
	from wnn.control._accel import AttitudeLqrRs, AttitudeMpcOfRs
	af = ec.airframe
	plant = dict(dt=float(ec.dt), arm_length=float(af.arm_length), k_thrust=float(af.k_thrust),
	             k_drag=float(af.k_drag), inertia=[float(x) for x in af.inertia],
	             gravity=float(af.gravity), hover=float(hover))
	if name == 'mpcof':
		return AttitudeMpcOfRs(**plant), True
	if name == 'lqr':
		return AttitudeLqrRs(**plant), False
	raise ValueError(name)


def rollout(teacher, has_observer, ec, rng, shift: float):
	"""One solo episode. Returns (teacher_pwm[steps,4], tilt_deg[steps], survived)."""
	from wnn.control._accel import AttitudeSim
	from wnn.control.training import _sample_initial_state, apply_disturbance
	sim = AttitudeSim(**ec.sim_kwargs())
	q0, w0 = _sample_initial_state(rng, ec.max_initial_tilt_rad, ec.max_initial_yaw_rad,
	                               ec.max_initial_body_rate, ec.max_initial_yaw_rate)
	sim.reset(q=list(q0), omega=list(w0))
	apply_disturbance(sim, ec.disturbance, rng)
	teacher.reset()
	n = ec.steps_per_episode
	pwm_t = np.zeros((n, 4)); tilt = np.zeros(n)
	for k in range(n):
		if sim.is_unstable():
			return pwm_t[:k], tilt[:k], False
		gyro, _ = sim.read_imu()
		p = np.asarray(teacher.step(list(sim.quaternion), list(gyro), [0.0, 0.0, 0.0]), float)
		applied = np.clip(p + shift, 0.0, 1.0)
		if has_observer:
			teacher.observe_py([float(g) for g in gyro], [float(a) for a in applied])
		sim.step([float(a) for a in applied])
		pwm_t[k] = p
		q = sim.quaternion
		# tilt = angle between body z and world z, from the quaternion
		w, x, y, z = q
		cz = 1.0 - 2.0 * (x * x + y * y)
		tilt[k] = math.degrees(math.acos(max(-1.0, min(1.0, cz))))
	return pwm_t, tilt, True


def summarise(dev: np.ndarray, levels: int, tilt: np.ndarray, survived: list):
	"""dev = teacher-relative deviation p - h_teacher, shape (steps, 4)."""
	lv = np.floor(dev * levels)
	out = {
		'dead_all': float(np.mean(lv == 0)),
		'dead_m13': float(np.mean(lv[:, [0, 2]] == 0)),
		'mean_abs_dev': float(np.mean(np.abs(dev))),
		'mean_abs_lv': float(np.mean(np.abs(lv))),
		'tilt_deg': float(np.mean(tilt)),
		'survived': float(np.mean(survived)),
	}
	for s in (1, 2, 4, 8):
		out['sat_s%d' % s] = float(np.mean(np.abs(dev) * s >= 0.5))
	return out


def run_variant(name, ec, hover, shift, episodes, seed, levels):
	teacher, has_obs = build_teacher(name, ec, hover)
	rng = np.random.default_rng(seed)
	devs, tilts, surv = [], [], []
	for _ in range(episodes):
		ep = np.random.default_rng(int(rng.integers(0, 2 ** 32 - 1)))
		p, t, ok = rollout(teacher, has_obs, ec, ep, shift)
		devs.append(p - hover); tilts.append(t); surv.append(ok)
	return summarise(np.concatenate(devs), levels, np.concatenate(tilts), surv)


def main():
	a = parse_args()
	teachers = a.teacher or ['mpcof', 'lqr']
	r = Recipe()
	ec = build_episode_config(r, a.seed)
	h = hover_nominal(ec.airframe)
	print('airframe %s  h_nom = sqrt(m g / 4k) = %.4f  (m=%.4f kg, k=%.3f N/pwm^2)'
	      % (ec.airframe.name, h, ec.airframe.mass, ec.airframe.k_thrust))
	print('disturbance %s, %d episodes x %d steps, label grid L=%d, tilt %.1f deg'
	      % (r.disturbance, a.episodes, r.steps, a.levels, r.tilt_deg))
	variants = [('legacy', 0.5, 0.0), ('today', 0.5, h - 0.5), ('fixed', h, 0.0)]
	cols = ['dead_all', 'dead_m13', 'mean_abs_dev', 'mean_abs_lv', 'sat_s1', 'sat_s2', 'sat_s4', 'sat_s8', 'tilt_deg', 'survived']
	for name in teachers:
		print('\n=== teacher %s ===' % name)
		print('  %-7s %-7s' % ('variant', 'hover') + ''.join('%13s' % c for c in cols))
		res = {}
		for vname, hover, shift in variants:
			res[vname] = run_variant(name, ec, hover, shift, a.episodes, a.seed, a.levels)
			print('  %-7s %-7.4f' % (vname, hover) + ''.join('%13.4f' % res[vname][c] for c in cols))
		ratio = res['today']['mean_abs_dev'] / max(res['fixed']['mean_abs_dev'], 1e-12)
		dz = (res['today']['dead_m13'] - res['fixed']['dead_m13']) * 100
		flag = 'MATERIAL' if (ratio < 0.8 or ratio > 1.25 or abs(dz) > 15) else 'immaterial'
		print('  today/fixed mean|dev| ratio = %.3f   motor-1/3 dead-zone shift = %+.1f pp   -> %s'
		      % (ratio, dz, flag))


if __name__ == '__main__':
	main()
