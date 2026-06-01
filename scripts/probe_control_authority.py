"""Empirical probe: control authority vs episode horizon for the WNN drone sim.

Why this exists (01/06/2026): it quantifies the finding that refuted the
curriculum-on-steps design. Attitude is the double-integral of control torque,
so control authority scales ≈ t². At dt=0.001 a 10-30 ms episode yields almost
no authority vs the 5° stable threshold, so a do-nothing hover is statistically
indistinguishable from a perfect PID — the GA has no control-skill gradient.
The curriculum was switched to fix the horizon at a signal-bearing length
(250 ms) and grow initial-condition severity instead.

Run:
  python scripts/probe_control_authority.py
Expected (representative):
  - 10 ms : hover≈PID stable (~53% vs ~53%), reward gap ~2
  - 250 ms: reward gap PID−hover grows 2→154 across tilt 5→60°
  - 500 ms: PID 100% stable vs hover ~10%

CPU-only (sim physics); safe to run alongside GPU jobs.
"""

import math
import numpy as np

import ram_accelerator as r
from wnn.control.training import run_episode, EpisodeConfig, make_pid_action_fn, _euler_to_quat_xyz
from wnn.control.pid import AttitudePID, AttitudePIDConfig


def hover_fn(gyro, accel, target_rpy, q):
	"""Do-nothing baseline: equal thrust on all 4 motors (no attitude control)."""
	return (0.5, 0.5, 0.5, 0.5)


def authority_table():
	"""Max attitude change any policy can produce from a fixed 4° roll, per horizon."""
	target_q = _euler_to_quat_xyz(0.0, 0.0, 0.0)
	pid = AttitudePID(AttitudePIDConfig())

	def run(policy, steps, is_pid=False):
		s = r.AttitudeSim()
		s.reset(q=list(_euler_to_quat_xyz(math.radians(4.0), 0.0, 0.0)), omega=[0.0, 0.0, 0.0])
		if is_pid:
			pid.reset()
		last = float("nan")
		for i in range(steps):
			if s.is_unstable():
				break
			gyro, accel = s.read_imu()
			q = s.quaternion
			pwm = pid.step(q, gyro, (0.0, 0.0, 0.0)) if is_pid else policy(gyro, accel, None, q)
			s.step(list(pwm))
			last = math.degrees(s.attitude_error(target_q))
		return last

	print("Control authority from 4° roll (max attitude change any policy can make):")
	print(f"  {'horizon':>9} {'hover':>8} {'full-diff':>10} {'PID':>8} {'max spread':>11}")
	for steps in (10, 30, 100, 250, 500):
		h = run(hover_fn, steps)
		f = run(lambda g, a, t, q: (1.0, 0.0, 1.0, 0.0), steps)
		p = run(None, steps, is_pid=True)
		vals = [v for v in (h, f, p) if v == v]
		print(f"  {steps:>6}ms {h:>8.3f} {f:>10.3f} {p:>8.3f} {max(vals)-min(vals):>11.3f}")


def ic_curriculum_signal(steps=250, n=150):
	"""Reward + stable-rate, hover vs PID, across the IC-difficulty schedule."""
	pid = AttitudePID(AttitudePIDConfig())
	pid_fn = make_pid_action_fn(pid)

	def measure(action_fn, tilt_deg, body_rate, is_pid=False, base=7):
		s = r.AttitudeSim()
		cfg = EpisodeConfig(dt=0.001, steps_per_episode=steps,
		                    max_initial_tilt_rad=math.radians(tilt_deg),
		                    max_initial_yaw_rad=math.radians(min(tilt_deg, 45.0)),
		                    max_initial_body_rate=body_rate, max_initial_yaw_rate=body_rate * 0.6)
		rng = np.random.default_rng(base)
		rewards, stable = [], 0
		for _ in range(n):
			ep = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
			if is_pid:
				pid.reset()
			res = run_episode(action_fn, s, cfg, rng=ep)
			rewards.append(res.cumulative_reward)
			if (not res.diverged) and res.mean_attitude_error_rad <= math.radians(5.0):
				stable += 1
		return float(np.mean(rewards)), stable / n

	print(f"\nIC-difficulty signal at fixed {steps}-step ({steps}ms) horizon:")
	print(f"  {'tilt':>5} {'rate':>5} | {'hover rwd':>10} {'PID rwd':>9} {'gap':>8} | "
	      f"{'hover stbl':>10} {'PID stbl':>9}")
	for tilt, br in [(5, 0.5), (15, 1.0), (30, 2.0), (45, 3.0), (60, 4.0)]:
		hr, hs = measure(hover_fn, tilt, br)
		pr, ps = measure(pid_fn, tilt, br, is_pid=True)
		print(f"  {tilt:>4}° {br:>5.1f} | {hr:>10.1f} {pr:>9.1f} {pr-hr:>8.1f} | "
		      f"{hs*100:>9.1f}% {ps*100:>8.1f}%")


if __name__ == "__main__":
	authority_table()
	ic_curriculum_signal()
