#!/usr/bin/env python3
"""W1 drift-mode analysis — WHAT drifts past the training horizon?

Hypothesis (docs/controller_research_roadmap.md W1.2): the horizon drift is a YAW
random-walk. The WNN observes gyro+accel only — absolute yaw is unobservable
(gravity is yaw-invariant), while the PID reads the true quaternion. If late-window
error is yaw-dominated with tight roll/pitch, then (a) drift is explained,
(b) committee drift-cancellation is explained (uncorrelated per-member yaw walks),
(c) the fix is the yaw-anchor channel (ANCH2K).

Rolls each subject controller for HORIZON steps x EPISODES fresh-IC episodes and
decomposes attitude error into |roll|, |pitch|, |yaw| per time bucket. Analysis
layer: the per-step loop is Python over the Rust sim/controller (~200K FFI calls,
seconds) — a one-off diagnostic, not a hot path; the committee/scoring hot loops
live in Rust (eval_ensemble_closed_loop).

Usage: PYTHONPATH=src/wnn /Users/lacg/wnn-venv/bin/python scripts/w1_drift_analysis.py
"""

import math

import numpy as np

from wnn.control.checkpoint_io import load_controller_checkpoint
from wnn.control.evaluator import (
	build_controller, controller_genome_from_arch, fit_thresholds_from_pid_rollouts,
)
from wnn.control.sim import AttitudeSim
from wnn.control.training import _sample_initial_state

HORIZON = 10000
EPISODES = 8
BUCKETS = 10
FRESH_SEED = 77770001
SUBJECTS = [
	("A_ctrl_s09 (trained@500)", "logs/controller/StateIntegral_20260701/A_ctrl_seed20260609/winner.yaml.gz"),
	("LONG_s09 (trained@2000)", "logs/controller/E2Reliability_20260702/LONG_seed20260609/winner.yaml.gz"),
]


def quat_to_euler_xyz(q):
	"""(w,x,y,z) -> roll, pitch, yaw (Tait-Bryan xyz), radians."""
	w, x, y, z = q
	roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
	s = 2 * (w * y - z * x)
	pitch = math.asin(max(-1.0, min(1.0, s)))
	yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
	return roll, pitch, yaw


def episode_ics(seed, n):
	"""Exact eval_closed_loop_reset numpy chain (fresh-seed protocol ICs)."""
	rng = np.random.default_rng(seed)
	out = []
	for _ in range(n):
		ep_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
		q, om = _sample_initial_state(
			ep_rng, math.radians(5.0), math.radians(5.0), 0.5, 0.3)
		out.append((list(q), list(om)))
	return out


def run_subject(label, path):
	payload = load_controller_checkpoint(path)
	spec, genome = payload["spec"], payload["best_genome"]
	thr = fit_thresholds_from_pid_rollouts(spec, num_episodes=10, seed=FRESH_SEED)
	sim = AttitudeSim()
	bucket_len = HORIZON // BUCKETS
	# accumulators [bucket] for |roll|, |pitch|, |yaw| (deg) + diverged count
	acc = np.zeros((BUCKETS, 3))
	cnt = np.zeros(BUCKETS)
	diverged = 0
	for (q0, om0) in episode_ics(FRESH_SEED, EPISODES):
		c = build_controller(controller_genome_from_arch(genome, spec, thr))
		c.reset()
		sim.reset(q=q0, omega=om0)
		for t in range(HORIZON):
			if sim.is_unstable():
				diverged += 1
				break
			gyro, accel = sim.read_imu()
			pwm = c.step(list(gyro), list(accel), [0.0, 0.0, 0.0])
			sim.step(list(pwm))
			r, p, y = quat_to_euler_xyz(sim.quaternion)
			b = min(t // bucket_len, BUCKETS - 1)
			acc[b, 0] += abs(math.degrees(r))
			acc[b, 1] += abs(math.degrees(p))
			acc[b, 2] += abs(math.degrees(y))
			cnt[b] += 1
		del c
	print(f"\n===== {label} — {EPISODES} eps × {HORIZON} steps, {BUCKETS} buckets =====")
	print(f"diverged episodes: {diverged}/{EPISODES}")
	print(f"{'bucket (steps)':<18} {'|roll|°':>8} {'|pitch|°':>9} {'|yaw|°':>8}   {'yaw share':>9}")
	for b in range(BUCKETS):
		if cnt[b] == 0:
			print(f"{b*bucket_len:>6}-{(b+1)*bucket_len:<10} (no surviving steps)")
			continue
		r, p, y = acc[b] / cnt[b]
		share = y / max(r + p + y, 1e-9) * 100.0
		print(f"{b*bucket_len:>6}-{(b+1)*bucket_len:<10} {r:8.2f} {p:9.2f} {y:8.2f}   {share:8.1f}%")
	# Verdict helper: late-window (last 3 buckets) yaw share
	late = acc[-3:].sum(axis=0) / max(cnt[-3:].sum(), 1e-9)
	share = late[2] / max(late.sum(), 1e-9) * 100.0
	print(f"LATE-WINDOW (last 30%): |roll| {late[0]:.2f}°  |pitch| {late[1]:.2f}°  |yaw| {late[2]:.2f}°"
	      f"  → yaw share {share:.1f}%")
	return share


def main():
	print("========== W1 drift-mode decomposition — is the drift a yaw random-walk? ==========")
	shares = {}
	for label, path in SUBJECTS:
		shares[label] = run_subject(label, path)
	print("\nVERDICT: yaw share of late-window error per subject:")
	for label, s in shares.items():
		print(f"  {label:<28} {s:5.1f}%  {'← YAW-DOMINATED (hypothesis holds)' if s > 60 else '← mixed/attitude drift'}")


if __name__ == "__main__":
	main()
