"""Export pose traces for the 3D policy-comparison visual: the WNN winner vs
all five teachers (pid, lqr, mpc, lqi, mpcof) flying IDENTICAL episodes.

Every policy flies its own AttitudeSim (same airframe, same initial conditions,
clean plant) so the ONLY thing that differs between traces is the policy —
same discipline as the transfer harness. Per recorded step we keep the
quaternion, the 4 motor PWMs and the attitude error, downsampled to 100 Hz.

Honesty notes baked into the output meta:
  - teachers are STATE-FEEDBACK (they read the true quaternion, exactly as
    they do as DAgger experts); the WNN flies on IMU only.
  - the WNN genome is final_population[0], the published-row convention
    (project_checkpoint_best_vs_pop0) — best_genome fallback is labeled.
  - clean plant (no disturbance): this is an illustration, not a claim; the
    banked numbers remain the held-out reports.

Usage:
  python scripts/viz_pose_export.py \
      --winner logs/controller/l1refly/L1R_..._winner.yaml.gz \
      --airframe cf21_brushless --out /tmp/poses.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from wnn.control._accel import (
	AttitudeLqiRs,
	AttitudeLqrRs,
	AttitudeMpcOfRs,
	AttitudeMpcRs,
	AttitudeSim,
)
from wnn.control.airframe import Airframe
from wnn.control.checkpoint_io import load_controller_checkpoint
from wnn.control.evaluator import build_controller
from wnn.control.pid_firmware import AttitudePidFirmware

STEPS = 2000          # 2 s @ 1 kHz — the training episode length
SAMPLE_EVERY = 10     # record at 100 Hz
STABLE_DEG = 5.0      # episode is "stable" when its mean error is under this
TILT_DEG = 5.0        # the production regime: recipes fly --tilt 5.0

NUM_EPISODES = 3
IC_SEED = 99990101    # first report seed — ICs drawn by the CANONICAL sampler

# Episodes are drawn by training.sample_ics_flat (the single source of truth
# for IC draw order) inside the operating regime the winner was trained and
# calibrated on (tilt ≤ 5°, body rate ≤ 0.5 rad/s, yaw rate ≤ 0.3), and FLY
# UNDER the run's disturbance preset — the banked triple was measured under
# L4C, and the winner's thermometer/memory were calibrated on that noise
# (a clean plant measurably degrades it; see the 13/08 A/B).


class TeacherPolicy:
	"""Uniform reset/act over the firmware PID and the Rust optimal teachers.

	mpcof is offset-free ONLY if told the applied action each step
	(observe_py) — same contract _ObserverExpert enforces for DAgger."""

	def __init__(self, inner, has_observer: bool):
		self._inner = inner
		self._has_observer = has_observer
		self._last_applied = [0.5, 0.5, 0.5, 0.5]

	def reset(self) -> None:
		self._inner.reset()
		self._last_applied = [0.5, 0.5, 0.5, 0.5]

	def act(self, q, gyro, accel, target) -> list[float]:
		if self._has_observer:
			self._inner.observe_py([float(g) for g in gyro], self._last_applied)
		out = [float(p) for p in self._inner.step(list(q), list(gyro), list(target))]
		self._last_applied = out
		return out


class WnnPolicy:
	"""The WNN flies on IMU alone — it never sees the true quaternion. The ONE
	exception, same as the scorers': reset(init_yaw) seeds the yaw-heading
	dead-reckoning with the episode's TRUE initial yaw (the yaw anchor). A 0.0
	seed on a yaw≠0 episode makes the controller fight a phantom yaw error —
	the 13/08 viz debugging found exactly that (4.1° vs the scorer's 1.1°)."""

	def __init__(self, controller):
		self._ctl = controller

	def reset(self, init_yaw: float = 0.0) -> None:
		self._ctl.reset(init_yaw)

	def act(self, q, gyro, accel, target) -> list[float]:
		return [float(p) for p in self._ctl.step(list(gyro), list(accel), list(target))]


def _materialize_wnn(payload: dict, af: Airframe, train_base_seed: int,
                     disturbance: str):
	"""pop[0] (the published-row genome) → a buildable, threshold-fitted
	controller. Mirrors scripts/rescore_winners.py: thresholds are fit ONCE from
	PID rollouts on the run's own train seed, ON THE RUN'S OPERATING REGIME
	(tilt 5°, the run's disturbance preset, the run's airframe) — fitting on any
	other regime mis-scales the thermometer (project_thermometer_regime_mismatch)."""
	from wnn.control.evaluator import (
		controller_genome_from_arch, fit_thresholds_from_pid_rollouts)
	from wnn.control.training import DisturbanceConfig, EpisodeConfig
	from wnn.seeds import resolve_seed_set

	spec = payload["spec"]
	pop = payload.get("population") or []
	genome = pop[0] if pop else payload.get("best_genome")
	label = "final_population[0]" if pop else "best_genome (no population in checkpoint)"
	if getattr(genome, "spec", None) is None:
		ec = EpisodeConfig(
			dt=0.001, steps_per_episode=STEPS,
			max_initial_tilt_rad=math.radians(TILT_DEG),
			max_initial_yaw_rad=math.radians(TILT_DEG),
			max_initial_body_rate=0.5, max_initial_yaw_rate=0.3,
			disturbance=DisturbanceConfig.preset(disturbance, seed=911),
			airframe=af)
		train_seed = resolve_seed_set(base=train_base_seed, run_index=0).train
		thresholds = fit_thresholds_from_pid_rollouts(
			spec, num_episodes=10, seed=train_seed, episode_config=ec)
		genome = controller_genome_from_arch(genome, spec, thresholds)
	return build_controller(genome), label


def make_policies(af: Airframe, winner_path: str, train_base_seed: int,
                  disturbance: str) -> tuple[dict, str]:
	"""All six policies on the SAME airframe. Returns (policies, genome_label)."""
	plant = dict(
		dt=0.001, arm_length=float(af.arm_length), k_thrust=float(af.k_thrust),
		k_drag=float(af.k_drag), inertia=[float(x) for x in af.inertia],
		gravity=float(af.gravity))
	payload = load_controller_checkpoint(winner_path)
	if payload is None:
		raise FileNotFoundError(winner_path)
	wnn, genome_label = _materialize_wnn(payload, af, train_base_seed, disturbance)
	policies = {
		"wnn": WnnPolicy(wnn),
		"pid": TeacherPolicy(AttitudePidFirmware(af, af.gains()), False),
		"lqr": TeacherPolicy(AttitudeLqrRs(**plant), False),
		"mpc": TeacherPolicy(AttitudeMpcRs(**plant), False),
		"lqi": TeacherPolicy(AttitudeLqiRs(**plant), False),
		"mpcof": TeacherPolicy(AttitudeMpcOfRs(**plant), True),
	}
	return policies, genome_label


def rollout(policy, af: Airframe, q0: list, omega0: list, dist_cfg,
            dseed: int, asym: tuple, ep_idx: int) -> dict:
	"""One episode on a FRESH sim under the run's disturbance regime, applied
	EXACTLY as the batch scorers do: stream seed = dist.seed XOR score seed
	(disturbance_stream), motor asym resolved ONCE per pass from that stream,
	per-episode seed via disturbance_episode_seed(dseed, ep). Same ep_idx ⇒
	same weather for every policy. Records q, pwm, err at 100 Hz."""
	sim = AttitudeSim(
		dt=0.001, arm_length=float(af.arm_length), k_thrust=float(af.k_thrust),
		k_drag=float(af.k_drag), inertia=[float(x) for x in af.inertia],
		gravity=float(af.gravity))
	sim.reset(list(q0), list(omega0))
	if dist_cfg is not None:
		from wnn.control._accel import disturbance_episode_seed
		sim.set_disturbance(
			tau_bias=[float(x) for x in dist_cfg.tau_bias],
			gust_sigma=float(dist_cfg.gust_sigma),
			gust_tau_c=float(dist_cfg.gust_tau_c),
			motor_asym=[float(x) for x in asym],
			gyro_sigma=float(dist_cfg.gyro_sigma),
			gyro_bias_walk=float(dist_cfg.gyro_bias_walk),
			accel_sigma=float(dist_cfg.accel_sigma),
			seed=disturbance_episode_seed(dseed, ep_idx),
			dropout_prob=float(dist_cfg.dropout_prob),
			dropout_len_steps=int(dist_cfg.dropout_len_steps),
			obs_delay_steps=int(dist_cfg.obs_delay_steps),
			torque_scale_jitter=float(dist_cfg.torque_scale_jitter))
	# Yaw anchor: the episode's TRUE initial yaw (ZYX from the quaternion) —
	# only WnnPolicy consumes it; teacher resets ignore the argument.
	w, x, y, z = (float(v) for v in q0)
	init_yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
	try:
		policy.reset(init_yaw)
	except TypeError:
		policy.reset()
	target = [0.0, 0.0, 0.0]
	qs, pwms, errs = [], [], []
	err_sum, tail_sum, tail_n = 0.0, 0.0, 0
	tail_start = int(STEPS * 0.80)
	diverged = False
	for t in range(STEPS):
		if sim.is_unstable():
			diverged = True
			break
		gyro, accel = sim.read_imu()
		q = sim.quaternion
		pwm = policy.act(q, gyro, accel, target)
		sim.step(pwm)
		err = float(sim.attitude_error(None))
		err_sum += err
		if t >= tail_start:
			tail_sum += err
			tail_n += 1
		if t % SAMPLE_EVERY == 0:
			qs.append([round(float(v), 5) for v in q])
			pwms.append([round(float(p), 4) for p in pwm])
			errs.append(round(math.degrees(err), 3))
	steps_done = t + 1 if not diverged else t
	mean_err = math.degrees(err_sum / max(steps_done, 1))
	steady = math.degrees(tail_sum / tail_n) if tail_n else float("nan")
	return {
		"q": qs, "pwm": pwms, "err_deg": errs, "diverged": diverged,
		"mean_err_deg": round(mean_err, 3),
		"steady_deg": round(steady, 3) if tail_n else None,
		"stable": (not diverged) and mean_err <= STABLE_DEG,
	}


def main() -> int:
	ap = argparse.ArgumentParser(description=__doc__)
	ap.add_argument("--winner", required=True, help="winner .yaml.gz checkpoint")
	ap.add_argument("--airframe", default="cf21_brushless")
	ap.add_argument("--disturbance", default="L4C",
	                help="the RUN's disturbance preset (threshold-fit regime only; "
	                     "the viz episodes themselves fly clean)")
	ap.add_argument("--train-base-seed", type=int, default=None,
	                help="run's base seed (default: parsed from the s<digits> tag)")
	ap.add_argument("--out", required=True, help="output JSON path")
	args = ap.parse_args()

	base_seed = args.train_base_seed
	if base_seed is None:
		import re
		m = re.search(r"_s(\d+)_winner", Path(args.winner).name)
		if not m:
			ap.error("cannot parse train seed from filename; pass --train-base-seed")
		base_seed = int(m.group(1))

	af = Airframe.preset(args.airframe)
	policies, genome_label = make_policies(af, args.winner, base_seed, args.disturbance)

	import numpy as np
	from wnn.control.training import (
		DisturbanceConfig, EpisodeConfig, sample_ics_flat)
	ec = EpisodeConfig(
		dt=0.001, steps_per_episode=STEPS,
		max_initial_tilt_rad=math.radians(TILT_DEG),
		max_initial_yaw_rad=math.radians(TILT_DEG),
		max_initial_body_rate=0.5, max_initial_yaw_rate=0.3)
	q0_flat, om0_flat = sample_ics_flat(IC_SEED, NUM_EPISODES, ec)
	ics = [(q0_flat[i * 4:(i + 1) * 4], om0_flat[i * 3:(i + 1) * 3])
		for i in range(NUM_EPISODES)]
	dist_cfg = DisturbanceConfig.preset(args.disturbance, seed=911)

	def _tilt_deg(q):
		return round(math.degrees(2.0 * math.acos(min(1.0, abs(q[0])))), 1)

	out = {
		"meta": {
			"airframe": args.airframe,
			"winner": str(Path(args.winner).name),
			"genome": genome_label,
			"dt_sampled_s": 0.001 * SAMPLE_EVERY,
			"steps": STEPS,
			"stable_deg": STABLE_DEG,
			"plant": f"{args.disturbance} disturbance — the regime the banked triple "
				"was measured under (identical per-episode streams across policies)",
			"observability": "teachers read the true quaternion (state feedback, "
				"as when they act as DAgger experts); the WNN flies on IMU only",
			"ic_sampler": f"training.sample_ics_flat(seed={IC_SEED}) — canonical draw order",
		},
		"episodes": [
			{"name": f"run {i + 1}: tilt {_tilt_deg(q)}°, |ω| "
				f"{round(math.sqrt(sum(w * w for w in om)), 2)} rad/s"}
			for i, (q, om) in enumerate(ics)],
		"policies": {},
	}
	from wnn.control.evaluator import disturbance_stream
	dseed, asym = disturbance_stream(dist_cfg, IC_SEED)
	for name, pol in policies.items():
		runs = [rollout(pol, af, q0, om0, dist_cfg, dseed, asym, i)
			for i, (q0, om0) in enumerate(ics)]
		n_stable = sum(1 for r in runs if r["stable"])
		mean_err = sum(r["mean_err_deg"] for r in runs) / len(runs)
		steadies = [r["steady_deg"] for r in runs if r["steady_deg"] is not None]
		steady = sum(steadies) / len(steadies) if steadies else None
		out["policies"][name] = {
			"episodes": runs,
			"triple": {
				"stable_pct": round(100.0 * n_stable / len(runs), 1),
				"err_deg": round(mean_err, 3),
				"steady_deg": round(steady, 3) if steady is not None else None,
			},
		}
		print(f"[viz] {name:6s} stable={out['policies'][name]['triple']['stable_pct']:5.1f}% "
		      f"err={mean_err:.2f}° steady={steady if steady is not None else float('nan'):.2f}°")
	Path(args.out).write_text(json.dumps(out))
	print(f"[viz] wrote {args.out} ({Path(args.out).stat().st_size / 1024:.0f} KiB)")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
