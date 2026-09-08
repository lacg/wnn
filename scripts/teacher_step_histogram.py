#!/usr/bin/env python3
"""Teacher per-step |Δpwm| histogram vs the delta alphabet (L3 pre-check).

QUESTION IT ANSWERS. A `delta_max` arm is only a fair test of actuation
granularity if the alphabet still COVERS what the teacher asks for per step.
DAgger's label is (controller.rs `train_output_step`):

    label = target_pwm[motor] - pwm_prev[motor]      then clip(±delta_max)

so any step where the teacher wants more than delta_max is a SATURATED label,
and the arm is handicapped by construction, not by granularity. L3's 0.025 arm
lost 4/4; this probe says how much of that could have been clipping.

WHAT IT MEASURES. The ATTITUDE channel, exactly: the Rust sim (AttitudeSim)
and the Rust teacher (AttitudeMpcOfRs, oracle-fed — DAgger's labelling
convention) on the recipe's airframe, L4C weather and IC draws, driven from
Python the way dagger.py's loop does. No Rust is re-implemented here.

WHAT IT BOUNDS. The COLLECTIVE channel. The stage-1 teacher's altitude PD
(altitude_pd.rs) is not exposed to Python and a wheel rebuild is off the table
while a chain is armed, so its demand is bounded from the same derivation the
Rust uses: b_z = 8·k·pwm_h/m, az = ωn²·alt_err − 2ζωn·vz, δ = az/b_z. That δ
is an ABSOLUTE collective offset; per step, holding it costs (1−leak)·δ.

Runs in a minute at nice 19 — it is a rollout, not a search.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TEACHER_ID = {"pid": 0, "lqr": 1, "mpc": 2, "lqi": 3, "mpcof": 4}


@dataclass(frozen=True)
class Recipe:
	"""The ladder's plant + IC draw, copied from phased_ga's EpisodeConfig build."""
	airframe: str = "cf21_brushless"
	disturbance: str = "L4C"
	dist_seed: int = 911
	steps: int = 2000
	dt: float = 0.001
	tilt_deg: float = 5.0
	body_rate: float = 0.5
	yaw_rate: float = 0.3
	alt_offset_m: float = 0.3
	init_vz: float = 0.2
	levels_per_motor: int = 16
	delta_gamma: float = 1.0


def build_episode_config(r: Recipe, seed: int):
	"""EpisodeConfig with the SAME resolved motor asymmetry the scorer flies."""
	from dataclasses import replace
	from wnn.control.airframe import Airframe
	from wnn.control.evaluator import disturbance_stream
	from wnn.control.training import DisturbanceConfig, EpisodeConfig
	dist = DisturbanceConfig.preset(r.disturbance, seed=r.dist_seed)
	_, asym = disturbance_stream(dist, seed)
	dist = replace(dist, resolved_asym=tuple(asym))
	return EpisodeConfig(
		dt=r.dt, steps_per_episode=r.steps,
		max_initial_tilt_rad=math.radians(r.tilt_deg),
		max_initial_yaw_rad=math.radians(r.tilt_deg),
		max_initial_body_rate=r.body_rate, max_initial_yaw_rate=r.yaw_rate,
		disturbance=dist, airframe=Airframe.preset(r.airframe),
	)


def rollout_teacher(teacher, ec, rng: np.random.Generator) -> np.ndarray:
	"""One oracle-fed teacher episode → (steps, 4) pwm trace. Mirrors run_episode's
	loop; the teacher observes its OWN applied pwm (solo flight)."""
	from wnn.control._accel import AttitudeSim
	from wnn.control.training import _sample_initial_state, apply_disturbance
	sim = AttitudeSim()
	q0, w0 = _sample_initial_state(
		rng, ec.max_initial_tilt_rad, ec.max_initial_yaw_rad,
		ec.max_initial_body_rate, ec.max_initial_yaw_rate)
	sim.reset(q=list(q0), omega=list(w0))
	apply_disturbance(sim, ec.disturbance, rng)
	teacher.reset()
	target = (0.0, 0.0, 0.0)
	trace = np.zeros((ec.steps_per_episode, 4), dtype=np.float64)
	for k in range(ec.steps_per_episode):
		if sim.is_unstable():
			return trace[:k]
		gyro, _accel = sim.read_imu()
		pwm = teacher.step(sim.quaternion, gyro, target)
		teacher.observe(gyro, pwm)
		sim.step(list(pwm))
		trace[k] = pwm
	return trace


def collect_traces(r: Recipe, teacher_name: str, episodes: int, seed: int) -> list[np.ndarray]:
	from wnn.control.dagger import make_expert
	ec = build_episode_config(r, seed)
	teacher = make_expert(teacher_name, ec)
	rng = np.random.default_rng(seed)
	traces = []
	for _ in range(episodes):
		ep_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
		traces.append(rollout_teacher(teacher, ec, ep_rng))
	return traces


def perfect_imitator_labels(u: np.ndarray, leak: float, dmax: float, step: float) -> dict:
	"""Replay the DAgger label rule through a leaky accumulator that always emits
	the (clipped, quantized) label. Returns clip share, dead-zone share and the
	tracking RMS the alphabet alone imposes. Starts on the teacher's first command
	so there is no startup artefact."""
	a = u[0].copy()
	n = len(u) - 1
	clipped = dead = 0
	err2 = 0.0
	for k in range(1, len(u)):
		label = u[k] - a                       # target - pwm_prev
		clipped += int(np.any(np.abs(label) > dmax))
		dead += int(np.all(np.abs(label) < step / 2.0))
		emit = np.clip(np.round(label / step) * step, -dmax, dmax)
		a = u[0] + leak * (a - u[0]) + emit    # anchor = first command
		a = np.clip(a, 0.0, 1.0)
		err2 += float(np.mean((a - u[k]) ** 2))
	return {"clip_share": clipped / max(n, 1), "dead_share": dead / max(n, 1),
	        "track_rms_pwm": math.sqrt(err2 / max(n, 1))}


def raw_step_stats(traces: list[np.ndarray]) -> dict:
	d = np.concatenate([np.abs(np.diff(t, axis=0)).ravel() for t in traces if len(t) > 1])
	pct = {p: float(np.percentile(d, p)) for p in (50, 90, 99, 99.9)}
	return {"n": int(d.size), "p50": pct[50], "p90": pct[90], "p99": pct[99],
	        "p99_9": pct[99.9], "max": float(d.max())}


def collective_bound(r: Recipe, leaks: list[float]) -> dict:
	"""AltitudePd's demand at the recipe's IC bounds, from altitude_pd.rs's own
	derivation (ωn = 2, ζ = 1 defaults). δ is an absolute offset; per-step cost
	of HOLDING it is (1−leak)·δ."""
	from wnn.control.airframe import Airframe
	af = Airframe.preset(r.airframe)
	hover = math.sqrt(af.mass * af.gravity / (4.0 * af.k_thrust))
	b_z = 8.0 * af.k_thrust * hover / af.mass
	omega_n, zeta = 2.0, 1.0
	az_max = omega_n ** 2 * r.alt_offset_m + 2.0 * zeta * omega_n * r.init_vz
	delta_abs = min(az_max / b_z, 0.25)
	return {"hover_pwm": hover, "b_z": b_z, "delta_abs_max": delta_abs,
	        "hold_cost_per_step": {str(l): (1.0 - l) * delta_abs for l in leaks}}


def print_report(out: dict) -> None:
	raw = out["raw_step"]
	print(f"\nTEACHER {out['teacher']} · {out['episodes']} episodes × {out['recipe']['steps']} steps · "
	      f"{out['recipe']['airframe']} / {out['recipe']['disturbance']} · tilt {out['recipe']['tilt_deg']}° · "
	      f"oracle-fed · {raw['n']} motor-steps")
	print(f"  raw per-step |Δpwm|   p50 {raw['p50']:.4f}   p90 {raw['p90']:.4f}   "
	      f"p99 {raw['p99']:.4f}   p99.9 {raw['p99_9']:.4f}   max {raw['max']:.4f}")
	print(f"\n  DAgger label through a perfect imitator (label = target − pwm_prev, clip ±dmax, "
	      f"17-value alphabet step = dmax/8)")
	print(f"  {'leak':>6} {'dmax':>7} {'step':>8} {'floor':>8} {'clip%':>7} {'dead%':>7} {'track RMS':>10}")
	for row in out["grid"]:
		print(f"  {row['leak']:>6.2f} {row['dmax']:>7.3f} {row['step']:>8.4f} {row['floor']:>8.4f} "
		      f"{100 * row['clip_share']:>6.2f}% {100 * row['dead_share']:>6.1f}% {row['track_rms_pwm']:>10.4f}")
	cb = out["collective_bound"]
	print(f"\n  COLLECTIVE channel (AltitudePd bound, not measured): hover {cb['hover_pwm']:.4f} pwm · "
	      f"b_z {cb['b_z']:.2f} m/s²/pwm · max |δ| at IC bounds {cb['delta_abs_max']:.4f} pwm (absolute)")
	for leak, cost in cb["hold_cost_per_step"].items():
		print(f"    hold cost per step at leak {leak}: {cost:.5f} pwm  "
		      f"({'under' if cost < 0.0125 else 'over'} one 0.0125 alphabet step)")


def main() -> int:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--teacher", default="mpcof", choices=sorted(TEACHER_ID))
	ap.add_argument("--episodes", type=int, default=20)
	ap.add_argument("--seed", type=int, default=99990101, help="a report seed, so the draw is the scorer's")
	ap.add_argument("--dmax", type=float, nargs="+", default=[0.1, 0.05, 0.025])
	ap.add_argument("--leak", type=float, nargs="+", default=[0.95, 0.90])
	ap.add_argument("--out", default="experiments/teacher_step_hist")
	args = ap.parse_args()
	os.nice(19)
	r = Recipe()
	t0 = time.time()
	traces = collect_traces(r, args.teacher, args.episodes, args.seed)
	grid = []
	for leak in args.leak:
		for dmax in args.dmax:
			step = dmax / 8.0
			acc = {"clip_share": 0.0, "dead_share": 0.0, "track_rms_pwm": 0.0}
			for t in traces:
				s = perfect_imitator_labels(t, leak, dmax, step)
				for k in acc:
					acc[k] += s[k] / len(traces)
			grid.append({"leak": leak, "dmax": dmax, "step": step, "floor": step / (1.0 - leak), **acc})
	out = {
		"teacher": args.teacher, "episodes": args.episodes, "seed": args.seed,
		"recipe": r.__dict__, "diverged": sum(len(t) < r.steps for t in traces),
		"raw_step": raw_step_stats(traces), "grid": grid,
		"collective_bound": collective_bound(r, args.leak),
		"wall_s": time.time() - t0,
	}
	Path(args.out).mkdir(parents=True, exist_ok=True)
	path = Path(args.out) / f"{args.teacher}_s{args.seed}_e{args.episodes}.json"
	path.write_text(json.dumps(out, indent=1))
	print_report(out)
	print(f"\n  diverged episodes: {out['diverged']}/{args.episodes} · {out['wall_s']:.0f} s · {path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
