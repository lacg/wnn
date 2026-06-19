"""H1 — output-saturation diagnostic for a trained WNN attitude controller.

Question (from docs/controller_quadcopter_inspired_experiments.md §H1): is the
residual ~3-4° steady-state error AUTHORITY-LIMITED (the per-step delta pins at
±delta_max while error persists → raise delta_max / resolution) or a PERCEPTION
problem (delta sits well inside its range while error persists → go to H2:
error/integral observation features)?

Method — pure-Python rollout, NO Rust rebuild (safe alongside a running worker):
  * load the winner via checkpoint_io.load_controller_checkpoint
  * build the Rust WnnController via the evaluator's build path
  * drive controller.step() in a Python loop (mirrors training.run_episode),
    and after each step decode the LAST output cells the same way controller.rs
    does (QSR sum -> decoded -> decoded_to_delta) to recover the per-motor delta
    and test |delta| >= delta_max. Correlate with the live attitude error.

Only meaningful in delta_control mode (delta_max is the per-step authority). In
absolute-PWM mode the script says so and reports error stats only.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

from wnn.control._accel import AttitudeSim
from wnn.control.checkpoint_io import load_controller_checkpoint
from wnn.control.evaluator import (
	build_controller,
	controller_genome_from_arch,
	fit_thresholds_from_pid_rollouts,
)
from wnn.control.training import (
	EpisodeConfig,
	_euler_to_quat_xyz,
	_sample_initial_state,
)

# Mirror controller.rs decode constants (QUAD_WEIGHTS / NEUTRAL_DECODE).
QSR_WEIGHTS = (0.0, 0.25, 0.75, 1.0)
NEUTRAL_DECODE = 0.75


def decoded_to_delta(decoded: float, delta_max: float) -> float:
	"""Exact port of controller.rs decoded_to_delta (piecewise-linear @0.75)."""
	n = NEUTRAL_DECODE
	if decoded >= n:
		return (decoded - n) / (1.0 - n) * delta_max
	return (decoded - n) / n * delta_max


def decode_motor_deltas(cells: list[int], num_motors: int, levels: int,
                        delta_max: float) -> list[float]:
	"""Recover the per-motor delta from the last output cells (controller.rs §7)."""
	deltas = []
	for m in range(num_motors):
		start = m * levels
		s = 0.0
		for c in cells[start:start + levels]:
			s += QSR_WEIGHTS[c & 0x3]
		decoded = min(max(s / levels, 0.0), 1.0)
		deltas.append(decoded_to_delta(decoded, delta_max))
	return deltas


def run_instrumented_episode(controller, spec, ec: EpisodeConfig,
                             rng, target_rpy=(0.0, 0.0, 0.0)):
	"""One episode; return a list of per-step (err_deg, max_sat_ratio, any_sat)."""
	sim = AttitudeSim()
	controller.reset()
	init_q, init_omega = _sample_initial_state(
		rng, ec.max_initial_tilt_rad, ec.max_initial_yaw_rad,
		ec.max_initial_body_rate, ec.max_initial_yaw_rate)
	sim.reset(q=list(init_q), omega=list(init_omega))
	target_q = _euler_to_quat_xyz(*target_rpy)

	dmax = float(spec.delta_max)
	steps = []
	for _ in range(ec.steps_per_episode):
		if sim.is_unstable():
			break
		gyro, accel = sim.read_imu()
		pwm = controller.step(list(gyro), list(accel), list(target_rpy))
		cells = controller.get_last_output_cells()
		deltas = decode_motor_deltas(cells, spec.num_motors,
		                             spec.levels_per_motor, dmax)
		sim.step(list(pwm))
		err_deg = math.degrees(sim.attitude_error(target_q))
		# Saturation ratio per motor = |delta| / delta_max; pinned if >= 0.99.
		ratios = [abs(d) / dmax if dmax > 0 else 0.0 for d in deltas]
		max_ratio = max(ratios)
		steps.append((err_deg, max_ratio, max_ratio >= 0.99))
	return steps


def main() -> int:
	ap = argparse.ArgumentParser(description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--winner", required=True, help="winner.yaml.gz path")
	ap.add_argument("--episodes", type=int, default=10)
	ap.add_argument("--steps", type=int, default=1000)
	ap.add_argument("--tilt", type=float, default=5.0, help="initial tilt (deg)")
	ap.add_argument("--seed", type=int, default=99990101, help="held-out seed")
	# steady-state window: the LAST `tail_frac` of each episode = the settled
	# region where the residual error lives (H1 cares about saturation THERE).
	ap.add_argument("--tail-frac", type=float, default=0.5)
	args = ap.parse_args()

	p = Path(args.winner)
	if not p.exists():
		print(f"ERROR: winner not found: {p}", file=sys.stderr)
		return 1

	print(f"[H1] loading winner {p.name} ...", flush=True)
	t0 = time.time()
	payload = load_controller_checkpoint(str(p))
	spec = payload["spec"]
	genome = payload.get("best_genome")
	print(f"[H1] loaded in {time.time()-t0:.1f}s | spec sn={spec.state_neurons} "
	      f"levels={spec.levels_per_motor} motors={spec.num_motors} "
	      f"delta_control={spec.delta_control} delta_max={spec.delta_max} "
	      f"delta_leak={spec.delta_leak}", flush=True)

	if not spec.delta_control:
		print("\n[H1] VERDICT: controller is in ABSOLUTE-PWM mode (delta_control="
		      "False) — there is no per-step delta_max authority cap, so the H1 "
		      "saturation question does not apply. Route to H2 (error/integral "
		      "observation features). No saturation analysis run.")
		return 0

	# Build controller (write trained cells) + PID-fit thresholds on held-out seed.
	print("[H1] fitting thresholds + building controller ...", flush=True)
	thresholds = fit_thresholds_from_pid_rollouts(spec, num_episodes=10, seed=args.seed)
	controller = build_controller(controller_genome_from_arch(genome, spec, thresholds))

	ec = EpisodeConfig(
		dt=0.001, steps_per_episode=args.steps,
		max_initial_tilt_rad=math.radians(args.tilt),
		max_initial_yaw_rad=math.radians(args.tilt),
		max_initial_body_rate=0.5, max_initial_yaw_rate=0.3,
	)
	rng = np.random.default_rng(args.seed)

	print(f"[H1] rolling {args.episodes} episodes x {args.steps} steps "
	      f"@ tilt {args.tilt}° seed {args.seed} ...", flush=True)
	all_steps = []
	for e in range(args.episodes):
		st = run_instrumented_episode(controller, spec, ec, rng)
		all_steps.append(st)
		merr = np.mean([s[0] for s in st]) if st else float("nan")
		print(f"   ep {e+1:2d}: {len(st):4d} steps, mean err {merr:5.2f}°", flush=True)

	# ---- Aggregate analysis ------------------------------------------------
	flat = [s for ep in all_steps for s in ep]
	if not flat:
		print("[H1] no steps recorded.")
		return 1

	errs = np.array([s[0] for s in flat])
	ratios = np.array([s[1] for s in flat])
	sat = np.array([s[2] for s in flat], dtype=bool)

	# steady-state region: last tail_frac of each episode
	tail = []
	for ep in all_steps:
		k = int(len(ep) * (1.0 - args.tail_frac))
		tail.extend(ep[k:])
	t_err = np.array([s[0] for s in tail]) if tail else errs
	t_ratio = np.array([s[1] for s in tail]) if tail else ratios
	t_sat = np.array([s[2] for s in tail], dtype=bool) if tail else sat

	# residual-error band: settled-but-imperfect steps (1°..5°) where the 5° gap lives
	band = (t_err >= 1.0) & (t_err < 5.0)
	b_ratio = t_ratio[band]
	b_sat = t_sat[band]

	def pct(x): return 100.0 * float(np.mean(x)) if len(x) else float("nan")

	print(f"\n{'='*64}\n  H1 SATURATION ANALYSIS\n{'='*64}")
	print(f"  total steps           : {len(flat)}  ({args.episodes} eps)")
	print(f"  overall mean err      : {errs.mean():.2f}°   stable(<5°) {pct(errs<5.0):.1f}%")
	print(f"  overall saturated     : {pct(sat):.1f}% of steps  "
	      f"(mean |delta|/dmax {ratios.mean():.2f})")
	print(f"  --- steady-state tail (last {int(args.tail_frac*100)}% of each ep) ---")
	print(f"  tail mean err         : {t_err.mean():.2f}°")
	print(f"  tail saturated        : {pct(t_sat):.1f}%  (mean |delta|/dmax {t_ratio.mean():.2f})")
	print(f"  --- residual band (1°..5° in the tail; n={int(band.sum())}) ---")
	print(f"  band saturated        : {pct(b_sat):.1f}%  (mean |delta|/dmax "
	      f"{b_ratio.mean() if len(b_ratio) else float('nan'):.2f})")

	# Verdict: in the residual band, is the controller pinned at authority?
	band_sat_pct = pct(b_sat)
	band_ratio = float(b_ratio.mean()) if len(b_ratio) else 0.0
	print(f"\n  VERDICT:")
	if len(b_ratio) == 0:
		print("    Inconclusive — no steps in the 1-5° residual band (either always "
		      "stable <1° or diverged). Inspect per-episode err above.")
	elif band_sat_pct >= 50.0 or band_ratio >= 0.8:
		print(f"    AUTHORITY-LIMITED — in the residual band the delta pins at "
		      f"±delta_max {band_sat_pct:.0f}% of the time (mean ratio {band_ratio:.2f}). "
		      f"The net WANTS a bigger correction than delta_max={spec.delta_max} allows.")
		print(f"    → H1-FIX: raise delta_max (and/or levels_per_motor for finer "
		      f"near-neutral resolution); retrain a small probe and re-measure.")
	elif band_ratio <= 0.5:
		print(f"    NOT SATURATION — in the residual band the delta sits well inside "
		      f"its range (mean ratio {band_ratio:.2f}, pinned only {band_sat_pct:.0f}%). "
		      f"The controller is NOT authority-limited; the residual error is a "
		      f"perception/integral problem.")
		print(f"    → H2: add error + leaky-integral OBSERVATION features (controller.rs "
		      f"NUM_FEATURES), retrain. See docs/controller_quadcopter_inspired_experiments.md.")
	else:
		print(f"    MIXED — band saturation {band_sat_pct:.0f}%, mean ratio "
		      f"{band_ratio:.2f}. Partial authority pressure; H2 likely the bigger "
		      f"lever but a modest delta_max bump may help. Inspect distribution.")
	return 0


if __name__ == "__main__":
	sys.exit(main())
