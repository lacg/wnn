"""DAGGER (Dataset Aggregation) trainer for WNN attitude controllers.

Ross, Gordon & Bagnell, "A Reduction of Imitation Learning and Structured
Prediction to No-Regret Online Learning", AISTATS 2011.

WHY THIS EXISTS
---------------
The earlier `bptt_trainer.train_bptt` is, despite its "DAGGER" comment,
pure **behavioral cloning**: the expert (PID) drives the sim at every
step, so the network is only ever trained on states the EXPERT visits.
Closed-loop, the student drives, drifts into states the expert never
showed it, and its per-step errors compound (~ε·T²) — exactly the
covariate-shift failure `diagnostics.open_loop_prediction_error` flagged
(the WNN learns PID open-loop to ~0.02 PWM error, yet loses to random
closed-loop).

DAGGER fixes this by training on the STUDENT's own state distribution:
  iter 0:  β=1  → expert drives (behavioral-cloning warm-start)
  iter i:  β=decay^i → with prob β the expert drives, else the student
           drives; at EVERY visited state we query the expert (PID) and
           train the controller toward the expert's action.
As β→0 the rollout distribution becomes the student's own, so the cells
get trained exactly where the deployed controller will operate.

ARCHITECTURE (no Rust logic reimplemented in Python — CLAUDE.md rule)
--------------------------------------------------------------------
This is OUTER-LOOP ORCHESTRATION only (roll out, β-mix, aggregate,
keep-curve), which `training.py` explicitly keeps in Python. Every
per-step hot operation is Rust:
  - sim physics            : AttitudeSim.step           (Rust)
  - controller forward     : WnnController.step         (Rust)
  - QSR-EDRA train step    : WnnController.edra_train_step (Rust — the
                             real per-motor beam-search constraint solver
                             ported in controller_training.rs)
The only Python per-step cost is the PID teacher (cheap, closed-form).
If GA throughput demands it, the whole rollout can move into a Rust
`dagger_train` entry point (would need a Rust PID) — noted as the
performance follow-up, not done pre-validation.

Cross-iteration accumulation is safe: WnnController.reset() zeroes only
the recurrent state + input history, never the trained memory cells.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from wnn.control._accel import AttitudeSim, WnnController

from .pid import AttitudePID, AttitudePIDConfig
from .evaluator import ControllerSpec, NUM_FEATURES
from .training import (
	EpisodeConfig,
	fitness_function,
	make_wnn_action_fn,
	_sample_initial_state,
)


@dataclass
class DaggerConfig:
	"""DAGGER training-loop configuration."""
	num_iterations: int = 5          # DAGGER rounds (iter 0 = BC warm-start)
	episodes_per_iter: int = 20      # rollouts collected+trained per round
	steps_per_episode: int = 2000    # 2 s @ 1 kHz
	beta_decay: float = 0.5          # β_i = max(beta_floor, beta_decay**i); β_0 = 1.0
	beta_floor: float = 0.0          # never drop expert-mix below this
	eval_episodes: int = 20          # closed-loop eval after each round
	topk_per_neuron: int = 4         # beam-search top-k in the QSR-EDRA solve
	seed: int = 0
	progress: bool = True
	target_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
	episode_config: Optional[EpisodeConfig] = None

	def __post_init__(self):
		if self.episode_config is None:
			self.episode_config = EpisodeConfig(
				dt=0.001, steps_per_episode=self.steps_per_episode,
				max_initial_tilt_rad=math.radians(30.0),
				max_initial_yaw_rad=math.radians(30.0),
				max_initial_body_rate=0.5, max_initial_yaw_rate=0.3,
			)

	def beta(self, iteration: int) -> float:
		"""Expert-mix probability for a DAGGER round. β_0 = 1.0 (pure expert)."""
		return max(self.beta_floor, self.beta_decay ** iteration)


def eval_closed_loop_reset(action_fn, reset_fn, episode_config, num_episodes: int, seed: int) -> tuple[float, dict]:
	"""Run num_episodes resetting the POLICY (reset_fn) AND sim each episode.

	Unlike fitness_function, this resets the policy's internal state — recurrent
	state, the delta-control throttle accumulator, or a PID's integral — at the
	start of every episode. That's required for the delta-control integrator
	(otherwise episode N+1 starts at episode N's saturated throttle) and gives a
	fair baseline comparison (PID/untrained/trained all start clean per episode).
	"""
	from wnn.control._accel import AttitudeSim
	from .training import run_episode

	sim = AttitudeSim()
	rng = np.random.default_rng(seed)
	errs, rewards, jerks, stable = [], [], [], 0
	for _ in range(num_episodes):
		reset_fn()
		ep_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
		res = run_episode(action_fn, sim, episode_config, rng=ep_rng)
		errs.append(res.mean_attitude_error_rad)
		rewards.append(res.cumulative_reward)
		jerks.append(res.mean_pwm_jerk)
		if (not res.diverged) and res.mean_attitude_error_rad <= math.radians(5.0):
			stable += 1
	mean_err = float(np.mean(errs))
	return float(np.mean(rewards)), {
		"mean_reward": float(np.mean(rewards)),
		"mean_attitude_error_rad": mean_err,
		"mean_attitude_error_deg": math.degrees(mean_err),
		"stable_rate": stable / max(num_episodes, 1),
		# Surface the per-episode motor-jerk mean so reward_gated_train can plumb
		# it into Metrics.motor_jerk_mean (the Python path used to drop it →
		# weight_jerk silently ignored; fixed 01/06/2026).
		"mean_pwm_jerk": float(np.mean(jerks)),
	}


def _eval_closed_loop(
	controller: WnnController,
	cfg: DaggerConfig,
) -> tuple[float, dict]:
	"""Score the controller closed-loop (student drives), resetting the
	controller's recurrent state + throttle accumulator each episode."""
	return eval_closed_loop_reset(
		make_wnn_action_fn(controller), controller.reset,
		cfg.episode_config, cfg.eval_episodes, cfg.seed + 7_000_000,
	)


def train_dagger(
	spec: ControllerSpec,
	thresholds: list[float],
	state_connections: list[int],
	output_connections: list[int],
	config: DaggerConfig,
) -> tuple[WnnController, dict]:
	"""Run DAGGER. Returns (trained_controller, stats).

	The controller is built fresh (empty QSR cells) from the supplied
	connectivity + thresholds, then trained in place across rounds.
	`stats` carries the per-iteration closed-loop fitness curve so we can
	SEE whether later (student-distribution) rounds actually help — the
	whole point of DAGGER over behavioral cloning.
	"""
	controller = WnnController(
		num_motors=spec.num_motors,
		levels_per_motor=spec.levels_per_motor,
		bits_per_feature=spec.bits_per_feature,
		input_window_k=spec.input_window_k,
		state_neurons=spec.state_neurons,
		state_bits_per_neuron=spec.state_bits_per_neuron,
		output_bits_per_neuron=spec.output_bits_per_neuron,
		thresholds=thresholds,
		state_connections=state_connections,
		output_connections=output_connections,
		delta_control=spec.delta_control,
		delta_max=spec.delta_max,
		delta_leak=spec.delta_leak,
		obs_tilt_p=spec.obs_tilt_p, obs_tilt_i=spec.obs_tilt_i, obs_peraxis_p=spec.obs_peraxis_p, obs_peraxis_i=spec.obs_peraxis_i, obs_peraxis_yaw=spec.obs_peraxis_yaw, obs_pwm=spec.obs_pwm, obs_yaw_err=spec.obs_yaw_err, obs_yaw_err_i=spec.obs_yaw_err_i, dt=spec.dt, integral_leak=spec.integral_leak, integral_scale=spec.integral_scale, decouple_outputs=spec.decouple_outputs,
	)
	pid = AttitudePID(AttitudePIDConfig())
	sim = AttitudeSim()
	rng = np.random.default_rng(config.seed)
	ec = config.episode_config
	target = config.target_rpy

	stats = {
		"iter_fitness": [],
		"iter_mean_err_deg": [],
		"iter_stable_rate": [],
		"iter_beta": [],
		"iter_cells_written": [],
		"train_steps": 0,
	}

	for it in range(config.num_iterations):
		beta = config.beta(it)
		cells_written = 0

		for _ in range(config.episodes_per_iter):
			ep_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
			init_q, init_omega = _sample_initial_state(
				ep_rng,
				ec.max_initial_tilt_rad, ec.max_initial_yaw_rad,
				ec.max_initial_body_rate, ec.max_initial_yaw_rate,
			)
			sim.reset(q=list(init_q), omega=list(init_omega))
			pid.reset()
			controller.reset()

			for _ in range(config.steps_per_episode):
				if sim.is_unstable():
					break
				gyro, accel = sim.read_imu()
				q = sim.quaternion

				# Student forward (advances controller recurrent state) ...
				student_pwm = controller.step(list(gyro), list(accel), list(target))
				# ... expert label at THIS state ...
				expert_pwm = pid.step(q, gyro, target)
				# ... train the QSR cells toward the expert (Rust beam-search EDRA).
				sw, ow = controller.edra_train_step(list(expert_pwm), config.topk_per_neuron)
				cells_written += int(sw) + int(ow)
				stats["train_steps"] += 1

				# DAGGER state distribution: β-mix which action drives the sim.
				# β_0 = 1 → expert drives (BC warm-start); β decays → student drives.
				use_expert = rng.random() < beta
				action = expert_pwm if use_expert else student_pwm
				sim.step(list(action))
				# Delta-control: if the expert drove, sync the controller's
				# throttle integrator to the actually-applied PWM so the next
				# step's delta is computed from the correct baseline.
				if spec.delta_control and use_expert:
					controller.set_pwm(list(expert_pwm))

		fit, metrics = _eval_closed_loop(controller, config)
		stats["iter_fitness"].append(float(fit))
		stats["iter_mean_err_deg"].append(float(metrics["mean_attitude_error_deg"]))
		stats["iter_stable_rate"].append(float(metrics["stable_rate"]))
		stats["iter_beta"].append(float(beta))
		stats["iter_cells_written"].append(int(cells_written))

		if config.progress:
			print(
				f"  DAGGER iter {it + 1}/{config.num_iterations}: "
				f"β={beta:.3f}  closed-loop fitness={fit:.2f}  "
				f"mean_err={metrics['mean_attitude_error_deg']:.2f}°  "
				f"stable={metrics['stable_rate']*100:.0f}%  "
				f"cells+={cells_written}"
			)

	# Report best round for convenience (deployed controller is the final
	# one; final≈best when accumulation is monotone — the curve shows if not).
	best_idx = int(np.argmax(stats["iter_fitness"])) if stats["iter_fitness"] else -1
	stats["best_iter"] = best_idx
	stats["best_fitness"] = stats["iter_fitness"][best_idx] if best_idx >= 0 else float("-inf")
	stats["final_fitness"] = stats["iter_fitness"][-1] if stats["iter_fitness"] else float("-inf")
	return controller, stats


__all__ = ["DaggerConfig", "train_dagger", "eval_closed_loop_reset"]
