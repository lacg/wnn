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

from .pid import AttitudePID, AttitudePIDConfig, PIDGains
from .evaluator import ControllerSpec, NUM_FEATURES
from .training import (
	EpisodeConfig,
	fitness_function,
	make_wnn_action_fn,
	make_pid_action_fn,
	make_residual_action_fn,
	compose_residual,
	residual_train_target,
	apply_disturbance,
	_sample_initial_state,
)


def _pd_config() -> AttitudePIDConfig:
	"""Memoryless PD baseline (Ki=0) — the analytic floor (84 @L2)."""
	c = AttitudePIDConfig()
	for ax in ("roll", "pitch", "yaw"):
		g = getattr(c, ax)
		setattr(c, ax, PIDGains(kp=g.kp, ki=0.0, kd=g.kd, i_clamp=g.i_clamp))
	return c


def _pid_plus_config() -> AttitudePIDConfig:
	"""PID+ expert (ki×4, i_clamp×4) — the integral ceiling (99.8 @L2). The
	residual-DAGGER teacher whose integral action the WNN learns."""
	c = AttitudePIDConfig()
	for ax in ("roll", "pitch", "yaw"):
		g = getattr(c, ax)
		setattr(c, ax, PIDGains(kp=g.kp, ki=g.ki * 4.0, kd=g.kd, i_clamp=g.i_clamp * 4.0))
	return c


def _residual_baseline_config(name: str) -> AttitudePIDConfig:
	"""Analytic baseline the WNN learns a residual on top of (E5 ablation)."""
	if name == "pd":
		return _pd_config()
	if name == "stock_pid":
		return AttitudePIDConfig()   # stock (97 @L2)
	raise ValueError(f"residual_baseline must be \'pd\' or \'stock_pid\', got {name!r}")


def make_residual_baseline(name: str, episode_config):
	"""The analytic baseline the residual composes on.

	L2 (06/08/2026): on an airframe that registers firmware cascade gains this is the
	FIRMWARE CASCADE (`AttitudePidFirmware`), not the legacy single-loop `AttitudePID`
	— the same controller the Metal kernel runs via `pidfw_step`, so the CPU and GPU
	baselines are one number rather than two. Airframes without cascade gains (the
	synthetic plant, and the `pd` ablation which is DEFINED as a memoryless PD) keep the
	legacy loop, so every pre-L2 residual run reproduces.

	`pd` deliberately stays legacy on every airframe: it is the Ki=0 analytic FLOOR the
	E5 ablation measures against, and a cascade carries integral action by construction.
	"""
	af = getattr(episode_config, "airframe", None)
	if name == "stock_pid" and af is not None:
		try:
			gains = af.gains()
		except KeyError:
			gains = None
		if gains is not None and gains.rate is not None:
			from wnn.control.pid_firmware import AttitudePidFirmware
			return AttitudePidFirmware(af, gains)
	return AttitudePID(_residual_baseline_config(name))


class _ObserverExpert:
	"""Wraps a Rust teacher so the DAGGER loop can drive it uniformly.

	Exists for ONE reason: `AttitudeMpcOfRs` is offset-free only if it is TOLD the
	action that was actually applied — `observe(gyro, applied_pwm)` is what builds its
	disturbance estimate d̂. Rust says so explicitly: "Solo flight without observe()
	⇒ d̂ stays 0 ⇒ mpcof degrades to plain MPC (safe no-op)". A silent degrade is the
	worst possible outcome here, because the run would LOOK like an mpcof arm and
	score like an mpc one. So observe is part of the interface, and `has_observer`
	lets a caller assert it is really getting the observer.
	"""

	def __init__(self, inner, has_observer: bool):
		self._inner = inner
		self.has_observer = has_observer
		self._last_applied = [0.5, 0.5, 0.5, 0.5]

	def reset(self) -> None:
		self._inner.reset()
		self._last_applied = [0.5, 0.5, 0.5, 0.5]

	def observe(self, gyro, applied_pwm) -> None:
		"""Feed back the APPLIED action (on-policy) before the next plan."""
		if self.has_observer:
			self._inner.observe_py([float(g) for g in gyro],
			                       [float(p) for p in applied_pwm])
			self._last_applied = [float(p) for p in applied_pwm]

	def step(self, q, gyro, target_rpy):
		out = list(self._inner.step(list(q), list(gyro), list(target_rpy)))
		self._last_applied = out
		return out


def make_expert(name: str, episode_config=None):
	"""The DAGGER teacher the WNN imitates. All expose step(q,gyro,target)+reset().

	AIRFRAME-AWARE since 06/08/2026 (L2). `wnn.control.optimal`'s LQRController /
	MPCController build their plant from `attitude_linear_model()`, which takes NO
	plant parameters — they are hardwired to the retired SYNTHETIC plant. Handing one
	of them a Crazyflie rollout yields a teacher derived for a different vehicle,
	which is exactly the defect `project_pid_not_airframe_retuned` records for PID.
	So when an EpisodeConfig carrying an airframe is supplied, the teacher is built
	from the RUST classes, which take the plant explicitly — the same ones
	`Teacher::from_id` uses on the GPU DAGGER path. Without an airframe the legacy
	Python objects are returned unchanged, so every pre-L2 residual run reproduces.

	lqi/mpcof exist ONLY on the airframe path: they have no Python twin.
	"""
	af = getattr(episode_config, "airframe", None) if episode_config is not None else None
	if af is None:
		if name == "pid_plus":
			return AttitudePID(_pid_plus_config())
		if name == "lqr":
			from wnn.control.optimal import LQRController
			return LQRController()
		if name == "mpc":
			from wnn.control.optimal import MPCController
			return MPCController()
		raise ValueError(
			f"residual_expert must be 'pid_plus'|'lqr'|'mpc' on the synthetic plant "
			f"(lqi/mpcof require an airframe — no Python twin), got {name!r}")

	if name == "pid_plus":
		# No airframe-derived twin: pid_plus IS the cranked-integral PID ablation.
		return AttitudePID(_pid_plus_config())
	from wnn.control._accel import (AttitudeLqrRs, AttitudeMpcRs, AttitudeLqiRs,
	                                AttitudeMpcOfRs)
	dt = float(getattr(episode_config, "dt", 0.001))
	plant = dict(dt=dt, arm_length=float(af.arm_length), k_thrust=float(af.k_thrust),
	             k_drag=float(af.k_drag), inertia=[float(x) for x in af.inertia],
	             gravity=float(af.gravity))
	if name == "lqr":
		return _ObserverExpert(AttitudeLqrRs(**plant), False)
	if name == "mpc":
		return _ObserverExpert(AttitudeMpcRs(**plant), False)
	if name == "lqi":
		return _ObserverExpert(AttitudeLqiRs(**plant), False)
	if name == "mpcof":
		return _ObserverExpert(AttitudeMpcOfRs(**plant), True)
	raise ValueError(
		f"residual_expert must be 'pid_plus'|'lqr'|'mpc'|'lqi'|'mpcof', got {name!r}")


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
	# E5 residual hybrid (see .claude/plans/e5_residual_hybrid.md). When True the
	# WNN learns a RESIDUAL on an analytic baseline instead of the full action:
	#   deployed = compose_residual(baseline(err), wnn); expert = PID+ (integral);
	#   per-step label = residual_train_target(PID+, baseline). Untrained hybrid
	#   ≡ baseline (the 84 @L2 floor), so training can only help.
	residual: bool = False
	residual_baseline: str = "pd"          # "pd" (84) | "stock_pid" (97)
	residual_scale: float = 1.0            # WNN [0,1] → residual (out−0.5)·scale
	residual_clamp: float = 0.2            # per-motor residual authority bound
	# The DAGGER teacher whose action the WNN imitates (as clamp(expert − baseline)).
	# "pid_plus" = the cranked-integral PID ceiling; the rest are optimal control.
	# lqi/mpcof require an airframe (no Python twin) — make_expert enforces that.
	residual_expert: str = "pid_plus"      # "pid_plus"|"lqr"|"mpc"|"lqi"|"mpcof"

	def __post_init__(self):
		if self.residual and self.residual_baseline not in ("pd", "stock_pid"):
			raise ValueError(f"residual_baseline must be 'pd' or 'stock_pid', got {self.residual_baseline!r}")
		if self.residual and self.residual_expert not in ("pid_plus", "lqr", "mpc", "lqi", "mpcof"):
			raise ValueError(f"residual_expert must be 'pid_plus'|'lqr'|'mpc'|'lqi'|'mpcof', "
			                 f"got {self.residual_expert!r}")
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

	# SCORING must fly the SAME airframe the batched Rust/Metal scorers fly: ONE
	# resolved motor asymmetry for the whole pass ("per-airframe wear", see
	# evaluator.disturbance_stream), not a fresh draw per episode. Averaging err
	# over a distribution of airframes sits ABOVE evaluating it at one (err is
	# convex in disturbance magnitude), which is what made the Python and Rust
	# columns of the L2 table incomparable. Resolved from the SAME (dist.seed XOR
	# score_seed) pair the kernel uses, so both paths get the identical vector.
	# Bound on a COPY — never mutate the caller's config.
	_dist = getattr(episode_config, "disturbance", None)
	if _dist is not None and _dist.resolved_asym is None:
		from dataclasses import replace as _replace
		from .evaluator import disturbance_stream
		_, _asym = disturbance_stream(_dist, seed)
		episode_config = _replace(episode_config,
		                          disturbance=_replace(_dist, resolved_asym=tuple(_asym)))

	sim = AttitudeSim()
	rng = np.random.default_rng(seed)
	errs, rewards, jerks, stable = [], [], [], 0
	# Transient-speed metrics (rise/settle/ITAE) — how FAST it corrects.
	rise, settle_abs, settle_rel, itae, iae, ise = [], [], [], [], [], []
	# steady = mean attitude error over the last steady_window_frac of the episode:
	# the HOLD term, as opposed to err which is ~80% recovery. run_episode has always
	# computed it (EpisodeResult.mean_steady_error_rad); this function just never
	# collected it, so every caller reporting from here could only show err/stable and
	# the required err/stable/steady triple was unreachable (06/08/2026).
	steady = []
	for _ in range(num_episodes):
		reset_fn()
		ep_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
		res = run_episode(action_fn, sim, episode_config, rng=ep_rng)
		errs.append(res.mean_attitude_error_rad)
		steady.append(res.mean_steady_error_rad)
		rewards.append(res.cumulative_reward)
		jerks.append(res.mean_pwm_jerk)
		rise.append(res.rise_time_s)
		settle_abs.append(res.settle_time_abs2deg_s)
		settle_rel.append(res.settle_time_rel5pct_s)
		itae.append(res.itae); iae.append(res.iae); ise.append(res.ise)
		if (not res.diverged) and res.mean_attitude_error_rad <= math.radians(5.0):
			stable += 1
	mean_err = float(np.mean(errs))
	return float(np.mean(rewards)), {
		"mean_reward": float(np.mean(rewards)),
		"mean_attitude_error_rad": mean_err,
		"mean_attitude_error_deg": math.degrees(mean_err),
		# Same key name evaluator.py's metrics dict uses, so a caller can read the
		# triple from either source without knowing which produced it.
		"mean_steady_error_rad": float(np.mean(steady)),
		"mean_steady_error_deg": math.degrees(float(np.mean(steady))),
		"stable_rate": stable / max(num_episodes, 1),
		# Surface the per-episode motor-jerk mean so reward_gated_train can plumb
		# it into Metrics.motor_jerk_mean (the Python path used to drop it →
		# weight_jerk silently ignored; fixed 01/06/2026).
		"mean_pwm_jerk": float(np.mean(jerks)),
		# Transient-speed metrics, averaged over episodes (seconds / natural units).
		"mean_rise_time_s": float(np.mean(rise)),
		"mean_settle_time_abs2deg_s": float(np.mean(settle_abs)),
		"mean_settle_time_rel5pct_s": float(np.mean(settle_rel)),
		"mean_itae": float(np.mean(itae)),
		"mean_iae": float(np.mean(iae)),
		"mean_ise": float(np.mean(ise)),
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


def _eval_closed_loop_residual(
	controller: WnnController, baseline: AttitudePID, cfg: DaggerConfig, num_motors: int,
) -> tuple[float, dict]:
	"""Score the COMPOSED hybrid closed-loop (analytic baseline + learned WNN
	residual), resetting BOTH the WNN recurrent state and the baseline PID's
	integral each episode. This is the number that must clear the baseline's own
	@L2 score (84 for PD / 97 for stock-PID) to prove the residual adds value."""
	base_fn = make_pid_action_fn(baseline)
	action_fn = make_residual_action_fn(
		base_fn, controller, cfg.residual_scale, cfg.residual_clamp, num_motors)

	def _reset():
		controller.reset()
		baseline.reset()

	return eval_closed_loop_reset(
		action_fn, _reset, cfg.episode_config, cfg.eval_episodes, cfg.seed + 7_000_000,
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
		delta_leak=spec.delta_leak, delta_gamma=getattr(spec, 'delta_gamma', 1.0),
		obs_tilt_p=spec.obs_tilt_p, obs_tilt_i=spec.obs_tilt_i, obs_peraxis_p=spec.obs_peraxis_p, obs_peraxis_i=spec.obs_peraxis_i, obs_peraxis_yaw=spec.obs_peraxis_yaw, obs_pwm=spec.obs_pwm, obs_yaw_err=spec.obs_yaw_err, obs_yaw_err_i=spec.obs_yaw_err_i,
		obs_collective_cmd=getattr(spec, 'obs_collective_cmd', False), obs_alt_err=getattr(spec, 'obs_alt_err', False), obs_vz=getattr(spec, 'obs_vz', False),
		obs_pos_err_xy=getattr(spec, 'obs_pos_err_xy', False), obs_vel_xy=getattr(spec, 'obs_vel_xy', False),
		dhat_b=(list(spec.dhat_b) if spec.dhat_b is not None else None), dhat_l_gain=spec.dhat_l_gain, dhat_ff=getattr(spec, 'dhat_ff', False), dhat_ff_clamp=getattr(spec, 'dhat_ff_clamp', 0.30), dt=spec.dt, integral_leak=spec.integral_leak, integral_scale=spec.integral_scale, decouple_outputs=spec.decouple_outputs,
		action_repeat=spec.action_repeat,
		memory_mode=spec.memory_mode_int(),
	)
	if config.residual:
		# E5 residual hybrid: the WNN learns clamp(expert − baseline) on top of the
		# analytic `baseline`. Expert = PID+ (integral ceiling) by default, or an
		# optimal-control teacher (LQR/MPC) that decisively beats PID+.
		pid = make_expert(config.residual_expert, config.episode_config)
		baseline = make_residual_baseline(config.residual_baseline, config.episode_config)
	else:
		pid = AttitudePID(AttitudePIDConfig())
		baseline = None
	nm = spec.num_motors
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
			# W2: arm this episode's disturbance on the TRAINING sim too (not just
			# eval), so the residual learns to reject the L2 bias — PID+ builds its
			# integral against the armed disturbance. No-op when disturbance is None
			# (clean sim = pre-W2 behavior, existing callers unaffected).
			if ec.disturbance is not None:
				apply_disturbance(sim, ec.disturbance, ep_rng)
			pid.reset()
			if baseline is not None:
				baseline.reset()
			controller.reset()

			for _ in range(config.steps_per_episode):
				if sim.is_unstable():
					break
				gyro, accel = sim.read_imu()
				q = sim.quaternion

				# Student forward (advances controller recurrent state) ...
				student_pwm = controller.step(list(gyro), list(accel), list(target))
				# ... expert label at THIS state (PID in absolute mode; PID+ in residual) ...
				expert_pwm = pid.step(q, gyro, target)
				if config.residual:
					# Residual hybrid: teach the WNN clamp(PID+ − baseline) in its
					# own output space; the on-policy action is the COMPOSED hybrid.
					base_pwm = baseline.step(q, gyro, target)
					train_tgt = residual_train_target(expert_pwm, base_pwm,
						config.residual_scale, config.residual_clamp, nm,
						neutral=float(controller.neutral_decode))
					sw, ow = controller.edra_train_step(train_tgt, config.topk_per_neuron)
					student_action = list(compose_residual(base_pwm, student_pwm,
						config.residual_scale, config.residual_clamp, nm,
						neutral=float(controller.neutral_decode)))
				else:
					# ... train the QSR cells toward the expert (Rust beam-search EDRA).
					sw, ow = controller.edra_train_step(list(expert_pwm), config.topk_per_neuron)
					student_action = list(student_pwm)
				cells_written += int(sw) + int(ow)
				stats["train_steps"] += 1

				# DAGGER state distribution: β-mix which action drives the sim.
				# β_0 = 1 → expert drives (BC warm-start); β decays → student drives.
				use_expert = rng.random() < beta
				action = list(expert_pwm) if use_expert else student_action
				sim.step(list(action))
				# mpcof's observer must see the action that was ACTUALLY APPLIED —
				# on-policy, i.e. whatever the β-mix chose to fly, not the teacher's own
				# plan. Without this it observes its own output, the model residual is
				# ~0, d̂ stays ~0, and the arm silently degrades to plain MPC.
				if hasattr(pid, "observe"):
					pid.observe(gyro, action)
				# Delta-control: if the expert drove, sync the controller's throttle
				# integrator to the actually-applied PWM. N/A in residual mode (the WNN
				# emits a signed residual, not an absolute throttle to sync).
				if spec.delta_control and use_expert and not config.residual:
					controller.set_pwm(list(expert_pwm))

		if config.residual:
			fit, metrics = _eval_closed_loop_residual(controller, baseline, config, nm)
		else:
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
