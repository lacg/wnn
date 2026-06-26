"""C1 — reward-gated imitation + connectivity GA for WNN attitude controllers.

WHY THIS EXISTS
---------------
Every IMITATION paradigm tried on the recurrent controller (per-step EDRA,
DAGGER, full/truncated BPTT) tumbles closed-loop (50–90° vs feedforward's
26°). Root cause (see `project_controller_state_2026_05_24`): per-step
imitation trains on PID's per-instant action, which does NOT constrain
closed-loop stability; the recurrence amplifies tiny learned biases into
divergence, and DAGGER then keeps imprinting PID's action onto the
unrecoverable tumbling states the student visits — poisoning the cells.

REWARD-GATING fixes the SIGNAL, not the algorithm. We still imitate PID via
`bptt_train_window`, but a visited state only trains the network if the WHOLE
episode it belonged to ended well (cumulative reward above a gate). Bad
(tumbling) episodes are thrown out, so the cells are only ever shaped by
trajectories that stayed in the recoverable basin. This is the smallest delta
that turns the per-step imitation loss into something filtered by the
closed-loop reward — no trajectory-level credit assignment required.

The student DRIVES the sim during rollout (pure student state distribution,
β=0), so we train exactly where the deployed controller will operate; the gate
is what keeps that from being DAGGER's failure mode.

ARCHITECTURE (CLAUDE.md: no Rust logic reimplemented in Python)
---------------------------------------------------------------
Outer-loop orchestration only — roll out, score, gate, GA. Every per-step hot
op is Rust: AttitudeSim.step, WnnController.step, WnnController.bptt_train_window
(the reachable QSR-EDRA beam-search solver). The only Python per-step cost is
the closed-form PID teacher.

Substrate (locked): absolute-PWM (delta_control=False), full-state structured
connectivity (random_connectivity), 4 state neurons. The GA mutates INPUT bits
only — the forced full-state bits are preserved so the network stays a single
coherent FSM rather than N disjoint mini-automata.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from wnn.control._accel import AttitudeSim, WnnController, compute_reward, monotonicity_violations

from .pid import AttitudePID, AttitudePIDConfig
from .evaluator import ControllerSpec, NUM_FEATURES, random_connectivity
from .training import (
	EpisodeConfig,
	_sample_initial_state,
	_euler_to_quat_xyz,
	make_wnn_action_fn,
)
from .dagger import eval_closed_loop_reset


# ----------------------------------------------------------------------------
# Trajectory recording
# ----------------------------------------------------------------------------

@dataclass
class Trajectory:
	"""One closed-loop student rollout, with PID labels at every visited state.

	gyros/accels/targets are the student's OWN visited state distribution (the
	student drove the sim); pid_pwms are the expert's action queried at each of
	those states — the imitation target. `cumulative_reward` is the gate key.
	"""
	gyros: list[list[float]]
	accels: list[list[float]]
	targets: list[list[float]]
	pid_pwms: list[list[float]]      # expert label at each visited state (C1 target)
	student_pwms: list[list[float]]  # student's OWN action at each state (C2 target)
	cumulative_reward: float
	mean_attitude_error_rad: float
	diverged: bool
	steps: int


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

@dataclass
class RewardGatedConfig:
	"""Reward-gated imitation trainer configuration."""
	num_rounds: int = 8                  # gated training rounds
	episodes_per_round: int = 24         # student rollouts collected per round
	steps_per_episode: int = 2000        # 2 s @ 1 kHz
	bptt_window: int = 32                # truncated-BPTT window (W); carry-state across chunks
	topk_per_neuron: int = 4             # beam-search top-k in the QSR-EDRA solve
	protect_learned: bool = False        # skip nudges that erode a learned cell (memory: HURT — default off)

	# Gate policy knobs (consumed by `episode_passes_gate`).
	gate_mode: str = "improvement"       # "improvement" (self-referential ratchet) | "quantile"
	gate_use_best: bool = False          # improvement mode: gate on running BEST (strict) vs running MEAN
	gate_window: int = 0                 # improvement mode: 0 = all history; N = last-N-episodes window
	gate_quantile: float = 0.5           # quantile mode: train on the top (1 - quantile) fraction; 0.5 = median
	gate_running: bool = True            # quantile mode: pool over ALL rounds (True) vs this round only (False)

	# Inner-loop write rule (the C1-vs-C2 switch):
	#   "pid"     → C1: imitate the EXPERT (PID) action at each visited state.
	#   "student" → C2: reinforce the STUDENT's OWN action ("do more of what
	#               worked") in above-gate episodes. REINFORCE-like; needs
	#               action variance to have something to reinforce — with
	#               deterministic cells the only variance is initial conditions,
	#               so C2 may plateau without light exploration noise (TODO).
	target_source: str = "pid"

	# Best-checkpoint selection: snapshot the controller at its best-fitness inner
	# round and restore it at the end (the chaotic final round is often worse).
	keep_best_checkpoint: bool = True

	# Exploration (C2 only): the student is deterministic given ICs, so
	# reinforce-own-wins has no action variance to select among. With prob
	# explore_eps per step, perturb the applied+recorded PWM by ±explore_scale
	# (uniform) so episodes explore different actions; good episodes then
	# reinforce their explored actions. Leave 0 for C1 (imitate PID exactly).
	explore_eps: float = 0.0
	explore_scale: float = 0.1

	# State-splitting trainer (Phase 5b). When WNN_STATE_SPLIT=1, the per-genome
	# trainer REPLACES truncated-BPTT with the conflict-driven splitting trainer:
	# the round's gated-good trajectories are handed to split_train_loop as a
	# batch (conflicts must be found ACROSS episodes), which constructs state
	# distinctions (event latches / integral counters) and retrains the output.
	# Old bptt path is byte-identical when the flag is off. Knobs = design §11b.
	split_tau: float = 0.1               # conflict PWM-spread threshold
	split_clean_gain: float = 0.999      # Type-1 (latch) vs Type-2 (integral) split
	split_accum_corr: float = 0.9        # min net-count correlation to install a counter
	split_max_rounds: int = 8            # inner split-loop rounds per GA round
	split_k_start: int = 1               # k(round) = k_start + round (greedy→batch)
	split_coarse_target: int = 32        # adaptive coarse-bucket target (0=exact); >0
	#                                      needed on REAL trajectories (exact frames
	#                                      never collide — Phase 5d/5e finding)
	split_selective_output: bool = True  # Phase 6c: only deviate the output where
	#                                      state is ACTIVE; preserve the stable
	#                                      hover-hold at state=0 (wholesale imitation
	#                                      destabilizes — Phase 6b finding)

	# Bootstrap curriculum: ramp the initial-tilt difficulty easy→full across
	# rounds so round 0 (empty cells → holds hover) has a spread of outcomes for
	# the gate to separate. Set curriculum=False for fixed full-difficulty.
	curriculum: bool = True
	easy_tilt_deg: float = 8.0
	full_tilt_deg: float = 30.0

	eval_episodes: int = 20              # closed-loop eval after each round
	seed: int = 0
	progress: bool = True
	target_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
	episode_config: Optional[EpisodeConfig] = None

	def __post_init__(self):
		if self.episode_config is None:
			self.episode_config = EpisodeConfig(
				dt=0.001, steps_per_episode=self.steps_per_episode,
				max_initial_tilt_rad=math.radians(self.full_tilt_deg),
				max_initial_yaw_rad=math.radians(30.0),
				max_initial_body_rate=0.5, max_initial_yaw_rate=0.3,
			)

	def round_tilt_rad(self, it: int) -> float:
		"""Initial-tilt bound for round `it` (radians)."""
		if not self.curriculum or self.num_rounds <= 1:
			return math.radians(self.full_tilt_deg)
		frac = it / (self.num_rounds - 1)
		deg = self.easy_tilt_deg + frac * (self.full_tilt_deg - self.easy_tilt_deg)
		return math.radians(deg)


# ----------------------------------------------------------------------------
# The gate — the heart of C1.
#
# Decides whether a finished episode is "good enough" that imitating PID on the
# states it visited will help (vs poison) the closed-loop policy. This is the
# one place where C1 differs from DAGGER.
#
# DEFAULT = improvement-gated (self-referential ratchet): an episode trains the
# network only if it beat the controller's OWN running performance. As the
# controller improves the bar rises with it, so it never reinforces a rollout
# that's merely "least-bad" — it reinforces rollouts that are above-average FOR
# THE CURRENT POLICY. No absolute standard, no external curriculum needed: the
# controller's own history is the curriculum. ("quantile" mode kept for A/B.)
# ----------------------------------------------------------------------------

def episode_passes_gate(
	score: float,
	round_scores: list[float],
	history_scores: list[float],
	cfg: RewardGatedConfig,
) -> bool:
	"""Return True iff this episode should feed `bptt_train_window`.

	Args:
		score:          this episode's cumulative reward (higher = better).
		round_scores:   all cumulative rewards collected THIS round.
		history_scores: all cumulative rewards collected so far (all rounds).
		cfg:            RewardGatedConfig (gate_mode + its knobs).
	"""
	if cfg.gate_mode == "improvement":
		# Pool = the controller's own recent history (windowed or full). The
		# ratchet: reinforce only episodes at/above what it's been achieving.
		pool = history_scores
		if cfg.gate_window > 0 and len(pool) > cfg.gate_window:
			pool = pool[-cfg.gate_window:]
		if len(pool) < 2:
			return True  # bootstrap: not enough history to judge improvement
		bar = max(pool) if cfg.gate_use_best else float(np.mean(pool))
		return score >= bar

	if cfg.gate_mode == "quantile":
		pool = history_scores if cfg.gate_running else round_scores
		if len(pool) < 2:
			return True
		return score >= float(np.quantile(pool, cfg.gate_quantile))

	raise ValueError(f"unknown gate_mode: {cfg.gate_mode!r}")


# ----------------------------------------------------------------------------
# Rollout + label
# ----------------------------------------------------------------------------

def _rollout_and_label(
	controller: WnnController,
	pid: AttitudePID,
	sim: AttitudeSim,
	ec: EpisodeConfig,
	rng: np.random.Generator,
	target: tuple[float, float, float],
	explore_eps: float = 0.0,
	explore_scale: float = 0.1,
) -> Trajectory:
	"""Roll the STUDENT closed-loop, recording its visited states + PID labels.

	The student drives the sim (β=0 — pure student distribution). At each step
	we query PID at the student's state for the imitation target. Returns the
	trajectory + its cumulative reward (the gate key).

	If explore_eps>0 (C2): the applied+recorded action is the student's PWM
	perturbed by ±explore_scale (per motor, with prob explore_eps), so episodes
	explore different actions and good ones reinforce what they actually did.
	"""
	init_q, init_omega = _sample_initial_state(
		rng, ec.max_initial_tilt_rad, ec.max_initial_yaw_rad,
		ec.max_initial_body_rate, ec.max_initial_yaw_rate,
	)
	sim.reset(q=list(init_q), omega=list(init_omega))
	pid.reset()
	controller.reset()
	target_q = _euler_to_quat_xyz(*target)

	gyros: list[list[float]] = []
	accels: list[list[float]] = []
	targets: list[list[float]] = []
	pid_pwms: list[list[float]] = []
	student_pwms: list[list[float]] = []
	cumulative = 0.0
	sum_err = 0.0
	steps = 0
	diverged = False

	for t in range(ec.steps_per_episode):
		if sim.is_unstable():
			diverged = True
			break
		gyro, accel = sim.read_imu()
		q = sim.quaternion
		# Student forward (advances recurrent state) — student drives the sim.
		student_pwm = controller.step(list(gyro), list(accel), list(target))
		# Expert label at THIS (student-visited) state.
		expert_pwm = pid.step(q, gyro, target)

		# Exploration: perturb the action the student actually takes (and that
		# C2 will reinforce). C1 (explore_eps=0) leaves student_pwm untouched.
		applied = [float(p) for p in student_pwm]
		if explore_eps > 0.0:
			for m in range(len(applied)):
				if rng.random() < explore_eps:
					applied[m] = float(np.clip(applied[m] + rng.uniform(-explore_scale, explore_scale), 0.0, 1.0))

		gyros.append([float(gyro[0]), float(gyro[1]), float(gyro[2])])
		accels.append([float(accel[0]), float(accel[1]), float(accel[2])])
		targets.append([float(target[0]), float(target[1]), float(target[2])])
		pid_pwms.append([float(expert_pwm[0]), float(expert_pwm[1]),
		                 float(expert_pwm[2]), float(expert_pwm[3])])
		student_pwms.append(applied)

		sim.step(applied)
		attitude_err = sim.attitude_error(target_q)
		cumulative += compute_reward(attitude_err, motor_command_jerk=0.0,
		                             mono_violations=0, lambda_smooth=0.0, lambda_mono=0.0)
		sum_err += attitude_err
		steps = t + 1

	return Trajectory(
		gyros=gyros, accels=accels, targets=targets,
		pid_pwms=pid_pwms, student_pwms=student_pwms,
		cumulative_reward=float(cumulative),
		mean_attitude_error_rad=sum_err / max(steps, 1),
		diverged=diverged, steps=steps,
	)


def _train_on_trajectory(controller: WnnController, traj: Trajectory, cfg: RewardGatedConfig) -> tuple[int, int]:
	"""Truncated-BPTT imitation over one (gated-good) trajectory.

	Chunks the episode into windows of W = cfg.bptt_window and calls
	`bptt_train_window` per chunk with carry-state (reset_state only on the first
	chunk) — this is the truncated BPTT the memory found beats full-episode
	(credit washes out over 2000 steps) and per-step EDRA (chaotic on recurrence).
	"""
	W = cfg.bptt_window
	n = traj.steps
	# C1 imitates the expert (PID); C2 reinforces the student's own action.
	targets_pwm = traj.student_pwms if cfg.target_source == "student" else traj.pid_pwms
	s_writes = o_writes = 0
	first = True
	for start in range(0, n, W):
		g = traj.gyros[start:start + W]
		a = traj.accels[start:start + W]
		tg = traj.targets[start:start + W]
		pp = targets_pwm[start:start + W]
		if not g:
			continue
		sw, ow = controller.bptt_train_window(
			g, a, tg, pp, cfg.topk_per_neuron,
			first,                       # reset_state on first chunk, carry after
			cfg.protect_learned,
		)
		s_writes += int(sw)
		o_writes += int(ow)
		first = False
	return s_writes, o_writes


# ----------------------------------------------------------------------------
# Inner loop — reward-gated imitation
# ----------------------------------------------------------------------------

def reward_gated_train(
	spec: ControllerSpec,
	thresholds: list[float],
	state_connections: list[int],
	output_connections: list[int],
	config: RewardGatedConfig,
	init_state_cells: list | None = None,
	init_output_cells: list | None = None,
) -> tuple[WnnController, dict]:
	"""Train a controller by reward-gated imitation. Returns (controller, stats).

	Builds a controller from the supplied connectivity, then for each round:
	rolls out the student, scores each episode, and runs truncated-BPTT imitation
	ONLY on the episodes that pass the gate. The same controller is trained in
	place so its rollout distribution improves across rounds (gated DAGGER-style
	aggregation).

	Warm-start (Lamarckism): if init_state_cells / init_output_cells are given
	(lists of (neuron, address, value) triples), they're written into the fresh
	controller BEFORE training — so training continues from inherited memory
	rather than empty cells. This is what makes Lamarckian write-back affect
	fitness (the acquired memory is carried into the next generation's training).
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
		obs_tilt_p=spec.obs_tilt_p, obs_tilt_i=spec.obs_tilt_i, obs_peraxis_p=spec.obs_peraxis_p, obs_peraxis_i=spec.obs_peraxis_i, obs_peraxis_yaw=spec.obs_peraxis_yaw, obs_pwm=spec.obs_pwm, integral_leak=spec.integral_leak, integral_scale=spec.integral_scale, decouple_outputs=spec.decouple_outputs,
	)
	# Warm-start from inherited cells (Lamarckian), if supplied.
	# Defense-in-depth: the controller's cell memory is u64-keyed, but the PyO3
	# write_*_cell(address: u64) raises OverflowError at CONVERSION for addr ≥ 2^64
	# (a neuron with >64 bits has a nominal space wider than u64). Such addresses
	# are physically unaddressable, so skip them here — _filter_inherited_cells caps
	# at u64 upstream, this is the belt-and-suspenders at the actual call site.
	_U64 = 1 << 64
	for (n, addr, v) in (init_state_cells or []):
		a = int(addr)
		if 0 <= a < _U64:
			controller.write_state_cell(int(n), a, int(v))
	for (n, addr, v) in (init_output_cells or []):
		a = int(addr)
		if 0 <= a < _U64:
			controller.write_output_cell(int(n), a, int(v))
	pid = AttitudePID(AttitudePIDConfig())
	sim = AttitudeSim()
	rng = np.random.default_rng(config.seed)
	base_ec = config.episode_config
	target = config.target_rpy

	history_scores: list[float] = []
	stats = {
		"iter_fitness": [], "iter_mean_err_deg": [], "iter_stable_rate": [],
		"iter_tilt_deg": [], "iter_n_trained": [], "iter_cells_written": [],
		"iter_mean_episode_reward": [], "train_steps": 0,
		# 01/06/2026: plumb the jerk + mono fitness metrics that the harmonic-rank
		# calculator needs (the Python path used to omit these → weight_jerk /
		# weight_mono silently ignored; the Rust path already produced them).
		"iter_motor_jerk_mean": [], "iter_mono_violations": [],
	}
	# Best-checkpoint (B): remember the cells at the best-fitness round.
	best_fit_so_far = float("-inf")
	best_snapshot = None

	# State-splitting trainer (Phase 5b), flag-gated. When OFF the training step
	# below is the byte-identical legacy bptt path. When ON, the round's gated
	# trajectories train via split_train_loop and the trainer's GA-handshake
	# pressure (saturation + connectivity wishes) accumulates in stats for the
	# caller (the GA) to feed back into mutation (Phase 5c).
	use_split = os.environ.get("WNN_STATE_SPLIT") == "1"
	split_saturation = 0
	split_wishes: set[int] = set()

	# 31/05/2026: cooperative-cancellation poll between rounds. Each round is
	# ~few hundred ms to a few seconds; checking at the round boundary gives
	# round-level cancel granularity (~100ms-1s response in practice).
	try:
		from wnn.control import _accel as _ra
		_check_cancel = _ra.is_cancelled
	except Exception:
		_check_cancel = lambda: False

	for it in range(config.num_rounds):
		if _check_cancel():
			break
		# Round-specific IC difficulty (curriculum bootstrap).
		ec = EpisodeConfig(
			dt=base_ec.dt, steps_per_episode=config.steps_per_episode,
			max_initial_tilt_rad=config.round_tilt_rad(it),
			max_initial_yaw_rad=base_ec.max_initial_yaw_rad,
			max_initial_body_rate=base_ec.max_initial_body_rate,
			max_initial_yaw_rate=base_ec.max_initial_yaw_rate,
		)

		# 1. Roll out the student, recording trajectories + scores.
		trajs: list[Trajectory] = []
		for _ in range(config.episodes_per_round):
			# Per-episode cancel poll: cheap (one atomic load per episode),
			# gives sub-100ms response inside the rollout phase. Skips
			# remaining episodes this round; the round loop then exits at
			# its own poll on the next iteration.
			if _check_cancel():
				break
			ep_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
			trajs.append(_rollout_and_label(controller, pid, sim, ec, ep_rng, target,
			                                config.explore_eps, config.explore_scale))
		round_scores = [t.cumulative_reward for t in trajs]

		# 2. Gate + 3. train on the survivors.
		n_trained = 0
		cells_written = 0
		if use_split:
			# State-splitting trainer: the WHOLE gated batch at once, so conflicts
			# are found ACROSS episodes (state must disambiguate histories that the
			# memoryless output cannot). split_train_loop constructs the state +
			# retrains the output in place, and reports GA-handshake pressure.
			gated = [t for t in trajs
			         if episode_passes_gate(t.cumulative_reward, round_scores, history_scores, config)]
			if gated:
				(_r, _cf, planted, _pr, saturation, wishes) = controller.split_train_loop(
					[t.gyros for t in gated], [t.accels for t in gated],
					[t.targets for t in gated], [t.pid_pwms for t in gated],
					config.split_tau, config.split_clean_gain, config.split_accum_corr,
					config.split_max_rounds, config.split_k_start, config.split_coarse_target,
					config.split_selective_output,
				)
				cells_written = int(planted)
				n_trained = len(gated)
				stats["train_steps"] += sum(t.steps for t in gated)
				split_saturation += int(saturation)
				split_wishes.update(int(w) for w in wishes)
		else:
			# Legacy truncated-BPTT path (byte-identical to pre-Phase-5b).
			for traj in trajs:
				if episode_passes_gate(traj.cumulative_reward, round_scores, history_scores, config):
					sw, ow = _train_on_trajectory(controller, traj, config)
					cells_written += sw + ow
					n_trained += 1
					stats["train_steps"] += traj.steps
		history_scores.extend(round_scores)

		# 4. Closed-loop eval (student drives, fresh recurrent state per episode).
		fit, metrics = eval_closed_loop_reset(
			make_wnn_action_fn(controller), controller.reset,
			base_ec, config.eval_episodes, config.seed + 5_000_000,
		)
		stats["iter_fitness"].append(float(fit))
		stats["iter_mean_err_deg"].append(float(metrics["mean_attitude_error_deg"]))
		stats["iter_stable_rate"].append(float(metrics["stable_rate"]))
		stats["iter_tilt_deg"].append(math.degrees(ec.max_initial_tilt_rad))
		stats["iter_n_trained"].append(n_trained)
		stats["iter_cells_written"].append(cells_written)
		stats["iter_mean_episode_reward"].append(float(np.mean(round_scores)))
		# Jerk: mean Σ(Δpwm)² over the eval rollouts (surfaced by eval_closed_loop_reset).
		stats["iter_motor_jerk_mean"].append(float(metrics.get("mean_pwm_jerk", 0.0)))
		# Mono: thermometer-monotonicity violations on the trained output cells
		# (a static property of the memory; same call the Rust path uses).
		try:
			mono = monotonicity_violations(controller.get_last_output_cells(),
			                               spec.levels_per_motor, spec.num_motors)
			stats["iter_mono_violations"].append(float(mono))
		except Exception:
			stats["iter_mono_violations"].append(0.0)

		# Best-checkpoint (B): snapshot whenever this round is the new best.
		if config.keep_best_checkpoint and fit > best_fit_so_far:
			best_fit_so_far = fit
			best_snapshot = controller.export_cells()

		if config.progress:
			print(
				f"  RG iter {it + 1}/{config.num_rounds}: "
				f"tilt={math.degrees(ec.max_initial_tilt_rad):.0f}°  "
				f"trained {n_trained}/{config.episodes_per_round} eps  "
				f"closed-loop fitness={fit:.2f}  "
				f"mean_err={metrics['mean_attitude_error_deg']:.2f}°  "
				f"stable={metrics['stable_rate'] * 100:.0f}%  cells+={cells_written}"
			)

	# Restore the best-round cells so the returned controller is the best
	# checkpoint, not the (often-worse) chaotic final round.
	if config.keep_best_checkpoint and best_snapshot is not None:
		controller.restore_cells(best_snapshot[0], best_snapshot[1])

	# State-splitting GA-handshake pressure (Phase 5b → consumed by 5c mutation).
	# saturation = #conflicts needing more state than available (grow state_neurons);
	# wish_bits = state-input positions a separator wanted but no neuron observed.
	stats["split_saturation"] = int(split_saturation)
	stats["split_wish_bits"] = sorted(split_wishes)

	best_idx = int(np.argmax(stats["iter_fitness"])) if stats["iter_fitness"] else -1
	stats["best_iter"] = best_idx
	stats["best_fitness"] = stats["iter_fitness"][best_idx] if best_idx >= 0 else float("-inf")
	stats["final_fitness"] = stats["iter_fitness"][-1] if stats["iter_fitness"] else float("-inf")
	return controller, stats


__all__ = [
	"Trajectory",
	"RewardGatedConfig",
	"episode_passes_gate",
	"reward_gated_train",
]
