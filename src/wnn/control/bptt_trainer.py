"""BPTT/EDRA trainer for WNN attitude controllers — Python reference impl.

This is the validation reference for the controller training algorithm. It
plugs into the existing `RAMRecurrentNetwork` (from `wnn.ram.core`) and the
`Memory.solve_constraints` beam-search constraint solver.

The Rust port (`controller_training.rs`) follows this as its specification.
The fast path will eventually be in Rust, but for paper-#1 validation +
algorithm soundness we run the Python version end-to-end first.

Pipeline per training episode:
  1. Reset sim + PID + network
  2. For step t = 0 ... T-1:
     a. Read sensors from sim (gyro, accel, target)
     b. Encode as bits via per-feature thermometer
     c. Get PID's action (the teacher)
     d. Encode action as 1024 target bits (4 motors × 256 levels,
        thermometer-encoded — bit i of motor m is TRUE iff
        target_pwm[m] * levels >= i)
     e. Call network.train([encoded_window], target_bits) — single-step
        EDRA (no multi-step BPTT in this minimal version; the recurrent
        state still propagates through forward passes)
     f. Apply PID action to sim (DAGGER-style — follow the teacher,
        not the student-in-training)
  3. After all episodes done, network has trained cells.

Why single-step (not multi-step BPTT-through-time): RAMRecurrentNetwork's
multi-window train() requires that consecutive windows are CONSECUTIVE
timesteps with state evolution between them — but it also RESETS the
recurrent state at the start of every train() call. So for true multi-step
BPTT we'd need to feed K windows in one train() call. Doable but complex.
For paper #1 we validate single-step first; multi-step is an optimization.

Limitation: the Python RAMRecurrentNetwork uses TRINARY memory
(FALSE/TRUE/EMPTY) while the Rust WnnController uses QUAD_WEIGHTED. So
we can't directly load Python-trained cells into the Rust controller
without a value mapping (FALSE→FALSE, TRUE→TRUE, EMPTY→WEAK_FALSE). The
mapping IS provided here (cells_to_rust_controller) but produces a
controller without the WEAK_* gradations the QSR architecture relies on
for partial-confidence outputs. So the loaded controller is a strict
subset of what BPTT could produce. For full Rust-native QSR training,
the Rust port is needed (project_drone_controller_paper1.md TODO).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from wnn.control._accel import AttitudeSim, WnnController

from .pid import AttitudePID, AttitudePIDConfig
from .evaluator import (
	ControllerSpec, ControllerGenome, NUM_FEATURES,
	fit_thresholds_from_pid_rollouts, random_connectivity,
)
from .training import (
	EpisodeConfig, _euler_to_quat_xyz, _sample_initial_state,
)


# Memory value constants matching wnn.ram.core.Memory's TRINARY scheme.
MEM_FALSE = 0
MEM_TRUE = 1
MEM_EMPTY = 2


@dataclass
class BPTTConfig:
	"""Training-loop configuration."""
	num_episodes: int = 50
	steps_per_episode: int = 2000
	episode_config: EpisodeConfig = None  # type: ignore[assignment]
	rng_seed: int = 0
	progress_every: int = 10
	# Single-step EDRA per timestep (True) vs multi-step BPTT through K windows (False).
	# False NOT YET IMPLEMENTED — see module docstring.
	single_step: bool = True

	def __post_init__(self):
		if self.episode_config is None:
			self.episode_config = EpisodeConfig(
				dt=0.001, steps_per_episode=self.steps_per_episode,
				max_initial_tilt_rad=math.radians(20.0),
				max_initial_yaw_rad=math.radians(20.0),
				max_initial_body_rate=0.3, max_initial_yaw_rate=0.2,
			)


def encode_sensors_thermometer(
	sensors: Tuple[float, ...],
	thresholds: list[float],
	bits_per_feature: int,
) -> Tensor:
	"""Per-feature thermometer encoding. Sensor i with value v_i produces
	bits [v_i >= thresholds[i*bpf], v_i >= thresholds[i*bpf+1], ...].

	Returns a 1D bool tensor of length NUM_FEATURES * bits_per_feature.
	"""
	bits = torch.zeros(NUM_FEATURES * bits_per_feature, dtype=torch.bool)
	for f in range(NUM_FEATURES):
		v = sensors[f]
		row = f * bits_per_feature
		for b in range(bits_per_feature):
			bits[row + b] = v >= thresholds[row + b]
	return bits


def encode_action_thermometer(
	pwm: Tuple[float, float, float, float],
	num_motors: int,
	levels_per_motor: int,
) -> Tensor:
	"""Thermometer-encode 4-motor PWM action as 1024-bit target.

	For motor m with pwm p ∈ [0, 1], bit (m * levels + i) is TRUE iff
	p * levels >= (i + 1). This puts the boundary correctly: e.g., p=0.5,
	levels=256 → first 128 bits TRUE, rest FALSE.
	"""
	bits = torch.zeros(num_motors * levels_per_motor, dtype=torch.bool)
	for m in range(num_motors):
		p = max(0.0, min(1.0, pwm[m]))
		num_true = int(p * levels_per_motor)
		bits[m * levels_per_motor : m * levels_per_motor + num_true] = True
	return bits


def build_python_network(spec: ControllerSpec, seed: int = 0):
	"""Construct a Python RAMRecurrentNetwork mirroring the controller spec.

	The Python network uses TRINARY memory; the input layer reads the
	current frame's encoded bits (no K-windowing for the minimal version);
	the state layer adds the recurrent state bits.

	Uses OutputMode.RAW so target_bits flow through as-is (the default
	HAMMING decoder expects a scalar class index, which doesn't apply to
	controller PWM bit-vectors).
	"""
	from wnn.ram.core.recurrent_network import RAMRecurrentNetwork
	from wnn.ram.encoders_decoders import OutputMode

	# Input bits: current frame only (no K-windowing in the minimal Python
	# trainer — the recurrent state provides temporal context). The Rust
	# controller's K>1 windowing is a separate optimization.
	input_bits = NUM_FEATURES * spec.bits_per_feature
	n_output_neurons = spec.num_motors * spec.levels_per_motor
	net = RAMRecurrentNetwork(
		input_bits=input_bits,
		n_state_neurons=spec.state_neurons,
		n_output_neurons=n_output_neurons,
		n_bits_per_state_neuron=spec.state_bits_per_neuron,
		n_bits_per_output_neuron=spec.output_bits_per_neuron,
		use_hashing=False,
		rng=seed,
		output_mode=OutputMode.RAW,
	)
	return net


def train_bptt(
	spec: ControllerSpec,
	thresholds: list[float],
	bptt_config: BPTTConfig,
) -> tuple:
	"""Run the BPTT/EDRA training loop. Returns (trained_network, run_stats).

	The network is updated in place; the returned object IS the trained
	network. run_stats is a dict with per-episode telemetry.
	"""
	if not bptt_config.single_step:
		raise NotImplementedError("Multi-step BPTT through K-window not yet implemented in Python trainer; the Rust port will land it.")

	net = build_python_network(spec, seed=bptt_config.rng_seed)
	sim = AttitudeSim()
	pid = AttitudePID(AttitudePIDConfig())

	rng = np.random.default_rng(bptt_config.rng_seed)
	stats = {
		"episode_rewards": [],
		"episode_mean_errs_deg": [],
		"episode_diverged": [],
		"train_calls": 0,
		"cell_writes": 0,
	}

	for ep_idx in range(bptt_config.num_episodes):
		ep_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
		init_q, init_omega = _sample_initial_state(
			ep_rng,
			bptt_config.episode_config.max_initial_tilt_rad,
			bptt_config.episode_config.max_initial_yaw_rad,
			bptt_config.episode_config.max_initial_body_rate,
			bptt_config.episode_config.max_initial_yaw_rate,
		)
		sim.reset(q=list(init_q), omega=list(init_omega))
		pid.reset()
		target = (0.0, 0.0, 0.0)

		ep_reward = 0.0
		sum_err = 0.0
		steps_done = 0
		diverged = False

		for _ in range(bptt_config.steps_per_episode):
			if sim.is_unstable():
				diverged = True
				break

			gyro, accel = sim.read_imu()
			q = sim.quaternion
			target_pwm = pid.step(q, gyro, target)

			# Build sensor input + target bits.
			sensors = (
				gyro[0], gyro[1], gyro[2],
				accel[0], accel[1], accel[2],
				target[0], target[1], target[2],
			)
			input_bits = encode_sensors_thermometer(
				sensors, thresholds, spec.bits_per_feature
			)
			target_bits = encode_action_thermometer(
				target_pwm, spec.num_motors, spec.levels_per_motor
			)

			# EDRA train step. input_bits shape: [input_bits], target shape: [1, n_output].
			# Wrap to the shapes the existing train() expects: list of windows.
			windows = [input_bits.unsqueeze(0)]
			target_bits_2d = target_bits.unsqueeze(0)
			net.train(windows, target_bits_2d)
			stats["train_calls"] += 1

			# Advance sim with PID action (DAGGER).
			sim.step(list(target_pwm))

			# Track stats.
			err = sim.attitude_error(_euler_to_quat_xyz(*target))
			ep_reward -= err * err
			sum_err += err
			steps_done += 1

		mean_err = sum_err / max(steps_done, 1)
		stats["episode_rewards"].append(ep_reward)
		stats["episode_mean_errs_deg"].append(math.degrees(mean_err))
		stats["episode_diverged"].append(diverged)

		if (ep_idx + 1) % bptt_config.progress_every == 0:
			recent_reward = np.mean(stats["episode_rewards"][-bptt_config.progress_every:])
			recent_err = np.mean(stats["episode_mean_errs_deg"][-bptt_config.progress_every:])
			print(
				f"  ep {ep_idx + 1}/{bptt_config.num_episodes}: "
				f"recent_mean_reward={recent_reward:.2f}, recent_mean_err={recent_err:.2f}°, "
				f"train_calls={stats['train_calls']}"
			)

	# Count actual cell writes (each train() writes some, some skipped).
	state_cells_written = 0
	output_cells_written = 0
	for n in range(spec.state_neurons):
		row = net.state_layer.memory.get_memory_row(n)
		state_cells_written += int((row != MEM_EMPTY).sum().item())
	for n in range(spec.num_motors * spec.levels_per_motor):
		row = net.output_layer.memory.get_memory_row(n)
		output_cells_written += int((row != MEM_EMPTY).sum().item())
	stats["state_cells_filled"] = state_cells_written
	stats["output_cells_filled"] = output_cells_written

	return net, stats


def python_network_action_fn(net):
	"""Build an ActionFn (compatible with run_episode) from a trained
	Python RAMRecurrentNetwork. Decodes via Strategy-1 (count TRUE) per
	motor — simpler than QSR Strategy-5 since Python is TRINARY-only.
	"""
	from .evaluator import NUM_FEATURES
	# These need to be in closure scope — pulled from net at call time.
	def fn(gyro, accel, target_rpy, q):
		# Need network thresholds passed in. We attach them to the net for convenience.
		thresholds = net._controller_thresholds  # type: ignore[attr-defined]
		spec = net._controller_spec              # type: ignore[attr-defined]
		sensors = (gyro[0], gyro[1], gyro[2], accel[0], accel[1], accel[2],
		           target_rpy[0], target_rpy[1], target_rpy[2])
		input_bits = encode_sensors_thermometer(sensors, thresholds, spec.bits_per_feature)
		# Forward through the network.
		_, _, _, out_bits = net._get_outputs(input_bits.unsqueeze(0), update_state=True)
		# out_bits is [1, n_output_neurons] bool. Per-motor count-TRUE decode.
		out = out_bits[0]
		levels = spec.levels_per_motor
		pwm = []
		for m in range(spec.num_motors):
			n_true = int(out[m * levels : (m + 1) * levels].sum().item())
			pwm.append(n_true / levels)
		return tuple(pwm)
	return fn


__all__ = [
	"BPTTConfig",
	"encode_sensors_thermometer",
	"encode_action_thermometer",
	"build_python_network",
	"train_bptt",
	"python_network_action_fn",
	"MEM_FALSE", "MEM_TRUE", "MEM_EMPTY",
]
