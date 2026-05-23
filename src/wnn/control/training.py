"""Episode runner + GA orchestration for WNN attitude controllers.

Stays in Python (per `project_drone_controller_paper1.md`): outer-loop
orchestration is cheap and benefits from Python's ecosystem (numpy,
matplotlib for diagnostics, pytest for unit tests). The per-step hot
path (sim physics, controller forward, decoder, reward) is in Rust —
see `wnn.control.sim`, `.controller`, `.decoders`, and `compute_reward`
re-exported below.

The GA itself reuses the existing pipeline in
`src/wnn/ram/experiments/` (same one IDS flows use). Controller-specific
glue is the episode runner + fitness function that calls into Rust.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ram_accelerator import (
	AttitudeSim,
	WnnController,
	compute_reward,
	monotonicity_violations,
)


@dataclass
class EpisodeConfig:
	"""Configuration for a single training episode."""

	# Sim
	dt: float = 0.001                       # 1 ms = 1 kHz update
	steps_per_episode: int = 2000           # 2 s of simulated flight
	max_initial_tilt_rad: float = math.radians(60.0)

	# Reward weights
	lambda_smooth: float = 0.0              # motor-jerk penalty (off by default)
	lambda_mono: float = 0.0                # output thermometer monotonicity penalty


def run_episode(controller: WnnController,
                sim: AttitudeSim,
                config: EpisodeConfig,
                target_attitude: tuple[float, float, float] = (0.0, 0.0, 0.0),
                rng: np.random.Generator | None = None) -> float:
	"""Run one episode and return the cumulative reward.

	Args:
		controller:     a WnnController (Rust) instance, already reset.
		sim:            an AttitudeSim (Rust) instance.
		config:         episode configuration.
		target_attitude: (roll, pitch, yaw) setpoint in radians.
		rng:            optional numpy RNG for reproducible initial conditions.

	Returns:
		cumulative_reward: sum of per-step rewards.
	"""
	# TODO: sample random initial attitude within the tilt cone, reset sim+controller,
	# run the inner loop calling sim.read_imu → controller.step → sim.step → reward.
	# Accumulate reward; return.
	raise NotImplementedError(
		"Episode runner stub. Implementation lands once the Rust sim physics + "
		"controller forward pass + thermometer encoding are real (currently stubs)."
	)


def fitness_function(controller: WnnController,
                     config: EpisodeConfig,
                     num_episodes: int = 30,
                     seed: int = 0) -> float:
	"""Average cumulative reward across `num_episodes` random initial conditions.

	Used as the GA's fitness signal. Reuses the existing GA pipeline in
	`src/wnn/ram/experiments/` — this function plugs into wherever the
	IDS pipeline currently computes per-genome fitness from a dataset.
	"""
	# TODO: implement
	raise NotImplementedError
