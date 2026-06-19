"""Open-loop teacher-forcing diagnostic for WNN controllers.

Localizes closed-loop control failure: covariate shift (DAGGER-fixable)
vs representation/training bug.

Method: run an episode with PID driving the sim (teacher-forcing). At
each step the WNN ALSO predicts an action, but the sim advances with
PID's action. Measure the per-step PWM prediction error |WNN - PID| on
PID's OWN state distribution (the distribution the WNN trained on).

  - Small open-loop error + failing closed-loop → COVARIATE SHIFT.
    The WNN learned the mapping; closed-loop errors compound into
    unseen states. Fix: DAGGER (retrain on the learner's visited states).
  - Large open-loop error → representation/training bug. The WNN never
    learned the mapping; chase the training, not the rollout regime.

24/05/2026 result on a reservoir+output-trained controller:
  open-loop error 0.022 (tiny), closed-loop 23.8° (fails) → COVARIATE SHIFT.
"""

from __future__ import annotations

import numpy as np

from .training import EpisodeConfig, _sample_initial_state
from .pid import AttitudePID
from wnn.control._accel import AttitudeSim


def open_loop_prediction_error(
	controller,
	config: EpisodeConfig,
	num_episodes: int = 10,
	seed: int = 999,
	pid: AttitudePID | None = None,
	delta_control: bool = False,
) -> dict:
	"""Run teacher-forced episodes; return WNN-vs-PID PWM prediction error.

	Args:
		controller: a (trained) WnnController.
		config:     EpisodeConfig (initial-condition bounds + steps).
		num_episodes, seed: reproducible test episodes.
		pid:        the teacher PID (default AttitudePID()).
		delta_control: if True, the controller outputs a per-step DELTA from its
			throttle accumulator. We sync the accumulator to PID's previous
			throttle each step (via set_pwm), so the error measures how well the
			controller's DELTA matches PID's ideal delta on PID's own state
			distribution (the meaningful open-loop quantity for delta-control).

	Returns dict with mean/max/std per-step |WNN_pwm - PID_pwm| and a
	verdict string.
	"""
	pid = pid or AttitudePID()
	sim = AttitudeSim()
	rng = np.random.default_rng(seed)
	errs: list[float] = []
	for _ in range(num_episodes):
		er = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
		q0, w0 = _sample_initial_state(
			er, config.max_initial_tilt_rad, config.max_initial_yaw_rad,
			config.max_initial_body_rate, config.max_initial_yaw_rate,
		)
		sim.reset(q=list(q0), omega=list(w0))
		pid.reset()
		controller.reset()
		target = (0.0, 0.0, 0.0)
		pid_prev = [0.5, 0.5, 0.5, 0.5]  # delta baseline: PID's previous throttle
		for _ in range(config.steps_per_episode):
			if sim.is_unstable():
				break
			gyro, accel = sim.read_imu()
			q = sim.quaternion
			if delta_control:
				# Apply the controller's delta from PID's current throttle, so
				# wnn_act = pid_prev + ctrl_delta and the error is
				# |ctrl_delta - (pid_act - pid_prev)| = delta-prediction error.
				controller.set_pwm(list(pid_prev))
			wnn_act = controller.step(list(gyro), list(accel), list(target))
			pid_act = pid.step(q, gyro, target)
			errs.append(float(np.mean(np.abs(np.array(wnn_act) - np.array(pid_act)))))
			sim.step(list(pid_act))  # teacher-forcing
			pid_prev = list(pid_act)
	errs_arr = np.array(errs)
	mean_err = float(errs_arr.mean())
	verdict = (
		"COVARIATE_SHIFT (DAGGER would fix)" if mean_err < 0.06
		else "REPRESENTATION_BUG (WNN doesn't reproduce PID even open-loop)"
	)
	return {
		"mean_pwm_error": mean_err,
		"max_pwm_error": float(errs_arr.max()),
		"std_pwm_error": float(errs_arr.std()),
		"n_steps": int(errs_arr.size),
		"verdict": verdict,
	}


__all__ = ["open_loop_prediction_error"]
