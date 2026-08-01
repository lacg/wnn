"""Score the classical reference controllers (PID/LQR/MPC/LQI/MPCOF) on a
held-out draw, flying EXACTLY the episodes the WNN scorer flies.

WHY THIS EXISTS. A baseline is only a comparator if it sees the same aircraft on
the same episodes. Three ways to get that wrong were found on 29/07/2026, each
worth roughly 10pp of PID stable%:

  * initial conditions drawn from the RAW report seed instead of the fold-0 pool
    (ControllerEvaluator swaps in `_fold_seeds[fold]` before scoring — evaluator.py:682 —
    so a held-out report always lands on fold 0, never on the raw seed);
  * the disturbance stream seed not XOR'd with the pool seed;
  * `d.motor_asym` passed raw — the fixed (1,1,1,1) multiplier — so the baseline
    flew a PERFECTLY SYMMETRIC quadrotor while every WNN cell carries an ~8% weak
    motor, which is the defect L2D exists to model.

`scripts/compute_baselines.py` was fixed for all three; `phased_ga._pid_baseline`
was not, and kept printing `vs PID 85.0%` in every cell banner against a true
90.4±7.5. Two implementations of "the comparator" is one too many — this module is
the single one, and both callers use it.

NOT a substitute for `eval_closed_loop_reset`: that path resets the policy per
episode and reports transient metrics (rise/settle/ITAE) this one does not. It is
the right tool for characterising a controller, and the wrong tool for producing a
number to sit next to a WNN row.
"""
from dataclasses import dataclass

_NAMES = {0: "PID", 1: "LQR", 2: "MPC", 3: "LQI", 4: "MPCOF"}


@dataclass(frozen=True)
class HoldoutDraw:
	"""The held-out episode set a baseline must be scored on.

	eval_folds/fold_index mirror the scoring run: K=1 keeps the raw seed (there is
	no pool), K>1 lands on the pool the evaluator actually samples from.
	"""
	seed: int
	episodes: int
	steps: int
	stable_deg: float = 5.0
	eval_folds: int = 5
	fold_index: int = 0

	def pool_seed(self) -> int:
		"""The seed the WNN scorer samples ICs and the disturbance stream from."""
		from .evaluator import fold_pool_seed
		if self.eval_folds <= 1:
			return self.seed
		return fold_pool_seed(self.seed, self.fold_index)


def _dist_fields(d, seed: int, motor_asym) -> dict:
	"""The 12 disturbance kwargs `score_classical_baseline` takes, from a
	DisturbanceConfig — mirrors evaluator._dist_packed_fields.

	motor_asym MUST be the RESOLVED per-airframe draw, never `d.motor_asym`: the
	raw field is the fixed multiplier and carries none of the asymmetry.
	"""
	asym = motor_asym if motor_asym is not None else d.motor_asym
	return dict(
		dist_enabled=True,
		dist_tau_bias=[float(x) for x in d.tau_bias],
		dist_gust_sigma=float(d.gust_sigma), dist_gust_tau_c=float(d.gust_tau_c),
		dist_motor_asym=[float(x) for x in asym],
		dist_gyro_sigma=float(d.gyro_sigma), dist_gyro_bias_walk=float(d.gyro_bias_walk),
		dist_accel_sigma=float(d.accel_sigma), dist_seed=int(seed),
		dist_dropout_prob=float(d.dropout_prob),
		dist_dropout_len_steps=int(d.dropout_len_steps),
		dist_obs_delay_steps=int(d.obs_delay_steps),
		dist_torque_scale_jitter=float(d.torque_scale_jitter))


def _episode_fields(ec, draw: HoldoutDraw) -> tuple:
	"""(q0, w0, disturbance kwargs) for the draw — the pool-seeded episode set."""
	from .evaluator import disturbance_stream
	from .training import sample_ics_flat
	pool = draw.pool_seed()
	q0, w0 = sample_ics_flat(pool, draw.episodes, ec)
	dseed, asym = disturbance_stream(ec.disturbance, pool)
	return q0, w0, _dist_fields(ec.disturbance, dseed, asym)


def score_all(ec, draw: HoldoutDraw) -> dict:
	"""{name: (stable%, err°, steady°)} for all 5 classical controllers."""
	from ._accel import score_classical_baseline
	q0, w0, fields = _episode_fields(ec, draw)
	out = {}
	for tid, name in _NAMES.items():
		st, err, steady = score_classical_baseline(
			tid, list(q0), list(w0), draw.steps, draw.stable_deg, **fields)
		out[name] = (st * 100.0, err, steady)
	return out


def pid_metrics(ec, draw: HoldoutDraw) -> dict:
	"""PID on the draw, in the metric-dict shape phased_ga's summaries print.

	mean_reward/mean_effort are None: this scorer returns stability and error, and
	inventing a reward here would be a made-up number in a comparison table.
	"""
	from ._accel import score_classical_baseline
	q0, w0, fields = _episode_fields(ec, draw)
	st, err, steady = score_classical_baseline(
		0, list(q0), list(w0), draw.steps, draw.stable_deg, **fields)
	return {"stable_rate": st, "mean_attitude_error_deg": err,
	        "mean_steady_error_deg": steady, "mean_reward": None,
	        "mean_effort": None, "label": "PID"}
