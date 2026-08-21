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
class TeacherFeed:
	"""What a COMPARISON teacher is allowed to read.

	The rule (Luiz, 13/08/2026): TRAIN with oracle teachers, COMPARE with
	estimator-fed ones. A rival that reads the true quaternion is not a rival —
	it is a disclosed upper bound, because the WNN never gets that quaternion:
	it reads the raw noisy IMU. Feeding the rival a Mahony estimate of the SAME
	noisy IMU is what makes the row apples-to-apples.

	`use_estimator=False` is the byte-identical legacy path, so every number
	banked before 13/08/2026 reproduces unchanged.

	`label_for` stamps the convention INTO the printed row. That is the whole
	point of this type: the numbers from the two conventions are not
	comparable, and an unlabelled `vs PID 90.4%` gives a future reader no way
	to tell which one they are holding.
	"""
	use_estimator: bool = False
	est_kp: float = 2.0
	est_ki: float = 0.1

	def fields(self) -> dict:
		"""The three kwargs `score_classical_baseline` takes for the feed."""
		return dict(use_estimator=bool(self.use_estimator),
		            est_kp=float(self.est_kp), est_ki=float(self.est_ki))

	def label_for(self, name: str) -> str:
		"""'PID' → 'PID[est]' or 'PID[oracle]' — never a bare name."""
		return f"{name}[est]" if self.use_estimator else f"{name}[oracle]"


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


def _stage1_fields(ec, draw: HoldoutDraw) -> dict:
	"""SCOPE C STAGE 1 (14/08/2026): the rival flies the SAME randomized plant
	the WNN flies — the vertical draws come from the SAME pool seed and sampler
	the WNN scorer uses, so the episode sets are identical, mass jitter and all.
	{} when translation is off, the bit-identical attitude-only path."""
	if float(getattr(ec, "max_initial_xy_offset_m", 0.0)) > 0.0 \
			and not getattr(ec, "translation", False):
		raise ValueError(
			"classical_baseline: the horizontal axis is armed (max_initial_xy_offset_m > 0) "
			"but translation is off, so the rivals would fly attitude-only while the WNN "
			"flies a position task. Refusing to score two different tasks against each other.")
	if not getattr(ec, "translation", False):
		return {}
	from .training import sample_vertical_ics_flat
	pool = draw.pool_seed()
	z0, vz0, coll, mass = sample_vertical_ics_flat(pool, draw.episodes, ec)
	af_mass = float(ec.airframe.mass) if ec.airframe is not None else 1.0
	fields = dict(
		translation=True,
		s1_target_altitude=float(getattr(ec, "target_altitude", 0.0)),
		s1_init_z=[float(v) for v in z0],
		s1_init_vz=[float(v) for v in vz0],
		s1_mass=[af_mass * float(m) for m in mass],
		s1_collective_frac=[float(c) for c in coll],
	)
	# SCOPE C STAGE 2 (14/08/2026): the HORIZONTAL draws, from the SAME pool seed
	# and sampler the WNN scorer uses, so the rival flies the identical episode
	# set — displaced starts included. Without them the WNN would fly back from
	# +/-xy while every rival started on target, and the comparison would be
	# between two different tasks. Gated on the axis so a stage-1 call stays
	# byte-identical.
	if float(getattr(ec, "max_initial_xy_offset_m", 0.0)) > 0.0:
		from .training import sample_horizontal_ics_flat
		x0, y0 = sample_horizontal_ics_flat(pool, draw.episodes, ec)
		fields.update(
			s2_init_x=[float(v) for v in x0],
			s2_init_y=[float(v) for v in y0],
			pos_omega_n=float(getattr(ec, "pos_omega_n", 1.0)),
			pos_zeta=float(getattr(ec, "pos_zeta", 1.0)),
			pos_max_tilt_rad=float(getattr(ec, "pos_max_tilt_rad", 0.5236)),
		)
	return fields


def score_all(ec, draw: HoldoutDraw, feed: TeacherFeed = TeacherFeed()) -> dict:
	"""{labelled_name: (stable%, err°, steady°)} for all 5 classical controllers.

	Keys carry the feed convention (`PID[est]` / `PID[oracle]`) so a results
	file cannot lose track of which rivals it is holding.
	"""
	from ._accel import score_classical_baseline
	q0, w0, fields = _episode_fields(ec, draw)
	# The plant the baselines fly MUST be the plant the WNN flies, or the
	# comparison is between two different aircraft.
	fields = {**fields, **ec.airframe_kwargs(), **feed.fields(), **_stage1_fields(ec, draw)}
	out = {}
	for tid, name in _NAMES.items():
		# Slice, don't destructure: the scorer gained a 6th value (jerk) on
		# 21/08/2026 and this tuple is documented as a 5-field row. Unpacking
		# by arity would break the moment the wheel grows another metric.
		_r = score_classical_baseline(
			tid, list(q0), list(w0), draw.steps, draw.stable_deg, **fields)
		st, err, steady, alt_m, pos3d_m = _r[:5]
		out[feed.label_for(name)] = (st * 100.0, err, steady, alt_m, pos3d_m)
	return out


def pid_metrics(ec, draw: HoldoutDraw, feed: TeacherFeed = TeacherFeed()) -> dict:
	"""PID on the draw, in the metric-dict shape phased_ga's summaries print.

	mean_reward/mean_effort are None: this scorer returns stability and error, and
	inventing a reward here would be a made-up number in a comparison table.
	"""
	from ._accel import score_classical_baseline
	q0, w0, fields = _episode_fields(ec, draw)
	s1 = _stage1_fields(ec, draw)
	fields = {**fields, **ec.airframe_kwargs(), **feed.fields(), **s1}
	# 6th value (21/08/2026): mean motor-command jerk, same definition as the
	# WNN scorer's, so teacher and WNN jerk are directly comparable. Tolerate a
	# 5-tuple wheel so this Python can land before the wheel does.
	_r = score_classical_baseline(
		0, list(q0), list(w0), draw.steps, draw.stable_deg, **fields)
	st, err, steady, alt_m, pos3d_m = _r[:5]
	jerk = _r[5] if len(_r) > 5 else None
	return {"stable_rate": st, "mean_attitude_error_deg": err,
	        "mean_steady_error_deg": steady, "mean_reward": None,
	        "mean_effort": None,
	        # None (not 0.0) on an older wheel: a zero here would read as a
	        # perfectly smooth teacher and poison the very calibration this
	        # number exists to serve.
	        "mean_pwm_jerk": (float(jerk) if jerk is not None else None),
	        # metres only when the rival actually flew the translating plant —
	        # 0.0-when-off would read as a perfect hold.
	        #
	        # 14/08/2026 RENAME. This key used to be "mean_position_error_m"
	        # while holding the ALTITUDE error alone. Harmless while only the
	        # vertical channel existed; the moment stage 2 flies, that name reads
	        # as full 3-D position error and would go into a table as one. The
	        # altitude number keeps its own key and the Euclidean 3-D number —
	        # the Molchanov-comparable one — gets the position name it earns.
	        "mean_altitude_error_m": (float(alt_m) if s1 else None),
	        "mean_position_error_m": (float(pos3d_m) if s1 else None),
	        "label": feed.label_for("PID")}
