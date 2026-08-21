"""Measure the TEACHER's jerk/err/stable/steady so the fitness bounds are evidence, not guesses.

WHY (21/08/2026). The controller fitness weights jerk at 0.20 and mono at 0.10 —
0.30 of the objective — and both are normalised by the POOL's spread, so a
practically meaningless jerk gap can outvote a large stability gap. The fix is
fixed practical scales, which needs a reference value for "how much jerk is
acceptable". The reference is the TEACHER: a controller already accepted as good
on this airframe. Its jerk distribution defines the bound.

Until today `score_classical_baseline` never returned jerk, so that reference did
not exist. It does now (dagger_train.rs, 6th return value, same definition as the
WNN scorer: mean over steps of ||dpwm||).

NOT calibrated here: mono_violations. It counts THERMOMETER-PATTERN violations in
the WNN's output encoding (controller.rs:6363) — a property of the representation,
not of the flight. A PID emits continuous commands and has no thermometer bank, so
mono is UNDEFINED for a teacher and no rollout can bound it. That is an argument
for removing mono from the flight objective, not for inventing a number.

Read-only: rollouts only. No GA, no training, no memory, no flows, no DB.
"""

import statistics as st
import sys

import math

from wnn.control.training import EpisodeConfig, DisturbanceConfig
from wnn.control.airframe import Airframe as _Airframe
from wnn.control.classical_baseline import HoldoutDraw, TeacherFeed, pid_metrics

# The report seeds the A/B published on — same episodes the WNN was judged on.
REPORT_SEEDS = [99990101, 99990102, 99990103, 99990104, 99990105]
EPISODES = 100
# pid_metrics() calls score_classical_baseline(0, ...) — teacher_id is HARDCODED
# to PID, so this measures PID and nothing else. score_all() loops all five
# teacher ids; use that if the other teachers' jerk is ever needed. Listing
# lqr/mpcof here would print three identical blocks and read as agreement
# between teachers that was never measured.
TEACHERS = ["pid"]

# The A/B's own flight envelope, read off scripts/fitness_agg_ab_chain.sh so the
# numbers describe the plant the fitness was actually judged on:
#   --airframe cf21_brushless --disturbance L4C --steps 2000 --tilt 5.0
#   --translation --reward-lambda-alt 0 --num-eval-folds 5
AIRFRAME, DISTURBANCE, STEPS, TILT_DEG, FOLDS = "cf21_brushless", "L4C", 2000, 5.0, 5


def episode_config() -> EpisodeConfig:
	"""The A/B's EpisodeConfig. Built here rather than imported because
	phased_ga assembles it inline inside main() — duplicating the FIELDS is
	safer than duplicating an argparse namespace, and every value below is
	traceable to a flag in the chain script."""
	return EpisodeConfig(
		dt=0.001, steps_per_episode=STEPS,
		max_initial_tilt_rad=math.radians(TILT_DEG),
		max_initial_yaw_rad=math.radians(TILT_DEG),
		disturbance=DisturbanceConfig.preset(DISTURBANCE),
		airframe=_Airframe.preset(AIRFRAME),
		translation=True,          # --translation
		lambda_alt=0.0,            # --reward-lambda-alt 0
	)


def rows_for(teacher: str) -> list:
	"""One metric dict per report seed, teacher ESTIMATOR-FED (the rival role —
	the WNN flies a raw noisy IMU, so an oracle-fed teacher is an upper bound,
	not a comparator)."""
	out = []
	for seed in REPORT_SEEDS:
		ec = episode_config()
		draw = HoldoutDraw(seed=seed, episodes=EPISODES,
		                   steps=ec.steps_per_episode, eval_folds=FOLDS)
		out.append(pid_metrics(ec, draw, TeacherFeed(use_estimator=True)))
	return out


def pct(vals: list, q: float) -> float:
	s = sorted(vals)
	i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
	return s[i]


def report(teacher: str, rows: list) -> dict | None:
	jerk = [r["mean_pwm_jerk"] for r in rows if r.get("mean_pwm_jerk") is not None]
	err = [r["mean_attitude_error_deg"] for r in rows]
	stb = [r["stable_rate"] * 100.0 for r in rows]
	sdy = [r["mean_steady_error_deg"] for r in rows]
	sd = lambda a: st.stdev(a) if len(a) > 1 else 0.0
	print(f"\n=== teacher={teacher}  (estimator-fed, {len(rows)} report seeds x {EPISODES} episodes) ===")
	print(f"  stable  {st.mean(stb):6.1f}% ± {sd(stb):5.2f}")
	print(f"  err     {st.mean(err):6.2f}°  ± {sd(err):5.2f}")
	print(f"  steady  {st.mean(sdy):6.2f}°  ± {sd(sdy):5.2f}")
	if not jerk:
		print("  jerk    — NOT RETURNED (stale ram_controller wheel: rebuild it)")
		return None
	print(f"  jerk    {st.mean(jerk):8.5f} ± {sd(jerk):.5f}"
	      f"   min {min(jerk):.5f}  P95 {pct(jerk, 0.95):.5f}  max {max(jerk):.5f}")
	return {"teacher": teacher, "jerk_mean": st.mean(jerk), "jerk_p95": pct(jerk, 0.95),
	        "jerk_max": max(jerk), "err": st.mean(err), "stable": st.mean(stb)}


def main() -> int:
	print(f"airframe={AIRFRAME}  disturbance={DISTURBANCE}  steps={STEPS}  tilt={TILT_DEG}°  translation=on")
	print(f"report seeds {REPORT_SEEDS}  x {EPISODES} episodes each")
	got = []
	for t in TEACHERS:
		try:
			r = report(t, rows_for(t))
			if r:
				got.append(r)
		except Exception as e:      # one bad teacher must not lose the others
			print(f"\n=== teacher={t} === FAILED: {type(e).__name__}: {e}")
	if not got:
		print("\nNo teacher produced a jerk number — nothing to calibrate from.")
		return 1
	print("\n" + "=" * 72)
	print("PROPOSED BOUND — from the teachers, not from the GA population")
	print("=" * 72)
	p95 = max(g["jerk_p95"] for g in got)
	print(f"  PID jerk P95                          : {p95:.5f}")
	print(f"  J_max at 1.0x (match the teacher)     : {p95:.5f}")
	print(f"  J_max at 1.2x (20% headroom)          : {p95 * 1.2:.5f}")
	print("\n  WNN population observed 0.017-0.037 across the A/B — 10-23x the PID.")
	print("  That gap is STRUCTURAL, not a quality gap: with levels=16 and")
	print("  delta_max=0.1, ONE level step per motor is 2*0.1/15 = 0.01333, and")
	print("  ||dpwm|| for 1..4 motors stepping together is 0.0133/0.0189/0.0231/")
	print("  0.0267 — the observed range IS the quantization lattice. A teacher")
	print("  emitting CONTINUOUS commands can sit below one quantum; a")
	print("  delta-coded WNN cannot. So J_max must come from what the ACTUATOR")
	print("  tolerates, NOT from the teacher's achieved smoothness — a")
	print("  teacher-derived bound would reject 100% of the population.")
	print("  mono_violations: NOT calibratable from a teacher (encoding metric, see docstring).")
	return 0


if __name__ == "__main__":
	sys.exit(main())
