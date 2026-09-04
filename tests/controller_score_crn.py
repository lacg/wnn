"""CRN fitness (03/09/2026): every genome scored on ALL K pools every generation,
combined by mean; no per-generation pool rotation; shared training seeds.

Pure-Python proof against a stub evaluator — no accelerator, no rollouts. Proves:
  1. combine_pool_scores averages reward + every numeric field, None-safe.
  2. Under score_crn every fold pool is visited once per _score_fitness call and
     _active_score_seed is restored to pool 0 afterwards.
  3. _advance_fold does NOT rotate under CRN (and still does without it).
  4. _train_base_seeds is shared under CRN and positional without it.
"""
import sys
sys.path.insert(0, "/Users/lacg/wnn/src/wnn")

from wnn.control.evaluator import ControllerEvaluator, combine_pool_scores, fold_pool_seed


def _stub(score_crn: bool, K: int = 5, seed: int = 7) -> ControllerEvaluator:
	ev = object.__new__(ControllerEvaluator)
	ev.seed = seed
	ev.num_eval = 4
	ev.num_eval_folds = K
	ev._fold_seeds = [fold_pool_seed(seed, k) for k in range(K)]
	ev._fold_counter = 0
	ev._active_score_seed = seed
	ev.score_crn = score_crn
	ev._crn_logged = True
	return ev


def test_combine_mean_none_safe():
	per_pool = [
		[(1.0, {"stable_rate": 1.0, "mean_attitude_error_deg": 2.0, "mean_pwm_jerk": None})],
		[(3.0, {"stable_rate": 0.5, "mean_attitude_error_deg": 4.0, "mean_pwm_jerk": 0.2})],
	]
	(reward, m), = combine_pool_scores(per_pool)
	assert reward == 2.0
	assert m["stable_rate"] == 0.75 and m["mean_attitude_error_deg"] == 3.0
	assert m["mean_pwm_jerk"] == 0.2, "None in one pool must average the rest"
	assert combine_pool_scores([]) == []


def test_crn_visits_every_pool_and_averages():
	ev = _stub(score_crn=True)
	seen = []

	def fake_score(controllers, shape_keys):
		seen.append(ev._active_score_seed)
		# reward = pool index so the mean is checkable; two genomes
		k = ev._fold_seeds.index(ev._active_score_seed)
		return [(float(k), {"stable_rate": float(k) / 10}) for _ in controllers]

	ev._score_grouped = fake_score
	out = ev._score_fitness(["c0", "c1"], ["s", "s"])
	assert seen == ev._fold_seeds, "each pool exactly once, in order"
	assert ev._active_score_seed == ev._fold_seeds[0], "restored to pool 0"
	assert [r for r, _ in out] == [2.0, 2.0]
	assert out[0][1]["stable_rate"] == 0.2


def test_legacy_scores_one_pool_only():
	ev = _stub(score_crn=False)
	seen = []
	ev._score_grouped = lambda c, s: (seen.append(ev._active_score_seed), [(0.0, {})] * len(c))[1]
	ev._advance_fold()
	ev._score_fitness(["c0"], ["s"])
	assert seen == [ev._fold_seeds[0]]


def test_advance_fold_rotation():
	ev = _stub(score_crn=False)
	assert [ev._advance_fold() for _ in range(6)] == [0, 1, 2, 3, 4, 0]
	ev = _stub(score_crn=True)
	assert [ev._advance_fold() for _ in range(6)] == [0] * 6
	assert ev._active_score_seed == ev._fold_seeds[0]


def test_train_base_seeds():
	ev = _stub(score_crn=False, seed=7)
	assert ev._train_base_seeds(3, seed_offset=10) == [710, 715, 720]
	ev = _stub(score_crn=True, seed=7)
	assert ev._train_base_seeds(3, seed_offset=10) == [700, 700, 700]


if __name__ == "__main__":
	for name, fn in list(globals().items()):
		if name.startswith("test_"):
			fn()
			print(f"  ok  {name}")
	print("controller_score_crn: ALL PASS")
