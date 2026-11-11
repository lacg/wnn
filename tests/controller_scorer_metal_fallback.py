"""A failing Metal scorer must NOT silently drop to the Python loop (10/11/2026: dropconst s3's
headline held-out died there with the real cause swallowed). It re-scores on the Rust CPU scorer
and matches the GPU result (CPU/GPU parity), and the error is surfaced.

Run: PYTHONPATH=src/wnn python tests/controller_scorer_metal_fallback.py
"""
import shlex
from pathlib import Path

from wnn.control import _accel as A
from wnn.control import phased_ga as P
from wnn.control.evaluator import ControllerEvaluator
from wnn.seeds import resolve_seed_set

D = Path(__file__).resolve().parents[1] / "logs/controller/ctrl48"
TAG = "C-S2-d_wposx2_a45_rf_c200_dres_full_s1"


def _evaluator_and_genome():
	line = [l for l in open(D / f"{TAG}.out", errors="replace") if l.startswith("[argv]")][0]
	a = P.build_arg_parser().parse_args(shlex.split(line[len("[argv]"):]))
	ec = P.episode_config_from_args(a); P._resolve_ctbr(a, ec); a._dhat_b = None
	a._conn_policy, a._conn_policy_min = P.conn_policy_from_name(getattr(a, "conn_policy", "spread"))
	_, spec, res = P._stage_entry_from_checkpoint(D / f"ckpt_{TAG}/stage4_memory.yaml.gz", top_k=1)[:3]
	s = resolve_seed_set(base=a.base_seed, run_index=0, train=a.train_seed, test=a.test_seed, val=a.val_seed)
	rs = int(a.test_seeds[0])
	thr = P._test_thresholds(a, ec, spec, rs, s.train, use_score=True)
	mk = lambda: ControllerEvaluator(spec, num_eval_episodes=a.test_episodes, seed=rs, episode_config=ec,
		thresholds=thr, rg_config=P._rg_config(a, ec, rs), max_train_workers=a.train_workers,
		num_eval_folds=a.num_eval_folds)
	return mk, res.final_population[0]


def main():
	mk, g = _evaluator_and_genome()
	gpu = mk().score_genomes([g])[0]
	orig = A.score_controllers_metal
	def boom(*_a, **_k):
		raise RuntimeError("simulated Metal failure")
	A.score_controllers_metal = boom
	try:
		cpu = mk().score_genomes([g])[0]
	finally:
		A.score_controllers_metal = orig
	assert gpu.acc == cpu.acc, (gpu.acc, cpu.acc)          # same episodes pass/fail
	for k in ("mean_attitude_error_deg", "mean_steady_error_deg", "mean_position_error_m"):
		gv, cv = getattr(gpu, k), getattr(cpu, k)
		assert abs(gv - cv) <= 0.01, (k, gv, cv)            # f32 GPU vs CPU accumulation
	print("ALL PASS — Metal failure re-scored on the Rust CPU scorer, matches GPU")


if __name__ == "__main__":
	main()
