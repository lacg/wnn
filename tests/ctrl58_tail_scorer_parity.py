"""CTRL-58 (ABI 59): the GA tail column end to end through the production evaluator.

On a real recipe's MEMORY genome (the adopted base, yfeM s1), scored on one of its VAL units:
  * tail OFF (default): metrics carry no episode_err_p99_deg, and every other metric is the
    same with the tail armed (CPU scorer: bit for bit) — the export is measurement only;
  * tail ARMED: a finite p99 on both scorers, GPU == CPU within the f32 tolerance (0.01°),
    and it is >= the mean err (an order statistic of the same per-episode errors);
  * CRN (all K pools): the column is the p99 over the pooled episodes of every pool.

Run: PYTHONPATH=src python tests/ctrl58_tail_scorer_parity.py   (needs ram_controller ABI >= 59)
"""
import os
import shlex
from pathlib import Path

from wnn.control import _accel as A
from wnn.control import phased_ga as P
from wnn.control.evaluator import ControllerEvaluator
from wnn.control.val_units import checked_val_unit_seeds
from wnn.seeds import resolve_seed_set

TAG = "C-S2-d_wposx2_a55_rf_c200_dc_biasin_ms01_P_both_L1_yfeM_full_s1"
# The run's .out and checkpoints live in the main tree's logs (a worktree may not carry them).
D = next(d for d in (Path(__file__).resolve().parents[1] / "logs/controller/ctrl48",
                     Path("/Users/lacg/wnn/logs/controller/ctrl48")) if (d / f"{TAG}.out").exists())


def _setup():
	line = [l for l in open(D / f"{TAG}.out", errors="replace") if l.startswith("[argv]")][0]
	a = P.build_arg_parser().parse_args(shlex.split(line[len("[argv]"):]))
	ec = P.episode_config_from_args(a); P._resolve_ctbr(a, ec); a._dhat_b = None
	a._conn_policy, a._conn_policy_min = P.conn_policy_from_name(getattr(a, "conn_policy", "spread"))
	_, spec, res = P._stage_entry_from_checkpoint(D / f"ckpt_{TAG}/stage4_memory.yaml.gz", top_k=1)[:3]
	s = resolve_seed_set(base=a.base_seed, run_index=0, train=a.train_seed, test=a.test_seed, val=a.val_seed)
	unit = checked_val_unit_seeds(a, s)[0]
	thr = P._test_thresholds(a, ec, spec, unit, s.train, use_score=True)
	def mk(gpu: bool, crn: bool = False, tail=None):
		os.environ["WNN_CONTROLLER_GPU_EVAL"] = "1" if gpu else "0"
		ev = ControllerEvaluator(spec, num_eval_episodes=a.test_episodes, seed=unit, episode_config=ec,
		                         thresholds=thr, rg_config=P._rg_config(a, ec, unit), max_train_workers=a.train_workers,
		                         num_eval_folds=a.num_eval_folds, score_crn=crn, max_eval_workers_gpu=gpu)
		ev.episode_tail_cap_deg = tail
		return ev
	return mk, res.final_population[0]


FIELDS = ("acc", "mean_attitude_error_deg", "mean_steady_error_deg", "mean_position_error_m",
          "mean_altitude_error_m", "mean_position_hold_m", "mean_dz_hold_m", "reward")


def main():
	assert A.ABI_VERSION >= 59 if hasattr(A, "ABI_VERSION") else True
	mk, g = _setup()
	off = mk(False).score_genomes([g])[0]
	on = mk(False, tail=8.0).score_genomes([g])[0]
	assert off.episode_err_p99_deg is None, "default must not carry the tail"
	for k in FIELDS:
		assert getattr(off, k) == getattr(on, k), (k, getattr(off, k), getattr(on, k))
	assert on.episode_err_p99_deg is not None and on.episode_err_p99_deg >= on.mean_attitude_error_deg
	gpu = mk(True, tail=8.0).score_genomes([g])[0]
	assert abs(gpu.episode_err_p99_deg - on.episode_err_p99_deg) <= 0.01, (gpu.episode_err_p99_deg, on.episode_err_p99_deg)
	print(f"[ctrl58-tail] single pool: p99 cpu {on.episode_err_p99_deg:.4f} gpu {gpu.episode_err_p99_deg:.4f} "
	      f"(err {on.mean_attitude_error_deg:.3f}); off == armed on {len(FIELDS)} fields")
	crn_off = mk(False, crn=True).score_genomes([g])[0]
	crn_on = mk(False, crn=True, tail=8.0).score_genomes([g])[0]
	for k in FIELDS:
		assert getattr(crn_off, k) == getattr(crn_on, k), (k, getattr(crn_off, k), getattr(crn_on, k))
	crn_gpu = mk(True, crn=True, tail=8.0).score_genomes([g])[0]
	assert abs(crn_gpu.episode_err_p99_deg - crn_on.episode_err_p99_deg) <= 0.01
	print(f"[ctrl58-tail] CRN: p99 cpu {crn_on.episode_err_p99_deg:.4f} gpu {crn_gpu.episode_err_p99_deg:.4f}")
	print("ALL PASS")


if __name__ == "__main__":
	main()
