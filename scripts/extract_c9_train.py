"""Extract C9 Stage-A winner's TRAIN (DAGGER) vs EVAL (K-fold) stable/err.

The gen-line 82%/3.96° is the K-fold EVAL on the train seed (optimistic GA-selection
metric). This re-evals the saved winner at the same train seed and pulls both:
  - eval  : Metrics.acc / mean_attitude_error_deg (held-out from DAGGER's episodes)
  - train : metrics['dagger']['iter_stable_rate'] / iter_mean_err_deg (DAGGER training)
"""
import pickle, math
from pathlib import Path
from wnn.control.evaluator import ControllerEvaluator, fit_thresholds_from_pid_rollouts
from wnn.control.training import EpisodeConfig
from wnn.control.reward_gated import RewardGatedConfig
from wnn.seeds import resolve_seed_set

PKL = "logs/controller/curriculum/full_C9_20260604_131958/stageA_winner.pkl"
TRAIN_BASE = 3003   # C9 full run --base-seed

d = pickle.load(open(PKL, "rb"))
g, spec = d["best_genome"], d["spec"]
seed = resolve_seed_set(base=TRAIN_BASE, run_index=0).train

# Stage A conditions: 5° / body-rate 0.5 / yaw 0.3 / 250 steps.
ec = EpisodeConfig(dt=0.001, steps_per_episode=250,
                   max_initial_tilt_rad=math.radians(5.0),
                   max_initial_yaw_rad=math.radians(5.0),
                   max_initial_body_rate=0.5, max_initial_yaw_rate=0.3)
thr = fit_thresholds_from_pid_rollouts(spec, num_episodes=10, seed=seed)
rg = RewardGatedConfig(seed=seed, episode_config=ec)
rg.steps_per_episode = 250; rg.num_rounds = 3; rg.episodes_per_round = 6; rg.progress = False
ev = ControllerEvaluator(spec, num_eval_episodes=100, seed=seed, episode_config=ec,
                         thresholds=thr, rg_config=rg, max_train_workers=3, num_eval_folds=5)

sc, oc = g.to_connections(); fit, metrics = ev.train_and_evaluate(thr, sc, oc)
dag = metrics.get("dagger", {})
print("=" * 64)
print("C9 Stage-A winner (46 neurons) @5° — TRAIN vs EVAL")
print("=" * 64)
print(f"  EVAL  (K-fold, train seed):  stable={metrics['stable_rate']*100:.1f}%  "
      f"err={metrics['mean_attitude_error_deg']:.2f}°")
if dag:
    print(f"  TRAIN (DAGGER last iter):    stable={dag.get('iter_stable_rate',float('nan'))*100:.1f}%  "
          f"err={dag.get('iter_mean_err_deg',float('nan')):.2f}°")
else:
    print("  TRAIN: (no dagger sub-metric returned)")
print("=" * 64)
