"""Re-score one finished ctrl48 run's HEADLINE genome on the fixed (ABI 47) scorer, as flown
(the run's own [argv]), over its 35 test seeds. argv: tag. Writes <tag>.json next to this file."""
import sys, shlex, json, re
sys.path.insert(0, "/tmp/ctrl_fix_site")
sys.path.insert(1, "/Users/lacg/wnn/.claude/worktrees/ctrl-scorer-fix/src")
from pathlib import Path
from wnn.control import phased_ga as P
from wnn.control.evaluator import ControllerEvaluator
from wnn.seeds import resolve_seed_set
import ram_controller
assert ram_controller.ABI_VERSION == 47, ram_controller.ABI_VERSION
D = "/Users/lacg/wnn/logs/controller/ctrl48/"
OUT = Path(__file__).parent
STAGE = {"GRID": "stage0_grid", "NEURONS": "stage1_neurons", "MEMORY": "stage4_memory"}
tag = sys.argv[1]
text = open(D + tag + ".out", errors="replace").read()
argv = shlex.split([l for l in text.splitlines() if l.startswith("[argv]")][0][len("[argv]"):])
head = re.findall(r"stage-select\] HEADLINE stage.*?genome=(\S+)", text)[-1]
stage, idx = head.split("#")[0], int(head.split("#")[1])
a = P.build_arg_parser().parse_args(argv)
ec = P.episode_config_from_args(a); P._resolve_ctbr(a, ec); a._dhat_b = None
a._conn_policy, a._conn_policy_min = P.conn_policy_from_name(getattr(a, "conn_policy", "spread"))
label, spec, res = P._stage_entry_from_checkpoint(Path(D + f"ckpt_{tag}/{STAGE[stage]}.yaml.gz"), top_k=idx + 1)[:3]
g = res.final_population[idx]
s = resolve_seed_set(base=a.base_seed, run_index=0, train=a.train_seed, test=a.test_seed, val=a.val_seed)
KEYS = ("acc", "mean_attitude_error_deg", "mean_steady_error_deg", "mean_position_error_m")
rows = []
if getattr(g, "cells", None) is None:
	# Arch-only headline (raw GRID): train on the TRAIN seed exactly as _holdout_report does
	# (K-fold accumulate, train-seed thresholds, write-back), then score-only on every test seed.
	rs0 = int(a.test_seeds[0])
	thr0 = P._test_thresholds(a, ec, spec, rs0, s.train, use_score=True)
	tev = ControllerEvaluator(spec, num_eval_episodes=a.test_episodes, seed=s.train, episode_config=ec,
	                          thresholds=thr0, rg_config=P._rg_config(a, ec, s.train), max_train_workers=a.train_workers,
	                          num_eval_folds=a.num_eval_folds)
	tev._evaluate_core([g], write_back=True)
	print(f"[arch-only] {head}: trained on train seed {s.train}", flush=True)
for rs in [int(x) for x in a.test_seeds]:
	thr = P._test_thresholds(a, ec, spec, rs, s.train, use_score=True)
	ev = ControllerEvaluator(spec, num_eval_episodes=a.test_episodes, seed=rs, episode_config=ec,
	                         thresholds=thr, rg_config=P._rg_config(a, ec, rs), max_train_workers=a.train_workers,
	                         num_eval_folds=a.num_eval_folds)
	m = ev.score_genomes([g])[0]
	rows.append([getattr(m, k) for k in KEYS])
n = len(rows); eps = a.test_episodes
mean = lambda i: sum(r[i] for r in rows) / n
fails = round(sum((1 - r[0]) * eps for r in rows))
res = {"tag": tag, "genome": head, "seeds": n, "episodes": eps, "stable": mean(0) * 100, "err": mean(1),
       "steady": mean(2), "pos": mean(3), "fails": fails, "per_seed": rows}
json.dump(res, open(OUT / f"{tag}.json", "w"))
print(f"{tag} {head} {res['stable']:.1f} / {res['err']:.2f} / {res['steady']:.2f} / {res['pos']:.3f} ({fails}/{n*eps})", flush=True)
