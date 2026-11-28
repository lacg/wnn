"""CTRL-84 driver: DAgger-train the MLP baseline for ONE (size, seed), select its hyperparameters on
the VAL units, and save the selected head for the TEST scorer. Prereg: docs/ctrl84_mlp_baseline_prereg.md.

  mlp_baseline.py --arm a|b --seed N [--hp all|i,j] [--dry-run] [--rounds R --episodes E --folds K]
                  [--val-units U] [--val-episodes V] [--device auto|mps|cpu] [--out-dir DIR]

Pairing: the MLP for seed N trains under the l2bnm sN run's OWN argv (parsed from its .out header):
the same EpisodeConfig (cf21, L4C weather, wear +-20 % per episode, lag 0.20 s, nominal-mass anchor,
specific-force IMU, rotor drag), the same trainer config (ControllerEvaluator.packed_train_config:
mpcof teacher, rounds x episodes, tilt curriculum 8->30 deg, gate off, checkpoint rule), the same CRN
training fold seeds (train_seed*100 + k, k < 5, accumulated) and the same feature layout (the
winner's spec). No GA. Per pre-declared HP config: train, then score on the run's 20 VAL units
(the stage-select units, same evaluator construction as TEST). Selection (lexicographic, VAL only):
fewest fails, then lowest mean err, then lowest steady. The TEST units are never touched here —
run_mlp.sh scores the selected head once through ctrl83/score_cell.py.

Writes (out-dir, default ctrl84/data): mlp<arm>_s<N>__hp<i>.npz per config, mlp<arm>_s<N>.npz = the
selected head, mlp<arm>_s<N>__val.json = the VAL table + selection + timings.
"""
import argparse, itertools, json, math, os, sys, time

os.environ.setdefault("WNN_CONTROLLER_GPU_EVAL", "0")  # CPU scorer (the GPU belongs to the IDS worker)
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "logs", "controller", "ctrl83"))
sys.path.insert(0, HERE)
import numpy as np
import mlp_ids as M

# Pre-declared HP grid (docs/ctrl84_mlp_baseline_prereg.md §HP grid). Index = config id.
GRID_LR = (1e-3, 3e-4)
GRID_EPOCHS = (4, 16)
GRID_WD = (0.0, 1e-4)
HP_GRID = tuple(dict(lr=lr, epochs=ep, weight_decay=wd) for lr, ep, wd in itertools.product(GRID_LR, GRID_EPOCHS, GRID_WD))
BATCH = 1024
ACT = "relu"


def log(msg):
	print(f"[ctrl84 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_pair(seed: int):
	"""(tag, args, ec0, spec, seeds, thresholds) of the paired l2bnm sN run."""
	import cells as C
	import ctrl_gap_student_trace as T
	from wnn.control import phased_ga as P
	from wnn.seeds import resolve_seed_set
	tag = C.wnn_tag(f"{M.BASE_ARM}_s{seed}")
	_, a, ec0 = T.run_args(tag)
	P._resolve_ctbr(a, ec0)
	seeds = resolve_seed_set(base=a.base_seed, run_index=0, train=a.train_seed, test=a.test_seed, val=a.val_seed)
	spec = P._ctl_load(f"/Users/lacg/wnn/logs/controller/ctrl48/{tag}_winner.yaml.gz")["spec"]
	return tag, a, ec0, spec, seeds


def thresholds_for(a, ec0, spec, seeds):
	from wnn.control.phased_ga import _test_thresholds
	# The WNN's published address function (train-seed fit); the head never reads it, the controller
	# constructor needs one — the same one the paired WNN is scored with.
	return _test_thresholds(a, ec0, spec, 99990101, seeds.train, use_score=True)


def train_evaluator(a, ec0, spec, seeds, thr):
	from wnn.control.evaluator import ControllerEvaluator
	from wnn.control.phased_ga import _rg_config
	return ControllerEvaluator(spec, num_eval_episodes=a.eval_episodes, seed=seeds.train, episode_config=ec0,
	                           thresholds=thr, rg_config=_rg_config(a, ec0, seeds.train), max_train_workers=1,
	                           num_eval_folds=a.num_eval_folds, score_crn=True, max_eval_workers_gpu=False)


def score_units(a, ec0, spec, thr, payload, units, episodes):
	"""Per unit [unit, stable, err, steady, pos, alt, pos_hold, fails] — the TEST evaluator construction
	(ctrl83/score_cell.py score_wnn) on these units."""
	from wnn.control.evaluator import ControllerEvaluator
	from wnn.control.phased_ga import _rg_config
	U = []
	for unit in units:
		ev = ControllerEvaluator(spec, num_eval_episodes=episodes, seed=unit, episode_config=ec0, thresholds=thr,
		                         rg_config=_rg_config(a, ec0, unit), max_train_workers=1, num_eval_folds=a.num_eval_folds,
		                         max_eval_workers_gpu=False)
		m = ev.score_mlp([payload])[0]
		st = float(m.stable_rate)
		U.append([unit, st, m.mean_attitude_error_deg, m.mean_steady_error_deg, m.mean_position_error_m,
		          m.mean_altitude_error_m, m.mean_position_hold_m, round((1 - st) * episodes)])
	return np.array(U, dtype=float)


def summary(U):
	m = U.mean(0)
	return dict(stable=100 * m[1], err=m[2], steady=m[3], pos=m[4], alt=m[5], fails=int(U[:, 7].sum()), units=len(U))


def select(rows):
	"""Pre-declared VAL rule: fewest fails, then lowest err, then lowest steady."""
	return min(rows, key=lambda r: (r["val"]["fails"], r["val"]["err"], r["val"]["steady"]))["hp"]


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--arm", choices=sorted(M.ARMS), required=True)
	ap.add_argument("--seed", type=int, choices=M.SEEDS, required=True)
	ap.add_argument("--hp", default="all", help="'all' or comma-separated HP_GRID indices")
	ap.add_argument("--dry-run", action="store_true", help="resolve the pair, print the plan, train/score nothing")
	ap.add_argument("--rounds", type=int, default=None, help="SMOKE ONLY: override the trainer's rounds")
	ap.add_argument("--episodes", type=int, default=None, help="SMOKE ONLY: override episodes per round")
	ap.add_argument("--folds", type=int, default=None, help="SMOKE ONLY: train on the first K fold seeds")
	ap.add_argument("--val-units", type=int, default=None, help="SMOKE ONLY: score the first U val units")
	ap.add_argument("--val-episodes", type=int, default=None, help="SMOKE ONLY: episodes per val unit")
	ap.add_argument("--device", default="auto")
	ap.add_argument("--out-dir", default=M.DATA)
	o = ap.parse_args()
	smoke = any(v is not None for v in (o.rounds, o.episodes, o.folds, o.val_units, o.val_episodes))
	hps = list(range(len(HP_GRID))) if o.hp == "all" else [int(i) for i in o.hp.split(",")]
	cid = f"mlp{o.arm}_s{o.seed}"
	import ctrl_gap_student_trace as T
	from wnn.control.mlp_payload import param_count
	tag, a, ec0, spec, seeds = load_pair(o.seed)
	val_units = T.val_units(tag)
	K = int(a.num_eval_folds)
	fold_seeds = [seeds.train * 100 + k for k in range(K)][: (o.folds or K)]
	units = val_units[: (o.val_units or len(val_units))]
	v_eps = int(o.val_episodes or a.test_episodes)
	hidden = M.ARMS[o.arm]
	nf = spec.num_features()
	log(f"{cid}: paired WNN {tag} | features {nf} -> {hidden} -> {spec.num_motors} ({ACT}) = "
	    f"{param_count(nf, hidden, spec.num_motors)} params | train folds {fold_seeds} | "
	    f"VAL {len(units)} units {units[0]}..{units[-1]} x {v_eps} eps | HP {hps} of {len(HP_GRID)}"
	    + (" | SMOKE (shrunk; not a cohort result)" if smoke else ""))
	if o.dry_run:
		for i in hps:
			log(f"  hp{i}: {HP_GRID[i]} batch {BATCH}")
		log("dry run: nothing trained or scored")
		return
	from wnn.control.mlp_dagger import MlpDaggerTrainer
	from wnn.control.mlp_fitter import MlpFitConfig
	os.makedirs(o.out_dir, exist_ok=True)
	thr = thresholds_for(a, ec0, spec, seeds)
	ev = train_evaluator(a, ec0, spec, seeds, thr)
	rows = []
	for i in hps:
		hp = HP_GRID[i]
		t0 = time.time()
		fc = MlpFitConfig(hidden=hidden, activation=ACT, lr=hp["lr"], epochs=hp["epochs"], weight_decay=hp["weight_decay"],
		                  batch_size=BATCH, seed=o.seed * 1000 + i, device=o.device)
		tr = MlpDaggerTrainer(ev, fc, num_rounds=o.rounds, episodes_per_round=o.episodes)
		payload, stats, fitter = tr.train(fold_seeds, log=log)
		t_train = time.time() - t0
		payload.meta.update(dict(cid=cid, hp=i, paired=tag, smoke=smoke, cost=payload.cost()))
		payload.save(os.path.join(o.out_dir, f"{cid}__hp{i}.npz"))
		t1 = time.time()
		U = score_units(a, ec0, spec, thr, payload, units, v_eps)
		val = summary(U)
		rows.append(dict(hp=i, cfg=hp, val=val, train_s=round(t_train, 1), val_s=round(time.time() - t1, 1),
		                 records=fitter.n_records, fit_log=fitter.fit_log[-3:], folds=stats))
		log(f"  hp{i} {hp}: VAL stable {val['stable']:.2f} err {val['err']:.3f} steady {val['steady']:.3f} "
		    f"pos {val['pos']:.4f} fails {val['fails']}/{len(units) * v_eps} | train {t_train:.0f}s val {rows[-1]['val_s']:.0f}s")
	best = select(rows)
	from wnn.control.mlp_payload import MlpPayload
	sel = MlpPayload.load(os.path.join(o.out_dir, f"{cid}__hp{best}.npz"))
	sel.meta["selected_on"] = "VAL: fewest fails, then err, then steady"
	sel.save(os.path.join(o.out_dir, f"{cid}.npz"))
	json.dump(dict(cid=cid, paired=tag, val_units=units, val_episodes=v_eps, smoke=smoke, selected=best, rows=rows,
	               grid=HP_GRID), open(os.path.join(o.out_dir, f"{cid}__val.json"), "w"), indent=1, default=float)
	log(f"{cid}: selected hp{best} {HP_GRID[best]} on VAL -> {os.path.join(o.out_dir, cid + '.npz')}")


if __name__ == "__main__":
	main()
