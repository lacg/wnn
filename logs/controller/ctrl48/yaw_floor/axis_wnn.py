"""ABI 56: per-axis split of a WNN run's HEADLINE genome THROUGH THE PRODUCTION CPU SCORER
(score_controllers_cpu(axis_out=)) — every flag flies natively, incl. --alt-integral pole3.

Per unit: ControllerEvaluator.score_genomes scores the genome (CPU), its exact
score_controllers_cpu call is captured and re-issued with axis_out=[]; the re-issued row
must equal the captured one BIT FOR BIT (the split is measurement only) and |v| must
reproduce err / steady. Rows: [unit, stable, err_deg, steady_deg, *14 AXIS_FIELDS (rad)].
--units val (default): stage-select VAL units (tuning). --units test: the run's --test-seeds
(report partition; descriptive)."""
import argparse, math, os, re, sys
os.environ["WNN_CONTROLLER_GPU_EVAL"] = "0"
sys.path.insert(0, "/Users/lacg/wnn/scripts")
import numpy as np
import ctrl_gap_student_trace as T


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
	ap.add_argument("--units", choices=["val", "test"], default="val")
	o = ap.parse_args()
	import wnn.control._accel as A
	import ram_controller as RC
	from wnn.control.phased_ga import _test_thresholds, _rg_config
	from wnn.control.evaluator import ControllerEvaluator
	from wnn.seeds import resolve_seed_set
	from probe_episode_failures import _load_stage_genome
	assert RC.ABI_VERSION >= 56, RC.ABI_VERSION
	txt, a, ec = T.run_args(o.tag)
	seeds = resolve_seed_set(base=a.base_seed, run_index=0, train=a.train_seed, test=a.test_seed, val=a.val_seed)
	units = T.val_units(o.tag) if o.units == "val" else [int(u) for u in a.test_seeds]
	g = re.findall(r"\[stage-select\] HEADLINE stage=\S+ genome=(\S+)", txt)[-1]
	lab, idx = g.split("#")
	_, spec, genome = _load_stage_genome(T.D + "ckpt_" + o.tag, T.STAGE[lab], int(idx))
	cap = {}; real = A.score_controllers_cpu
	def wrap(controllers, *aa, **kk):
		cap.setdefault("calls", []).append((controllers, aa, kk)); return real(controllers, *aa, **kk)
	A.score_controllers_cpu = wrap
	R = []
	for unit in units:
		thr = _test_thresholds(a, ec, spec, unit, seeds.train, use_score=True)
		ev = ControllerEvaluator(spec, num_eval_episodes=a.test_episodes, seed=unit, episode_config=ec, thresholds=thr,
		                         rg_config=_rg_config(a, ec, unit), max_train_workers=1, num_eval_folds=a.num_eval_folds,
		                         max_eval_workers_gpu=False)
		cap["calls"] = []
		m = ev.score_genomes([genome])[0]
		ctrls, aa, kk = cap["calls"][0]
		row0 = real(ctrls, *aa, **kk)[0]
		ax = []
		row1 = real(ctrls, *aa, **kk, axis_out=ax)[0]
		assert [float(x).hex() for x in row0] == [float(x).hex() for x in row1], f"{unit}: axis_out moved the row"
		assert len(ax) == 14
		assert abs(math.degrees(ax[3]) - m.mean_attitude_error_deg) < 0.01, (unit, math.degrees(ax[3]), m.mean_attitude_error_deg)
		assert abs(math.degrees(ax[7]) - m.mean_steady_error_deg) < 0.01, (unit, math.degrees(ax[7]), m.mean_steady_error_deg)
		R.append([unit, m.stable_rate, m.mean_attitude_error_deg, m.mean_steady_error_deg, *ax])
		print(f"[axis-wnn] {o.tag} {unit}: err {m.mean_attitude_error_deg:.3f} steady {m.mean_steady_error_deg:.3f} | "
		      f"err r/p/y {math.degrees(ax[0]):.3f}/{math.degrees(ax[1]):.3f}/{math.degrees(ax[2]):.3f} "
		      f"steady r/p/y {math.degrees(ax[4]):.3f}/{math.degrees(ax[5]):.3f}/{math.degrees(ax[6]):.3f}", flush=True)
	np.savez(o.out, rows=np.array(R), genome=g, partition=o.units, tag=o.tag,
	         alt_integral=str(getattr(a, "alt_integral", "off")))
	print(f"[axis-wnn] done -> {o.out}", flush=True)


if __name__ == "__main__":
	main()
