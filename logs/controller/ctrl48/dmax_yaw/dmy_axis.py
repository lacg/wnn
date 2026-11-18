"""17/11 (Luiz): why did yaw-only Y s1's steady miss? Per-axis split (ABI 56 score_controllers_cpu(axis_out=))
of any stage genome of a run, optionally re-decoded with another yaw-bank gain (spec.delta_max_yaw override,
NOT retrained), on the run's VAL units (tuning) or --test-seeds (report).

  --genome HEADLINE | GRID#i | NEURONS#i | MEMORY#i
  --dmy asflown | none (yaw bank at --delta-max) | <float>
Rows: [unit, stable, err_deg, steady_deg, alt_m, pos_m, dz_hold_m, *14 AXIS_FIELDS (rad)]."""
import argparse, dataclasses, math, os, re, sys
os.environ["WNN_CONTROLLER_GPU_EVAL"] = "0"
sys.path.insert(0, "/Users/lacg/wnn/scripts")
import numpy as np
import ctrl_gap_student_trace as T


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
	ap.add_argument("--units", choices=["val", "test"], default="val")
	ap.add_argument("--genome", default="HEADLINE"); ap.add_argument("--dmy", default="asflown")
	o = ap.parse_args()
	import wnn.control._accel as A
	import ram_controller as RC
	from wnn.control.phased_ga import _test_thresholds, _rg_config
	from wnn.control.evaluator import ControllerEvaluator
	from wnn.seeds import resolve_seed_set
	from probe_episode_failures import _load_stage_genome
	assert RC.ABI_VERSION >= 57
	txt, a, ec = T.run_args(o.tag)
	seeds = resolve_seed_set(base=a.base_seed, run_index=0, train=a.train_seed, test=a.test_seed, val=a.val_seed)
	units = T.val_units(o.tag) if o.units == "val" else [int(u) for u in a.test_seeds]
	head = re.findall(r"\[stage-select\] HEADLINE stage=\S+ genome=(\S+)", txt)[-1]
	if o.genome == "WINNER":
		# The published HEADLINE genome from --save-winner (best_genome). Needed where the stage
		# checkpoint lost the genome's cells (17/11: GRID#2 of the top-3 GRID stream saved cells=None).
		from wnn.control.phased_ga import _ctl_load
		pay = _ctl_load(T.D + o.tag + "_winner.yaml.gz")
		spec, genome, g = pay["spec"], pay["best_genome"], f"WINNER={head}"
		assert genome.cells is not None
	else:
		g = head if o.genome == "HEADLINE" else o.genome
		lab, idx = g.split("#")
		rep = os.path.join(T.D + "ckpt_" + o.tag, "ctrl56_stage0_grid.yaml.gz")
		if lab == "GRID" and os.path.exists(rep):
			# CTRL-56 repaired copy (arch-only entries retrained on the train seed after a verified
			# MATCH of GRID#0's retrain) — the controllers stage-select scored in flight.
			from wnn.control.phased_ga import _stage_entry_from_checkpoint
			e = _stage_entry_from_checkpoint(rep, top_k=int(idx) + 1)
			spec, genome = e[1], list(getattr(e[2], "final_population", None) or [e[2].best_genome])[int(idx)]
			g += "(ctrl56-repaired)"
		else:
			_, spec, genome = _load_stage_genome(T.D + "ckpt_" + o.tag, T.STAGE[lab], int(idx))
		assert getattr(genome, "cells", None) is not None, f"{g}: the stage checkpoint holds no cells for this genome (use --genome WINNER)"
	if o.dmy != "asflown":
		spec = dataclasses.replace(spec, delta_max_yaw=None if o.dmy == "none" else float(o.dmy))
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
		ax = []
		row = real(ctrls, *aa, **kk, axis_out=ax)[0]
		assert abs(math.degrees(ax[7]) - m.mean_steady_error_deg) < 0.01 and abs(math.degrees(ax[3]) - m.mean_attitude_error_deg) < 0.01
		assert abs(math.degrees(row[5]) - m.mean_steady_error_deg) < 1e-9, "re-issued call is not the scored one"
		R.append([unit, m.stable_rate, m.mean_attitude_error_deg, m.mean_steady_error_deg, m.mean_altitude_error_m,
		          m.mean_position_error_m, m.mean_dz_hold_m, *ax])
	r = np.array(R, dtype=float); mm = r.mean(0); d = np.degrees
	print(f"[dmy-axis] {o.tag.split('_L1_')[1]} {o.units} {g} dmy={o.dmy}: err {mm[2]:.3f} steady {mm[3]:.3f} alt {mm[4]:.4f} pos {mm[5]:.4f}"
	      f" dz_hold {mm[6]:.4f} | steady r/p/y {d(mm[11]):.3f}/{d(mm[12]):.3f}/{d(mm[13]):.3f} err r/p/y {d(mm[7]):.3f}/{d(mm[8]):.3f}/{d(mm[9]):.3f}"
	      f" drift {d(mm[18]):.3f}", flush=True)
	np.savez(o.out, rows=r, genome=g, tag=o.tag, units=o.units, dmy=o.dmy)


if __name__ == "__main__":
	main()
