"""ABI 57 offline pre-check (Luiz 17/11/2026): does a YAW-ONLY delta_max keep the delta-gain block's
attitude gain without its altitude cost? EXISTING GENOMES, NOT RETRAINED — the decode gain is changed
at score time only (the spec's delta_max / delta_max_yaw), on the run's VAL units (tuning partition),
same episodes per variant (one evaluator per unit, CPU scorer).

Variants (deg / m), per genome type:
  M (trained --delta-max 0.6): flown | all 0.5 (delta_max=0.5) | yaw 0.5 (delta_max_yaw=0.5)
  D (trained --delta-max 0.5): flown | yaw-only 0.5 (delta_max=0.6, delta_max_yaw=0.5)
Rows: [unit, stable, err, steady, alt, pos, pos_hold, dz_hold] per variant."""
import argparse, dataclasses, os, re, sys
os.environ["WNN_CONTROLLER_GPU_EVAL"] = "0"
sys.path.insert(0, "/Users/lacg/wnn/scripts")
import numpy as np
import ctrl_gap_student_trace as T

VARIANTS = {"M": {"flown": {}, "all05": {"delta_max": 0.5}, "yaw05": {"delta_max_yaw": 0.5}},
            "D": {"flown": {}, "yaw05_others06": {"delta_max": 0.6, "delta_max_yaw": 0.5}}}


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--tag", required=True); ap.add_argument("--arm", choices=["M", "D"], required=True)
	ap.add_argument("--out", required=True)
	o = ap.parse_args()
	import ram_controller as RC
	from wnn.control.phased_ga import _test_thresholds, _rg_config
	from wnn.control.evaluator import ControllerEvaluator
	from wnn.seeds import resolve_seed_set
	from probe_episode_failures import _load_stage_genome
	assert RC.ABI_VERSION >= 57, RC.ABI_VERSION
	txt, a, ec = T.run_args(o.tag)
	seeds = resolve_seed_set(base=a.base_seed, run_index=0, train=a.train_seed, test=a.test_seed, val=a.val_seed)
	units = T.val_units(o.tag)
	g = re.findall(r"\[stage-select\] HEADLINE stage=\S+ genome=(\S+)", txt)[-1]
	lab, idx = g.split("#")
	_, spec, genome = _load_stage_genome(T.D + "ckpt_" + o.tag, T.STAGE[lab], int(idx))
	assert abs(spec.delta_max - (0.6 if o.arm == "M" else 0.5)) < 1e-9 and spec.delta_max_yaw is None, (spec.delta_max, spec.delta_max_yaw)
	out = {}
	for vname, change in VARIANTS[o.arm].items():
		vspec = dataclasses.replace(spec, **change)
		vg = genome  # the arch genome carries no spec: the evaluator's base spec (spec_from_arch) does
		rows = []
		for unit in units:
			thr = _test_thresholds(a, ec, vspec, unit, seeds.train, use_score=True)
			ev = ControllerEvaluator(vspec, num_eval_episodes=a.test_episodes, seed=unit, episode_config=ec, thresholds=thr,
			                         rg_config=_rg_config(a, ec, unit), max_train_workers=1, num_eval_folds=a.num_eval_folds,
			                         max_eval_workers_gpu=False)
			m = ev.score_genomes([vg])[0]
			rows.append([unit, m.stable_rate, m.mean_attitude_error_deg, m.mean_steady_error_deg, m.mean_altitude_error_m,
			             m.mean_position_error_m, m.mean_position_hold_m, m.mean_dz_hold_m])
		r = np.array(rows, dtype=float); out[vname] = r
		mm = r[:, 1:].mean(0)
		print(f"[dmy] {o.tag} {g} {vname:<15} stable {100*mm[0]:.2f}% err {mm[1]:.3f} steady {mm[2]:.3f} alt {mm[3]:.4f} pos {mm[4]:.4f}"
		      f" pos_hold {mm[5]:.4f} dz_hold {mm[6]:.4f}", flush=True)
	np.savez(o.out, genome=g, arm=o.arm, **out)


if __name__ == "__main__":
	main()
