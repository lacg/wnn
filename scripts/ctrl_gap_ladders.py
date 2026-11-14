"""docs/ctrl_gap_after_block9.md §B.1: print the thermometer cut-points of the inputs a
student would need to rebuild the tilt reference (e_x, e_y, v_x, v_y), its constant
target features and roll_err, for the block-8 "both" headline genomes. Read-only."""
import sys, os, re, math
os.environ["WNN_CONTROLLER_GPU_EVAL"] = "0"
sys.path.insert(0, "/Users/lacg/wnn/scripts")
import numpy as np
from ctrl_gap_student_trace import run_args, val_units, D, STAGE
from wnn.control.phased_ga import _test_thresholds
from wnn.control.feature_layout import feature_names
from wnn.seeds import resolve_seed_set
from probe_episode_failures import _load_stage_genome
T = "C-S2-d_wposx2_a50_rf_c200_dc_biasin_ms01_P_both_full"
unit = val_units(T + "_s1")[0]
for s in (1, 2, 3):
	tag = f"{T}_s{s}"
	txt, a, ec = run_args(tag)
	seeds = resolve_seed_set(base=a.base_seed, run_index=0, train=a.train_seed, test=a.test_seed, val=a.val_seed)
	g = re.findall(r"\[stage-select\] HEADLINE stage=\S+ genome=(\S+)", txt)[-1]; lab, idx = g.split("#")
	_, spec, genome = _load_stage_genome(D + "ckpt_" + tag, STAGE[lab], int(idx))
	thr = np.asarray(_test_thresholds(a, ec, spec, unit, seeds.train, use_score=True), dtype=np.float64)
	names = feature_names(spec)
	per = len(thr) // len(names)
	thr = thr.reshape(len(names), per)
	if s == 1:
		print("features:", names, "cuts/feature:", per)
	np.savez(f"/tmp/ctrl_gap/ladders_s{s}.npz", names=np.array(names), thr=thr)
	for i, n in enumerate(names):
		t = np.sort(thr[i])
		if n in ("e_x", "e_y", "v_x", "v_y", "tgt_r", "tgt_p", "roll_err", "pitch_err", "yaw_err", "yaw_err_i", "abp_r", "abp_p"):
			print(f"s{s} {n:10s} cuts: {np.round(t, 4).tolist()}")
