"""CTRL-82: which features each output bank (coll, p, q, r) reads, per s1 winner (connection-bit share %, genome only)."""
import sys
sys.path.insert(0, "/Users/lacg/wnn/scripts")
import numpy as np
import ctrl_gap_student_trace as T
import wnn.control.phased_ga as PG
from wnn.control.feature_layout import feature_names
TAGS = {"G+T": "a62_rf_c200_dc_biasin_ms01_P_both_L1_yld_n0g3gz_full_s1", "lk90": "a63_rf_c200_dc_biasin_ms01_P_both_L1_yld_gt_lk90_full_s1",
        "l2b": "a63_rf_c200_dc_biasin_ms01_P_both_L1_yld_gt_l2b_full_s1"}
YAW = ("gyro_z", "yaw_err", "yaw_err_i"); RP = ("gyro_x", "gyro_y", "roll_err", "pitch_err", "roll_i", "pitch_i", "abp_r", "abp_p")
for lab, tg in TAGS.items():
	pay = PG._ctl_load(T.D + "C-S2-d_wposx2_" + tg + "_winner.yaml.gz"); spec, g = pay["spec"], pay["best_genome"]
	names = list(feature_names(spec)); fb = len(names) * spec.bits_per_feature; sp = g.shape.output_input_space
	S = g.output_sampled; nb = len(S) // 4
	print(f"== {lab}: {len(S)} output neurons ({nb}/bank), {len(S[0])} suffix bits, prefix {g.forced_prefix}, space {sp} = {sp/fb:.2f} frames")
	for b, bn in enumerate(("coll", "p(roll)", "q(pitch)", "r(yaw)")):
		bits = np.concatenate([np.array(s) for s in S[b * nb:(b + 1) * nb]]); bits = bits[bits < sp]
		f = (bits % fb) // spec.bits_per_feature; cnt = np.bincount(f, minlength=len(names)) / len(bits) * 100
		top = np.argsort(cnt)[::-1][:6]
		print(f"   {bn:9s} yaw-feature bits {cnt[[names.index(x) for x in YAW]].sum():5.1f}% ({' '.join(f'{x} {cnt[names.index(x)]:.1f}' for x in YAW)}) | top: " + " ".join(f"{names[i]} {cnt[i]:.1f}" for i in top))
