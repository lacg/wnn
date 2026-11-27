"""CTRL-80 follow-up: where the yaw bank's connections and cells go (winner genomes, no flight).
Output input space = num_features x 8 thermometer bits, feature f -> bits [8f, 8f+8). Neuron n -> bank n // L, banks [T, p, q, r]."""
import sys
import numpy as np
import wnn.control.phased_ga as PG
from wnn.control.feature_layout import feature_names
D = "/Users/lacg/wnn/logs/controller/ctrl48/"
ARMS = {"l2b": "a63_rf_c200_dc_biasin_ms01_P_both_L1_yld_gt_l2b", "lead": "a64_rf_c200_dc_biasin_ms01_P_both_L1_yld_gt_l2bylead"}
SHOW = ("gyro_z", "yaw_err", "yaw_err_i", "yaw_lead", "tgt_y", "abp_r", "abp_p")
print(f"{'run':10s} {'L':>4s} {'yaw cells':>10s} {'all cells':>10s} {'yaw conn':>8s} | yaw-bank connection share % : " + " ".join(f"{s:>9s}" for s in SHOW) + " | yaw neurons with >=1 lead bit %")
for s in (1, 2, 3):
	for arm, mid in ARMS.items():
		p = PG._ctl_load(D + f"C-S2-d_wposx2_{mid}_full_s{s}_winner.yaml.gz"); g = p["best_genome"]; names = list(feature_names(p["spec"]))
		L = g.output_neurons // 4; bpf = 8
		conns = np.array(g.output_sampled)                     # (neurons, bits) indices into the output input space
		yb = conns[3 * L:4 * L]; feat = np.where(yb < len(names) * bpf, yb // bpf, -1)
		share = {n: 100.0 * (feat == names.index(n)).mean() if n in names else float("nan") for n in SHOW}
		with_lead = 100.0 * (feat == names.index("yaw_lead")).any(1).mean() if "yaw_lead" in names else float("nan")
		_, oc = g.cells.to_triples(); nn = np.fromiter((c[0] for c in oc), dtype=np.int64, count=len(oc))
		yc = int(((nn >= 3 * L) & (nn < 4 * L)).sum())
		print(f"{arm}_s{s:<6d} {L:4d} {yc:10d} {len(oc):10d} {yb.size:8d} | {'':28s}" + " ".join(f"{share[n]:9.1f}" for n in SHOW) + f" | {with_lead:.0f}")
