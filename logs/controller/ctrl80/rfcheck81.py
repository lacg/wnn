"""CTRL-80 follow-up: resolve the candidate arm's receptive fields from the flown l2bylead argv + --drop-features gyro_z
(no flight, no training). Prints the yaw-bank family and the drop token; refusals surface as SystemExit."""
import shlex, sys
sys.path.insert(0, "/Users/lacg/wnn/scripts")
import ctrl_gap_student_trace as T
import wnn.control.phased_ga as PG
from wnn.control.arch_strategy import rf_summary, _rf_fields
from wnn.control.feature_layout import feature_layout
tag = "C-S2-d_wposx2_a64_rf_c200_dc_biasin_ms01_P_both_L1_yld_gt_l2bylead_full_s1"
txt = open(T.D + tag + ".out", errors="replace").read()
argv = shlex.split([l for l in txt.splitlines() if l.startswith("[argv]")][0][len("[argv]"):])
for extra in ([], ["--drop-features", "gyro_z"], ["--drop-features", "yaw_err"]):
	a = PG.build_arg_parser().parse_args(argv + extra)
	PG._refuse_projection_combos(a)
	print(extra, "drop token:", PG.drop_token(a), "| resolved:", PG.resolved_drop_features(a))
pay = PG._ctl_load(T.D + tag + "_winner.yaml.gz"); spec = pay["spec"]
import dataclasses
for drops in ((), ("gyro_z",)):
	s2 = dataclasses.replace(spec, drop_features=tuple(sorted(set(spec.drop_features or ()) | set(drops))))
	f = _rf_fields(s2); lay = feature_layout(s2)
	yaw = [d.name for d, m in zip(lay, f["rf_feature_masks"]) if m & int(lay[[d.name for d in lay].index("yaw_err")].family)]
	print(drops, "spec drops", s2.drop_features, "| yaw-family features still addressable:", yaw, "|", rf_summary(s2, [24]))
