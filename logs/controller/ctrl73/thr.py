import sys, math
sys.path.insert(0, "/Users/lacg/wnn/scripts")
import ctrl_gap_student_trace as T
from wnn.control.phased_ga import _test_thresholds, _ctl_load
from wnn.control.feature_layout import feature_names
from wnn.seeds import resolve_seed_set
tag = sys.argv[1]
txt, a, ec = T.run_args(tag)
s = resolve_seed_set(base=a.base_seed, run_index=0, train=a.train_seed, test=a.test_seed, val=a.val_seed)
pay = _ctl_load(T.D + tag + "_winner.yaml.gz"); spec = pay["spec"]
thr = _test_thresholds(a, ec, spec, 0, s.train, use_score=True)
names = list(feature_names(spec))
print(len(thr), len(names), spec.bits_per_feature if hasattr(spec, "bits_per_feature") else None)
bpf = len(thr)//len(names); th = [thr[i*bpf:(i+1)*bpf] for i in range(len(names))]

for i, n in enumerate(names):
	if True:
		v = th[i] if not hasattr(th, "shape") else th[i]
		print(n, [round(float(x), 5) for x in v])
