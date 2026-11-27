"""CTRL-82 VAL (never a result): NEUTRAL roll-tail statistics per seed, 10,000 VAL eps per winner (100 VAL units x 100).
Neutral = unpaired tail quantiles of each arm's own per-episode distribution, and paired means on the top 1% selected by
the ALL-ARM mean (no arm selects its own bad luck)."""
import os
import numpy as np
L = lambda n: np.load(f"/tmp/ctrl82/summ_{n}.npz")["summ"] if os.path.exists(f"/tmp/ctrl82/summ_{n}.npz") else None
SEEDS = {1: {"G+T": "gt", "lk85": "lk85_s1", "lk90": "lk90", "l2b": "l2b"},
         2: {"G+T": "gt_s2", "lk85": "lk85_s2", "lk90": "lk90_s2", "l2b": "l2b_s2"},
         3: {"G+T": "gt_s3", "lk85": "lk85_s3", "lk90": "lk90_s3", "l2b": "l2b_s3"}}
print("columns: roll_pre mean | roll_pre p99 | roll_pre p99.9 | roll_max p99 | roll_max p99.9 | yaw_hold mean | whole err p99.9 || top1% by all-arm mean roll_pre: roll_pre / roll_max")
for s, arms in SEEDS.items():
	A = {k: L(v) for k, v in arms.items()}; A = {k: v for k, v in A.items() if v is not None}
	if not A: continue
	ref = next(iter(A.values()))
	assert all((v[:, :2] == ref[:, :2]).all() for v in A.values()), "VAL units differ across arms within a seed"
	ix = np.argsort(sum(v[:, 8] for v in A.values()))[::-1][:100]
	print(f"== seed {s} (VAL units {int(ref[0,0])}..{int(ref[-1,0])})")
	for k, v in A.items():
		q = lambda c, p: np.percentile(v[:, c], p)
		print(f"   {k:5s} {v[:,8].mean():6.3f} | {q(8,99):5.2f} | {q(8,99.9):5.2f} | {q(14,99):5.2f} | {q(14,99.9):5.2f} | {v[:,13].mean():6.3f} | {q(3,99.9):5.2f} || {v[ix,8].mean():5.2f} / {v[ix,14].mean():5.2f}")
