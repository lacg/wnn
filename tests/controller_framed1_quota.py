"""Verify exact-quota framed1 init (Luiz 16/08 round-2 FQ spec)."""
import sys
sys.path.insert(0, "/Users/lacg/wnn/src/wnn")
import numpy as np
from collections import Counter
from wnn.control.recurrent_genome import (
	RecurrentArchShape, RecurrentArchGenome, RecurrentArchConfig,
	_framed1_slot_schedule, _sample_framed1,
)

K, BPF, NFEAT, MOTORS = 4, 8, 18, 4
FRAME = NFEAT * BPF           # 144
SPACE = K * FRAME             # 576
N = 240                       # 60 levels/motor

# --- 1. schedule: exact quotas globally AND per motor block -------------------
rng = np.random.default_rng(31337002)
sched = _framed1_slot_schedule(N, K, MOTORS, rng)
assert len(sched) == N
tot = Counter(sched)
# slot 3 = newest = Luiz's window0
assert (tot[3], tot[2], tot[1], tot[0]) == (128, 64, 32, 16), tot
for m in range(MOTORS):
	blk = Counter(sched[m * 60:(m + 1) * 60])
	assert (blk[3], blk[2], blk[1], blk[0]) == (32, 16, 8, 4), (m, blk)
print("1. schedule: 128/64/32/16 global, 32/16/8/4 per motor — EXACT ✓")

# --- 2. genome init: frame purity + coverage per width ------------------------
shape = RecurrentArchShape(prefix_factor=0, state_input_space=0,
                           output_input_space=SPACE, output_quantum=MOTORS)
for width, want in [(18, {1: 18}), (30, {2: 12, 1: 6}), (36, {2: 18})]:
	cfg = RecurrentArchConfig(conn_policy="framed1", bits_per_feature=BPF,
	                          input_window_k=K)
	g = RecurrentArchGenome.random(shape, 0, N, 0, width,
	                               np.random.default_rng(7), config=cfg)
	slot_counts = Counter()
	for suffix in g.output_sampled:
		assert len(suffix) == width
		frames = {b // FRAME for b in suffix}
		assert len(frames) == 1, f"width={width}: neuron spans frames {frames}"
		slot_counts[frames.pop()] += 1
		per_feat = Counter((b % FRAME) // BPF for b in suffix)
		assert Counter(per_feat.values()) == Counter(want), \
			f"width={width}: coverage {Counter(per_feat.values())} != {want}"
	assert (slot_counts[3], slot_counts[2], slot_counts[1], slot_counts[0]) == \
		(128, 64, 32, 16), (width, slot_counts)
	print(f"2. b={width}: frame-pure, quotas exact, thresholds/feature {want} ✓")

# --- 3. neurogenesis path (slot=None) still works, k=1 degenerates to min1 ----
s = _sample_framed1(SPACE, 18, BPF, K, np.random.default_rng(3))
assert len({b // FRAME for b in s}) == 1
s1 = _sample_framed1(FRAME, 18, BPF, 1, np.random.default_rng(3))
assert len(s1) == 18 and len({b // BPF for b in s1}) == 18
print("3. weighted per-neuron draw (neurogenesis) + k=1 degenerate — OK ✓")

# --- 4. non-divisible sizes don't crash, quotas still sum ---------------------
for n in (7, 61, 240):
	sc = _framed1_slot_schedule(n, K, MOTORS, np.random.default_rng(1))
	assert len(sc) == n and all(0 <= x < K for x in sc)
print("4. odd sizes safe ✓\nALL PASS")
