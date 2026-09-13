"""Backend benchmark: CPU vs Metal vs wgpu on the public forward, parity asserted.

	python scripts_bench_backends.py

Measured 12/09/2026 on the M4 Max (min of 3, ms): wgpu sits 1.5-3.3x behind the
hand-written Metal kernel at every size (u64 emulation in WGSL + per-call buffer
upload/readback), so Metal stays the primary backend on Apple silicon and wgpu is
the portable one — it does NOT replace Metal (docs/wnn_public_api_draft.md §6).
"""
import time
import numpy as np
from weightless import WiSARDClassifier, CellMode, backends

rng = np.random.default_rng(0)


def bench(n_train, n_pred, bits, neurons, total_bits=512, mode=CellMode.QUAD_WEIGHTED):
	protos = rng.integers(0, 2, (2, total_bits), dtype=np.uint8)
	y = rng.integers(0, 2, n_train)
	X = protos[y] ^ (rng.random((n_train, total_bits)) < 0.15).astype(np.uint8)
	yp = rng.integers(0, 2, n_pred)
	Xp = protos[yp] ^ (rng.random((n_pred, total_bits)) < 0.15).astype(np.uint8)
	clf = WiSARDClassifier(neurons_per_class=neurons, bits_per_neuron=bits, cell_mode=mode, random_state=0, backend="cpu").fit(X, y)
	row = f"n={n_pred:>7} b={bits:>2} npc={neurons:>3} keys={clf.export_keys()['keys'].size:>8}"
	ref = None
	avail = [b.split(":")[0] for b in backends()]
	for be in ("cpu", "metal", "wgpu"):
		if be not in avail:
			row += f" | {be:5s}      n/a"
			continue
		clf.set_params(backend=be)
		clf._scores(Xp, False)
		t = []
		for _ in range(3):
			t0 = time.perf_counter()
			s = clf._scores(Xp, False)
			t.append(time.perf_counter() - t0)
		if ref is None:
			ref = s
		else:
			assert np.allclose(ref, s, atol=1e-6), be
		row += f" | {be:5s} {min(t) * 1000:8.1f} ms"
	print(row)


if __name__ == "__main__":
	print("backends:", backends())
	for args in [(20000, 20000, 16, 50), (20000, 100000, 16, 50), (20000, 100000, 48, 50), (50000, 200000, 48, 100), (20000, 100000, 96, 50)]:
		bench(*args)
