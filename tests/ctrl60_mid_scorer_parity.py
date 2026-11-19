"""ABI 60 (CTRL-58 A2 amendment): the mid-window column end to end through the production
evaluator — the twin of ctrl58_tail_scorer_parity.py, same recipe genome and VAL unit.

  * mid OFF (default): the metrics carry no mean_mid_error_deg, and every other metric is the
    same with the column armed (CPU scorer: bit for bit) — the column is measurement only;
  * mid ARMED: a finite value on both scorers, GPU == CPU within the f32 tolerance (0.01°);
    on this fail-free unit err == 0.15·first + 0.65·mid + 0.20·steady bounds it: mid lies
    between steady and err/0.65;
  * armed together with the tail export (the tail FOLLOWS the 20 metric slots): both columns
    equal their single-armed values;
  * CRN (all K pools): the column is the mean over the pools.

Run: PYTHONPATH=src python tests/ctrl60_mid_scorer_parity.py   (needs ram_controller ABI >= 60)
"""
import importlib.util
from pathlib import Path

from wnn.control import _accel as A

_spec = importlib.util.spec_from_file_location("ctrl58_tail_scorer_parity", Path(__file__).with_name("ctrl58_tail_scorer_parity.py"))
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)


def main():
	assert A.ABI_VERSION >= 60, A.ABI_VERSION
	mk, g = T._setup()

	def score(gpu: bool, mid: bool, crn: bool = False, tail=None):
		ev = mk(gpu, crn=crn, tail=tail)
		ev.score_mid_window = mid
		return ev.score_genomes([g])[0]

	off, on = score(False, False), score(False, True)
	assert off.mean_mid_error_deg is None, "default must not carry the mid column"
	for k in T.FIELDS:
		assert getattr(off, k) == getattr(on, k), (k, getattr(off, k), getattr(on, k))
	mid, err, steady = on.mean_mid_error_deg, on.mean_attitude_error_deg, on.mean_steady_error_deg
	assert mid is not None and steady < mid < err / 0.65, (steady, mid, err)
	gpu = score(True, True)
	assert abs(gpu.mean_mid_error_deg - mid) <= 0.01, (gpu.mean_mid_error_deg, mid)
	print(f"[ctrl60-mid] single pool: mid cpu {mid:.4f} gpu {gpu.mean_mid_error_deg:.4f} "
	      f"(err {err:.3f}, steady {steady:.3f}); off == armed on {len(T.FIELDS)} fields")
	both, tail_only = score(False, True, tail=8.0), score(False, False, tail=8.0)
	assert both.mean_mid_error_deg == mid and both.episode_err_p99_deg == tail_only.episode_err_p99_deg
	assert tail_only.mean_mid_error_deg is None and tail_only.episode_err_p99_deg >= err
	print(f"[ctrl60-mid] with the tail export: mid {both.mean_mid_error_deg:.4f}, p99 {both.episode_err_p99_deg:.4f}")
	crn_off, crn_on = score(False, False, crn=True), score(False, True, crn=True)
	for k in T.FIELDS:
		assert getattr(crn_off, k) == getattr(crn_on, k), (k, getattr(crn_off, k), getattr(crn_on, k))
	crn_gpu = score(True, True, crn=True)
	assert crn_off.mean_mid_error_deg is None and abs(crn_gpu.mean_mid_error_deg - crn_on.mean_mid_error_deg) <= 0.01
	print(f"[ctrl60-mid] CRN: mid cpu {crn_on.mean_mid_error_deg:.4f} gpu {crn_gpu.mean_mid_error_deg:.4f}")
	print("ALL PASS")


if __name__ == "__main__":
	main()
