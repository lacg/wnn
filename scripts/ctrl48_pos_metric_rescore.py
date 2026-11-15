"""CTRL-48 block 10b (Luiz 15/11/2026): re-rank finished runs' stage-select candidates with
--fit-pos-metric hold (+ --zrank-pos-floor auto), offline, VAL only, descriptive.

Input: the candidate VAL metrics and p99 tails captured by scripts/ctrl48_posfloor_rescore.py
(/tmp/ctrl_posfloor/cands_<tag>.json — the stage-select's own unrounded 20-val-unit means; that
script's gate reproduced every logged pick and fit). The ranking is the run's calculator
(FitnessCalculatorControllerHarmonic, recipe weights + viability gate) and stage-select rule
(select_tail.tail_rank_winner). Reported for each run: the logged pick (pos, no floor), the
pick under pos + floor, under hold (no floor) and under hold + floor, with VAL err / steady /
pos / pos_hold of the logged pick and of the hold + floor pick. TEST is never read.
"""
import json
import math
import sys
from types import SimpleNamespace

from wnn.ram.fitness.FitnessCalculatorControllerHarmonic import FitnessCalculatorControllerHarmonic as FC
from wnn.control.pos_floor import POS_HOLD_SCALE_FLOOR_M, POS_SCALE_FLOOR_M
from wnn.control.select_tail import tail_rank_winner

W = dict(weight_err_sq=0.2557, weight_stable=0.2045, weight_steady=0.358, weight_alt=0.0, weight_pos=0.1818,
         aggregation="zscore", zrank_clamp=3.0, gate_stable_min=0.70, gate_err_max=8.0)


def pick(d, metric, floor):
	ms = [SimpleNamespace(**m) for m in d["metrics"]]
	fit = FC(**W, pos_metric=metric, zrank_pos_floor=floor).fitness(ms)
	return tail_rank_winner(d["labels"], dict(zip(d["labels"], fit)), d["tails"])[0]


def main(tags):
	print(f"{'run (VAL, 20 units)':16s} | logged    | pos+floor | hold      | hold+floor | VAL err/steady/pos/pos_hold: logged -> hold+floor pick")
	for t in tags:
		d = json.load(open(f"/tmp/ctrl_posfloor/cands_C-S2-d_wposx2_a51_rf_c200_dc_biasin_ms01_P_both_{t}.json"))
		m = dict(zip(d["labels"], d["metrics"]))
		nan = [l for l in d["labels"] if not math.isfinite(m[l]["mean_position_hold_m"])]
		gate = pick(d, "pos", 0.0)
		assert gate == d["logged"], f"{t}: pos/no-floor re-rank {gate} != logged {d['logged']}"
		a, b, c = pick(d, "pos", POS_SCALE_FLOOR_M), pick(d, "hold", 0.0), pick(d, "hold", POS_HOLD_SCALE_FLOOR_M)
		v = lambda k: "{:.3f}/{:.3f}/{:.4f}/{:.4f}".format(*(m[k][x] for x in ("mean_attitude_error_deg", "mean_steady_error_deg", "mean_position_error_m", "mean_position_hold_m")))
		flag = "" if c == d["logged"] else " CHANGED"
		print(f"{t:16s} | {d['logged']:9s} | {a:9s} | {b:9s} | {c:10s}{flag:8s} | {v(d['logged'])} -> {v(c)}" + (f"  (NaN hold: {nan})" if nan else ""))


if __name__ == "__main__":
	main(sys.argv[1:] or [f"L1_full_s{s}" for s in range(1, 11)] + ["L1_L3_full_s1", "L1_L2y_full_s1", "L1_L2y_L3_full_s1"])
