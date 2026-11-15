"""CTRL-48 (Luiz 15/11/2026): re-score a FINISHED run's stage-select with the pos-only
scale floor (--zrank-pos-floor), VAL side only; no TEST unit is scored.

In-process --recalc-headline on the run's exact argv (logs/controller/ctrl48/rebuild_argv.py
recalc + --val-units 20 --select-tail p99): the candidates are re-scored on the 20 val units
exactly as stage-select scores them, with two cuts that change no number used for selection:
  * the classical rival rows of each val report block are skipped (_pid_baseline → []),
    they are printed context, never ranked;
  * the run stops at the stage-select ranking (tail_rank_winner), before the test-unit
    scoring of the pick.
The captured (candidate metrics, p99 tails) are then ranked with the run's own calculator at
floor 0 — GATE: must give the logged headline — and at the measured floor (and 2.2 mm, the
stage-select-frame estimate).

  PYTHONPATH=src python scripts/ctrl48_posfloor_rescore.py <run.out> [more .out ...]
"""
import re
import shlex
import subprocess
import sys

import wnn.control.phased_ga as P
from wnn.ram.fitness.FitnessCalculatorControllerHarmonic import FitnessCalculatorControllerHarmonic as FC
from wnn.control.pos_floor import POS_SCALE_FLOOR_M
from wnn.control import select_tail as ST

REBUILD = "/Users/lacg/wnn/logs/controller/ctrl48/rebuild_argv.py"


class _Stop(Exception):
	pass


def rescore(out_path):
	text = open(out_path, errors="replace").read()
	logged = re.findall(r"\[stage-select\] HEADLINE stage=\S+ genome=(\S+)", text)[-1]
	logged_fit = dict(re.findall(r"^\s+((?:GRID|NEURONS|MEMORY)#\d)\s+val .*?fit=(-?[\d.]+)", text, re.M))
	tag = out_path.split("/")[-1][:-4]
	ckpt = out_path.rsplit("/", 1)[0] + f"/ckpt_{tag}"
	argv = shlex.split(subprocess.check_output([sys.executable, REBUILD, out_path, "recalc", ckpt, "--val-units", "20", "--select-tail", "p99"], text=True))
	cap = {}
	P._pid_baseline = lambda *a, **k: []
	orig_fit = FC.fitness

	def spy(self, ms):
		cap["calc"], cap["ms"] = self, list(ms)
		return orig_fit(self, ms)
	FC.fitness = spy

	def stop(labels, whms, tails):
		cap["labels"], cap["tails"] = list(labels), dict(tails)
		raise _Stop
	P.tail_rank_winner = stop
	sys.argv = ["phased_ga"] + argv
	try:
		P.main()
	except _Stop:
		pass
	finally:
		FC.fitness = orig_fit
	calc, ms, labels, tails = cap["calc"], cap["ms"], cap["labels"], cap["tails"]
	import json
	keep = ("stable_rate", "mean_attitude_error_deg", "mean_steady_error_deg", "reward", "mean_position_error_m",
	        "mean_altitude_error_m", "mean_position_hold_m")
	json.dump({"tag": tag, "logged": logged, "labels": labels, "tails": tails,
	           "metrics": [{k: getattr(x, k, None) for k in keep} for x in ms]},
	          open(f"/tmp/ctrl_posfloor/cands_{tag}.json", "w"), indent=1)
	res = {}
	for fl in (0.0, 0.0022, POS_SCALE_FLOOR_M):
		calc.zrank_pos_floor = fl
		whms = dict(zip(labels, orig_fit(calc, ms)))
		win, comb = ST.tail_rank_winner(labels, whms, tails)
		res[fl] = (win, whms)
	m = dict(zip(labels, ms))
	return tag, logged, logged_fit, res, m


def main(paths):
	rows = []
	for p in paths:
		tag, logged, logged_fit, res, m = rescore(p)
		w0, f0 = res[0.0]
		gate = "OK" if w0 == logged and all(abs(f0[k] - float(v)) < 6e-4 for k, v in logged_fit.items() if k in f0) else "MISMATCH"
		w1 = res[POS_SCALE_FLOOR_M][0]
		w2 = res[0.0022][0]
		v = lambda k: f"{m[k].mean_attitude_error_deg:.3f}/{m[k].mean_steady_error_deg:.3f}/{m[k].mean_position_error_m:.4f}"
		short = tag.replace("C-S2-d_wposx2_a51_rf_c200_dc_biasin_ms01_P_both_", "")
		rows.append(f"{short:24s} | {logged:10s} | {w0:10s} {gate:8s} | {w1:10s} {'same' if w1 == logged else 'CHANGED':7s} | {w2:10s} | {v(logged)} -> {v(w1)}")
		print(rows[-1], flush=True)
	print("\nrun                      | logged     | floor 0 (gate)      | floor 3.1 mm       | floor 2.2 mm | VAL err/steady/pos: logged -> 3.1 mm pick")
	print("\n".join(rows))


if __name__ == "__main__":
	main(sys.argv[1:])
