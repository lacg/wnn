#!/usr/bin/env python
"""OFFLINE CONNECTIVITY — VALIDITY PROBE (15/09/2026; Luiz: go, true-delta label,
held-out conditional entropy).

QUESTION. Can a tap layout's closed-loop fitness be predicted WITHOUT flying it, from
how well its addresses predict the teacher's label on records it never saw? If yes,
the CONNECTIONS search can rank candidates in seconds instead of ~20 min a population.

PRE-REGISTERED PROTOCOL
  population  the run's CONNECTIONS checkpoint (default: the arm-B `_bd` seed-2 run —
              searched under the true-delta label on ABI 29), every genome with cells;
  dataset     `--episodes` DAgger episodes rolled out by the ELITE (pop[0]) with the
              run's own training config (same sim/teacher/draws as the trainer,
              ram_controller.offline_tap_probe), fold = episode % `--folds`;
  proxy       per genome = mean over output neurons of the 5-fold held-out conditional
              entropy H(label | address) (bits/record; Laplace alpha, unseen address ->
              training prior). LOWER = better. Per-motor means reported too;
  truth       the same genomes' in-search fitness: ControllerEvaluator.evaluate_batch
              under the run's calculator (rank combine) — plus the raw components;
  verdict     Spearman(proxy, fitness) over the population, and the overlap of the proxy's
              top-K with the fitness top-K. BAR (pre-registered): rho >= 0.7 -> build the
              surrogate GA; rho <= 0.4 -> CLOSED (as racing was); between -> report only.

Usage:  PYTHONPATH=src/wnn python scripts/offline_connectivity_probe.py [--tag TAG]
            [--episodes 200] [--folds 5] [--alpha 1.0] [--seed 0] [--out JSON]
Nothing is written to any marker; the JSON goes to experiments/offline_probe/.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import rescore_first_report_seed as rs  # noqa: E402
import recalc_headlines as rh  # noqa: E402

DEFAULT_TAG = "SL_C_b24n256_cf21_brushless_L4C_g10_s31337002_bd"


def spearman(a: list, b: list) -> float:
	def ranks(v):
		order = sorted(range(len(v)), key=lambda i: v[i])
		r = [0.0] * len(v)
		i = 0
		while i < len(order):
			j = i
			while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
				j += 1
			for k in range(i, j + 1):
				r[order[k]] = (i + j) / 2.0 + 1.0
			i = j + 1
		return r
	ra_, rb = ranks(a), ranks(b)
	n = len(a)
	ma, mb = statistics.mean(ra_), statistics.mean(rb)
	num = sum((x - ma) * (y - mb) for x, y in zip(ra_, rb))
	den = (sum((x - ma) ** 2 for x in ra_) * sum((y - mb) ** 2 for y in rb)) ** 0.5
	return num / den if den else float("nan")


def load_population(tag: str):
	from wnn.control.phased_ga import _stage_entries_from_checkpoints
	ck = rs._find_ckpt_dir(tag)
	for label, spec, res in _stage_entries_from_checkpoints(ck):
		if label == "CONNECTIONS":
			pop = list(res.final_population or [])
			return spec, pop, ck
	raise SystemExit(f"{tag}: no CONNECTIONS checkpoint under {ck}")


def build_evaluator(tag: str, marker: dict, spec, ec, args):
	"""The SEARCH's own evaluator (phased_ga._run_arch_phase's construction): PID-fitted
	thresholds on the train seed, the run's rg_config, CRN scoring."""
	from wnn.control.evaluator import ControllerEvaluator, fit_thresholds_from_pid_rollouts
	from wnn.control.phased_ga import _rg_config, _calib_ec
	seed = int(args.train_seed)
	thresholds = fit_thresholds_from_pid_rollouts(
		spec, num_episodes=10, seed=seed, geometry=getattr(ec, "geometry", None),
		alloc=getattr(ec, "alloc_residual", None), episode_config=_calib_ec(args, ec))
	return ControllerEvaluator(spec, num_eval_episodes=args.eval_episodes, seed=seed,
	                           episode_config=ec, thresholds=thresholds,
	                           rg_config=_rg_config(args, ec, seed),
	                           max_train_workers=args.train_workers,
	                           num_eval_folds=args.num_eval_folds, score_crn=args.score_crn)


def run_fitness(args, ev, pop):
	"""In-search fitness of every genome under the run's calculator (one rank combine
	over the whole population, as the GA does) + the raw components."""
	from wnn.control.phased_ga import _select_aggregation, _gate_args
	from wnn.ram.fitness import FitnessCalculatorControllerHarmonic
	ms = ev.evaluate_batch(pop)
	calc = FitnessCalculatorControllerHarmonic(
		weight_err_sq=args.fit_weight_err_sq, weight_stable=args.fit_weight_stable,
		weight_jerk=args.fit_weight_jerk, weight_mono=args.fit_weight_mono,
		weight_steady=getattr(args, "fit_weight_steady", 0.0),
		weight_effort=getattr(args, "fit_weight_effort", 0.0),
		weight_alt=getattr(args, "fit_weight_alt", 0.0),
		weight_pos=getattr(args, "fit_weight_pos", 0.0),
		aggregation=_select_aggregation(args), zrank_clamp=getattr(args, "zrank_clamp", 3.0),
		gate_stable_min=_gate_args(args)[0], gate_err_max=_gate_args(args)[1])
	fit = list(calc.fitness(ms))
	rows = [dict(fitness=float(f), reward=float(m.reward), stable=float(m.stable_rate),
	             err=float(m.mean_attitude_error_deg),
	             steady=(float(m.mean_steady_error_deg) if m.mean_steady_error_deg is not None else None),
	             alt=(float(m.mean_altitude_error_m) if getattr(m, "mean_altitude_error_m", None) is not None else None))
	        for f, m in zip(fit, ms)]
	return rows, calc.name


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
	ap.add_argument("--tag", default=DEFAULT_TAG)
	ap.add_argument("--episodes", type=int, default=200)
	ap.add_argument("--folds", type=int, default=5)
	ap.add_argument("--alpha", type=float, default=1.0)
	ap.add_argument("--seed", type=int, default=0, help="dataset rollout seed (0 = the run's train seed)")
	ap.add_argument("--top-k", type=int, default=10)
	ap.add_argument("--tilt-deg", type=float, default=None,
	                help="override the dataset's initial tilt (default: the trainer's curriculum spread, 8->30 deg; "
	                     "the SCORER flies 5 deg — pass 5 for a regime-matched read)")
	ap.add_argument("--skip-fitness", action="store_true", help="proxy only (no rollout re-score)")
	ap.add_argument("--fitness-from", default=None, help="reuse the rollout fitness rows of an earlier probe JSON (same tag)")
	ap.add_argument("--out", default=None)
	ap.add_argument("--data-root", default=os.environ.get("WNN_DATA_ROOT", ROOT),
	                help="checkout that holds logs/ + experiments/ (a worktree has neither)")
	pa = ap.parse_args()
	rs.ROOT = rh.ROOT = pa.data_root
	from wnn.control import _accel as ra
	if not hasattr(ra, "offline_tap_probe"):
		raise SystemExit("the installed ram_controller has no offline_tap_probe — build the probe wheel")
	from wnn.control.phased_ga import episode_config_from_args
	import glob
	paths = glob.glob(os.path.join(pa.data_root, "experiments", "*_markers", f"{pa.tag}.json"))
	marker = json.load(open(paths[0])) if paths else {}
	out = rs._find_out(pa.tag)
	facts = rs.parse_out(out)
	args = rs.build_run_args(pa.tag, marker, facts)
	spec, pop, ck = load_population(pa.tag)
	rh._apply_fitness_identity(args, ck, out)
	args.teacher = marker.get("teacher") or "mpcof"
	args.teacher_hover = rh._teacher_hover(marker)
	# The recalc's rebuild passes no training-side flag; the spec carries the rest.
	ec = episode_config_from_args(args)
	t0 = time.time()
	ev = build_evaluator(pa.tag, marker, spec, ec, args)
	print(f"[probe] {pa.tag}: {len(pop)} CONNECTIONS genomes, hover={args.teacher_hover}, "
	      f"train seed {args.train_seed}, thresholds fitted ({time.time() - t0:.0f}s)", flush=True)

	# ---- proxy: one dataset from the elite, every layout scored on it ----
	materialized = [ev._materialize(g) for g in pop]
	specs = {(s.state_neurons, s.output_bits_per_neuron, s.levels_per_motor) for s, _sc, _oc in materialized}
	assert len(specs) == 1, f"mixed shapes in the population: {specs}"
	elite_spec, sc0, oc0 = materialized[0]
	cells0 = getattr(pop[0], "cells", None)
	init_s, init_o = cells0.to_triples() if cells0 is not None else (None, None)
	elite = ev.controller_for(elite_spec, sc0, oc0, init_s, init_o)
	taps = [[int(x) for x in oc] for _s, _sc, oc in materialized]
	cfg = ev.packed_train_config()
	seed = pa.seed or int(args.train_seed)
	t1 = time.time()
	per_genome, ds = ra.offline_tap_probe(elite, taps, cfg, ev.target_rpy(), seed, pa.episodes, pa.folds, pa.alpha, pa.tilt_deg)
	episodes, records, mean_reward, mean_err_rad, diverged = ds
	print(f"[probe] dataset: {episodes} episodes, {records} records, mean reward {mean_reward:.2f}, "
	      f"mean err {mean_err_rad * 57.2958:.2f} deg, diverged {diverged} ({time.time() - t1:.0f}s)", flush=True)
	levels = elite_spec.levels_per_motor
	proxy, proxy_motor, gain = [], [], []
	for rows in per_genome:
		ce = [r[0] for r in rows]
		hy = [r[1] for r in rows]
		proxy.append(statistics.mean(ce))
		gain.append(statistics.mean(hy) - statistics.mean(ce))  # information the address buys
		proxy_motor.append([statistics.mean(ce[m * levels:(m + 1) * levels]) for m in range(elite_spec.num_motors)])
	print(f"[probe] proxy (mean held-out H(y|a), bits/record): elite {proxy[0]:.4f}, "
	      f"pop {min(proxy):.4f}..{max(proxy):.4f}; label entropy H(y) {statistics.mean(r[1] for r in per_genome[0]):.4f}", flush=True)

	# ---- truth: the search's own rollout fitness ----
	fit_rows, calc_name = ([], None)
	if pa.fitness_from:
		prev = json.load(open(pa.fitness_from))
		assert prev["tag"] == pa.tag and prev["n"] == len(pop), "fitness-from: different population"
		fit_rows, calc_name = prev["fitness"], prev.get("calculator")
		print(f"[probe] rollout fitness reused from {pa.fitness_from}", flush=True)
	elif not pa.skip_fitness:
		t2 = time.time()
		fit_rows, calc_name = run_fitness(args, ev, pop)
		print(f"[probe] rollout fitness: {calc_name} over {len(pop)} genomes ({time.time() - t2:.0f}s)", flush=True)

	result = dict(tag=pa.tag, at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
	              episodes=episodes, records=records, folds=pa.folds, alpha=pa.alpha, dataset_seed=seed, tilt_deg=pa.tilt_deg,
	              dataset=dict(mean_reward=mean_reward, mean_err_deg=mean_err_rad * 57.2958, diverged=diverged),
	              hover=args.teacher_hover, calculator=calc_name, n=len(pop),
	              proxy=proxy, proxy_gain=gain, proxy_per_motor=proxy_motor, fitness=fit_rows)
	if fit_rows:
		f = [r["fitness"] for r in fit_rows]
		verdict = dict(rho_fitness=spearman(proxy, f),
		               rho_err=spearman(proxy, [r["err"] for r in fit_rows]),
		               rho_stable=spearman(proxy, [-r["stable"] for r in fit_rows]),
		               rho_reward=spearman(proxy, [-r["reward"] for r in fit_rows]))
		k = pa.top_k
		top_proxy = set(sorted(range(len(pop)), key=lambda i: proxy[i])[:k])
		top_fit = set(sorted(range(len(pop)), key=lambda i: f[i])[:k])
		verdict["topk_overlap"] = len(top_proxy & top_fit)
		verdict["top_k"] = k
		bar = "BUILD (rho >= 0.7)" if verdict["rho_fitness"] >= 0.7 else ("CLOSED (rho <= 0.4)" if verdict["rho_fitness"] <= 0.4 else "INDETERMINATE (0.4 < rho < 0.7)")
		verdict["pre_registered_read"] = bar
		result["verdict"] = verdict
		print("\n[probe] VERDICT  Spearman(proxy, rollout fitness) = %.3f  (err %.3f, -stable %.3f, -reward %.3f); "
		      "top-%d overlap %d/%d  ->  %s" % (verdict["rho_fitness"], verdict["rho_err"], verdict["rho_stable"],
		                                        verdict["rho_reward"], k, verdict["topk_overlap"], k, bar), flush=True)
		print("  idx   proxy   gain    fitness   stable   err    steady")
		for i in sorted(range(len(pop)), key=lambda i: f[i]):
			r = fit_rows[i]
			print(f"  {i:3d}  {proxy[i]:.4f}  {gain[i]:.4f}  {r['fitness']:8.4f}  {r['stable'] * 100:5.1f}%  {r['err']:5.2f}  "
			      f"{(r['steady'] if r['steady'] is not None else float('nan')):5.2f}")
	os.makedirs(os.path.join(pa.data_root, "experiments", "offline_probe"), exist_ok=True)
	outp = pa.out or os.path.join(pa.data_root, "experiments", "offline_probe", f"PROBE_{pa.tag}{'_tilt%g' % pa.tilt_deg if pa.tilt_deg else ''}.json")
	with open(outp, "w") as fh:
		json.dump(result, fh, indent=1)
	print(f"[probe] wrote {outp}")


if __name__ == "__main__":
	main()
