#!/usr/bin/env python3
"""How should a DFA controller SPLIT its address between state and sensor bits?

RESULT THAT REDIRECTED THIS (30/07/2026). Built to test a collision hypothesis —
"narrower addresses generalize better" — which the first two smokes FALSIFIED. On a
1layer winner (sn=0, so ob is purely sensor) held-out RISES with bits to a shallow
peak: 8->10.0%, 12->40.0%, 16->45.0%, 20->85.0%, 24->90.0%, 30->85.0%. On a dfa
winner narrowing ob is catastrophic (14->0.0%) because ob = sn + suffix with the
prefix FORCED, so shrinking ob starves sensors rather than colliding them.

So the live question is not "how wide" but "how split". The dfa winner runs
suffix=18, BELOW the 20-24 sensor optimum the 1layer sweep located, because its
12-bit forced prefix consumed most of a 30-bit address. Trading state bits for
sensor bits (or simply widening ob past 30) may be free performance.

MEASUREMENT THAT MOTIVATES THIS (30/07/2026). On the study's best cell the trained
memory holds ~127 distinct addresses per output neuron, each visited ~236 times.
Votes are saturated; capacity is untouched (~1e-5% of the address space). So the
held-out collapse is NOT starvation and NOT capacity — held-out episodes simply
visit addresses outside the trained set, and at ob=30 nothing collides to bridge
them. Narrowing the address is the only lever that makes an unseen state land on a
trained address.

WHAT VARIES
  ob      total output-neuron address width. The identity ob = sn + suffix holds
          for every winner (verified across 13), because the sn state bits are a
          FORCED prefix — the FSM-coherence invariant, deliberate and NOT to be
          "fixed": a state neuron that cannot see the whole state cannot implement
          a transition function. So only the SUFFIX varies here, and ob = sn + k.
  levels  levels_per_motor. Accumulator states = levels^motors, so 16 -> 8 -> 4
          cuts the FSM state requirement 2^16 -> 2^12 -> 2^8. The gap decomposition
          measured output quantization at ~0pp, so this should be nearly free —
          and it is the only way to afford both enough state AND a narrow address.

NESTED SUFFIXES. Smaller k TRUNCATES the winner's own suffix (first k indices)
rather than resampling; larger k extends it with fresh draws. So the ob arms are a
strict subset chain and the comparison isolates HOW MANY bits from WHICH bits.

Every arm: cells wiped, one training pass, scored on 5 report seeds (mean±SD), on
the fold-0 pool like everything since 29/07. Multiple training seeds because the
budget-probe smoke showed single-seed scatter swamping real effects.

Usage: bits_levels_sweep.py --winner ...gz --obs 10 12 14 18 24 30 --levels 4 8 16 \
           --train-seeds 31337002 31337003 --out experiments/dfa1l_markers/bits_sweep.json
"""
import argparse
import copy
import json
import math
import random
import statistics
import sys
import time

from wnn.control.checkpoint_io import load_controller_checkpoint
from wnn.control.evaluator import (ControllerEvaluator, EpisodeConfig,
                                   fit_thresholds_from_pid_rollouts)
from wnn.control.reward_gated import RewardGatedConfig
from wnn.control.training import DisturbanceConfig
from wnn.seeds import resolve_seed_set


def _ec(a):
	return EpisodeConfig(
		dt=0.001, steps_per_episode=a.steps,
		max_initial_tilt_rad=math.radians(a.tilt),
		max_initial_yaw_rad=math.radians(a.tilt),
		max_initial_body_rate=0.5, max_initial_yaw_rate=0.3,
		disturbance=DisturbanceConfig.preset(a.disturbance, seed=911))


def _rg(seed, ec, a):
	rg = RewardGatedConfig(seed=seed, episode_config=ec)
	rg.steps_per_episode, rg.progress = a.steps, False
	rg.teacher = a.teacher
	return rg


def _resuffix(genome, k, pool, rng):
	"""Nested resuffix: truncate to the first k, or extend with fresh draws."""
	out = []
	for row in genome.output_sampled:
		if k <= len(row):
			out.append(list(row[:k]))
		else:
			spare = [i for i in range(pool) if i not in set(row)]
			rng.shuffle(spare)
			out.append(list(row) + spare[:k - len(row)])
	return out


def _arm(payload, sn, suffix, levels, train_seed, a, rng):
	"""One (sn, suffix, levels) cell: rebuild, wipe, train, score.

	Parameterised by the SPLIT, not by ob — on a DFA architecture ob = sn + suffix
	with the prefix forced, so sweeping ob alone silently trades state for sensor
	and the two effects cannot be separated (the 30/07 dfa smoke: narrowing ob to
	14 left TWO sensor bits and scored 0.0%). Here sn and suffix move independently
	and ob is derived."""
	proto = payload["_proto"]
	sn_max = len(proto.state_sampled)
	if sn > sn_max:
		return None, f"sn={sn} > winner's {sn_max} state neurons (cannot invent state)"
	# levels_per_motor is DERIVED at materialize as output_neurons // num_motors
	# (evaluator.spec_from_genome), so assigning it on the spec is silently ignored —
	# it is the SAME KNOB as output_neurons. Refuse rather than pretend.
	derived_levels = payload["_proto"].output_neurons // payload["spec"].num_motors
	if levels != derived_levels:
		return None, (f"levels={levels} unreachable: derived from output_neurons "
		              f"({payload['_proto'].output_neurons}//{payload['spec'].num_motors}"
		              f"={derived_levels}); vary output_neurons instead")
	spec = copy.deepcopy(payload["spec"])
	genome = proto.clone()
	# Drop state neurons from the tail: the prefix shrinks, freeing address budget.
	genome.state_sampled = [list(r) for r in proto.state_sampled[:sn]]
	genome.state_neurons = sn
	genome.output_sampled = _resuffix(proto, suffix, genome.shape.output_input_space, rng)
	spec.state_neurons = sn
	spec.state_bits_per_neuron = sn + (len(proto.state_sampled[0]) if sn_max else 0)
	spec.output_bits_per_neuron = sn + suffix
	ob = sn + suffix
	genome.cells = None

	ec = _ec(a)
	thr = fit_thresholds_from_pid_rollouts(spec, num_episodes=10, seed=train_seed)
	ev = ControllerEvaluator(spec, num_eval_episodes=a.episodes, seed=train_seed,
	                         episode_config=ec, thresholds=thr,
	                         rg_config=_rg(train_seed, ec, a), num_eval_folds=5)
	t0 = time.time()
	ev._evaluate_core([genome], write_back=True)
	dt = time.time() - t0

	tris = []
	for rs in a.report_seeds:
		thr_r = fit_thresholds_from_pid_rollouts(spec, num_episodes=10, seed=rs)
		evr = ControllerEvaluator(spec, num_eval_episodes=a.episodes, seed=rs,
		                          episode_config=ec, thresholds=thr_r,
		                          rg_config=_rg(rs, ec, a), num_eval_folds=5)
		m = evr.score_genomes([genome])[0]
		tris.append((m.acc * 100.0, m.mean_attitude_error_deg,
		             getattr(m, "mean_steady_error_deg", None)))
	cells = genome.cells.cell_count() if genome.cells is not None else 0
	return {"ob": ob, "sn": sn, "suffix": suffix, "levels": levels, "train_seed": train_seed,
	        "stable": _ms([t[0] for t in tris]), "err_deg": _ms([t[1] for t in tris]),
	        "steady_deg": _ms([t[2] for t in tris]), "cells": cells,
	        "train_s": round(dt, 1)}, None


def _ms(xs):
	xs = [x for x in xs if x is not None]
	if not xs:
		return None
	return [statistics.mean(xs), statistics.pstdev(xs) if len(xs) > 1 else 0.0]


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--winner", required=True)
	ap.add_argument("--sns", type=int, nargs="+", default=None,
	                help="state-neuron counts (prefix width). Default: the winner's own")
	ap.add_argument("--suffixes", type=int, nargs="+", default=[18, 24, 30],
	                help="SENSOR bits per output neuron. ob = sn + suffix (derived)")
	ap.add_argument("--levels", type=int, nargs="+", default=[4, 8, 16])
	ap.add_argument("--train-seeds", type=int, nargs="+", default=[31337002, 31337003])
	ap.add_argument("--report-seeds", type=int, nargs="+",
	                default=[99990101, 99990102, 99990103, 99990104, 99990105])
	ap.add_argument("--episodes", type=int, default=100)
	ap.add_argument("--steps", type=int, default=2000)
	ap.add_argument("--tilt", type=float, default=5.0)
	ap.add_argument("--disturbance", default="L2D")
	ap.add_argument("--teacher", default="lqr")
	ap.add_argument("--out", required=True)
	a = ap.parse_args()

	payload = load_controller_checkpoint(a.winner)
	pop = payload.get("population") or []
	payload["_proto"] = pop[0] if pop else payload["best_genome"]
	sn = len(payload["_proto"].state_sampled)
	print(f"# bits x levels sweep: {a.winner}")
	print(f"# frozen: sn={sn} state neurons, {len(payload['_proto'].output_sampled)} output "
	      f"neurons; ob = sn + suffix, so suffix = ob - {sn}")
	print(f"# {len(a.sns or [sn])}sn x {len(a.suffixes)}suf x {len(a.levels)}lvl x "
	      f"{len(a.train_seeds)} tseed arms, each scored on {len(a.report_seeds)} report seeds")
	print()
	print(f"{'ob':>3} {'sn':>4} {'suf':>4} {'lvl':>4} {'tseed':>9} {'stable%':>13} {'err°':>13} {'cells':>9} {'train':>7}")
	rows, skipped = [], []
	rng = random.Random(20260730)
	sns = a.sns if a.sns else [sn]
	for lv in a.levels:
		for s_n in sns:
			for suf in a.suffixes:
				for ts in a.train_seeds:
					r, err = _arm(payload, s_n, suf, lv,
					              resolve_seed_set(base=ts, run_index=0).train, a, rng)
					if r is None:
						skipped.append({"sn": s_n, "suffix": suf, "levels": lv, "reason": err})
						print(f"{'--':>3} {suf:>4} {lv:>4} {ts:>9}   SKIP: {err}")
						continue
					rows.append(r)
					st, er = r["stable"], r["err_deg"]
					print(f"{r['ob']:>3} {r['sn']:>4} {r['suffix']:>4} {lv:>4} {ts:>9} "
					      f"{st[0]:>7.1f}±{st[1]:<5.1f} {er[0]:>7.2f}±{er[1]:<5.2f} "
					      f"{r['cells']:>9,} {r['train_s']:>6.0f}s", flush=True)
	with open(a.out, "w") as f:
		json.dump({"meta": vars(a) | {"frozen_sn": sn,
		           "note": "ob = sn + suffix (forced full-state prefix, FSM-coherence "
		                   "invariant). Suffixes are NESTED: smaller ob truncates the "
		                   "winner's own suffix, so arms isolate HOW MANY bits from "
		                   "WHICH bits."},
		           "rows": rows, "skipped": skipped}, f, indent=1)
	print(f"\n# wrote {a.out}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
