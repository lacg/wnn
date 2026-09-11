"""Paired-seed power analysis for controller A/B arms.

Answers the question an arm cannot answer for itself: given the per-seed spread
we actually observe, how many seeds does it take to resolve an effect of a given
size, and what is the false-positive rate of the win/loss gate we are using?

Reads banked markers, pairs each arm run against its OWN same-seed control on the
MEMORY multi-seed held-out row (never the headline — that carries the stage-select
draw), and reports per-metric mean delta, SD, CI, a paired t-test, the sign-test
tally, and the seeds required for a target effect.

Usage:
  PYTHONPATH=src/wnn python scripts/paired_power.py \\
      --arm _ls2 --arm _ls4 --arm _ls8 \\
      --base SL_C_b24n256_cf21_brushless_L4C_g10_s{seed} \\
      --seed 31337002 --seed 31337003 --seed 31337004 --seed 31337005 \\
      --control-override 31337002=SL_C_b24n256_cf21_brushless_L4C_g10_s31337002_crn
"""

import argparse
import glob
import json
import math
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARKER_GLOB = os.path.join(ROOT, 'experiments', '*_markers', '*.json')
METRICS = ('stable', 'err', 'steady', 'alt')
# Lower is better for err/steady/alt; higher is better for stable.
LOWER_BETTER = {'stable': False, 'err': True, 'steady': True, 'alt': True}


def parse_args():
	ap = argparse.ArgumentParser(description='Paired-seed power analysis for controller arms')
	ap.add_argument('--arm', action='append', required=True,
	                help='arm tag suffix, e.g. _ls2; repeatable (one tally per arm)')
	ap.add_argument('--base', required=True,
	                help='tag template with {seed}, e.g. SL_C_b24n256_..._s{seed}')
	ap.add_argument('--seed', action='append', required=True, help='seed; repeatable')
	ap.add_argument('--control-override', action='append', default=[],
	                help='seed=explicit_control_tag; repeatable')
	ap.add_argument('--metric', action='append', default=None,
	                help='restrict to these metrics (default: all four columns)')
	ap.add_argument('--targets', default='0.5,0.3,0.2,0.1',
	                help='effect sizes (in the metric unit) to size for')
	ap.add_argument('--power', type=float, default=0.80)
	ap.add_argument('--primary', default=None,
	                help='the ONE pre-registered verdict column; the others print as descriptive')
	return ap.parse_args()


def load_markers():
	out = {}
	for path in glob.glob(MARKER_GLOB):
		try:
			doc = json.load(open(path))
		except Exception:
			continue
		tag = doc.get('tag') or os.path.basename(path)[:-5]
		out[tag] = doc
	return out


def memory_row(doc):
	"""The MEMORY multi-seed held-out row — the stage-matched comparison surface."""
	raw = doc.get('held_memory_multiseed') or ''
	vals = {}
	for m in METRICS:
		hit = re.search(r'%s=([0-9.]+)' % m, raw)
		if not hit:
			return None
		vals[m] = float(hit.group(1))
	return vals


def deltas_for_arm(markers, base, arm, seeds, controls, metrics):
	"""Paired arm-minus-control deltas, signed so NEGATIVE always means the arm won."""
	rows = []
	for seed in seeds:
		arm_tag = base.format(seed=seed) + arm
		ctl_tag = controls.get(seed, base.format(seed=seed))
		a, c = markers.get(arm_tag), markers.get(ctl_tag)
		if a is None or c is None:
			rows.append({'seed': seed, 'missing': arm_tag if a is None else ctl_tag})
			continue
		av, cv = memory_row(a), memory_row(c)
		if av is None or cv is None:
			rows.append({'seed': seed, 'missing': 'MEMORY row'})
			continue
		d = {}
		for m in metrics:
			raw = av[m] - cv[m]
			d[m] = raw if LOWER_BETTER[m] else -raw
		rows.append({'seed': seed, 'delta': d, 'arm': av, 'ctl': cv})
	return rows


def mean_sd(xs):
	n = len(xs)
	if n == 0:
		return 0.0, 0.0
	mu = sum(xs) / n
	if n < 2:
		return mu, 0.0
	var = sum((x - mu) ** 2 for x in xs) / (n - 1)
	return mu, math.sqrt(var)


try:
	from scipy import stats as _st
except Exception:  # scipy is in the venv; the fallback only keeps the tool importable
	_st = None

# Fallback two-sided t critical values (alpha 0.05) if scipy is absent.
T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
          8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042}


def t_crit(df, alpha=0.05):
	"""Exact two-sided t critical value at this df (scipy), table fallback otherwise."""
	if df <= 0:
		return float('nan')
	if _st is not None:
		return float(_st.t.ppf(1 - alpha / 2, df))
	for k in sorted(T_CRIT):
		if df <= k:
			return T_CRIT[k]
	return 1.96


def binom_tail(k, n):
	"""P(X >= k) for X ~ Binomial(n, 0.5) — the sign gate's null."""
	total = sum(math.comb(n, i) for i in range(k, n + 1))
	return total / (2 ** n)


def paired_t_power(n, sd, effect, alpha=0.05):
	"""Exact power of a two-sided paired t-test at n pairs (noncentral t)."""
	if n < 2 or sd <= 0:
		return 0.0
	if _st is None:  # normal approximation only if scipy is missing
		z = effect / (sd / math.sqrt(n))
		return max(0.0, min(1.0, 0.5 + 0.5 * math.erf((z - 1.96) / math.sqrt(2))))
	df = n - 1
	ncp = effect / (sd / math.sqrt(n))
	tc = t_crit(df, alpha)
	# scipy's nct loses precision at large noncentrality (returns nan or drifts);
	# there the test is essentially certain, so clamp to 1 instead of trusting it.
	if ncp > 30.0:
		return 1.0
	pw = float(_st.nct.sf(tc, df, ncp) + _st.nct.cdf(-tc, df, ncp))
	if math.isnan(pw):
		return 1.0 if ncp > tc else 0.0
	return max(0.0, min(1.0, pw))


def seeds_needed(sd, effect, power, alpha=0.05, n_max=2000):
	"""Smallest n whose EXACT paired-t power reaches `power` (iterates on the t
	distribution — the normal approximation understates n badly below ~n=10)."""
	if effect <= 0 or sd <= 0:
		return 0
	for n in range(2, n_max + 1):
		if paired_t_power(n, sd, effect, alpha) >= power:
			return n
	return n_max


def mde(n, sd, power, alpha=0.05):
	"""Minimum detectable effect at n pairs: the effect with exactly `power`."""
	if n < 2 or sd <= 0:
		return float('nan')
	lo, hi = 0.0, 20.0 * sd
	for _ in range(60):
		mid = 0.5 * (lo + hi)
		if paired_t_power(n, sd, mid, alpha) >= power:
			hi = mid
		else:
			lo = mid
	return hi


def report_arm(arm, rows, metrics, targets, power, primary=None):
	print('\n=== arm %s ===' % arm)
	usable = [r for r in rows if 'delta' in r]
	for r in rows:
		if 'missing' in r:
			print('  seed %s: SKIPPED (missing %s)' % (r['seed'], r['missing']))
	if not usable:
		print('  no paired seeds — nothing to analyse')
		return
	n = len(usable)
	print('  paired seeds: %d' % n)
	k = len(metrics)
	print('  %-7s %9s %9s %9s %9s  %-10s %-11s %s'
	      % ('metric', 'mean d', 'SD', 'CI lo', 'CI hi', 'Holm-%d CI' % k, 'sign tally', 'verdict'))
	for m in metrics:
		xs = [r['delta'][m] for r in usable]
		mu, sd = mean_sd(xs)
		half = t_crit(n - 1) * sd / math.sqrt(n) if n > 1 else float('nan')
		half_h = t_crit(n - 1, 0.05 / k) * sd / math.sqrt(n) if n > 1 else float('nan')
		lo, hi = mu - half, mu + half
		wins = sum(1 for x in xs if x < 0)
		crosses_zero = not (lo > 0 or hi < 0)
		holm_ok = (mu - half_h > 0) or (mu + half_h < 0)
		verdict = 'indistinguishable from zero' if crosses_zero else (
			'ARM better' if hi < 0 else 'CONTROL better')
		role = ''
		if primary is not None:
			role = ' [PRIMARY]' if m == primary else ' [descriptive]'
		if m == 'stable':
			role += ' (bounded %, t-CI is approximate; prefer failure counts)'
		print('  %-7s %9.3f %9.3f %9.3f %9.3f  %-10s %-11s %s%s'
		      % (m, mu, sd, lo, hi, 'excl 0' if holm_ok else 'straddles',
		         '%d/%d wins' % (wins, n), verdict, role))
	print('\n  MDE at this n (effect with %.0f%% power, exact paired t):' % (power * 100))
	print('  ' + '  '.join('%s %.3f' % (m, mde(n, mean_sd([r['delta'][m] for r in usable])[1], power))
	                        for m in metrics))
	print('\n  SEEDS NEEDED (exact paired t, alpha 0.05 two-sided, power %.0f%%), from the SD above:'
	      % (power * 100))
	head = '  %-7s' % 'metric' + ''.join('%12s' % ('d=%.2f' % t) for t in targets)
	print(head)
	for m in metrics:
		xs = [r['delta'][m] for r in usable]
		_, sd = mean_sd(xs)
		cells = ''.join('%12d' % seeds_needed(sd, t, power) for t in targets)
		print('  %-7s%s' % (m, cells))
	print('  (a %d-pair SD is itself uncertain: its 95%% CI spans roughly [0.57x, 3.7x] at n=4)' % n)


def report_gate(n_seeds, n_rungs):
	print('\n=== the win/loss gate, as a hypothesis test ===')
	print('  A "k of n paired wins" rule is a SIGN TEST. Under the null (the lever does')
	print('  nothing) each seed is a coin flip, so the gate fires by chance at:')
	print('  %-14s %-12s %s' % ('rule', 'P(fire|null)', 'across %d rungs' % n_rungs))
	for k in range(n_seeds, max(0, n_seeds // 2), -1):
		p = binom_tail(k, n_seeds)
		any_p = 1 - (1 - p) ** n_rungs
		print('  %-14s %-12.4f %.4f' % ('%d of %d' % (k, n_seeds), p, any_p))
	print('  The sign test also DISCARDS MAGNITUDE: a 0.01 deg loss and a 0.77 deg loss')
	print('  count the same. Prefer the mean delta and its CI above for the verdict.')


def main():
	args = parse_args()
	metrics = tuple(args.metric) if args.metric else METRICS
	targets = [float(t) for t in args.targets.split(',')]
	controls = {}
	for ov in args.control_override:
		seed, _, tag = ov.partition('=')
		controls[seed] = tag
	markers = load_markers()
	print('markers loaded: %d' % len(markers))
	print('comparison surface: MEMORY multi-seed held-out row (stage-matched)')
	print('sign convention: NEGATIVE delta = the ARM is better')
	for arm in args.arm:
		rows = deltas_for_arm(markers, args.base, arm, args.seed, controls, metrics)
		report_arm(arm, rows, metrics, targets, args.power, args.primary)
	report_gate(len(args.seed), len(args.arm))


if __name__ == '__main__':
	main()
