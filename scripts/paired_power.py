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

UNPAIRED mode (--welch, multi-axis programme R4/§4): the pairing by search seed is
cosmetic (pair correlation ≈ 0), so the primary analysis is a Welch two-sample t of
the condition seeds (--seed) against an anchor that may carry MORE seeds
(--anchor-seed, default = the --seed set). Same MEMORY row, same four columns, same
sign convention; the CI uses the Welch–Satterthwaite df and the MDE/seeds-needed
block sizes the CONDITION arm with the anchor held at its banked n.
  ... --welch --seed 31337002 ... --seed 31337005 \\
      --anchor-seed 31337002 ... --anchor-seed 31337009 --control-suffix _hd29
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
	ap.add_argument('--control-suffix', default='',
	                help='control tag = base.format(seed) + this suffix (e.g. _hd); overrides win')
	ap.add_argument('--metric', action='append', default=None,
	                help='restrict to these metrics (default: all four columns)')
	ap.add_argument('--targets', default='0.5,0.3,0.2,0.1',
	                help='effect sizes (in the metric unit) to size for')
	ap.add_argument('--power', type=float, default=0.80)
	ap.add_argument('--primary', default=None,
	                help='the ONE pre-registered verdict column; the others print as descriptive')
	ap.add_argument('--welch', action='store_true',
	                help='UNPAIRED Welch t: condition seeds (--seed) vs anchor seeds (--anchor-seed)')
	ap.add_argument('--anchor-seed', action='append', default=None,
	                help='anchor/control seed for --welch; repeatable (default: the --seed set)')
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


def deltas_for_arm(markers, base, arm, seeds, controls, metrics, control_suffix=''):
	"""Paired arm-minus-control deltas, signed so NEGATIVE always means the arm won."""
	rows = []
	for seed in seeds:
		arm_tag = base.format(seed=seed) + arm
		ctl_tag = controls.get(seed, base.format(seed=seed) + control_suffix)
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


def arm_tags(base, arm, seeds):
	"""(seed, tag) per condition seed."""
	return [(seed, base.format(seed=seed) + arm) for seed in seeds]


def control_tags(base, seeds, controls, control_suffix=''):
	"""(seed, tag) per anchor seed — explicit override wins, else base + suffix."""
	return [(seed, controls.get(seed, base.format(seed=seed) + control_suffix)) for seed in seeds]


def group_values(markers, tags, metrics):
	"""Per-seed MEMORY-row values for one UNPAIRED group (the Welch surface). A
	missing marker or row is reported as such, never silently dropped."""
	rows = []
	for seed, tag in tags:
		doc = markers.get(tag)
		vals = memory_row(doc) if doc is not None else None
		if vals is None:
			rows.append({'seed': seed, 'missing': tag if doc is None else 'MEMORY row'})
			continue
		rows.append({'seed': seed, 'vals': {m: vals[m] for m in metrics}})
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


def nct_power(df, ncp, alpha=0.05):
	"""Exact two-sided power of a t-test with `df` degrees of freedom at
	noncentrality `ncp` — shared by the paired (df=n-1) and Welch (WS df) sizing."""
	if _st is None:  # normal approximation only if scipy is missing
		return max(0.0, min(1.0, 0.5 + 0.5 * math.erf((ncp - 1.96) / math.sqrt(2))))
	tc = t_crit(df, alpha)
	# scipy's nct loses precision at large noncentrality (returns nan or drifts);
	# there the test is essentially certain, so clamp to 1 instead of trusting it.
	if ncp > 30.0:
		return 1.0
	pw = float(_st.nct.sf(tc, df, ncp) + _st.nct.cdf(-tc, df, ncp))
	if math.isnan(pw):
		return 1.0 if ncp > tc else 0.0
	return max(0.0, min(1.0, pw))


def paired_t_power(n, sd, effect, alpha=0.05):
	"""Exact power of a two-sided paired t-test at n pairs (noncentral t)."""
	if n < 2 or sd <= 0:
		return 0.0
	return nct_power(n - 1, effect / (sd / math.sqrt(n)), alpha)


def welch_se_df(na, sa, nc, sc):
	"""Welch two-sample standard error and Welch–Satterthwaite df for
	(condition n=na, SD=sa) vs (anchor n=nc, SD=sc)."""
	va, vc = sa * sa / na, sc * sc / nc
	se = math.sqrt(va + vc)
	if se <= 0.0 or na < 2 or nc < 2:
		return se, float('nan')
	df = (va + vc) ** 2 / (va * va / (na - 1) + vc * vc / (nc - 1))
	return se, df


def welch_power(na, sa, nc, sc, effect, alpha=0.05):
	"""Exact power of the two-sided Welch t at these group sizes and SDs."""
	se, df = welch_se_df(na, sa, nc, sc)
	if se <= 0.0 or math.isnan(df):
		return 0.0
	return nct_power(df, effect / se, alpha)


def welch_seeds_needed(sa, nc, sc, effect, power, alpha=0.05, n_max=2000):
	"""Smallest CONDITION n (anchor fixed at its banked nc) whose Welch power
	reaches `power` for `effect`, from the two observed SDs. None = unreachable:
	with the anchor fixed, SE >= sc/sqrt(nc) whatever the condition n, so the
	anchor's own spread caps the power — the remedy is MORE ANCHOR seeds."""
	if effect <= 0 or (sa <= 0 and sc <= 0):
		return 0
	# Asymptote na -> inf: the arm term vanishes, df -> nc-1 (the anchor alone).
	if sc > 0 and nct_power(nc - 1, effect / (sc / math.sqrt(nc)), alpha) < power:
		return None
	for na in range(2, n_max + 1):
		if welch_power(na, sa, nc, sc, effect, alpha) >= power:
			return na
	return n_max


def welch_mde(na, sa, nc, sc, power, alpha=0.05):
	"""Minimum detectable effect at (na, nc): the effect with exactly `power`."""
	if na < 2 or nc < 2 or (sa <= 0 and sc <= 0):
		return float('nan')
	lo, hi = 0.0, 20.0 * max(sa, sc)
	for _ in range(60):
		mid = 0.5 * (lo + hi)
		if welch_power(na, sa, nc, sc, mid, alpha) >= power:
			hi = mid
		else:
			lo = mid
	return hi


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


def welch_row(m, arm_rows, ctl_rows, k, power):
	"""One metric's Welch line: arm/anchor means, signed mean difference (NEGATIVE =
	arm better, as in the paired mode), SE, WS df, 95% CI, Holm-k CI and MDE."""
	xs = [r['vals'][m] for r in arm_rows]
	ys = [r['vals'][m] for r in ctl_rows]
	ma, sa = mean_sd(xs)
	mc, sc = mean_sd(ys)
	raw = ma - mc
	d = raw if LOWER_BETTER[m] else -raw
	se, df = welch_se_df(len(xs), sa, len(ys), sc)
	half = t_crit(df) * se if not math.isnan(df) else float('nan')
	half_h = t_crit(df, 0.05 / k) * se if not math.isnan(df) else float('nan')
	return {'metric': m, 'arm_mean': ma, 'arm_sd': sa, 'ctl_mean': mc, 'ctl_sd': sc,
	        'd': d, 'se': se, 'df': df, 'lo': d - half, 'hi': d + half,
	        'holm_ok': (d - half_h > 0) or (d + half_h < 0),
	        'mde': welch_mde(len(xs), sa, len(ys), sc, power)}


def welch_verdict(row, primary):
	"""verdict + role tag, same wording as the paired mode."""
	lo, hi = row['lo'], row['hi']
	crosses_zero = math.isnan(lo) or not (lo > 0 or hi < 0)
	verdict = 'indistinguishable from zero' if crosses_zero else (
		'ARM better' if hi < 0 else 'CONTROL better')
	role = ''
	if primary is not None:
		role = ' [PRIMARY]' if row['metric'] == primary else ' [descriptive]'
	if row['metric'] == 'stable':
		role += ' (bounded %, t-CI is approximate; prefer failure counts)'
	return verdict + role


def print_skipped(rows, label):
	for r in rows:
		if 'missing' in r:
			print('  %s seed %s: SKIPPED (missing %s)' % (label, r['seed'], r['missing']))


def report_welch_arm(arm, arm_rows, ctl_rows, metrics, targets, power, primary=None):
	"""UNPAIRED report: condition seeds vs anchor seeds, Welch t per metric, then the
	MDE at the banked (na, nc) and the CONDITION seeds needed with the anchor fixed."""
	print('\n=== arm %s (UNPAIRED Welch t vs anchor) ===' % arm)
	print_skipped(arm_rows, 'arm')
	print_skipped(ctl_rows, 'anchor')
	arm_ok = [r for r in arm_rows if 'vals' in r]
	ctl_ok = [r for r in ctl_rows if 'vals' in r]
	na, nc = len(arm_ok), len(ctl_ok)
	if na < 2 or nc < 2:
		print('  need >= 2 seeds in BOTH groups (arm %d, anchor %d) — nothing to analyse' % (na, nc))
		return
	print('  condition seeds: %d   anchor seeds: %d' % (na, nc))
	k = len(metrics)
	print('  %-7s %9s %9s %9s %9s %7s %9s %9s  %-10s %s'
	      % ('metric', 'arm mean', 'anch mean', 'mean d', 'SE', 'df', 'CI lo', 'CI hi',
	         'Holm-%d CI' % k, 'verdict'))
	rows = [welch_row(m, arm_ok, ctl_ok, k, power) for m in metrics]
	for r in rows:
		print('  %-7s %9.3f %9.3f %9.3f %9.3f %7.2f %9.3f %9.3f  %-10s %s'
		      % (r['metric'], r['arm_mean'], r['ctl_mean'], r['d'], r['se'], r['df'],
		         r['lo'], r['hi'], 'excl 0' if r['holm_ok'] else 'straddles',
		         welch_verdict(r, primary)))
	print('  group SDs: ' + '  '.join('%s arm %.3f / anchor %.3f' % (r['metric'], r['arm_sd'], r['ctl_sd'])
	                                 for r in rows))
	print('\n  MDE at (arm n=%d, anchor n=%d) (effect with %.0f%% power, exact Welch t):'
	      % (na, nc, power * 100))
	print('  ' + '  '.join('%s %.3f' % (r['metric'], r['mde']) for r in rows))
	report_welch_seeds(rows, nc, targets, power)


def report_welch_seeds(rows, nc, targets, power):
	"""CONDITION seeds needed per target effect, anchor held at its banked nc."""
	print('\n  CONDITION SEEDS NEEDED (exact Welch t, alpha 0.05 two-sided, power %.0f%%, '
	      'anchor FIXED at n=%d), from the group SDs above:' % (power * 100, nc))
	print('  %-7s' % 'metric' + ''.join('%14s' % ('d=%.2f' % t) for t in targets))
	for r in rows:
		cells = ''.join('%14s' % seeds_cell(welch_seeds_needed(r['arm_sd'], nc, r['ctl_sd'], t, power))
		                for t in targets)
		print('  %-7s%s' % (r['metric'], cells))
	print('  ("anchor-capped" = with the anchor fixed at n=%d its own SD/sqrt(n) floors the SE,' % nc)
	print('   so NO condition n reaches the target — extend the ANCHOR, not the arm. A 4-seed SD')
	print('   carries a 95% CI of roughly [0.57x, 3.7x]: quote every MDE as a range.)')


def seeds_cell(n):
	return 'anchor-capped' if n is None else '%d' % n


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
	if args.welch:
		main_welch(args, markers, controls, metrics, targets)
		return
	for arm in args.arm:
		rows = deltas_for_arm(markers, args.base, arm, args.seed, controls, metrics, args.control_suffix)
		report_arm(arm, rows, metrics, targets, args.power, args.primary)
	report_gate(len(args.seed), len(args.arm))


def main_welch(args, markers, controls, metrics, targets):
	"""--welch: every arm's condition seeds against ONE anchor group. No sign gate —
	there are no pairs to tally."""
	anchor_seeds = args.anchor_seed or args.seed
	print('test: UNPAIRED Welch two-sample t (Welch–Satterthwaite df); '
	      'anchor seeds %s' % ','.join(anchor_seeds))
	ctl_rows = group_values(markers, control_tags(args.base, anchor_seeds, controls,
	                                              args.control_suffix), metrics)
	for arm in args.arm:
		arm_rows = group_values(markers, arm_tags(args.base, arm, args.seed), metrics)
		report_welch_arm(arm, arm_rows, ctl_rows, metrics, targets, args.power, args.primary)


if __name__ == '__main__':
	main()
