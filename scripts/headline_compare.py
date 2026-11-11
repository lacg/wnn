"""Compare two controller groups on the REPORT HEADLINE read from .out files.

Sibling of paired_power.py for runs that bank no sweepladder marker (the CTRL-8
stage-2 runs and the CTRL-5 multi-axis programme: they leave <tag>.out + a winner
file). Reads the LAST `[stage-select] HEADLINE held-out:` line of each .out (the
stage-select crowned genome scored on the test seeds — the house surface) and
the run's search seed from its LAST `--base-seed` in the `[argv]` line.

Five metrics together: stable %, err deg, steady deg, alt m, pos3d m (the .out's
`pos=` is the 3-D position error, cpu_score.rs). Sign convention as paired_power:
NEGATIVE delta = the ARM is better on every metric.

Verdict = the CTRL-8 rule_v2 template (experiments/ctrl8_ab_rule.json), margins
from that file unless overridden (multi-axis: the axis MEI):
  per metric, 95% t CI of d:  BETTER (CI < -margin) | WORSE (CI > +margin)
                              | SIMILAR (CI inside +-margin) | UNRESOLVED
  1. attitude block (stable, err, steady): no WORSE, >=1 BETTER, rest BETTER/SIMILAR -> that side WINS
  2. all three SIMILAR -> pos3d decides (beyond its margin; else no difference)
  3. BETTER and WORSE both present -> MIXED, no automatic verdict (report to Luiz)
  otherwise UNDECIDED. Interim stop (rule_v2 `stopping`): all three attitude
  metrics BETTER in the same direction.
--mode paired (default; seeds matched, CTRL-8) or welch (unpaired, multi-axis R4).
The statistics come from paired_power.py (exact t, Welch-Satterthwaite df).

Usage:
  PYTHONPATH=src/wnn python scripts/headline_compare.py \\
      --arm 'logs/controller/ctrl8_drag_sweep_pop50/arm_kd078.out' --arm 'logs/controller/ctrl8_ab_pop50/arm_s*.out' \\
      --ctl 'logs/controller/ctrl8_drag_sweep_pop50/twin_kd078.out' --ctl 'logs/controller/ctrl8_ab_pop50/twin_s*.out'
"""

import argparse
import glob
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paired_power as pp  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RULE = os.path.join(ROOT, 'experiments', 'ctrl8_ab_rule.json')
METRICS = ('stable', 'err', 'steady', 'alt', 'pos3d')
ATTITUDE = ('stable', 'err', 'steady')
LOWER_BETTER = {'stable': False, 'err': True, 'steady': True, 'alt': True, 'pos3d': True}
HEAD_RE = re.compile(r'\[stage-select\] HEADLINE held-out: stable=([0-9.]+)%\s+err=([0-9.]+)\S*\s+'
                     r'steady=([0-9.]+)\S*.*?alt=([0-9.]+)m\s+pos=([0-9.]+)m')
SEED_RE = re.compile(r'--base-seed\s+(\d+)')


def parse_args():
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument('--arm', action='append', required=True, help='.out glob for the ARM group; repeatable')
	ap.add_argument('--ctl', action='append', required=True, help='.out glob for the CONTROL group; repeatable')
	ap.add_argument('--mode', choices=('paired', 'welch'), default='paired')
	ap.add_argument('--margin', action='append', default=[],
	                help='metric=value override, e.g. err=0.6 (default: rule_v2 margins; alt 0.0)')
	ap.add_argument('--json', default=None, help='also write the result here')
	return ap.parse_args()


def read_out(path):
	"""(seed, {metric: value}) from one .out, or (seed, None) if no headline yet."""
	seed, head = None, None
	with open(path, 'rb') as f:
		for raw in f:
			line = raw.decode('utf-8', 'replace')
			if line.startswith('[argv]'):
				hits = SEED_RE.findall(line)
				seed = hits[-1] if hits else seed
			m = HEAD_RE.search(line)
			if m:
				head = dict(zip(METRICS, (float(x) for x in m.groups())))
	return seed, head


def load_group(globs):
	"""{seed: (path, values|None)}; a duplicate seed is an error, never a silent pick."""
	out = {}
	for g in globs:
		for path in sorted(glob.glob(g)):
			seed, vals = read_out(path)
			if seed is None:
				raise SystemExit('%s: no [argv] --base-seed — cannot place this run' % path)
			if seed in out:
				raise SystemExit('seed %s appears twice: %s and %s' % (seed, out[seed][0], path))
			out[seed] = (path, vals)
	return out


def margins(overrides):
	m = {'stable': 0.5, 'err': 0.1, 'steady': 0.1, 'alt': 0.0, 'pos3d': 0.0}
	try:
		rule = json.load(open(RULE))['rule_v2']['margins']
		m.update({'stable': rule['stable_pp'], 'err': rule['err_deg'],
		          'steady': rule['steady_deg'], 'pos3d': rule['pos3d_m']})
	except Exception as e:
		print('WARNING: rule_v2 margins unreadable (%s) — using built-in copies' % e)
	for ov in overrides:
		k, _, v = ov.partition('=')
		if k not in m:
			raise SystemExit('--margin: unknown metric %r (one of %s)' % (k, ', '.join(METRICS)))
		m[k] = float(v)
	return m


def signed(m, arm_v, ctl_v):
	raw = arm_v - ctl_v
	return raw if LOWER_BETTER[m] else -raw


def paired_rows(arm, ctl):
	seeds = sorted(s for s in arm if s in ctl and arm[s][1] and ctl[s][1])
	rows = {}
	for m in METRICS:
		xs = [signed(m, arm[s][1][m], ctl[s][1][m]) for s in seeds]
		n = len(xs)
		mu, sd = pp.mean_sd(xs)
		half = pp.t_crit(n - 1) * sd / math.sqrt(n) if n > 1 else float('nan')
		rows[m] = {'d': mu, 'lo': mu - half, 'hi': mu + half, 'n': n}
	return seeds, rows


def welch_rows(arm, ctl):
	a = [s for s in arm if arm[s][1]]
	c = [s for s in ctl if ctl[s][1]]
	rows = {}
	for m in METRICS:
		xs = [arm[s][1][m] for s in a]
		ys = [ctl[s][1][m] for s in c]
		ma, sa = pp.mean_sd(xs)
		mc, sc = pp.mean_sd(ys)
		d = signed(m, ma, mc)
		se, df = pp.welch_se_df(len(xs), sa, len(ys), sc)
		half = pp.t_crit(df) * se if not math.isnan(df) else float('nan')
		rows[m] = {'d': d, 'lo': d - half, 'hi': d + half, 'n': (len(xs), len(ys))}
	return (sorted(a), sorted(c)), rows


def classify(row, margin):
	lo, hi = row['lo'], row['hi']
	if math.isnan(lo):
		return 'UNRESOLVED'
	if hi < -margin:
		return 'BETTER'
	if lo > margin:
		return 'WORSE'
	if lo >= -margin and hi <= margin:
		return 'SIMILAR'
	return 'UNRESOLVED'


def loose_classes(cls, rows, mg):
	"""Reading B of rule_v2 step 1: an UNRESOLVED attitude metric that cannot be WORSE beyond
	its margin (CI hi <= +margin) is treated as SIMILAR-for-the-arm; one that cannot be BETTER
	beyond its margin (CI lo >= -margin) as SIMILAR-for-the-control. Only attitude metrics."""
	out = dict(cls)
	for m in ATTITUDE:
		if cls[m] != 'UNRESOLVED':
			continue
		r = rows[m]
		if r['hi'] <= mg[m] and r['d'] < 0:
			out[m] = 'SIMILAR'
		elif r['lo'] >= -mg[m] and r['d'] > 0:
			out[m] = 'SIMILAR'
	return out


def verdict(cls):
	"""rule_v2 steps 1-3 on the classified metrics (labels are from the ARM's side)."""
	att = [cls[m] for m in ATTITUDE]
	if 'BETTER' in att and 'WORSE' in att:
		return 'MIXED — no automatic verdict, report to Luiz', False
	if 'WORSE' not in att and 'BETTER' in att and all(c in ('BETTER', 'SIMILAR') for c in att):
		return 'ARM WINS (attitude block)', all(c == 'BETTER' for c in att)
	if 'BETTER' not in att and 'WORSE' in att and all(c in ('WORSE', 'SIMILAR') for c in att):
		return 'CONTROL WINS (attitude block)', all(c == 'WORSE' for c in att)
	if all(c == 'SIMILAR' for c in att):
		p = cls['pos3d']
		if p == 'BETTER':
			return 'ARM WINS (attitude similar; pos3d tie-break)', False
		if p == 'WORSE':
			return 'CONTROL WINS (attitude similar; pos3d tie-break)', False
		return 'NO DIFFERENCE (attitude similar; pos3d not resolved)', False
	return 'UNDECIDED — continue to n=5', False


def print_group(label, grp):
	for s in sorted(grp):
		path, v = grp[s]
		body = '  '.join('%s %.3f' % (m, v[m]) for m in METRICS) if v else 'NO HEADLINE YET (skipped)'
		print('  %-4s %s  %s   [%s]' % (label, s, body, os.path.relpath(path, ROOT)))


def main():
	a = parse_args()
	arm, ctl = load_group(a.arm), load_group(a.ctl)
	mg = margins(a.margin)
	print('surface: REPORT headline (stage-select crowned genome on the test seeds)')
	print('sign: NEGATIVE d = ARM better · mode: %s · margins: %s' % (
		a.mode, '  '.join('%s %g' % (m, mg[m]) for m in METRICS)))
	print_group('arm', arm)
	print_group('ctl', ctl)
	seeds, rows = (paired_rows if a.mode == 'paired' else welch_rows)(arm, ctl)
	cls = {m: classify(rows[m], mg[m]) for m in METRICS}
	print('\n  %-7s %9s %9s %9s  %s' % ('metric', 'mean d', 'CI lo', 'CI hi', 'class'))
	for m in METRICS:
		r = rows[m]
		print('  %-7s %9.3f %9.3f %9.3f  %s' % (m, r['d'], r['lo'], r['hi'], cls[m]))
	v, stop = verdict(cls)
	loose = verdict(loose_classes(cls, rows, mg))[0]
	n = rows['err']['n']
	print('\n  n = %s · VERDICT (strict: UNRESOLVED is neither better nor similar): %s' % (n, v))
	print('  reading B (an UNRESOLVED metric whose CI cannot be beyond-margin on the other side counts')
	print('  as better-or-similar for that side): %s' % loose)
	print('  rule_v2 step 1 wording admits both; which one applies is Luiz\'s call, not this tool\'s.')
	print('  interim-stop condition (all three attitude CIs clear their margins, same direction): %s'
	      % ('MET' if stop else 'not met'))
	if a.json:
		with open(a.json, 'w') as f:
			json.dump({'mode': a.mode, 'seeds': seeds, 'margins': mg, 'rows': rows, 'class': cls,
			           'verdict': v, 'verdict_reading_b': loose, 'interim_stop': stop}, f, indent=1)


if __name__ == '__main__':
	main()
