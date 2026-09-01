#!/usr/bin/env python3
"""Rank EVERY banked controller marker on one gate-distance scale.

WHY THIS EXISTS. Each chain's STATE block carries its own little table of
gate-distances, covering only the runs that chain launched. Reading one of those
as "the record" is how the sweep ladder's hd 0.4034 got called a programme best
on 01/09/2026 when it was 38th across the archive — the gated weight sweep had
put nine runs inside the gate five weeks earlier. This script ranks all of them
together so a claim can be checked against the whole record instead of a slice.

THE SCALE is lifted verbatim from scripts/sweep_ladder_gamma.sh so a number here
is the number that chain would print:

    hd = 0.5556 * (err / 8.0) + 0.4444 * min(K * -log2(stable), 20.0)
    K  = log(0.5) / log(0.70)

It is the GATE's own geometry (stable >= 0.70 AND err <= 8.0 deg), not a neutral
summary: because stable enters through a log, points near the 70% gate are worth
far more than the same percentage points down at 30%. hd 1.0 sits ON the gate.
Use it to RANK; report the triple.

TWO REGIMES, NEVER POOLED. A run that flew --translation reports an alt= field in
its held-out line; an attitude-only run has no such field. Those are different
tasks, so the leaderboards are kept apart — 100% stable exists only in the
attitude-only half, and quoting it beside an altitude run compares two objectives.

ONLY headline_holdout IS READ — the stage-select winner scored on the report
seeds. Never the val seeds (that is what selection ran on) and never the
during-search gen lines (anti-predictive, repeatedly).
"""
import glob, json, math, os, re, sys

K = math.log(0.5) / math.log(0.70)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gate_distance(stable_pct, err_deg):
	s = max(stable_pct / 100.0, 1e-6)
	return 0.5556 * (err_deg / 8.0) + 0.4444 * min(K * -math.log2(s), 20.0)


def out_files():
	return {os.path.basename(p)[:-4]: p
	        for p in glob.glob(os.path.join(ROOT, 'logs/controller/*/*.out'))}


def state_neurons(path):
	"""Max state-neuron count the run actually flew, or None if unreadable."""
	if not path:
		return None
	try:
		txt = open(path, errors='ignore').read()
	except OSError:
		return None
	vals = [int(x) for x in re.findall(r'arch sn=(\d+)', txt)]
	vals += [int(x) for x in re.findall(r'GRID WINNER.*?sn=(\d+)', txt)]
	return max(vals) if vals else None


TEACHERS = ('mpcof', 'mpc', 'lqi', 'lqr', 'pid', 'afcal')


def teacher_of(tag):
	"""Teacher named in the tag, if any. Longest match first so mpcof beats mpc."""
	for t in TEACHERS:
		if t in tag:
			return t
	return '?'


def parse_marker(path, outs):
	try:
		d = json.load(open(path))
	except Exception:
		return None
	h = d.get('headline_holdout', '')
	ms = re.search(r'stable=([0-9.]+)%', h)
	me = re.search(r'err=([0-9.]+)', h)
	if not (ms and me):
		return None
	md = re.search(r'steady=([0-9.]+)', h)
	ma = re.search(r'alt=([0-9.]+)m', h)
	tag = os.path.basename(path)[:-5]
	stage = re.search(r'stage=(\w+) genome=(\S+)', d.get('headline_stage', ''))
	return dict(
		tag=tag,
		cohort=path.split(os.sep)[-2].replace('_markers', ''),
		stable=float(ms.group(1)),
		err=float(me.group(1)),
		steady=float(md.group(1)) if md else float('nan'),
		alt=float(ma.group(1)) if ma else None,
		altitude=ma is not None,
		sn=state_neurons(outs.get(tag)),
		stage=stage.group(1) if stage else '?',
		teacher=teacher_of(tag),
		date=(d.get('done') or '')[:10],
		hd=gate_distance(float(ms.group(1)), float(me.group(1))),
	)


def table(rows, n, show_alt):
	w = ['  rank      hd  stable     err  steady']
	w[0] += '     alt' if show_alt else '       '
	w[0] += '   sn  stage        date        cohort           tag'
	for i, r in enumerate(sorted(rows, key=lambda r: r['hd'])[:n], 1):
		alt = ('%7.3f' % r['alt']) if r['alt'] is not None else '      —'
		sn = ('%2d' % r['sn']) if r['sn'] is not None else ' ?'
		w.append('  %4d  %6.4f  %5.1f%%  %6.2f  %6.2f%s  %3s  %-11s  %-10s  %-15s  %s'
		         % (i, r['hd'], r['stable'], r['err'], r['steady'],
		            alt if show_alt else '       ', sn, r['stage'][:11], r['date'],
		            r['cohort'][:15], r['tag'][:52]))
	return '\n'.join(w)


def seed_spread(rows):
	"""Same recipe, different base seed — the band any n=1 claim sits inside."""
	fam = {}
	for r in rows:
		base = re.sub(r'_s\d+$', '', r['tag'])
		fam.setdefault(base, []).append(r)
	out = []
	for base, rs in sorted(fam.items(), key=lambda kv: min(x['hd'] for x in kv[1])):
		if len(rs) < 3:
			continue
		hds = sorted(x['hd'] for x in rs)
		sts = sorted(x['stable'] for x in rs)
		out.append('  %-46s n=%d   hd %.4f-%.4f (mean %.4f)   stable %.1f-%.1f%%'
		           % (base[:46], len(rs), hds[0], hds[-1], sum(hds) / len(hds), sts[0], sts[-1]))
	return '\n'.join(out[:14])


def main():
	outs = out_files()
	rows = [r for r in (parse_marker(p, outs)
	                    for p in glob.glob(os.path.join(ROOT, 'experiments/*_markers/*.json')))
	        if r]
	alt = [r for r in rows if r['altitude']]
	att = [r for r in rows if not r['altitude']]
	cells = {(a, b): 0 for a in (0, 1) for b in (0, 1)}
	unknown = 0
	for r in rows:
		if r['sn'] is None:
			unknown += 1
			continue
		cells[(1 if r['sn'] > 0 else 0, 1 if r['altitude'] else 0)] += 1

	doc = __doc__.split('\n')
	print('# Controller gate-distance leaderboard')
	print()
	print('Generated by `scripts/gate_distance_leaderboard.py` — re-run it, do not hand-edit.')
	print()
	print('```')
	print('\n'.join(l for l in doc[2:] if True).rstrip())
	print('```')
	print()
	print('## Coverage')
	print()
	print('```')
	print('markers with a headline held-out : %d' % len(rows))
	print('  altitude regimen (alt= present) : %d' % len(alt))
	print('  attitude-only                   : %d' % len(att))
	print('  state-neuron count unreadable   : %d  (no .out on disk)' % unknown)
	print('```')
	print()
	print('## The 2x2 that is not filled')
	print()
	print('```')
	print('                 attitude-only   altitude')
	print('  sn = 0            %6d       %6d' % (cells[(0, 0)], cells[(0, 1)]))
	print('  sn > 0            %6d       %6d' % (cells[(1, 0)], cells[(1, 1)]))
	print('```')
	print()
	print('State neurons and altitude have NEVER been flown together: every altitude run is')
	print('single-layer, every state-layer run is attitude-only. So nothing here says what a')
	print('state layer would do UNDER altitude — that cell is untested, not refuted.')
	print()
	print('## What actually separates the 100% runs')
	print()
	print('```')
	hundred = [r for r in rows if r['stable'] >= 100.0]
	print('markers at stable = 100.0%%           : %d' % len(hundred))
	print('  ... in the altitude regimen        : %d' % sum(r['altitude'] for r in hundred))
	print('  ... attitude-only                  : %d' % sum(not r['altitude'] for r in hundred))
	print('  ... with a state layer (sn > 0)    : %d' % sum(bool(r['sn']) for r in hundred))
	print('  ... single-layer (sn = 0)          : %d' % sum(not r['sn'] for r in hundred))
	seen = sorted({r['teacher'] for r in hundred})
	print('  ... teachers represented           : %s' % ', '.join(seen))
	print()
	print('best stable, altitude regimen        : %.1f%%' % max(r['stable'] for r in alt))
	print('best stable, attitude-only           : %.1f%%' % max(r['stable'] for r in att))
	print('```')
	print()
	print('The regime is the ONLY clean separator. 100% is reached single-layer as well as with')
	print('a state layer, and by more than one teacher — but never once with altitude in the')
	print('objective, where the ceiling is 98.0%. So "the state layer bought 100%" is REFUTED by')
	print('the sn=0 runs that also reach it; what the archive supports is that adding altitude')
	print('moved the ceiling, with the caveat that no run has ever changed only that one flag.')
	print()
	print('## Altitude regimen — the bar for anything flown today')
	print()
	print('```')
	print(table(alt, 25, True))
	print('```')
	print()
	print('## Attitude-only — a DIFFERENT task, never a comparator')
	print()
	print('```')
	print(table(att, 15, False))
	print('```')
	print()
	print('## Seed spread, same recipe (n>=3)')
	print()
	print('Wider than most effects any sweep has measured. An n=1 point sits inside this band,')
	print('so a "new best" needs the paired same-seed comparator, not the leaderboard top.')
	print()
	print('```')
	print(seed_spread(alt))
	print('```')


if __name__ == '__main__':
	sys.exit(main())
