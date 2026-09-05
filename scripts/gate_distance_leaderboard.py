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

TWO COLUMNS, TWO RULES (05/09/2026, Luiz). `hd` ranks headline_holdout — the
stage-select winner on the report seeds. But stage-select crowns the union-rank
winner of the top-3 of EVERY stage on the val seeds, so it is a DRAW: the two
byte-identical seed-31337002 runs at b32 n256 sit 7 ranks apart (0.1129 vs 0.1442)
while their MEMORY rows match to 0.01 deg. Ranking eras against each other on `hd`
therefore reads "rotation beat CRN", which is backwards.
So every row also carries `hdMEM` — the SAME rule for everyone, computed from
held_memory_multiseed (the MEMORY stage's mean over the report seeds, present in
every marker). Compare eras on hdMEM; publish the headline. Both are printed side
by side so the selection draw is visible rather than silently ranked.
The `fit` column says which scorer the search ran under: CRN (every genome on all
5 pools every generation, default since 03/09 21:05 EDT) or rot (the K-fold
rotation, where an elite kept the score of the pool it was born on).

ONLY held-out lines ARE READ — the report seeds. Never the val seeds (that is what
selection ran on) and never the during-search gen lines (anti-predictive, repeatedly).
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


def scorer_era(path):
	"""CRN or rotation, read off the .out startup line. Never assumed from a date."""
	if not path:
		return '?'
	try:
		txt = open(path, errors='ignore').read(200000)
	except OSError:
		return '?'
	m = re.search(r'fitness_pools=(CRN|rotation)', txt)
	if m:
		return 'CRN' if m.group(1) == 'CRN' else 'rot'
	# The fitness_pools line SHIPPED WITH CRN (commit 5c3e7e61, 03/09 21:05 EDT). A
	# real run log without it therefore predates CRN and is rotation-era — inferred
	# from the log's own contents, never from the marker's date.
	return 'rot' if 'Phased-GA controller search:' in txt else '?'


def stage_triple(line):
	"""stable/err/steady/alt out of a MULTI-SEED held-out line (means, not the SDs)."""
	ms = re.search(r'stable=([0-9.]+)', line)
	me = re.search(r'err=([0-9.]+)', line)
	if not (ms and me):
		return None
	md = re.search(r'steady=([0-9.]+)', line)
	ma = re.search(r'alt=([0-9.]+)', line)
	return dict(stable=float(ms.group(1)), err=float(me.group(1)),
	            steady=float(md.group(1)) if md else float('nan'),
	            alt=float(ma.group(1)) if ma else None)


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
	mem = stage_triple(d.get('held_memory_multiseed', '') or d.get('held_memory', ''))
	return dict(
		tag=tag,
		cohort=path.split(os.sep)[-2].replace('_markers', ''),
		stable=float(ms.group(1)),
		err=float(me.group(1)),
		steady=float(md.group(1)) if md else float('nan'),
		alt=float(ma.group(1)) if ma else None,
		# A run with NO z axis still prints alt=0.000m (the field exists, the plant
		# does not), so presence of the field is not the test — the translation A/B's
		# OFF arm otherwise lands in the altitude table and tops it with a task it
		# never flew. Every genuine altitude run carries a non-zero altitude error.
		altitude=ma is not None and float(ma.group(1)) > 0.0,
		sn=state_neurons(outs.get(tag)),
		stage=stage.group(1) if stage else '?',
		teacher=teacher_of(tag),
		date=(d.get('done') or '')[:10],
		hd=gate_distance(float(ms.group(1)), float(me.group(1))),
		fit=scorer_era(outs.get(tag)),
		mem=mem,
		hd_mem=gate_distance(mem['stable'], mem['err']) if mem else None,
	)


def table(rows, n, show_alt, key='hd'):
	w = ['  rank      hd   hdMEM  fit  stable     err  steady']
	w[0] += '     alt' if show_alt else '       '
	w[0] += '   sn  stage        date        cohort           tag'
	for i, r in enumerate(sorted(rows, key=lambda r: r[key] if r[key] is not None else 9e9)[:n], 1):
		alt = ('%7.3f' % r['alt']) if r['alt'] is not None else '      —'
		sn = ('%2d' % r['sn']) if r['sn'] is not None else ' ?'
		hm = ('%6.4f' % r['hd_mem']) if r['hd_mem'] is not None else '     —'
		w.append('  %4d  %6.4f  %s  %-3s  %5.1f%%  %6.2f  %6.2f%s  %3s  %-11s  %-10s  %-15s  %s'
		         % (i, r['hd'], hm, r['fit'], r['stable'], r['err'], r['steady'],
		            alt if show_alt else '       ', sn, r['stage'][:11], r['date'],
		            r['cohort'][:15], r['tag'][:52]))
	return '\n'.join(w)


def same_rule_table(rows, n):
	"""Rank on hdMEM — one fixed stage for everyone, so eras are comparable."""
	ok = [r for r in rows if r['hd_mem'] is not None]
	w = ['  rank   hdMEM  fit  stable     err  steady      alt   headline-hd  Δrank  tag']
	order_hd = {r['tag']: i for i, r in
	            enumerate(sorted(ok, key=lambda r: r['hd']), 1)}
	for i, r in enumerate(sorted(ok, key=lambda r: r['hd_mem'])[:n], 1):
		m = r['mem']
		alt = ('%7.3f' % m['alt']) if m['alt'] is not None else '      —'
		d = order_hd[r['tag']] - i
		w.append('  %4d  %6.4f  %-3s  %5.1f%%  %6.2f  %6.2f%s   %6.4f  %+5d  %s'
		         % (i, r['hd_mem'], r['fit'], m['stable'], m['err'], m['steady'],
		            alt, r['hd'], d, r['tag'][:52]))
	return '\n'.join(w)


def era_pairs(rows):
	"""CANDIDATE same-shape same-seed pairs across scorer eras.

	Keyed on (bits, neurons, airframe, disturbance, seed) parsed from the tag. That
	key does NOT verify the flag set — two runs can share it and differ in features
	— so this is a SHORTLIST to check, never a verdict. Quote a pair only after
	confirming both recipes (the STATE block records which are byte-identical).
	"""
	def key(tag):
		m = re.search(r'b(\d+)n(\d+)_(\w+?)_(L\w+?)_.*?s(\d+)', tag)
		return (m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)) if m else None

	fam = {}
	for r in rows:
		k = key(r['tag'])
		if k and r['hd_mem'] is not None and r['fit'] in ('CRN', 'rot'):
			fam.setdefault(k, []).append(r)
	out = []
	for k, rs in sorted(fam.items()):
		crn = [r for r in rs if r['fit'] == 'CRN']
		rot = [r for r in rs if r['fit'] == 'rot']
		if not (crn and rot):
			continue
		for a in crn:
			for b in rot:
				out.append('  b%s n%s s%s' % (k[0], k[1], k[4]))
				out.append('    CRN  hdMEM %.4f  (%5.1f%% / %5.2f° / %5.2f°)   headline-hd %.4f  %s'
				           % (a['hd_mem'], a['mem']['stable'], a['mem']['err'],
				              a['mem']['steady'], a['hd'], a['tag'][:44]))
				out.append('    rot  hdMEM %.4f  (%5.1f%% / %5.2f° / %5.2f°)   headline-hd %.4f  %s'
				           % (b['hd_mem'], b['mem']['stable'], b['mem']['err'],
				              b['mem']['steady'], b['hd'], b['tag'][:44]))
				out.append('    Δ same-rule (CRN − rot) %+.4f   Δ headline %+.4f   %s'
				           % (a['hd_mem'] - b['hd_mem'], a['hd'] - b['hd'],
				              'AGREE' if (a['hd_mem'] < b['hd_mem']) == (a['hd'] < b['hd'])
				              else 'DISAGREE — the headline gap is a selection draw'))
	return '\n'.join(out) if out else '  (no cross-era pair on this key yet)'


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
	print('## Same rule for everyone — ranked on the MEMORY stage (altitude regimen)')
	print()
	print('`hd` above ranks the stage-select winner, and stage-select is a val DRAW: it can')
	print('move a run several ranks without the controller changing. This table ranks the same')
	print('runs on `hdMEM` — the MEMORY multiseed held-out, one fixed stage for everyone — so')
	print('CRN-era and rotation-era runs are comparable. `Δrank` is headline-rank minus')
	print('same-rule rank: a large value means that row owes its placing to the draw.')
	print()
	print('```')
	print(same_rule_table(alt, 25))
	print('```')
	print()
	print('## CRN vs rotation on the same rule')
	print()
	print('CRN (every genome scored on all 5 pools every generation) replaced the K-fold')
	print('rotation on 03/09/2026. Rotation is NOT a rival to beat — it was untrustworthy, so')
	print('its scores are not a bar. These candidate pairs share (bits, neurons, airframe,')
	print('disturbance, seed); the KEY DOES NOT VERIFY THE FLAG SET, so confirm both recipes')
	print('before quoting one. Read the same-rule delta; the headline delta is shown only so a')
	print('selection draw is visible when the two disagree.')
	print()
	print('```')
	print(era_pairs(alt))
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
