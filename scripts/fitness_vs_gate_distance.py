#!/usr/bin/env python3
"""Do the FITNESS rule and the GATE-DISTANCE yardstick order the archive the same way?

WHY THIS EXISTS (05/09/2026, Luiz's question). The leaderboard ranks finished runs on
gate distance; the GA ranks genomes on the weighted gated combine. They are different
functions of different subsets of the metrics, so "does the yardstick agree with the
optimiser, and if not, which one is right?" is a real question and it was unanswerable
from banked data — until you notice the per-seed RESULT lines carry `reward`, which is
the one column the fitness needs and the marker summary lines drop.

    GATE DISTANCE (the leaderboard yardstick — an ABSOLUTE scale)
        gd = 0.5556*(err/8.0) + 0.4444*min(K*-log2(stable), 20.0),  K = log0.5/log0.70
        · 2 of the 5 reported metrics: stable and err ONLY.
        · err LINEAR; stable through a LOG (a point is worth ~3.3x more at 30% than 98%).
        · gate violation folded into the same sum, continuously.
        · absolute: a number means the same thing in any cohort, any year.

    FITNESS (what actually selects genomes — a RELATIVE, gated combine)
        feasible  = stable >= 0.70 AND err <= 8.0 deg           (the PHYSICAL pair)
        Deb's rules: any feasible beats any infeasible; among infeasible, smaller
        normalised violation wins; among feasible, the base combine decides, computed
        over the FEASIBLE SUBSET ONLY.
        base combine = zrank over columns, z = (x-median)/(1.4826*MAD) clamped to +-3,
        weighted:   reward  w=0.3125  (higher better)   <- NOT err; the reward field
                    stable  w=0.2500  (higher better)
                    steady  w=0.4375  (lower  better)   <- gate distance cannot see this
        · relative: a score only means something against the population it was ranked in.

WHAT THIS SCRIPT DOES. Reads every banked marker, recovers the MEMORY stage's held-out
row per report seed from the .out (means over the 5 seeds, so it matches the marker's
MULTI-SEED line), then ranks the whole archive TWICE: once by gate distance, once by
handing those same rows to the SHIPPED wheel combine (ram_controller.gated_fitness_combine
— never a reimplementation, per CLAUDE.md). Reports Kendall tau, the inversion count, and
the rows the two rules disagree about most.

⚠️ WHAT THIS IS NOT. It does NOT re-rank anything, change any fitness, or touch a run.
The live fitness ranks a GA POPULATION on DURING-SEARCH pool scores; this ranks the
ARCHIVE on HELD-OUT scores. Same rule, different population — so it answers "do the two
rules order the same rows differently", not "what would the GA have done". A zrank is
relative, so every score here shifts if the archive membership changes. Read the
DISAGREEMENTS, not the absolute fitness numbers.
"""
import glob, json, math, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

K = math.log(0.5) / math.log(0.70)

# The live weight vector — the one every ladder/A/B run in flight uses. Read off the
# recipes (scripts/translation_ab_chain.sh, sweep_ladder_gamma.sh): S16noJM minus the
# jerk/mono terms, i.e. --fit-weight-err-sq 0.3125 --fit-weight-stable 0.25
# --fit-weight-steady 0.4375, --fit-aggregation zscore --zrank-clamp 3.0,
# --gate-stable 0.70 --gate-err 8.0.
W_REWARD, W_STABLE, W_STEADY = 0.3125, 0.25, 0.4375
CLAMP, GATE_STABLE, GATE_ERR = 3.0, 0.70, 8.0

RESULT = re.compile(
	r'RESULT — during-search winner \(held-out\):\s+'
	r'stable=([0-9.]+)%\s+err=([0-9.]+)°\s+steady=([0-9.]+)°.*?reward=(-?[0-9.]+)')
HEADER = re.compile(r'HELD-OUT REPORT \[(\w+)[\]#-]')


def gate_distance(stable_pct, err_deg):
	s = max(stable_pct / 100.0, 1e-6)
	return 0.5556 * (err_deg / 8.0) + 0.4444 * min(K * -math.log2(s), 20.0)


def memory_rows(path):
	"""Mean stable/err/steady/reward over the MEMORY stage's report-seed rows.

	Anchored on the stage HEADER, not on line order: with --report-seeds N every stage
	emits N RESULT lines, and taking "the last few" is exactly the mislabelling bug the
	arm library had to fix on 04/08/2026. A bare [MEMORY] header (no #k suffix) is the
	published stage; the [MEMORY#k-VALxxx] blocks are stage-select's val draws and must
	NOT be averaged in — selection never touches the report seeds.
	"""
	try:
		txt = open(path, errors='ignore').read()
	except OSError:
		return None
	rows, stage = [], None
	for line in txt.splitlines():
		h = HEADER.search(line)
		if h:
			stage = h.group(0)
			continue
		m = RESULT.search(line)
		if m and stage == 'HELD-OUT REPORT [MEMORY]':
			rows.append(tuple(float(g) for g in m.groups()))
	if not rows:
		return None
	n = len(rows)
	return dict(stable=sum(r[0] for r in rows) / n, err=sum(r[1] for r in rows) / n,
	            steady=sum(r[2] for r in rows) / n, reward=sum(r[3] for r in rows) / n,
	            seeds=n)


def collect():
	outs = {os.path.basename(p)[:-4]: p
	        for p in glob.glob(os.path.join(ROOT, 'logs/controller/*/*.out'))}
	rows = []
	for mp in glob.glob(os.path.join(ROOT, 'experiments/*_markers/*.json')):
		tag = os.path.basename(mp)[:-5]
		try:
			d = json.load(open(mp))
		except Exception:
			continue
		if 'headline_holdout' not in d:
			continue
		mem = memory_rows(outs.get(tag, ''))
		if not mem:
			continue
		alt = re.search(r'alt=([0-9.]+)m', d.get('headline_holdout', ''))
		mem['tag'] = tag
		# A run with no z axis still prints alt=0.000m — the field's presence is not
		# the test (see gate_distance_leaderboard.py).
		mem['altitude'] = bool(alt) and float(alt.group(1)) > 0.0
		mem['gd'] = gate_distance(mem['stable'], mem['err'])
		rows.append(mem)
	return rows


def fitness_scores(rows):
	"""Hand the rows to the SHIPPED wheel combine. Never reimplement the ranking."""
	from wnn.control._accel import gated_fitness_combine
	n = len(rows)
	flat = ([r['reward'] for r in rows]
	        + [r['stable'] / 100.0 for r in rows]
	        + [r['steady'] for r in rows])
	return list(gated_fitness_combine(
		flat, n, [W_REWARD, W_STABLE, W_STEADY], [True, True, False],
		'zscore', CLAMP,
		[r['stable'] / 100.0 for r in rows], [r['err'] for r in rows],
		GATE_STABLE, GATE_ERR))


def kendall(a, b):
	"""Concordant/discordant over all pairs. Ties broken by neither count."""
	n, con, dis, tie = len(a), 0, 0, 0
	for i in range(n):
		for j in range(i + 1, n):
			da, db = a[i] - a[j], b[i] - b[j]
			if da == 0 or db == 0:
				tie += 1
			elif (da > 0) == (db > 0):
				con += 1
			else:
				dis += 1
	total = con + dis
	return (con - dis) / total if total else float('nan'), con, dis, tie


def report(rows, label):
	if len(rows) < 3:
		print('  too few rows (%d) to rank.' % len(rows))
		return
	fit = fitness_scores(rows)
	for r, f in zip(rows, fit):
		r['fit'] = f
	by_gd = {r['tag']: i for i, r in enumerate(sorted(rows, key=lambda r: r['gd']), 1)}
	by_ft = {r['tag']: i for i, r in enumerate(sorted(rows, key=lambda r: r['fit']), 1)}
	tau, con, dis, tie = kendall([r['gd'] for r in rows], [r['fit'] for r in rows])
	print('  %s — %d runs, MEMORY stage held-out, %d report seeds each'
	      % (label, len(rows), rows[0]['seeds']))
	print('  Kendall tau (gate-distance vs fitness) = %+.3f' % tau)
	print('    concordant pairs %d   DISCORDANT %d   tied %d   -> %.1f%% of pairs invert'
	      % (con, dis, tie, 100.0 * dis / max(con + dis, 1)))
	print()
	print('  Worst disagreements (|rank difference|), gate-dist rank vs fitness rank:')
	print('    gd#  fit#   Δ   gate-dist   fitness   stable     err  steady   tag')
	worst = sorted(rows, key=lambda r: -abs(by_gd[r['tag']] - by_ft[r['tag']]))[:12]
	for r in worst:
		print('    %3d  %4d  %+4d   %9.4f  %8.4f  %5.1f%%  %6.2f  %6.2f   %s'
		      % (by_gd[r['tag']], by_ft[r['tag']], by_ft[r['tag']] - by_gd[r['tag']],
		         r['gd'], r['fit'], r['stable'], r['err'], r['steady'], r['tag'][:44]))
	print()
	print('  Top 10 by each rule, side by side:')
	print('    #   by GATE-DISTANCE                              |  by FITNESS')
	g = sorted(rows, key=lambda r: r['gd'])[:10]
	f = sorted(rows, key=lambda r: r['fit'])[:10]
	for i, (a, b) in enumerate(zip(g, f), 1):
		print('    %2d  %-44s  |  %s' % (i, a['tag'][:44], b['tag'][:44]))


def main():
	rows = collect()
	print(__doc__.rstrip())
	print()
	print('=' * 78)
	print('ALTITUDE REGIMEN')
	print('=' * 78)
	report([r for r in rows if r['altitude']], 'altitude regimen')
	print()
	print('=' * 78)
	print('ATTITUDE-ONLY (a different task — never pooled with the above)')
	print('=' * 78)
	report([r for r in rows if not r['altitude']], 'attitude-only')


if __name__ == '__main__':
	sys.exit(main())
