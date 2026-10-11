"""IDS-25 analysis: does the k-fold search fitness rank grid genomes like held-out TEST does?

Input: experiments/ids_plateau/ids25/flow_<id>.json from ids25_rescore_grid.py.
Per run (50 grid genomes = the grid phase final population): Spearman rho between
  search side (k-fold CV on the 80% train): fitness_score (lower = better), kf_f1, kf_fpr, kf_ce
  held-out TEST: val_cal F1 / FPR (Protocol v2: threshold from VAL), CE (threshold-free)
oriented so rho > 0 always means "search agrees with held-out".
Pooled per dataset: mean per-run rho with a bootstrap CI over runs (B=10000).
Plus selection regret: held-out val_cal F1 of the search-fitness-best genome vs the best
held-out genome in the same population (pp), and its held-out rank.
Usage: python scripts/ids_plateau/ids25_analyze.py [dir]
"""
import glob, json, os, re, sys
import numpy as np
from scipy import stats

RNG = np.random.default_rng(20261010)
DS = re.compile(r"-(cicids|ciciot|unswr|unswt)-")


def rho(a, b):
	a, b = np.asarray(a, float), np.asarray(b, float)
	if np.ptp(a) == 0 or np.ptp(b) == 0:
		return np.nan
	return stats.spearmanr(a, b).statistic


def boot(x, B=10000):
	x = np.asarray([v for v in x if not np.isnan(v)], float)
	if len(x) < 2:
		return np.nan, np.nan, len(x)
	m = RNG.choice(x, (B, len(x))).mean(1)
	return np.percentile(m, 2.5), np.percentile(m, 97.5), len(x)


def run_stats(gs):
	fit = [g["fitness"] for g in gs]
	have_fit = all(f is not None for f in fit)
	hoF1 = [g["ho_modes"]["val_cal"]["f1"] for g in gs]
	hoFPR = [g["ho_modes"]["val_cal"]["fpr"] for g in gs]
	hoCE = [g["ho_ce"] for g in gs]
	nf = [-f for f in fit] if have_fit else None
	r = dict(n=len(gs),
		fit_F1=rho(nf, hoF1) if have_fit else np.nan,
		fit_FPR=rho(nf, [-x for x in hoFPR]) if have_fit else np.nan,
		fit_CE=rho(nf, [-x for x in hoCE]) if have_fit else np.nan,
		kF1_F1=rho([g["kf_f1"] for g in gs], hoF1),
		kFPR_FPR=rho([g["kf_fpr"] for g in gs], hoFPR),
		kCE_CE=rho([g["kf_ce"] for g in gs], hoCE),
		spread_kF1=100 * np.ptp([g["kf_f1"] for g in gs]), spread_hoF1=100 * np.ptp(hoF1))
	if have_fit:
		b = int(np.argmin(fit))
		r["regret_F1"] = 100 * (max(hoF1) - hoF1[b])
		r["regret_FPR"] = 100 * (hoFPR[b] - min(hoFPR))
		r["ho_rank_F1"] = 1 + sum(h > hoF1[b] + 1e-12 for h in hoF1)
	return r


def main():
	d = sys.argv[1] if len(sys.argv) > 1 else "/Users/lacg/wnn/experiments/ids_plateau/ids25"
	runs = {}
	for p in sorted(glob.glob(os.path.join(d, "flow_*.json"))):
		j = json.load(open(p))
		if len(j["genomes"]) < 10:
			continue
		runs[j["flow"]] = (j["name"], DS.search(j["name"]).group(1), run_stats(j["genomes"]), j["wall_s"])
	keys = ["fit_F1", "fit_FPR", "fit_CE", "kF1_F1", "kFPR_FPR", "kCE_CE"]
	print("Per-run Spearman (rho>0 = search agrees with held-out); held-out = TEST val_cal F1/FPR, CE")
	print(f"    {'run':42s} | {'N':>3} | " + " | ".join(f"{k:>8}" for k in keys) + " | kF1 spread | hoF1 spread | regret F1 pp | ho rank | wall s")
	for fid, (name, ds, r, w) in sorted(runs.items(), key=lambda kv: (kv[1][1], kv[1][0])):
		print(f"    {name:42s} | {r['n']:3d} | " + " | ".join(f"{r[k]:+8.3f}" for k in keys)
			+ f" | {r['spread_kF1']:10.3f} | {r['spread_hoF1']:11.3f} | {r.get('regret_F1', np.nan):12.3f} | {r.get('ho_rank_F1', 0):7d} | {w:6.0f}")
	print("\nPooled per dataset: mean per-run rho [bootstrap 95% CI over runs] (N runs)")
	for ds in ["unswr", "unswt", "cicids", "ciciot"]:
		rs = [r for (_, d_, r, _) in runs.values() if d_ == ds]
		if not rs:
			continue
		print(f"### {ds} (runs={len(rs)}, genomes/run={int(np.mean([r['n'] for r in rs]))})")
		for k in keys:
			v = [r[k] for r in rs]
			lo, hi, n = boot(v)
			print(f"    {k:9s} {np.nanmean(v):+.3f} [{lo:+.3f},{hi:+.3f}] (N={n})  min {np.nanmin(v):+.3f} max {np.nanmax(v):+.3f}")
		for k in ["regret_F1", "regret_FPR", "spread_hoF1", "spread_kF1"]:
			v = [r[k] for r in rs if k in r]
			print(f"    {k:11s} {np.mean(v):.3f}±{np.std(v, ddof=1) if len(v) > 1 else 0:.3f} pp (N={len(v)})")
		rk = [r["ho_rank_F1"] for r in rs if "ho_rank_F1" in r]
		print(f"    held-out F1 rank of the fitness-best genome: median {np.median(rk):.0f} of ~{int(np.mean([r['n'] for r in rs]))} (ranks {sorted(rk)})")
	print(f"\ncompute: {len(runs)} flows, {sum(w for (*_, w) in runs.values()) / 3600:.2f} h re-score wall")


if __name__ == "__main__":
	main()
