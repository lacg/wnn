# IDS-2 final readout (26/09/2026) — reproduce the numbers

Read-only against the dashboard DB (`file:/Volumes/.../wnn/db/wnn.db?mode=ro`). Held-out TEST val_cal only,
from `validation_summaries` final rows — never `iterations.best_f1`.

- `ids2_final_readout.py` — protocol readout (ids-security agent). Subcommands `arms`, `seeds`, `pareto`,
  `rule7`, `extra` — e.g. `python ids2_final_readout.py arms cicids quad`. Feeds docs/ids_results.md § "IDS-2 FINAL (26/09/2026)".
- `load.py`, `anal.py`, `anal2.py` — statistical adjudication (experiment-design agent): seed-paired
  contrasts, two-way ANOVA (arm × seed), Bonferroni, best-of-k null simulation, power (noncentral t).
  Run from this directory (`anal*.py` import `load`).

Verdict (both agents, 26/09): NO statistically supported winner on cicids or ciciot. Chosen configs:
cicids CE20 (FPR direction 3/3 seeds, fewer generations), ciciot B15-AC (non-dominated 32/35, FPR −1.10pp
5/5 — provisional, post-hoc). unswr: blocked on IDS-16. Plane: IDS-2.

Era-restricted best-row readout (26/09, Luiz: no SP/SP100 era, best rows WITH their config):
- `era.py` — strict: `_3way`, started >= 30/08/2026 02:27 UTC (OR-fold fix), val_cal only.
- `era_cicids.py` — relaxed CICIDS set (IDSX+IDSXD from 22/08; the fix cannot touch <=34-bit cicids).
- `era_cicids_d.py` — IDSXD (desirability) only.
- `topf1.py` / `top2.py` — the all-era top-F1 extractor/ranker (includes SP/SP100; kept for reference).
- `ids20_idsagg_readout.py` — IDS-20 (IDSAGG) zscore-vs-desirability readout per experiments/ids_aggregation_ab_rule.json; subcommands primary/arms/seeds/b05vb15/rule7/pareto (IDSAGGX = B05-AC extension, descriptive).
- `ids20_best_of_best.py` / `ids20_best_of_best_paired.py` — IDS-20 best-of-best (5 genome types x 7 modes x GS/GA) per dataset, Pareto, best-vs-mean, paired per-seed by arm.
