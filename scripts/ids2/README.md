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
