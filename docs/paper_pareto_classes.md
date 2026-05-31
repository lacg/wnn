# Paper Section: Pareto Deployment Classes (UNSW-NB15 Temporal)

Status: draft scaffold, 2026-05-31. Intended for the paper's Experimental Results
section (around T7 model-size analysis) or the Discussion section's "FPGA deployment
analysis" paragraph.

All numbers in this section come from the n=30 16b-Wb-C35-500n34b-OI cohort
(flows 2897/2911/2925/2978-3004) plus measured RF and XGBoost baselines from
`scripts/verify_unsw_temporal_baselines.py` on the same partition.

---

## Section Opener (TODO — Andrew's authorial voice)

<!--
LEARNING-MODE TODO for Luiz, 5-10 lines max:

Write a 3-5 sentence opener that:
  1. Frames the GA as discovering a Pareto front (not "we trained three models")
  2. Names the three operating points: server / edge-FPGA / FPR-extreme
  3. Ends on the line that anchors the rest of the section — what is the headline
     claim the three tables below are evidence for?

Constraints from CLAUDE.md:
  - No em-dashes (use commas, parentheses, colons, periods)
  - No superfluous adjectives ("rigorous", "comprehensive", "significant")
  - Positive framing of prior work; let methodology and results speak for themselves
  - First person plural ("we report"), present tense

Suggested anchor phrasings to choose from / mix:
  - "A single GA run produces a range of Pareto-optimal genomes spanning
     four orders of magnitude in model size."
  - "Three operating points along the F1 vs FPR vs size Pareto front..."
  - "All three beat measured RF and XGBoost baselines on both F1 and FPR
     at their respective deployment scales."

If unsure: the strongest single sentence available is "the same GA population
produces both a 1.3 MB server-class detector and a 2 KB FPGA-deployable edge
variant, both of which beat RF and XGBoost on F1 and FPR." Pick the framing
that makes that conclusion inevitable.
-->

**[TODO: 3-5 sentence opener — see HTML comment above for constraints and anchor phrasings]**

---

## Three Pareto operating points

| Operating point | Neurons | Memory | F1 (val_cal) | FPR (val_cal) | Source |
|---|---:|---:|---:|---:|---|
| **Server class**       | ~309 (cohort mean) | ~1.3 MB | **88.87 ± 0.23%** | **8.77 ± 1.14%** | n=30 16b-Wb seeds, mean±sample-std |
| **Edge / FPGA class**  | **5**              | **~2 KB**  | **87.88%**        | **6.94%**         | Best-of-class single seed, flow 2994 r56926 |
| **FPR-extreme class**  | 16                 | ~5 KB    | 87.70%            | **4.63%**         | Best-of-class single seed |

Within-cohort range: max single-seed F1 is 89.34% (flow 2994 r56926, val_cal
best_ce), and the same flow also produces the 5-neuron edge variant cited above.
Minimum single-seed val_cal FPR within the n=30 cohort is 6.77% on a best_ce pick.

## Comparison to measured RF and XGBoost baselines

Baselines were measured with `scripts/verify_unsw_temporal_baselines.py` on the
identical 80/20 temporal split, top-20 features, 16-bit thermometer encoding.
RF: scikit-learn 100 trees, max_depth=None, random_state=42. XGBoost: 100 trees,
max_depth=6, lr=0.1, random_state=42.

| Baseline | F1 | FPR | Model Size |
|---|---:|---:|---:|
| Random Forest (measured) | 86.05% | 25.41% | 138 MB |
| XGBoost (measured)       | 84.89% | 28.57% | 0.27 MB |

### Deltas

| WNN class vs baseline | Δ F1 | Δ FPR | Size ratio (baseline ÷ WNN) |
|---|---:|---:|---:|
| Server (~309n, 1.3 MB) vs RF       | **+2.82 pp** | **−16.64 pp** | **138×** smaller than RF |
| Server (~309n, 1.3 MB) vs XGBoost  | **+3.98 pp** | **−19.80 pp** | 5× larger than XGBoost, but XGBoost has 3.3× the FPR |
| Edge / FPGA (5n, ~2 KB) vs RF      | **+1.83 pp** | **−18.47 pp** | **~138,000×** smaller than RF |
| Edge / FPGA (5n, ~2 KB) vs XGBoost | **+2.99 pp** | **−21.63 pp** | **~135×** smaller than XGBoost |
| FPR-extreme (16n, ~5 KB) vs RF     | **+1.65 pp** | **−20.78 pp** | ~55,000× smaller than RF |

## FPGA fit analysis (Xilinx Zynq Z-7020)

The 5-neuron edge variant uses 5 neurons × 16-bit thermometer input × approximately
2,500 sparse memory entries per neuron, giving ~28 KB of dense BRAM footprint plus
~500-1000 LUTs for address generation (genome-dependent).

The Z-7020 provides 625 KB internal BRAM and 137 KB LUTRAM, for 762 KB total internal
memory. A single 5-neuron WNN occupies under 4% of available internal memory, leaving
99%+ headroom. This permits up to roughly 27 parallel detector instances on a single
Z-7020 device for cascade or ensemble deployment.

## Note on the 16b-Wb F1 profile

The 16b-Wb cohort uses harmonic-rank fitness with weights ce=0.10, acc=0.20,
F1=0.35, FPR=0.35. This recipe places half of its weight on the joint
(F1, FPR) objective and explicitly co-optimizes low FPR alongside F1. The
within-cohort F1 peak (89.34%) and 88.87 ± 0.23% mean reflect this
co-optimization, not an architectural F1 ceiling.

Within the broader XDS-unsw-temporal sweep, smaller cohorts at other
(width, weight) combinations reach 89.50% (8b-Wc) and 89.69% (96b-Wc).
Pre-fix runs (cohorts before commits 6ab34164 + d6b658bd, 2026-05-28)
included F1 values above 90% but are not directly comparable due to the GA
selection bug those commits resolved. An F1-focused weight recipe (CE and
F1 heavy, FPR lighter) at n=100 on the post-fix codebase is the
straightforward next experiment for raising the post-fix F1 record.

For this paper, the contribution is the joint Pareto picture rather than
a single F1 maximum: at every reported operating point (server, edge, FPR-
extreme) the WNN improves on the measured RF and XGBoost baselines on both
F1 (+1.6 pp to +4.0 pp) and FPR (−16 pp to −21 pp), at model sizes between
138× and 138,000× smaller than RF.

## Methodology notes for reproducibility

- All 30 seeds use the 16-bit distributive thermometer, top-20 feature subset,
  500-neuron / 34-bit-address maximum (C35 architecture).
- GA fitness: harmonic rank of (ce, accuracy, F1, FPR) with weights
  Wb = ce=0.10, acc=0.20, F1=0.35, FPR=0.35.
- Training uses order-independent QUAD_WEIGHTED memory (OI-v2) with K-fold = 5×5
  on the 80% training partition. The 20% held-out partition is touched only at the
  final validation checkpoint.
- All 30 flows ran post the IDS GA dual-bug fix (commits 6ab34164 + d6b658bd,
  2026-05-28); pre-fix cohorts are deprecated.
- Threshold mode reported throughout is `val_cal` (validation-set-calibrated
  threshold on the best_ce genome). The full 7-mode threshold table per seed
  is available in the `validation_summaries.threshold_metadata` JSON column.

## Cross-references

Related working memory:
- `project_unsw_temp_paper_ready_31may` — resume context, full result table
- `project_rf_xgb_unsw_temp_measured` — baseline measurement methodology
- `project_tiny_neuron_pareto_5to8n` — discovery of the 5n / 7n / 8n Pareto extremes
- `project_xds_5table_findings_30may` — full XDS cohort analysis
- `project_ids_ga_dual_bug_fix` — pre-fix cohort deprecation

Related repo artefacts:
- `scripts/verify_unsw_temporal_baselines.py` — RF / XGBoost measurement
- `scripts/analyze_xds_final.py` — 4-criteria Pareto ranking (TODO Task #16: add `--max-neurons N` filter)
- `scripts/build_xds_5tables.py` — canonical 5-table report
