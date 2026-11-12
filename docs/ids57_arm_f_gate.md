# IDS-57 arm F: offline VAL gate (per-attack-category clusters, capacity only on the hard category)

Written 12/11/2026 (ids-security agent) for Luiz. Arm F is Luiz's idea, approved to gate on 12/11/2026.

**Status: gate run. Verdict KILLED. Nothing was queued, paused, cancelled, resumed or reordered. No wheel was
swapped. No worker code changed (the only files added are this doc, the probe scripts and their outputs). The worker (ABI 28) ran
IDS56U flow 7543 throughout; 7544-7548 were still queued after the last probe.**

Vocabulary (`docs/terminology_train_val_test.md`): **train** fits, **VAL** tunes, **TEST** reports.

- Every number below is a **VAL gate read**: threshold fitted on VAL and scored on VAL, on random unsearched
  genomes. It is a tuning read. It is shown as a difference against the 34-bit baseline and a verdict, never as a
  result. **No TEST row was scored** (the probe rebinds the dataset's test slot to VAL before any evaluator is built).
- unswr = UNSW-NB15 `random_3way`, the params of control flow 7501 (BINARY, lattice, rate 0.25, order-independent
  train, `balance_classes=true`). Train 1,267,407 rows; VAL 152,391 benign / 6,035 attack (1,331 Fuzzers).
- 5 genome draws per variant (seeds 1000-1004, the A1 gate's draws). Differences are paired by draw.
- Probes ran at `nice -n 10`, one process, 4 threads, 3-5 s per genome.

## 1. Verdict

**KILLED.** Arm F as proposed (benign cluster at 34 bits, attack side split by category, extra bits only on
Fuzzers) loses to the plain 34-bit baseline on every variant and every draw.

| Read (VAL, mean of 5 draws) | dF1 vs 34-bit baseline | Wins | Fuzzers recall at FPR 0.3 % | Cells |
|---|---|---|---|---|
| A1, blanket 64 bits | **+1.19 pp** | 5/5 | +8.0 pts | 2.15M |
| Arm F, Fuzzers at 64 bits + "other attacks" | **-0.48 pp** | 0/5 | **-14.0 pts** | 0.76M |
| Arm F, top-3 separate + other | -0.43 to -0.50 pp | 0/5 | -14.3 to -16.6 pts | 0.64-0.69M |
| Arm F, all 9 separate | -0.51 to -0.56 pp | 0/5 | -12.5 to -13.6 pts | 0.63-0.64M |

Three measured facts explain it:

1. **The bits pay on the benign cluster, not on the attack cluster.** 64 bits on the benign side alone: +0.49 pp
   (5/5). 64 bits on the attack side alone: -0.63 pp (0/5). Both: +1.19 pp.
2. **The cells are in the benign cluster too.** Of the 1.48M cells A1 adds, 1.34M (91 %) are benign-side. The
   attack side at 64 bits costs 0.13M. So "capacity only on the hard category" has almost nothing to save.
3. **The category split itself adds nothing.** Fuzzers split out at 34 bits: +0.04 pp. The same split at A1's
   widths: +1.11 pp against A1's +1.19. More clusters is slightly worse (all 9 at 64 bits: +0.77).

The hypothesis is refuted at the gate: a dedicated wide Fuzzers cluster makes Fuzzers recall **worse** at every
false-alarm budget tested (section 5).

## 2. Feasibility with today's code (measured)

| Question | Answer | Evidence |
|---|---|---|
| Train a K-cluster memory with per-cluster (neurons, bits) on per-category labels, real path? | **Yes, today (ABI 28).** | 10 clusters, Fuzzers at 64 bits and the rest at 34, trained through `IdsEvaluatorFactory` -> `IDSEvaluator(classification="multi")` -> `IDSCacheWrapper` |
| Merged categories ("Fuzzers + other attacks")? | **Yes, probe-level.** The label array is remapped before the evaluator is built. No flow param does this. | `armf_common.relabel` |
| Per-row score of every cluster read back? | **Yes, today, through the streamer.** `IDSGenomeStreamerWrapper.take_scores()` returns the flat rows x K buffer. The in-memory cache path does not (`score_examples` and `evaluate_multiclass_at_thresholds` return one score or a confusion matrix). ABI 29 adds the export to the cache path; the gate did not need it. | below |
| Is the streamer the same model as the cache path? | **Yes, bit for bit on the cases checked.** | below |
| Binary decision over K clusters (best attack cluster vs benign, margin calibrated on VAL)? | The **rule** exists in Rust (`benign_margins` + `margin_decode`, mode `margin_val_cal`). The **threshold objective** does not: for K > 2 the sweep maximises K-class macro-F1. The gate got the binary objective from the same Rust code by handing it a 2-column matrix [benign, best attack]. | `armf_common.binary_read` |
| Does the hierarchical path expose it? | No. It is a cascade: a 2-cluster gate decides benign vs attack, the second stage only names the attack. The category clusters never vote on the binary decision. | `IdsEvaluatorFactory._hierarchical` |
| Does `ids_classification=multi` expose it? | Partly. Same max-over-attack margin, but threshold, search fitness and reported F1 are all 10-class. Always 9 attack clusters; no merging; no explicit per-cluster caps. | `multiclass_metrics::modes_from_scores`, `thresholds::search_f1_fpr` |

**Fidelity checks (VAL, `armf_fidelity.py`):**

| Check | Result |
|---|---|
| 2-cluster 250 x 34, streamer scores -> Rust modes vs `IDSCacheWrapper.evaluate_multiclass_at_thresholds` | 10/10 modes: confusion matrix and threshold identical; cells identical (667,807) |
| That genome vs the A1 gate's baseline (probe2, same draw) | identical: 94.0204 / 0.9108 |
| 10-cluster, mixed widths (Fuzzers 64, rest 34), streamer vs cache | 7/7 comparable modes identical; cells identical (605,929). `margin_train_cal` is not comparable (the probe does not score the train rows) |
| Whole gate run twice | second run reproduces the first on every row |

**What Python does in the gate, stated plainly.** Training, scoring, the threshold sweep and the metrics are Rust.
Python reduces the attack columns to one (max, sum, or z-normalised max) and counts rows per category. That
reduction is the decision rule under test; it is not a copy of existing accelerator logic.

**What a flow would need (not built, section 9).** A label-mapping param (Python). A binary objective for the
K-cluster search fitness and for the reported modes (Rust, worker wheel). Explicit per-cluster caps (Python at
least).

## 3. Variants

Cluster 0 is always benign. "by support" = the 250 attack neurons split by train support with a floor of 10
(`tier_sizing.allocate_neurons`, the August rule). The benign cluster at 250 x 34 is the same neurons in every
variant that keeps it.

| Variant | Benign | Attack side |
|---|---|---|
| (a) baseline | 250 x 34 | one cluster, 250 x 34 |
| (b) A1 | 250 x 64 | one cluster, 250 x 64 |
| benign-only 64 | 250 x 64 | one cluster, 250 x 34 |
| attack-only 64 | 250 x 34 | one cluster, 250 x 64 |
| F: Fuzzers 250 | 250 x 34 | Fuzzers 250 x 64; other eight 250 x 34 |
| F: Fuzzers 125 | 250 x 34 | Fuzzers 125 x 64; other eight 125 x 34 |
| F: split only | 250 x 34 | Fuzzers 250 x 34; other eight 250 x 34 |
| F: top-3, by support | 250 x 34 | Fuzzers 55 x 64; Exploits 70; Reconnaissance 34; other six 91 (all x 34) |
| F: top-3, 125 each | 250 x 34 | Fuzzers 125 x 64; Exploits, Reconnaissance, other six 125 x 34 |
| F: 5 majors + rare merged, by support | 250 x 34 | Fuzzers 55 x 64; Exploits 70; Recon 34; DoS 15; Generic 60; rare four 15 |
| F: 5 majors + rare merged, 60 each | 250 x 34 | six clusters of 60; Fuzzers x 64 |
| F: all 9, by support | 250 x 34 | Fuzzers 49 x 64; Exploits 63; Recon 31; DoS 14; Generic 54; four rare 10 each |
| F: all 9, 60 each | 250 x 34 | nine clusters of 60; Fuzzers x 64 |
| F: all 9, tiered bits | 250 x 34 | as "by support", Generic x 24, rare x 20 |

## 4. Gate table (VAL, differences vs (a), paired by draw)

Baseline (a) on this read: F1 94.06, FPR 0.90 (every unsearched 34-bit draw calibrates to the high-recall point).
Decision rule: attack if (best attack cluster - benign) >= tau, tau fitted on VAL for 2-class macro-F1.

| Variant | dF1 mean [min..max] | SD | Wins | FPR at its tau | Cells | Added cells | dF1 per added 1M cells |
|---|---|---|---|---|---|---|---|
| (a) baseline | 0 | - | - | 0.90 | 0.67M | 0 | - |
| **(b) A1** | **+1.19 [+1.05..+1.25]** | 0.08 | 5/5 | 0.30 | 2.15M | +1.48M | **+0.80** |
| benign-only 64 | +0.49 [+0.42..+0.52] | 0.04 | 5/5 | 0.56 | 2.02M | +1.34M | +0.37 |
| attack-only 64 | -0.63 [-0.70..-0.55] | 0.07 | 0/5 | 1.09 | 0.81M | +0.13M | negative |
| F: Fuzzers 250 | -0.48 [-0.56..-0.39] | 0.08 | 0/5 | 0.80 | 0.76M | +0.08M | negative |
| F: Fuzzers 125 | -0.50 [-0.54..-0.45] | 0.04 | 0/5 | 0.71 | 0.67M | 0 | negative |
| F: split only (34 bits) | +0.04 [-0.05..+0.09] | 0.06 | 4/5 | 0.82 | 0.70M | +0.02M | inside draw noise |
| F: top-3, by support | -0.43 [-0.54..-0.25] | 0.13 | 0/5 | 0.67 | 0.64M | -0.03M | negative |
| F: top-3, 125 each | -0.50 [-0.56..-0.43] | 0.05 | 0/5 | 0.77 | 0.69M | +0.02M | negative |
| F: 5 majors + rare merged, by support | -0.50 [-0.59..-0.44] | 0.06 | 0/5 | 0.75 | 0.63M | -0.04M | negative |
| F: 5 majors + rare merged, 60 each | -0.50 [-0.54..-0.44] | 0.04 | 0/5 | 0.79 | 0.64M | -0.03M | negative |
| F: all 9, by support | -0.56 [-0.62..-0.49] | 0.06 | 0/5 | 0.91 | 0.63M | -0.05M | negative |
| F: all 9, 60 each | -0.51 [-0.55..-0.44] | 0.04 | 0/5 | 0.84 | 0.64M | -0.03M | negative |
| F: all 9, tiered bits | -0.56 [-0.65..-0.46] | 0.08 | 0/5 | 0.89 | 0.62M | -0.05M | negative |

- The A1 gate reproduces: +1.19 here, +1.18 in `docs/ids57_options.md` (same draws, different read path).
- Every arm F variant with a 64-bit Fuzzers cluster is between -0.43 and -0.56, 0/5. The spread across variants
  (0.13) is about the size of the draw SD. No sizing, grouping or bits tiering moves it.
- F1 gained per added cell vs A1: A1 buys +0.80 pp per 1M cells. Arm F buys a loss, so there is no ratio to quote.
- Cells are logical cells of unsearched genomes (physical = logical on every row here).

**Decomposition of A1's gain (2-cluster, same draws):**

| Benign bits | Attack bits | dF1 | Added cells |
|---|---|---|---|
| 34 | 34 | 0 | 0 |
| 64 | 34 | +0.49 | +1.34M |
| 34 | 64 | -0.63 | +0.13M |
| 64 | 64 | +1.19 | +1.48M |

The two sides are not additive (+0.49 and -0.63 would sum to -0.14; together they give +1.19). The wide attack
cluster only helps when the benign cluster is wide too.

## 5. Fuzzers recall at equal false-alarm budgets (VAL)

Recall at each variant's own tau is not comparable (the baseline sits at FPR 0.9, A1 at 0.3). The table reads the
same margin score at exact benign FPR budgets, interpolating between adjacent thresholds. Fuzzers recall %, mean of
5 draws, difference vs (a) in brackets, draws that beat (a).

| Variant | FPR 0.2 % | FPR 0.3 % | FPR 0.5 % | FPR 0.9 % |
|---|---|---|---|---|
| (a) baseline | 46.2 | 52.9 | 65.6 | 90.8 |
| (b) A1 | 52.0 (+5.8, 5/5) | 60.9 (+8.0, 5/5) | 72.6 (+7.0, 5/5) | 92.7 (+1.8, 5/5) |
| benign-only 64 | 47.1 (+0.9, 3/5) | 55.4 (+2.6, 5/5) | 68.8 (+3.3, 5/5) | 90.0 (-0.9, 0/5) |
| attack-only 64 | 33.3 (-12.9, 0/5) | 40.5 (-12.4, 0/5) | 54.6 (-10.9, 0/5) | 82.7 (-8.1, 0/5) |
| F: Fuzzers 250 | 32.7 (-13.5, 0/5) | 38.9 (-14.0, 0/5) | 51.2 (-14.4, 0/5) | 81.4 (-9.4, 0/5) |
| F: split only (34 bits) | 45.1 (-1.1, 1/5) | 51.7 (-1.1, 0/5) | 64.9 (-0.6, 1/5) | 90.2 (-0.7, 1/5) |
| F: top-3, by support | 31.9 (-14.3, 0/5) | 36.3 (-16.6, 0/5) | 49.5 (-16.1, 0/5) | 81.6 (-9.2, 0/5) |
| F: all 9, by support | 34.3 (-12.0, 0/5) | 40.4 (-12.5, 0/5) | 52.6 (-13.0, 0/5) | 81.3 (-9.6, 0/5) |
| F: all 9, tiered bits | 34.7 (-11.5, 0/5) | 40.8 (-12.1, 0/5) | 52.9 (-12.7, 0/5) | 80.4 (-10.4, 0/5) |

- A 64-bit Fuzzers cluster against a 34-bit benign cluster loses 8-17 points of Fuzzers recall at every budget.
- The baseline's 46 % at FPR 0.2 on VAL matches the production GA genomes' 45 % at FPR 0.2 on TEST (options doc
  section 2b; different partition, already reported, indicative only).
- All-attack TPR moves the same way: the arm F variants with a 64-bit Fuzzers cluster are -1.0 to -2.8 points at
  every budget; A1 is +4.3 to +5.5 at FPR 0.2-0.5.

## 6. The garbage-attractor check: which attack cluster wins the benign rows

Benign VAL rows flagged at each variant's tau, by the attack cluster with the top score (mean of 5 draws).

| Variant | Benign rows flagged | Fuzzers cluster | Exploits | Rare clusters | Worms |
|---|---|---|---|---|---|
| F: Fuzzers 250 x 64 + other | 1,225 | 721 (59 %) | - | - | - |
| F: split only, Fuzzers 250 x 34 | 1,246 | 996 (80 %) | - | - | - |
| F: top-3, by support | 1,014 | 444 (44 %) | 379 (37 %) | - | - |
| F: 5 majors + rare merged, by support | 1,146 | 422 (37 %) | 389 (34 %) | merged four: 210 (18 %) | - |
| F: 5 majors + rare merged, 60 each | 1,210 | 842 (70 %) | 310 (26 %) | merged four: 40 (3 %) | - |
| F: all 9, by support | 1,392 | 938 (67 %) | 326 (23 %) | Analysis 47, Shellcode 23, Backdoor 5 | **0** |
| F: all 9, 60 each | 1,279 | 897 (70 %) | 325 (25 %) | Analysis 32, Shellcode 4, Backdoor 0 | **0** |
| F: all 9, tiered bits (rare x 20) | 1,360 | 871 (64 %) | 281 (21 %) | Analysis 77, Shellcode 35, Backdoor 17 | **0** |

- **The August attractor does not appear.** Worms (85 train rows) wins zero benign rows in every draw. August was
  QUAD on unswt, where an untrained cell scores a weak positive; this arm is BINARY, where an untrained cell is 0.
  The reason is inferred from the mode semantics, not measured per cell.
- **The attractor here is Fuzzers.** In the by-support and 60-each variants the Fuzzers cluster takes 37-70 % of
  the flagged benign rows with 11-22 % of the attack neurons. Fuzzers is the category that looks like benign
  traffic; giving it its own cluster gives benign-looking rows a dedicated place to win.
- Rare merged vs separate makes no difference to F1 (-0.50 vs -0.56 by support; -0.50 vs -0.51 at 60 each).
- Small rare clusters do leak a little: the merged-four cluster at 15 neurons takes 18 % of the benign rows, at 60
  neurons 3 %. Narrower rare clusters (20 bits) leak more (Analysis 47 -> 77).
- Balance weights reach 14,342 for Worms with nine separate clusters. They are inert under the BINARY lattice
  rule on this read (stated from the rule's semantics; not tested by switching them off).

## 7. Decision-rule variants (one line each, VAL, dF1 vs the baseline under the production rule)

| Rule | Result | Verdict |
|---|---|---|
| max over attack clusters - benign, tau on VAL | the table in section 4 | the best rule on every split variant |
| sum over attack clusters - benign | -0.35 to -0.76 on split variants at 34-bit benign; -0.28 with Fuzzers + other at 64 (max: +1.11) | **worse than max**, 0/5 everywhere |
| neuron-weighted pooled vote | -0.32 to -1.01; -0.35 with Fuzzers + other at 64 | **worse than max**, 0/5 everywhere |
| per-cluster z-normalisation (mu, sigma per score column on VAL), then max | -0.79 to -1.08 on arm F variants; -1.18 on the plain 2-cluster baseline; -1.12 on A1 | **killed**, 0/5 on all 22 variants |
| benign-vs-best margin calibrated on VAL | this is the rule used throughout | - |

- z-normalisation is harmful even with two clusters, so it is not a split-specific problem: the benign column has
  the smaller spread and gets inflated.
- **The sigma-floor edge case from August did not fire.** The smallest score-column sigma on VAL was 0.044
  (Worms, all 9 at 64 bits) against the 1e-6 floor. No column is near-constant on unswr.
- Not run: a fitted readout over the K cluster scores (per-cluster offsets and scales, or a logistic stack). That is
  options-doc arm E in a smaller form. It is gateable today for cluster-level scores (the streamer export); it was
  outside this arm.

## 8. Follow-ups run after the first read (not arm F as proposed)

Run to answer two questions the result raised. Same draws, same read.

**Does the category structure add anything on top of A1's widths?**

| Variant | dF1 [min..max] | Wins | FPR | Cells | dF1 per added 1M cells (A1 = 0.80) |
|---|---|---|---|---|---|
| A1: 250 x 64, 250 x 64 | +1.19 [+1.05..+1.25] | 5/5 | 0.30 | 2.15M | 0.80 |
| benign 64; Fuzzers 250 x 64; other 250 x 64 | +1.11 [+0.97..+1.16] | 5/5 | 0.24 | 2.19M | 0.73 |
| benign 64; Fuzzers 125 x 64; other 125 x 64 | +0.98 [+0.76..+1.19] | 5/5 | 0.26 | 2.06M | 0.71 |
| benign 64; Fuzzers 250 x 64; other 250 x 34 | +0.99 [+0.90..+1.06] | 5/5 | 0.36 | 2.10M | 0.69 |
| benign 64; all 9 by support x 64 | +0.77 [+0.65..+0.88] | 5/5 | 0.31 | 1.99M | 0.58 |

No. Every split is at or below A1, and it falls as the number of clusters grows. Fuzzers recall at FPR 0.3 is
+7.0 for the first row against A1's +8.0.

**Can A1's gain be had at fewer cells? (2 clusters, sizes only)**

| Variant | dF1 [min..max] | SD | Wins | Draws on the low-FPR point | Cells | dF1 per added 1M cells |
|---|---|---|---|---|---|---|
| A1: 250 x 64, 250 x 64 | +1.19 [+1.05..+1.25] | 0.08 | 5/5 | 5/5 | 2.15M | 0.80 |
| benign 250 x 64, attack 125 x 64 | +1.08 [+0.96..+1.23] | 0.12 | 5/5 | 5/5 | 2.04M | 0.79 |
| **benign 125 x 64, attack 250 x 64** | **+0.79 [+0.42..+1.04]** | 0.26 | 5/5 | 4/5 | **1.20M** | **1.50** |
| benign 250 x 48, attack 250 x 48 | +0.68 [+0.19..+1.16] | 0.39 | 5/5 | 3/5 | 1.29M | 1.10 |
| benign 250 x 48, attack 250 x 64 | +0.32 [+0.08..+0.74] | 0.27 | 5/5 | 3/5 | 1.36M | 0.47 |

- Halving the **benign** cluster at 64 bits keeps two thirds of A1's gate gain for 36 % of the added cells
  (1.9x A1's gain per cell). It is less stable: one draw in five falls back to the high-recall point, and the SD
  triples.
- Halving the attack cluster saves almost nothing (0.11M cells) and costs almost nothing.
- Mixed widths are bad again: 48-bit benign against 64-bit attack is worse than 48 against 48.
- This is the only size-efficient direction the gate found. It is an A1 companion, not arm F, and it is not
  proposed here (section 11).

## 9. For the record: what an arm F flow would have needed

Not proposed. Listed so the cost is known if the idea returns in another form.

| Piece | Exists today | Would need |
|---|---|---|
| K clusters, per-cluster neurons and bits in the genome | Yes (`ClusterGenome`, per-neuron bits) | - |
| Per-category training labels | Yes (`ids_classification=multi`, all 9 categories) | - |
| Merged categories (cluster-label mapping) | No | New param (e.g. `ids_attack_clusters`) + a relabel in `IdsEvaluatorFactory`. Python only. Register in `KNOWN_PARAMS` |
| Binary decision as the **search fitness** | No: `adaptive/thresholds.rs::search_f1_fpr` scores K-class macro-F1 on the argmax | Rust: a binary-collapse switch in `EvalSettings` (`ram_core`, so both wheels rebuild) + worker wheel swap + ABI bump |
| Binary decision in the **reported modes** | The max-margin rule yes; the tau objective is K-class macro-F1 | Rust: `multiclass_metrics::modes_from_scores` with a 2-class objective (`find_optimal_margin_tau_by` already takes the objective). Worker wheel |
| Per-cluster caps for the GA | `ArchitectureConfig.neuron_bounds_per_cluster` / `bits_bounds_per_cluster` exist, fed only by `ids_tier_sizing` (support centres x 0.5-1.5 inside one global band `ids_tier_bits_min/max`) | New param for explicit per-cluster bounds (Python). Whether the Rust offspring search enforces per-cluster bounds was not traced: no such argument was found in `IDSEvaluator.search_offspring` or `ids_cache_api.rs` |
| Per-row all-cluster scores | Yes, streamer (`take_scores`); cache path in ABI 29 | - |

Existing `KNOWN_PARAMS` keys an arm F flow would have reused: `ids_classification`, `ids_tier_sizing`,
`ids_tier_neuron_cap`, `ids_tier_bits_min`, `ids_tier_bits_max`, `max_bits`, `max_neurons`, `max_neurons_total`.

**No pre-registration is drafted.** The task asked for one if the gate passed. It did not.

## 10. Why it fails (measured vs inferred)

Measured:
- Bits help on the benign side and hurt on the attack side when the other side stays at 34 (section 4).
- A wide Fuzzers cluster against a 34-bit benign cluster lowers Fuzzers recall at equal FPR (section 5).
- The Fuzzers cluster collects most of the flagged benign rows (section 6).
- The split alone, at equal widths, changes nothing (+0.04 at 34 bits; -0.08 against A1 at 64 bits).

Inferred (not measured per address):
- A 34-bit benign neuron has seen 1.2M benign rows and fires on most benign-looking addresses, Fuzzers rows
  included. A 64-bit attack neuron fires only on addresses it has seen, so on an unseen Fuzzers row the wide
  attack cluster votes less, not more, while the benign vote is unchanged. The margin shrinks on exactly the hard
  rows. With both sides wide, the benign vote on those rows falls too, and the margin opens.
- This is the same reading as the options doc's mechanism (benign address collisions), seen from the other side:
  the fix has to be applied where the collisions are, which is the benign cluster.

## 11. Caveats

- **Random genomes.** No search, no co-adaptation. A GA could size and wire category clusters in ways the draws do
  not show. Against that: the result is 0/5 on eleven arm F variants with a draw SD of 0.04-0.13 against an effect
  of -0.4 to -0.6, and the decomposition in section 4 says the sign comes from where the bits are, not from sizing.
- **Fit on VAL, scored on VAL.** Optimistic in level for every variant alike; the differences are what is read.
  With 158K rows and one fitted threshold the optimism is small. The z-normalised rule fits 2K extra numbers and
  still loses.
- **Training labels.** Each category cluster treats every other row as a negative (the existing K-cluster
  trainer). A variant that drops other attacks from a category cluster's negatives was not run; under the BINARY
  lattice rule negatives do not clear a cell that a positive set, so it is not expected to matter (stated, not
  tested).
- **One write rule.** Lattice only. The IDS56U unweighted vote was not crossed with the split.
- **Full-train fit, no CV.** The flow's search uses 5-fold CV on train; the gate trains once on the full train
  partition.
- **Cells are for unsearched genomes.** GA genomes differ (the 34-bit GA genomes hold 0.63-0.71M cells against
  0.67M here).
- **Selection on TEST across the line.** IDS-55, IDS-56, IDS56U and A1 each make (or will make) a keep-or-drop
  decision on the same unswr TEST partition. A killed gate adds no TEST read, so arm F does not add to that count.
  Any arm that does fly from section 8 would be the fifth such decision and needs the A1 pre-registration's VAL
  concordance step.
- **Direction informed by TEST.** The idea came from the per-category TEST diagnosis of already-reported genomes
  (options doc section 2b). The gate itself used train and VAL only.

## 12. What Luiz must decide

1. **Accept the kill.** Recommended. No flow, no code, no wheel work for arm F.
2. **The size question.** A1's pre-registration already says the b64 row cannot be quoted at the 34-bit size. The
   gate now says where the size is (91 % benign-side) and that a half-size benign cluster keeps about two thirds
   of the gate gain at 1.9x the gain per cell, with a wider spread. Options:
   - (a) do nothing until A1 reads out, then look at the neuron counts and cells the A1 GA genomes actually chose
     (the GA Neurons phase can already shrink the benign cluster) - recommended;
   - (b) ask `experiment-design` to pre-register a size-normalised companion (64 bits with a smaller benign
     cluster) to queue behind A1. Today's params cap both clusters with one `max_neurons`; `max_neurons_total`
     shrinks both in proportion. An asymmetric cap needs the per-cluster bound param of section 9 (Python).
3. **A fitted readout over cluster scores** (arm E in small) is now gateable offline without a wheel. Run that
   gate, or leave it parked until A1 reads out (the options doc's standing recommendation).
4. **Correction to carry into `docs/ids57_options.md`:** arm E's "not gateable today" is true for per-neuron
   outputs only. Per-row per-cluster scores are available on ABI 28 through the streamer.

## 13. Not verified

- The TEST effect of anything here. Nothing was scored on TEST.
- Whether the Rust GA offspring search honours per-cluster bounds (section 9).
- The per-address mechanism in section 10.
- Arm F under the vote write rule, under QUAD, or on unswt / cicids / ciciot.
- Streamer-vs-cache identity beyond the two genomes checked (one 2-cluster, one 10-cluster mixed-width). The gate's
  2-cluster rows also match the A1 gate's cache-path numbers at 34 bits (exactly, draw 0) and at 64 bits (mean
  +1.19 vs +1.18).
- GA wall time and memory for K-cluster flows at these sizes.

## 14. Files

- `experiments/ids57_probes/armf_common.py` - dataset, relabel, evaluator, streamer scoring, Rust threshold read.
- `experiments/ids57_probes/armf_fidelity.py` -> `armf_fidelity.jsonl` - section 2.
- `experiments/ids57_probes/armf_gate.py` -> `armf_gate.jsonl` - every row behind sections 4-8 (22 variants x 5 draws).
- `experiments/ids57_probes/armf_sum.py` -> `armf_summary.txt` - the tables.
- Interpreter: `/Users/lacg/wnn-venv/bin/python`, `ram_accelerator` ABI 28 (the worker's installed wheel, read only).
- `experiments/` is gitignored; these files and the earlier IDS-57 probes (`probe1-4.py`, `sum*.py`) are
  force-added so the gate can be re-run.
