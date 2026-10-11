# IDS-26 — Final GA population diversity (search-collapse test)

Parent spike: IDS-23 (GA plateau: ceiling vs search collapse vs objective mismatch). Read-only analysis,
10/10/2026. No runs, code or configs changed.

Scripts: `scripts/ids_plateau/ids26_population_diversity.py` (per-run metrics + cross-run best-genome
similarity), `ids26_report.py` (tables), `ids26_duplicate_neurons.py` (duplicate neurons inside one genome).

## Verdict

**PARTIAL collapse, the same pattern on all four datasets.**

- **Within a run: yes, collapse.** In every current-era sweep run (131/131: unswr QUAD 28, unswr QSR-abi13 29,
  unswt 24, cicids 20, ciciot 30), the final GA population is ONE basin. Bits are frozen (within-pop SD 0.00 to 0.08),
  neuron counts span about 11 to 28 neurons, and the whole population descends from 3 to 5 grid genomes, which are
  the grid's top 5 in 91/102 QUAD runs. The 50-genome grid population (15 architecture cells, no shared neurons)
  shrinks to a gene pool smaller than one genome's worth of distinct neurons. That is why rare grid genomes do not
  survive: GA Neurons throws away the grid's diversity in its first generations.
- **Across seeds and arms: no collapse.** The final best genomes of different seeds share no neurons (J_neu = 0.000
  in every dataset). Different arms that start from the SAME seed's grid (unswr: grid best genomes share 60 to 72%
  of neurons) end almost disjoint (J_neu 0.02 to 0.03). Feature usage (J_use) across seeds is LOWER after the GA than
  between random grid draws (unswr 0.32 vs 0.61, cicids 0.15 vs 0.57, ciciot 0.24 vs 0.61). Runs do not converge on
  one genome or one feature set. They reach the same F1/FPR plateau from disjoint genomes.
- **What this means for IDS-23.** The plateau is not "every run is trapped in the same basin". Many unrelated basins
  land on the same metric, which points toward a ceiling or the objective (the sibling stories), not toward a
  single-basin search failure. Within-run collapse does explain Luiz's "GA normalizes" observation, and why the
  best-F1 cell usually comes from Grid. The GA is a local refiner of the grid's top lineage, with no mechanism that
  keeps a second lineage alive.
- **New finding: duplicate neurons.** 67 to 78% of the neurons in the GA's best genomes are exact copies of another
  neuron in the same genome. Grid genomes have none. The unswr "~400 neurons" GA genome has about 136 distinct
  neurons, and cicids 232 neurons have 52 distinct. A duplicate trains and answers identically, so it adds no
  feature, only extra weight for one neuron's vote. Most of the GA "neurons" dimension is searching vote weights over
  about 50 to 140 grid neurons. See the mechanism below.

## Method

- Data: `phase_result.final_population` of the Grid Search checkpoint (= the GA's seed population, 50 genomes) and of
  the GA Neurons checkpoint (`experiment_end` rows of the `checkpoints` table), plus `phase_result.best_genome`.
  Streamed one run at a time. 247 completed runs.
- Cohorts: **SWEEP** = IDSXD-unswr-quad (9 arms; B34-CTRL only the `-immfix` reruns), IDSXD-unswr-qsr `-abi13`
  (shown separately), IDSXD-unswt-quad-16b (8 arms; B34-CTRL unswt is excluded by the regex, see note),
  IDSAGG/IDSAGGX cicids and ciciot. **SP100** = the August production-config runs (29 completed per dataset, plain
  harmonic-rank aggregation, `GA gens = 60` in all of them). SP100 is legacy era and shown for contrast only.
- A neuron is identified as (bits, sorted multiset of input-bit indices). In the NEURONS phase, crossover
  (positional and pool-shuffle) and mutation copy neurons verbatim. Mutation only appends or removes trailing
  neurons, and appended neurons get random connections. So exact neuron identity is a faithful lineage marker.
- `J_neu`: Jaccard of two genomes' neuron SETS (shared lineage). `J_in`: Jaccard of the union of input bits observed.
  This saturates for large genomes and is 1.0 everywhere on unswt-16b (small input space). `J_use`: weighted
  (Ruzicka) Jaccard of the normalized per-input-bit usage histograms (feature-selection similarity). Grid genomes
  are independent random draws, so grid-vs-grid pairs give the chance baseline.
- **Basins**: single-linkage components over the edge rule `J_neu >= 0.5 AND |dN|/max(N) <= 0.10 AND |d mean bits| <= 1`
  (same lineage AND same architecture). There is a sensitivity row at `J_neu >= 0.2`. **Arch cells** = distinct
  (modal bits, 50-neuron bucket). **Gene pool** = |distinct neurons in the whole population| / mean neurons per genome
  (50 = no sharing, 1 = the population holds one genome's worth of neurons, < 1 = duplicates inside genomes).
- **Ancestry**: the share of the GA best genome's neurons found verbatim in the grid population, and how many grid
  genomes contribute.
- Fitness spread is the GA's own k-fold `cached_metrics` (not held-out). No held-out metric is used in this story.

Note on B34-CTRL: the sweep regex drops every plain `IDSXD-*-B34-CTRL-r*` flow and keeps only the unswr `-immfix`
reruns. That leaves unswt with 8 arms × 3 seeds = 24 runs. The unswt B34-CTRL runs started 26/09, after the 17/09
immigrant fix, so they could be added. Leaving them out does not change any conclusion: every unswt arm already
ends at about 34 bits (max_bits 34).

## Summary table (mean over runs; Grid = seed population, GA = final population)

```
dataset  cohort      runs | neurons SD in pop | bits SD in pop | J_neu in pop | J_use in pop | gene pool  | basins     | grid ancestors | GA-best distinct
                          |  Grid  ->   GA     | Grid  ->  GA   | Grid -> GA   | Grid -> GA   | Grid -> GA | Grid -> GA | alive in GA pop| neuron fraction
---------------------------------------------------------------------------------------------------------------------------------------------------------------------
unswr    SWEEP-QUAD  28   |  133.8 ->    5.5    | 3.73 -> 0.04   | 0.00 -> 0.86 | 0.52 -> 0.84 | 50 -> 0.5  | 50 -> 1.0  |  4.6 / 50      | 0.33
unswr    SWEEP-QSR   29   |  135.5 ->    5.6    | 3.50 -> 0.05   | 0.00 -> 0.84 | 0.52 -> 0.83 | 50 -> 0.5  | 50 -> 1.0  |  4.6 / 50      | 0.33
unswt    SWEEP-QUAD  24   |  135.1 ->    4.5    | 2.51 -> 0.00   | 0.00 -> 0.93 | 0.81 -> 0.98 | 50 -> 0.7  | 50 -> 1.0  |  3.4 / 50      | 0.43
cicids   SWEEP-QUAD  20   |  140.9 ->    4.3    | 2.35 -> 0.01   | 0.00 -> 0.82 | 0.56 -> 0.80 | 50 -> 0.4  | 50 -> 1.0  |  3.0 / 50      | 0.23
ciciot   SWEEP-QUAD  30   |   65.8 ->    2.4    | 14.20 -> 0.08  | 0.00 -> 0.87 | 0.61 -> 0.91 | 50 -> 0.5  | 50 -> 1.0  |  2.9 / 50      | 0.30
---------------------------------------------------------------------------------------------------------------------------------------------------------------------
unswr    SP100       29   |  169.8 ->  138.0    | 6.76 -> 3.12   | 0.00 -> 0.32 | 0.42 -> 0.40 | 50 -> 5.4  | 50 -> 7.9  |  6.9 / 50      | 0.46
unswt    SP100       29   |  163.5 ->  140.5    | 5.42 -> 6.12   | 0.00 -> 0.34 | 0.66 -> 0.54 | 50 -> 1.8  | 50 -> 5.1  |  5.2 / 50      | 0.46
cicids   SP100       29   |  144.8 ->   39.6    | 2.76 -> 0.09   | 0.00 -> 0.50 | 0.55 -> 0.54 | 50 -> 0.8  | 50 -> 1.9  |  4.6 / 50      | 0.22
ciciot   SP100       29   |   64.5 ->   20.9    | 17.20 -> 2.50  | 0.00 -> 0.65 | 0.60 -> 0.74 | 50 -> 0.6  | 50 -> 2.2  |  5.0 / 50      | 0.23
```

GA runs with exactly one basin: SWEEP 131/131. SP100: unswr 0/29, unswt 0/29, cicids 5/29, ciciot 11/29.

**SP100's apparent diversity is not useful diversity.** Its GA populations are bimodal. For example, in
SP100-unswr-r10596 the final population holds about 10-neuron genomes at k-fold F1 0.86 next to about 360-neuron
genomes at F1 0.93. Plain harmonic-rank aggregation keeps rank specialists (the small genomes win a rank), which
matches the known "WHM crowns specialists" finding. The extra basins are degenerate small genomes, not competing good
lineages. That is also why SP100's within-pop k-fold F1 range is 7 to 14 pp vs 0.06 to 0.3 pp in the sweeps.

## Cross-seed / cross-arm similarity of the final BEST genomes

```
dataset / cohort        pair relation        |  J_neu Grid -> GA |  J_use Grid -> GA | pairs
------------------------------------------------------------------------------------------------
unswr  / SWEEP-QUAD     same-arm, diff-seed  |  0.000 -> 0.000   |  0.611 -> 0.318   |   30
                        same-seed, diff-arm  |  0.720 -> 0.028   |  0.891 -> 0.268   |  100
                        diff both            |  0.000 -> 0.000   |  0.599 -> 0.270   |  248
unswr  / SWEEP-QSR      same-arm, diff-seed  |  0.000 -> 0.000   |  0.617 -> 0.339   |   34
                        same-seed, diff-arm  |  0.602 -> 0.024   |  0.841 -> 0.266   |  108
unswt  / SWEEP-QUAD     same-arm, diff-seed  |  0.000 -> 0.000   |  0.823 -> 0.686   |   24
                        same-seed, diff-arm  |  0.112 -> 0.016   |  0.844 -> 0.684   |   84
cicids / SWEEP-QUAD     same-arm, diff-seed  |  0.000 -> 0.000   |  0.567 -> 0.152   |   40
                        same-seed, diff-arm  |  0.113 -> 0.013   |  0.601 -> 0.158   |   30
ciciot / SWEEP-QUAD     same-arm, diff-seed  |  0.000 -> 0.000   |  0.605 -> 0.239   |   60
                        same-seed, diff-arm  |  0.086 -> 0.014   |  0.652 -> 0.265   |   75
unswr  / SP100          diff-seed            |  0.000 -> 0.000   |  0.497 -> 0.323   |  406
unswt  / SP100          diff-seed            |  0.000 -> 0.000   |  0.277 -> 0.676   |  406
cicids / SP100          diff-seed            |  0.000 -> 0.000   |  0.478 -> 0.126   |  406
ciciot / SP100          diff-seed            |  0.000 -> 0.000   |  0.660 -> 0.242   |  406
```

Reading: the GA moves best genomes of different seeds and arms APART, not together. The only exception is unswt
SP100 J_use (0.28 -> 0.68), where SP100's grid best genomes are tiny (86 neurons on average) and their usage
histograms are mostly noise. Same-seed arms share most of the starting grid on unswr (J_neu 0.6 to 0.7) and still
diverge to J_neu about 0.02. The weight vector changes which lineage wins, but not the plateau it reaches.

## Duplicate neurons inside one genome

```
cohort                runs | Grid best distinct | GA best distinct | GA best n (distinct n) | GA pop distinct
------------------------------------------------------------------------------------------------------------
unswr  SWEEP-QUAD      28  |      1.000         |   0.330±0.139    |    396 (136)           |  0.333±0.133
unswr  SWEEP-QSR       29  |      1.000         |   0.331±0.131    |    387 (135)           |  0.332±0.123
unswt  SWEEP-QUAD      24  |      1.000         |   0.434±0.136    |    339 (143)           |  0.432±0.135
cicids SWEEP-QUAD      20  |      1.000         |   0.231±0.098    |    232 ( 52)           |  0.242±0.101
ciciot SWEEP-QUAD      30  |      1.000         |   0.296±0.142    |    167 ( 43)           |  0.296±0.145
unswr  SP100           29  |      1.000         |   0.462±0.177    |    332 (144)           |  0.614±0.130
unswt  SP100           29  |      1.000         |   0.458±0.221    |    327 (143)           |  0.653±0.094
cicids SP100           29  |      1.000         |   0.220±0.100    |    208 ( 41)           |  0.304±0.100
ciciot SP100           29  |      1.000         |   0.225±0.041    |    221 ( 49)           |  0.251±0.038
```

Mechanism (`core/neighbor_search.rs`):
- `crossover_ps_neurons` (pool-shuffle, `pool_shuffle_ratio=0.8`) pools both parents' neurons with no
  de-duplication, then takes N from the shuffled pool. When the parents share neurons (which is what happens once
  they belong to one lineage, and `assortative_mating_ratio=0.85` prefers similar parents), the child draws both
  copies.
- `mutate_neurons_phase` copies the kept neurons verbatim. The fingerprint guard in `search_offspring` and the elite
  de-duplication compare WHOLE genomes, never neurons inside a genome.
- Existing connections are never mutated in the NEURONS phase. The only new features are randomly connected
  appended neurons (±3% delta, applied with probability 0.1 per offspring on the single IDS cluster). 70 to 93% of
  the GA best genome's neurons are verbatim grid neurons.

So in the NEURONS phase the GA cannot discover new features. It re-weights (by copying) a subset of its top grid
ancestors' neurons. Whether duplication-as-weighting helps or hurts held-out is NOT known from this data, and needs
the A/B in proposal P1.

## Why the population collapses (grounded in the code)

1. μ+λ truncation (`generic_ga.py`): 20% elites + `population_size` offspring compete, top `population_size` kept.
   On top of that, `fitness_percentile=0.75` generates 67 and keeps the best 50. Selection pressure is high and there
   is no similarity term anywhere in survival.
2. Elite de-duplication is by exact fingerprint, so near-clones (J_neu 0.9) fill the elite slots.
3. `assortative_mating_ratio=0.85` uses an ARCHITECTURE-only distance (neurons and mean bits, `genome_distance`).
   Once the architecture collapses (neuron SD about 5, bits frozen), every pair is "similar" and the knob has no
   niching effect. It only speeds up inbreeding, and with pool-shuffle it speeds up duplication.
4. **Random immigrants are dead on the IDS path.** `GAConfig.immigrant_fraction` is read only in the Python
   `_generate_offspring`. `ArchitectureGAStrategy._generate_offspring` routes to Rust `search_offspring`, which has
   no immigrant branch. `immigrant_fraction` is also not a flow param (not in `KNOWN_PARAMS`). The only immigrants
   IDS flows ever get come from the adaptive-scaler population growth (50 -> 57/65 on WARNING/CRITICAL), and those
   are evaluated once and must beat the incumbents in the same truncation.
5. The GA starts from the 50 grid genomes. They are 15 architecture cells × random draws, independent, and diverse
   (J_neu 0). The collapse happens inside GA Neurons: the grid's top-5 lineages win the first tournaments.

## Proposals (for Luiz. Nothing is changed. The n=100 tandem keeps the current GA.)

Ordered by expected information per cost. Each is a paired A/B against the same-seed control on n=5 seeds per
dataset (k-fold 5 unchanged). The judge is held-out val_cal F1/FPR (best_f1 and best_fitness genomes) from the
final validation rows. The diagnostics are this script's basins, gene pool and distinct-neuron fraction.
One GA Neurons stage is enough to judge.

- **P1. Neuron de-duplication in crossover (Rust, worker wheel).** In `crossover_ps_neurons` /
  `crossover2_neurons`, build the pool as a set of neuron keys, and fill any shortfall with fresh random neurons at the
  mode bits. This is a test of duplication-as-weighting, not a fix. If the held-out result is unchanged or better at
  about 1/3 the neurons, the plateau genomes are 2 to 4x over-sized (relevant to the FPGA/MCU footprint claim). If it
  is worse, duplication is acting as a learned vote weight, and an explicit per-neuron weight would be a cleaner
  representation (a design discussion, not code). Observable: GA best distinct fraction about 1.0.
- **P2. Wire random immigrants into the Rust offspring path.** Add `immigrant_fraction` to `KNOWN_PARAMS` and to
  the worker -> GAConfig plumbing. In `ArchitectureGAStrategy._generate_offspring`, generate
  `round(immigrant_fraction * n)` genomes via `create_immigrant` (it already follows the phase: mode bits, band
  neurons) and evaluate them through the same batch evaluator before the Rust offspring. Suggested 0.1. Observable:
  gene pool > 1 and alive grid ancestors > 5. Cheap and uses existing hooks.
- **P3. Similarity-aware survival (deterministic crowding / neuron-Jaccard elite de-dup).** Replace the
  exact-fingerprint elite de-dup with "skip an elite whose J_neu to an already-kept elite is >= 0.8", or let each
  child compete only with its closer parent (Mahfoud deterministic crowding; the distance from
  `project_ga_diversity_preservation.md` with the connection term = 1 - J_neu). Observable: basins >= 3 at the end
  of GA. This is the real niching test of the "rare grid genomes are lost" claim. If held-out does not move with
  3+ surviving basins, search collapse is ruled out as the plateau cause.
- **P4. Make `assortative_mating_ratio` meaningful or turn it off.** Its distance ignores connections. Use
  1 - J_neu, or run an arm at 0.0 with `pool_shuffle_ratio` 0.0 (positional crossover duplicates less). This is
  zero code (both are flow params) and could be the first arm if Luiz wants a no-wheel probe.
- **P5. A GA Connections stage after GA Neurons** (the `optimize_connections` / `PhaseType.CONNECTIONS` path already
  exists). It is the only existing mechanism that creates new features from a fixated lineage. It is a flow-shape
  change (third experiment), queued via POST /api/flows, so it is post-tandem.

Cross-seed evidence already argues that the plateau level is not a single-basin artifact. So the prior on any of
these lifting held-out F1 by a large amount is low. P1 is the one with a likely concrete payoff (genome size), and
P3 is the decisive collapse test.

## Full per-cohort tables (script output)

```

unswr / SP100 (quad)  runs=29  arms=1
  metric                         |  Grid (seed pop) |     GA final pop
  --------------------------------------------------------------------
  pop size                       |             50±0 |             57±0
  distinct genomes               |         50.0±0.0 |         35.4±7.4
  neurons mean                   |           261±28 |           191±88
  neurons SD (within pop)        |       169.8±13.8 |       138.0±50.9
  neurons range (max-min)        |           492±18 |          379±137
  bits mean                      |       19.62±1.34 |       14.53±3.50
  bits SD (within pop)           |        6.76±0.94 |        3.12±2.47
  bits range (max-min)           |       24.28±3.95 |        9.28±7.39
  J_neu within-pop               |      0.000±0.000 |      0.317±0.105
  J_in within-pop                |      0.685±0.104 |      0.513±0.093
  J_use within-pop               |      0.418±0.066 |      0.398±0.072
  gene pool (genome-equiv.)      |         50.0±0.0 |          5.4±4.5
  basins (J_neu>=0.5)            |         50.0±0.0 |          7.9±6.0
  basins (J_neu>=0.2)            |         50.0±0.0 |          5.9±3.8
  arch cells                     |         15.0±0.0 |          5.2±2.6
  k-fold F1 range in pop (pp)    |      2.653±3.667 |      7.342±8.116
  best genome neurons            |          272±141 |          332±145
  best genome bits               |         14.8±2.4 |         16.5±3.6
  GA best: neurons verbatim from grid 0.886±0.085, grid ancestors 5.2±1.9, top-ancestor share 0.443±0.134, grid genomes still alive in GA pop 6.9±2.5/50, top ancestor = grid pop[0..4] in 14/29
  GA gens run 60±0; runs with GA basins==1: 0/29

unswr / SWEEP (quad)  runs=28  arms=9
  metric                         |  Grid (seed pop) |     GA final pop
  --------------------------------------------------------------------
  pop size                       |             50±0 |             61±4
  distinct genomes               |         50.0±0.0 |         38.7±5.6
  neurons mean                   |           344±24 |           394±66
  neurons SD (within pop)        |       133.8±11.2 |          5.5±2.3
  neurons range (max-min)        |           414±34 |             25±8
  bits mean                      |       15.42±7.73 |       15.95±7.58
  bits SD (within pop)           |        3.73±1.56 |        0.04±0.04
  bits range (max-min)           |       13.00±5.93 |        0.16±0.16
  J_neu within-pop               |      0.000±0.000 |      0.856±0.057
  J_in within-pop                |      0.849±0.032 |      0.939±0.038
  J_use within-pop               |      0.520±0.034 |      0.840±0.067
  gene pool (genome-equiv.)      |         50.0±0.0 |          0.5±0.2
  basins (J_neu>=0.5)            |         50.0±0.0 |          1.0±0.0
  basins (J_neu>=0.2)            |         50.0±0.0 |          1.0±0.0
  arch cells                     |         13.7±3.2 |          1.5±0.5
  k-fold F1 range in pop (pp)    |      0.248±0.115 |      0.186±0.159
  best genome neurons            |           407±47 |           396±67
  best genome bits               |         16.0±7.7 |         16.0±7.6
  GA best: neurons verbatim from grid 0.915±0.077, grid ancestors 4.5±1.8, top-ancestor share 0.483±0.133, grid genomes still alive in GA pop 4.6±1.9/50, top ancestor = grid pop[0..4] in 25/28
  GA gens run 74±25; runs with GA basins==1: 28/28

unswr / SWEEP-QSR (qsr)  runs=29  arms=9
  metric                         |  Grid (seed pop) |     GA final pop
  --------------------------------------------------------------------
  pop size                       |             50±0 |             61±4
  distinct genomes               |         50.0±0.0 |         40.9±7.1
  neurons mean                   |           340±26 |           385±74
  neurons SD (within pop)        |       135.5±11.3 |          5.6±2.1
  neurons range (max-min)        |           416±37 |            28±11
  bits mean                      |       15.97±8.38 |       16.42±8.24
  bits SD (within pop)           |        3.50±1.64 |        0.05±0.08
  bits range (max-min)           |       12.14±5.90 |        0.24±0.30
  J_neu within-pop               |      0.000±0.000 |      0.839±0.079
  J_in within-pop                |      0.846±0.034 |      0.927±0.059
  J_use within-pop               |      0.519±0.038 |      0.831±0.091
  gene pool (genome-equiv.)      |         50.0±0.0 |          0.5±0.2
  basins (J_neu>=0.5)            |         50.0±0.0 |          1.0±0.0
  basins (J_neu>=0.2)            |         50.0±0.0 |          1.0±0.0
  arch cells                     |         13.4±3.5 |          1.4±0.6
  k-fold F1 range in pop (pp)    |      0.266±0.113 |      0.154±0.097
  best genome neurons            |           397±63 |           387±73
  best genome bits               |         16.1±8.4 |         16.4±8.2
  GA best: neurons verbatim from grid 0.907±0.086, grid ancestors 4.6±1.8, top-ancestor share 0.475±0.185, grid genomes still alive in GA pop 4.6±1.8/50, top ancestor = grid pop[0..4] in 23/29
  GA gens run 70±20; runs with GA basins==1: 29/29

unswt / SP100 (quad)  runs=29  arms=1
  metric                         |  Grid (seed pop) |     GA final pop
  --------------------------------------------------------------------
  pop size                       |             50±0 |             57±0
  distinct genomes               |         50.0±0.0 |         28.0±5.7
  neurons mean                   |           264±23 |           159±88
  neurons SD (within pop)        |        163.5±8.8 |       140.5±57.1
  neurons range (max-min)        |            495±0 |          325±131
  bits mean                      |       30.43±1.11 |       26.91±5.07
  bits SD (within pop)           |        5.42±1.66 |        6.12±4.74
  bits range (max-min)           |       22.41±7.59 |      15.26±11.78
  J_neu within-pop               |      0.000±0.000 |      0.336±0.063
  J_in within-pop                |      0.848±0.078 |      0.669±0.215
  J_use within-pop               |      0.661±0.065 |      0.536±0.156
  gene pool (genome-equiv.)      |         50.0±0.0 |          1.8±0.4
  basins (J_neu>=0.5)            |         50.0±0.0 |          5.1±2.0
  basins (J_neu>=0.2)            |         50.0±0.0 |          4.6±2.1
  arch cells                     |         15.0±0.0 |          3.6±0.8
  k-fold F1 range in pop (pp)    |     10.538±9.737 |     14.025±7.839
  best genome neurons            |           86±145 |          327±120
  best genome bits               |         28.0±6.4 |         32.4±2.2
  GA best: neurons verbatim from grid 0.874±0.080, grid ancestors 3.7±1.6, top-ancestor share 0.606±0.209, grid genomes still alive in GA pop 5.2±1.7/50, top ancestor = grid pop[0..4] in 5/29
  GA gens run 66±11; runs with GA basins==1: 0/29

unswt / SWEEP (quad)  runs=24  arms=8
  metric                         |  Grid (seed pop) |     GA final pop
  --------------------------------------------------------------------
  pop size                       |             50±0 |             60±4
  distinct genomes               |         50.0±0.0 |         36.1±7.4
  neurons mean                   |           321±11 |          338±114
  neurons SD (within pop)        |        135.1±4.4 |          4.5±2.0
  neurons range (max-min)        |            400±0 |             22±8
  bits mean                      |       31.65±0.21 |       33.59±0.67
  bits SD (within pop)           |        2.51±0.17 |        0.00±0.01
  bits range (max-min)           |        8.00±2.04 |        0.02±0.03
  J_neu within-pop               |      0.000±0.000 |      0.932±0.025
  J_in within-pop                |      1.000±0.000 |      1.000±0.000
  J_use within-pop               |      0.810±0.005 |      0.977±0.010
  gene pool (genome-equiv.)      |         50.0±0.0 |          0.7±0.2
  basins (J_neu>=0.5)            |         50.0±0.0 |          1.0±0.0
  basins (J_neu>=0.2)            |         50.0±0.0 |          1.0±0.0
  arch cells                     |         14.8±0.4 |          1.4±0.5
  k-fold F1 range in pop (pp)    |      0.610±0.308 |      4.892±2.657
  best genome neurons            |          325±119 |          339±114
  best genome bits               |         33.6±0.8 |         33.6±0.7
  GA best: neurons verbatim from grid 0.811±0.082, grid ancestors 3.4±1.9, top-ancestor share 0.527±0.156, grid genomes still alive in GA pop 3.4±1.9/50, top ancestor = grid pop[0..4] in 21/24
  GA gens run 109±24; runs with GA basins==1: 24/24

cicids / SP100 (quad)  runs=29  arms=1
  metric                         |  Grid (seed pop) |     GA final pop
  --------------------------------------------------------------------
  pop size                       |             50±0 |             57±0
  distinct genomes               |         50.0±0.0 |         34.3±4.8
  neurons mean                   |            302±8 |           166±87
  neurons SD (within pop)        |        144.8±5.0 |        39.6±49.3
  neurons range (max-min)        |           443±48 |           93±108
  bits mean                      |       31.86±0.22 |       33.76±0.45
  bits SD (within pop)           |        2.76±0.86 |        0.09±0.14
  bits range (max-min)           |       11.52±7.89 |        0.23±0.36
  J_neu within-pop               |      0.000±0.000 |      0.496±0.036
  J_in within-pop                |      0.891±0.029 |      0.664±0.057
  J_use within-pop               |      0.548±0.018 |      0.537±0.037
  gene pool (genome-equiv.)      |         50.0±0.0 |          0.8±0.3
  basins (J_neu>=0.5)            |         50.0±0.0 |          1.9±0.5
  basins (J_neu>=0.2)            |         50.0±0.0 |          1.6±0.5
  arch cells                     |         14.9±0.4 |          2.1±0.6
  k-fold F1 range in pop (pp)    |      2.109±5.059 |      0.238±0.068
  best genome neurons            |           172±96 |          208±141
  best genome bits               |         33.7±0.7 |         33.8±0.4
  GA best: neurons verbatim from grid 0.927±0.128, grid ancestors 3.9±1.3, top-ancestor share 0.509±0.182, grid genomes still alive in GA pop 4.6±1.4/50, top ancestor = grid pop[0..4] in 25/29
  GA gens run 67±10; runs with GA basins==1: 5/29

cicids / SWEEP (quad)  runs=20  arms=4
  metric                         |  Grid (seed pop) |     GA final pop
  --------------------------------------------------------------------
  pop size                       |             50±0 |             59±4
  distinct genomes               |         50.0±0.0 |         37.0±7.8
  neurons mean                   |            308±7 |          232±120
  neurons SD (within pop)        |        140.9±3.7 |          4.3±2.7
  neurons range (max-min)        |           414±35 |            18±11
  bits mean                      |       31.91±0.23 |       33.84±0.45
  bits SD (within pop)           |        2.35±0.30 |        0.01±0.01
  bits range (max-min)           |        7.00±2.20 |        0.03±0.06
  J_neu within-pop               |      0.000±0.000 |      0.817±0.071
  J_in within-pop                |      0.915±0.017 |      0.892±0.043
  J_use within-pop               |      0.562±0.011 |      0.796±0.056
  gene pool (genome-equiv.)      |         50.0±0.0 |          0.4±0.2
  basins (J_neu>=0.5)            |         50.0±0.0 |          1.0±0.0
  basins (J_neu>=0.2)            |         50.0±0.0 |          1.0±0.0
  arch cells                     |         14.8±0.4 |          1.5±0.5
  k-fold F1 range in pop (pp)    |      0.327±0.114 |      0.058±0.033
  best genome neurons            |          315±160 |          232±120
  best genome bits               |         33.9±0.4 |         33.8±0.4
  GA best: neurons verbatim from grid 0.903±0.073, grid ancestors 3.0±1.2, top-ancestor share 0.609±0.218, grid genomes still alive in GA pop 3.0±1.2/50, top ancestor = grid pop[0..4] in 18/20
  GA gens run 78±24; runs with GA basins==1: 20/20

ciciot / SP100 (quad)  runs=29  arms=1
  metric                         |  Grid (seed pop) |     GA final pop
  --------------------------------------------------------------------
  pop size                       |             50±0 |             57±0
  distinct genomes               |         50.0±0.0 |         33.8±4.5
  neurons mean                   |            172±9 |           207±30
  neurons SD (within pop)        |         64.5±6.8 |        20.9±25.1
  neurons range (max-min)        |           199±22 |            62±60
  bits mean                      |       70.70±2.76 |       62.52±2.62
  bits SD (within pop)           |       17.20±1.30 |        2.50±2.73
  bits range (max-min)           |       63.45±6.57 |        7.68±9.27
  J_neu within-pop               |      0.000±0.000 |      0.653±0.093
  J_in within-pop                |      0.946±0.028 |      0.869±0.047
  J_use within-pop               |      0.603±0.021 |      0.736±0.076
  gene pool (genome-equiv.)      |         50.0±0.0 |          0.6±0.1
  basins (J_neu>=0.5)            |         50.0±0.0 |          2.2±1.4
  basins (J_neu>=0.2)            |         50.0±0.0 |          2.1±1.4
  arch cells                     |         15.0±0.0 |          2.2±0.9
  k-fold F1 range in pop (pp)    |      1.918±0.993 |      0.582±0.231
  best genome neurons            |           221±34 |           221±28
  best genome bits               |         64.0±0.0 |         63.6±1.7
  GA best: neurons verbatim from grid 0.816±0.114, grid ancestors 4.0±1.8, top-ancestor share 0.442±0.154, grid genomes still alive in GA pop 5.0±1.6/50, top ancestor = grid pop[0..4] in 25/29
  GA gens run 79±17; runs with GA basins==1: 11/29

ciciot / SWEEP (quad)  runs=30  arms=6
  metric                         |  Grid (seed pop) |     GA final pop
  --------------------------------------------------------------------
  pop size                       |             50±0 |             59±4
  distinct genomes               |         50.0±0.0 |         32.0±8.1
  neurons mean                   |            170±8 |           167±60
  neurons SD (within pop)        |         65.8±6.3 |          2.4±1.4
  neurons range (max-min)        |           193±17 |             11±5
  bits mean                      |      73.52±10.84 |      66.61±14.80
  bits SD (within pop)           |       14.20±1.56 |        0.08±0.18
  bits range (max-min)           |      43.87±10.74 |        0.31±0.72
  J_neu within-pop               |      0.000±0.000 |      0.874±0.061
  J_in within-pop                |      0.952±0.041 |      0.944±0.033
  J_use within-pop               |      0.610±0.039 |      0.912±0.065
  gene pool (genome-equiv.)      |         50.0±0.0 |          0.5±0.2
  basins (J_neu>=0.5)            |         50.0±0.0 |          1.0±0.0
  basins (J_neu>=0.2)            |         50.0±0.0 |          1.0±0.0
  arch cells                     |         14.8±0.4 |          1.4±0.5
  k-fold F1 range in pop (pp)    |      1.304±0.307 |      0.301±0.176
  best genome neurons            |           180±69 |           167±60
  best genome bits               |        66.1±15.6 |        66.6±14.8
  GA best: neurons verbatim from grid 0.700±0.130, grid ancestors 2.9±1.3, top-ancestor share 0.467±0.193, grid genomes still alive in GA pop 2.9±1.3/50, top ancestor = grid pop[0..4] in 27/30
  GA gens run 100±35; runs with GA basins==1: 30/30

Cross-run best-genome similarity (mean pairwise; n pairs)

unswr / SP100 (quad)
  pair relation          | phase |   J_neu |    J_in |   J_use | n
  same-arm/diff-seed     | GS    |   0.000 |   0.833 |   0.497 | 406
  same-arm/diff-seed     | GA    |   0.000 |   0.656 |   0.323 | 406

unswr / SWEEP (quad)
  pair relation          | phase |   J_neu |    J_in |   J_use | n
  same-arm/diff-seed     | GS    |   0.000 |   0.978 |   0.611 | 30
  same-arm/diff-seed     | GA    |   0.000 |   0.623 |   0.318 | 30
  same-seed/diff-arm     | GS    |   0.720 |   0.992 |   0.891 | 100
  same-seed/diff-arm     | GA    |   0.028 |   0.564 |   0.268 | 100
  diff-both              | GS    |   0.000 |   0.977 |   0.599 | 248
  diff-both              | GA    |   0.000 |   0.585 |   0.270 | 248

unswr / SWEEP-QSR (qsr)
  pair relation          | phase |   J_neu |    J_in |   J_use | n
  same-arm/diff-seed     | GS    |   0.000 |   0.972 |   0.617 | 34
  same-arm/diff-seed     | GA    |   0.000 |   0.646 |   0.339 | 34
  same-seed/diff-arm     | GS    |   0.602 |   0.988 |   0.841 | 108
  same-seed/diff-arm     | GA    |   0.024 |   0.549 |   0.266 | 108
  diff-both              | GS    |   0.000 |   0.968 |   0.587 | 264
  diff-both              | GA    |   0.000 |   0.567 |   0.267 | 264

unswt / SP100 (quad)
  pair relation          | phase |   J_neu |    J_in |   J_use | n
  same-arm/diff-seed     | GS    |   0.000 |   0.401 |   0.277 | 406
  same-arm/diff-seed     | GA    |   0.000 |   1.000 |   0.676 | 406

unswt / SWEEP (quad)
  pair relation          | phase |   J_neu |    J_in |   J_use | n
  same-arm/diff-seed     | GS    |   0.000 |   1.000 |   0.823 | 24
  same-arm/diff-seed     | GA    |   0.000 |   1.000 |   0.686 | 24
  same-seed/diff-arm     | GS    |   0.112 |   1.000 |   0.844 | 84
  same-seed/diff-arm     | GA    |   0.016 |   1.000 |   0.684 | 84
  diff-both              | GS    |   0.000 |   1.000 |   0.823 | 168
  diff-both              | GA    |   0.000 |   1.000 |   0.680 | 168

cicids / SP100 (quad)
  pair relation          | phase |   J_neu |    J_in |   J_use | n
  same-arm/diff-seed     | GS    |   0.000 |   0.810 |   0.478 | 406
  same-arm/diff-seed     | GA    |   0.000 |   0.279 |   0.126 | 406

cicids / SWEEP (quad)
  pair relation          | phase |   J_neu |    J_in |   J_use | n
  same-arm/diff-seed     | GS    |   0.000 |   0.898 |   0.567 | 40
  same-arm/diff-seed     | GA    |   0.000 |   0.345 |   0.152 | 40
  same-seed/diff-arm     | GS    |   0.113 |   0.897 |   0.601 | 30
  same-seed/diff-arm     | GA    |   0.013 |   0.348 |   0.158 | 30
  diff-both              | GS    |   0.000 |   0.897 |   0.564 | 120
  diff-both              | GA    |   0.000 |   0.339 |   0.147 | 120

ciciot / SP100 (quad)
  pair relation          | phase |   J_neu |    J_in |   J_use | n
  same-arm/diff-seed     | GS    |   0.000 |   0.998 |   0.660 | 406
  same-arm/diff-seed     | GA    |   0.000 |   0.680 |   0.242 | 406

ciciot / SWEEP (quad)
  pair relation          | phase |   J_neu |    J_in |   J_use | n
  same-arm/diff-seed     | GS    |   0.000 |   0.946 |   0.605 | 60
  same-arm/diff-seed     | GA    |   0.000 |   0.609 |   0.239 | 60
  same-seed/diff-arm     | GS    |   0.086 |   0.944 |   0.652 | 75
  same-seed/diff-arm     | GA    |   0.014 |   0.614 |   0.265 | 75
  diff-both              | GS    |   0.000 |   0.943 |   0.601 | 300
  diff-both              | GA    |   0.000 |   0.604 |   0.236 | 300
```
