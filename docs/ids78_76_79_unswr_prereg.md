# IDS-78 / IDS-76 / IDS-79 on unswr: pre-registration

Written 23/11/2026 (EST) by ids-security, at Luiz's request (23/11 ~20:15 EST), before any flow of these arms was
created. Goal: surpass RF raw top-20 on unswr TEST (95.81 F1 / 0.288 % FPR). Evidence base: `docs/ids77_unswr_gap.md`.

## Control, pairing, reporting cell

- **Control** = IDS-70 arm E: flows 7591 / 7592 / 7593 (seeds r20421 / r20422 / r20423). Cut E (`ids_cut_method=info_gain_mc`),
  fixed 80 bits, `max_neurons=250` per cluster, Grid Search -> GA Neurons, protocol v2 (`random_3way`, 5-fold CV on TRAIN
  drives the GA, VAL calibrates thresholds only, TEST report-only), worker ABI 29. The controls are not re-flown.
- **Pairing**: each arm is flown on the same 3 seeds, n=3, and compared seed by seed.
- **Reporting cell (fixed)**: the arm's last GA phase, `test_summaries`, `test_point='final'`, `genome_type='best_f1'`,
  threshold mode `margin_val_cal`, TEST. For IDS-78 that is the GA Connections experiment. For the control and IDS-79, the
  GA Neurons experiment.
- **No VAL in selection or stopping.** Every arm keeps the control's GA fitness (5-fold CV mean on TRAIN,
  `ids_offspring_folds=all`) and its patience rule. VAL is used only for threshold calibration.

Control cells (TEST, read from `test_summaries` on 23/11):

```
seed     flow   F1 (%)   FPR (%)  acc (%)   cells       hours
r20421   7591   95.366   0.278    99.332    1,306,543   11.4
r20422   7592   95.466   0.257    99.349    1,298,510   10.3
r20423   7593   95.262   0.270    99.320    1,290,622   14.7
mean            95.365   0.268    99.334
RF raw top-20 (5 seeds)  95.812 / 0.288     XGB raw top-20  95.603 / 0.427
```

## Decision rule (same for every arm)

Following Luiz's rule (adopt = consistently better in the expected direction and not worse elsewhere, no magnitude bar):
- **ADOPT**: dF1 > 0 on 3/3 seeds AND mean dFPR <= 0.
- **Luiz decides**: dF1 > 0 on 3/3 seeds with mean dFPR > 0, or dF1 > 0 on 2/3 seeds with mean dF1 > 0 and mean dFPR <= 0.
- **NOT ADOPT**: anything else.
- Cells and hours are tie-breakers only. Accuracy is reported, not ruled on (base-rate caveat).
- Surpassing RF raw is a separate statement: arm mean F1 > 95.81 with mean FPR <= 0.288. It is not the adopt rule.

## IDS-78: GA Connections

**Hypothesis.** A GA phase that mutates only connections (neurons and bits fixed) after GA Neurons can move connections
onto bits that separate Fuzzers from benign, raising recall at the same FPR.

**IDS-77 evidence (against).** G2 found that where connections land barely matters on random genomes (per-feature or
RF-importance draws lose 0.26-0.32 pp VAL; only a sqrt draw gains +0.05, about one baseline SD). The operator is
local: each mutated connection moves by at most 2 bit positions (`delta in [-2, 2]`, clamped to [0, 1104)), which
under the thermometer encoding is usually a neighbouring cut of the same feature. IDS-77 listed this phase as killed at
the gate. Luiz asked to measure it anyway.

**Expected size.** Unknown. Honest prior: 0 to +0.05 pp F1 on TEST, FPR flat.

**Design.**
- One experiment: `{"name": "GA Connections", "experiment_type": "ga", "phase_type": "ga_connections",
  "optimize_connections": true}` (the same spec the 5/7/10-phase templates use; last run on IDS flows 430-433).
- **Seeded from the control's own GA Neurons final population** (`seed_checkpoint_id` = the control's GA Neurons
  `experiment_end` checkpoint: 8426 / 8428 / 8490; 57 genomes each, all neurons 80 bits, connections 0-1103). Grid and
  GA Neurons are **not re-flown**: that would repeat the control recipe on seeds already flown. The arm is therefore
  "the control + one GA Connections phase", paired on the exact same starting population.
- Params: the control's, verbatim (52 keys). Fixed 80b and cut E are kept: the connections operator never changes
  bits or neurons (Rust `mutate_connections_phase`, Python `_mutate_connections`), and crossover keeps each child's
  architecture.
- Mutation rate 0.1 per connection, with the GA's usual crossover and immigrants (immigrants inherit neurons/bits from
  the population mode; fresh random connections).
- Runs on the installed worker (ABI 29) with no code change and no swap: the IDS cache maps `phase_type=2` to
  `PhaseType::Connections`, and `total_input_bits` comes from the cache's own encoded width.

**Built-in fidelity check.** The arm's GA Connections `init` test cells are computed on the control's final GA Neurons
population. They should equal the control's GA Neurons `final` cells. If they do not, the pairing is flagged before any
readout.

**Confound.** The arm gets extra GA generations. The control's GA Neurons stopped on its patience rule (plateau), so
"more GA Neurons time" is not expected to explain a gain, but this arm cannot separate "connections" from "more search".

**Cost.** GA generation at this genome size is about 7.5 min (control: 11.3 h / 90 gens). Minimum 50 generations under
the patience rule (5 checks x 10 gens), the control ran 80-110. Estimate 6.5-14 h per flow, **about 20-40 h for n=3**.

## IDS-76: K-fold fold rotation (not queued)

**Hypothesis.** Re-drawing the 5-fold partition during the GA stops the GA fitting its fixed folds, converting more of
its CV edge into held-out gain.

**IDS-77 evidence.** Own-fold GA edge +0.38 / +0.32, fresh-fold +0.28 / +0.27 (7591 / 7592). The fold-specific part
is 0.05-0.10. **Expected TEST effect <= +0.1 pp.**

**Status: not implementable on ABI 29.** There is no `ids_fold_rotation` key in `KNOWN_PARAMS` and no code path. The
5 folds are built once in Rust when the IDS cache is constructed (`IDSCache::new` ->
`create_stratified_folds(..., seed)`), and the Python `_kfold_rotation_offset` only rotates which of the fixed folds
is used (a no-op at `ids_kfold_per_gen=5`). It needs a Rust re-partition API, the Python param and schedule, and a
worker wheel swap. Per Luiz's instruction this arm is stopped and reported, not queued.

## IDS-79: max_neurons 500 per cluster

**Hypothesis.** More independent voters per cluster raise Fuzzers recall at flat FPR.

**IDS-77 evidence (for).** G1 (random genomes, VAL): 500 x 500 x 80 gains +0.140 pp F1 on 5/5 draws, Fuzzers recall
+2.6 pts, FPR +0.008. It saturates at 500. The control's grid best and its GA cluster 0 sit on the 250 cap on 3/3
seeds. **Expected TEST effect 0 to +0.15 pp F1, FPR flat to +0.03.** At the top of the bound this reaches XGB raw
F1, not RF raw.

**Design.** The control's params verbatim with `max_neurons` 250 -> 500 (the only diff). `max_neurons` is the
per-cluster cap. The worker derives the grid neuron list from min/max when `neurons_grid` is unset:
`[5, 100, 200, 300, 400, 500]` x `[80]` (was `[5, 50, 100, 150, 200, 250]`), so the grid reaches 500 x 500. Same two
experiments (Grid Search -> GA Neurons). Note: the GA Neurons mutation step scales with the band
(`round(0.1 * (min + max))` = 51 instead of 26), which is part of this change.

**Cost.** About 2.6M cells (the old quantile-80b control's footprint). Training cost about 2-2.6x per genome, so
**about 25-40 h per flow, 75-115 h for n=3.**

## Queue order

Appended at the end of the queue (after IDS-70 B 7595/7596 and IDS-71 7597-7599), arm-major: IDS-78 s1, s2, s3, then
IDS-79 s1, s2, s3. IDS-79 is to run after IDS-78 and IDS-76. Because IDS-76 cannot be created yet, the IDS-79 bodies
are written (`docs/ids78_79_flows.json`) but held, to be queued with `scripts/queue_ids78_79_unswr.py --queue ids79`
once Luiz decides on IDS-76.

Scripts: `scripts/queue_ids78_79_unswr.py` (bodies derived from the read-only DB, asserted against the control,
`KNOWN_PARAMS`-checked, POST /api/flows, verified, then PATCH queued).

## Addendum 23/11/2026 ~21:10 EST: IDS-76 built (Luiz, 23/11 ~20:45 EST)

Luiz asked to build fold rotation, deploy it at a flow boundary, and queue IDS-76 and IDS-79 after IDS-78
(7633-7635). This addendum fixes the design before any code is written. The status line in the IDS-76 section
above ("not implementable on ABI 29") is superseded by this section. Plane IDS-76 was not read here (no Plane access
from this session); the design below follows the prereg hypothesis and the IDS-77 numbers.

### Design: `ids_fold_rotation` (worker ABI 30)

- **Param.** `ids_fold_rotation`: bool, default `false`, registered in `KNOWN_PARAMS`. Only `true`/`false` are
  accepted. `true` needs in-memory storage (not streaming), flat classification (not hierarchical),
  `ids_k_folds > 1` and `ids_offspring_folds = "all"`. Any other combination is refused at evaluator construction.
- **Partition per generation.** Generation `g` of a GA stage scores every genome on partition `P(g)`. `P(0)` is the
  partition the cache builds today (seed = flow seed). For `g >= 1`, `P(g)` is re-drawn in Rust with the same
  stratified construction (`create_stratified_folds`: shuffle within class, deal round-robin, shuffle within fold,
  per-fold negative seeds), seeded by `ram_core::counter_rng::draw_u64(flow seed, g, ...)` on a dedicated
  coordinate. `P(g)` is a function of (flow seed, g) only, so runs are reproducible and a resumed run re-draws the
  same partitions. Each `P(g)` is still a proper 5-fold CV partition of the TRAIN rows: train on 4 folds, score the
  held-out 5th, average the 5. VAL and TEST are never touched; the test checkpoints train on the full TRAIN set and
  do not use folds.
- **Carried genomes are re-scored.** Fitness is frozen per genome, so after a re-draw the carried genomes would
  hold scores from `P(g-1)` while the offspring hold scores from `P(g)`. At the start of generation `g >= 1` the GA
  (1) selects the elites as today, on the population's `P(g-1)` scores (the whole population is on one partition),
  (2) re-draws the partition, (3) re-scores the elites and the incumbent on `P(g)` with the same call that scores
  offspring and immigrants (`evaluate_offspring_batch`, the 5-fold CV mean). The mu+lambda pool (elites +
  offspring + immigrants) is then on one partition, and the "new incumbent" test compares pool[0] with the
  incumbent re-scored on the same partition. Tournament parent selection keeps the `P(g-1)` scores (it only picks
  parents). Non-elite population members are not re-scored: under mu+lambda only the elites enter the next pool.
- **Cost.** 10 elites + 1 incumbent re-scored per generation vs 67 offspring evaluated (`population_size` 50,
  `fitness_percentile` 0.75): about +16 % per generation.
- **Fitness must be absolute.** The incumbent re-score is only meaningful when a genome's score does not depend on
  the pool, so rotation requires `fitness_aggregation = "desirability"` (the S&P cohort's) and refuses otherwise.
- **Patience.** Magnitude-aware patience watches the incumbent's F1/FPR, now re-scored each generation. Its
  watermarks ratchet across partitions, so partition noise can only raise the bar (max of noisy values) and tends
  to stop the stage earlier, not later. This is a known difference from the control, reported with the readout.
- **Stage end.** After the GA stage the cache is restored to `P(0)`, so a later phase starts from today's folds.
- **Flag off is unchanged.** With the flag off the new Rust method is never called and the GA hook returns
  immediately, so folds, fitness and results are today's. Proven before the swap by (a) a Rust test that `P(0)`
  equals the construction-time folds (rows, targets, negatives) and a re-draw followed by `g = 0` restores them byte
  for byte, (b) a Python pin run twice, once on ABI 29 (installed wheel + current tree) and once on ABI 30 (new
  wheel + new tree): k-fold CV means, Rust offspring search and a multi-generation GA run on a synthetic dataset,
  compared exactly. A unit test checks that ON re-draws a valid stratified partition per generation (disjoint,
  covers every TRAIN row, class counts per fold within 1, deterministic per (seed, g), different across g).

### Flows that cross the ABI boundary

The worker swap (ABI 29 -> 30) happens at the next flow boundary after 7594. 7595-7599 (IDS-70 B, IDS-71) and IDS-78
7633-7635 were created on ABI 29 and may run on ABI 30. This is valid because ABI 30 only adds a method and two
cache fields that are read only when `ids_fold_rotation` is true; none of these flows sets it, and the pin above
shows the flag-off path gives identical folds, fitness and GA trajectories on both wheels. The test-result cache
pins the worker ABI, so those flows re-compute their test cells instead of reusing ABI-29 cache entries (cost only).

### IDS-76 arm

- **Recipe.** Clones of the control 7591 / 7592 / 7593 (same seeds r20421-23, params verbatim) plus
  `ids_fold_rotation: true`, same two experiments (Grid Search -> GA Neurons).
- **Why a full clone and not a seeded GA phase.** Rotation acts inside GA Neurons from generation 1, so the arm must
  run GA Neurons from the control's own start. The grid does not use rotation and is deterministic (OI training,
  counter-RNG GA seeds since ABI 14), about 3 minutes per flow; it reproduces the control's grid exactly, and GA
  Neurons generation 0 is on `P(0)`, identical to the control. The arm is a new recipe (rotation on), so this is not
  a repeat of a flown recipe on the same seed; the shared prefix is what makes the pairing exact.
- **Built-in fidelity check.** The arm's `after-grid` test cells must equal the control's. If not, the pairing is
  flagged before any readout.
- **Expected.** <= +0.1 pp TEST F1 (IDS-77: fold-specific share of the GA edge is 0.05-0.10). Same decision rule and
  reporting cell as the other arms (GA Neurons, `final`, `best_f1`, `margin_val_cal`, TEST).
- **Cost.** About the control's 10-15 h per flow plus about 16 % per GA generation, so roughly 35-50 h for n=3.

### Queue order (revised)

IDS-78 7633-7635 (queued) -> IDS-76 s1, s2, s3 -> IDS-79 s1, s2, s3, so flow ids are IDS-78 < IDS-76 < IDS-79.
IDS-76 and IDS-79 are created only after the swap is verified, with `scripts/queue_ids78_79_unswr.py --queue ids76`
then `--queue ids79` (their descriptions say worker ABI 30).
