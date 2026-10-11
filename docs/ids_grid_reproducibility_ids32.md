# IDS-32: the IDS grid search does not reproduce from the flow seed

Investigation date: 11/10/2026. This was investigation only: no source was changed, no flow was created, and the database was opened read-only.

## Verdict

**Root cause (primary):** the worker builds every IDS `IDSEvaluator` **without a `seed=`**.
`BaseEvaluator` therefore falls back to a **wall-clock seed**, and that seed drives how the Rust
`IDSCache` splits the 80% train set into the 5 stratified K-fold parts. Every flow therefore scores
its genomes on a different, randomly drawn fold partition. The search fitness (the 5-fold average) differs
by about 1e-3 CE and 0.1 pp acc for the *same genome*. The top grid configs sit about 1e-3 CE apart, so the grid
ranking reshuffles. The reshuffle then changes which random connections the expansion variants get. The result is a
different 50-genome population and a different winner.

Genome *generation* is NOT the cause. The grid RNG is seeded from the flow seed and is consumed in a
deterministic order. The Rust train and score engine is deterministic across processes for the shapes in this grid
(b ≥ 16), given the same partition. A second, smaller engine defect exists at b ∈ {8, 12} (§4). It does not
explain this pair.

## 1. Evidence from 6835, 6836 and 6430 (all seed 20419, same grid config)

**Same genome, different score.** All 33 genomes shared by 6835 and 6836 (identical connections, by hash)
have DIFFERENT `cached_metrics` (none of the 33 match bit for bit). The CE delta is up to 6e-4. Examples:

```
hash      (pos6835, n, b)   pos6836   ce 6835   ce 6836   f1 6835  f1 6836
36003547  (2, 400, 34)      1         0.133770  0.133645  0.91771  0.91798
691d0736  (12, 400, 32)     5         0.134555  0.133979  0.91747  0.91841
f6fd4988  (15, 400, 34)     3         0.134384  0.133788  0.91999  0.92057
1f5d4fe1  (43, 100, 32)     36        0.139378  0.138634  0.92101  0.91930
```

Genome overlap by hash: 6835∩6836 = 33/50, 6430∩6835 = 24/50, 6430∩6836 = 31/50, and 22 genomes are shared by all three.

**Every one of the 54 grid configs scores differently across the 3 flows** (worker log `[k/54]` lines, 54/54
DIFF). This includes n=5 b=4 (CE 0.5232 / 0.5232 / 0.5229). The top-12 ranking reorders:

```
rank  6430 (ABI14)  6835 (ABI15)  6836 (ABI15)
1     400x34        400x34        400x34
2     200x34        300x34        200x34
3     300x32        400x32        400x32
4     300x34        300x32        300x34
5     400x32        200x34        300x32
...   (rank 10-12 also swap: 200x32 / 500x28 / 100x32)
```

**Why a rank swap changes genomes that were never re-scored.** `GenericGridSearch._expand_population`
(`src/wnn/ram/strategies/connectivity/generic_grid_search.py:105-127`) walks the top-K in RANK order.
`_make_variant` draws fresh connections from the shared grid RNG (`grid_search.py:120`, `self._rng.randint`).
When two configs swap rank, every variant drawn after the swap point gets a different connection draw.
The 54 config originals still match, because they are drawn in enumeration order (`generic_grid_search.py:105`).
This is why roughly 2/3 of the genomes overlap and the rest do not.

## 2. Code path (worker → grid → eval → selection)

1. `worker.py:1274`: `seed = params.get("seed", 42)` is in scope, but
2. `worker.py:1427` (`optimizer_eval = IDSEvaluator(...)`) and `:1454` (test_eval) pass **no `seed=`**.
   The hierarchical path at `:1524, :1528, :1541, :1546, :1561` is the same. Only `run_seed` (the QSR/PLN coin) is threaded.
   Blame: the constructor has looked like this since `4c434efd9` (10/03/2026).
   The legacy CLI `src/wnn/ids/train.py:285/295/…` DOES pass `seed=cfg.seed`. Only the worker is affected.
3. `ram/architecture/base_evaluator.py:145-147`: `if seed is None: seed = int(time.time()*1000) % 2**32`.
4. `ram/architecture/ids_evaluator.py:363`: `IDSCacheWrapper.new_from_numpy(..., seed=self._seed, ...)`.
5. `accelerator/ids_cache.rs:368-375` → `create_stratified_subsets(..., seed)` (`:574-638`).
   `StdRng::seed_from_u64(seed + 50)` shuffles each class and round-robins the rows into the 5 parts.
   This draw IS the K-fold partition.
6. Grid: `_IDSGridSearchCore._eval_one` → `IDSEvaluator.evaluate_batch` (`ids_evaluator.py:558-597`,
   `k_folds=5, kfold_per_gen=5`). It averages `evaluate_genomes_kfold_hybrid` over folds 0..4
   (`ids_cache.rs:840-898`: train on 4 parts, score the 5th). The fitness therefore depends entirely on the partition.
7. Rank → expand (rank-ordered RNG draws) → re-rank → trim 55 → 50 (`generic_grid_search.py:139-158`).

The VALCACHE log line prints `seed=20419` because it is built from the flow params. It does NOT reflect the
evaluator's seed, so it hides the bug.

## 3. Offline proof (real UNSW-NB15 temporal_3way top20 16b; RAYON_NUM_THREADS=4; worker untouched)

The test built the evaluator exactly as the worker does (QUAD, nsr 0.25, balanced, single-cluster, same fitness
weights). It then scored the first 3 genomes of 6835's grid `final_population` in 4 separate processes:

```
evaluator seed        g0 200x34 ce   g1 300x34 ce   g2 400x34 ce
None (728477362)      0.134116       0.133930       0.133683
None (728484624)      0.133417       0.133441       0.133627
20419                 0.134073       0.133535       0.133522
20419 (new process)   0.134073       0.133535       0.133522   <- bit-identical
6835 stored           0.133100       0.133432       0.133770
```

With a fixed seed the result is bit-identical across processes. With `seed=None` it moves by up to 7e-4 CE, the same size
as the production deltas. Synthetic tests at production size (175k × 288 bits, thermometer-coded) confirm the same thing.
Sequential, cross-process and 4-thread-concurrent evals all give bit-identical results for every b ∈ {4, 16, 20, 28, 34} × n ∈ {5…500}.
So the grid's `ThreadPoolExecutor` concurrency and `WNN_HYBRID_SPEED_RATIO` do **not** cause it. The hybrid split needs
a batch ≥ 4, and the grid evaluates 1 genome per call. Candidates (a) and (c) are ruled out. (b) is confirmed, and
it comes from the fold partition, not from GPU math.

## 4. Secondary defect: the train/score engine gives different results for the same input at b ∈ {8, 12}

The same genome evaluated repeatedly *in one process*, with a fixed cache, gives different results when b = 8
(all n ≥ 5 on random data, n ≥ 100 on thermometer data) or b = 12 (some n). Examples: n=100 b=8 CE
1.0315 / 1.0264 / 1.0207 on 3 back-to-back calls. b = 4 and b ≥ 16 are always exact.
Both routes go through `train_single_via_marker` → `marker_train::batched_train_offspring`
(`adaptive/eval_single.rs:545`, path-2 marker hashtable). Its parity was only ever verified for b ≥ 16
(doc comment at `eval_single.rs:533`). The suspect is a marker-hashtable insert/OI-counter race at high load factor
(capacity at b=8 is 512 slots for 256 addresses). This needs a Rust investigation (follow-up IDS-32b). It does not
affect the 6835/6836 grid winners (top-K is b ≥ 28), but it does affect any low-bit cohort.

## 5. Blast radius

- **Every worker IDS flow since 10/03/2026, in every phase.** The grid, GA neurons and every GA generation are all
  scored on a wall-clock partition. The GA offspring RNG fix (ABI 14, `project_ids_ga_offspring_seed_wallclock`)
  made the *operators* reproducible, but the GA fitness landscape is still per-run random. The GA is hit twice:
  through its seed population (the grid output) and through its own fitness.
- **Held-out numbers are NOT biased.** The test evaluator trains on the full 80% and scores val/test. Its subsets
  use `seed` only for negative sampling, which is 0 in single-cluster binary and exhaustive in default multiclass.
  So the held-out score of a given genome does not depend on the partition. Every reported number is an honest
  draw. The thing that is not reproducible is *which genome* gets selected.
- **Paired-seed A/Bs** (the GAT-* quick tests, IDS-26, the IDSZ/CE20 "paired seeds" n=5, the IDS-3/4/5/7 n=100 tandem):
  these are still **valid**. Each pair is still an iid draw and nothing is biased. But they are **not common-random-number paired**
  at the fold level, so the variance reduction that pairing was meant to buy is mostly not there. Arm differences
  carry the full fold-partition noise. Already-significant results stand (they were tested on noisier pairs than
  assumed), and null results may be under-powered. The "sigma_seed" estimates include partition noise.
  No paper number has to be withdrawn. Any **reproducibility claim** (re-running seed X gives result Y) is false until the fix lands.
- **Streaming flows (46M / ciciot streaming):** the Python fold permutation at `ids_evaluator.py:434` uses
  `(seed if seed is not None else 42) + 7777` with the ctor's `seed`, which is None. So every streaming flow uses the SAME
  partition no matter what its flow seed is. That is deterministic, but it does not vary across a cohort's seeds. The streaming
  neuron-sample RNG at `:923` (`rng_seed=self._seed`) is wall-clock.
- **Latent:** `_kfold_rotation_offset` (`ids_evaluator.py:561-564`) is shared mutable state. It is harmless at
  `kfold_per_gen == k_folds` (the window is always 0..4), but with `kfold_per_gen < k_folds` the fold window would
  depend on the call count and on how the 4 grid threads interleave.

## 6. Minimal fix (proposal, not implemented)

1. `worker.py`: pass `seed=int(seed)` to every `IDSEvaluator(...)` (`:1427, :1454, :1524, :1528, :1541,
   :1546, :1561`). That is 7 keyword arguments.
2. Make the bug impossible to repeat: `BaseEvaluator.__init__` (`base_evaluator.py:145`) should **raise** on
   `seed=None`, or at least warn loudly, instead of silently using the wall clock. Audit the other
   `time.time()` fallbacks (`base_evaluator.py:444, 575`; `ids_evaluator.py:1231, 1306`) the same way.
3. Optional hardening of the grid: key each expansion variant's connection seed by `(config, variant_idx)`
   (for example, pre-draw them in enumeration order). Then a near-tie rank swap can no longer re-randomise the other variants.
4. Follow-up IDS-32b: find and fix the b ∈ {8, 12} engine nondeterminism (§4).

**Does it change results for future flows?** Yes, but only as a re-draw, not as a change in distribution.
The partition becomes a deterministic function of the flow seed. It is still a seeded random stratified 5-fold
split, so post-fix cohorts stay statistically comparable to pre-fix ones (same estimator, different realisation).
They will not be bit-identical to any earlier flow. Same-seed A/B arms will then share the exact partition (true
CRN pairing), and a rerun of seed X will reproduce the grid. Streaming flows move from the constant 42+7777
partition to seed+7777. The fix only touches Python, so it needs no wheel rebuild and no worker swap. It does need
a flow_runner restart at worker-idle, per the never-edit-live-imported-source rule. Verification: run two tiny-budget
flows with the same seed. Their grid `final_population` hashes and `[k/54]` scores must be 54/54 and 50/50 identical.
The offline script in §3 already shows the fixed-seed path is bit-exact on real data.
