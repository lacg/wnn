# Controller GA — K-fold cross-validation design

## Problem

Current `ControllerEvaluator.evaluate_batch(genomes)` evaluates every genome on a single pool of N
random episodes (seeded). The GA's fitness signal is computed from those exact N
episodes, generation after generation. **Genomes can overfit to that specific episode
pool**: an arch/cell configuration that's lucky on the chosen initial conditions scores
high, propagates via elitism, dominates the population — but fails when re-evaluated on
fresh seeds.

Observed evidence (Plan A v1, Stage 1 final):
- GA-eval (training episodes):  err = 12.57° / stable = 15%
- Final re-eval (fresh episodes):  err = 16.22° / stable = 5%

Gap: **3.65° err / 10pp stable**. With per-episode std ~3° and N=20, expected sampling
noise is ~0.7° — so this is ~5σ overfit, not noise.

## Solution: K-fold rotation

Pre-generate K disjoint episode pools at evaluator init. Each generation evaluates all
genomes on a different pool, rotating cyclically. Across K consecutive generations the
GA's selection pressure averages over K × N = K·N distinct episodes — wider coverage,
no single-pool memorization possible.

### Design choices

#### Fold count K
- K=5 mirrors the IDS convention (`ids_k_folds=5`, see CLAUDE.md).
- With N=40 per pool, total coverage = 200 episodes across all folds.
- K=3 is acceptable for fast probes; K=10 is overkill.

#### Per-genome vs per-batch fold
- **Per-batch (chosen)**: every genome in the gen's batch uses the SAME pool. Fair
  comparison within the gen; pool rotates between gens.
- **Per-genome**: each genome in the batch gets a different pool. Fairer across genomes
  but breaks within-gen ranking — unsuitable for elitism-based selection.

#### Aggregation
- **Pure rotation** (default): gen N evaluates on pool (N mod K). Fitness =
  mean of N episodes in that pool. Elite selection per gen as today.
  *Pros*: K× cheaper than mean-over-K. Per-gen ranking still well-defined.
  *Cons*: Across-gen comparisons become noisier (different pools yield different
  absolute scores).
- **Validation gens** (recommended): every K-th gen runs a "validation" eval — mean
  over ALL K pools. This validation score is used to recompute elite ranking for that
  gen, exposing any single-pool overfits before they propagate.

#### Episode reproducibility
- The K pools are seeded deterministically at evaluator init: `pool_seed[k] =
  hash((base_seed, k))`. Same base_seed → same pools across runs. Critical for
  reproducibility.

## Implementation

### File: `src/wnn/control/evaluator.py`

```python
@dataclass
class ControllerEvalConfig:
    num_eval_episodes: int = 20          # N: episodes per pool
    num_eval_folds: int = 1              # K: number of pools (1 = current behavior)
    validation_every_k_gens: int = 5     # how often to run mean-over-K validation
    base_seed: int = 0

class ControllerEvaluator:
    def __init__(self, spec, num_eval_episodes, seed, episode_config, ...,
                 num_eval_folds: int = 1, validation_every_k_gens: int = 5):
        self.num_eval_folds = num_eval_folds
        self._pool_seeds = [
            hash((seed, k)) & 0xFFFFFFFF for k in range(num_eval_folds)
        ]
        self._gen_counter = 0
        # ... existing init ...

    def evaluate_batch(self, genomes, *, validation=False):
        """Evaluate genomes on the current gen's fold (or all folds if validation=True)."""
        if validation or self.num_eval_folds == 1:
            pools = self._pool_seeds   # use all pools for validation
        else:
            pools = [self._pool_seeds[self._gen_counter % self.num_eval_folds]]
            self._gen_counter += 1

        all_metrics = []
        for pool_seed in pools:
            metrics = self._eval_on_pool(genomes, pool_seed)
            all_metrics.append(metrics)

        # Average across pools (for validation, or trivially for single-pool gens)
        return [_mean_metrics(per_pool) for per_pool in zip(*all_metrics)]
```

### File: `src/wnn/control/ga_strategy.py` (or wherever the GA loop lives)

```python
# Inside the per-gen loop:
is_validation_gen = (gen % cfg.validation_every_k_gens == 0)
batch_metrics = ev.evaluate_batch(population, validation=is_validation_gen)
# Use batch_metrics for fitness ranking as today.
# If validation gen, also use the (more accurate) score to overrule
# any elite whose validation rank dropped > threshold.
```

### File: `tests/run_phased_ga.py`

Add CLI flags:
```python
ap.add_argument("--num-eval-folds", type=int, default=1,
                help="K episode pools for fold rotation (1 = current behavior).")
ap.add_argument("--validation-every-k-gens", type=int, default=5,
                help="Run mean-over-K validation eval every N gens (default 5).")
```

Plumb through to `ControllerEvaluator` constructor.

## Compute cost

Per gen:
- Pure rotation: 1 pool eval = N episodes (same as today)
- Validation gen: K pool evals = K × N episodes

Total across G gens (K=5, validation every 5 gens):
- Pure rotation gens: (G - G/5) × N episodes
- Validation gens: (G/5) × K × N episodes = (G/5) × 5N = G·N episodes
- **Total: G·N + G·N = 2·G·N episodes**

**Net overhead: 2× over current eval=N**, even though the GA effectively saw K×N distinct
episodes across the run. This is dramatically cheaper than mean-over-K every gen
(which would be K× more expensive).

## Plan A v2 recipe (recommendation)

```bash
python -u tests/run_phased_ga.py \
  --num-eval-folds 5 \
  --validation-every-k-gens 5 \
  --eval-episodes 40 \
  --pop 100 \
  --neurons-gens 200 --bits-gens 200 --conns-gens 200 --memory-gens 400 \
  --neurons-patience 10 --bits-patience 10 --conns-patience 10 --memory-patience 20 \
  --steps 500 --tilt 15 --universe-episodes 3 \
  --rg-rounds 3 --rg-episodes-per-round 6 --rg-eval-episodes 5 \
  --fit-weight-err-sq 0.30 --fit-weight-stable 0.50 \
  --fit-weight-jerk   0.10 --fit-weight-mono   0.10 \
  --base-seed 20260601 \
  --save-winner logs/controller/planAB/winner_planAv2.pkl
```

Compared to Plan A v1:
- pop: 200 → 100 (2× faster per gen)
- eval-episodes: 20 → 40 (2× slower per gen) — net same per-gen cost
- num-eval-folds: 1 → 5 (2× overhead from validation gens)
- **Total wall time: ~2× longer than v1, but with K=5 × N=40 = 200 episode coverage and overfit-protected fitness**

For a faster probe (smoke test): K=3, eval=30, pop=60, gens 100/stage → ~3-4h total.

## Smoke test plan

Before launching Plan A v2 in earnest:
1. Run `python tests/test_kfold_evaluator.py` (new file): asserts pool seeds are
   reproducible across runs and that pure-rotation/validation modes return correct
   shapes.
2. Run a tiny phased-GA: `--num-eval-folds 3 --eval-episodes 10 --pop 20 --neurons-gens 10`.
   Verify: per-gen log shows fold index, validation gens show all-K-pool average,
   wall time ~3-5 min.
3. Compare to a `--num-eval-folds 1` baseline at same other knobs. Verify the K=3
   run produces a more-stable elite (smaller train/re-eval gap).

## Migration

- Default `--num-eval-folds 1` → behavior identical to current code. No regression.
- All existing logs/runs/pickles remain compatible.
- `ControllerEvaluator.evaluate_batch` API is unchanged unless `validation=True` is
  passed explicitly, so external callers don't break.

## Open questions

1. **Cell warm-start across folds**: when a genome's cells are written via reward-gated
   inner training, do we warm-start cells from one pool's training to the next pool's
   eval? Current code resets cells per evaluate_batch call. Decision: keep reset
   behavior (each genome is evaluated from cell-blank state per pool). Pre-validated
   cells would introduce pool-coupling bias.
2. **K-fold for the universe-recording (Stage 4 Memory) phase**: the QSR universe is
   recorded over a single seed. Should we record K universes and union them? Probably
   overkill — the universe is just the address space, not the cell values. Leave at
   K=1 for Memory stage.

## Status

DESIGN ONLY — not implemented. Implement before Plan A v2 launch. Estimated effort:
6-10 hours code + 2-3 hours smoke testing.
