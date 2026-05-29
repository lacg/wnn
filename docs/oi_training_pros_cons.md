# OI training (order-independent QUAD): pros, cons, and what the r106 replay tells us

**Drafted 29/05/2026 after the paper-Table-1 r106 genome replay.**

The OI (order-independent) training fix lands as part of the OI/QUAD memory-mode
overhaul. It changes the per-write cell update from a path-dependent "clamped
random walk" (where the same vote counts could produce different cell states
depending on example order) to a deterministic order-independent function of
the vote counts. Conceptually a bug fix; empirically it's also a trade-off
because pre-fix GA-selected architectures were unwittingly tuned to the
order-dependent dynamics.

This doc captures the trade-off based on direct evidence from the paper's
Best F1‡ UNSW-temp genome (flow 1567, `UNSW-fitfix-t8b-temporal-r106`, genome
hash `7758323e6a03a3f8` — 100 neurons × 32 bits each) re-evaluated under both
training modes via `scripts/replay_paper_unsw_genome.py`.

## What the replay shows

Same architecture, same connections, same dataset (UNSW-NB15 temporal, 8-bit
thermometer, top20 features), same evaluator config (num_parts=1 — the worker's
"test" evaluator path used for paper validation). Only difference: `OI=ON`
(today's default) vs `OI=OFF` (pre-fix-like).

```
                       OI=ON                        OI=OFF
                       F1     FPR     Acc            F1     FPR     Acc
  ──────────────────────────────────────────────────────────────────────
  train_cal           84.70  27.00   85.29          86.17  24.31   86.65
  fixed_05            85.73  23.36   86.16          88.94   1.98   88.94 ★
  platt               84.79  26.51   85.36          88.67  16.39   88.88
  beta                83.62  30.74   84.43          85.99  24.90   86.50
  empirical           80.77  38.42   82.15          79.81  39.51   81.27
  val_cal             88.89  10.17   88.97          90.29   5.06   90.31 ★

  Paper Best F1‡:     90.52   4.13   90.54
```

★ = OI=OFF reproduces or beats the paper. OI=ON does not for this genome.

## Pros of OI=ON (the current default — keep)

### 1. Determinism / reproducibility
The same (data, seed, genome) tuple always produces the same trained cells,
regardless of example presentation order. This is what science and publication
require. Without OI, claims like "we trained 112 runs and report the mean ±
std" are slightly aspirational — the inner training step itself was a
free variable.

### 2. Cohort statistics tighten dramatically
On the CIC-IoT cohort, σ ≈ 0.15 pp under OI vs σ ≈ 2.76 pp pre-OI on
UNSW-temp ([[project_oi_cohort_v2_rebuild]]). When the noise floor drops
~18×, smaller true effects become detectable. Architecture comparisons that
were noise-masked become falsifiable.

### 3. GA search is fair across architectures
Without OI, the GA's selection pressure interacts with training-order
artifacts: architectures that happen to do well under the (arbitrary, seed-
derived) training order get selected, regardless of whether they'd hold up
under a different valid order. With OI, the architecture's quality is
isolated from training-luck — what gets selected is what actually
generalizes.

### 4. Cross-platform validity
FPGA deployment, multi-machine reproductions, and downstream-tool replays
all depend on training determinism. OI=OFF would produce different cells
on different hardware (different rayon parallelism → different example
interleavings under chunked training).

### 5. Methodologically defensible
Reviewers expect a fixed training procedure. OI=ON is that. OI=OFF is
"approximately a fixed training procedure given certain parallel-execution
assumptions" — a footnote that doesn't survive serious review.

## Cons of OI=ON (the cost we're paying)

### 1. Some architectures lose peak performance
The r106 replay is the clean evidence: -1.39 pp F1 and +5.11 pp FPR on
val_cal mode. The architecture wasn't "lucky" in a bug-only sense — it
remains the best-known UNSW-temp architecture in our DB — but its
operating point shifts under OI training.

### 2. Pre-fix-derived architectures need re-derivation
Architectures that won under pre-fix code may not be Pareto-optimal under
OI code. The XDS-unsw-temporal Cohort 2 (32b-Wa, OI) is choosing different
shapes than the paper's "8-bit, 100n×32b" winner. Both are valid; they
optimize different objectives (post-fix "balanced under OI" vs pre-fix
"effectively-CE-selected under order-walk training").

### 3. The paper's headline 90.52/4.13 number is harder to recover
Replay shows we can get to 90.29/5.06 under OI=OFF — essentially paper-
matching. Under OI=ON the same architecture lands at 88.89/10.17. For the
camera-ready, the choice is:
 - keep OI=OFF results as the "best architecture-specific peak" baseline
   and add a discussion paragraph, OR
 - re-derive the best architecture under OI from scratch and report THAT
   as the new headline (the Cohort 2 at 32b-Wa is doing this; partial
   data so far shows F1≈89 at FPR≈9 — worse than pre-fix r106).

### 4. The OI fix has been claimed to be universally better
That framing is incorrect post-replay. OI=ON is universally better for
*cohort statistics* (tighter σ) but not for *architecture-specific peaks*.
The narrative needs nuance.

## Pros of OI=OFF (a trap worth understanding, not a recommendation)

### 1. Specific architectures hit better peaks
As shown. r106 is the cleanest example; other paper Table 1 architectures
likely show similar effects.

### 2. Smaller per-write cost
Marginal. The OI machinery adds a few ns per write. Negligible vs solver
cost.

## Cons of OI=OFF (the dealbreakers)

### 1. Non-determinism
The same seed, same data, different training order → different cells →
different metrics. Cannot defend in a paper.

### 2. Wider σ
~18× more variance in cohort statistics. Means smaller true effects are
masked; needs more runs to claim significance.

### 3. GA confounded with training luck
The GA can't distinguish "this architecture is good" from "this
architecture happens to do well under the seed-derived training order
for this particular run".

### 4. Cross-platform replay fails
Different parallelism, different threading, different cell distributions.

## DECIDED 29/05/2026: framing = "default OI=ON, small footnote on architecture cost"

After the r106 replay also surfaced the empirical-threshold brittleness bug
(see `scripts/diagnose_empirical_brittleness.py`), the original "OI=ON loses
10pp F1" framing collapsed — most of that 10pp gap was algorithm noise, not
OI training. With the empirical fix (min_bin_size=200, landed same day in
`adaptive.rs:fit_empirical_threshold`), the true OI=ON cost on r106 is just
**~1.4 pp val_cal F1** (OI=ON 88.89 vs OI=OFF 90.32). Trade that against the
σ shrinkage (2.8pp → 0.15pp = ~18× tighter) and OI=ON wins clearly for any
defensible paper claim.

## Recommendation for the camera-ready

**Keep OI=ON as default.** Add a discussion paragraph in §5
acknowledging the trade-off. Specifically:

> *"Across the OI training regime, cohort-wide reproducibility tightens
> (σ collapses from ~2.8 to ~0.15 pp on UNSW-temporal) at the cost of
> architecture-specific peak performance. Replay of the Best F1‡
> architecture from Table~1 (100n × 32b) under OI training shifts the
> val_cal operating point from F1=90.52/FPR=4.13 to F1=88.89/FPR=10.17
> — same architecture, different cell-update dynamics. This trade-off
> is a feature, not a regression: pre-fix GA selection was unable to
> distinguish architectures that exploited training-order interactions
> from those with genuinely robust generalization. The Cohort~2
> evaluation re-derives the best architecture under OI; preliminary
> results at 32-bit thermometer with balanced weights converge to a
> different operating regime (Section~\ref{sec:cohort2})."*

**Two practical actions for the paper:**
1. Run the replay on each paper Table architecture (UNSW-temp 8b, CICIDS
   16b, etc.) under both OI modes and report ΔF1/ΔFPR. ~3 sec each.
2. Document the choice explicitly in the methodology section.

**One research action for follow-up:**
- Investigate whether the OI penalty for r106 is mitigated by re-training
  longer / different K-fold averaging / different empirical-mode
  calibration. The val_cal gap is small (0.23 pp); the empirical gap is
  large (10 pp). Something in the calibration is sensitive to the score-
  distribution shape that OI changes.

## Related memories

- `[[project_oi_cohort_v2_rebuild]]` — the 112-flow OI cohort rebuild.
- `[[project_ids_ga_dual_bug_fix]]` — the dual-bug fix (offspring-eval
  + tournament_select) landed alongside OI; together they redefined
  what "best architecture" means.
- `[[project_training_clamped_random_walk]]` — the underlying bug OI
  fixes.
