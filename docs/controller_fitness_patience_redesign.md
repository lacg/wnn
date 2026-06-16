# Controller fitness & patience redesign — magnitude awareness

**Status:** design / not implemented (do AFTER the current 50/100 + 100n200m cohort).
**Motivation:** the controller harmonic-rank fitness `WHM = Σw / Σ(w_i/rank_i)` is
**magnitude-blind** — it ranks genomes, so a real jump (seed2 gen-23: stable
20%→70%, err 10.94°→4.03°) barely moved the objective (−0.0004). Two costs:
(1) the **patience/early-stopper** watches the WHM, so genuine gains don't reset
it → premature early-stop + "improvements look rare"; (2) weak magnitude gradient
in selection (secondary — rank-selection *did* find the jump). jerk/mono ARE
plumbed and ranked in clean runs (the "RESERVED/None" docstring was stale, fixed
981e93e2); the magnitude-blindness is the rank scheme itself, not a dead metric.

Two options, **not mutually exclusive**: (a) is the cheap, comparable, do-now fix
for the *symptom*; (b) is the deep fix that re-baselines. (b) subsumes (a) — a
magnitude-aware *fitness* makes patience magnitude-aware for free.

---

## (a) Magnitude-scaled patience — keep rank selection, fix the early-stopper

Leave the harmonic-rank WHM driving **selection** (so selection + cross-run
comparison are UNCHANGED). Only change what the early-stopper watches and how
much patience a check recovers — scaled by the *magnitude* of real improvement.

### Current (EarlyStoppingTracker)
- improvement → `counter = max(0, counter − 1)`  (recover exactly 1)
- no improvement → `counter += 1`
- stop when `counter ≥ patience`  (remaining = patience − counter; 5/5 = full)

### Proposed
Track the best genome's **err°** and **stable%** (magnitude metrics, not WHM).
Each check (every `check_interval` gens), vs the best at the previous check:

```
ρ_err = err_prev / max(err_cur, ε_err)            # >1 when err drops; halving → 2
ρ_stb = (stb_cur + s0) / (stb_prev + s0)          # additive s0≈0.05 tames stb=0; 20→70% → ~3.5
ρ     = max(ρ_err, ρ_stb)                          # biggest real gain drives recovery
ρ     = min(ρ, ρ_cap)                               # ρ_cap ≈ patience_max, guards a fluke ratio

if ρ ≥ 1 + δ:        counter = max(0, counter − ρ)   # RECOVER by the ratio (your formula)
else:                counter += 1                     # drain by 1 (the floor)
stop when counter ≥ patience
```

This is exactly your idea: floor recovery is effectively "decrease 1" (drain),
err halved → recover `1/0.5 = 2`, stable 20→70% (3.5×) → recover `3.5` capped at
`patience_max`. `counter` is kept as a float; "remaining = patience − counter"
just displays the same.

- **Combine rule:** `max(ρ_err, ρ_stb)` — most generous; any big single-metric
  gain keeps the search alive. (Alt: weighted blend `w·log ρ`; max is simpler.)
- **δ (noise gate ≈0.05):** below it, treat as no-improvement so eval jitter
  doesn't fake-recover. K-fold eval already dampens this.
- **Edge cases:** `ε_err≈0.5°` floor (avoid div-0 / huge ratios near 0); `ρ<1`
  (got worse) → treat as no-improvement (`+1`); `ρ_cap` so one outlier check
  can't extend forever.

### Wiring / effort (small)
- `EarlyStoppingTracker.update(...)` gains `best_err_deg`, `best_stable` args
  (or take the best genome's `Metrics`). The GA loop already has the best
  genome's metrics — pass err°/stable% through.
- Keep the rank-WHM for selection → **selection identical → comparable** with the
  whole existing cohort + C10 sweep. Only the *early-stop timing* changes (by
  design — that's the fix).
- Gate behind `magnitude_aware_patience=True` (default off) so it's opt-in/testable.
- **Test:** synthetic trajectories — flat → drains to stop in `patience` checks;
  a gen-23-style jump → recovers proportionally; a worse check → drains.

### Risk: low. Selection untouched; only when-to-stop moves.

---

## (b) Value-based fitness — replace rank with weighted sum of shaped "goodness"

Score each genome on a weighted SUM of per-metric **goodness ∈ [0,1]** (1 = best),
each via a FIXED nonlinear transform — so the objective itself moves with
magnitude. Your fixed-transform framing sidesteps the per-generation min-max
normalization problem (no relativity reintroduced).

### Goodness transforms (1 = best, fixed & physically motivated)
```
g_err  = exp(−err° / τ_err)          τ_err ≈ 3–5°   # err 0→1.0, 3°→0.37, 5°→0.19, 10°→0.04, 180°→~0
g_stb  = stable²   (or exp((stb−1)/τ))              # convex → rewards the high end (20%→0.04, 70%→0.49, 100%→1.0)
g_jerk = exp(−jerk / τ_jerk)         τ_jerk = TBD   # MEASURE the range first
g_mono = exp(−mono / τ_mono)         τ_mono = TBD   # MEASURE the range first
```
- **err 0–180°**, we care about ≲5° and "closer to 0 much better" → exponential
  with τ≈3° does exactly that (steep reward near 0). ✓ your idea.
- **stable 0–100%, "closer to 100 much better"** → convex (`stable²` or exp) so a
  70% genome scores ~12× a 20% one. ✓
- **jerk/mono ranges unknown** → STEP 0 is to log their population distributions
  (a few-genome probe), then set τ from the measured spread.

### Combine — weights NEED NOT sum to 1 (your insight is correct)
```
F = w_err·g_err + w_stb·g_stb + w_jerk·g_jerk + w_mono·g_mono     # higher = better
```
For a weighted SUM, only weight *ratios* matter for ranking — absolute scale is
irrelevant. So `w_err=1, w_stb=2, w_jerk=0.5, w_mono=10` is fine ("mono matters
10× err"). **Caveat that bites:** influence = weight × the transform's
*sensitivity in the operating region*. If g_mono sits at ~0.99 and barely varies
across genomes, w_mono=10 still does little. So **weights and τ's must be
co-designed** (each metric is a 2-knob tune: scale × weight).

### Why this fixes it
A 20%→70% stable jump moves `g_stb` 0.04→0.49 → moves `F` a lot (vs the rank
nudge). Patience watching `F` becomes magnitude-aware automatically → (b) ⊃ (a).

### Costs / honest caveats
1. **Measure jerk/mono ranges first** (STEP 0) — can't set τ_jerk/τ_mono blind.
2. **Hyperparameters interact:** per metric tune both τ (sensitivity) and w
   (weight). Bounded (a small grid), not open-ended — but real.
3. **Breaks comparability:** changes selection → NOT comparable with any
   rank-based run (the cohort *and* the C10 sweep). New baseline; the weight
   sweep would need redoing in value-space (or accept fresh value-space weights).
4. **More eval-noise-sensitive** than rank (F moves directly with a noisy err°).
   Lean on K-fold averaging; maybe smooth.
5. New `FitnessCalculator` subclass, opt-in via calculator-type flag.

---

## Recommendation / sequencing
1. **(a) now** (next cohort): cheap, low-risk, comparable, kills the
   premature-early-stop symptom you caught. ~1 method + 2 passed values + a flag.
2. **(b) as a deliberate experiment** after: STEP 0 measure jerk/mono ranges →
   pick τ's + initial weights (your guesses) → new calculator behind a flag →
   small τ/weight sweep → if it beats rank-based on the held-out, adopt it.
3. If (b) lands, (a) becomes redundant (fitness magnitude-aware ⇒ patience too) —
   but (a) is worth doing first because it ships in days and doesn't reset baselines.
