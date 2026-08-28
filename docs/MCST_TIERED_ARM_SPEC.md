# MCST — support-tiered multiclass arm (spec, 27/08/2026)

One arm, three components, ONE pre-registered read. Ruled by Luiz 27/08/2026:
tiered bits + per-class score normalization carry the ACCURACY claim; tiered
neurons ride along as an EFFICIENCY-ONLY claim with **250n as the joint cap**.

## 0. Evidence this is built on (banked, do not re-derive)

From the MCSD-auto cohort (docs/ids_results.md §9) and the flow-5985 S1 grid:

- **Neurons are nearly inert on S1**: n=5→500 at b=34 buys +0.98pp acc and
  WORSE CE; n=5,b=34 had the best CE of all 54 grid configs. The GA parking
  every S1 cluster at min_neurons=5 was correct optimization of a flat surface.
- **Bits is ~55x the lever**: b=4→34 at n=5 buys +54.0pp acc.
- **The S0 grid lottery is a flat-surface artifact**: seed 20401's 600n gate
  cost 30x runtime and was the WORST of five gates held-out; 12-26n gates won.
- **No per-class decode normalization exists**: multi argmax compares a 5n
  cluster's score against a 205n cluster's score on one raw scale.
  (`class_weights` in wnn/ids/metrics.py is a REPORTING weight, not decode.)

Therefore: neurons = cost/variance control, bits = accuracy lever,
normalization = the decode fix that tiering makes MORE necessary, not less.

## 1. Sizing algorithm (support-tiered centres)

Let S_model = the model's OWN train-row count (S0: 175,341; S1: 119,341).
Centres derive from FULL train supports, never the k-fold subset (71,606).

**Neurons** (cap N=250 JOINT across S0+S1, floor n_floor=10):
1. raw_c = N * s_c / S_total over all 10 classes.
2. Classes with raw_c < n_floor get n_floor.
3. Renormalize the rest to spend N minus the floored total; iterate if
   scaling pushes another class under the floor.

**Bits** (floor 10, cap 34; 34b = "a neuron fed ALL training rows"):
  b_c = clamp( round( 34 * log2(s_c) / log2(S_model) ), 10, 34 )

**Hier budget split — CORRECTED 28/08 (Luiz).** The cap is a PLANNING budget
split ONCE at the beginning; it is NOT a winner cap and NOT a runtime ceiling.

- The joint cap is allocated across the FULL class set up front.
- S0 (binary gate) takes `[benign_share, sum(attack_shares)]`.
- S1 takes the per-attack shares of that SAME allocation.
- **`S1 = 250 − S0_winner` is WRONG and was removed**: a greedy S0 winner would
  starve S1 down to its 10n floors, which is precisely the failure this rule
  exists to prevent.
- **The winners MAY sum above the cap at the end.** That is intended, so no
  feasibility guard and no grid exclusions — n×1.0 is feasible by construction.

(Superseded: the earlier "S1 budget = 250 − winner" text and the ≤160 S0 guard.
The 5995 defect and its 41db56f1 fix both lived inside that wrong frame; the
frame itself is now replaced.)

## 2. Worked example — UNSW-NB15 temporal_3way (train supports)

10-class allocation, N=250, floor 10 (floors: Analysis 2000, Backdoor 1746,
Shellcode 1133, Worms 130 → 4x10=40; remaining 210 scaled by 0.8647):

```
class            support   neurons-centre   bits-centre(S1, S=119341)
Normal            56,000        69                (S0 benign cluster)
Generic           40,000        49                31
Exploits          33,393        41                30
Fuzzers           18,184        22                29
DoS               12,264        15                27
Reconnaissance    10,491        13                27
Analysis           2,000        10 (floor)        22
Backdoor           1,746        10 (floor)        22
Shellcode          1,133        10 (floor)        20
Worms                130        10 (floor)        14   <- Luiz predicted 10~15
                             ─────
                              249 ≈ 250 ✓
```

S0 gate (2 clusters): benign centre 69, attack centre 180 (sum of attack
allocations). With per-cluster multipliers {0.5, 0.75, 1.0} and the ≤160
guard, the feasible S0 grid is (0.5,0.5)=125, (0.75,0.5)=142, (1.0,0.5)=159
plus any other combo ≤160.

## 3. Grid design

Multipliers are **GLOBAL per-stage scalars** applied to every class centre —
per-class differentiation comes from the centres, NOT from independent
per-class grid axes (which would explode the grid).

- neurons: {0.5, 0.75, 1.0}  — down-only; **250 is the cap** (the earlier
  +50%/375 idea is superseded by Luiz 27/08: cap rules).
- bits:    {0.5, 0.75, 1.0, 1.25, 1.5}, per-class clamp [10, 34], dedupe.
- Grid = 3 x 5 = 15 configs/stage (vs 54 today).
- **Anti-lottery tiebreak**: among grid configs within epsilon (default 0.5%
  relative fitness) of the best, pick the SMALLEST total neurons. This is the
  efficiency claim acting at selection time; log the tie group.

GA phases keep per-cluster mutation as-is, but min/max neuron bounds become
per-cluster: [0.5*centre_c, 1.5*centre_c] clamped to [10, and the global cap
via a genome-total check in the mutation accept step]. Bits bounds likewise
[0.5*b_c, 1.5*b_c] clamp [10,34].

## 4. Per-class score normalization (the decode fix)

New decode mode **`argmax_classnorm`** (+ `margin_classnorm` variant),
Protocol v2 (calibrated on VAL, never test):

1. On VAL, per class c, collect the score distribution s_c over val rows;
   compute mu_c, sigma_c.
2. Decode: argmax_c z_c where z_c = (score_c - mu_c) / sigma_c
   (sigma floor 1e-6; margin variant thresholds top1-top2 z-gap at a
   val-F1-optimal tau).

This is the "trust a small-but-confident class" mechanism: a 10n Worms
cluster that fires far above its own val distribution beats a 49n Generic
cluster firing at its mean. It applies to flat-multi decode AND the hier S1
decode; the S0 gate keeps its binary calibration path.

Implementation: Rust (`multiclass_modes_from_scores` + the threshold-modes
enum) with the val pass supplying mu/sigma; Python decode-sweep registry in
`experiments/experiment.py`; dashboard mode column. Full-stack (Rule 6).

## 5. Cohort design + pre-registered read (READ ONCE)

- **One arm**: `MCST-unswt-quad-16b-hier-s{20401..20405}` — tiered centres +
  classnorm decode + 250n joint cap. 5 seeds, same as MCSD.
- Comparators: banked MCSD-auto hier + multi (ids_results.md §9). Paired by
  seed. NOTE the read is (tiering + classnorm) COMBINED vs baseline; if it
  wins, mechanism attribution needs a follow-up ablation (classnorm-only on
  uniform sizes) — do not claim which component did it from this cohort.
- **Primary: macro-F1 + benign-FPR.** weighted-F1 secondary. Per-class recall
  table MANDATORY. Never during-search k-fold numbers.
- **Neuron half is pre-registered EFFICIENCY-ONLY**: report total neurons,
  memory cells/bytes, wall-clock, and gate-size variance across seeds (the
  grid-lottery test: 5985's 608n outlier must not recur). NO accuracy claim
  may cite the neuron tiering.
- Also report the gate-FPR identity per seed (must stay exact).

## 6. Implementation checklist (full-stack)

1. `KNOWN_PARAMS`: `ids_tier_neurons` (bool), `ids_tier_neuron_cap` (int,
   250), `ids_tier_bits` (bool), `ids_score_norm` (enum: none|classnorm).
2. Python flow/grid builder: support-derived centres (from the loader's
   train-label counts, NOT hardcoded), multiplier grid, feasibility guard,
   epsilon-tiebreak.
3. ClusterGenome seeding from per-class centre lists (already supported);
   per-cluster GA bounds.
4. Rust: classnorm decode modes; rebuild WORKER wheel; ABI bump iff any
   existing signature changes (additive-only preferred).
5. **Prerequisite fix riding along** (from §9C of the readout): persist the
   frozen S0 genome as its own `genome_type` so the deployed gate is in the
   DB for every seed.
6. Deploy order: build anytime; **install at worker-idle** (IDSXD is
   draining, ~7-8 days — swap between flows via scripts/worker_swap.py);
   **SMOKE ONE** (s20401) and verify centres, guard, classnorm log lines,
   and the frozen-genome row before releasing the other 4. Flows via
   POST /api/flows with experiments included.

## 7. Open defaults (chosen, flag if wrong)

- epsilon for the tiebreak: 0.5% relative fitness.
- S0 multipliers per-cluster-independent (9 combos, guard-filtered);
  S1/multi multipliers global scalars.
- classnorm sigma floor 1e-6; margin tau by val-F1.
- The flat-multi variant of this arm is NOT queued now (one arm ruled);
  build the classnorm mode dataset-agnostic so it can be.
