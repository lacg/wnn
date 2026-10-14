# =====================================================================
# SECTION 11 — MCST BITS-BAND ARM (tiered3, n=5, Protocol v2)
# =====================================================================

Appended 28/08/2026. Cohort **MCST-unswt-quad-16b-hier-s{20401..20405}-tiered3**,
flows 6005-6009, 5/5 complete, 0 failures. Comparator: the tiered2 arm of §10
(flows 6000-6004), paired by seed. Dataset/split/classification/seeds are
identical (`unsw-nb15` · `temporal_3way` · `hierarchical`), verified per pair
from `flows.config_json.params`.

⚠️ **This is a TWO-FACTOR change, not a bits ablation.** Diffing the params of
all five pairs gives exactly two differences, and both were changed together:

```
param                    tiered2        tiered3
min_bits / max_bits      4 / 34         34 / 50        <- the intended change
ids_tier_bits_min/max    (unset)        34 / 50        <- the intended change
ids_tier_neuron_cap      250            150            <- ALSO changed
```

The neuron-cap move is the smaller of the two in practice — S0 grid centres fall
~6% (totals `{52,76,102}` → `{48,72,96}`) and the S0 GA ceiling 163 → 150, while
the **S1 grid neuron axis is bit-identical between arms** (`{45,72,90}`). So bits
is the dominant difference, but the cap is not nil, and it pushes benign-FPR the
same way bits does. **Every FPR statement below is therefore attributable to
"wider bits AND a smaller cap", never to bits alone.** A clean bits-only rerun
was not done and is not planned (see 11H).

## 11A. Cascade result vs tiered2 — held-out TEST, paired by seed

```
seed      macroF1 t2   t3    delta      FPR t2    t3    delta     acc t2    t3    delta
20401         39.188  39.804  +0.615    11.778  13.865  +2.086   73.830  73.471  -0.360
20402         39.202  41.318  +2.116    12.730  13.119  +0.389   73.063  73.762  +0.700
20403         38.257  40.563  +2.306    11.908  12.562  +0.654   73.437  73.612  +0.175
20404         38.381  42.189  +3.809    10.805  14.173  +3.368   73.293  73.595  +0.301
20405         40.466  41.382  +0.916    11.903  15.011  +3.108   74.357  73.262  -1.096
--------------------------------------------------------------------------------------
MCST tiered3 (n=5)  macroF1 41.051±0.904 · benignFPR 13.746±0.947 · acc 73.540±0.187
MCST tiered2 (n=5)  macroF1 39.099±0.882 · benignFPR 11.825±0.684 · acc 73.596±0.509
MCSD auto    (n=5)  macroF1 37.256±1.75  · benignFPR 13.54±1.07   · acc 71.42±1.39
RF multi / RF cascade   macroF1 51.45 / 51.73 · benignFPR 22.58 / 24.95
```

Paired, exact sign-flip permutation (n=5):

```
macroF1    mean +1.952 pp   SD 1.271   5/5 seeds better   p = 0.0625   floor for n=5
benignFPR  mean +1.921 pp   SD 1.368   0/5 seeds better   p = 0.0625   floor for n=5
accuracy   mean -0.056 pp   SD 0.694   3/5 seeds better   p = 0.8750   flat
```

## 11B. BITS IS A TRADE DIAL, NOT A GAIN

**Both primaries are unanimous, and they point in OPPOSITE directions.** macro-F1
improves on 5/5 seeds; benign-FPR degrades on 5/5 seeds; the two deltas are the
same size to three significant figures (+1.952 vs +1.921 pp) and accuracy — which
absorbs both — does not move (-0.056 pp, p=0.875).

The widened band does not find a better model. It **slides the operating point
along the existing F1/FPR trade**. That is the whole result, and it is why
widening again is not worth a run: you buy the same exchange at the same rate.

Mechanism (consistent with, not proven by, this data): more bits → larger address
space → fewer collisions → less of the collision-driven generalization that IS
the WNN's learning mechanism. Training fit sharpens while benign generalization
erodes.

⚠️ **Do not read p=0.0625 as significance.** It is the *smallest attainable*
two-sided p for n=5 sign-flip, and it is attained here by BOTH primaries at once —
including the one that got worse. Unanimity at n=5 is a direction, not a
demonstration.

## 11C. STAGE DECOMPOSITION — the FPR cost is entirely the gate

The gate-FPR identity of §10E holds **5/5 in tiered3 too**: each flow's
`frozen_s0_gate.fpr` equals its cascade `benignFPR` to full precision. So the
cascade's benign-FPR *is* the S0 binary gate's FPR, and the S1 stage cannot
contribute to it. Splitting the two stages:

```
                            tiered2            tiered3          delta    seeds     p
S0 gate macro-F1        89.950±0.304       89.585±0.247       -0.365      2/5   0.2500
S0 gate benignFPR       11.825±0.685       13.746±0.948       +1.921      0/5   0.0625
S0 rows routed to S1   22953±226          23529±260           +576        5/5   0.0625
S1 stage macro-F1       40.435±1.076       42.749±0.953       +2.313      5/5   0.0625
S1 stage accuracy       68.191±0.953       69.154±0.354       +0.963      4/5   0.1250
S0 gate size (neurons)   95.4±22.1          84.8±16.4         -10.6       1/5   0.4375
```

**The two effects live in different stages and are cleanly separable:**

- 100% of the benign-FPR regression is the **S0 gate** — a sharper, slightly
  smaller gate misroutes 576 more benign rows per seed into the attack branch
  (5/5 seeds), and every one of those is a false positive by construction.
- 100% of the macro-F1 gain is **S1** — +2.313 pp at the stage, 5/5 seeds, which
  the cascade dilutes to +1.952 pp because the extra misrouted benign rows land
  in S1 as noise.

The S1 gain is the sturdier half of the result: **S1 improved by 2.3 pp while its
neuron budget was cut** (joint cap 250 → 150 with the gate taking ~85), i.e. the
gain survived a capacity reduction rather than being bought by one.

## 11D. Per-class — 9 of 10 classes improve; only Normal degrades

Held-out TEST, cascade, mean±SD over 5 seeds. `n` is the class's test support.

```
class              n     tiered2 F1      tiered3 F1     delta  seeds       p
Normal         18500   88.86±0.34      88.31±0.34      -0.55    2/5   0.2500
Analysis         351    7.15±5.05      11.32±1.85      +4.17    4/5   0.1875
Backdoor         283    7.31±1.50       8.09±2.11      +0.78    3/5   0.4375
DoS             2061    8.80±1.93       9.87±0.82      +1.07    4/5   0.2500
Exploits        5545   59.22±1.26      60.38±0.48      +1.16    4/5   0.1250
Fuzzers         3000   28.94±0.53      30.25±0.92      +1.31    5/5   0.0625
Generic         9438   93.70±1.73      96.22±1.28      +2.51    5/5   0.0625
Reconnaissance  1766   70.44±3.20      74.23±1.74      +3.79    5/5   0.0625
Shellcode        199   25.67±2.17      30.56±6.27      +4.89    3/5   0.3750
Worms             23    0.90±0.24       1.28±0.10      +0.38    5/5   0.0625
```

**Normal is the only class that loses, and Normal's loss IS the FPR cost** — the
same benign rows counted twice, once as a class-F1 drop and once as the headline
benign-FPR rise. Unanimous movers are Fuzzers, Generic, Reconnaissance and Worms
(5/5 each); Shellcode (+4.89) and Analysis (+4.17) are larger but noisier (3/5,
4/5) and Analysis's tiered2 SD of 5.05 on a 7.15 mean makes its delta unreadable.

MANDATORY recall table (per the QSR lesson — an aggregate-F1 win with recall
losses is NOT "detects better"):

```
class             tiered2 recall   tiered3 recall    delta  seeds
Normal            88.18±0.68       86.25±0.95        -1.92    0/5
Analysis          14.76±14.69      32.02±9.80       +17.26    4/5
Backdoor           7.63±2.31       11.10±3.76        +3.46    4/5
DoS                5.12±1.38        6.17±0.80        +1.05    4/5
Exploits          46.72±1.98       47.82±0.75        +1.10    3/5
Fuzzers           27.61±2.10       30.32±1.62        +2.71    5/5
Generic           95.01±0.60       94.60±0.30        -0.41    2/5
Reconnaissance    72.59±5.00       79.59±3.30        +7.00    5/5
Shellcode         61.61±4.75       66.33±6.27        +4.72    4/5
Worms             62.61±15.86      79.13±7.14       +16.52    4/5
```

Recall clears the QSR bar: 8 of 10 classes gain recall, and the two that lose
(Normal -1.92, Generic -0.41) are the two largest. This arm really does detect
more attacks — it just charges for them at the gate.

## 11E. ⚠️ TWO n=1 CLAIMS FROM THE SMOKE ARE RETRACTED

Both were single-seed reads of flow 6005 and both were carried forward before the
other four seeds landed.

1. **"98.3% of tier-slots pin at the band ceiling"** — that is seed 20401 ONLY.
   Counting every tier-slot of every stored S1 GA genome:

   ```
   arm       flow   modal bits   share of all tier-slots
   tiered2   6000        34             54.9%
   tiered2   6001        34             55.0%
   tiered2   6002        34             55.2%
   tiered2   6003        34             55.5%
   tiered2   6004        34             55.3%      -> mean 55.2%, SD 0.2
   tiered3   6005        50             98.3%
   tiered3   6006        50             56.9%
   tiered3   6007        50             77.0%
   tiered3   6008        50             54.0%
   tiered3   6009        50             25.9%      -> mean 62.4%, SD 26.6
   ```

   The modal value IS the ceiling in every seed of both arms, but tiered2 sat at
   55% of its own ceiling with SD 0.2 while tiered3 ranges 25.9-98.3%. **The band
   attracts; it does not bind** — and tiered3's pinning is not meaningfully
   higher than tiered2's, it is merely far more variable.

2. **"bits is a NON-LEVER"** — too strong. It moves S1 macro-F1 unanimously
   (11C). The defensible claim is 11B's: **a trade, not a gain.**

The n=1 smoke did predict the FPR cost accurately (+2.086 pp on s20401 vs
+1.921 pp at n=5). It over-read the pinning by a factor it had no way to see.

## 11F. Grid caveat — tiered3's S1 grid is NARROWER, not wider

Counting distinct `(bits, neurons)` configs actually stored per S1 grid:

```
tiered2   flows 6000-6004   15 configs   bits {11,17,22,28,33} x neurons {5,8,10}
tiered3   flows 6005-6009    9 configs   bits {34,43,50}       x neurons {5,8,10}
```

The `x[0.5..1.5]` bits multipliers clamp into the `[34,50]` band and collapse the
5-value bits axis to 3. So "widening the band" **narrowed the search**: tiered3
explored 9 shapes where tiered2 explored 15, and it lost every config below 34
bits. The same clamp hit S0 (`{16,23,31,34}` → `{34,45,50}`).

This is a second reason not to widen again — the band is a floor as well as a
ceiling, and raising the floor deletes configurations that tiered2 was free to
pick. Per-class bits diversity only ever appears in GA-neurons, by which point
bits is frozen from the grid winner.

## 11G. Bits does NOT fix the Worms sink

Over-absorption below is `predictions / true support` (Worms true support = 23
rows in every seed):

```
                   tiered2              tiered3          delta   seeds      p
Worms predictions  3216±580             2830±232         -386     4/5   0.1875
Worms precision %  0.453±0.122          0.644±0.049      +0.191   5/5   0.0625
over-absorption    140x±25              123x±10          -12%
top FP sources     Exploits 41-46%, DoS 24-35%  (stable in all 10 runs)
```

Precision rises unanimously but from 0.45% to 0.64% — **17 true positives out of
2830 predictions**. The sink is barely dented and is orthogonal to bits, which is
why it needs its own mechanism fix (the coverage-aware scorer) rather than more
address space.

> Denominator note: an earlier working note quoted this ratio as 527x → 462x. The
> prediction counts and precisions there are identical to the table above; only
> the denominator convention differs, and the relative change is the same to a
> tenth of a point (-12.2% here, -12.3% there). Use `predictions / true support`
> as defined here, or state the alternative explicitly — never mix the two.

## 11H. What this cohort supports / does not

**Supported:**
- Widening the bits band from 34 to 34-50 raises cascade macro-F1 on 5/5 paired
  seeds (+1.95 pp) and raises benign-FPR on 5/5 paired seeds (+1.92 pp).
- The two effects are stage-separable: the F1 gain is S1's (+2.31 pp at the
  stage, 5/5), the FPR cost is the S0 gate's (5/5 gate-FPR identity).
- 8 of 10 classes gain recall; only the two largest lose any.
- The S1 gain survived a joint-neuron-cap cut from 250 to 150.

**NOT supported:**
- Any significance claim. p=0.0625 is the n=5 floor and both primaries hit it.
- Attributing the FPR cost to bits alone — `ids_tier_neuron_cap` moved with it.
- Any net-improvement claim: accuracy is flat (-0.056 pp, p=0.875), so this is a
  reallocation, not a gain.
- Any claim of closing the tree gap. **41.05 vs RF 51.45 is still -10.4 pp**,
  narrowed from -12.4 pp but not closed.
- Anything about the Worms sink being addressable by sizing.

**RULING (Luiz, 28/08/2026): bits is CLOSED.** No further widening, and no
bits-only rerun to de-confound the cap — the reason is 11B, not the confound: a
dial that trades F1 for FPR at 1:1 does not move the frontier, so a cleaner
measurement of the same dial buys nothing. The next lever is the sink (11G), not
the address space.
