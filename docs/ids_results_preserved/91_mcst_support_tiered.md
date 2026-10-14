# =====================================================================
# SECTION 10 — MCST SUPPORT-TIERED ARM (B side, n=5, Protocol v2)
# =====================================================================

Appended 28/08/2026 (overnight, autonomous). Cohort **MCST-unswt-quad-16b-hier-
s{20401..20405}-tiered2**, flows 6000-6004, 5/5 complete, 0 failures.
Comparator: the banked MCSD-auto hier arm (§9), paired by seed. B side only —
nothing was re-run on the A side. Spec: docs/MCST_TIERED_ARM_SPEC.md.

**Everything else is matched to §9**: same dataset/split/bits/weights/anchor,
same 4 stage-tagged phases, same pop/gens/patience. The ONLY changes are
support-tiered per-class sizing (cap 250 joint, S1 = 250 − S0 winner) and the
two new classnorm decode modes.

⚠️ **A `-tiered` generation (flows 5995-5999) exists and is NOT comparable** —
its S0 grid was degenerate (see 10D). NEVER merge it with `-tiered2`.

## 10A. Cascade result vs baseline — held-out TEST, paired by seed

```
seed      macroF1 base    MCST    delta      FPR base    MCST   delta     acc base   MCST   delta
20401           34.60    39.19    +4.59         14.10   11.78   -2.32
20402           36.47    39.20    +2.73         14.56   12.73   -1.83
20403           39.07    38.26    -0.81         11.77   11.91   +0.14
20404           38.10    38.38    +0.28         13.41   10.81   -2.60
20405           38.06    40.47    +2.41         13.84   11.90   -1.94
-----------------------------------------------------------------------------------------------
MCST  tiered2 (n=5)   macroF1 39.10±0.88 · benignFPR 11.82±0.68 · acc 73.60±0.51
MCSD  auto    (n=5)   macroF1 37.26±1.75 · benignFPR 13.54±1.07 · acc 71.42±1.39
RF multi / RF cascade macroF1 51.45 / 51.73 · benignFPR 22.58 / 24.95
```

Paired, exact sign-flip permutation (n=5):

```
macroF1    mean +1.84 pp   SD 2.13   4/5 seeds better   p = 0.1875   NOT significant
benignFPR  mean -1.71 pp   SD 1.08   4/5 seeds better   p = 0.1250   NOT significant
accuracy   mean +2.18 pp   SD 1.40   5/5 seeds better   p = 0.0625   floor for n=5
```

p=0.0625 is the SMALLEST attainable p for a two-sided n=5 sign-flip test.
Accuracy is "as significant as n=5 permits"; macro-F1 and FPR are not
significant. **Three consistent directional wins with halved variance is a
promising signal, not a demonstrated effect.** More seeds are needed.

**The variance collapse is the sturdiest observation**: SD roughly HALVES on
all three metrics (1.75→0.88, 1.07→0.68, 1.39→0.51). Support-derived centres
remove the seed-to-seed architecture lottery, which is exactly what §9D
predicted they would do.

## 10B. ATTRIBUTION — this is tiering, NOT classnorm

**The cascade's combined metric uses RAW ARGMAX.** `_compute_ids_hierarchical_
combined` → `predict_classes` → `predict` → `IDSCache::predict_examples`, which
is un-calibrated argmax. The classnorm modes are computed in the per-stage
decode sweep and persisted, but they **do not feed the combined 10-class
number**. Therefore:

- 10A's gains are attributable to **support-tiered sizing alone**.
- classnorm is measured but unexercised by the headline. No combined-vs-
  ablation confound: the spec's worry about attributing a joint win is moot.

~~**classnorm's own evidence (S1 stage, flow 6000, 9-class):**~~ **← see the correction below**

```
argmax           40.0442      argmax_platt      39.9537
margin_fixed0    40.0442      argmax_beta       32.6246
margin_train_cal 38.1838      margin_classnorm  40.2705
                              argmax_classnorm  41.1988   <- claimed best of 7, RETRACTED
```

⚠️ The S1 stage has **no benign class** (9-class attack-only; index 0 is
Analysis), so every "BenignFPR" in the S1 sweep — including argmax's 74% — is
meaningless. Judge S1 modes on macro-F1 only.

~~**OPEN HEADROOM:** routing the cascade's S1 decode through `argmax_classnorm`
instead of raw argmax is untested and worth ~1pp of S1 macro-F1 on this
evidence. That is a one-line change to the combined path and its own arm.~~

### ⚠️ CORRECTION 28/08/2026 — the evidence block above is flow 5995, and it does not replicate

Two defects, both re-verified against the DB. **The attribution conclusion of
10B is UNAFFECTED** — the cascade still uses raw argmax, so 10A's gains are
still tiering alone. What is withdrawn is only the claim that classnorm is a
good decode and worth routing to.

**1. Wrong flow — and a VOID one.** All seven values match, to four decimals,
`flow 5995` · `S1: GA Neurons` · `validation_point='final'` ·
`genome_type='best_fitness'`. Not flow 6000. **Flow 5995 is the VOID `-tiered`
generation of 10D**, whose S0 grid was degenerate — the same section that
forbids merging it supplied this block's numbers. Within 5995's own row the
seven modes ARE paired and classnorm genuinely does win; the row is simply not
admissible evidence about anything.

> An earlier working note diagnosed the source as flow 6003's `best_acc` argmax
> row (40.0394, which rounds to the same 40.04). That was a near-miss; the exact
> provenance is 5995.

**2. It does not replicate on the valid flows.** Read PAIRED over flows
6000-6009 — every row one `(flow, genome_type)` pair at
`validation_point='final'`, so each mode is scored on the SAME genome as argmax:

```
mode                mean F1%    vs argmax   rows won   flows won   p (flow-level, n=10)
argmax                41.626      +0.000       —          —              —
margin_fixed0         41.626      +0.000      0/50       0/10         1.0000   (identical to argmax)
argmax_platt          41.815      +0.189     29/50       5/10         0.6426
margin_train_cal      40.540      -1.087      4/50       1/10         0.0137
argmax_classnorm      40.218      -1.408      8/50       2/10         0.0312
margin_classnorm      39.990      -1.637      7/50       2/10         0.0098
argmax_beta           34.258      -7.368      0/50       0/10         0.0020
```

**argmax_classnorm LOSES by 1.408 pp (p=0.0312), winning 8 of 50 rows and 2 of
10 flows** (6001 and 6005). **No mode beats argmax**: `argmax_platt` is the only
positive one and it is a coin flip (+0.189 pp, 5/10 flows, p=0.64).
`margin_fixed0` is bit-identical to argmax on all 50 rows — it is a duplicate,
not a seventh option, so the sweep offers six distinct decodes, not seven.

**3. The mechanism is real; the exchange rate is against us.** Per-class F1,
classnorm minus argmax, S1 `best_fitness` genome, mean over the 10 valid flows:

```
  DoS            +14.63 pp   10/10        Reconnaissance  -0.89 pp    7/10
  Worms           +1.21 pp    9/10        Analysis        -1.11 pp    2/10
  Generic         +1.28 pp    6/10        Fuzzers         -4.63 pp    1/10
  Backdoor        -0.22 pp    6/10        Exploits       -10.64 pp    0/10
                                          Shellcode      -12.40 pp    0/10
```

classnorm does drain the Worms sink — predictions fall 2926 → 1449 (**-50.5%**
mean) — and DoS, the sink's second-largest donor, recovers +14.63 pp on 10/10
flows. But it pays for that in Exploits (-10.64, 0/10) and Shellcode (-12.40,
0/10), and the net is negative. The instrument is too blunt.

> Two sub-claims from the earlier note are themselves corrected here:
> (a) **Worms recall is NOT unchanged.** At n=10 it falls **72.6% → 67.8%**, so
> the removed predictions are not purely false positives. The "unchanged at
> 73.9%" reading was flow 6005 alone.
> (b) The **σ-floor edge case reproduces**: flow 6004 moves the WRONG way,
> Worms predictions 2695 → 3722 (**+38.1%**). `fit_class_zstats` floors σ at
> 1e-6, so a *near*-constant score column (a rare class's largely-untrained
> memory) is amplified rather than damped; only an exactly-constant column
> degrades to the intended mean-shift.

**RULING: the classnorm-in-cascade arm is REFUTED BEFORE RUNNING — do not queue
it.** There is no ~1 pp of headroom to route to.

## 10C. Efficiency claim (pre-registered as efficiency-only)

```
                MCSD-auto gates          MCST-tiered2 gates
seed 20401      608 n                    128 n
seed 20402       12 n                     78 n
seed 20403       22 n                     83 n
seed 20404       26 n                    106 n
seed 20405       12 n                     79 n
-------------------------------------------------------------
spread          12-608 n (51x)           78-128 n (1.6x)
runtime         1.8-30.6 min             3.8-24.0 min
```

The 51x gate-size spread collapses to 1.6x. The 608-neuron grid-lottery
outlier (§9D — worst gate of five at 30x the cost) cannot recur: centres are
support-derived and the epsilon tiebreak takes the smaller genome on a tie.
**Observed firing, flow 6000:** `[tie-break] 2 configs within 0.5% of best
fitness 4.8018 — picked smallest genome (94 neurons, was 122)`.

Per the pre-registration, **no accuracy claim may cite the neuron tiering**;
10A's numbers are reported as the arm's total effect.

## 10D. The `-tiered` generation is VOID (kept, never merged)

Flows 5995-5999. The first smoke (5995) revealed that S0's centres were
allocated against the JOINT cap (250) while the grid guard capped S0 at 160,
so `n×0.75` (188) and `n×1.0` (250) were both excluded and the S0 grid
collapsed from 15 configs to 5, all at `n=125` — a degenerate neuron axis.

Fixed in `41db56f1` (reserve the next stage's floors BEFORE allocating);
regression test carries 5995's real supports. The corrected S0 grid enumerates
all 15 configs and ranks `n=120` above `n=160`, reproducing §9D's
capacity-saturation finding as soon as the axis was free to vary.

5995 completed with the degenerate grid (cascade macroF1 .3892 / benignFPR
.1164) and 5996-5999 never ran. They are left on disk as a record.
**Never merge `-tiered` with `-tiered2`.**

## 10E. Defect closed: the deployed gate is now in the DB

`frozen_s0_gate` rows persisted **5/5** (§9C had 2/5 recoverable):

```
seed 20401  gate 128n  fpr=0.1178  routed=23046
seed 20402  gate  78n  fpr=0.1273  routed=23102
seed 20403  gate  83n  fpr=0.1191  routed=22956
seed 20404  gate 106n  fpr=0.1081  routed=22562
seed 20405  gate  79n  fpr=0.1190  routed=23098
```

Each row's `fpr` equals its cascade `benignFPR` exactly — the gate-FPR
identity, now verifiable from a persisted row rather than reconstructed.

## 10F. What this cohort supports / does not

**Supported:** support-tiered sizing moves all three cascade metrics in the
right direction on 4-5 of 5 paired seeds and roughly halves their variance;
the gate-size lottery is eliminated; the deployed gate is recorded for every
seed. ~~classnorm is the best S1 decode of seven~~ — **RETRACTED 28/08/2026,
see the correction in 10B: read paired on the ten valid flows, classnorm loses
by 1.408 pp (p=0.0312) and no decode beats argmax.**

**NOT supported:** any significance claim (n=5, p≥0.0625); any claim that
classnorm improved the cascade (it is not in that path); any claim that
classnorm is a good S1 decode at all (10B correction — it is significantly
worse than argmax, and the arm to route it is refused, not open); any claim that WNN
multiclass now approaches the tree bar — **39.10 vs RF 51.45 is still -12.4pp**,
narrowed from -14.2pp but not closed.
