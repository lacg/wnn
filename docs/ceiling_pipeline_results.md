# Ceiling pipeline results — S / B / A / C

Run 31/07/2026 18:59 → 01/08/2026 04:06 UTC (12h08, `rc=0` on all four phases).
Script: `scripts/run_ceiling_pipeline.sh all`. Report: `scripts/build_ceiling_report.py`.

Motivation: the dfa1l study's during-search numbers collapsed on held-out
(`dfa_10feat_BINARY_s31337003` reported `FINAL err=2.38° stable=100%` and
delivered **20.86° / 0% stable**). The pipeline tests four candidate explanations,
one per phase.

## What each phase is

| Phase | Name | Question it answers | Method |
|---|---|---|---|
| **S** | split sweep | Is the controller **sensor-starved**? Every study cell ran `suffix=18`; a smoke run suggested 18→32 lifts held-out 40%→90-95%. | 65 arms: control `sn=12/suf=18` + `sn{6,7} × suf{24,30,32,34,36,38}`, 5 training seeds each, every arm scored on 5 report seeds. `ob = sn + suffix` (state prefix forced — the FSM-coherence invariant). Cells wiped, ONE imitation training pass per arm, no GA. |
| **B** | data-budget curve | Is the gap **data starvation**? Would more DAgger episodes close it? | Retrain the winner architecture at 1× / 4× / 16× episode budget (192 / 768 / 3072), each scored on 5 report seeds. Read the SHAPE, with a noise test on each slope. |
| **A** | nominal long memory-GA | Is the gap the **imitation ceiling**? Does closed-loop reward beat the teacher? | MEMORY-stage value-GA seeded from the winner, 800 gens, patience 20, **no teacher ceiling** — reward is the objective, not teacher agreement. Nominal plant. |
| **C** | motor-fault memory-GA | Can the WNN **beat classical control** where classical control fails? | Same memory-GA on a faulted plant (motor 2 at 30% effectiveness) + matched classical baselines re-measured on that plant. |

S and B are diagnostics (is the substrate/data the problem?). A and C are attempts
to actually win.

---

## Phase A — the one classical-beating result

```
HELD-OUT (fresh report seed 99990101, nominal plant, L2D, tilt 5°)
  during-search winner:  stable=89.0%  err=3.27°  steady=3.34°  mono_viol=3
  vs PID:                stable=85.0%  err=3.96°
  population (descriptive): stable=65.4±37.8%  err=23.42±34.89°  (pop max 91.0%)
```

**First WNN result in the campaign to beat a classical baseline on held-out** — on
both stability (+4.0 pp) and error (−0.69°). Every dfa1l study cell topped out at
77.0% and lost to PID.

Caveats, both material:
- **n=1.** One genome, one report seed. Not reproduced.
- The population is **bimodal** (65.4±37.8%): the winner is a good draw, not
  evidence that the method reliably produces 89%.
- Early-stopped at **gen 42 of 800** (patience 19/20, ~5.5 gen/s) — the long budget
  was never needed, which is why the phase took 7 minutes.

## Phase B — the data-budget hypothesis is dead

```
saved       : stable= 84.8± 9.4  err= 3.43±0.42  (accumulated reference, NOT an arm)
retrain-1x  : stable= 38.6±16.3  err= 6.18±1.02  ( 192 eps, train  255s)
retrain-4x  : stable=  7.4± 7.3  err= 8.96±0.90  ( 768 eps, train 1345s)
retrain-16x : stable= 15.4±10.6  err= 8.36±1.25  (3072 eps, train 3428s)

slope  192->768 eps: -15.6 pp/doubling  (delta -31.2 vs noise ±17.8 -> REAL)
slope 768->3072 eps:  +4.0 pp/doubling  (delta  +8.0 vs noise ±12.9 -> within noise)
```

**More imitation data makes it worse**, and the decline passes the script's own
noise test. Buying episodes does not close the gap — stop buying episodes.

`saved` is the accumulated reference (cells built over 5 folds × every GA
generation), NOT a comparable arm. The honest comparison is retrain-vs-retrain
across the budget axis.

## Phase C — beats 3 of 5 classicals, loses to the best 2

```
WNN memory-GA (held-out, 1 report seed):  stable= 4.0%  err=10.03°  steady=13.41°
  population (descriptive):               stable= 3.2± 2.8%  (pop max 8.0%)

fault-plant classical baselines (5 report seeds):
  MPCOF  10.8± 7.0    LQI   7.2± 4.7    LQR   1.0± 1.3
  MPC     0.4± 0.5    PID   0.0± 0.0
```

The WNN (4.0%) **beats LQR, MPC and PID**, all of which collapse to ~0–1% when a
motor drops to 30%. It loses to MPCOF (10.8%) and LQI (7.2%).

Two things that block a claim here:
- **Asymmetric scoring.** The WNN is scored on ONE report seed; the baselines on
  five. Not like-for-like. Fix: rescore the phase-C winner on the same 5 seeds
  (`scripts/rescore_winners.py`).
- The recipe cited **MPCOF 20.0%** as the bar; freshly measured on 5 seeds it is
  **10.8±7.0%**. The bar moved, and `docs/motor_fault_experiment.md` should be
  updated to the measured value.

## Phase S — modest, replicable, not significant

Architecture: **`dfa / 9feat / BINARY`**, seeded from the `s31337002` winner —
i.e. variations on the study's BEST row (`dfa 9feat BINARY`, 77.0±15.0%).

```
   sn  suf  ob  n div       stable%         lo–hi         err°      steady°    cells
    6   24  30  5   0    39.6± 22.1  10.2– 70.0    6.41± 1.45   8.80± 2.82   21,002
    6   30  36  5   0    51.1± 15.0  33.6– 68.2    5.44± 0.88   7.14± 1.80   44,926
    6   32  38  5   0    38.9± 29.9   3.8– 73.6    9.92± 7.87  15.78±15.31   54,759
    6   34  40  5   0    55.0± 26.0   7.6– 78.0    5.97± 3.07   8.52± 6.01   61,573
    6   36  42  5   0    24.9± 12.8  11.2– 41.0    8.20± 1.85  12.56± 2.96   80,888
    6   38  44  5   0    43.2± 25.5  12.6– 71.0    6.76± 2.10   9.52± 3.91   90,265
    7   24  31  5   0    36.4± 25.5   1.6– 70.8    6.56± 1.98   9.02± 3.76   19,947
    7   30  37  5   0    41.5± 21.7   5.6– 73.2    6.60± 2.05   9.41± 3.27   42,158
    7   32  39  5   0    26.7± 20.0   2.0– 56.6   12.06± 7.82  19.04±14.03   55,680
    7   34  41  5   0    51.4± 21.2  17.6– 80.4    5.80± 1.76   8.09± 3.78   64,383
    7   36  43  5   0    30.8± 13.6   5.4– 44.4    7.12± 1.76  10.28± 2.95   68,952
    7   38  45  5   0    34.2± 17.3   7.8– 50.6    7.22± 2.13  10.92± 4.63   85,663
   12   18  30  5   1    34.6± 18.2   0.0– 49.4   20.94±30.33  25.38±35.37    7,931  <= CONTROL
```

Zero divergences in all 60 swept arms. The only `div` is the control's seed
`2350568529` (0.0% stable / 81.60° err), reproduced identically in both
invocations — a deterministic property of `sn=12/suf=18` on that seed. Hence the
control reads 34.6% over 5 seeds but **43.3% over its 4 converged seeds**.

Pooled across both `sn` blocks (n=10 per suffix):

| suf | n | mean | sd | Δ vs ctrl 34.6 | Δ vs converged 43.3 |
|---|---|---|---|---|---|
| 24 | 10 | 38.0 | 25.2 | +3.4 | −5.3 |
| 30 | 10 | 46.3 | 20.3 | +11.7 | +3.0 |
| 32 | 10 | 32.8 | 27.6 | −1.8 | −10.5 |
| **34** | 10 | **53.2** | 25.1 | **+18.6** | **+9.9** |
| 36 | 10 | 27.8 | 14.2 | −6.8 | −15.5 |
| 38 | 10 | 38.7 | 23.4 | +4.1 | −4.6 |

`suf=34` is top-of-block in BOTH independent blocks (55.0 / 51.4, 3.6 pp apart) —
the only suffix that replicates as good. `32` and `36` replicate as bad.

### Significance — and why pairing does not rescue it

```
suf=34 vs control, PAIRED on training seed:
  all 5 control seeds                  n=10  diff=+18.6pp  sd=34.0  t=1.73  need n≈27
  excluding the diverged control seed  n= 8  diff= +7.1pp  sd=25.6  t=0.78  need n≈103

  Pearson r(control, suf34) over paired arms = -0.17
```

Two things fall out of this:

1. **The +18.6 pp headline is carried by one seed.** Excluding the control seed
   that diverges to 0%, the paired advantage is **+7.1 pp, t=0.78** — nowhere near
   significant, and it would need **~100 training seeds** to demonstrate.
2. **Pairing cannot reduce the variance here.** `r = -0.17` between control and
   `suf=34` performance on the same training seed: a seed that trains well for one
   architecture is *not* the seed that trains well for another. Seed luck is
   **architecture-specific, not seed-specific**, so the usual variance-reduction
   trick (common random numbers / paired t-test) does not apply to this substrate.

Honest claim: *"suffix 34 is the best configuration tested and replicates across
two state-neuron settings"*. NOT *"suffix 34 is significantly better"*.

The motivating prediction **failed**: the smoke run said 18→32 would give
90–95%; `suf=32` pooled is **32.8%**, the worst suffix in the sweep, and nothing
anywhere approached 90%.

Cost: `suf=34` is ~61–64k cells vs the control's 7,931 — **≈8×** the memory
(≈8 KB vs ≈1 KB at 1 bit/cell BINARY) for ~+10 pp.

---

## S and A are not comparable — they measure different things

Phase S arms are **single imitation training passes** (cells wiped, one pass, no
GA). Phase A is a **full reward-driven memory-GA**. Comparing S's best (53.2%) to
A's 89.0% is apples-to-oranges: it conflates "which architecture" with "how much
optimization". They compose rather than compete — S says which architecture to
hand to A; A says what optimization then achieves on it.

The untested combination — **run phase A's memory-GA on phase S's `suf=34`
architecture** — is the obvious follow-up, and neither phase measured it.

## Open items

- **Phase A n=1.** Reproduce across ≥5 training seeds before citing 89%.
- **Phase C asymmetry.** Rescore the winner on the same 5 report seeds as the
  baselines; update `docs/motor_fault_experiment.md` (MPCOF bar is 10.8±7.0, not 20.0).
- **Phase S on 1layer.** The whole sweep ran on `dfa`. The `1layer` substrate is
  ~20× cheaper per cell and is the FPGA-relevant one; the suffix result may not
  transfer.
- **A ∘ S.** Memory-GA on the `suf=34` architecture.
