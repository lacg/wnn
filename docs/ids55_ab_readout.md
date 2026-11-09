# IDS-55 A/B readout — receptive fields `family` (cross 0.2) vs `off`, UNSW-temporal BINARY

Flows 7451–7460 (worker ABI 27, families_v1, dedupe off), BINARY lattice 2m max250, blend_neuron,
16b-Wb, sample rate 0.25, seeds r20419–r20423, paired by seed. The family arms 7452/7454 failed
at GA start (seed projection with `_rng=None`, fixed 9517d9d3) and were rerun from the beginning.

Metric: final `best_f1` genome, `margin_val_cal` (val-calibrated, report on the held-out test
partition), from `validation_summaries`. Cells = logical cells of that genome.

## GA Neurons (published phase), per seed

| seed | off: F1 / FPR / Acc | family: F1 / FPR / Acc | off: neurons / cells | family: neurons / cells |
|---|---|---|---|---|
| r20419 | 90.82 / 4.29 / 90.84 | 90.58 / 6.89 / 90.62 | 498 / 216,702 | 496 / 115,458 |
| r20420 | 90.87 / 4.95 / 90.89 | 89.65 / 6.49 / 89.68 | 498 / 210,609 | 500 / 114,211 |
| r20421 | 90.85 / 6.21 / 90.89 | 90.04 / 4.71 / 90.05 | 497 / 207,249 | 480 / 106,296 |
| r20422 | 90.79 / 4.71 / 90.82 | 89.33 / 4.23 / 89.34 | 500 / 215,704 | 500 / 114,320 |
| r20423 | 90.45 / 4.61 / 90.47 | 90.02 / 6.84 / 90.06 | 500 / 217,427 | 481 / 108,539 |
| **mean±SD** | **90.75±0.17 / 4.95±0.74 / 90.78±0.18** | **89.92±0.47 / 5.83±1.26 / 89.95±0.48** | 498.6 / 213,538±4,415 | 491.4 / 111,765±4,076 |

## Grid (for reference)

| arm | F1 | FPR | Acc | cells |
|---|---|---|---|---|
| off | 90.65±0.13 | 5.19±0.85 | 90.67±0.13 | 199,696±16,319 |
| family | 89.85±0.53 | 7.82±1.77 | 89.91±0.53 | 105,660±2,271 |

## Reading (n=5)

- **F1:** family is lower on all 5 seeds (GA), −0.83 pp on the mean.
- **FPR:** family is higher on the mean (5.83 vs 4.95) but split 3 worse / 2 better — the
  offline probe's lead ("unswt BINARY FPR") is NOT confirmed.
- **Memory:** family uses **~48% fewer cells** (111.8K vs 213.5K) at the same neuron count —
  family-confined neurons see correlated bits, so fewer distinct addresses get written.
- Duration equal (~0.43 h/flow).

So on unswt the families trade ~0.8 pp F1 for half the memory. Not adoptable as an accuracy
change; an open question is whether the memory saving is worth it (e.g. vs an `off` genome at
equal cells), and whether ciciot (coverage-limited in the probe) behaves differently.
