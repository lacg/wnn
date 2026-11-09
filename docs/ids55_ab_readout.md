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

## Round 2a — EQUAL MEMORY (flows 7461/64/67/70/73, rf off, max_neurons_total 262, ABI 27, 09/11)

GA best_f1 margin_val_cal, F1 / FPR / Acc · neurons / cells:

| seed | off (500n) | family 0.2 (v1) | off (262n, equal memory) |
|---|---|---|---|
| r20419 | 90.82 / 4.29 / 90.84 · 498 / 216.7K | 90.58 / 6.89 / 90.62 · 496 / 115.5K | 90.47 / 8.70 / 90.54 · 262 / 116.4K |
| r20420 | 90.87 / 4.95 / 90.89 · 498 / 210.6K | 89.65 / 6.49 / 89.68 · 500 / 114.2K | 90.20 / 4.52 / 90.22 · 247 / 109.8K |
| r20421 | 90.85 / 6.21 / 90.89 · 497 / 207.2K | 90.04 / 4.71 / 90.05 · 480 / 106.3K | 90.73 / 5.77 / 90.76 · 258 / 116.0K |
| r20422 | 90.79 / 4.71 / 90.82 · 500 / 215.7K | 89.33 / 4.23 / 89.34 · 500 / 114.3K | 90.41 / 5.74 / 90.44 · 232 / 102.6K |
| r20423 | 90.45 / 4.61 / 90.47 · 500 / 217.4K | 90.02 / 6.84 / 90.06 · 481 / 108.5K | 90.86 / 4.65 / 90.88 · 261 / 114.8K |
| **mean±SD** | **90.75±0.17 / 4.95±0.74 / 90.78** · 213.5K | **89.92±0.47 / 5.83±1.26 / 89.95** · 111.8K | **90.53±0.26 / 5.88±1.68 / 90.57** · 111.9K |

At the SAME memory (111.9K vs 111.8K cells), plain spread neurons beat family by +0.61 pp F1 (4/5 seeds), at
equal FPR (5.88 vs 5.83). Halving memory by halving neurons costs only 0.22 pp. So family's memory saving is not
special: you get the same saving, with better F1, from fewer spread neurons. Matches the ids-security prediction
(~90.6; docs/ids55_why_family_loses.md): the loss is structural (benign-side saturation + protocol starvation),
not capacity.
