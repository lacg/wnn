# CTRL-45 axis receptive fields — validity audit (08/11/2026, experiment-design agent)

**Verdict: REAL, with caveats.** It is not caused by selection, the wheel, the scorer or a leak. It appears in every stage, on every
unit, in the random grid genomes, and on 10 never-seen units. Formal pre-registered H1 needs n=3; s3 is running.

## Evidence (measured)

**Same evaluation.**
- The only argv difference is `--receptive-fields axis --rf-cross 0.2`; the plant, CTBR, weather, gate, RG and seed lines are byte-identical.
- PID[est] and MPCOF rows match on all 35 units in both seeds.
- The ABI 44 vs 45 confound is ruled out: the base s1 re-score under 45 reproduces exactly, and s2 is a same-wheel pair.

**No leak.**
- Stage-select ran on val seeds 163226054..58, disjoint from the test seeds 99990101..135.
- Thresholds are fit on the train seed.
- RF is a connection-map projection only; no scorer, kernel or decode reads it.

**Same thing measured.**
- Bits, decode, action_repeat, ranges and mode are equal. The base headline actually had MORE neurons (264 vs 256).
- In-family connection share is exactly 0.83 in RF vs 0.18–0.33 under spread.
- Effort is equal, and jerk shows no consistent cost.

**Per stage and per unit.**
- Err is better on 35/35 units in every stage of both seeds.
- No unit gets worse on stable.
- Worst-wear units go from 20–63% to 86–100%.

**Population.** 55 random grid genomes per arm, no overlap between arms:

| seed | spread | rf |
|---|---|---|
| s1 | 62.0±7.0% | 97.9±2.1% |
| s2 | 82.2±3.9% | 98.8±1.6% |

So the effect is structural, not produced by search or selection.

**10 fresh units (99990201..210).**

| | stable | err |
|---|---|---|
| base | 82.9% | 3.81° |
| rf | 99.3% | 2.14° |
| PID | 88.7% | 3.61° |

RF err is better on 10/10. On unit 205: PID 21%, base 23%, rf 99%.

**6× fewer cells.**

| s1 GRID pop[0] | total cells | median per neuron |
|---|---|---|
| spread | 1.10M | 2,194 |
| rf | 219k | 221 |

All stored cells are TRUE. Each RF neuron reads a few co-moving features of one axis, so it sees fewer distinct addresses and each one is
revisited more: generalization, not an empty memory.

## Headlines

| | stable | err | steady | fails/3500 |
|---|---|---|---|---|
| s1 base | 87.7% | 3.57° | 2.17° | 429 |
| s1 rf | 98.9% | 2.04° | 1.22° | 38 |
| s2 base | 82.5% | 3.81° | 2.47° | 611 |
| s2 rf | 99.8% | 1.98° | 1.21° | 7 |

## Caveats

- n=2 so far.
- Stable is now SATURATED, so future RF A/Bs must rank on err and steady (2.0° vs the teacher's 1.35°).
- Side oddity in both arms: the GRID "population (held-out, descriptive)" line averages about 25%, apparently from scoring cell-less members. To be fixed.

## Ranked improvements

1. Re-base CTRL-48 block 2 onto RF.
2. RF + drop-constant: the dead tgt bits are 22% of the yaw bank's connections under RF.
3. rf-cross dose 0 / 0.1 / 0.4.
4. RF + calibration fixes.
5. RF + leak999 / rate-I.
6. grid-bits 32 under RF.
7. Connections stage on.
8. Family-map tweaks.
9. A read hit-rate counter, plus a harder stress row (±30% asymmetry) so stable can rank again.
