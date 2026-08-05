# L4 teacher screen — Crazyflie 2.1 Brushless, L4C (05/08/2026)

First controller result measured entirely on sourced ground: a plant from Bitcraze's
own firmware (`cf21_brushless`) under a disturbance ladder where every value cites a
paper or datasheet (L4C — see `disturbance_param_sources.md`). Stateless (sn=0),
nf=15 pidmix, BINARY, 5-seed held-out, 2 training seeds per teacher.

Chain: `scripts/l4_teacher_chain.sh` (PID 86480), 06:17:37Z → 10:38:45Z, **4 h 21 m**,
6/6 cells rc=0, markers in `experiments/l4teach_markers/`.

## Result — MEMORY-stage 5-seed held-out

```
teacher | s31337002      | s31337003      | mean  | seed   | classical | student vs
        | err°   stable% | err°   stable% | err°  | spread | err°      | own teacher
--------+----------------+----------------+-------+--------+-----------+-------------
LQR     | 0.86   100.0   | 1.35    99.0   | 1.11  | 0.49   | 0.93      | WORSE 0.18°
LQI     | 0.99   100.0   | 1.25    99.6   | 1.12  | 0.26   | 0.81      | WORSE 0.31°
PID     | 2.59    97.0   | 2.82    95.8   | 2.71  | 0.23   | 1.64      | WORSE 1.07°
```

Per-cell report-seed SDs: LQR 0.86±0.04 / 1.35±0.12; LQI 0.99±0.04 / 1.25±0.09;
PID 2.59±0.63 / 2.82±0.65. These are report-seed spreads on a FIXED winner and are
much smaller than the TRAINING-seed spread in the `seed spread` column — the latter is
the one that governs whether a ranking is real.

## What is separable, and what is not

- **LQR (1.11) vs LQI (1.12): NOT SEPARABLE.** The 0.01° gap is ~50x smaller than the
  seed spread (0.49 / 0.26°), and the two arms CROSS between seeds — LQR wins seed 2 by
  0.13°, LQI wins seed 3 by 0.10°. That crossover is the signature of an unresolvable
  pair. **Do not report an ordering between them.**
- **LQ* (1.11-1.12) vs PID (2.71): SEPARABLE.** A 1.59° gap, 3.2x the largest seed
  spread, with both PID cells far outside the LQ* range on both seeds.

## Conclusions that survive n=2

1. **Teacher quality propagates, but only across a large quality gap.** A weak teacher
   (PID, 1.64°) yields a decisively worse student. Between two strong teachers 0.12°
   apart, the students are indistinguishable.
2. **The substrate floor on this plant is ~1.1°.** Both strong-teacher arms converge
   there regardless of whether the teacher sits at 0.81° or 0.93°. Improving the
   teacher beyond LQR buys nothing measurable; the lever is the WNN's own resolution
   (thermometer bits, output neurons, decode), not teacher selection.
3. **NO student beat its own teacher** once both seeds are averaged (LQR +0.18,
   LQI +0.31, PID +1.07). **The single-seed "lqr beat its teacher at 0.86 vs 0.93"
   observation is WITHDRAWN** — seed 2 gave 1.35° and the mean is 1.11°, clearly worse.
   It was an n=1 artifact inside a 0.49° spread.
4. **stable% does not discriminate** on this ladder: every classical controller holds
   100% and the students sit at 95.8-100%. Error is the only useful axis, as the
   baselines predicted.

## Committee implication

The pool was meant to exploit decorrelated failure modes across structurally different
teachers. With LQR and LQI producing statistically identical students, a committee of
those two is unlikely to decorrelate much — PID is the only genuinely different member
and it is the weak one. **Forming the committee should wait** until the deferred
MPC-family teachers (MPCOF/MPC, structurally different from the LQ* family) are
available, or it will be an ensemble of near-duplicates.

## Open items this raised

- **`--max-cells 180000` is not a hard bound in practice.** Both LQ* arms overshoot
  entering MEMORY — LQR μ222k for 1 gen; LQI μ237k for 5 gens (seed 2) and μ235-249k
  for 4 gens (seed 3), *rising* before the cull, max-in-population 468k. The PID arm
  never exceeds μ59k. So the cap appears to be enforced post-mutation rather than at
  construction, and it binds exactly on the arms that search hardest — it may be
  shaping WHICH genomes survive into MEMORY, not merely bounding memory.
- **MPC-family teachers still deferred** (~20 min/gen, ~24 h/cell). Options in
  `logs/controller/l4teach/DEFERRED_mpc_teachers/README.md`. They are the only
  structurally distinct teachers left and matter for both the ranking and the committee.
- **n=2 is thin.** Every conclusion above except the PID separation rests on two
  training seeds. A third would firm up the floor estimate.

## Provenance note

Run on the corrected `TeacherBank` (see `disturbance_param_sources.md` — pre-05/08 the
bank clamped teacher ids > 2 to PID). Collision check across all six cells: every grid
CE and held-out triple distinct, so the fix held for the whole screen.
