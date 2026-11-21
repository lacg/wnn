# IDS-74: one-setting A/Bs on the production unswt recipe — prereg (Luiz 21/11/2026 ~17:20 EST)

## Arms
All arms are clones of the IDS-56 ms01-encon controls (7492 / 7496 / 7500, r20421-23; temporal_3way, BINARY vote
θ 0.10, max_bits 34, ids_n_bits 16, encoding dedupe on, balance_classes on, no fitness_aggregation key, so the worker
default `harmonic` applies). Each arm changes exactly its own keys.

| arm | change | why |
|---|---|---|
| D | `fitness_aggregation=desirability`, `fitness_zrank_clamp=3.0` | IDSD (26/08) won F1 5-0 on unswt (+0.281 pp, paired SD 0.041). From IDS38 (13/10) unswt recipes stopped setting it, and there is no recorded decision; Luiz does not remember dropping it. |
| BC | `balance_classes=false` | On in every unswt flow since Sep, never A/B'd. unswr uses false. unswt TRAIN is 68% attacks. |
| V02 | `ids_vote_share=0.02` | IDS-56 tried 0.05 and 0.10 on unswt and 0.10 won; 0.02 was never tried. This rules out a U-shaped curve. The vote path was certified bit-identical ABI 28→29, with no fix since IDS-56. |

## Order
After IDS-73 (7600-7602), ahead of the paused 7593-7599. Within IDS-74: arm-major D, BC, V02. The IDS-73 resume
waiter re-queues 7593-7599 once no flow with id > 7599 is still queued.

## Read
GA `best_f1` · `margin_val_cal` · TEST, per seed, paired vs control: F1, FPR, acc, logical cells, hours. Adopt
reading (Luiz): F1 up consistently, not worse elsewhere. Each arm is judged on its own; there is no combined arm
until the singles are read.
