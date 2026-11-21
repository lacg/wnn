# IDS-73: supervised cut points (arm E) on the production unswt recipe — prereg (Luiz 21/11/2026 ~16:40 EST)

## Why
unswt is the fast screen: its train set is 175k rows against unswr's 1.27M, so a flow takes about 0.6 h on this
recipe instead of 11 h at 80b on unswr. On unswr, IDS-70 E s1 gave F1 +0.21 pp and 55% fewer cells. This checks
whether E's gain carries to a second dataset, which has a different base rate (68% attacks) and a temporal
train/test shift.

## Arm and control
- **Arm:** the production unswt controls (IDS-56 ms01-encon: flows 7492 / 7496 / 7500, r20421-23; temporal_3way,
  max_bits 34, ids_n_bits 16, encoding dedupe on) cloned VERBATIM, plus `ids_cut_method=info_gain_mc`. Bodies are in
  `docs/ids73_flows.json`; the queue script is `scripts/queue_ids73_cut_e_unswt.py`.
- **Order (Luiz):** these run FIRST. Queued flows 7593-7599 are paused with POST /pause, and the waiter
  `logs/ids/ids73_resume_waiter.sh` resumes them with POST /resume, in id order, once all 3 IDS-73 flows have left
  'queued'.
- **Control:** the same-seed IDS-56 flow, not re-flown.

## Read
GA `best_f1` · `margin_val_cal` · TEST, per seed, paired vs control: F1, FPR, acc, logical cells, bits, hours.
Adopt reading (Luiz): F1 up consistently, not worse elsewhere. On the first flow, check the `[CUTS]` line, which
must report a budget equal to the 16-bit encoder's total bits.
