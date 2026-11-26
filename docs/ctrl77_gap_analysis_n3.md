# CTRL-77 n=3 — gap analysis vs MPCOF (26/11/2026)

All numbers are TEST (35 CF2.1 units), CPU re-score (`logs/controller/ctrl73/wnn_units.py`),
mean of s1-s3. MPCOF numbers from `docs/ctrl73_gap_analysis_gt.md` §1-2 (B = est-fed eq-info for
means and hold; C = oracle for whole / pre-hold per-axis). G+T from
`logs/controller/ctrl73/data/n0g3gz_s{1,2,3}.npz`; l2b / l2bl3 from `logs/controller/ctrl77/gap/data/`
(scripts `ax.py`, `pu.py` alongside). All 8 re-scored winners reproduced their logged headline.

## 1. Gap table (values)

```
gap row                        MPCOF     G+T      l2b      l2bl3    carried by
stable %                       100.00    99.98    100.00   100.00   u123, u112 (roll/pitch mid-window)
fails / 10500 eps              0/3500    2        0        0        same
episode err max deg            n/a       7.34     4.68     3.88     u123 ep16 / ep33
err deg                        0.691     1.011    0.980    1.006    ~70% yaw (l2b excess: y .282, r .076, p .042)
  whole roll                   0.392     0.476    0.468    0.442
  whole pitch                  0.351     0.412    0.393    0.404
  whole yaw                    0.287     0.585    0.569    0.614
  pre-hold roll                0.466     0.569    0.560    0.529
  pre-hold pitch               0.419     0.493    0.470    0.485
  pre-hold yaw                 0.340     0.680    0.663    0.712    largest single piece
  early 0-0.3 s (norm)         n/a       3.461    3.325    3.389    rival split not measured
  mid 0.3-1.6 s (norm)         n/a       0.672    0.660    0.679
steady deg                     0.165     0.274    0.260    0.280    yaw only
  hold roll                    0.094     0.104    0.101    0.095    at rival level
  hold pitch                   0.083     0.087    0.082    0.082    at rival level
  hold yaw                     0.078     0.207    0.193    0.222
  signed hold yaw per c1       ~0        0.252    0.286    0.409    high-wear units (u115, u132, u130)
  mean |hold yaw offset|       n/a       0.067    0.067    0.091
err slope on |c1| (deg)        0.016     0.412    0.463    0.381    wear tail
pos m (3-D)                    0.357     0.3412   0.3420   0.3394   no gap (WNN ahead)
pos_hold m                     0.309     0.2756   0.2768   0.2713   no gap
```

l2b closes 0.031 of the 0.320 deg err gap (~10%). What remains is ~70% yaw.

## 2. Alignment

- **err: structural by unit.** Per-unit err correlates 0.93-0.99 across seeds and 0.99-1.00 across
  arms; same worst five units on every arm (123, 134, 121, 118, 112).
- **steady: not aligned by unit** (cross-seed r 0.25-0.69, partly genome-specific); only the
  wear-proportional yaw offset is consistent (offset vs c1, r 0.79-0.93).
- **l2b matches its mechanism, weakly.** err lower than G+T on 33/35 units; early window lower 3/3
  seeds (3.34/3.29/3.35 vs 3.42/3.44/3.52); pitch lower 3/3; roll 2/3; steady direction only (25/35 units).
- **L3 does not match its mechanism.** Meant to cut the yaw transient; in l2bl3 whole yaw is 0.614 vs
  l2b 0.569, wear-offset slope 0.409 vs 0.252 (higher 3/3). Its err comes from roll.
- **L1 matches the lag diagnosis.** s1: whole yaw 0.935 vs 0.597, hold yaw 0.382 vs 0.202, roll/pitch unchanged.
- **Fails: same episodes everywhere** — single-axis roll excursions that recover by the hold.
  u123 ep16: l2b 2.95/2.48/4.18 deg vs G+T 7.34/4.14/5.93.

## 3. Coverage map

```
gap (l2b value vs MPCOF)             follow-up                 can it see / close it?
hold yaw 0.193 vs 0.078              CTRL-79 yil; CTRL-78      targeted; chatter vs offset needs the re-score per END
wear err slope 0.46 vs 0.016         CTRL-79 (stated)          partly; full closure = wear estimate, out of scope
pre-hold yaw 0.663 vs 0.340          none (L3 missed)          UNCOVERED
pre-hold roll 0.560 vs 0.466         n=5 l2b                   seen only as err; roll barely moved
pre-hold pitch 0.470 vs 0.419        n=5 l2b                   partly closing
tail / fails                         n=5 l2b, l2bl3            underpowered as a count
stacking on the base                 none                      CTRL-78/79 pair vs G+T, not vs l2b
pos, hold roll/pitch                 none needed               no gap
```

Candidate follow-ups (model-free; none decided):
1. Yaw transient — VAL trace (`trace_val.py`, `trace_yawsat.py`) on the l3 s1 winner: did the corrective
   yaw command above 2.6 deg rise once the cut moved to 4.5 deg? If flat, the limit is output/label side.
2. Roll residual — re-run L2 pre-check (a) on an l2b VAL trace: does gyro_x still saturate under dense_hold?
3. Fails — read a fixed hard-episode panel as values per winner (u123 ep16/33, u112 ep86, u118 ep1/39/60/94).
4. Stacking — fly a surviving yil / leak arm on l2b before assuming it adds.
5. Per-END CPU re-score (20 s) on every CTRL-78/79/n=5 END as a descriptive readout.

## 4. Thin spots

- The n=5 bar is decided by fails at noise level: G+T passes err and steady and misses on 2 fails in 10500;
  at its n=10 rate (4/35000), 0 fails in 10500 happens ~30% of the time by chance.
- l2b's zero has a thin margin: s3 u123 ep33 at 4.68 deg vs the 5 deg threshold.
- l2a s1's 7 fails = u118 x3 + u123 x4, all roll, all recovered (one weak genome; cause inferred).
- s4/s5 may be harder steady seeds (G+T 0.30/0.33 there): l2bl3 at 0.283 has little room under 0.30.

Verified from files: every table number, correlations, fail episodes. Inferred: l2a cause, L3 offset
mechanism, s4/s5 difficulty, the chance read on fails. Not measured: MPCOF early/mid split, per-axis
for the other eight arms, any VAL trace of an F77 winner.
