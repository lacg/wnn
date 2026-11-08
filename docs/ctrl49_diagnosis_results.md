# CTRL-49 E diagnosis — results (08/11/2026)

## Method

- Test-only Rust module `imitation_diag_ctbr.rs` (`WNN_IMIT_DIAG=1`, `#[ignore]`).
- posx2 cascade recipe, seeds s1–s3, production ladder vs `--calib-episodes 200` (A).
- All states teacher-driven.
- Strata: ALL / TRANSIENT (t < 300 ms or tilt > 10°) / HOLD (5°, t > 300 ms; separate 5° fit and test sets).
- Logs: /tmp/ctrl49/diag_s{1,2,3}_{prod,c200}.log.

## HOLD ridge R², roll p / pitch q, mean of s1–s3

| inputs | production ladder | 200-episode ladder (A) |
|---|---|---|
| raw features (linear) | 0.690 / 0.710 | 0.690 / 0.710 |
| thermometer | 0.568 / 0.612 | 0.602 / 0.632 |
| D6 true attitude instead of estimate | 0.536 / 0.620 | 0.589 / 0.633 |
| D7 E1 tilt-referenced preview | 0.597 / 0.642 | 0.624 / 0.635 |
| D5 thermometer + teacher hidden state (Fh) | 0.757 / 0.778 | 0.777 / 0.790 |
| D5 thermometer + all hidden state | 0.767 / 0.788 | 0.783 / 0.798 |
| D5 raw + all hidden state | 0.979 / 0.980 | 0.979 / 0.980 |
| rate-I alone, thermometer (A) | — | 0.744 / 0.761 |

## Decision tree

- **Estimator:** no. The Mahony error is large (RMS 0.38–0.40° vs hold attitude 0.56–0.60°), but the hold label barely depends on attitude.
- **E1:** no. On top of A it adds only +0.025 / +0.009, below the 0.029 / 0.026 bar.
  - The self-majority oracle gain (+12 to +14 pp) is an address-variety artifact.
  - The generalisation version DROPS with E1: −2.6 / −4.5 pp under axis receptive fields, −1 pp under spread.
- **Single combined feature:** no (amplification about 7, no gain).
- **Single-feature resolution:** no (D1 diffuse, under 0.09 each).
- **Missing state: yes.** The teacher's hidden state (ua, I, Fh, d̂) adds +0.18 / +0.17, twice the whole raw-vs-thermometer gap. The pieces overlap: each one carries the disturbance.
- **D3 interactions:** within-family products don't beat raw.

## Recommendation

Fly the EXISTING hidden-state blocks under A (`--calib-episodes 200`): `--obs-rate-i`, `--obs-cascade-dhat-residual`, `--obs-cascade-dhat`. Their earlier failure to stack was on the 10-episode ladders. Offline R² gates; only closed-loop A/B decides.
