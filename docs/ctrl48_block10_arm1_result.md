# CTRL-48 block 10 arm 1 — result: L1 ADOPTED (`both_L1` is the new base)

Prereg docs/ctrl48_block10_arm1_prereg.md (written before any run). Flown 14/11 19:05 → 15/11 05:39 EST, controller
ABI 51, rule P, 35 TEST units × 100 episodes per seed. A = `both` (block 9, ABI 50, not re-flown; ABI 51 flags-off
reproduced it exactly) · B = `both_L1` (+ `--obs-tilt-ref-err`, dense ladder).

```
seed  stable %       err °         steady °      pos m           fails   | Δ (B − A): err · steady · pos · fails
oracle-fed (label source, bound)           100.0   0.69   0.16   0.357   0
MPCOF equal-info (COMPARATOR)              100.0   0.69   0.17   0.357   0
MPCOF no bias comp                         100.0   1.35   0.68   0.341   0
s1    99.9 | 100.0   1.40 | 1.04   0.82 | 0.43   0.361 | 0.342   2 | 0   | −0.36 · −0.39 · −0.019 · −2
s2   100.0 | 100.0   1.46 | 1.20   0.82 | 0.57   0.363 | 0.338   0 | 1   | −0.26 · −0.25 · −0.025 · +1
s3   100.0 | 100.0   1.26 | 1.07   0.68 | 0.49   0.358 | 0.336   0 | 1   | −0.19 · −0.19 · −0.022 · +1
s4   100.0 | 100.0   1.40 | 1.10   0.83 | 0.49   0.358 | 0.337   1 | 0   | −0.30 · −0.34 · −0.021 · −1
s5   100.0 | 100.0   1.34 | 1.00   0.72 | 0.50   0.360 | 0.339   0 | 0   | −0.34 · −0.22 · −0.021 ·  0
s6   100.0 | 100.0   1.28 | 1.05   0.71 | 0.40   0.358 | 0.339   1 | 0   | −0.23 · −0.31 · −0.019 · −1
s7   100.0 | 100.0   1.36 | 1.01   0.75 | 0.41   0.363 | 0.337   1 | 0   | −0.35 · −0.34 · −0.026 · −1
s8   100.0 |  99.9   1.45 | 1.22   0.86 | 0.58   0.361 | 0.337   1 | 3   | −0.23 · −0.28 · −0.024 · +2
s9   100.0 | 100.0   1.37 | 1.05   0.74 | 0.44   0.367 | 0.341   0 | 0   | −0.32 · −0.30 · −0.026 ·  0
s10  100.0 | 100.0   1.36 | 0.96   0.80 | 0.37   0.359 | 0.338   0 | 0   | −0.40 · −0.43 · −0.021 ·  0
mean               1.368 | 1.070  0.773 | 0.468  0.361 | 0.338  6 | 5   | −0.30 · −0.31 · −0.022 · −1
```

## Decision (pre-registered rule, applied in order)
1. Safety gate: B pooled fails 5/35000 ≤ 17 ✓; not worse than A's 6/35000 (one-sided conditional binomial p = 0.73) ✓.
2. Steady better by 0.305° > 0.03° ✓ (paired 95% CI −0.36 to −0.25°); err not worse — it is better by 0.298°
   (CI −0.35 to −0.25°) ✓. → **ADOPT L1. `both_L1` is the base of the 2×2.**
- Descriptive: 10/10 seeds better on steady AND err (sign test p = 0.002); pos −0.022 m on 10/10.
- Prediction (steady 0.5-0.65°) was beaten: 0.47°, i.e. below the block-9 bound of 0.49°.
- Gap to the equal-info COMPARATOR: steady 0.60° → 0.30°, err 0.68° → 0.38° (halved); pos now 0.019 m better than it
  (0.338 vs 0.357 m). Allowed wording: "halves the attitude gap to the equal-information comparator".
- B hold columns (new, no A value): pos_hold 0.263-0.275 m, dz_hold 0.123-0.143 m.
