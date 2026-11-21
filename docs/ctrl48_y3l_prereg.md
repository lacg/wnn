# Y3+L: Y3 with the dense yaw ladder (Plane CTRL-65) — prereg (Luiz pre-approved 20/11/2026, conditional)

## Why (per-axis VAL split, logs/controller/ctrl48/y3_gap/, Y3 vs Y+F winners s1-s5, tuning numbers)
The whole Y3 − Y+F gap is on YAW; roll/pitch differ by −0.003/−0.001° (err) and +0.002/+0.014° (hold).
```
Y3 − Y+F (VAL, mean s1-s5)   err      steady   yaw err (whole)  yaw hold   per-unit standing yaw offset
                             +0.069   +0.043   +0.098           +0.043     +0.036
per seed yaw err:  +0.064 +0.100 +0.082 +0.110 +0.132
per seed yaw hold: +0.014 +0.054 +0.056 +0.049 +0.041
```
Hold is the last 20% of the episode, so only ~0.2 × 0.043 ≈ 0.009° of the 0.098° yaw err gap sits in hold:
**~90% of the yaw err gap is the transient** (what L targets: the dense ladder adopted on Y+F, Plane CTRL-62).
The steady gap is mostly a **standing per-unit yaw offset** (+0.036°) — the target of the separate
`--delta-max-yaw` gain pre-check (G slightly above 1), not of L.

## Arms
- **Y3+L (flown):** each seed's Y3 argv (`..._a59_..._dmy03_full_sN.out`) with the bare `--yaw-centred-ladder`
  replaced by `--yaw-centred-ladder dense`; no F; `--delta-max-yaw 0.3`. ABI 59, s1-s5. Tags `..._dmy03_yld_full_sN`.
  Driver `logs/controller/ctrl48/ab_y3l.sh`.
- **Compared against (not re-flown):** Y3 (CTRL-63) and Y+F+L (CTRL-62), same seeds.

## Expectation
Against Y3: err −0.03 to −0.06° (yaw transient), steady 0 to −0.01°, pos flat, fails no worse.
Against Y+F+L: still behind on steady (the offset is untouched), err gap roughly halved.

## Reading (Luiz's rule)
Adopt as the pure-table line if consistently better than Y3 in err and not worse on the rest of the vector.
"Catches Y+F+L" only if it is level with Y+F+L on the whole vector; otherwise Y+F+L stays the base.
