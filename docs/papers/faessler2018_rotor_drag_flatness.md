# Faessler, Franchi, Scaramuzza 2018 — Differential Flatness of Quadrotor Dynamics Subject to Rotor Drag

Faessler, M., Franchi, A., Scaramuzza, D. (2018). IEEE RA-L 3(2). arXiv:1712.02402 — https://arxiv.org/abs/1712.02402
Local PDF: `papers/faessler2018_rotor_drag_flatness.pdf`

**Relevance.** This paper defines our model form exactly: a mass-normalized linear rotor drag `v̇ = −g zW + c zB − R D Rᵀ v`, with
`D = diag(dx, dy, dz)` in 1/s. **Regime is NOT micro:** the vehicle weighs 610 g with "stiff six inch propellers" and thrust-to-weight 4.
The coefficients were identified in free flight by gradient-free optimization of tracking error. Velocities reach 4 m/s.

## Verbatim extraction (§VII-C)
| identification trajectory | dx [s−1] | dy [s−1] |
|---|---|---|
| circle (4 m/s) | 0.544 | 0.386 |
| lemniscate | 0.491 | 0.236 |
| circle at 2.8 m/s | 0.425 | 0.256 |

"we assume the rotor drag to be independent of the thrust. This is not true in reality".

## Absent
- Not a Crazyflie or micro vehicle. Rotor speed is not measured. There is no per-rotor-speed coefficient, only lumped values that already have units of 1/s.
