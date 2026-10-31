# Folk 2026 — Real Time Local Wind Inference for Robust Autonomous Navigation (PhD thesis)

Folk, S. (2026). *Real Time Local Wind Inference for Robust Autonomous Navigation.* PhD dissertation,
University of Pennsylvania (supervisors V. Kumar, M. Yim). arXiv:2604.00343 — https://arxiv.org/abs/2604.00343
Local PDF: `papers/folk2026_wind_inference_thesis.pdf`

**Relevance.** This is the only FREE-FLIGHT Crazyflie-family rotor-drag identification found, and it uses the
Svacha et al. [25] procedure. **Regime:** Crazyflie 2.1-Brushless at **0.057 kg** (34 g nominal plus mocap, a 6 g INA260 and a 12 g whisker deck),
55 mm rotor diameter (Table 6.1). It hovers in front of a WindShaper fan array at 0–50% power ("approximately 8 m/s" at 50%).
The fit uses onboard accelerometer data against airspeed × Σ rotor speed. The lateral fit includes any parasitic drag. It is not separated out.

## Verbatim extraction
- Rotor drag model (eq 2.9): `Dri = −K ηi vai`, `K = diag(kd, kd, kz)`. The drag is per rotor and summed over the rotors.
- Table 6.2 ("Lumped parameter aerodynamic coefficients identified using the procedure in [25]"):

| Parameter | Var | Value | Unit |
|---|---|---|---|
| Thrust Coefficient | kη | 4.26(10−8) | N/(rad/s)² |
| Induced Drag Coefficient | kd | 5.09(10−6) | N·s²/(m·rad) |
| Inflow Coefficient | kz | 1.19(10−5) | N·s²/(m·rad) |
| Translational Lift Coefficient | kh | 1.11(10−3) | N·s²/m² |

- Fig 6.6 fit (read from the page image): `y = 5.09e-06x + -2.87e-01`, where y = a_x (m/s²) and x = (1/m)·V_x·Σηi
  (unit rad/s / (s·kg)). The x range is 0 to about 0.95e6.
- Bench thrust stand (§6.3.1): `kη = 4.052·10−8 N-(rad/s)−2`. Motor model `η = Kv (Vsupply + V0)(PWM − DZ)^(2/3)`, with
  Kv = 0.3637 (rad/s)-V−1, V0 = 0.5535 V, DZ = 4,673.
- Hover power "approximately 10.56 Watts" (§6.3.2).

## Derivation (ours)
Hover: ηi = √(m g / (4 kη)) = √(0.057·9.81/(4·4.26e-8)) = 1811 rad/s, so Σηi = 7245 rad/s.
**k_d = kd·Σηi/m = 5.09e-6 × 7245 / 0.057 = 0.647 1/s.** With the bench kη = 4.052e-8, ηi = 1857 and k_d = 0.663 1/s.
Sanity check: at 8 m/s, x = 7245·8/0.057 = 1.02e6, which matches the Fig 6.6 axis range.

## Absent / caveats
- A different vehicle from the CF2.0/2.1 brushed: brushless motors, 57 g, and much more frame and deck area.
- No uncertainty on kd. No separate parasitic-drag fit for the Crazyflie. Airspeed range is about 0–8 m/s.
- No hover rotor-speed value is printed. Our hover Σηi comes from kη.
