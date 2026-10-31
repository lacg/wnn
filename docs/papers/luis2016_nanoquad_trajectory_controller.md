# Luis & Le Ny 2016 — Design of a Trajectory Tracking Controller for a Nanoquadcopter

Luis, C., Le Ny, J. (2016). Technical report, Polytechnique Montréal. arXiv:1608.05786 — https://arxiv.org/abs/1608.05786
Local PDF: `papers/luis2016_nanoquad_trajectory_controller.pdf`

**Relevance.** This is a commonly cited Crazyflie 2.0 model. It gives the HOVER ROTOR SPEED, an independent input for converting
Förster's κ⊥ into k_d. **It contains NO translational/rotor drag term.** Its "CD" is the rotor TORQUE coefficient.

## Verbatim extraction (Table 2.3.1 and §2.3–2.4)
- `mquad` "Mass of the quadcopter alone" **0.27 [Kg]**, muwb 0.04 [Kg], mvicon 0.02 [Kg], m "Total mass" **0.33 [Kg]**.
  These are printed values. They are evidently 0.027/0.004/0.002/0.033 kg (see the check below).
- d (arm length) 39.73×10−3 [m]; r (rotor radius) 23.1348×10−3 [m]; kT 0.2025; kD 0.11
- `CT = 3.1582 × 10−10 [N/rpm²]` (2.3.4); `CD = 7.9379 × 10−12 [Nm/rpm²]` (2.3.7). CD is a torque coefficient, not translational drag.
- Hover: `ωe = sqrt(mg/(4CT)) = 16073 [rpm]` (2.4.7)

## Derivation (ours)
Check: 4·CT·16073² = 0.3264 N, so m = 0.0333 kg. This confirms the "0.33 Kg" entry is 0.033 kg.
Θ̇Σ = 4·16073·2π/60 = 6732 rad/s. Combined with Förster's κ⊥: k_d = 9.1785e-7·6732/0.033 = **0.187 1/s**.
Note: this hover rotor speed is about 16% lower than Förster's motor maps give at 33 g (about 2085 rad/s per rotor, Θ̇Σ ≈ 8340).
Förster at 33 g gives k_d ≈ 0.232. The gap comes from the thrust model (kT taken from Crazyflie 1.0 literature [2]) versus the load-cell maps.

## Absent
- No translational drag, rotor drag, blade-flapping or induced-drag term. No wind or airspeed experiments.
