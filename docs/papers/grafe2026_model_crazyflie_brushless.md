# Gräfe et al. 2026 — How to Model Your Crazyflie Brushless

Gräfe, A., Scherer, C., Hönig, W., Trimpe, S. (2026). *How to Model Your Crazyflie Brushless.*
Accepted at IEEE ICRA 2026. arXiv:2603.05944v1 (6 Mar 2026) — https://arxiv.org/abs/2603.05944
Local PDF: `papers/grafe2026_model_crazyflie_brushless.pdf` (arXiv HTML 404'd; read via pdftotext).

**Relevance.** The only source found that MEASURES the motor time constant of the Crazyflie 2.1
Brushless (our anchor airframe). **Regime:** real CFB, in-flight logging of firmware motor commands
vs rotor RPM from a custom deck (QRD1114 infrared sensor + reflective band under the propeller);
first-order fit. Model filters ROTOR SPEED Ω, thrust is a cubic polynomial of Ω.

## Verbatim extraction
- Eq (5): `Ω̇ = −(1/T)·Ω + (K/T)·u`, u ∈ [0,1]⁴ motor commands. Text: "we approximate the dynamics via a
  first-order linear system with time constant T and amplification K".
- §IV-B: "We identify the motor system dynamics by recording the motor commands and the RPM during flight
  ... We then fit the time constant T and amplification K to the measured curves leading to T = 0.05 s and
  K = 2900 rad s−1."
- Table II: m = 44 g (40 g without propeller guards); J = diag([3.3, 3.6, 5.9])·10−5 kgm² (diag([1.8, 2.4, 3])·10−5);
  **T = 50 ms**; K = 2.9·10³ rad s−1; Jm = 0.5·10−7 kgm²; ℓ = 7.07 cm.
- Eq (9) thrust: `Fthrust(Ω) = −0.23(Ω/σ)³ + 0.562(Ω/σ)² − 0.043(Ω/σ)`.
- Sim: mocap/Crazyswarm delay modelled as "a fixed delay of 8 ms"; sim step 4 ms.
- DR (§VI-A): "mass (±10 %), inertia and motor dynamics (±20 %), motor thrusts and torques scalings (±20 %)
  and center of gravity position (±1 cm)"; conclusion: "domain randomization between 10 %–20 % effectively
  bridges the sim-to-real gap".
- Abstract: brushless motors "provide around 50% more thrust compared to the brushed motors" of CF2.1.

**Derivation (ours):** T here is a TIME CONSTANT τ (63%). In our Molchanov convention (T_ours = 4τ):
T_ours = 4 × 0.05 = **0.20 s**; ±20% DR → τ ∈ [0.04, 0.06] s → T_ours ∈ [0.16, 0.24] s.

## Absent
- No separate rising/falling-edge time constants; no uncertainty/CI on T; no step size or operating point
  stated for the fit; no pure transport delay term for the motor; no ESC/DSHOT rate; no comparison τ for
  the brushed CF2.1.
