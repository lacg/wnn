# Eschmann, Albani, Loianno 2024 — Learning to Fly in Seconds

Eschmann, J., Albani, D., Loianno, G. *Learning to Fly in Seconds.* arXiv:2311.13081v2 (rev. 08 Apr 2024;
"Accepted April, 3, 2024" — IEEE RA-L). https://arxiv.org/abs/2311.13081
Local PDF: `papers/eschmann2024_learning_to_fly_in_seconds.pdf`.

**Relevance.** RL sim-to-real on a 27 g Crazyflie with a deliberately SLOW motor model. **Regime:** an
ASSUMED simulator constant chosen for transfer, not a measurement.

## Verbatim extraction (§IV)
- "Common values for the RC equivalent time constant (1 − e−1 ≈ 63% step response) are between **0.05 s and
  0.25 s**. We empirically find **0.15 s** to be suitable for Sim2Real transfer which is also supported by the
  manufacturers measurements" (footnote → Bitcraze M1 step-response PNG).
- Transfer is "without domain randomization" of the motor constant.

**Derivation (ours):** 0.15 s is τ (63%), so T_ours = 4 × 0.15 = **0.60 s** — beyond our T=0.30 axis.
Note the same group's later flight-data fit gives τ = 0.072 s (eschmann2024_sysid_motor_delays.md).

## Absent
- No citation for the 0.05–0.25 s range; no numbers read from the manufacturer plot; no DR on τ;
  no CF version/brushed label (mass 27 g only).
