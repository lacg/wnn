# Eschmann, Albani, Loianno 2024 — Data-Driven System Identification of Quadrotors Subject to Motor Delays

Eschmann, J., Albani, D., Loianno, G. (2024). *Data-Driven System Identification of Quadrotors Subject to
Motor Delays.* arXiv:2404.07837v2 — https://arxiv.org/abs/2404.07837 (venue not printed on arXiv copy).
Local PDF: `papers/eschmann2024_sysid_motor_delays.pdf`.

**Relevance.** Flight-data fit of the first-order motor time constant for a 27 g Crazyflie 2.1 (brushed,
inferred from mass; motor type not stated) and a 3.35 kg quad. **Regime:** onboard flight logs at 1000 Hz,
RPM NOT measured — inferred from setpoints through the motor model; T_m swept to minimise thrust RMSE.

## Verbatim extraction
- Model: motors "as decoupled first-order systems"; Table I: "Tm — Motor delay time constant"; discrete
  form α = exp(−Δt/Tm) on RPM. Thrust = polynomial of RPM.
- §IV-A (Crazyflie 2.1, 27 g): "shows a clear optimum at **Tm = 0.072 s**. Based on the manufacturer's step
  response [footnote: bitcraze.io/wp-content/uploads/2015/02/M1-step-response.png] we read the time constant
  to be **≈ 0.073 s (rising and falling edge averaged)** ... Moreover, in [2] [Förster 2015] the time constant
  has been estimated at **0.065 s**".
- Fig. 8 caption region (large 3.35 kg quad): "Minimum: **Tm = 0.065 s**".
- Single Tm; separate rising/falling models left to future work.

**Derivation (ours):** T_ours = 4τ → CF2.1: 4×0.072 = **0.29 s**; large quad 4×0.065 = 0.26 s.

## Absent
- Measured RPM (inferred only); numeric sweep bounds for Tm; brushed/brushless label; uncertainty on Tm;
  separate rise/fall values (the 0.073 s is an average they read off the plot — see
  bitcraze2015_m1_step_response.md for our own digitisation, which does NOT reproduce 0.073 s).
