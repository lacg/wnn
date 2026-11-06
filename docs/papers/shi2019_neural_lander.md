# Shi, Shi, O'Connell, et al. 2019 — Neural Lander: Stable Drone Landing Control Using Learned Dynamics

Shi, G., Shi, X., O'Connell, M., Yu, R., Azizzadenesheli, K., Anandkumar, A., Yue, Y., Chung, S.-J.
IEEE ICRA 2019, pp. 9784–9790 (venue/pages per Zhang 2022 reference list and Kaufmann 2022 ref [18]).
arXiv:1811.08027. Local PDF: `papers/shi2019_neural_lander.pdf`. Read 06/11/2026.

**Relevance.** Learned residual-force DNN (spectrally normalized) inside a nonlinear feedback-
linearization POSITION controller; stock attitude/motor loops below. **Regime:** real Intel Aero 1.47 kg,
mocap at 100 Hz, ground-effect landing.

## Verbatim extraction
- "combines a nominal dynamics model with a Deep Neural Network (DNN) that learns high-order interactions.
  We apply spectral normalization (SN) to constrain the Lipschitz constant".
- Results: "decrease error in z axis from 0.13 m to 0, mitigate x and y drifts by as much as 90%, in the
  landing case … decrease z error from 0.153 m to 0.027 m, in the cross-table trajectory tracking task."
- Loop rates: "update rate of attitude controller (> 100 Hz) and motor speed control (> 5 kHz) are much
  higher than that of the position controller (≈ 10 Hz)" — this timescale separation is used in the
  stability proof.

## Absent
- No learned attitude/motor loop; no disturbance beyond ground effect; no motor τ.
