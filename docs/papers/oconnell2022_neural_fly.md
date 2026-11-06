# O'Connell, Shi, et al. 2022 — Neural-Fly Enables Rapid Learning for Agile Flight in Strong Winds

O'Connell, M., Shi, G., Shi, X., Azizzadenesheli, K., Anandkumar, A., Yue, Y., Chung, S.-J.
Science Robotics 7(66), eabm6597 (2022). DOI 10.1126/scirobotics.abm6597. arXiv:2205.06908.
Local PDF: `papers/oconnell2022_neural_fly.pdf`. Read 06/11/2026.

**Relevance.** The reference "learned outer loop over a stock inner loop" for disturbance rejection:
learned wind-force basis + online adaptive linear coefficients in the POSITION loop; PX4's stock
attitude PID underneath. **Regime:** real 2.6 kg custom quad (T/W 2.2), Caltech wind tunnel, mocap/EKF.

## Verbatim extraction
- Architecture: "We implemented our control algorithm and the baseline control methods in the position
  control loop in Python, and run it on the onboard Linux computer at 50 Hz. The PX4 was set to the
  offboard flight mode and received thrust and attitude commands … The built-in PX4 multicopter attitude
  controller was then executed at the default rate, which is a linear PID regulation controller on the
  quaternion error."
- Learned part: DAIML learns a shared representation of aerodynamic residual force "only using 12 minutes
  of flight data"; online "composite adaptation law" updates linear coefficients. "Our method … replaces
  the integral feedback term with an adapted learning term."
- Headline: "average improvement of 66 % over a nonlinear tracking controller, 42 % over an L1 adaptive
  controller, and 35 % over an Incremental Nonlinear Dynamics Inversion (INDI) controller."
- Winds up to "43.6 km/h (12.1 m/s)"; training data "only covers constant wind speeds up to 6.1 m/s".
- Table 1 tracking error [cm], RMS / mean:

| Method | 0 m/s | 4.2 | 8.5 | 12.1 | 8.5+2.4 sin(t) |
|---|---|---|---|---|---|
| Nonlinear | 11.9/10.8 | 10.7/9.9 | 16.3/14.7 | 23.9/21.6 | 31.2/28.2 |
| INDI | 7.3/6.3 | 6.4/5.9 | 8.5/8.2 | 10.7/10.1 | 11.1/10.3 |
| L1 | 4.6/4.2 | 5.8/5.2 | 12.1/11.1 | 22.7/21.3 | 13.0/11.6 |
| NF-Constant | 5.4/5.0 | 6.1/5.7 | 7.5/6.9 | 12.7/11.2 | 12.7/12.1 |
| NF-Transfer | 3.7/3.4 | 4.8/4.4 | 6.2/5.9 | 10.2/9.4 | 8.8/8.0 |
| NF | 3.2/2.9 | 4.0/3.7 | 5.8/5.3 | 9.4/8.7 | 7.6/6.9 |

- Limits: "combined code execution time and communication delay to be at least 15 ms and often as much
  as 30 ms"; "Attitude tracking delay can be substantially reduced through the use of a nonlinear
  attitude controller".
- Outdoor (GPS): "7.5 cm mean tracking error".

## Absent
- No learned inner loop; no motor-level learning; no motor time constant; exponential-stability claim is
  for the outer loop assuming the inner loop tracks.
