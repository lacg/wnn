# Mysore, Mabsout, Saenko, Mancuso 2021 — How to Train your Quadrotor (RE+AL)

arXiv:2012.06656v2 (22 Feb 2021). Venue not printed in the PDF. Local PDF: `papers/mysore2021_how_to_train_your_quadrotor.pdf`.
Read 19/11/2026.

**Relevance.** Neuroflight successor with a REAL-flight table vs a tuned PID on the rate loop. **Regime:** real FPV quad,
Betaflight-derived firmware with the PID block replaced by the NN; PPO in sim.

## Verbatim extraction
- Net: "network with 2 hidden layers, and 64 artificial neurons per hidden layer. The output layer of the policy network
  has 4 neurons — one for each of the motor command signals." "we replace this control block with a call to our trained
  neural network agents."
- Table 1, Test-Flight (MAE deg/s roll/pitch/yaw/overall; current A): PID 6.68/4.54/3.80/5.01, 8.07 A;
  Neuroflight 6.26/5.42/3.87/5.19, 22.87 A; RE+AL 6.01±1.80/4.52±1.18/2.64±0.18/4.20±1.14, 5.86±3.10 A.
- Simulated: PID 11.41, Neuroflight 7.30, RE+AL 6.75±2.13 deg/s overall.
- Abstract: "the first RL-based controllers trained in simulation to outperform a well-tuned PID controller on a real-world" drone.

## Absent
- MCU inference timing (see Koch 2019 for the 2×32 net); attitude-angle (deg) error; outer loop learning.
