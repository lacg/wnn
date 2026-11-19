# Ferede, de Croon, De Wagter, Izzo 2023 — End-to-end Neural Network Based Optimal Quadcopter Control (G&CNet)

arXiv:2304.13460v2 (22 Jun 2023). Journal not printed in the extracted text (Elsevier layout). Local PDF:
`papers/ferede2023_e2e_nn_quadcopter_control_gcnet.pdf`. Read 19/11/2026.

**Relevance.** Defines what the field means by "end-to-end" (no inner loop), trained by SUPERVISED imitation of optimal
trajectories — and still needs an onboard-estimated bias fed to the net. **Regime:** Parrot Bebop 1, Paparazzi,
"dual-core CPU Cortex A9", OptiTrack + onboard fusion.

## Verbatim extraction
- Highlights: "First flight tested, end-to-end neural network controller for quadcopters that does not rely on inner loop
  controllers".
- Abstract: sim-to-real gap normally requires "robust inner loop controllers during real flights, which limits the
  network's control authority"; nets "learn an energy-optimal policy mapping the quadcopter's state to rpm commands using
  an optimal trajectory dataset"; "this model mismatch is estimated onboard and fed to the network".
- Net: "3 hidden layers of 120 neurons with ReLU activation and an output layer of 4 neurons". Dataset "100,000
  energy-optimal trajectories" (90,000 train).
- Baseline: differential-flatness-based controller with INDI.

## Absent
- Inference time / rate not found by grep; no deg/m tracking-error table extracted (results are lap time and energy plots).
