# Ferede, Blaha, Lucassen, De Wagter, de Croon 2025 — One Net to Rule Them All: Domain Randomization in Quadcopter Racing Across Different Platforms

arXiv:2504.21586. ICRA 2025, DOI 10.1109/ICRA55743.2025.11128790 (per search hit; not printed in the PDF).
Local PDF: `papers/ferede2025_one_net_to_rule_them_all.pdf`. Read 19/11/2026.

**Relevance.** State-of-the-art motor-level policy on a flight-controller MCU. **Regime:** real 3-inch and 5-inch race
quads, STM32H743 running INDIflight ("a fork of Betaflight"), onboard EKF fed by OptiTrack, PPO.

## Verbatim extraction
- "Our controller maps state information directly to motor commands, enabling end-to-end guidance and control."
- Net: "three-layer fully connected network with ReLU activation functions, with 64 neurons per layer"; 20 observations
  (incl. propeller speeds); 4 motor commands; runs "onboard, at an update rate of 1000Hz".
- "An onboard EKF fuses data from a TDK InvenSense ICM-42688-P IMU, and position/attitude observations from an
  external optical motion capture system (Optitrack)."
- Training: PPO, 100 million time steps. 0% randomization fails transfer (5–7 / 8–13 gates on 3-inch / 5-inch);
  10% gives 38–39 / 46–47 gates at mean 7.05–7.13 / 8.63–8.73 m/s (Tables IV, V; three flights per policy).

## Absent
- Inference time, flash/RAM, MAC count; baseline classical controller in real flight; lap-time table per policy.
