# Stroobants, De Wagter, de Croon 2025 — Neuromorphic Attitude Estimation and Control

IEEE Robotics and Automation Letters ("PREPRINT VERSION. ACCEPTED MARCH, 2025"). arXiv:2411.13945.
Local PDF: `papers/stroobants2025_neuromorphic_attitude_estimation_control.pdf`. Read 19/11/2026.

**Relevance.** The nearest "exotic-substrate attitude controller on a Crazyflie" paper and the ONLY one found that
counts additions vs multiplications against the stock PID. **Regime:** real Crazyflie + Teensy 4.0 (Cortex-M7) deck,
total 35 g, mocap for the outer position loop, SNN trained by imitation of the stock complementary filter + cascaded PID.

## Verbatim extraction
- Abstract: tracks attitude "with an average error of 3.0 degrees, compared to 2.7 degrees for the regular flight
  stack"; "control commands from sensor inputs at 500Hz". "The network currently runs at 500Hz on a Teensy microcontroller."
- I/O: IMU gyro+accel + attitude setpoint in; pitch/roll/yaw TORQUE commands out; "The motor mixer is a linear
  transformation that converts torque commands into rotor velocities." Outer position controller stays classical.
- Integral: a hand-fixed subgroup of neurons acts as integrators — "These integrator neurons can now operate
  analogously to the integral component of a standard PID controller."
- Size: "trained with 150-150-130 neurons per layer … we reduced the size to 150, 100, and 80 per layer".
- Table I (RMSE / avg SD / avg RT): SNN (time-shifted & augm.) 3.03° / 0.77 / 145 ms; SNN (augmented) 3.10°;
  SNN (time-shifted) 3.24°; SNN (baseline) 3.14° / 1.16 / 135 ms; PID 2.67° / 0.23 / 125 ms.
- Operation count (estimate, not measured): "approximately 42,500 additions" per update, with 15–20% spiking sparsity
  "around 7,500 additions. In contrast, the cascaded PID controller on the Crazyflie requires about 28 additions and
  52 multiplications per timestep"; a float multiply taken as "roughly 37 times more energy than a" addition →
  traditional pipeline ≈ 3,000 additions; conclusion: on a conventional MCU the SNN is "in the same energy order of
  magnitude".
- Training: 20 min manual flight logged at 500 Hz; supervised BPTT; second data collection with SNN in the loop.
- Wording: "first fully neuromorphic system for attitude estimation and control", although it runs on a conventional MCU.

## Absent
- No measured energy/power, no measured inference latency (only the ≤2 ms bound), no parameter count, no flash/RAM.
- No neuromorphic chip in the loop. No rate/motor-level SNN (mixer classical).
