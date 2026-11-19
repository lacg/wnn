# Alavilli*, Nguyen*, Schoedel*, Plancher, Manchester — TinyMPC: Model-Predictive Control on Resource-Constrained Microcontrollers

arXiv:2310.16985 (v1 25 Oct 2023). Venue (ICRA 2024) — unverified in the PDF text. Author order verified on the PDF first page (* = equal contribution).
Local PDF: `papers/alavilli2024_tinympc.pdf`. Read 19/11/2026.

**Relevance.** NON-NN comparator: what classical optimal control costs on the same STM32F405. **Regime:** real 27 g
Crazyflie 2.1, mocap pose at 100 Hz, float32 ADMM.

## Verbatim extraction
- Crazyflie MCU: "ARM Cortex-M4 (STM32F405) clocked at 168 MHz with 192 kB of SRAM and 1 MB" flash.
- Bench MCU: Teensy 4.1 "ARM Cortex-M7 running at 600 MHz with 32-bit floating point support, 7.75 MB of flash, and 512 kB".
- "TinyMPC ran at 500 Hz with a horizon length of N = 15 for the figure-eight tracking task"; obstacle task 100 Hz, N = 20.
  State n = 12, m = 4 "representing the quadrotor's full state and PWM motor commands". Converged within max 7 iterations
  inside the "allotted 2 ms solve time defined by the 500 Hz control".
- "maximum speed-up of 8.85x over OSQP"; OSQP exceeded Teensy memory at n = 16.
- 90° attitude-error recovery: TinyMPC max position error "23 cm"; PID and Brescianini "40 cm and 65 cm, respectively,
  while violating input limits"; INDI and Mellinger crashed.
- Future work: "a fixed-point version, since many small microcontrollers lack hardware floating-point support".

## Absent
- No measured µs solve time on the Crazyflie (only the 2 ms budget); no flash/RAM used on the Crazyflie; no RMS tracking error;
  no trial counts.
