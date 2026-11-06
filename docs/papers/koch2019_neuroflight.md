# Koch, Mancuso, Bestavros 2019 — Neuroflight: Next Generation Flight Control Firmware

Koch, W., Mancuso, R., Bestavros, A. arXiv:1901.06553v2 (16 Sep 2019). Venue: not stated in the PDF.
Local PDF: `papers/koch2019_neuroflight.pdf`. Read 06/11/2026.

**Relevance.** A tiny learned controller on a flight-controller MCU that outputs motor power directly —
but as an INNER (rate) loop: pilot/autopilot supplies rate setpoint and throttle is mixed in classically.
**Regime:** real FPV racing quad NF1, MATEKF722 (STM32F722), Betaflight-derived firmware, PPO in GymFC.

## Verbatim extraction
- "The NN embedded in Neuroflight replaces attitude control and motor mixing".
- Input/output: "x(t) = [e(t), ∆e(t)]" (angular-velocity error and its difference); "yi ∈ [0, 1] is the
  control signal representing the percent power to be applied to the ith actuator"; "the NN has no
  concept of thrust (T), therefore … the thrust command must be mixed into the NN output".
- Motor-state choice: "a NN trained on absolute RPMs does not easily transfer from simulation to the real
  world, unless an accurate propulsion subsystem model is available"; replacing ω(t) with ∆e(t): "on
  average 45.07% less error with only an increase of 3.41% in its control signal outputs."
- History: "using a history of errors as input to the NN also led to satisfactory performance. This
  approach has the disadvantage of requiring a state history table".
- Net: "6 inputs, 4 outputs, 2 hidden layers with 32 nodes each … 1344 tunable weights".
- Compute: "can execute at over 2kHz on an ARM Cortex-M7"; MCU "1MB of flash memory, 320KB of SRAM …
  216MHz"; TF AOT runtime deps "24.86 KB"; NN task "can be configured to execute at 2kHz".

## Absent
- No outer-loop learning; no head-to-head real tracking-error table vs PID in the extracted text checked;
  no motor time-constant value.
