# Kaufmann et al. 2023 — Champion-level drone racing using deep reinforcement learning (Swift)

Kaufmann, E., Bauersfeld, L., Loquercio, A., Müller, M., Koltun, V., Scaramuzza, D. *Champion-level
drone racing using deep reinforcement learning.* Nature 620, 982–987 (31 Aug 2023).
https://doi.org/10.1038/s41586-023-06419-4 — Local PDF: `papers/kaufmann2023_swift_champion_drone_racing.pdf`. Read 06/11/2026.

**Relevance.** The flagship real-world learned quad controller uses CTBR + Betaflight PID inner loop,
citing the ICRA'22 benchmark as justification. **Regime:** real racing quad, PPO in sim with
data-driven residual dynamics/perception models, fine-tuned.

## Verbatim extraction
- Methods, "Betaflight low-level controller": "the neural network outputs collective thrust and body
  rates. This control signal is known to combine high agility with good robustness to
  simulation-to-reality transfer44" (ref 44 = Kaufmann, Bauersfeld & Scaramuzza ICRA 2022).
- "processed by an onboard low-level controller that computes individual motor commands … this
  low-level proportional–integral–derivative (PID) controller and ESC are implemented using the
  open-source Betaflight and BLHeli32 firmware45. In simulation, we use an accurate model of both the
  low-level controller and the motor speed controller."
- Betaflight peculiarities modelled: "the reference for the D-term is constantly zero (pure damping),
  the I-term gets reset when the throttle is cut and, under motor thrust saturation, the body rate
  control is assigned priority"; "gains … identified from the detailed logs"; "The simulation can
  predict the individual motor commands with less than 1% error."
- Action: "at ∈ R4 in the form of desired mass-normalized collective thrust and body rates."
- Observation "ot ∈ R31": state estimate (R15: position, velocity, rotation matrix), next-gate corners
  (R12), and "(3) the action applied in the previous step".
- Rates (Fig. 1 diagram): IMU 200 Hz, VIO 100 Hz, control policy 100 Hz ("MLP: 2 × 128"), gate detector 30 Hz.
- Training: 1 × 10^8 env interactions; fine-tuning 2 × 10^7. Residual dynamics via k-NN regression,
  residual perception via Gaussian processes.
- Motor model: kmot appears as "The motor time constant is kmot" in Methods eq.; ESC is open-loop
  ("the ESC does not perform closed-loop control of the motor speeds") with battery-voltage model.

## Absent
- Numeric kmot not found in the extracted text (Extended Data tables not checked line-by-line).
- No comparison against a motor-command (SRT) policy in this paper.
