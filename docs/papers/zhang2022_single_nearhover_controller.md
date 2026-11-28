# Zhang, Loquercio, Wu, Kumar, Malik, Mueller 2022/23 — Learning a Single Near-hover Position Controller for Vastly Different Quadcopters

Zhang, D., Loquercio, A., Wu, X., Kumar, A., Malik, J., Mueller, M. W. arXiv:2209.09232 (v2, 2 May 2023).
https://arxiv.org/abs/2209.09232 — Local PDF: `papers/zhang2022_single_nearhover_controller.pdf`.
Venue: NOT confirmed from the PDF or search (HiPeR Lab hosts a 2023 copy). Read 06/11/2026.

**Relevance.** RMA-style adaptive quad control — and notably the INVERSE cascade: a classical outer
loop emits CTBR, the learned policy is the inner loop outputting motor speeds, with an adaptation
module reading state-action history. **Regime:** PPO in Flightmare, real 792 g and 177 g quads, mocap.

## Verbatim extraction
- "This controller takes a history of platform states and commands from a platform-independent high level
  controller as inputs, and outputs individual motor speed."
- Outer loop (sim): "cascaded linear acceleration controller … natural frequency 2rad/s and damping
  ratio 0.7 … body rates … proportional to the attitude error angle, with a time constant of 0.2s."
  Real: "a PD controller which takes as input the goal position and outputs the mass normalized
  collective trust and the body rates."
- Motors: "we model the motors' response using a first-order system". Policy "500Hz", sim step "2ms",
  "observation latency of 10ms".
- Adaptation module: "projects the latest 400 state-action pairs" → 1-D CNN → latent z (R^6).
  [Derivation (ours): 400 × 2 ms = 0.8 s of history.]
- Table I training ranges: mass [0.142, 0.950] kg; arm [0.046, 0.200] m; motor constant [1.15e-7, 7.64e-6];
  max motor speed [707, 4895] rad/s; payload [10, 50] % of mass.
- Table II (real, free hover, small/large): Height err PID 0.06/0.01, LTF 0.09/0.03, Ours 0.05/0.03 m;
  Ang-vel err PID 0.51/0.12, Ours 0.30/0.19 rad/s. Inertia board: PID small success 0, Ours 100;
  large: PID height err 0.31 m vs Ours 0.08 m.
- Latency: "L1 is not implemented in our real-world experiments as a baseline because it is highly
  sensitive to latency … our approach can handle latency very well by simulating it during training."

## Absent
- Motor time constant value and any DR on it: not in Table I.
- Venue not confirmed.

## Robustness protocol addendum (read 28/11/2026 from local PDF)
- Table I "Ranges of the drone and environmental parameters" — Training Range / Testing Range:
  Mass (kg) [0.142, 0.950] / [0.114, 1.140]; Arm length (m) [0.046, 0.200] / [0.037, 0.240];
  Inertia x,y (kg·m²) [7.42e-5, 5.60e-3] / [5.94e-5, 6.72e-3]; Inertia z [1.20e-4, 8.80e-3] / [9.60e-5, 1.06e-2];
  Propeller constant κ (m) [0.0041, 0.0168] / [0.0033, 0.0201]; Payload (% of Mass) [10, 50] / [5, 60];
  Payload location (% arm) [-10, 10] / [-10, 10]; Motor Constant [1.15e-7, 7.64e-6] / [9.16e-8, 9.17e-6];
  Body drag coefficient [0, 0.62] / [0, 0.74]; Max. motor speed (rad/s) [707, 4895] / [566, 5874].
- Derivation (ours): test bounds = train lower × 0.8, train upper × 1.2 (e.g. 0.142×0.8 = 0.114, 0.950×1.2 = 1.14,
  4895×1.2 = 5874) — every parameter widened ~20% at each end, drawn JOINTLY ("We randomize each of these parameters at
  the end of each episode"; plus a mid-episode re-randomization of mass, inertia, CoM).
- "observation latency of 10ms" in training/sim.
- Table III (sim, test ranges, "Metrics are averages over 100 experiments"): success / height err m — Robust 18% / 0.44;
  SysID 38% / 0.19; LTF 59% / 0.35; L1 59% / 0.17; Ours 66% / 0.09; Expert 69% / 0.09.
- Table IV (sim, "disturbances unseen at training time", each type SEPARATE): External Forces "uniformly sampled
  between 0 and 50% of the weight"; Partially Failing Motors "multiply ... a randomly sampled motor's thrust force to a
  random number between 0 and 1". External Forces success: Robust 5%, SysID 2%, LTF 32%, L1 42% (rest of table not
  transcribed).
- Real (Table II): "Metrics are averaged over 5 experiments".
- Still absent: motor time-constant value/DR; sensor-noise multiplier.
