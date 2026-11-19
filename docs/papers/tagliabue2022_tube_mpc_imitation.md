# Tagliabue, Kim, Everett, How — Demonstration-Efficient Guided Policy Search via Imitation of Robust Tube MPC

arXiv:2109.09910. Venue not printed in the arXiv PDF (ICRA 2022 — unverified). Local PDF:
`papers/tagliabue2022_tube_mpc_imitation.pdf`. Read 19/11/2026 (ar5iv HTML + PDF text grep).
Sibling NOT opened this run: "Output Feedback Tube MPC-Guided Data Augmentation…" arXiv:2210.10127 (vision-based).

**Relevance.** Closest training-method precedent to ours: DAgger/BC imitation of an MPC teacher, policy outputs
attitude+thrust setpoints to a classical attitude controller. **Regime:** real multirotor, Nvidia TX2 CPU onboard,
mocap or onboard VIO, MIT/ACL snap-stack attitude controller.

## Verbatim extraction
- Policy: "a 2-hidden layers, fully connected DNN, with 32 neurons per layer"; input dimension "188 (position,
  velocity, current tilt …" + reference); output dimension "3 (desired thrust and tilt …)".
- Compute: expert RTMPC (MATLAB) average 55.3 ms on i7-8750H laptop; compressed policy 0.25 ms on the laptop,
  "TX2 CPU (PyTorch) is 1.66ms"; runs "at 100Hz on the onboard Nvidia TX2 (on its CPU)".
- Classical remainder: "MIT/ACL open-source snap-stack [41] for controlling" attitude; state from mocap/VIO.
- Results (Table I, T1 wind): SA-Sparse reaches 100% robustness from 1 demonstration (DAgger alone 98%, baselines need
  6–14 demos); expert gap 3.75% (SA-Sparse+DAgger). Disturbances "≈25–30% of the UAV weight". Zero-shot sim-to-real from
  "a single demonstration collected in simulation"; real figure-eight against "an array of 3 leaf blowers".

## Absent
- No tracking RMSE in m or deg; no real-flight trial counts; no MCU deployment; no parameter count or MAC count.
- "Expert gap" is relative action error, not flight error.
