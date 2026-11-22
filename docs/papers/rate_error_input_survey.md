# Survey: body-rate tracking error / inner-loop integrator state as learned-controller INPUT

Compiled 22/11/2026. Question: do learned quadrotor controllers observe the body-rate tracking error
(ω* − ω, its Δ or ∫), or the inner-loop PID integrator, as a model-free way to sense or adapt to
disturbances? And which disturbance estimators run without a plant model?
Every quote below comes from a PDF or HTML page opened during this run. Lines marked
"(prior note)" come from library notes written in an earlier run, with the PDF re-grepped this run.

## Answer
- **Rate tracking error as input: YES, but only for learned controllers that REPLACE the rate loop.**
  Koch 2019 (GymFC), Neuroflight and RE+AL are inner-loop policies whose input is e = Ω* − Ω. Yu & Lee
  2023 feed e_Ω to a full-stack policy, in simulation only. None of them frames e as a disturbance sensor.
- **A learned OUTER loop reading the rate error of a firmware rate PID underneath it: NOT FOUND.**
- **The inner-loop integrator state as an input: NOT FOUND in any quadrotor paper.** The nearest
  match is Saj 2026: a learned outer loop over the Crazyflie PID rate loop, whose disturbance estimator
  (a GRU) reads the inner-loop PID's motor PWM outputs. Those outputs contain the I-term implicitly.
  Char & Schneider 2023 (NeurIPS) feed P/I/D of the tracking error to RL policies, but on generic
  control tasks and a tokamak, not on drones.

## Per-paper extraction

### 1. Koch, Mancuso, West, Bestavros — Reinforcement Learning for UAV Attitude Control
ACM Trans. Cyber-Physical Systems 3(2), Art. 22, Feb 2019. DOI 10.1145/3301273. arXiv:1804.04154.
PDFs: `papers/koch2019_rl_uav_attitude_control_tcps.pdf` (TCPS), `papers/koch2018_..._arxiv.pdf`.
**Regime:** simulation only (Gazebo, Iris 1.5 kg). Policy = inner rate loop → 4 motor signals.
- "the agent receives an observation x t from the environments consisting of the angular velocity error
  of each axis e = Ω∗ − Ω and the angular velocity of each rotor ωi" (gyro + ESC).
- "The state is of size m × (M + D) where m is the memory size"; M = 4, D = 3. Memory m ∈ {1,2,3}; m = 1 was best.
- Table 3 (1,000 commands, 95% CI), Error in rad/s, φ/θ/ψ: PPO m=1 317.0±11.0 / 326.3±13.2 / 217.5±9.1;
  PID 416.1±20.4 / 269.6±11.9 / 245.1±11.5. Rise in ms: PPO 66.6/70.8/72.9, PID 79.0/99.8/67.7.
- On board: no plant model. The policy is model-free (PPO). Disturbances are motivation only ("wind, variable
  payloads, voltage sag"); none are tested. No integral of e is observed.

### 2. Koch, Mancuso, Bestavros — Neuroflight (arXiv:1901.06553) (prior note, `koch2019_neuroflight.md`)
**Regime:** real FPV quad, STM32F722. NN replaces the rate PID and the mixer.
- "x(t) = [e(t), ∆e(t)]"; replacing ω(t) with ∆e(t) gives "on average 45.07% less error".
- "6 inputs, 4 outputs, 2 hidden layers with 32 nodes each". Real MAE avg 15 deg/s, with gusts
  "exceeded 30mph". No PID comparison in real flight. No integral input. No disturbance estimator.

### 3. Mysore et al. 2021 — RE+AL (arXiv:2012.06656) — observation NOT VERIFIED
Built on Neuroflight/GymFC ("RE+AL improves and completes Neuroflight"). The reward penalises
"the difference between the set point and the actual angular rate". The observation vector was **not
found printed** in the PDF (grep this run). Do not cite it as observing e. Real MAE 4.20 deg/s vs PID 5.01.

### 4. Yu & Lee 2023 — Multi-Agent RL for the Low-Level Control of a Quadrotor UAV
arXiv:2311.06144v2 (26 Feb 2024). No venue listed. `papers/yu2023_marl_lowlevel_quadrotor.pdf`. **Simulation only.**
- Observations: "o1=(ex,eIx,ev,b3,eω12)∈ℝ15", "o2=(b1,eΩ3)∈ℝ4". The SARL baseline uses "o=(ex,eIx,ev,R,eΩ)".
  This includes the angular-rate error AND a leaky integral of POSITION error: ė_Ix = −α e_Ix + e_x, α = 0.01.
- No integral of the rate error. No disturbance tests (±5% parameter randomisation only). No real flight.
  (Taken from the arXiv HTML via the fetch summariser. The PDF is stored.)

### 5. Char & Schneider 2023 — PID-Inspired Inductive Biases for Deep RL in Partially Observable Control Tasks
NeurIPS 2023. arXiv:2307.05891. `papers/char2023_pid_inspired_inductive_biases.pdf`.
- The PIDE encoder gives the policy the tracking error, its integral and its derivative. "For SISO problems,
  policies with this encoder have access to the same information as a PID controller."
- Claims the encoding "is significantly more resilient to changes in the dynamics during test time".
  It reports "1.7x better performance on average" on PyBullet locomotion.
- **No quadrotor. No real hardware.** The tokamak "real" setting is a learned simulator.

### 6. Saj, Vemuri, Kalathil, Benedict 2026 — Adaptive Outer-Loop Control of Quadrotors via RL
arXiv:2605.16015v1 (15 May 2026). Presented at VFS 82nd Annual Forum, May 2026. `papers/saj2026_adaptive_outer_loop_rl.pdf`.
**Regime:** real Crazyflie. The policy runs off-board and sends thrust + body rates to the "onboard, high-bandwidth PID rate controllers".
- Disturbance estimator (RDP, GRU) input: "xt ∈ R26 consists of the 18-dimensional vehicle kinematic state,
  the 4-dimensional previous outer-loop action, and crucially, the 4-dimensional Pulse Width Modulation (PWM)
  signals commanded to the individual motors by the inner-loop PID controller".
  The paper explains why: "higher-level thrust and body-rate setpoints fail to uniquely capture transient
  actuator states, motor lag, and inner-loop saturations".
- There is also an online integral thrust bias, F_bias ← F_bias + η(z_des − z). This is a hand-made outer integrator.
- Position-hold RMSE under central payload 0–25%: Base 0.074 → 0.160 m; Adaptive+RDP 0.024–0.030 m
  (from the fetch summariser; Table 2 is in the stored PDF).
- **Absent:** no rate tracking error and no inner-loop integrator state in the policy or RDP input.
  No plant model on board (the GRU is learned in simulation). The sim-to-real "calibration bridge" was fitted on 3 flight samples.

### 7. Huang, Rana, Spitzer, Shi, Boots 2023 — DATT (arXiv:2310.09053v3; venue NOT verified)
`papers/huang2023_datt.pdf`. **Regime:** real Crazyflie 2.1, OptiTrack, off-board policy.
- "We use L1 adaptive control … to estimate d, which is directly passed into our policy".
- The inner loop is "PI attitude rate controller on the firmware: τ = −KPω (ω − ωdes ) − KIω (ω − ωdes )".
  As printed, the I term also multiplies the instantaneous error.
- **The L1 estimator NEEDS a nominal model.** Its predictor uses the thrust term R e3 f_Σ / m, so it needs mass and a thrust map.
- Plate + wind, smooth trajectory: DATT 0.095±0.053 m vs L1-MPC 0.181±0.04 m.
- Absent: ω and the rate error are not in the policy observation; no integrator state.

### 8. Zhang et al. 2022/23 — single near-hover controller, RMA-style (prior note; arXiv:2209.09232)
The adaptation module "projects the latest 400 state-action pairs" through a 1-D CNN to z ∈ R^6. This is model-free
implicit adaptation. The policy is the INNER loop (outputs motor speeds) under a classical outer loop.
Inertia-board test: PID success 0 vs Ours 100 (small quad). Venue not confirmed.

### 9. Eschmann et al. 2024 (prior note) — an action history is added "to mitigate this large level of partial
observability" from motor delay. It is not used for disturbance adaptation.

### 10. Kaufmann et al. 2023 Swift (prior note) — CTBR over the Betaflight PID. The observation "ot ∈ R31" = state,
gate corners and the previous action. **No rate error and no PID state.**

### 11. Hwangbo 2017 / Molchanov 2019 (PDFs re-grepped this run)
- Hwangbo: "nine elements of the rotation matrix … position, linear velocity and angular velocity … 18-dimensional
  state vector". Measured ω only, no error and no integral.
- Molchanov: "we input only the error between the current and goal state, except for the rotation matrix".
  The angular-velocity error is relative to the goal state, not to a cascade rate setpoint. No integral and no history.

### 12. O'Connell et al. 2022 Neural-Fly (Sci. Robotics, DOI 10.1126/scirobotics.abm6597) (PDF re-grepped)
Neural-Fly "replaces the integral feedback term with an adapted learning term" in the POSITION loop, over the PX4 attitude PID.
- The adaptation target is the measured residual force from "mv̇ = mg + Rfu + f", where "fu … predicted by the
  nominal model". **This needs a partial plant model: mass plus the rotor thrust model.**
- At 12.1 m/s wind: 9.4 cm RMS vs INDI 10.7 cm.

### 13. Smeur, Chu, de Croon 2016 — Adaptive INDI for Attitude Control of MAVs
J. Guidance, Control, and Dynamics 39(3), 2016. DOI 10.2514/1.G001490. `papers/smeur2016_adaptive_indi.pdf` (mirror copy).
**Regime:** real Parrot Bebop, Paparazzi, loops "at 512 Hz". This is NOT a learned controller.
- "only uses a control effectiveness model and uses estimates of the angular accelerations to replace the rest
  of the model". "The control effectiveness is the sole model still required by INDI."
  So INDI is sensor-based but needs a PARTIAL model, G1/G2. Adaptive INDI estimates G online by least mean squares,
  but "It is generally not possible to take off without any estimate of the control effectiveness".
- CoM-offset test: "a weight of 42.5 g" at "about 11 cm to the center of gravity". INDI pitch back to zero
  "After 0.3 s"; PID "about 1.5 s … approximately five times longer"; motor-speed change "578 rounds per minute".

## Could NOT verify
- Pi et al. 2021, Applied Sciences 11(7):3257 (RL + disturbance compensation): the MDPI PDF returned HTML/403. Not read.
- MDPI Actuators 11(4):105 (PPO-PPD; cites a "PPO-IC" state integrator, ref [39]): 403. Not read.
- Mysore 2021 observation vector: not printed in the PDF.
- Venues for Zhang 2022/23 and DATT: not printed in the text that was read.

## Absent across everything read
No paper feeds the firmware rate-PID integrator state, or the rate tracking error of an underlying
rate PID, to a learned outer-loop controller. No paper uses rate error as an explicit model-free
disturbance sensor. The model-free adaptation routes that were found are: (a) state-action history
encoders (Zhang/RMA, Saj RDP GRU); (b) P/I/D-of-error encoders (Char & Schneider, not drones);
(c) a hand-made outer integral (Saj F_bias; Yu & Lee leaky ∫e_x). INDI, L1 and Neural-Fly all need a partial model.
