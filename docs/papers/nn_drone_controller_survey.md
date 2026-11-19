# Survey — how neural-network quadrotor controllers treat the control stack and its classical parts

Compiled 19/11/2026 by the paper-scout agent. Purpose: positioning input for a weightless (RAM/lookup-table)
attitude controller that outputs CTBR to a classical rate loop, with a Mahony estimator and one float feed-forward
term on yaw. No paper of ours is written yet; nothing here is a claim about our own results.

**Verification status.** Every number below was read in this run from the paper's own text (local PDF text
extraction or arXiv/ar5iv HTML). Items I could not confirm are tagged *unverified*. "Derivation (ours)" marks
arithmetic that is not printed in any paper. Per-paper extractions with absent-lists are in the sibling notes
(`docs/papers/<slug>.md`); PDFs are in `papers/` (gitignored).

**Regime warning.** The rows mix real flight and simulation, mocap and onboard estimation, 27 g and 3 kg vehicles,
rate-loop error (deg/s) and attitude error (deg) and position error (m). Do not compare across rows without checking
the regime column.

## 1. Table

Abbreviations: CTBR = collective thrust + body rates; SRT = single-rotor thrust/motor command; MoCap = motion capture;
DR = domain randomization; n.r. = not reported in the paper.

| # | Paper (id) | Regime | What the net replaces → action space | Classical parts that remain; how the paper words it | Pure / residual / hybrid | Net size and compute | Headline vs baseline | Training; sim-to-real |
|---|---|---|---|---|---|---|---|---|
| 1 | Hwangbo et al. 2017, RA-L. arXiv:1707.05110 | Real AscTec Hummingbird 0.665 kg; Vicon + onboard IMU fusion | Whole position+attitude stack → four rotor thrusts | State estimator (Vicon+IMU); vendor motor speed controllers. Says the net maps "state to actuator command making any predefined control structure obsolete". PD controller used in training only: "does not aid the final controller" | Pure policy (motor level) | 18 inputs, 2×64 tanh; "7 µs per time step" on "a single core of Intel Xeon E5-1620"; onboard Intel Compute Stick (1.44 GHz Atom) | Sim failure rate 4 % vs linear MPC 71 % (100 random starts); recovers from upside-down throw at 5 m/s; tracking "higher than optimization based controllers" | RL (own deterministic policy optimization "using a natural gradient descent"); no DR, no motor model |
| 2 | Koch et al. 2019, Neuroflight. arXiv:1901.06553 | Real FPV quad, STM32F722 216 MHz | Rate (inner) loop + motor mixing → 4 motor powers in [0,1] | Outer loop = pilot; gyro filtering; THRUST MIXED IN CLASSICALLY: "the NN has no concept of thrust … the thrust command must be mixed into the NN output". Still "first open-source neural network-based flight controller firmware" | Pure rate policy + classical throttle mix | 6 in, 2×32, 4 out, "1344 tunable weights"; graph 12 KB; control algorithm WCET 210 µs (NN) vs 15 µs (PID) armed; task WCET 423 µs vs 238 µs; runs at 2 kHz; +70 mA (≈11 %) current | Sim MAE 2.83 vs PID 4.20 (avg). Real flight NN only: MAE avg 15 deg/s. No real PID comparison | PPO in GymFC "digital twin" |
| 3 | Mysore et al. 2021, RE+AL. arXiv:2012.06656 | Real FPV quad, Betaflight fork | Rate loop → 4 motor commands | Rest of Betaflight kept: "we replace this control block with a call to our trained neural network" | Pure rate policy | 2×64, 4 outputs; MCU timing n.r. | Real MAE 4.20±1.14 deg/s vs PID 5.01 vs Neuroflight 5.19; current 5.86±3.10 A vs PID 8.07 A vs Neuroflight 22.87 A | PPO + smoothness-shaped reward |
| 4 | Molchanov et al. 2019, IROS. arXiv:1903.04628 | Real Crazyflie 2.0 (STM32F405 168 MHz) + 2 larger quads | Position+attitude+rate → 4 normalized motor thrusts | Onboard EKF fusing IMU + MoCap. Novelty worded as "(without the use of a stabilizing PD controller)" | Pure policy (motor level) | 18 in, 2×64 tanh; "Evaluating the neural network takes about 0.8 ms"; loop at 500 Hz | Table IV on CF: NN ēθ 1.53°, position error 0.19 m vs Mellinger 1.93°, 0.11 m; recovers 80 % of throws | PPO; DR ("20 %" best), motor-lag model |
| 5 | Kaufmann et al. 2022, ICRA. arXiv:2202.10796 | Sim + BEM sim + real 0.768 kg quad | Compares three levels: LV, CTBR, SRT | CTBR is "tracked by a low-level controller using measurements from an inertial sensor"; openly argues the classical inner loop is what makes transfer work | Pure policy at each level; CTBR = NN + classical rate loop | Two 3×64 encoders + 2×128 MLP; policies at 50 Hz; inference time n.r. | BEM+20 ms delay: SRT crashes on all racing tracks, CTBR 5.6–14.9 cm; real CTBR 4.4±1.4 cm hover; SRT "could not be deployed" (60 ms latency) | PPO; DR (mass/inertia ±30 %) |
| 6 | Kaufmann et al. 2023, Nature (Swift). DOI 10.1038/s41586-023-06419-4 | Real racing quad, onboard VIO, Jetson TX2 | Guidance+position+attitude → CTBR | VIO + Kalman filter; Betaflight "proportional–integral–derivative (PID) controller" on an "STM32 processor operating at 216 MHz"; simulated to "<1% error" | Pure policy at CTBR level over classical PID | "MLP: 2 × 128", 100 Hz; "one inference pass taking 8 ms" on TX2 CPU | "raced this track against three human champions" and won races against each (abstract: "fastest recorded race time") | PPO, 1×10^8 steps; data-driven residual dynamics/perception models, fine-tuning |
| 7 | Song et al. 2023, Sci. Robotics. arXiv:2310.10943 | Real racing quad, Vicon (400 Hz) + EKF | Planning+position+attitude → CTBR | "tracked by a low-level controller that operates at a higher frequency" (not named). Abstract still: "a neural network controller trained with reinforcement learning (RL) outperforms optimal control (OC) methods" | Pure policy at CTBR level over classical loop | 2×256 MLP, 100 Hz; inference time and compute location n.r. | Realistic sim success: RL 100 % vs contouring MPC 20 % vs trajectory tracking 0 %; real: RL 5.35±0.15 s / 85 % vs CC 5.54±0.21 s / 50 %; 108 km/h, 12.58 g | PPO, "ten minutes on a standard workstation"; DR of thrust map and drag; zero-shot |
| 8 | Eschmann et al. 2024, RA-L (Learning to Fly in Seconds). arXiv:2311.13081 | Real 27 g Crazyflie, stock firmware | Position+attitude+rate → 4 RPM setpoints | Stock firmware state estimation ("without modifications to the state estimation in the firmware"); calls CTBR a higher, easier level (3.1) vs its own 5.1 | Pure policy (motor level) | Size not printed in main text (same group's RLtools benchmark uses 13→64→64→4); 500 Hz | Real Lissajous RMSE slow/normal/fast: Ours 0.08/0.17/0.24 m vs PID 0.23/0.72/0.88, Geometric 0.06/0.16/0.36, INDI 0.21/1.13/1.04 | TD3, 18 s on a laptop; no DR; action history + rotor-delay model (without either: 0/10 trajectories) |
| 9 | Eschmann et al. 2024, JMLR (RLtools). arXiv:2306.03530 | MCU inference benchmark, no flight result | — (benchmark of a 13→64→64→4 ReLU MLP) | — | — | Crazyflie STM32F405 (168 MHz, 192 KB RAM): 743 µs generic / 478 µs DSP library / 293 µs optimized; Pixhawk 6C 133/93/53 µs; Teensy 4.1 64/45/41 µs | — | — |
| 10 | Ferede et al. 2023, G&CNet. arXiv:2304.13460 | Real Parrot Bebop 1 (Cortex-A9), OptiTrack | Guidance+control → rpm commands | Onboard state fusion; ONBOARD-ESTIMATED MOMENT OFFSET FED AS A NETWORK INPUT ("this model mismatch is estimated onboard and fed to the network"). Claims "First flight tested, end-to-end neural network controller for quadcopters that does not rely on inner loop controllers" | Pure policy + classical disturbance estimate as input | 3×120 ReLU; rate n.r. (by grep) | More energy-optimal and robust than differential-flatness+INDI in waypoint flight (plots; no deg/m table extracted) | SUPERVISED imitation of 100,000 energy-optimal trajectories |
| 11 | Ferede et al. 2023, E2E RL. arXiv:2311.16948 | Real Bebop 1, measured RPM | Guidance+control → motor commands, vs same net → CTBR → INDI | EKF (OptiTrack+IMU); measured rotor speeds and estimated external force/moment as inputs. States the usual fix is "a robust inner loop controller — an abstraction that, in theory, constrains the optimality" | Pure policy; the CTBR arm is NN + INDI | 3-layer policy; residual nets 1×32; venue *unverified* | E2E ahead by "1.39-second" in sim, "0.17-second" real (fastest run); "greater sensitivity to modeling errors" | PPO |
| 12 | Ferede et al. 2025, ICRA (One Net). arXiv:2504.21586 | Real 3-inch and 5-inch race quads, STM32H743, INDIflight | Guidance+control → 4 motor commands | "An onboard EKF fuses … IMU, and position/attitude observations from an external optical motion capture system". Wording: "end-to-end guidance and control" | Pure policy (motor level) | 3×64 ReLU, 20 inputs, "1000Hz" onboard; inference time n.r. | 0 % DR fails transfer (5–13 gates); 10 % DR: 38–39 / 46–47 gates, mean 7.05–7.13 / 8.63–8.73 m/s | PPO, 100 M steps; DR 0–30 % |
| 13 | Zhang et al. 2022/23 (RMA near-hover). arXiv:2209.09232 | Real 792 g and 177 g quads, MoCap | INVERSE cascade: classical outer loop → learned inner loop → motor speeds | Classical "PD controller which takes as input the goal position and outputs the mass normalized collective trust and the body rates" | Hybrid (classical outer + learned inner) | 3-layer MLP, 256-dim hidden + 1-D CNN adaptation over 400 steps; 500 Hz; venue *unverified* | Inertia-board test: PID success 0 vs ours 100 (small quad) | PPO + RMA adaptation; DR over mass/arm/motor constants |
| 14 | Shi et al. 2019, ICRA (Neural Lander). arXiv:1811.08027 | Real Intel Aero 1.47 kg, MoCap 100 Hz | Nothing replaced: DNN predicts residual force inside a nonlinear position controller | Whole stack classical (attitude > 100 Hz, motor > 5 kHz). Worded as "combines a nominal dynamics model with a Deep Neural Network" | RESIDUAL on a classical controller | Spectrally-normalized ReLU DNN; position loop ≈ 10 Hz | z error 0.13 m → 0 in landing; 0.153 → 0.027 m cross-table | Supervised on flight data |
| 15 | O'Connell et al. 2022, Sci. Robotics (Neural-Fly). arXiv:2205.06908 | Real 2.6 kg quad, wind tunnel, Raspberry Pi 4 + PX4 | Learned basis + online-adapted coefficients in the position loop | PX4 attitude PID and estimator untouched. Worded as "replaces the integral feedback term with an adapted learning term" | RESIDUAL / adaptive hybrid | Position loop in Python at 50 Hz on the onboard Linux computer | "66 % over a nonlinear tracking controller, 42 % over an L1 adaptive controller, and 35 % over" INDI; 12.1 m/s wind: 9.4 cm RMS vs INDI 10.7 cm | Meta-learning on 12 min of flight data + online adaptation |
| 16 | Tagliabue et al. (tube-MPC imitation). arXiv:2109.09910 | Real multirotor, Nvidia TX2, MoCap/VIO | Position MPC → thrust + tilt (roll, pitch) setpoints | Classical attitude controller (MIT/ACL snap-stack) and estimator | Pure policy at attitude-setpoint level over classical attitude loop | "2-hidden layers … 32 neurons per layer", 188 inputs, 3 outputs; 1.66 ms on TX2 CPU vs MPC 55.3 ms (laptop); 100 Hz | 100 % robustness from 1 demonstration (DAgger alone 98 %, baselines need 6–14 demos); wind ≈ 25–30 % of weight; no RMSE in m | DAgger / BC of robust tube MPC with tube-guided data augmentation; zero-shot; venue ICRA 2022 *unverified* |
| 17 | Cobo-Briesewitz et al. 2026, L4DC (Learned INDI). arXiv:2503.09441 | Real Crazyflie 2.1, 34.7 g, STM32 168 MHz | NN predicts residual force/torque in place of RPM-sensor-based INDI | Whole controller classical; "hybrid approach that integrates learning-based predictions with INDI" | RESIDUAL / hybrid, named as such | 19→24→24→6 MLP; "203 microseconds on the Crazyflie hardware" | Tracking tables not extracted (*unverified*) | Supervised on flight data |
| 18 | Stroobants et al. 2025, RA-L (neuromorphic attitude). arXiv:2411.13945 | Real Crazyflie + Teensy 4.0 (Cortex-M7), 35 g | Attitude ESTIMATION + attitude control → body torques | Outer position controller; linear motor mixer; integral realised as hand-fixed integrator neurons. Still "first fully neuromorphic system for attitude estimation and control", while stating it runs on a conventional MCU | Pure SNN for estimator+attitude loop, between classical outer loop and mixer | 150-100-80 spiking neurons (after pruning); 500 Hz on Teensy 4.0; estimate "around 7,500 additions" per update vs PID "about 28 additions and 52 multiplications"; energy NOT measured | Attitude RMSE 3.03° vs PID 2.67°; response time 145 ms vs 125 ms | Supervised imitation of the stock filter + cascaded PID (20 min of flight at 500 Hz) + in-loop data augmentation |
| 19 | Paredes-Vallés et al. 2024, Sci. Robotics. arXiv:2303.08778 | Real drone, event camera, Loihi + UP Squared + PX4 | Vision → ego-motion → thrust, roll/pitch angles, yaw rate | PX4 attitude estimate and all low-level loops: commands "processed by the low-level controllers there". Scoped openly: "limited ourselves to the vision-to-control pipeline, ending at thrust and attitude commands" | Pure SNN (vision) + linear decoder, over classical autopilot | 28.8k neurons; 200 Hz; 27 µJ per inference on Loihi vs 75,347–85,577 µJ on Jetson Nano | Qualitative setpoint tracking; steady-state error noted; no RMS | Vision self-supervised; control decoder by mutation-only GA (population 100) in sim + DR |
| 20 | Kresse & Lampert 2026, ICML (Differentiable Weightless Controllers). arXiv:2512.01467 | MuJoCo SIMULATION only; FPGA synthesis results | Whole policy (Ant, HalfCheetah, Hopper, Humanoid, Walker2d) | None discussed (no robot) | Pure policy | Thermometer inputs, two layers of 6-input LUTs, 1024 per layer; Artix-7 XC7A15T @ 100 MHz: e.g. Hopper 3.2k LUTs, 0 DSP, 0 BRAM, 0.03 µs, 2.3×10^−9 J/action; "multiplication-free inference" | Returns "competitive with standard deep policies"; quantized-NN baseline 45 DSP, 0.21 µs, 6.5×10^−8 J | RL (SAC/PPO) with gradient surrogates |
| C1 | Alavilli, Nguyen, Schoedel et al., TinyMPC. arXiv:2310.16985 (NON-NN comparator) | Real 27 g Crazyflie 2.1, STM32F405, MoCap 100 Hz | Full-state MPC → PWM motor commands | MoCap state | Classical | 500 Hz, N = 15, n = 12, m = 4, within "2 ms"; "8.85x over OSQP"; measured µs n.r. | 90° recovery: max position error 23 cm vs PID 40 cm, Brescianini 65 cm | — (venue ICRA 2024 *unverified*) |

Related but only abstract-level read: Kresse & Lampert, "Learning Quantized Continuous Controllers for Integer
Hardware", arXiv:2511.07046 — 2–3 bit weights, Artix-7, five MuJoCo tasks, no robot (numbers beyond the abstract
*unverified*). Tagliabue & How, "Output Feedback Tube MPC-Guided Data Augmentation…", arXiv:2210.10127 — NOT opened.

### Small-MLP-on-MCU reference numbers (all printed in the papers)

| Net | MCU | Time | Source row |
|---|---|---|---|
| 6→32→32→4 (1,344 weights) | STM32F722, 216 MHz | 210 µs WCET (algorithm), 423 µs (whole task); PID 15 µs | 2 |
| 13→64→64→4 | STM32F405, 168 MHz (Crazyflie) | 293 µs optimized / 478 µs DSP lib / 743 µs generic | 9 |
| 18→64→64→4 | STM32F405, 168 MHz (Crazyflie 2.0) | "about 0.8 ms" | 4 |
| 19→24→24→6 | STM32, 168 MHz (Crazyflie 2.1) | 203 µs | 17 |
| 3×64, 20 inputs | STM32H743 | n.r.; runs at 1000 Hz | 12 |
| 13→64→64→4 | Teensy 4.1 (600 MHz M7) | 41 µs optimized | 9 |
| SNN 150-100-80 | Teensy 4.0 | ≤ 2 ms (500 Hz) | 18 |
| MPC, N = 15 | STM32F405 | ≤ 2 ms budget (500 Hz) | C1 |

Derivation (ours): weight-multiply counts are 6·32+32·32+32·4 = 1,344 (matches Koch's printed figure);
13·64+64·64+64·4 = 5,184; 18·64+64·64+64·4 = 5,504; 19·24+24·24+24·6 = 1,176. Per-multiply time is then roughly
293/5,184 ≈ 0.06 µs (RLtools optimized, F405), 743/5,184 ≈ 0.14 µs (generic, F405), 203/1,176 ≈ 0.17 µs (F405),
210/1,344 ≈ 0.16 µs (F722). A 13→64→64→4 float32 MLP is 5,316 parameters ≈ 21 kB (5,316 × 4 B). None of these
per-multiply or kB figures are printed in the papers.

## 2. Answers

### (a) Is "NN + classical inner loop + estimator" the norm, and is it still called a neural controller?

Yes, on both counts, in the papers that fly fastest and are cited most.

- Rows 5, 6, 7, 16, 19 put the network above a classical rate or attitude loop. Song (row 7) writes "a neural
  network controller" in the abstract of a Science Robotics paper while the body says the output is "tracked by a
  low-level controller"; Swift (row 6) names Betaflight PID explicitly.
- Kaufmann 2022 (row 5) is the paper that turns this into an argument: CTBR over a classical rate loop transfers,
  motor-level did not fly. That is direct support for our CTBR choice.
- EVERY row keeps a classical state estimator (EKF, VIO, mocap fusion, PX4 attitude estimate), including the ones
  that claim "end-to-end" (rows 10–12). The only exception is row 18, which learns the attitude estimator too.
- "End-to-end" has a specific meaning in this literature: no inner loop, network commands motors (rows 10–12, and
  the taxonomy in row 8). Ferede's highlight is "does not rely on inner loop controllers". A CTBR-output controller
  should not use the phrase.
- I found no reviewer statements; this answer is about how authors word it, not how referees received it.

### (b) How common are explicit classical terms added to the NN output, and how are they framed?

Common, in three distinct patterns, and the framing follows the pattern.

1. **Residual on a classical controller** (rows 14, 15, 17). The classical law is the controller; the net is called
   "learned dynamics", "adapted learning term", or "hybrid approach". Nobody in this group claims the NN is the
   controller.
2. **Classical term mixed into a policy's output** (row 2). Neuroflight adds the thrust command to the network
   output and says so in one sentence, while still titling itself a neural flight controller. This is the closest
   precedent to our yaw feed-forward (constant × estimated trim added to the table output).
3. **Classical bias/disturbance estimate routed through the network's INPUT** (rows 10, 11) or built into the
   network as fixed integrator units (row 18). These keep the "network is the controller" wording intact, because the
   last operation before the actuator interface is still the network.

Two further data points: row 1 uses a PD controller during training and states it "does not aid the final
controller"; row 4 lists "(without the use of a stabilizing PD controller)" as part of its novelty claim. So the
presence of a classical feedback term is something this community notices and authors pre-empt.

Implication for the P-term debate (interpretation, not a quoted finding): a constant-gain feed-forward on an
estimated trim sits in pattern 2 and is defensible with one sentence plus an ablation. A proportional FEEDBACK
term on yaw error summed with the table output moves that axis into pattern 1 — a reader can then ask what the
P term alone achieves, and the honest framing becomes "hybrid" on that axis. The row-18 and row-10 precedents
suggest the lower-risk route if the yaw gap must be closed: feed the quantity to the network as an input, or
report the three-way ablation (table only / + feed-forward / + P) so the contribution of each part is on the page.

### (c) Does anyone claim multiplier-free or lookup-only inference for flight control?

No. I found no weightless, WiSARD, RAM-based, LUT-network, binarized or logic-gate-network controller flown on a
multirotor. Five targeted searches (weightless/WiSARD + UAV; LUT/binarized/multiplier-free + quadrotor + FPGA;
LogicNets/PolyLUT/NeuraLUT/logic-gate networks + control; FPGA quantized policy + quadrotor; RAM-based/n-tuple/CMAC
+ quadrotor) returned classification benchmarks, fuzzy/PID FPGA designs and CMAC reviews, but no flight controller.
Absence from a web search is not proof of absence.

The nearest evidence, by type:

- **Row 20 (Kresse & Lampert)** makes the multiplication-free claim for continuous control with real hardware
  numbers — LUT counts, 0 DSP blocks, 2–3 clock cycles, about 2 nJ per action on an Artix-7 — but in MuJoCo only.
  It names UAVs as motivation and cites Swift. It is trained by gradient surrogates under RL.
- **Row 18 (Stroobants)** is the only flight paper that counts operations: about 7,500 additions per update against
  the PID's 28 additions and 52 multiplications, and it concedes the two are in the same energy order of magnitude
  on a conventional MCU. No measurement.
- **Row 19 (Paredes-Vallés)** has measured energy (27 µJ per inference on Loihi) but for the vision network; the
  control part is a linear decoder and the flight loops are PX4.
- **Row 2 (Koch)** gives measured MCU cost of a conventional NN controller: 210 µs against 15 µs for PID, and an
  11 % current increase.

### (d) What would be defensible and novel, and what would be attacked?

Defensible, given this table (each still needs our own measurements):

- **First weightless / lookup-table network as the attitude controller of a quadrotor**, to the extent our evidence
  is hardware or high-fidelity simulation. Row 20 is the prior art to cite and differentiate from: simulation-only
  locomotion tasks, gradient-trained, FPGA target. If our result is also simulation-only, the novelty is the task
  (attitude control in a cascade with estimator and motor lag) and the training route, not "first on a robot".
- **Training route**: DAgger from an MPC teacher into a network trained by memory writes, with architecture found
  by a GA. Row 16 is the precedent for MPC imitation with an MLP; row 20 for LUT policies via RL; row 19 for a GA
  (on a linear decoder only). I found no paper combining them.
- **The cascade itself**. CTBR over a classical rate loop with a classical estimator is the mainstream
  architecture (rows 5–7) and can be stated plainly, citing row 5 for why.
- **Timing determinism**, if measured. Row 2 reports a variability window of 7.1 % for the NN against 40.0 % for
  PID; a table lookup has a fixed path too.
- **A scoped arithmetic claim**: "no multiplications inside the network" is supportable in the same way row 20 and
  row 18 word it, provided it is scoped to network inference.

Likely to be attacked:

- **"Multiplier-free controller"** unscoped. The Mahony filter, the rate PID, the mixer, input scaling before
  thermometer encoding and the float yaw feed-forward all multiply. Row 18 shows the expected standard: count the
  operations and admit what the surrounding stack costs.
- **Compute advantage on an MCU.** A 2×64 MLP already runs in 293 µs on the Crazyflie's STM32F405 and a 2×32 in
  about 210 µs on an F722 (reference table above); row 4 flew at 500 Hz with 0.8 ms inference in 2019, and TinyMPC
  fits a 500 Hz MPC on the same chip. An attitude loop at 500 Hz to 1 kHz is therefore not compute-limited for an
  MLP. A speed claim needs our measured microseconds on the same MCU next to these, and is only interesting if it
  is well under the roughly 200–300 µs band or if the target has no FPU or is an FPGA.
- **Memory.** A 2×64 MLP is about 21 kB of float32 weights (our derivation) and Neuroflight's whole graph is 12 KB.
  RAM-neuron tables will be compared against that directly; a footprint in the hundreds of kB or more hands the
  reviewer the trade-off "faster but far larger".
- **Tracking quality versus the classical baseline.** The norm is to tie or beat PID/INDI/MPC on the reported
  metric (rows 3, 7, 8, 15); row 18 is published with a loss (3.03° vs 2.67°) but frames it as feasibility. A
  weightless controller that trails PID must be framed the same way.
- **"The NN is the controller"** once a P term is added to its output (see b).
- **Energy.** No flight paper here measured controller energy on an MCU; rows 19 and 20 measured on Loihi and FPGA.
  An energy claim without a measurement on a named device will not survive.

## 3. Not found / not verified

- No weightless or LUT-network flight controller (see c).
- Venues *unverified*: rows 11, 13, 16, C1. DOIs for rows 7, 12, 19 come from repository/search hits, not the PDFs.
- Row 8 network size is not in the main text; row 10 control rate not found; row 17 tracking tables not read.
- Rows 5, 6, 11, 13, 14, 15: details beyond the cells above rely on the library notes of 06/11/2026; the cell values
  were re-checked against the local PDF text on 19/11/2026.
- Not opened: arXiv:2210.10127 (Tagliabue & How, output-feedback variant), WarpMPC (arXiv:2607.11603, seen only as
  a search snippet), Neural-Swarm, DATT, and any Crazyflie AI-deck (vision) work.
