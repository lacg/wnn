# Eschmann, Albani, Loianno 2024 — Learning to Fly in Seconds

Eschmann, J., Albani, D., Loianno, G. *Learning to Fly in Seconds.* arXiv:2311.13081v2 (rev. 08 Apr 2024;
"Accepted April, 3, 2024" — IEEE RA-L). https://arxiv.org/abs/2311.13081
Local PDF: `papers/eschmann2024_learning_to_fly_in_seconds.pdf`.

**Relevance.** RL sim-to-real on a 27 g Crazyflie with a deliberately SLOW motor model. **Regime:** an
ASSUMED simulator constant chosen for transfer, not a measurement.

## Verbatim extraction (§IV)
- "Common values for the RC equivalent time constant (1 − e−1 ≈ 63% step response) are between **0.05 s and
  0.25 s**. We empirically find **0.15 s** to be suitable for Sim2Real transfer which is also supported by the
  manufacturers measurements" (footnote → Bitcraze M1 step-response PNG).
- Transfer is "without domain randomization" of the motor constant.

**Derivation (ours):** 0.15 s is τ (63%), so T_ours = 4 × 0.15 = **0.60 s** — beyond our T=0.30 axis.
Note the same group's later flight-data fit gives τ = 0.072 s (eschmann2024_sysid_motor_delays.md).

## Absent
- No citation for the 0.05–0.25 s range; no numbers read from the manufacturer plot; no DR on τ;
  no CF version/brushed label (mass 27 g only).

## Addendum 06/11/2026 — action space, action history, ablation (for the CTBR-vs-motor decision)
- Output level: "RPM setpoints as a = {ωsp1 , ωsp2 , ωsp3 , ωsp4 } which places our policy/controller on
  the lowest order (level 5.1)". Their taxonomy places CTBR at 3.1 with "only one domain parameter (mass)";
  "the reality gap and complexity of the system greatly increases towards the lowest levels like 5.1".
- Motor lag handling: "these delays lead to actions only impacting the state after 5 to 25 control steps.
  To mitigate this large level of partial observability, we add a history of control actions to the
  actor's observation." Actor obs "18+NH ·4 dimensional"; critic (asymmetric) sees true RPMs. NH value
  not found in the main text.
- Transfer claim: "Sim2Real transfer without domain randomization".
- Table II ablation (component REMOVED; real-world # successful / mean xy error e [m]):
  All components 10/10 0.26 (pos @300k), 10/10 0.34 (pos @3M), 10/10 0.27 (traj @3M);
  **Action History 5/10 1.18; 5/10 0.63; 0/10 ∞**; **Rotor Delay 0/10 ∞; 0/10 ∞; 0/10 ∞**.
  Text: "training without simulating the rotor delay leads to no usable policy … not including the action
  history … considerable drop in reliability and positional accuracy".
- Table III (real Lissajous, RMSE e [m], slow/normal/fast): PID 0.23/0.72/0.88; Geometric 0.06/0.16/0.36;
  INDI 0.21/1.13/1.04; Ours 0.08/0.17/0.24.
