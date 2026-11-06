# Hwangbo, Sa, Siegwart, Hutter 2017 — Control of a Quadrotor with Reinforcement Learning

Hwangbo, J., Sa, I., Siegwart, R., Hutter, M. *Control of a Quadrotor with Reinforcement Learning.*
IEEE Robotics and Automation Letters (preprint June 2017). arXiv:1707.05110.
Local PDF: `papers/hwangbo2017_quadrotor_control_rl.pdf`. Read 06/11/2026.

**Relevance.** Early real-hardware demo of a NN mapping state directly to rotor thrusts (SRT) with no
motor model at all. **Regime:** AscTec Hummingbird (has onboard motor-speed controllers), Vicon + MSF state.

## Verbatim extraction
- Abstract: "a common network can be trained to directly map state to actuator command making any
  predefined control structure obsolete"; "Computation time of evaluating the policy is only 7 µs per
  time step"; recovery from "manually throwing it upside-down in the air with an initial velocity of 5 m/s".
- Sim: "We used a very simple model … ignore all drag forces … floating body model with four thrust
  forces"; integration time step "(0.01 s)". Inertia "simplified it to a diagonal matrix since we did not
  measure the inertia".
- PD used during training only: "The PD controller is used to stabilize the learning process but it does
  not aid the final controller"; gains kp = −0.2, kd = −0.06 (z one sixth).
- State: 18-dim (rotation matrix 9 + position, lin vel, ang vel); 2 hidden layers of 64.
- Table I: mass 0.665 kg; Ixx,Iyy,Izz 0.007, 0.007, 0.012 kg m².
- Sim result: failure rate learned policy 4 % vs linear MPC 71 % over 100 random-state rollouts.
- Real gap: "how the motor controllers regulate the motor speed is unknown. The rotors can be
  accelerated relatively fast but decelerates quite slowly. We do not get any feedback of the motor speed
  and the identification of its dynamics is missing at this point."
- Steady-state error in sim "(1.3 cm)". Tracking error "higher than optimization based controllers".
- Future work: "train RNN which can adapt to errors in modeling automatically".

## Absent
- NO motor dynamics/time constant in sim; NO action history; NO domain randomization; NO noise model.
- No quantitative real-world tracking RMSE table.
