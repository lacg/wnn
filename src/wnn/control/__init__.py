"""WNN-based drone controllers.

This subpackage implements WNN controllers for drone attitude stabilization,
the first concrete application of the WNN architecture to closed-loop
real-time control (vs the IDS classification tasks elsewhere in the codebase).

Design locked in memory `project_drone_controller_paper1.md` (and the
broader long-horizon vision in `project_future_fanless_drone_vision.md`):

  - Input:    thermometer-encoded IMU + setpoint, windowed over K=3-5
              timesteps, plus recurrent state from previous timestep.
  - Network:  RAMRecurrentNetwork (state layer + output layer), with
              GA-evolved state-layer size N in [50, 300], bits per
              neuron evolved in the same range as IDS, QSR-graded
              state output (2 bits per state neuron).
  - Output:   thermometer-encoded PWM, 256 levels per motor initially
              (upgradeable to 2048 = DShot 11-bit grade), decoded via
              Strategy 5 (QSR-weighted sum per motor).
  - Training: GA + EDRA pipeline reused from IDS. Reward = -attitude_error²
              + smoothness term + λ_mono × thermometer-violations.
  - Sim:      custom Python attitude sim (rigid-body rotational dynamics)
              for paper #1; PX4 SITL deferred to paper #2.

Submodules:
  sim         — custom Python attitude simulator
  controller  — WNN controller wrapper (thermometer I/O over RAMRecurrentNetwork)
  decoders    — QSR-weighted sum (Strategy 5) + Strategy 1 (count) for ablation
  training    — episode runner + reward function + GA driver
  __main__    — CLI entry point for standalone training (no dashboard needed)

This is additive code: nothing in src/wnn/ids/ or src/wnn/ram/ imports
from here. Safe to develop on main without disturbing the IDS cohort.
"""
