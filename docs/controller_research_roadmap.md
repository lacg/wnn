# Controller research roadmap — pre-paper workstreams (02/07/2026)

_Luiz's call: the paper is in its infancy. The horizon-drift + committee findings
(docs/controller_horizon_findings.md) are real but raise more questions than a paper
can leave open. Paper gate: W1-W3 minimum; W4/W5 may define the paper#1/paper#2 split._

## W0 — in flight (the running pipeline)
E2 seed10s → low-edge seed10 rescue → **C2K**: 8-cell pool of @2000-trained members
(PWM2K/TILT2K/LEAN2K/ANCH2K × seeds 09/10, LONG recipe) + free LONG_s09/s10 →
assembly: fresh-seed truth serum per member → mean-PWM committees at the horizon
quadruplet (500/2000/5000/10000). Hypothesis: committee of non-drifters ≥96%,
horizon-free. Markers: /tmp/wnn_e2_done.json → /tmp/wnn_rescue_done.json →
/tmp/wnn_c2k_done.json.

## W1 — understand the horizon gap ("good should be good at any step count")
1. **Decay-law surface**: train at H ∈ {500, 1000, 2000, 4000} (one recipe, 2 seeds),
   eval each at {0.5×, 1×, 2.5×, 5×, 10×, 20×} H. Extract: is drift immunity always
   ~2.5× the trained horizon? Does the decay slope depend on H? (Committee too.)
2. **Drift-mode analysis**: record drifting trajectories from a @500 winner past its
   horizon; characterize the wander (per-axis bias? oscillatory? locked to state-
   address cycles or thermometer-threshold boundaries?). If drift has an identifiable
   mechanism, singles might be made horizon-free without committees.
3. **Correlated-drift probe** (threat): committees with HETEROGENEOUS encodings
   (per-member bits-per-feature / threshold-gamma) vs the shared-encoding committee,
   at horizons ≫10⁴ — does shared encoding eventually correlate member drift?

## W2 — disturbances (the "real weather" chapter) — THE priority after W0
Add to the Rust sim + Metal rollout mirrors (full parity discipline): constant wind
torque, gust process (e.g. Ornstein-Uhlenbeck), motor-efficiency asymmetry, sensor
noise (gyro bias + accel noise), CG offset. Then re-run the whole measured ladder
under each: PID vs PD-only (integral action finally has WORK — first separation
expected), singles, committee, horizon sweep. This is where a learned controller can
legitimately beat a fixed-gain PID — and where the paper's comparisons become honest.

## W3 — hybrid PID + WNN residual (E5), under W2 at long horizons
PID stays in the actuation path; WNN learns clamped Δu (±10% PWM) via the closed-loop
MEMORY GA (DAGGER can't train a residual — the teacher IS the PID, target ≡ 0).
Floor = PID; measure the lift under wind/asymmetry where fixed gains are suboptimal.
Compare: PID | PD | best single | committee | hybrid | hybrid-committee.

## W4 — task validity (is attitude-hold the right problem?)
- Attitude stabilization IS the standard low-level inner-loop task (defensible), but
  the RL-comparison lineage (incl. Sajus) demos goal-directed tasks.
- Our sim is ROTATION-ONLY — position hold is not currently expressible. Add
  translational dynamics (thrust vector integration, drag) → tasks: position hold,
  then waypoint acquisition (Sajus-comparable score: targets reached per time).
- Literature scan (short): what do recent learned-flight-controller papers use as
  the primary task + metrics; pick our second task to match the modal choice.

## W5 — realism ladder
- **W5a (cheap, pull into paper #1)**: real airframe parameters — Crazyflie-class
  mass/inertia/arm/k_thrust — re-run headline tables. One param swap.
- **W5b (paper #2 bridge)**: PX4 SITL / jMAVSim; then FPGA-in-the-loop per the
  original design (project_drone_controller_paper1).
- **W5c (question, not commitment)**: physical drone before paper #1? Original
  design says sim-only paper #1, hardware paper #2 — revisit after W2 results.

## Standing methodology (locked this investigation)
Fresh-seed truth serum mandatory for any winner; report the horizon quadruplet
(500/2000/5000/10000) + steady° everywhere; --steps 2000 default TRAINING length;
one controller at a time; rust-first including harnesses; source+wheel land
atomically at driver-idle.
