# Paredes-Vallés, Hagenaars, Dupeyroux, Stroobants, Xu, de Croon 2024 — Fully neuromorphic vision and control for autonomous drone flight

Science Robotics 9(90), eadi0591 (2024), DOI 10.1126/scirobotics.adi0591 (per TU Delft repository search hit).
arXiv:2303.08778. Local PDF: `papers/paredesvalles2024_fully_neuromorphic_flight.pdf`. Read 19/11/2026.

**Relevance.** "Fully neuromorphic" in a top venue still ends at thrust+attitude setpoints handed to PX4.
**Regime:** real drone, DAVIS event camera, Loihi (Kapoho Bay) + UP Squared, PX4 autopilot.

## Verbatim extraction
- Abstract: vision part "consisting of five layers and 28.8k neurons"; Loihi "execution frequency of 200 Hz, spending only
  27 µJ per inference". Per-corner net 506.4k synapses.
- Control part: single linear decoding layer → "desired collective thrust, pitch and roll angles and yaw rate", sent
  "over ROS2 to PX4, and processed by the low-level controllers there"; uses "the attitude estimate coming from PX4".
  Trained by a mutation-only GA, "population size of 100" (in Flightmare, with domain randomization).
- Energy table: Loihi (Nahuku 32) 7.165 / 20.602 / 27.207 µJ per inference (slow/medium/fast); Jetson Nano 85,576.977 µJ
  (5 W mode) / 75,347.505 µJ (10 W mode); "three to four orders of magnitude" better energy.
- Scoping sentence: "Such hardware is currently not available, so we have limited ourselves to the vision-to-control
  pipeline, ending at thrust and attitude commands."

## Absent
- No neuromorphic attitude/rate loop; no in-flight RMS tracking error; full-system power (camera, UP board, PX4) excluded.
- Idle 0.94 W / +7–12 mW figures seen in a search snippet for the journal version — unverified in the arXiv PDF beyond the table.
