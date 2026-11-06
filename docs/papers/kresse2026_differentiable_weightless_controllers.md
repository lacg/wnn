# Kresse & Lampert 2026 — Differentiable Weightless Controllers: Learning Logic Circuits for Continuous Control

Kresse, F., Lampert, C. H. arXiv:2512.01467v2 (8 Jun 2026); PDF footer: "Proceedings of the 43rd
International Conference on Machine Learning" (ICML 2026). Local PDF:
`papers/kresse2026_differentiable_weightless_controllers.pdf`. Read 06/11/2026.

**Relevance.** The closest prior art to our WNN controller: LUT-based (DWN-derived) policies with
thermometer-encoded inputs, trained by gradient surrogates (SAC/PPO), synthesized to FPGA.
**Regime:** MuJoCo SIMULATION ONLY (Ant, HalfCheetah, Hopper, Humanoid, Walker2d); no robot hardware.

## Verbatim extraction
- "compile directly into FPGA-compatible circuits with few- or even single-clock-cycle latency and
  nanojoule-level energy cost per action. Across five MuJoCo benchmarks … DWCs achieve returns competitive
  with standard deep policies".
- Encoding: "quantizing them using quantile binning and thermometer encoding"; "two layers of k=6 input
  LUTs"; default 1024 LUTs per layer.
- Table 2 (Artix-7 XC7A15T−1 @ 100 MHz, one model, median reward of 10): Dℓ=1024 e.g. Hopper 3.1k reward,
  3.2k LUTs, 2.0k FFs, 0 BRAM, 0 DSP, Lat 0.03 µs, 0.228 W, 1.0×10^8 actions/s, 2.3×10^−9 J/action;
  Dℓ=256 Hopper 3.3k, 0.9k LUTs, Lat 0.01 µs, 1.2×10^−9 J. Quantized-NN baseline (Kresse & Lampert 2026)
  Hopper 1.9k reward, 45 DSP, 0.21 µs, 6.5×10^−8 J.
- "For our standard Dℓ = 1024 setup, latency is only 2 or 3 clock cycles … energy usage is in the range of
  2 nanojoule per action." HalfCheetah noted as an exception in training.

## Absent
- No physical robot, no drone, no sim-to-real, no actuator lag, no cascade/inner-loop discussion.
- Training is RL with surrogate gradients — not RAM-write/one-shot, not imitation (DAgger).
