# Wu, Cheng, Zhao, ... Hovakimyan et al. 2023 — L1Quad: L1 Adaptive Augmentation of Geometric Control for Agile Quadrotors with Performance Guarantees

arXiv:2302.07208 (revised Dec 2024; text states accepted to IEEE Trans. Control Systems Technology).
https://arxiv.org/abs/2302.07208 — Local PDF: `papers/wu2023_l1quad.pdf`. Read 28/11/2026. (Author list beyond
"Wu et al." not transcribed — check PDF page 1 before citing.)

**Relevance.** Representative MODEL-BASED adaptive-control robustness protocol. **Regime:** real custom quad,
m = 0.62 kg, J = 1e-3 diag(3.0, 1.8, 3.2) kg m² (Table I), mocap; baselines L1 off (geometric) and PID.

## Verbatim extraction
- "extensive experiments for eleven types of uncertainties" — each scenario tested SEPARATELY: injected uncertainty,
  slosh payload (0.317 kg, "approximately 50%" of vehicle), chipped propeller, mixed propellers, ground effect,
  downwash, hanging off-center weight, tunnel, battery/0.5 kg payload, etc.
- Sec. J benchmark (one factor at a time): "weights range from 100 to 500 g with 100 g increments" (added and slung,
  0.4 m cord); circular trajectory speed "0 (hover) to 2.5 m/s with a 0.5 m/s increment"; figure-8 "0 (hover) to 3 m/s
  with a 1 m/s increment"; "data within 5 s (4000 data points in total)" per box plot.
- Appendix Tables III-VI: RMSE [m] grid weight × speed. E.g. Table III (added, circle) 500 g @ 2.5 m/s: L1 off 0.43,
  L1 on 0.17, PID 0.16; 0 g @ 2.5 m/s: 0.24 / 0.15 / 0.09.
- Derivation (ours): 100–500 g on 0.62 kg = +16% to +81% mass.

## Absent
- No joint/combined perturbation draw; no randomized sim test distribution; no inertia, motor time-constant, latency
  or sensor-noise sweep; single runs per cell (no seeds/episodes beyond the 5 s window).
