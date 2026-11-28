# Eschmann, Albani, Loianno 2025 — RAPTOR: A Foundation Policy for Quadrotor Control

arXiv:2509.11481 (v1 15 Sep 2025; v2 6 Apr 2026). https://arxiv.org/abs/2509.11481 —
Local PDF: `papers/eschmann2025_raptor_foundation_policy.pdf` (v2). Read via arxiv.org/html/2509.11481v2, 28/11/2026.

**Relevance.** State of the art for "one policy, many airframes": recurrent student distilled from 1000 SAC teachers,
each on a sampled quadrotor. **Regime:** sim training (ground-truth state), 10 real quads 31.9 g – 2.4 kg.

## Verbatim extraction (Supplementary, Materials and Methods)
- r_t2w ~ U(1.5, 5) (S5); mass m = s³, s ~ U(∛0.02, ∛5) → m ∈ [0.02, 5] (S6–S7).
- Torque-to-inertia r_t2i ~ U(40, 1200) (S19); Jzz = (Jxx+Jyy)/2 · 1.832 (S22).
- c_m ~ U(0.005, 0.05) (S24). Motor delays: T_m↑ ~ U(0.03, 0.1) (S25), T_m↓ ~ U(0.03, 0.3) (S26).
- Arm-length ratio: authors note u ~ N(−0.1, 0.1) used by mistake instead of a uniform ±10%.
- OOD tests stated: Flightmare quad with thrust-to-weight ≈ 12 vs ≤ 5 in training; flexible frame; mixed 2-/3-blade
  propellers; Mahony/Madgwick estimators vs ground truth in training; 500-step training sequences tested up to 50 s.
- Latency: a linear-velocity delay "10−30 ms" used to reproduce z-oscillations in sim (Discussion).
- Held-out sim test set: 7 quadrotors; sensitivity in main text = number of teachers (Fig. 2C), hidden dim (Fig. 2D).

## Absent
- No observation-noise spec; no action-latency DR; no episode/seed counts for the sim test; no one-parameter-at-a-time
  dynamics sweep; OOD is per-vehicle/per-condition, not a joint widened-range draw.
- Summarizer read only (PDF stored, tables not re-verified against page images) — re-check S-equations before quoting.
