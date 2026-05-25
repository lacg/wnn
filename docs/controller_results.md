# WNN Drone-Controller — Results

## C1 vs C2 connectivity-GA signal-check — 25/05/2026

**Setup:** pop=8, gens=4, inner rounds=4, 16 episodes/round, 1500 steps, fixed
15° initial tilt, absolute-PWM, full-state, 4 state neurons × 24b, 16 levels.
GPU-batched scoring + CPU training (workers=1, RAYON=3, alongside the 46M flow).
Driver: `tests/run_controller_ga.py --paradigm both`. Log: `/tmp/c1c2_signalcheck.log`.

| policy | mean_err | stable | reward |
|---|---|---|---|
| PID (teacher) | 2.45° | 100% | −9.36 |
| Untrained | 21.53° | 0% | −262.66 |
| **C1 reward-gated** (re-trained best) | 24.32° | **12%** | −686.70 |
| **C2 reinforce-own** (re-trained best) | 30.93° | 0% | −858.58 |

GA-internal best during search: C1 reward −63.69 (elite Acc up to 31.25%);
C2 reward −858.60.

### Findings
1. **C1 — qualified positive on `stable>0`.** First non-zero closed-loop
   stability on this substrate (12% re-trained, up to 31% in-search). The
   connectivity GA surfaces partially-stabilizing controllers the
   random-connectivity inner loop never found (that was flat 0%). Capability
   exists.
2. **Fitness is noise-dominated.** C1's best genome scored −63.69 during search
   but −686.70 on an independent re-train (~10× swing) — the chaotic inner loop
   yields different controllers per training seed. GA "best" frozen at gen 1, no
   climbing gens 2–4, elite survivals 0/4. A noise-dominated fitness can't be
   climbed reliably. (C2 is reproducibly bad: −858.60 → −858.58, a stable-bad
   attractor.)
3. **C2 — negative as built.** Worse than untrained (31° vs 21°, 0% stable);
   training degrades monotonically (64→75→75→74°) and the improvement-gate
   deadlocks (iter 3: 0 episodes trained). Deterministic cells → no action
   variance to reinforce → amplifies bad behavior. Needs exploration noise, or
   shelve.

### Recommendation (pre-bigger-run)
Do **not** scale the same design — fix the fitness-variance first:
- **Multi-seed genome fitness** (mean over K≈3 inner-train+score) to de-noise.
- **Best-checkpoint selection** (snapshot the controller at its best inner round;
  needs a small Rust cell-export). C1 curve 28→43→**18**→21 shows the best round
  (18°) beats the final (21°), yet we currently re-train to the final state.
- **C2**: add exploration or shelve; start scoping **B (GA-Memory)** which has no
  imitation pathology.
- Bigger run needs `max_train_workers≈12` (once the 46M flow frees) for
  near-linear training speedup.
