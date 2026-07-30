# Motor-fault experiment — the no-teacher test (designed 30/07/2026, HELD)

> **QUAD-dfa corner retired 30/07/2026.** Both dfa x QUAD cells are sentinelled for
> seeds 31337003-006 and stay at n=1 (seed 31337002 only): dfa_9feat_QUAD 42.0% in
> 30.23h, dfa_10feat_QUAD 2.0% in 90.95h. The 003 run of dfa_9feat_QUAD was
> SIGKILLed by the memory watchdog at 14h40 (avail hit the 6GB hard floor during an
> IDS flow handover) with its best pinned since gen 01 — 21.0%/9.08 deg flat for four
> generations and cells MAX at 1.49x the 180k cap. ~121h of campaign time bought the
> study's two WORST held-out results, so the corner is reported descriptively and the
> compute goes to the bits-axis / ceiling work instead.

**Status: designed and baselined, NOT launched — the one-controller-at-a-time rule
holds while the dfa1l sweep owns the controller slot.**

## Why this experiment exists

The 30/07 diagnostics closed two of the three axes the WNN could win on:

1. **Nominal performance** — closed by construction: DAgger imitates an LQR teacher,
   and imitation is bounded above by its expert (best cell 77.0±15.0% vs teacher 100%).
2. **Noise robustness (L3D)** — closed by measurement: every saved winner collapses to
   ~0% stable under L3D while LQR holds 60.8±18.2 and LQI 70.6±15.0
   (`experiments/dfa1l_markers/rescore_L3D_*.json`, `baselines_L3D.json`).

The gap decomposition (`gap_L2D.json` / `gap_L3D.json`, script
`scripts/quantized_teacher_gap.py`) says the failure is the LEARNER, not the substrate:
LQR forced through the WNN's own I/O resolution (8 quantile thresholds per input,
16-level motor decode) still scores **96% under L2D and 43% under L3D**. Output
quantization costs ~0pp everywhere; input quantization costs ~4pp nominal / ~21pp
under stress. So the representational ceiling sits far above what the learner reaches
— which cuts both ways: more search under the current training scheme chases a gap
that imitation cannot close, but the substrate itself is not the excuse.

The remaining axis is **model mismatch**: classical K is computed from the nominal
plant and is exactly wrong when the plant changes. A learner trained ON the changed
plant carries no such error — but only if it is NOT imitating a nominal-model teacher.
The machinery for teacher-free training already exists: the MEMORY-stage GA optimizes
closed-loop fitness directly (no DAgger).

## The measured opening (motor-2 effectiveness sweep, 30/07)

Nominal-K controllers, L2D base conditions, fold-0 pool, 100 ep × 2000 steps:

```
 fault |    PID      |    LQR      |    MPC      |    LQI      |   MPCOF
  1.0  | 100.0%/3.4° | 100.0%/1.4° | 100.0%/1.5° | 100.0%/1.2° | 100.0%/0.8°
  0.8  |  69.0%/4.6° | 100.0%/1.8° | 100.0%/1.9° | 100.0%/1.5° | 100.0%/0.8°
  0.7  |  22.0%/5.6° | 100.0%/2.1° | 100.0%/2.2° | 100.0%/1.7° | 100.0%/0.8°
  0.5  |   0.0%/8.5° | 100.0%/3.0° | 100.0%/3.2° | 100.0%/2.4° | 100.0%/1.1°
  0.3  |   0.0%/16.0°|   3.0%/10.8°|   1.0%/11.0°|  14.0%/9.2° |  20.0%/11.5°
```

Feedback margin absorbs a 50% single-motor loss outright; the cliff is between 0.5
and 0.3. **fault=0.3 is the arena**: best classical (nominal-model) result is MPCOF
at 20.0%. Anything a fault-adapted WNN scores above 20% is a win against every
nominal-model classical controller on the canonical fault-tolerant-control scenario
("one rotor at 30% effectiveness"), stated in one sentence.

## Design

- **Plant fault:** motor 2 at 0.3 effectiveness, injected as a fixed multiplier on the
  resolved motor-asym (composes with L2D's per-airframe draw, exactly as the sweep
  above measured). Sweep 0.3 / 0.4 / 0.5 if budget allows — 0.5 doubles as a sanity
  cell where classical still wins.
- **WNN arm:** seed from the best saved winner (`dfa_9feat_BINARY` — carries trained
  cells) via `--seed-winner ... --seed-winner-stage memory` = MEMORY-only value-GA,
  **NO DAGGER, no teacher** — closed-loop fitness on the faulted plant. This is raw
  transfer + adaptation, the only training mode in the toolkit whose ceiling is not an
  expert.
- **Baselines on the same faulted plant, same fold-0 pool:** nominal-K LQR/LQI/MPCOF
  (already measured above) and PID. An **oracle-K** upper bound (K or mixer recomputed
  knowing the fault) is the right "adaptation prize" reference — needs flight-dynamics
  care (at 0.3 the faulted motor cannot even hold its hover share, so naive mixer
  compensation saturates); design it with the flight-dynamics agent before trusting it.
- **Report:** held-out triple on FRESH report seeds, the same fold-0/conditions
  machinery as everything since 29/07 (`fold_pool_seed` / `disturbance_stream` —
  the conditions-parity test guards it).

## What would need building (small)

1. A `--motor-fault "idx:factor"` flag on `phased_ga` (post-edits the resolved asym
   path) — and the SAME injection in `compute_baselines.py`, or the comparison is
   unmatched (see the three 29/07 condition bugs; add a parity assertion).
2. A recipe script wrapping the seed-winner memory-GA run + baselines + report.

## Success / failure reading

- WNN > 20% stable at fault=0.3 → first genuine WNN-beats-classical result, on a real
  scenario, with an honest no-teacher story. Paper-worthy regardless of margin.
- WNN ≤ 20% → the model-mismatch axis closes too, and with all three axes measured
  shut, the honest conclusion is the multiplier-free-hardware framing or a redesign of
  the training stack (the gap decomposition says the learner, not the substrate, is
  what fails).
