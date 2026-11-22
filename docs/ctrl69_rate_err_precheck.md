# CTRL-69 — offline pre-check: does the rate-loop tracking error e carry the residual's information?

Status: **PRE-REGISTERED 22/11/2026 (before any data was produced).** Results and verdict are appended
below the line "RESULTS" afterwards; nothing above that line is edited after data exists.

## Question

N0 (model-free controller = Y+F+L − `--obs-cascade-dhat-residual` − `--yaw-trim-ff`) costs
~+0.28° err / +0.11° steady vs Y+F+L. The dropped input is the per-axis unabsorbed disturbance
`r = (d̂ + c·ki·I)/(c·ki·i_limit) ≡ I − I_ss(d̂)` from MPCOF's model-based observer
(cascade_obs.rs a3). Candidate model-free replacement: the firmware rate loop's own per-axis
tracking error `e = rate_sp − gyro` (computed on board by the rate PID; dI/dt = ki·e).
Does e (in some variant) carry the information r carried, (1) about r itself and (2) about the
teacher's DAgger label / the part of the label the N0 inputs do not already explain?

**This is a NECESSARY-not-sufficient screen.** An offline connectivity probe once failed to predict
flight (rho ≈ 0.3, project_offline_connectivity_probe_closed). PASS here means "worth an ABI + a
flight A/B", never "will fly better". FAIL means "do not spend the ABI".

## Data (VAL only; never the 99990101+ test units)

- Regime: the N0 s1 recipe (`logs/controller/ctrl48/C-S2-d_wposx2_a60_rf_c200_dc_biasin_ms01_P_both_L1_yld_n0_full_s1.out`
  `[argv]`), parsed by phased_ga's own parser; the packed config is `reference_packed_config(ec)` (the
  DAgger training regime the calibration sampler flies: cf21 brushless, L4C, translation, motor lag
  0.2 s, MPCOF teacher, CTBR 100 Hz outer over the 500 Hz firmware rate PID).
- Probe controller = the N0 layout **plus** `--obs-rate-i --obs-cascade-dhat-residual`, so r and I come
  out of the controller's own compute_features (production values, no re-derivation). Teacher-driven
  (`expert_drives=True`, `explore_eps=0`): the trajectory does not depend on the student's cells or
  layout.
- e comes from a scratch diagnostic export (private worktree + private CARGO_TARGET_DIR, temp
  `--target`, never installed into the venv): the flown rate loop's `setpoint − measured` at each
  rate update, recorded every physical step before the student's step (causal: what a feature
  computed at that decision could read).
- Two disjoint episode streams drawn by the DAgger rollout's own episode draw (per-episode airframe
  asymmetry, weather, IC): **FIT** (seed 69000001, 100 episodes) fits every quantizer, the ridge and
  p(y|x); **VAL** (seed 69000002, 100 episodes) is where every number below is measured. Neither seed
  is a train seed (31337001+) or a test unit (99990101+). Records = decision steps (100 Hz).
- Optional secondary: the same on N0-winner-driven rollouts, only if a residual can be produced for
  a layout without it without new physics in Python (otherwise reported as not done).

## Variables (per axis a ∈ roll, pitch, yaw)

x candidates:
- `R`   residual feature (reference — the information N0 lost)
- `I`   rate-I feature (I/i_limit; known-bad reference: the NR arm failed in flight)
- `E0`  raw e at the last rate update before the decision
- `Ewin` mean e over the last action-repeat window (the 5 rate updates of the previous 10 ms)
- `Eτ`  first-order low-pass of e at the rate-loop rate, τ ∈ {20, 50, 100, 200, 500} ms
- secondary: `I×Eτ` joint (both thermometers seen together, 9×9 cells)

y targets:
- (1) `R` itself
- (2a) the DAgger label for that axis (bank p/q/r, the teacher's rate setpoint the student is trained toward)
- (2b) **PRIMARY** the label REMAINDER: label − ridge prediction from the thermometer bits (8 quantile
  cuts per feature, FIT) of the N0 feature vector (the probe's features minus the R and I blocks).
  This is what a new input must add on top of what N0 already sees.

Quantization (what a WNN can see): every x → 8 quantile cuts fitted on FIT (9 levels, a thermometer);
every y → 16 quantile bins fitted on FIT (levels = 16).

## Metrics

- **Held-out information gain** `IG(y;x) = H_VAL(y) − CE_VAL(y|x)` in bits, with p(y) and p(y|x-level)
  estimated on FIT (add-0.5 Dirichlet smoothing) and scored by log-loss on VAL. Held-out, so plug-in
  bias cannot inflate it (it can go negative). This is the primary metric.
- secondary: plug-in MI on VAL with the same bins; Spearman ρ on VAL (raw values).
- **Recovery** `ρ_rec(x) = IG(y;x) / IG(y;R)` on the primary target.

## Pass rule (fixed before data)

PASS iff **one variant** (the same one on all three axes) among {E0, Ewin, E20, E50, E100, E200, E500}
has, on the PRIMARY target (2b), for roll, pitch AND yaw:
1. recovery `IG(2b; E*) / IG(2b; R) ≥ 0.50`, and
2. `IG(2b; E*) > IG(2b; I)` (beats rate-I).

An axis where R itself is uninformative (`IG(2b; R) < 0.005` bits) is reported N/A and excluded
from (1) (it still needs (2)); if all three are N/A the check is VOID (the residual carried nothing
measurable offline, so the N0 cost is not an information loss this screen can see).

Why 50%: the residual is the reference for what N0 lost (+0.28° err). A replacement carrying under
half that information is unlikely to recover a meaningful share of the cost once it pays for its
own address bits (each feature consumes receptive-field slots); at or above half it is worth an ABI
and a flight A/B. 7 variants are tried, so the "best variant" is a small best-of-7 on VAL; the
same-variant-on-all-axes requirement and the full table guard against that. The joint `I×Eτ` and
targets (1)/(2a) are reported but do not enter the rule.

---

## RESULTS
