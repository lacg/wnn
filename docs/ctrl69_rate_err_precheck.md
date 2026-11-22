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

*(Appended 22/11/2026 after the data. Nothing above this line was edited after data existed:
prereg commit ed6e26b9.)*

### What was run

- Scratch export `trace_dagger_cascade` (patch: `logs/controller/ctrl69/trace_dagger_cascade.patch`,
  private worktree of 269e9e02 = ABI 60 source, private CARGO_TARGET_DIR, temp `--target`; the venv
  wheel was never touched). It runs the production DAgger rollout (`rollout_and_label_inner`) with
  the diagnostic `ImitCapture` and returns per decision the feature vector + label, and per physical
  step the FLOWN rate loop's `err` (new diagnostic field `PidCh.last_err` = sp − measured at the last
  rate update) and I/i_limit. A **shadow residual block** (the production `CascadeObs` with the
  residual on, ticked with the same three inputs `tick_cascade` hands the real block) gives r for
  layouts that do not emit it.
- Self-checks: rate-I feature ≡ recorded flown I (max |diff| 0.0); shadow residual ≡ `cresid`
  feature (max |diff| 0.0, teacher-driven AND student-driven); the teacher-driven analysis re-run on
  the final wheel reproduced the first run byte-for-byte.
- Streams: FIT seed 69000001, VAL seed 69000002, 100 episodes × 2000 steps (2 s, 1 kHz) = 20 000
  decision records each; 0 diverged episodes in every arm. Regime = N0 s1 argv, tilt 5°.
  Scripts + raw outputs: `logs/controller/ctrl69/` (`run.sh` reproduces everything).

### PRE-REGISTERED ARM — teacher-driven (MPCOF flies): **PASS**

Primary target (2b) = label remainder after a ridge on the N0 feature thermometer bits (ridge VAL
R² of the label p/q/r = 0.51 / 0.52 / 0.02). Held-out IG in bits:

| x | roll | pitch | yaw |
|---|---|---|---|
| R (reference) | 0.207 | 0.154 | 0.498 |
| I (rate-I) | 0.026 | 0.017 | 0.346 |
| E0 | 0.174 | 0.143 | 0.478 |
| E20 | 0.174 | 0.151 | 0.499 |
| **E50** | 0.172 | 0.154 | 0.509 |
| E100 | 0.166 | 0.149 | 0.510 |
| E500 | 0.113 | 0.074 | 0.414 |

Recovery IG(x)/IG(R): E0 0.84/0.93/0.96, Ewin 0.85/0.92/0.96, E20 0.84/0.98/1.00,
**E50 0.83/1.00/1.02**, E100 0.80/0.97/1.02, E200 0.72/0.84/0.99, E500 0.55/0.48/0.83 (fails);
rate-I 0.13/0.11/0.69. Every e variant except E500 beats I on every axis.
**Prereg verdict: PASS via E0, Ewin, E20, E50, E100, E200.** (Bootstrap over VAL episodes, E50:
0.84 [0.64, 1.15] / 1.04 [0.72, 1.44] / 1.02 [0.97, 1.10].)

### POST-HOC (not pre-registered) — the teacher-driven PASS is partly a copycat channel

Found while reading the table: Spearman(e, label) ≈ +0.88. Under teacher-driven rollouts the flown
setpoint IS the teacher's previous label, so `e = sp_prev − gyro` hands the student the teacher's
previous action (causal-confusion / copycat; de Haan et al. 2019). In deploy and in student-driven
DAgger, sp_prev is the STUDENT's own command, not the teacher's. Two post-hoc checks:

**(a) Copycat control** — teacher-driven, remainder ridge also given the previous label's bits
(ridge VAL R² 0.72 / 0.71 / 0.85): R keeps 0.182 / 0.140 bits on roll/pitch but only 0.015 on yaw;
e recovery E20 1.02 / 1.08 / −0.38, E50 0.99 / 1.04 / −0.07. → On roll/pitch the information is
real (survives the control); on yaw, the residual's AND e's label information was the slow yaw
setpoint itself (the copycat channel), nothing beyond it.

**(b) N0-winner-driven (the deploy-relevant distribution; N0 s1 winner flies, MPCOF labels, shadow
R)** — ridge VAL R² 0.54 / 0.58 / 0.61. Primary target (2b), held-out IG bits:

| x | roll | pitch | yaw |
|---|---|---|---|
| R (reference) | 0.176 | 0.197 | 0.107 |
| I (rate-I) | −0.001 | −0.003 | 0.064 |
| E0 | 0.121 | 0.168 | −0.026 |
| E20 | 0.128 | 0.163 | −0.033 |
| **E50** | 0.139 | 0.161 | −0.016 |
| E100 | 0.141 | 0.155 | 0.014 |
| E200 | 0.119 | 0.135 | 0.040 |
| E500 | 0.063 | 0.069 | 0.052 |
| I×E20 (joint, secondary) | 0.102 | 0.124 | 0.090 |
| I×E50 (joint, secondary) | 0.104 | 0.122 | 0.070 |

Recovery: E50 **0.79 / 0.82 / −0.15**, E100 0.80 / 0.79 / 0.13, E500 0.36 / 0.35 / 0.49;
rate-I **−0.00 / −0.01 / 0.60**; joint I×E20 0.58 / 0.63 / 0.85 (beats I alone on all three).
Bootstrap E50: 0.80 [0.64, 1.08] / 0.83 [0.67, 1.03] / −0.14 [−0.41, −0.02]; rate-I yaw 0.60
[0.40, 0.88]. Same rule applied to this arm: **FAIL (yaw)**; roll/pitch pass for E0..E200.
About R itself (target 1) e carries 1.1-1.3 bits on roll/pitch (E50-E200) vs I 0.52-0.56; on yaw
I (1.22) beats every e variant except E200/E500 (1.35/1.50).

**(c) Y+F+L-winner-driven** (s1 winner, ABI-59 genome on the ABI-60 wheel, thresholds re-fit as the
run did): roll/pitch E50 0.76 / 0.95, E100 0.77 / 0.97, I 0.12 / 0.23; **yaw is not readable** —
its yaw label is the remainder after `--yaw-trim-ff`, which is built from cresid_z, so R's yaw
information is already removed from the label (IG(R) = −0.01, N/A).

### VERDICT

- **Pre-registered rule: PASS** (teacher-driven; best single variant E50, recovery 0.83 / 1.00 / 1.02,
  beats rate-I on every axis).
- **But the yaw half of that PASS does not survive the post-hoc checks**: it rides on the teacher's
  previous setpoint (copycat), and on the N0-winner-driven distribution e carries ~none of the
  residual's yaw label information (E50 −0.15) while rate-I carries 0.60 of it.
- **Roll/pitch: e is a robust model-free replacement** — recovery 0.79-1.04 for E20-E100 in every
  arm (teacher, copycat-controlled, N0-driven, Y+F+L-driven), where rate-I carries ~0 (consistent
  with the NR arm failing in flight).
- **Best variant: E50** (one-pole low-pass of e at τ = 50 ms; E100 within noise). The raw value (E0)
  and the 10 ms window mean (Ewin) lose ~0.1 of recovery on roll; τ ≥ 200 ms decays.
- **Candidate to fly, if any:** e-LPF τ = 50 ms on roll/pitch, with yaw covered by rate-I (I_z,
  0.60) or the per-axis pair (I, E20/E50) — the joint pair is the only form that cleared ≥ 0.5 on
  all three axes on the N0-driven arm (0.58 / 0.63 / 0.85). These are post-hoc reads: a flight A/B
  would need its own prereg.
- **Necessary-not-sufficient**: an offline information screen once failed to predict flight (rho
  ≈ 0.3). This says e is not information-dead on roll/pitch and is on yaw; it does not say any arm
  will close the +0.28° gap.
