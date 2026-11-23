# CTRL-69 — rate-loop tracking error as the model-free stand-in for the MPCOF residual (arm R) — pre-registration

Status: **PRE-REGISTERED 22/11/2026, before any flight of arm R** (Luiz approved prereg + ABI-62 landing + the R
flight on 22/11 ~21:15 EST). Results are appended below the line "RESULTS" after the data; nothing above that
line is edited once a flight has started.

Evidence that motivated the arm (offline, VAL streams only): `docs/ctrl69_rate_err_precheck.md` — pre-registered
screen PASS; post-hoc, roll/pitch information real on every rollout distribution (E50 recovery 0.79-1.04 of the
residual's label information, rate-I ~0), yaw a copycat channel for e (N0-driven E50 −0.15) where rate-I carries
0.60. That is a necessary-not-sufficient screen; this flight is the test.

## Arms

- **A (control) = N0 on ABI 61** — the already-flying CTRL-67 rows, tags
  `C-S2-d_wposx2_a61_rf_c200_dc_biasin_ms01_P_both_L1_yld_n0_full_sN` (driver `logs/controller/ctrl61/ab_mf61.sh`).
  Not re-flown.
- **R = A + the ABI-62 model-free rate-loop inputs**, tags
  `C-S2-d_wposx2_a62_rf_c200_dc_biasin_ms01_P_both_L1_yld_n0rerr_full_sN`:
  `--obs-rate-err --obs-rate-err-tau 0.05 --obs-rate-err-axes roll,pitch --obs-rate-i --obs-rate-i-axes yaw`
  - roll/pitch: e = rate_sp − gyro of the flown firmware rate PID, / the CTBR bank range, one-pole low-pass
    τ = 50 ms (the precheck's E50);
  - yaw: the firmware rate integrator I_z / i_limit (the only yaw input that survived the N0-driven precheck).
  - No observer, no plant model: everything is read from the on-board rate loop.
  - The argv is A sN's own flown `[argv]` (rebuilt with `logs/controller/ctrl48/rebuild_argv.py fly`) + those 5
    flags; only the save paths (tag) differ. Driver `logs/controller/ctrl62/ab_r62.sh` checks this statically and
    through phased_ga's own resolver (`_resolve_ctbr`) before each flight.
- **B (descriptive only) = N0 + `--delta-max-yaw 0.3`** (CTRL-68, same block as A). B is the yaw-decode lever —
  a different axis of change. R vs B is reported descriptively, never as an adoption test.

## Why R pairs validly with A across the ABI change

ABI 62 (`ctrl62-rate-err` aa2854d0, branched from the ABI-61 source dbd123b0) with every new flag off is
byte-identical to ABI 61: two-build pin ABI 61 vs 62 flags-off identical (base, rate_i, rate_i+dhat, dres;
`tests/controller_ctrl62_abi61_pin.py`), flags-off smoke `logs/controller/ctrl62/smoke_flags_off.out`. The
estimator (f32 MahonyF32), plant, scorer, teacher and test units are the same. R s N and A s N share the base seed
31337000+N, so the pair differs in the 5 flags only.

## Hypothesis

- **H-R:** the rate-loop tracking error (roll/pitch) + yaw rate-I gives the model-free controller back part of
  the disturbance information N0 lost when the MPCOF residual input was dropped (N0 cost ~+0.28° err / +0.11°
  steady vs Y+F+L on ABI 60). **Expected direction: R < A on err AND on steady.** Not expected to change
  stability or position.

## Readout

- TEST only: the HEADLINE block of each run, 35 CF2.1 units 99990101..99990135 × 100 episodes. The whole
  vector: stable % / err ° / steady ° / pos m (3-D, includes altitude) / fails (stable_fail). Gen-line and VAL
  numbers are not results.
- **Headline n = 5 pairs** (R sN vs A sN, s1-s5), per-seed deltas + mean delta.
- Context rows: the MPCOF equal-information row (ruling row B) re-scored on the installed f32 lineage
  (`docs/ctrl61_rival_rows.md`, produced by the same driver after the landing) and PID. Context only, not part
  of the adoption rule.

## Adoption rule (Luiz's rule, fixed here)

**ADOPT R** if it is consistently better than A in the expected direction on **err and steady** — the majority
of the 5 per-seed pairs in that direction AND the mean delta in that direction, on each of the two — **and not
worse elsewhere**: stable, fails and pos not worse (Luiz weighs 0 fails / 100% stable heavily; stable is floored
from fails). **No magnitude bar.** Flat or worse → not adopted; A stays the ABI-61/62 lineage control. Luiz rules
on the readout.

## What this does NOT test

- Whether e works on yaw (the precheck says it does not on the deploy distribution; yaw uses rate-I in R).
- Any tau other than 50 ms, or the joint (I, e) pair on roll/pitch.
- Anything about MPCOF on board: R has no MPCOF code on board (the DAgger teacher stays MPCOF, offline only).

## Failure handling

- A run that fails (rc ≠ 0) is retried by hand only if the failure is new and fixable; the same failure three
  times → that seed is deferred and the readout is n < 5, said so.
- If an A seed has no finished run, R for that seed is not flown (no pair).

---

## RESULTS

*(empty until the flights land)*
