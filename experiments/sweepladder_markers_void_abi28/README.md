# VOID — arm B first flight on the obs_pwm-broken wheel (ABI 28)

`SL_C_b24n256_cf21_brushless_L4C_g10_s31337002_bd` flew 13/09/2026 00:40-03:01 EDT and
banked 0.0% / 57.15° / 57.15° / 0.238 m at every stage: the student output near-constant
pwm (effort 0.92, mono_viol 10). Cause, proven by a 4-arm smoke (logs/controller/armb_smoke):
`--obs-pwm` alone kills (0.0%/70°), `--dagger-label-delta` alone flies. Two defects on the
pwm feature — a degenerate thermometer ladder (the fitter's untrained feature controller
never left the anchor ⇒ all 8 thresholds = 0.5) and a replay-frozen accumulator (the
documented `OBS_PWM_FIXED=false` gap). Fixed in e554661b, controller ABI 29. This row is
not a result and is never paired; the re-fly under the same tag supersedes it.
