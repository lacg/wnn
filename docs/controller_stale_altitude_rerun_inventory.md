# Stale-altitude-features bug — rerun inventory (measured 12/09/2026)

Every banked run whose winner checkpoint has any of obs_collective_cmd / obs_alt_err /
obs_vz true trained its vertical bits at a STALE address (spec §0b). These are the runs
to requeue on the fixed wheel (ABI 28). Measured from the checkpoints, not from memory.

TOTAL: 148 runs, 594 h of box time.

| cohort | runs | hours | span | what it decided (and therefore what is re-opened) |
|---|---|---|---|---|
| altweight | 9 | 25 | 2026-08-18..2026-08-19 | altitude weight (alt=0 chosen) |
| fitnessab | 10 | 34 | 2026-08-20..2026-08-21 | zscore vs harmonic combine |
| gatedwsweep | 29 | 98 | 2026-08-21..2026-08-25 | S16noJM weights = the ladder's weights |
| leakrevisit | 2 | 7 | 2026-09-06..2026-09-06 | leak 0.80/0.90 revisit |
| specialist1 | 5 | 20 | 2026-08-16..2026-08-16 | specialist weights |
| stage1lambda | 6 | 20 | 2026-08-14..2026-08-15 | stage-1 altitude lambda |
| sweepladder | 82 | 361 | 2026-08-26..2026-09-12 | bits/neurons/gamma ladder, anchors, arm A, leak, mut, window-k, D0 A/B |
| translationab | 5 | 29 | 2026-09-04..2026-09-06 | translation ON arms (b32 n256 replication) |

## Suggested order (active programme first, then the decisions it rests on)

1. sweepladder anchors + D0 A/B + arm A + leak/mut/window (the multi-axis anchor set) — 82 runs
2. translationab ON + leakrevisit — 7 runs
3. gatedwsweep (re-decides the weight vector) — 29 runs
4. fitnessab, altweight, specialist1, stage1lambda — 30 runs

Each cohort's chain is marker-gated and idempotent: move the old markers aside (git keeps them),
relaunch the chain on the fixed wheel, and the provenance field in every new marker carries the
ABI-28 wheel hash so old and new can never be confused again.

## Runs

### altweight
- 2026-08-18  AW_C10_alt000_b18n32_s31337002  (2.9 h)
- 2026-08-19  AW_C10_alt010_b18n32_s31337002  (2.7 h)
- 2026-08-19  AW_S16_alt010_b18n32_s31337002  (2.8 h)
- 2026-08-19  AW_S16_alt020_b18n32_s31337002  (2.8 h)
- 2026-08-19  AW_REF_lam16_b18n32_s31337002  (2.8 h)
- 2026-08-19  AW_C10_alt020_b18n32_s31337002  (2.9 h)
- 2026-08-19  AW_S16_alt000_b18n32_s31337002  (2.9 h)
- 2026-08-19  AW_C10_alt035_b18n32_s31337002  (2.9 h)
- 2026-08-19  AW_S16_alt035_b18n32_s31337002  (2.9 h)

### fitnessab
- 2026-08-20  FAB_harmonic_c10_cf21_brushless_L4C_s31337002  (3.2 h)
- 2026-08-20  FAB_zscore_c10_cf21_brushless_L4C_s31337004  (3.2 h)
- 2026-08-20  FAB_zscore_c10_cf21_brushless_L4C_s31337003  (3.4 h)
- 2026-08-20  FAB_zscore_c10_cf21_brushless_L4C_s31337002  (3.4 h)
- 2026-08-20  FAB_harmonic_c10_cf21_brushless_L4C_s31337003  (3.4 h)
- 2026-08-20  FAB_harmonic_c10_cf21_brushless_L4C_s31337004  (3.7 h)
- 2026-08-21  FAB_harmonic_c10_cf21_brushless_L4C_s31337005  (3.2 h)
- 2026-08-21  FAB_zscore_c10_cf21_brushless_L4C_s31337006  (3.4 h)
- 2026-08-21  FAB_zscore_c10_cf21_brushless_L4C_s31337005  (3.5 h)
- 2026-08-21  FAB_harmonic_c10_cf21_brushless_L4C_s31337006  (3.8 h)

### gatedwsweep
- 2026-08-21  GWS_C10_cf21_brushless_L4C_s31337002  (3.2 h)
- 2026-08-22  GWS_S16_cf21_brushless_L4C_s31337003  (3.1 h)
- 2026-08-22  GWS_C10_cf21_brushless_L4C_s31337003  (3.3 h)
- 2026-08-22  GWS_E50S50_cf21_brushless_L4C_s31337002  (3.4 h)
- 2026-08-22  GWS_C10noJM_cf21_brushless_L4C_s31337002  (3.4 h)
- 2026-08-22  GWS_STEADY40_cf21_brushless_L4C_s31337002  (3.5 h)
- 2026-08-22  GWS_S16noJM_cf21_brushless_L4C_s31337002  (3.5 h)
- 2026-08-22  GWS_S16_cf21_brushless_L4C_s31337002  (3.7 h)
- 2026-08-23  GWS_S16noJM_cf21_brushless_L4C_s31337004  (3.0 h)
- 2026-08-23  GWS_S16noJM_cf21_brushless_L4C_s31337003  (3.0 h)
- 2026-08-23  GWS_C10noJM_cf21_brushless_L4C_s31337004  (3.1 h)
- 2026-08-23  GWS_E50S50_cf21_brushless_L4C_s31337003  (3.1 h)
- 2026-08-23  GWS_S16_cf21_brushless_L4C_s31337004  (3.2 h)
- 2026-08-23  GWS_C10_cf21_brushless_L4C_s31337004  (3.3 h)
- 2026-08-23  GWS_C10noJM_cf21_brushless_L4C_s31337003  (3.3 h)
- 2026-08-23  GWS_STEADY40_cf21_brushless_L4C_s31337003  (3.5 h)
- 2026-08-24  GWS_STEADY40_cf21_brushless_L4C_s31337004  (3.2 h)
- 2026-08-24  GWS_E50S50_cf21_brushless_L4C_s31337004  (3.3 h)
- 2026-08-24  GWS_S16noJM_cf21_brushless_L4C_s31337005  (3.5 h)
- 2026-08-24  GWS_S16_cf21_brushless_L4C_s31337005  (3.5 h)
- 2026-08-24  GWS_C10noJM_cf21_brushless_L4C_s31337005  (3.6 h)
- 2026-08-24  GWS_E50S50_cf21_brushless_L4C_s31337005  (3.7 h)
- 2026-08-24  GWS_C10_cf21_brushless_L4C_s31337005  (3.8 h)
- 2026-08-25  GWS_STEADY40_cf21_brushless_L4C_s31337006  (3.3 h)
- 2026-08-25  GWS_S16noJM_cf21_brushless_L4C_s31337006  (3.4 h)
- 2026-08-25  GWS_S16_cf21_brushless_L4C_s31337006  (3.4 h)
- 2026-08-25  GWS_STEADY40_cf21_brushless_L4C_s31337005  (3.5 h)
- 2026-08-25  GWS_C10_cf21_brushless_L4C_s31337006  (3.5 h)
- 2026-08-25  GWS_C10noJM_cf21_brushless_L4C_s31337006  (3.5 h)

### leakrevisit
- 2026-09-06  LKR_l080_b32n64_cf21_brushless_L4C_g10_s31337002  (3.3 h)
- 2026-09-06  LKR_l090_b32n64_cf21_brushless_L4C_g10_s31337002  (3.4 h)

### specialist1
- 2026-08-16  SP1_C3_b15min3_cf21_brushless_L4C_s31337002  (3.4 h)
- 2026-08-16  SP1_D_b30fullwin_cf21_brushless_L4C_s31337002  (3.6 h)
- 2026-08-16  SP1_C2_b15min2_cf21_brushless_L4C_s31337002  (4.4 h)
- 2026-08-16  SP1_B_b15spread_cf21_brushless_L4C_s31337002  (4.4 h)
- 2026-08-16  SP1_D40_b30fullwin_stride10_cf21_brushless_L4C_s31337002  (4.6 h)

### stage1lambda
- 2026-08-14  S1L_lam4_mpcof_cf21_brushless_L4C_s31337002  (3.1 h)
- 2026-08-14  S1L_lam16_mpcof_cf21_brushless_L4C_s31337002  (3.3 h)
- 2026-08-14  S1L_lam0_mpcof_cf21_brushless_L4C_s31337002  (3.4 h)
- 2026-08-14  S1L_lam1_mpcof_cf21_brushless_L4C_s31337002  (3.5 h)
- 2026-08-14  S1L_lam64_mpcof_cf21_brushless_L4C_s31337002  (3.5 h)
- 2026-08-15  S1L_lam1_mpcof_cf21_brushless_L4C_s31337003  (3.1 h)

### sweepladder
- 2026-08-26  SL_A_b10n32_cf21_brushless_L4C_s31337002  (4.1 h)
- 2026-08-26  SL_A_b12n32_cf21_brushless_L4C_desir_s31337002  (4.1 h)
- 2026-08-26  SL_A_b14n32_cf21_brushless_L4C_s31337002  (4.2 h)
- 2026-08-26  SL_A_b16n32_cf21_brushless_L4C_s31337002  (4.3 h)
- 2026-08-26  SL_A_b12n32_cf21_brushless_L4C_s31337002  (4.3 h)
- 2026-08-26  SL_A_b18n32_cf21_brushless_L4C_s31337002  (4.3 h)
- 2026-08-27  SL_A_b20n32_cf21_brushless_L4C_desir_s31337002  (4.3 h)
- 2026-08-27  SL_A_b20n32_cf21_brushless_L4C_s31337002  (4.3 h)
- 2026-08-27  SL_A_b18n32_cf21_brushless_L4C_desir_s31337002  (4.4 h)
- 2026-08-27  SL_A_b16n32_cf21_brushless_L4C_desir_s31337002  (4.6 h)
- 2026-08-27  SL_A_b14n32_cf21_brushless_L4C_desir_s31337002  (4.6 h)
- 2026-08-28  SL_A_b26n32_cf21_brushless_L4C_s31337002  (3.8 h)
- 2026-08-28  SL_A_b26n32_cf21_brushless_L4C_desir_s31337002  (4.0 h)
- 2026-08-28  SL_A_b22n32_cf21_brushless_L4C_desir_s31337002  (4.1 h)
- 2026-08-28  SL_A_b22n32_cf21_brushless_L4C_s31337002  (4.1 h)
- 2026-08-28  SL_A_b24n32_cf21_brushless_L4C_desir_s31337002  (4.2 h)
- 2026-08-28  SL_A_b24n32_cf21_brushless_L4C_s31337002  (4.2 h)
- 2026-08-29  SL_A_b32n32_cf21_brushless_L4C_desir_s31337002  (3.7 h)
- 2026-08-29  SL_A_b30n32_cf21_brushless_L4C_s31337002  (3.8 h)
- 2026-08-29  SL_A_b32n32_cf21_brushless_L4C_s31337002  (3.8 h)
- 2026-08-29  SL_A_b30n32_cf21_brushless_L4C_desir_s31337002  (3.9 h)
- 2026-08-29  SL_A_b28n32_cf21_brushless_L4C_s31337002  (4.0 h)
- 2026-08-29  SL_A_b28n32_cf21_brushless_L4C_desir_s31337002  (4.0 h)
- 2026-08-30  SL_A_b36n32_cf21_brushless_L4C_s31337002  (3.7 h)
- 2026-08-30  SL_A_b36n32_cf21_brushless_L4C_desir_s31337002  (3.7 h)
- 2026-08-30  SL_A_b34n32_cf21_brushless_L4C_desir_s31337002  (3.8 h)
- 2026-08-30  SL_A_b34n32_cf21_brushless_L4C_s31337002  (3.9 h)
- 2026-08-30  SL_A_b40n32_cf21_brushless_L4C_desir_s31337002  (4.1 h)
- 2026-08-30  SL_A_b40n32_cf21_brushless_L4C_s31337002  (5.2 h)
- 2026-08-31  SL_A_b48n32_cf21_brushless_L4C_s31337002  (4.0 h)
- 2026-08-31  SL_A_b48n32_cf21_brushless_L4C_desir_s31337002  (4.2 h)
- 2026-08-31  SL_A_b64n32_cf21_brushless_L4C_s31337002  (5.3 h)
- 2026-08-31  SL_A_b64n32_cf21_brushless_L4C_desir_s31337002  (5.9 h)
- 2026-09-01  SL_C_b32n32_cf21_brushless_L4C_g20_s31337002  (3.5 h)
- 2026-09-01  SL_C_b32n64_cf21_brushless_L4C_g10_s31337002  (3.6 h)
- 2026-09-01  SL_C_b36n64_cf21_brushless_L4C_g10_s31337002  (3.8 h)
- 2026-09-01  SL_C_b36n32_cf21_brushless_L4C_g20_s31337002  (3.8 h)
- 2026-09-01  SL_C_b36n96_cf21_brushless_L4C_g10_s31337002  (3.8 h)
- 2026-09-01  SL_A_b48n32_cf21_brushless_L4C_g20_s31337002  (9.3 h)
- 2026-09-02  SL_C_b32n64_cf21_brushless_L4C_g20_s31337002  (3.4 h)
- 2026-09-02  SL_C_b32n96_cf21_brushless_L4C_g10_s31337002  (3.7 h)
- 2026-09-02  SL_C_b36n64_cf21_brushless_L4C_g20_s31337002  (3.8 h)
- 2026-09-02  SL_C_b32n256_cf21_brushless_L4C_g10_s31337002  (5.7 h)
- 2026-09-02  SL_C_b36n256_cf21_brushless_L4C_g10_s31337002  (6.9 h)
- 2026-09-03  SL_C_b32n96_cf21_brushless_L4C_g20_s31337002  (3.7 h)
- 2026-09-03  SL_C_b24n256_cf21_brushless_L4C_g10_s31337002  (4.0 h)
- 2026-09-03  SL_C_b36n96_cf21_brushless_L4C_g20_s31337002  (4.1 h)
- 2026-09-03  SL_C_b28n256_cf21_brushless_L4C_g10_s31337002  (4.3 h)
- 2026-09-03  SL_C_b40n256_cf21_brushless_L4C_g10_s31337002  (6.8 h)
- 2026-09-04  SL_C_b32n256_cf21_brushless_L4C_g10_s31337003  (4.6 h)
- 2026-09-04  SL_C_b24n256_cf21_brushless_L4C_g10_s31337003  (4.6 h)
- 2026-09-04  SL_C_b28n256_cf21_brushless_L4C_g10_s31337003  (5.1 h)
- 2026-09-06  SL_C_b32n64_cf21_brushless_L4C_g10_s31337002_crn  (3.4 h)
- 2026-09-06  SL_C_b24n256_cf21_brushless_L4C_g10_s31337002_crn  (4.7 h)
- 2026-09-06  SL_C_b28n256_cf21_brushless_L4C_g10_s31337002_crn  (4.9 h)
- 2026-09-07  SL_C_b24n256_cf21_brushless_L4C_g10_s31337005  (4.4 h)
- 2026-09-07  SL_C_b24n256_cf21_brushless_L4C_g10_s31337003_mut1tap  (4.5 h)
- 2026-09-07  SL_C_b24n256_cf21_brushless_L4C_g10_s31337002_mut1tap  (4.6 h)
- 2026-09-07  SL_C_b24n256_cf21_brushless_L4C_g10_s31337004  (4.6 h)
- 2026-09-08  SL_C_b24n256_cf21_brushless_L4C_g10_s31337005_mut1tap  (4.4 h)
- 2026-09-08  SL_C_b24n256_cf21_brushless_L4C_g10_s31337003_leak090  (4.5 h)
- 2026-09-08  SL_C_b24n256_cf21_brushless_L4C_g10_s31337002_leak090  (4.5 h)
- 2026-09-08  SL_C_b24n256_cf21_brushless_L4C_g10_s31337004_leak090  (4.5 h)
- 2026-09-08  SL_C_b24n256_cf21_brushless_L4C_g10_s31337004_mut1tap  (4.6 h)
- 2026-09-09  SL_C_b24n256_cf21_brushless_L4C_g10_s31337005_leak090  (4.4 h)
- 2026-09-09  SL_C_b24n256_cf21_brushless_L4C_g10_s31337002_ls2  (4.7 h)
- 2026-09-09  SL_C_b24n256_cf21_brushless_L4C_g10_s31337002_ls4  (5.0 h)
- 2026-09-09  SL_C_b24n256_cf21_brushless_L4C_g10_s31337002_ls8  (5.1 h)
- 2026-09-09  SL_C_b24n256_cf21_brushless_L4C_g10_s31337002_win2  (5.5 h)
- 2026-09-10  SL_C_b24n256_cf21_brushless_L4C_g10_s31337003_ls2  (4.7 h)
- 2026-09-10  SL_C_b24n256_cf21_brushless_L4C_g10_s31337004_ls2  (4.8 h)
- 2026-09-10  SL_C_b24n256_cf21_brushless_L4C_g10_s31337003_ls4  (4.9 h)
- 2026-09-10  SL_C_b24n256_cf21_brushless_L4C_g10_s31337003_ls8  (5.1 h)
- 2026-09-10  SL_C_b24n256_cf21_brushless_L4C_g10_s31337004_ls4  (5.2 h)
- 2026-09-11  SL_C_b24n256_cf21_brushless_L4C_g10_s31337002_hd  (2.8 h)
- 2026-09-11  SL_C_b24n256_cf21_brushless_L4C_g10_s31337005_ls2  (4.5 h)
- 2026-09-11  SL_C_b24n256_cf21_brushless_L4C_g10_s31337005_ls4  (4.6 h)
- 2026-09-11  SL_C_b24n256_cf21_brushless_L4C_g10_s31337005_ls8  (4.8 h)
- 2026-09-11  SL_C_b24n256_cf21_brushless_L4C_g10_s31337004_ls8  (5.1 h)
- 2026-09-12  SL_C_b24n256_cf21_brushless_L4C_g10_s31337003_hd  (2.8 h)
- 2026-09-12  SL_C_b24n256_cf21_brushless_L4C_g10_s31337005_hd  (2.9 h)
- 2026-09-12  SL_C_b24n256_cf21_brushless_L4C_g10_s31337004_hd  (3.0 h)

### translationab
- 2026-09-04  TAB_on_b32n256_cf21_brushless_L4C_s31337002  (5.9 h)
- 2026-09-05  TAB_on_b32n256_cf21_brushless_L4C_s31337005  (5.8 h)
- 2026-09-05  TAB_on_b32n256_cf21_brushless_L4C_s31337003  (5.9 h)
- 2026-09-05  TAB_on_b32n256_cf21_brushless_L4C_s31337004  (6.0 h)
- 2026-09-06  TAB_on_b32n256_cf21_brushless_L4C_s31337006  (5.8 h)

