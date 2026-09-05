Do the FITNESS rule and the GATE-DISTANCE yardstick order the archive the same way?

WHY THIS EXISTS (05/09/2026, Luiz's question). The leaderboard ranks finished runs on
gate distance; the GA ranks genomes on the weighted gated combine. They are different
functions of different subsets of the metrics, so "does the yardstick agree with the
optimiser, and if not, which one is right?" is a real question and it was unanswerable
from banked data — until you notice the per-seed RESULT lines carry `reward`, which is
the one column the fitness needs and the marker summary lines drop.

    GATE DISTANCE (the leaderboard yardstick — an ABSOLUTE scale)
        gd = 0.5556*(err/8.0) + 0.4444*min(K*-log2(stable), 20.0),  K = log0.5/log0.70
        · 2 of the 5 reported metrics: stable and err ONLY.
        · err LINEAR; stable through a LOG (a point is worth ~3.3x more at 30% than 98%).
        · gate violation folded into the same sum, continuously.
        · absolute: a number means the same thing in any cohort, any year.

    FITNESS (what actually selects genomes — a RELATIVE, gated combine)
        feasible  = stable >= 0.70 AND err <= 8.0 deg           (the PHYSICAL pair)
        Deb's rules: any feasible beats any infeasible; among infeasible, smaller
        normalised violation wins; among feasible, the base combine decides, computed
        over the FEASIBLE SUBSET ONLY.
        base combine = zrank over columns, z = (x-median)/(1.4826*MAD) clamped to +-3,
        weighted:   reward  w=0.3125  (higher better)   <- NOT err; the reward field
                    stable  w=0.2500  (higher better)
                    steady  w=0.4375  (lower  better)   <- gate distance cannot see this
        · relative: a score only means something against the population it was ranked in.

WHAT THIS SCRIPT DOES. Reads every banked marker, recovers the MEMORY stage's held-out
row per report seed from the .out (means over the 5 seeds, so it matches the marker's
MULTI-SEED line), then ranks the whole archive TWICE: once by gate distance, once by
handing those same rows to the SHIPPED wheel combine (ram_controller.gated_fitness_combine
— never a reimplementation, per CLAUDE.md). Reports Kendall tau, the inversion count, and
the rows the two rules disagree about most.

⚠️ WHAT THIS IS NOT. It does NOT re-rank anything, change any fitness, or touch a run.
The live fitness ranks a GA POPULATION on DURING-SEARCH pool scores; this ranks the
ARCHIVE on HELD-OUT scores. Same rule, different population — so it answers "do the two
rules order the same rows differently", not "what would the GA have done". A zrank is
relative, so every score here shifts if the archive membership changes. Read the
DISAGREEMENTS, not the absolute fitness numbers.

==============================================================================
ALTITUDE REGIMEN
==============================================================================
  altitude regimen — 102 runs, MEMORY stage held-out, 5 report seeds each
  Kendall tau (gate-distance vs fitness) = +0.929
    concordant pairs 4967   DISCORDANT 184   tied 0   -> 3.6% of pairs invert

  Worst disagreements (|rank difference|), gate-dist rank vs fitness rank:
    gd#  fit#   Δ   gate-dist   fitness   stable     err  steady   tag
     13    28   +15      0.2252    0.0470   93.6%    2.06    2.14   SL_C_b36n256_cf21_brushless_L4C_g10_s3133700
     29    19   -10      0.2896   -0.2147   90.8%    2.44    1.97   FAB_harmonic_c10_cf21_brushless_L4C_s3133700
     38    48   +10      0.3876    1.5596   89.0%    3.49    3.65   GWS_C10_cf21_brushless_L4C_s31337006
     28    20    -8      0.2845   -0.1948   91.8%    2.56    2.02   GWS_C10_cf21_brushless_L4C_s31337003
     21    13    -8      0.2638   -0.3705   92.2%    2.34    1.55   GWS_STEADY40_cf21_brushless_L4C_s31337005
     26    33    +7      0.2727    0.3163   93.6%    2.74    2.56   FAB_harmonic_c10_cf21_brushless_L4C_s3133700
     91    98    +7      3.5160    7.3468   35.2%   31.89   51.92   AW_C10_alt020_b18n32_s31337002
     66    73    +7      1.7070    4.8141   55.2%   13.92   15.61   AW_C10_alt035_b18n32_s31337002
     89    82    -7      3.3253    5.3793   15.0%   13.85   15.36   SL_A_b10n32_cf21_brushless_L4C_s31337002
     19    26    +7      0.2431   -0.0963   93.8%    2.35    1.81   SL_C_b36n96_cf21_brushless_L4C_g10_s31337002
     44    51    +7      0.4397    1.8204   86.6%    3.75    3.92   SL_C_b36n64_cf21_brushless_L4C_g10_s31337002
     25    18    -7      0.2670   -0.2196   92.4%    2.43    2.01   GWS_S16noJM_cf21_brushless_L4C_s31337003

  Top 10 by each rule, side by side:
    #   by GATE-DISTANCE                              |  by FITNESS
     1  TAB_on_b32n256_cf21_brushless_L4C_s31337002   |  TAB_on_b32n256_cf21_brushless_L4C_s31337004
     2  SL_C_b32n256_cf21_brushless_L4C_g10_s3133700  |  SL_C_b32n256_cf21_brushless_L4C_g10_s3133700
     3  SL_C_b28n256_cf21_brushless_L4C_g10_s3133700  |  TAB_on_b32n256_cf21_brushless_L4C_s31337002
     4  TAB_on_b32n256_cf21_brushless_L4C_s31337004   |  SL_C_b28n256_cf21_brushless_L4C_g10_s3133700
     5  SL_C_b24n256_cf21_brushless_L4C_g10_s3133700  |  SL_C_b28n256_cf21_brushless_L4C_g10_s3133700
     6  TAB_on_b32n256_cf21_brushless_L4C_s31337003   |  TAB_on_b32n256_cf21_brushless_L4C_s31337003
     7  SL_C_b28n256_cf21_brushless_L4C_g10_s3133700  |  SL_C_b32n256_cf21_brushless_L4C_g10_s3133700
     8  SL_C_b32n256_cf21_brushless_L4C_g10_s3133700  |  SL_C_b24n256_cf21_brushless_L4C_g10_s3133700
     9  GWS_C10noJM_cf21_brushless_L4C_s31337005      |  GWS_C10noJM_cf21_brushless_L4C_s31337005
    10  GWS_S16noJM_cf21_brushless_L4C_s31337005      |  GWS_S16noJM_cf21_brushless_L4C_s31337005

==============================================================================
ATTITUDE-ONLY (a different task — never pooled with the above)
==============================================================================
  attitude-only — 89 runs, MEMORY stage held-out, 5 report seeds each
  Kendall tau (gate-distance vs fitness) = +0.913
    concordant pairs 3732   DISCORDANT 169   tied 15   -> 4.3% of pairs invert

  Worst disagreements (|rank difference|), gate-dist rank vs fitness rank:
    gd#  fit#   Δ   gate-dist   fitness   stable     err  steady   tag
     34    21   -13      0.0931   -0.3996  100.0%    1.34    0.74   CMT_mpcof_cf21_brushless_L4C_s31337003
      5    16   +11      0.0635   -0.5163   99.6%    0.84    0.59   S1_lqi_sn4_cf21_brushless_L4C_s31337002
     11    22   +11      0.0686   -0.3959   99.2%    0.84    0.50   S1_lqi_sn8_cf21_brushless_L4C_s31337003
     51    61   +10      0.1183    1.2993   99.8%    1.67    1.39   DOBF_lqr_on_cf21_brushless_L4C_s31337002
     45    37    -8      0.1075   -0.1785   99.8%    1.51    0.80   CLS_lqi_c25_L16_cf21_brushless_L4C_s31337002
     33    40    +7      0.0925   -0.1278   99.4%    1.22    0.68   CAB_afcal_mpcof_cf21_brushless_L4C_s31337002
     46    39    -7      0.1099   -0.1688  100.0%    1.58    0.81   CMT_lqi_cf21_brushless_L4C_s31337003
     37    43    +6      0.0971   -0.0409   99.6%    1.33    0.79   CAB_synth_mpcof_cf21_brushless_L4C_s31337002
     47    53    +6      0.1127    0.5065   98.4%    1.33    0.73   CAB_afcal_mpcof_cf21_brushless_L4C_s31337003
     40    34    -6      0.0995   -0.2385  100.0%    1.43    0.91   ALP_lqi_L64_cf21_brushless_L4C_s31337003
     27    33    +6      0.0889   -0.2471   99.6%    1.21    0.61   ALP2_lqi_L16_cf21_brushless_L4C_s31337002
     38    44    +6      0.0971   -0.0409   99.6%    1.33    0.79   L1R_s16plain_mpcof_cf21_brushless_L4C_s31337

  Top 10 by each rule, side by side:
    #   by GATE-DISTANCE                              |  by FITNESS
     1  S1_lqi_sn4_cf21_brushless_L4C_s31337004       |  ALP_lqi_L64_cf21_brushless_L4C_s31337002
     2  S1_lqi_sn8_cf21_brushless_L4C_s31337002       |  E1_lqi_c30_refiton_cf21_brushless_L4C_s31337
     3  ALP_lqi_L64_cf21_brushless_L4C_s31337002      |  E1_lqi_c30_refitoff_cf21_brushless_L4C_s3133
     4  E1_lqi_c30_refiton_cf21_brushless_L4C_s31337  |  OQ_lqi_c30_cf21_brushless_L4C_s31337003
     5  S1_lqi_sn4_cf21_brushless_L4C_s31337002       |  E2_lqi_g10_c30_refitoff_cf21_brushless_L4C_s
     6  E1_lqi_c30_refitoff_cf21_brushless_L4C_s3133  |  S1_lqi_sn4_cf21_brushless_L4C_s31337004
     7  OQ_lqi_c30_cf21_brushless_L4C_s31337003       |  ALP2_lqi_L64_cf21_brushless_L4C_s31337002
     8  E2_lqi_g10_c30_refitoff_cf21_brushless_L4C_s  |  S1_lqi_sn8_cf21_brushless_L4C_s31337002
     9  S1_lqi_sn8_cf21_brushless_L4C_s31337004       |  E1_lqi_c30_refitoff_cf21_brushless_L4C_s3133
    10  ALP2_lqi_L64_cf21_brushless_L4C_s31337002     |  OQ_lqi_c30_cf21_brushless_L4C_s31337002
