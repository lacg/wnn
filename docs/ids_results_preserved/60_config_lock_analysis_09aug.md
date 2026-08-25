# =====================================================================
# SECTION 6 — PRESERVED: config-lock analysis written 09/08/2026
# (hand-written; kept verbatim. Its ciciot n=9 cells are now n=10 —
#  the 3 outstanding abl runs completed; Sections 2-4 supersede those cells.)
# =====================================================================

# Config ranking — held-out GA best_f1 val_cal (mean±std over runs)

## unswt — UNSW-NB15 temporal_3way (16b Wb)

    Config   | n  | F1 val_cal     | FPR val_cal    | Acc val_cal
    ---------+----+----------------+----------------+---------------
    quad     | 10 |  86.54± 1.87 |  13.58± 7.08 |  86.70± 1.73
    ablqsr   | 10 |  83.30± 5.36 |  27.83±18.68 |  84.37± 4.47
    abl3s    | 10 |  79.29± 0.04 |  42.01± 0.02 |  81.02± 0.04
    ablpln   | 10 |  79.24± 0.03 |  42.18± 0.04 |  80.99± 0.03
    abl2s    | 10 |  79.20± 0.10 |  42.01± 0.02 |  80.92± 0.10
    abl2big  | 10 |  79.12± 0.08 |  42.07± 0.10 |  80.85± 0.09

## unswr — UNSW-NB15 random_3way (64b Wb)

    Config   | n  | F1 val_cal     | FPR val_cal    | Acc val_cal
    ---------+----+----------------+----------------+---------------
    ablqsr   | 10 |  94.33± 0.07 |   0.62± 0.07 |  99.14± 0.02
    quad     | 10 |  93.54± 0.15 |   1.08± 0.12 |  98.93± 0.04
    ablpln   | 10 |  93.52± 0.10 |   1.04± 0.04 |  98.94± 0.02
    abl3s    | 10 |  93.50± 0.03 |   1.12± 0.00 |  98.92± 0.00
    abl2big  | 10 |  93.47± 0.03 |   1.12± 0.00 |  98.92± 0.01
    abl2s    | 10 |  93.47± 0.03 |   1.12± 0.00 |  98.91± 0.00

## cicids — CICIDS2017 random_3way (96b Wa)

    Config   | n  | F1 val_cal     | FPR val_cal    | Acc val_cal
    ---------+----+----------------+----------------+---------------
    quad     | 10 |  99.59± 0.04 |   0.09± 0.02 |  99.74± 0.02
    ablqsr   | 10 |  99.33± 0.06 |   0.25± 0.05 |  99.58± 0.04
    ablpln   | 10 |  99.27± 0.03 |   0.28± 0.04 |  99.54± 0.02
    abl2big  | 10 |  99.13± 0.01 |   0.60± 0.01 |  99.45± 0.01
    abl3s    | 10 |  99.11± 0.02 |   0.61± 0.02 |  99.43± 0.02
    abl2s    | 10 |  99.08± 0.03 |   0.62± 0.03 |  99.41± 0.02

## ciciot — CIC-IoT-2023 neto_subsample random_3way (96b Wc)

    Config   | n  | F1 val_cal     | FPR val_cal    | Acc val_cal
    ---------+----+----------------+----------------+---------------
    quad     | 10 |  92.82± 0.11 |   8.66± 1.03 |  96.42± 0.05
    ablqsr   |  9 |  92.07± 0.38 |  11.16± 0.78 |  96.09± 0.21
    ablpln   |  9 |  92.02± 0.50 |  13.42± 1.02 |  96.15± 0.25
    abl3s    |  9 |  81.38± 1.36 |  36.64± 6.31 |  91.54± 0.80
    abl2s    | 10 |  77.70± 1.49 |  44.83± 8.69 |  90.15± 1.74


# Decision — config lock for the 100-run SP cohorts (90 new + existing 10 per dataset)

## QUAD-run identification (veto if wrong)

The "existing 10" per dataset are the completed SP-*-bin-*-n30 flows. They carry NO
memory_mode param, and worker.py resolves `params.get("memory_mode", "QUAD_WEIGHTED")`,
so they ran QUAD_WEIGHTED. The only config difference vs abl2s is memory_mode
(verified key-by-key diff; seeds overlap with the abl groups). Counted flows:

    unswt : 4404 4409 4414 4419 4424 4429 4434 4439 4444 4449
    unswr : 4405 4410 4415 4420 4425 4430 4435 4440 4445 4450
    cicids: 4406 4411 4416 4421 4426 4431 4436 4441 4446 4451
    ciciot: 4407 4412 4417 4422 4427 4432 4437 4442 4447 4452

(4539 SP-unswt-mcsmoke is a multiclass smoke test — excluded. Flows 4454-4538
bin-n30/ciciot46m are paused, not counted.)

## Per-config cost (avg minutes per completed run)

    Config   |  unswt |  unswr | cicids | ciciot
    ---------+--------+--------+--------+--------
    quad     |    3.9 |   22.2 |   71.6 |   66.1
    abl2s    |   17.0 |   24.6 |  177.7 |  166.5
    abl2big  |   13.3 |   19.2 |  119.1 |      —
    abl3s    |   24.0 |  159.8 |  512.4 |  195.5
    ablpln   |   46.5 |  377.4 |  601.8 |  306.1
    ablqsr   |   44.8 |  342.0 |  485.2 |  118.9

## Ranking verdict per dataset (GA best_f1, val_cal held-out, n=10 unless noted)

    unswt  : QUAD wins outright — 86.54±1.87 F1 / 13.58±7.08 FPR. Every non-QUAD mode
             except QSR COLLAPSES to a saturated detector (F1 ~79.2, FPR ~42, std <0.15;
             fixed_05 shows FPR ~100% => memory saturates to "attack" at 16b temporal).
             QSR is bimodal (83.30±5.36, FPR 27.8±18.7) — some seeds escape, some don't.
             Difference QUAD vs best non-QUAD is >> within-config std. DECISIVE.
    unswr  : QSR statistically beats QUAD — 94.33±0.07 vs 93.54±0.15 F1 (+0.79pp),
             FPR 0.62±0.07 vs 1.08±0.12 (-0.46pp). At n=10 the gap is >5x the larger
             within-config std — a real effect. Everything else ties QUAD within ~0.07pp.
    cicids : QUAD wins — 99.59±0.04 vs QSR 99.33±0.06 F1, FPR 0.09±0.02 vs 0.25±0.05.
             Margins >> std. DECISIVE.
    ciciot : QUAD wins — 92.82±0.11 vs QSR 92.07±0.38 (n=9) F1, FPR 8.66±1.03 vs
             11.16±0.78. Margins >> std. DECISIVE. QUAD's val_cal point (92.82/8.66/96.42)
             sits right at the standing WNN-vs-XGBoost reference (93.34/8.37/96.71).

## Recommendation

LOCK QUAD_WEIGHTED for all four dataset tracks.

  - QUAD wins 3/4 tracks decisively and is the ONLY mode that does not collapse on
    unswt; it is also the cheapest mode on every dataset (90 new QUAD runs cost
    ~6h/33h/107h/99h for unswt/unswr/cicids/ciciot vs 8-27x more for PLN/QSR).
  - Only QUAD lets the existing 10 runs count toward the 100 — any other lock means
    100 NEW runs for that track, not 90.
  - The one honest caveat: on unswr, QSR beat QUAD by +0.79pp F1 / -0.46pp FPR
    (significant at n=10). If the paper wants the best unswr headline, a separate
    100-run QSR-unswr cohort (est. ~570h at 342m/run) is the price; as an ablation
    finding ("QSR helps only on the easiest split, at 15x cost") the n=10 result
    already carries the point. Recommend NOT switching the main cohort.

Suggested new-flow naming: SP-{ds}-quad-{width}W{x}-n90-r{seed} (or continue bin-n30
resume-style naming); memory_mode may be set explicitly to "QUAD_WEIGHTED" to make the
cohort self-documenting — it is behaviorally identical to omitting the key.

## Data-quality flags

  1. ciciot has NO abl2big group (never created) — 19 config groups, not 20.
  2. 3 abl runs outstanding, all ciciot: 1 running (abl3s), 2 queued (ablpln, ablqsr)
     — those groups report n=9. Untouched per worker discipline.
  3. unswt non-QUAD collapse (above) is itself a paper-worthy ablation finding:
     graduated QUAD nudging prevents write-saturation where BINARY/TERNARY/PLN commit.
  4. Architecture (neurons/bits) headers are unresolvable for several unswt-abl and
     ciciot GS groups — the winners' genome_hash was never persisted to `genomes`
     (annotated "arch resolvable for X/N" inline). Metrics coverage is 100%.
  5. best_fpr genomes are frequently degenerate (FPR ~0 with F1 ~31-75, or FPR ~100%)
     — known pattern; Pareto mining uses best_ce/best_acc/best_f1 points instead.
  6. All numbers are held-out report-set values (Protocol v2, val-calibrated modes);
     no iterations.best_f1 anywhere in this report.
  7. Base-rate vigilance (ciciot): accuracy runs ~3.5pp above F1 — rank on F1/FPR,
     as done above.



---

