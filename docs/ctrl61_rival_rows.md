# Rival rows on the f32-estimator lineage (ABI 61/62) — old vs new (CTRL-69, 22/11/2026)

Written by `scripts/ctrl61_rival_rows.py` from the driver `logs/controller/ctrl62/ab_r62.sh` after the ABI-62 landing.
**Partition: TEST** = 35 CF2.1 units 99990101..99990135 × 100 episodes = 3500 episodes per row. Deterministic laws:
one row each, no seed spread. Nothing tuned on these numbers. Cells: stable % / err ° / steady ° / pos m (fails).

```
row                                                            | OLD ABI 60 (f64 Mahony)                 | NEW ABI 61/62 (f32 MahonyF32)          
---------------------------------------------------------------+-----------------------------------------+----------------------------------------
A  MPCOF>rate(h10) est-fed, no bias comp                       | 100.00 / 1.352 / 0.684 / 0.341 (0/3500)   | 100.00 / 1.352 / 0.684 / 0.341 (0/3500)
B  MPCOF>rate(h10) est-fed + bias comp (eq-info COMPARATOR)    | 100.00 / 0.691 / 0.165 / 0.357 (0/3500)   | 100.00 / 0.691 / 0.165 / 0.357 (0/3500)
C  MPCOF>rate(h10) oracle-fed (bound, label law)               | 100.00 / 0.690 / 0.163 / 0.357 (0/3500)   | 100.00 / 0.690 / 0.163 / 0.357 (0/3500)
F  PID firmware cascade, as shipped (run-printed rows)         |  91.14 / 3.333 / 2.759 / 0.352 (310/3500)   |  91.14 / 3.333 / 2.759 / 0.352 (310/3500)
A* cross-check: MPCOF est-fed, run-printed rows                | 100.00 / 1.353 / 0.685 / 0.341 (0/3500)   | 100.00 / 1.353 / 0.685 / 0.341 (0/3500)
```

- Rows A/B/C: `rival_probe_tests::test_units_rival_rows_rescore_f32_lineage` (cargo test --release, fixture
  `scripts/ctrl_indi_val_fixture.py --units test` on the ABI-61 N0 s1 run log; self-check = this build reproduces
  the installed wheel's est-fed row on unit 99990101). OLD = the published rows (docs/ctrl_rival_row_ruling.md §0,
  spec §12.1, f6e084e8), quoted, not re-flown (the ABI-60 wheel is no longer installed).
- Row F and A*: means of the per-unit rival lines the flown runs print (MEMORY-stage test-unit blocks, 0.01 ° /
  0.001 m print precision). OLD = 5 ABI-60 N0 run(s), NEW = 5 ABI-61/62 run(s); max per-unit
  spread across runs of one lineage: old 0.0000, new 0.0000 (0 = deterministic, as expected).
- Units whose printed rival line moved old -> new: PID 0/35, MPCOF est-fed 0/35
  (at print precision; the probe rows above carry full precision).
- **Comparator (ruling §1): row B (equal-information MPCOF).** From this file on, WNN rows of the ABI-61/62
  lineage are set beside the NEW column, never the OLD one.
