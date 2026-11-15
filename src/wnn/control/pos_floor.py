"""CTRL-48 (Luiz 15/11/2026): the pos-only scale floor of the ZRank fitness combine.

The zscore combine scales each column by 1.4826·MAD across the ranked candidates. Under
the L3 altitude integrator the candidates' pos values cluster within millimetres, the
MAD collapses, and a 5 mm difference (inside measured comparison noise) clamps at
±zrank_clamp — the s1 stage-select that crowned GRID#2 over a 0.19°-better MEMORY#2
(docs/ctrl48_block10_2x2_s1_diagnosis.md). The floor bounds that denominator for the pos
column ONLY: scale = max(1.4826·MAD, POS_SCALE_FLOOR_M).

POS_SCALE_FLOOR_M is a MEASUREMENT, by the CTRL-21 estimator (scripts/crn_noise_probe.py,
the method behind FitnessCalculatorControllerHarmonic.MEASURED_SCALE_FLOORS): a run's
57 MEMORY-stage genomes re-scored on three CRN frames of 5 pools × 200 episodes; per
frame pair the robust spread across genomes of D_g = pos_g(F1) − pos_g(F2) is the
within-frame comparison noise. Two populations (block-10 s1: L1 and L1+L3) × three pairs
= six estimates: 1.74 / 3.38 / 2.90 (L1+L3), 2.19 / 6.43 / 5.00 mm (L1); MEDIAN 3.14 mm
→ 0.0031 m. The stage-select frame (20 val units × 100 episodes = 2× the episodes) would
scale it by 1/√2 ≈ 2.2 mm; one conservative value serves both sites.
"""

POS_SCALE_FLOOR_M = 0.0031


def zrank_pos_floor_from_args(args) -> float:
	"""--zrank-pos-floor: unset → 0.0 (byte-identical); `auto` → POS_SCALE_FLOOR_M;
	else a non-negative number of metres."""
	v = getattr(args, "zrank_pos_floor", None)
	if v is None:
		return 0.0
	if str(v).strip().lower() == "auto":
		return POS_SCALE_FLOOR_M
	f = float(v)
	if not f >= 0.0:
		raise SystemExit(f"--zrank-pos-floor must be >= 0 metres or `auto`, got {v!r}")
	return f
