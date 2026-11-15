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

# The SAME measurement for the hold-window column (--fit-pos-metric hold, block 10b,
# 15/11/2026): same estimator, same two populations, same three frames (the probe was
# re-run with mean_position_hold_m added; its pos estimates reproduced the six above
# exactly). pos_hold estimates: 2.49 / 5.08 / 4.02 mm (L1+L3), 6.57 / 17.27 / 11.79 mm (L1);
# MEDIAN 5.83 mm → 0.0058 m. The hold column is noisier than pos (a mean over the last
# fifth of the episode only); in the L1 population its noise is about its whole spread
# (17 mm vs 17 mm on the A-C pair). `auto` follows the ranked column.
POS_HOLD_SCALE_FLOOR_M = 0.0058


def fit_pos_metric_from_args(args) -> str:
	"""--fit-pos-metric: "pos" (default, the whole-episode column) or "hold"."""
	m = getattr(args, "fit_pos_metric", None) or "pos"
	if m not in ("pos", "hold"):
		raise SystemExit(f"--fit-pos-metric must be pos or hold, got {m!r}")
	return m


def zrank_pos_floor_from_args(args) -> float:
	"""--zrank-pos-floor: unset → 0.0 (byte-identical); `auto` → the measured floor of the
	column the fitness ranks (POS_SCALE_FLOOR_M, or POS_HOLD_SCALE_FLOOR_M under
	--fit-pos-metric hold); else a non-negative number of metres."""
	v = getattr(args, "zrank_pos_floor", None)
	if v is None:
		return 0.0
	if str(v).strip().lower() == "auto":
		return POS_HOLD_SCALE_FLOOR_M if fit_pos_metric_from_args(args) == "hold" else POS_SCALE_FLOOR_M
	f = float(v)
	if not f >= 0.0:
		raise SystemExit(f"--zrank-pos-floor must be >= 0 metres or `auto`, got {v!r}")
	return f
