"""ThresholdCalibration — how the thermometer ladder is fitted (CTRL-48, 08/11/2026).

WHY (docs/ctrl44b_thermometer_encoding_audit.md): every production fit used 10
calibration episodes at the 5 deg scorer tilt. Features dominated by between-episode
variance (d-hat, rate-I, coll_cmd, alt_err, e_x/e_y, yaw_err) then get ~10 independent
values, so their quantile ladders are a fresh random draw per seed; and the ladder
saturates 27-59% of the 8-30 deg states the DAgger trainer actually visits.

One typed config replaces the five hard-coded `num_episodes=10`. The defaults
reproduce the ABI-44 thresholds exactly (episodes 0-9 of the sampler's stream ARE
today's fit); any non-default value is a lineage break and needs a paired A/B.

  episodes  number of calibration episodes in the flown regime (default 10)
  regime    "flown"     — the run's own episode config (calib_episode_config)
            "train_mix" — the flown episodes POOLED with an equal number spread
                          uniformly over the trainer's curriculum tilts, easy -> full
                          (--rg-easy-tilt-deg .. --rg-full-tilt-deg, 8 -> 30 deg)
  ladder    "quantile"  — uniform quantiles of each feature (legacy)
            "centred"   — for ZERO-REFERENCED features (feature_layout.ZERO_REFERENCED)
                          symmetric thresholds at +-Q_|x|(p_j), p_j = (j/h)^1.5 * p_max,
                          finer near 0; every other feature keeps the quantile ladder
  outer_quantile  the outermost quantile positions [q, 1-q] (both ladders); None keeps
            1/(b+1), b/(b+1). Lives HERE, not as a loose fit argument, so every fit site
            gets it with the rest of the calibration (09/11/2026: as a loose argument it
            reached 2 of 5 fit sites from 10/08 to 09/11 — commit 8542f4c1)
"""

from __future__ import annotations

from dataclasses import dataclass

CALIB_REGIMES = ("flown", "train_mix")
CALIB_LADDERS = ("quantile", "centred")


@dataclass(frozen=True)
class ThresholdCalibration:
	episodes: int = 10
	regime: str = "flown"
	ladder: str = "quantile"
	easy_tilt_deg: float = 8.0
	full_tilt_deg: float = 30.0
	outer_quantile: float | None = None

	def __post_init__(self):
		if int(self.episodes) < 1:
			raise ValueError(f"calibration episodes must be >= 1 (got {self.episodes})")
		if self.regime not in CALIB_REGIMES:
			raise ValueError(f"calibration regime {self.regime!r} — one of {list(CALIB_REGIMES)}")
		if self.ladder not in CALIB_LADDERS:
			raise ValueError(f"calibration ladder {self.ladder!r} — one of {list(CALIB_LADDERS)}")
		if self.outer_quantile is not None and not 0.0 < float(self.outer_quantile) < 0.5:
			raise ValueError(f"calibration outer_quantile must be in (0, 0.5) (got {self.outer_quantile})")

	def mix_tilts_deg(self) -> list[float]:
		"""train_mix: one extra episode per tilt, `episodes` of them, easy -> full."""
		if self.regime != "train_mix":
			return []
		n = int(self.episodes)
		if n == 1:
			return [float(self.full_tilt_deg)]
		step = (float(self.full_tilt_deg) - float(self.easy_tilt_deg)) / (n - 1)
		return [float(self.easy_tilt_deg) + i * step for i in range(n)]

	def token(self) -> str:
		"""Provenance calib= value."""
		base = f"eps:{int(self.episodes)},regime:{self.regime},ladder:{self.ladder}"
		# Appended only when set, so every default-calibration token stays byte-identical.
		return base if self.outer_quantile is None else f"{base},outerq:{float(self.outer_quantile):g}"


@dataclass(frozen=True)
class CalibrationFit:
	"""A fitted ladder (tightly-coupled result of the calibration): the flat
	thresholds and the features whose calibration samples had ZERO spread (the
	measured constant set --drop-constant-features excludes)."""
	thresholds: list
	constant_features: tuple


def calibration_from_args(args) -> ThresholdCalibration:
	"""The run's calibration config from phased_ga's CLI (defaults = ABI 44)."""
	return ThresholdCalibration(
		episodes=int(getattr(args, "calib_episodes", 10)),
		regime=str(getattr(args, "calib_regime", "flown")),
		ladder=str(getattr(args, "calib_ladder", "quantile")),
		easy_tilt_deg=float(getattr(args, "rg_easy_tilt_deg", 8.0)),
		full_tilt_deg=float(getattr(args, "rg_full_tilt_deg", 30.0)),
		outer_quantile=getattr(args, "threshold_outer_quantile", None),
	)


__all__ = ["ThresholdCalibration", "CalibrationFit", "CALIB_REGIMES", "CALIB_LADDERS",
           "calibration_from_args"]
