"""Where a thermometer feature's cut points come from (flow param ids_cut_method, Plane IDS-70).

QUANTILE      -- (default) the unsupervised placement of ThermometerType (DISTRIBUTIVE =
                 empirical quantiles). Byte-identical to every encoding before IDS-70.
INFO_GAIN_MC  -- pre-check variant E (Plane IDS-63): best-first information-gain cuts against
                 the multiclass label (attack categories), one global budget = the bits the
                 QUANTILE encoding spends on the numeric + categorical features; categoricals
                 become a thermometer over their TRAIN attack-rate rank.
XGB_SPLITS    -- pre-check variant B: every split threshold of an XGBoost model fitted on the
                 numeric features (binary label); categoricals keep their binary code.

Supervised cuts are fitted on the TRAIN partition only (ThermometerEncoder.fit).
"""

from enum import Enum


class CutMethod(Enum):
	QUANTILE = "quantile"
	INFO_GAIN_MC = "info_gain_mc"
	XGB_SPLITS = "xgb_splits"


# Bump when a supervised method's fitting semantics change: it re-keys only the
# flows that use that method (test_cache_key "cut=").
CUT_METHOD_VERSION = "v1"
