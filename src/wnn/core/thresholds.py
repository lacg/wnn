"""
Threshold enums for optimization control.

These enums define thresholds for overfitting detection and early stopping.
All values are in percentage points (e.g., 5 = 5%).

Usage:
    from wnn.core import OverfitThreshold, EarlyStopThreshold

    # Get threshold value
    threshold = OverfitThreshold.HEALTHY  # -5 (meaning -5%)

    # Use in comparison (delta is already in percentage)
    if delta < OverfitThreshold.HEALTHY:
        print("Improving!")

    # Access as float for calculations
    as_float = OverfitThreshold.HEALTHY.as_float()  # -5.0
"""

from enum import IntEnum


class OverfitThreshold:
    """
    Thresholds for overfitting detection based on val/train ratio change.

    Float-based (not an enum) for decimal precision.

    The val/train ratio is compared to a baseline. The delta (% change from
    baseline) determines the overfitting status:

    HEALTHY: delta < -0.1% (improving - val/train ratio decreasing)
    WARNING: delta > 0% (mild overfitting - ratio starting to increase)
    SEVERE: delta > 0.1% (significant overfitting - needs aggressive action)
    CRITICAL: delta > 0.3% (severe overfitting - should stop)

    Values are 10x more sensitive than before to catch smaller regressions.
    """
    HEALTHY = -0.1   # < -0.1%: ratio improving, exit diversity mode
    WARNING = 0.0    # > 0%: any ratio increase, enter mild diversity
    SEVERE = 0.1     # > 0.1%: significant increase, enter aggressive diversity
    CRITICAL = 0.3   # > 0.3%: severe overfitting, early stop

    @classmethod
    def get_status(cls, delta: float) -> str:
        """
        Get the threshold status for a given delta value.

        Args:
            delta: The percentage change from baseline (e.g., 0.35 for +0.35%)

        Returns:
            The status name as string.
        """
        if delta > cls.CRITICAL:
            return "CRITICAL"
        elif delta > cls.SEVERE:
            return "SEVERE"
        elif delta > cls.WARNING:
            return "WARNING"
        else:
            return "HEALTHY"


class EarlyStopThreshold(IntEnum):
    """
    Thresholds for early stopping based on improvement rate.

    When the optimization improvement falls below these thresholds for
    a patience period, the optimization stops.

    Values represent minimum required improvement percentage per check period.

    STRICT: 5% - requires significant improvement each period
    MODERATE: 2% - allows moderate plateaus
    LENIENT: 1% - allows longer plateaus (legacy default)
    VERY_LENIENT: 0 - never early stop based on improvement rate
    """
    VERY_LENIENT = 0    # No minimum improvement required
    LENIENT = 1         # 1% improvement required (legacy)
    MODERATE = 2        # 2% improvement required
    STRICT = 5          # 5% improvement required (current default)

    def as_float(self) -> float:
        """Return threshold as float percentage."""
        return float(self.value)
