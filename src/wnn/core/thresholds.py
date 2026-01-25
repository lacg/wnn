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


class OverfitThreshold(IntEnum):
    """
    Thresholds for overfitting detection based on val/train ratio change.

    The val/train ratio is compared to a baseline. The delta (% change from
    baseline) determines the overfitting status:

    HEALTHY: delta < -1% (improving - val/train ratio decreasing)
    WARNING: delta > 0% (mild overfitting - ratio starting to increase)
    SEVERE: delta > 1% (significant overfitting - needs aggressive action)
    CRITICAL: delta > 3% (severe overfitting - should stop)

    Example:
        Baseline ratio: 250x (val=2000, train=8)
        Current ratio: 262.5x (val=2100, train=8)
        Delta: +5% -> SEVERE threshold exceeded
    """
    HEALTHY = -1    # < -1%: ratio improving, exit diversity mode
    WARNING = 0     # > 0%: any ratio increase, enter mild diversity
    SEVERE = 1      # > 1%: significant increase, enter aggressive diversity
    CRITICAL = 3    # > 3%: severe overfitting, early stop

    def as_float(self) -> float:
        """Return threshold as float percentage."""
        return float(self.value)

    @classmethod
    def get_status(cls, delta: float) -> 'OverfitThreshold':
        """
        Get the threshold status for a given delta value.

        Args:
            delta: The percentage change from baseline (e.g., 3.5 for +3.5%)

        Returns:
            The most severe threshold that delta exceeds.
        """
        if delta > cls.CRITICAL:
            return cls.CRITICAL
        elif delta > cls.SEVERE:
            return cls.SEVERE
        elif delta > cls.WARNING:
            return cls.WARNING
        else:
            return cls.HEALTHY


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
