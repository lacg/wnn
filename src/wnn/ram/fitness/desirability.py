"""Desirability fitness — the shape tables (spec: docs/DESIRABILITY_FITNESS_SHAPES.md).

Lives in wnn.ram.fitness (the shared calculator package) so BOTH substrates
read one table — the controller calculator consumes CONTROLLER_SHAPES via
aggregation="desirability"; IDS will consume IDS_SHAPES when its arm is built.
No wnn.control import happens here (the worker imports this package and must
not need ram_controller installed).

score = sum(w_c * h_c) = weighted half-lives of desirability lost; lower =
better. shape "power": higher-is-better fraction, u = x^k, u(anchor) = 0.5.
shape "exp": lower-is-better cost, u = 2^(-x/anchor), anchor IS the half-life.
Anchors measured from 1,830 held-out rows of the gated weight sweep; the
retained 0.70/8.0 gate calibration becomes the half-anchors of its own curves.
Decisions locked 26/08/2026 (Luiz): steady anchor 8.0, eps floor 2^-20,
A/B rides on the relaunched ladder.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DesirabilityShape:
	"""One metric's utility curve: where ideal is 1 and where u = 0.5."""
	metric: str          # Metrics attribute name (domain mapping stays in Python)
	shape: str           # "power" (higher-better fraction) | "exp" (lower-better cost)
	half_anchor: float   # u(half_anchor) = 0.5
	weight: float        # exponent w_c; 0.0 = shape defined, metric silent


# Controller — weights start as S16noJM UNCHANGED so the pre-registered A/B
# isolates the aggregation change from any weight change (jerk/mono/alt stay
# silent until after the A/B; their shapes are ready).
CONTROLLER_SHAPES: tuple[DesirabilityShape, ...] = (
	DesirabilityShape("stable_rate",             "power", 0.70, 0.25),
	DesirabilityShape("mean_attitude_error_deg", "exp",   8.00, 0.3125),
	DesirabilityShape("mean_steady_error_deg",   "exp",   8.00, 0.4375),
	DesirabilityShape("motor_jerk_mean",         "exp",   0.06, 0.0),
	DesirabilityShape("mono_violations_total",   "exp",   2.00, 0.0),
	DesirabilityShape("mean_altitude_error_m",   "exp",   1.00, 0.0),
)

# IDS — anchors from banked comparators (production Wb F1 93.34 / FPR 8.37;
# UNSW-temp 16b-Wb 88.86 / 8.78). accuracy shape defined but silent (banked:
# weighting accuracy HURTS). Per-class recalls are appended dynamically in
# multiclass mode: one ("power", 0.50, w_recall / K) column per class — the
# anti-QSR device (an aggregate win bought with recall collapse multiplies
# near-zero utilities and dies).
IDS_SHAPES: tuple[DesirabilityShape, ...] = (
	DesirabilityShape("f1",       "power", 0.80, 0.35),
	DesirabilityShape("fpr",      "exp",   0.10, 0.35),
	DesirabilityShape("ce",       "exp",   1.00, 0.10),   # anchor: fit per dataset family, then FROZEN
	DesirabilityShape("accuracy", "power", 0.80, 0.0),
)


def flatten(shapes: tuple[DesirabilityShape, ...]) -> tuple[list[float], list[str], list[float]]:
	"""(weights, shape kinds, half anchors) for desirability_fitness_combine —
	only the metrics with weight > 0 (silent shapes are documentation until a
	weight decision turns them on)."""
	active = [s for s in shapes if s.weight > 0.0]
	return ([s.weight for s in active],
	        [s.shape for s in active],
	        [s.half_anchor for s in active])
