"""Training-only horizontal episode regime (`--train-xy-offset`, `--train-init-vxy`;
docs/ctrl_dres_failures.md idea #4).

The rf_c200_dres held-out fails all reach 0.4-0.5 m/s of sideways speed, a regime the
DAgger trainer almost never visits: its episodes start displaced by at most --xy-offset
AT REST, and the teacher's critically damped position loop (omega_n = 1 rad/s) peaks
near 0.37 x0 m/s, about 0.18 m/s from a 0.5 m start. These flags widen ONLY the
trainer's starts — the same split --rg-easy/full-tilt-deg already makes for tilt:

  trainer rollouts + per-round checkpoint eval      -> --train-xy-offset, --train-init-vxy
  GA fitness, held-out, stage-select (scorers),
  threshold calibration, MEMORY recorder            -> --xy-offset, at rest (unchanged)

The calibration sampler is deliberately left alone: refitting the thermometers on the
wider states is idea #1/#2 (the outer-quantile / student-visited calibration), and
mixing the two would make the A/B unreadable.
"""

import math


def train_ic_overrides(args) -> tuple:
	"""(train_xy_offset or None, train_init_vxy) exactly as the CLI set them."""
	xy = getattr(args, "train_xy_offset", None)
	return (None if xy is None else float(xy)), float(getattr(args, "train_init_vxy", 0.0) or 0.0)


def validate_train_ics(args, ec) -> None:
	"""Refuse non-finite / out-of-range bounds, and either flag without the horizontal
	channel (translation and --xy-offset > 0) — there they would be silently ignored."""
	xy, vxy = train_ic_overrides(args)
	if xy is not None and not (math.isfinite(xy) and xy > 0.0):
		raise SystemExit(f"--train-xy-offset {xy:g} must be a finite bound > 0 (m); omit it to "
		                 "train at --xy-offset.")
	if not (math.isfinite(vxy) and vxy >= 0.0):
		raise SystemExit(f"--train-init-vxy {vxy:g} must be a finite bound >= 0 (m/s).")
	if (xy is not None or vxy > 0.0) and not (ec.translation and ec.max_initial_xy_offset_m > 0.0):
		raise SystemExit("--train-xy-offset / --train-init-vxy need --translation and --xy-offset > 0: "
		                 "without the horizontal channel no horizontal state is drawn.")


def train_ic_token(args) -> str:
	"""Provenance train_ic= value: 'off', or 'xy:<m>,vxy:<m/s>' when either is set."""
	xy, vxy = train_ic_overrides(args)
	if xy is None and vxy == 0.0:
		return "off"
	xy_s = "scorer" if xy is None else f"{xy:g}"
	return f"xy:{xy_s},vxy:{vxy:g}"


def train_ic_header(args, ec) -> str:
	"""The [RG-IC] line next to [RG-TILT]: the trainer's horizontal starts vs the scorer's."""
	xy, vxy = train_ic_overrides(args)
	scorer_xy = float(getattr(ec, "max_initial_xy_offset_m", 0.0))
	train_xy = scorer_xy if xy is None else xy
	return (f"[RG-IC] DAgger trainer xy start ±{train_xy:g} m, init v_xy ±{vxy:g} m/s per axis "
	        f"(rollouts + checkpoint eval); scorers/calibration/recorder xy ±{scorer_xy:g} m at rest")
