"""feature_layout — the controller's input features: NAME, ORDER and AXIS FAMILY, one table.

WHAT IT IS
----------
The single Python source of truth for "which input feature sits at index f, what
is it called, and which control axis does it inform" (CTRL-45, 08/11/2026). Before
this, Python only duplicated the feature COUNT (ControllerSpec.num_features) and two
analysis scripts each kept their own partial name list.

The ORDER authority stays in Rust (WnnController::compute_features, mirrored by the
Metal twin: hot path). This module mirrors it block by block, each block gated by
the SAME spec flag, and tests/controller_feature_layout.py holds the two equal on
COUNT (every obs-flag combination) and POSITION (sentinel values read back through
get_last_feature_vector).

AXIS FAMILIES
-------------
A family is a bitmask (a feature may serve more than one axis). Output bank b of
the cascade (CTBR [T, p, q, r]; decouple [T, tau_r, tau_p, tau_y]) serves
BANK_FAMILIES[b]. The axis receptive-field projection (receptive_fields.rs) keeps
each output neuron mostly inside its bank's family; cross-axis signals travel
through its count budget (--rf-cross), so this table is one family per feature
and no extra coupling edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from .evaluator import ControllerSpec


class AxisFamily(IntFlag):
	"""Control axis a feature informs (bitmask; mirrors the u8 masks Rust reads)."""
	NONE = 0
	ROLL = 1
	PITCH = 2
	YAW = 4
	THRUST = 8


ALL_AXES = AxisFamily.ROLL | AxisFamily.PITCH | AxisFamily.YAW | AxisFamily.THRUST

# Output bank -> family. CTBR banks are [collective, p, q, r]; the decouple banks
# [T, tau_roll, tau_pitch, tau_yaw] are the same axes in the same order.
BANK_FAMILIES: tuple[AxisFamily, ...] = (
	AxisFamily.THRUST, AxisFamily.ROLL, AxisFamily.PITCH, AxisFamily.YAW)


@dataclass(frozen=True)
class FeatureDesc:
	"""One input feature (tightly-coupled helper of the layout). `zero_ref`: the
	feature's natural reference is 0 (an error, a rate, a deviation), so the CTRL-48
	centred ladder (--calib-ladder centred) applies to it."""
	name: str
	family: AxisFamily
	zero_ref: bool = False


_R, _P, _Y, _T = AxisFamily.ROLL, AxisFamily.PITCH, AxisFamily.YAW, AxisFamily.THRUST

# THE family map (Luiz 08/11: the family comes from the OUTPUT BANK; this table says
# which bank each feature is "own-axis" for).
#
# CAVEAT, world-frame position block (e_x, v_x -> PITCH; e_y, v_y -> ROLL):
# position error and velocity are WORLD-frame, and world-x maps onto the body
# pitch axis only while yaw stays small. That holds in the CTRL-45 recipe because
# its initial yaw is bounded by --tilt 5 (phased_ga: max_initial_yaw =
# radians(tilt); cos 5 deg = 0.996) and the yaw loop holds heading. Under a
# large-yaw regime (initial yaw tens of degrees, or a yaw-tracking task) world x/y
# mix into BOTH roll and pitch and these four should become ROLL|PITCH.
# The same caveat covers acc_x -> PITCH / acc_y -> ROLL only through the rotor-drag
# term (specific force, ABI 32); their tilt component is body-frame and exact.
FAMILY: dict[str, AxisFamily] = {
	"gyro_x": _R, "gyro_y": _P, "gyro_z": _Y,           # body rates p, q, r
	"acc_x": _P, "acc_y": _R, "acc_z": _T,              # specific force (+ rotor drag)
	"tgt_r": _R, "tgt_p": _P, "tgt_y": _Y,              # per-axis reference
	"tilt": _R | _P, "tilt_i": _R | _P,                 # coupled roll/pitch magnitude
	"roll_err": _R, "pitch_err": _P, "yaw_err_pa": _Y,
	"roll_i": _R, "pitch_i": _P, "yaw_i_pa": _Y,
	"pwm": ALL_AXES,                                    # motor space mixes every axis
	"yaw_err": _Y, "yaw_err_i": _Y,
	"dhat_r": _R, "dhat_p": _P, "dhat_y": _Y,           # L1 student d-hat
	"coll_cmd": _T, "alt_err": _T, "vz": _T,            # vertical channel
	"e_x": _P, "e_y": _R, "v_x": _P, "v_y": _R,         # world frame — see CAVEAT
	"rate_i_x": _R, "rate_i_y": _P, "rate_i_z": _Y,     # CTRL-38 a1 inner-loop I
	"cdhat_x": _R, "cdhat_y": _P, "cdhat_z": _Y,        # CTRL-38 a2 cascade d-hat
	"cresid_x": _R, "cresid_y": _P, "cresid_z": _Y,     # CTRL-48 a3 unabsorbed d-hat
	# CTRL-46: the predicted Mahony bias of the student's roll / pitch reading — BODY
	# frame (rotated by the estimate's own quaternion), so no world-frame caveat.
	"abp_r": _R, "abp_p": _P,
}


# Features whose natural reference is 0 (errors, body rates, deviations, observers,
# integrators). NOT acc_z (≈ g), coll_cmd (≈ hover pwm) or tilt (≥ 0): their
# operating point is elsewhere, so they keep the quantile ladder.
ZERO_REFERENCED: frozenset[str] = frozenset({
	"gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "tgt_r", "tgt_p", "tgt_y",
	"roll_err", "pitch_err", "yaw_err_pa", "roll_i", "pitch_i", "yaw_i_pa",
	"pwm", "yaw_err", "yaw_err_i", "dhat_r", "dhat_p", "dhat_y", "alt_err", "vz",
	"e_x", "e_y", "v_x", "v_y", "rate_i_x", "rate_i_y", "rate_i_z",
	"cdhat_x", "cdhat_y", "cdhat_z", "cresid_x", "cresid_y", "cresid_z",
	"abp_r", "abp_p",
})

# ARM 2 (Luiz 12/11/2026, --yaw-centred-ladder): the yaw-ERROR inputs — the
# yaw-anchor pair and the per-axis yaw error/integral. Only these take the
# yaw-centred ladder (0 a bin edge); every other feature keeps its ladder.
YAW_ERROR_FEATURES: frozenset[str] = frozenset({"yaw_err", "yaw_err_i", "yaw_err_pa", "yaw_i_pa"})

# Block 10 (Luiz 14/11/2026, docs/ctrl48_block10_levers_spec.md): the DENSE ladder's
# families. The yaw INTEGRAL inputs take the yaw cuts scaled by the integral's
# steady-state gain; the tilt-error pair is the L1 input (--obs-tilt-ref-err).
YAW_INTEGRAL_FEATURES: frozenset[str] = frozenset({"yaw_err_i", "yaw_i_pa"})
TILT_ERROR_FEATURES: frozenset[str] = frozenset({"roll_err", "pitch_err"})


def _block_name(name: str) -> str:
	"""Per-motor names ("pwm_3") belong to their block ("pwm")."""
	return name if name in FAMILY else name.rsplit("_", 1)[0]


def _family_of(name: str) -> AxisFamily:
	"""Per-motor names ("pwm_3") share their block's family ("pwm")."""
	return FAMILY[_block_name(name)]


def _descs(names: list[str]) -> tuple[FeatureDesc, ...]:
	return tuple(FeatureDesc(n, _family_of(n), _block_name(n) in ZERO_REFERENCED) for n in names)


def _base_block(spec: "ControllerSpec") -> list[str]:
	return ["gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z", "tgt_r", "tgt_p", "tgt_y"]


def _attitude_block(spec: "ControllerSpec") -> list[str]:
	"""tilt, tilt_i, per-axis errors, per-axis integrals (compute_features order)."""
	pa = 3 if spec.obs_peraxis_yaw else 2
	return (["tilt"] * spec.obs_tilt_p + ["tilt_i"] * spec.obs_tilt_i
	        + ["roll_err", "pitch_err", "yaw_err_pa"][:pa] * spec.obs_peraxis_p
	        + ["roll_i", "pitch_i", "yaw_i_pa"][:pa] * spec.obs_peraxis_i)


def _actuator_block(spec: "ControllerSpec") -> list[str]:
	"""obs_pwm, the yaw-anchor pair and the L1 d-hat."""
	return ([f"pwm_{m}" for m in range(spec.num_motors)] * spec.obs_pwm
	        + ["yaw_err"] * spec.obs_yaw_err + ["yaw_err_i"] * spec.obs_yaw_err_i
	        + ["dhat_r", "dhat_p", "dhat_y"] * (spec.dhat_b is not None))


def _translation_block(spec: "ControllerSpec") -> list[str]:
	"""Scope C vertical (stage 1) then horizontal (stage 2) channels."""
	return (["coll_cmd"] * spec.obs_collective_cmd + ["alt_err"] * spec.obs_alt_err
	        + ["vz"] * spec.obs_vz + ["e_x", "e_y"] * spec.obs_pos_err_xy
	        + ["v_x", "v_y"] * spec.obs_vel_xy)


def _cascade_block(spec: "ControllerSpec") -> list[str]:
	"""CTRL-38/48 cascade blocks, appended LAST (cascade_obs.rs::append): rate-I,
	d-hat, then the unabsorbed residual."""
	c = spec.ctbr or {}
	return (["rate_i_x", "rate_i_y", "rate_i_z"] * bool(c.get("obs_rate_i", False))
	        + ["cdhat_x", "cdhat_y", "cdhat_z"] * (c.get("obs_dhat_plant") is not None)
	        + ["cresid_x", "cresid_y", "cresid_z"] * (c.get("obs_dhat_residual_plant") is not None))


def _att_bias_block(spec: "ControllerSpec") -> list[str]:
	"""CTRL-46 Mahony-bias predictor, appended LAST (after the cascade blocks;
	controller.rs append_att_bias_features)."""
	return ["abp_r", "abp_p"] * bool(getattr(spec, "obs_att_bias_pred", False))


_BLOCKS = (_base_block, _attitude_block, _actuator_block, _translation_block, _cascade_block,
           _att_bias_block)


def feature_layout(spec: "ControllerSpec") -> tuple[FeatureDesc, ...]:
	"""compute_features' canonical order for this spec's obs flags."""
	return _descs([n for block in _BLOCKS for n in block(spec)])


def feature_names(spec: "ControllerSpec") -> list[str]:
	return [d.name for d in feature_layout(spec)]


__all__ = ["AxisFamily", "ALL_AXES", "BANK_FAMILIES", "FeatureDesc", "FAMILY", "ZERO_REFERENCED",
           "YAW_ERROR_FEATURES",
           "feature_layout", "feature_names"]
