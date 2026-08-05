"""Airframe + matched PID gains as SOURCED, reusable parameter sets.

WHY THIS EXISTS. The airframe used to be hardcoded in at least three places —
`AttitudeSim::new`'s defaults (controller.rs), `Teacher::from_id` (dagger_train.rs)
and `AttitudePIDConfig`'s hand-tuned gains (pid.py) — with no single definition and
no citation. Three copies of a number is three chances for them to disagree silently,
and a teacher tuned for a plant the sim is not flying is a bug that produces plausible
results rather than a crash. One struct, passed as a parameter, makes that
unrepresentable.

EVERY FIELD CITES ITS SOURCE. `source` is a required field, not a comment: an airframe
that cannot say where it came from does not construct. Values are SI throughout —
kg, m, N, rad, s (Luiz, 05/08/2026). Full provenance and the rejected alternatives are
in docs/disturbance_param_sources.md.

`PidGains` is deliberately in this file rather than its own: it is meaningless apart
from the airframe it was tuned against, and it carries `airframe` so a mismatch is
caught at construction instead of producing a quietly mistuned controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PidAxis:
	"""One axis of a PID loop. `i_limit` is the anti-windup clamp."""

	kp: float
	ki: float
	kd: float
	i_limit: float = 0.0


@dataclass(frozen=True)
class PidGains:
	"""Attitude + rate PID gains, BOUND to the airframe they were tuned against.

	`airframe` must match the `Airframe.name` it is used with — `Airframe.gains()`
	enforces it. Cascaded designs fill both loops; single-loop designs leave `rate`
	as None.

	UNITS ARE NOT SI AND ARE NOT INTERCHANGEABLE. Firmware gains map degrees and
	deg/s to actuator counts; DSL's `*_COEFF_TOR` feed a PWM path scaled by
	MAX_PWM=65535. Our sim's PID emits normalized PWM in [0, 1]. A documented,
	TESTED unit mapping is required before either set drives our plant — inventing
	that mapping would reintroduce exactly the problem this module removes. See
	`unit_note`.
	"""

	airframe: str
	attitude: tuple  # (roll, pitch, yaw) PidAxis — angle -> rate setpoint
	source: str
	rate: Optional[tuple] = None  # (roll, pitch, yaw) PidAxis — rate -> actuator
	unit_note: str = ""


@dataclass(frozen=True)
class Airframe:
	"""A physical vehicle's parameters, in SI, with its provenance attached.

	Field names mirror the Rust `AttitudeSim::new` signature so threading one into
	the sim is a direct splat rather than a translation step (translation layers are
	where unit bugs live).
	"""

	name: str
	mass: float           # kg
	arm_length: float     # m — per-axis torque moment arm (X-config: d*sqrt(2))
	k_thrust: float       # N per pwm^2, per motor (pwm=1 -> k_thrust newtons)
	k_drag: float         # thrust->yaw-torque ratio, m
	inertia: tuple        # (Ixx, Iyy, Izz) kg*m^2
	gravity: float        # m/s^2
	source: str           # REQUIRED — citation for the numbers above
	inertia_source: str   # separate: inertia is often published apart from the rest
	notes: str = ""

	@property
	def thrust_to_weight(self) -> float:
		"""Total available thrust / weight. Below ~1 the vehicle cannot hover."""
		return 4.0 * self.k_thrust / (self.mass * self.gravity)

	@property
	def hover_pwm(self) -> float:
		"""Per-motor PWM that holds hover: thrust ∝ pwm^2, so sqrt(1/t2w)."""
		return (1.0 / self.thrust_to_weight) ** 0.5

	@property
	def max_torque(self) -> float:
		"""One motor at full PWM, N*m — the control-authority scale."""
		return self.arm_length * self.k_thrust

	@property
	def angular_authority(self) -> float:
		"""max_torque / Ixx, rad/s^2. The honest agility comparator: torque alone
		is meaningless without the inertia it has to accelerate."""
		return self.max_torque / self.inertia[0]

	def gains(self, registry_key: Optional[str] = None) -> PidGains:
		"""The PID gains bound to THIS airframe. Refuses a mismatch loudly."""
		key = registry_key or self.name
		g = _GAINS.get(key)
		if g is None:
			raise KeyError(
				f"no PID gains registered for {key!r}; known: {sorted(_GAINS)}")
		if g.airframe != self.name:
			raise ValueError(
				f"gains {key!r} were tuned for airframe {g.airframe!r}, not "
				f"{self.name!r} — using them would mistune the controller silently")
		return g

	@classmethod
	def preset(cls, name: str) -> "Airframe":
		af = _AIRFRAMES.get(name)
		if af is None:
			raise KeyError(
				f"unknown airframe {name!r}; known: {sorted(_AIRFRAMES)}")
		return af

	@classmethod
	def names(cls) -> list:
		return sorted(_AIRFRAMES)


# ---------------------------------------------------------------------------
# LINEAGE A — Crazyflie 2.x as modelled by gym-pybullet-drones.
# Plant and gains come from ONE lineage on purpose: the DSL gains were tuned
# against this exact URDF, so pairing them is sourced rather than assumed.
# ---------------------------------------------------------------------------
_CF2X_URDF = Airframe(
	name="cf2x_urdf",
	mass=0.027,
	arm_length=0.0397,
	# t2w 2.25 * m * g / 4 motors. The URDF gives thrust2weight, not a per-motor
	# newton figure, so this IS a derivation — trivial, but stated.
	k_thrust=2.25 * 0.027 * 9.81 / 4.0,   # 0.148989 N/pwm^2
	k_drag=7.94e-12 / 3.16e-10,           # km/kf = 2.5127e-2 m
	inertia=(1.4e-5, 1.4e-5, 2.17e-5),
	gravity=9.81,
	source="gym_pybullet_drones/assets/cf2x.urdf (Panerati et al. 2021, "
	       "arXiv:2103.02142) — the model Molchanov/QuadSwarm-adjacent work flies",
	inertia_source="same URDF (ixx/iyy/izz on base_link)",
	notes="Props at (+-0.028, +-0.028, 0) — rotor plane AT CoM height, so rotor "
	      "drag has zero moment arm and this model yields no wind-induced attitude "
	      "torque by construction. See the weather-axis note in "
	      "docs/disturbance_param_sources.md.",
)

# ---------------------------------------------------------------------------
# LINEAGE B — Crazyflie 2.1 BRUSHLESS, from Bitcraze's own flight firmware.
# Luiz's focus: heavier, proportionally more thrust, LOWER thrust-to-weight and
# lower agility than the 2.x — the more conservative, more realistic vehicle.
# ---------------------------------------------------------------------------
# INERTIA IS THE ONE UNSOURCED NUMBER HERE, and it is flagged rather than hidden.
# The firmware does not publish inertia (PID needs no plant model, so Bitcraze
# never states it), Bitcraze's crazyflie-simulation repo ships only a 2.x Gazebo
# model with no brushless variant, and no system-ID paper for the brushless was
# found. It is SCALED from Bitcraze's MEASURED 2.x inertia by the geometry ratio
#     I_bl = I_2x * (m_bl/m_2x) * (L_bl/L_2x)^2
# anchoring on a measured value rather than a modelled one. A point-mass
# derivation was rejected: it needs a per-motor mass that is equally unpublished,
# so it would trade one assumption for two.
_CF2X_MEASURED_I = (16.571710e-06, 16.655602e-06, 29.261652e-06)  # Bitcraze Gazebo
_CF2X_MEASURED_M = 0.025 + 4 * 0.0008   # body + 4 rotors = 0.0282 kg
_CF2X_MEASURED_L = 0.031 * (2 ** 0.5)   # props at +-0.031 -> 0.04384 m
_BL_M, _BL_L = 0.0393, 0.050
_BL_SCALE = (_BL_M / _CF2X_MEASURED_M) * (_BL_L / _CF2X_MEASURED_L) ** 2

_CF21_BRUSHLESS = Airframe(
	name="cf21_brushless",
	mass=_BL_M,
	arm_length=_BL_L,
	k_thrust=0.2,                      # THRUST_MAX, N per motor
	k_drag=0.00569278844371417,        # THRUST2TORQUE, m
	inertia=tuple(i * _BL_SCALE for i in _CF2X_MEASURED_I),
	gravity=9.81,
	source="bitcraze/crazyflie-firmware src/platform/interface/"
	       "platform_defaults_cf21bl.h @master (CF_MASS, ARM_LENGTH, THRUST_MAX, "
	       "THRUST2TORQUE)",
	inertia_source="DERIVED, NOT SOURCED — Bitcraze's measured 2.x Gazebo inertia "
	               "(crazyflie-simulation simulator_files/gazebo/crazyflie/model.sdf) "
	               f"scaled by (m_bl/m_2x)*(L_bl/L_2x)^2 = {_BL_SCALE:.4f}. No "
	               "brushless inertia is published anywhere found. TREAT AS AN "
	               "ASSUMPTION: state it in any paper and sensitivity-check it.",
	notes="Bitcraze's 2.x Gazebo model puts rotors at z=+0.021 m, ABOVE the body "
	      "CoM — unlike the gym-pybullet-drones URDF's z=0. If a wind->attitude "
	      "coupling is ever revisited, that offset is the moment arm the URDF "
	      "lineage lacks.",
)

# The 2.x variant of the FIRMWARE lineage, kept as the real-hardware cross-check
# against LINEAGE A. Note it disagrees with the URDF on mass and arm length —
# they are different builds, which is precisely why blending lineages is refused.
_CF2X_FIRMWARE = Airframe(
	name="cf2x_firmware",
	mass=0.029,
	arm_length=0.046,
	k_thrust=0.18,                     # THRUST_MAX (brushed, upper #ifdef branch)
	k_drag=0.0051648627905205285,      # THRUST2TORQUE
	inertia=_CF2X_MEASURED_I,
	gravity=9.81,
	source="bitcraze/crazyflie-firmware platform_defaults_cf2.h @master",
	inertia_source="bitcraze/crazyflie-simulation gazebo model.sdf (MEASURED — "
	               "body 0.025 kg, Ixx 16.5717e-6, Iyy 16.6556e-6, Izz 29.2617e-6, "
	               "with a non-zero Ixy 0.8308e-6 our diagonal model drops)",
	notes="Cross-check only. Disagrees with cf2x_urdf on mass (0.029 vs 0.027) and "
	      "arm (0.046 vs 0.0397) — different builds behind #ifdefs.",
)

_AIRFRAMES = {
	af.name: af for af in (_CF2X_URDF, _CF21_BRUSHLESS, _CF2X_FIRMWARE)
}


# ---------------------------------------------------------------------------
# Re-deriving the SIM's PID gains for a new airframe.
#
# WHY THIS IS NEEDED. `AttitudePIDConfig` / `AttitudePidRs::new_default()` carry
# hand-tuned constants (roll/pitch kp 1.2, kd 0.30) matched to the RETIRED synthetic
# plant. LQR/LQI/MPC/MPCOF re-derive from the airframe automatically; PID does not, so
# on a sourced airframe PID is the only teacher flying another vehicle's tuning. That
# is an uncontrolled variable in the L4 teacher screen — see
# docs/l4_teacher_screen_results.md "PID-teacher tuning currency".
#
# These gains are the SIM's normalized-PWM gains, NOT the firmware/DSL `_GAINS` above
# (those still await a tested unit mapping). Keeping the two separate is deliberate.
# ---------------------------------------------------------------------------
# The plant the stock gains were hand-tuned against (controller.rs:620-626).
LEGACY_TUNED_PLANT = (0.075, 2.4, 0.0023)   # (arm_length, k_thrust, Ixx)
LEGACY_HOVER = 0.5


def roll_pitch_loop_gain(
	arm_length: float, k_thrust: float, inertia_xx: float, hover: float,
) -> float:
	"""Small-signal roll/pitch loop gain G, rad/s^2 per unit of PID output.

	The sim integrates tau = I*omega_dot and thrust ~ pwm^2, so around hover `p` the
	differential pair contributes d(pwm^2) = 4*p per unit command:
	    G = 4 * arm_length * k_thrust * hover / Ixx
	With u = kp*err - kd*rate the closed loop is theta'' + G*kd*theta' + G*kp*theta = 0,
	hence omega_n = sqrt(G*kp) and zeta = kd*sqrt(G) / (2*sqrt(kp)).
	"""
	return 4.0 * arm_length * k_thrust * hover / inertia_xx


def derive_sim_pid_rp(airframe: Airframe, kp_ref: float, kd_ref: float) -> tuple:
	"""Re-derive (kp, kd) for `airframe`'s roll/pitch axis from the reference tuning.

	Returns (kp, kd). `kp_ref`/`kd_ref` are the legacy hand-tuned values that were
	matched to LEGACY_TUNED_PLANT at LEGACY_HOVER.

	TODO(Luiz): choose the invariant to preserve. The options are NOT equivalent and
	the choice decides what the LQ-vs-PID comparison actually measures:

	  (a) preserve (omega_n, zeta) — scale kp by G_ref/G_new and kd by
	      sqrt(G_ref/G_new)... i.e. reproduce the legacy loop shape exactly. Most
	      defensible as "the same controller, ported".
	  (b) preserve the dominant slow pole kp/kd (~4 rad/s) — the response the vehicle
	      actually shows. Note the measured slow pole barely moves anyway (4.42 ->
	      4.08 rad/s), so this is close to a no-op and would leave PID at ~1.64 deg.
	  (c) match the LQR closed-loop bandwidth on this airframe — the fairest
	      *teacher-vs-teacher* comparison, but it tunes PID using LQR's answer, which
	      arguably concedes the plant-model advantage the screen is trying to measure.
	  (d) re-hand-tune per airframe — most faithful to what a practitioner does,
	      least reproducible, and it reintroduces an unsourced number.

	Whatever is chosen must be recorded as the `source` on the resulting PidGains
	(as `inertia_source` does for the derived inertia) — a derived gain that cannot
	say what it preserved is exactly the unsourced constant this module exists to kill.
	"""
	raise NotImplementedError(
		"derive_sim_pid_rp: pick the invariant to preserve — see the TODO above and "
		"docs/l4_teacher_screen_results.md 'PID-teacher tuning currency'")


# ---------------------------------------------------------------------------
# Gains, each bound to one airframe.
# ---------------------------------------------------------------------------
_DSL_UNIT_NOTE = (
	"NOT SI. P/D_COEFF_TOR feed a PWM path with PWM2RPM_SCALE=0.2685, "
	"PWM2RPM_CONST=4070.3, MIN_PWM=20000, MAX_PWM=65535. Our sim's PID emits "
	"normalized PWM in [0,1]; the mapping must be derived and TESTED before use."
)
_FW_UNIT_NOTE = (
	"NOT SI. Attitude loop maps degrees -> deg/s setpoint; rate loop maps deg/s -> "
	"firmware actuator counts. Our sim's PID emits normalized PWM in [0,1]; the "
	"mapping must be derived and TESTED before use."
)

_GAINS = {
	# Single-loop (position/attitude) design; no separate rate loop.
	"cf2x_urdf": PidGains(
		airframe="cf2x_urdf",
		attitude=(
			PidAxis(kp=70000.0, ki=0.0, kd=20000.0),
			PidAxis(kp=70000.0, ki=0.0, kd=20000.0),
			PidAxis(kp=60000.0, ki=500.0, kd=12000.0),
		),
		source="gym_pybullet_drones/control/DSLPIDControl.py (UTIAS DSL: Zhou, Xu, "
		       "Du, Vukosavljev, Ngan, Hou) — P/I/D_COEFF_TOR, tuned against cf2x.urdf",
		unit_note=_DSL_UNIT_NOTE,
	),
	# Cascaded attitude + rate, straight from the brushless flight firmware.
	"cf21_brushless": PidGains(
		airframe="cf21_brushless",
		attitude=(
			PidAxis(kp=6.0, ki=3.0, kd=0.0, i_limit=20.0),
			PidAxis(kp=6.0, ki=3.0, kd=0.0, i_limit=20.0),
			PidAxis(kp=6.0, ki=1.0, kd=0.35, i_limit=360.0),
		),
		rate=(
			PidAxis(kp=200.0, ki=400.0, kd=2.5, i_limit=33.3),
			PidAxis(kp=200.0, ki=400.0, kd=2.5, i_limit=33.3),
			PidAxis(kp=120.0, ki=16.7, kd=0.0, i_limit=166.7),
		),
		source="bitcraze/crazyflie-firmware platform_defaults_cf21bl.h @master — the "
		       "gains the real brushless vehicle flies with",
		unit_note=_FW_UNIT_NOTE,
	),
	"cf2x_firmware": PidGains(
		airframe="cf2x_firmware",
		attitude=(
			PidAxis(kp=6.0, ki=3.0, kd=0.0, i_limit=20.0),
			PidAxis(kp=6.0, ki=3.0, kd=0.0, i_limit=20.0),
			PidAxis(kp=6.0, ki=1.0, kd=0.35, i_limit=360.0),
		),
		rate=(
			PidAxis(kp=250.0, ki=500.0, kd=2.5, i_limit=33.3),
			PidAxis(kp=250.0, ki=500.0, kd=2.5, i_limit=33.3),
			PidAxis(kp=120.0, ki=16.7, kd=0.0, i_limit=166.7),
		),
		source="bitcraze/crazyflie-firmware platform_defaults_cf2.h @master",
		unit_note=_FW_UNIT_NOTE,
	),
}
