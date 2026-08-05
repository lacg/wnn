"""The Crazyflie flight firmware's cascaded attitude PID, faithful to its source.

WHY A SECOND PID CLASS EXISTS. `pid.py`'s `AttitudePID` is a SINGLE loop in radians
emitting PWM directly, with gains hand-tuned against a retired synthetic plant and no
citation. This one is the controller the real vehicle flies: a CASCADE (angle -> rate
setpoint -> actuator counts) in DEGREES at 500 Hz, whose gains come from
`platform_defaults_cf21bl.h`. The two are not interchangeable and neither is a tuning
of the other — kp 6.0 (deg -> deg/s) and kp 1.2 (rad -> pwm) are different quantities.
`pid.py` is retained only to replay pre-05/08/2026 results; all new work uses this.

EVERY CONSTANT IS SOURCED. The full derivation, with the file each line came from, is
in docs/disturbance_param_sources.md "THE UNIT MAPPING — DERIVED AND SOURCED". The
short version:
  - both loops at ATTITUDE_RATE = 500 Hz, output HELD across the 1 kHz sim tick
    (stabilizer_types.h RATE_DO_EXECUTE)
  - degrees and deg/s (stabilizer_types.h: attitude_t "// deg", gyro "// deg/s")
  - integ += err*dt; out = kp*err + ki*integ + kd*deriv, deriv on the MEASUREMENT;
    iLimit and outputLimit both constrain (pid.c)
  - counts are linear in FORCE: f_i = counts_i/65535, and our sim has
    thrust_i = k_thrust*pwm_i^2 with k_thrust == THRUST_MAX, so pwm_i = sqrt(f_i)
    (power_distribution_quadrotor.c + platform_defaults_cf21bl.h)

MIXER NOTE — READ BEFORE USING ON A NEW AIRFRAME. The firmware mixes X-config
(all four motors contribute to roll AND pitch, halved). Our `AttitudeSim::body_torque`
mixes '+'-config (roll from motors 1/3 only, pitch from 0/2 only). We emit the '+' form
the sim expects; the geometry difference is absorbed by `Airframe.arm_length`, whose
documented meaning is the PER-AXIS moment arm (X-config: d*sqrt(2)). Whether
cf21_brushless's arm_length should therefore be 0.050 (the firmware's motor radius) or
0.0707 is an OPEN plant-fidelity question — see the same doc. This class does not
resolve it and does not depend on it.
"""

from __future__ import annotations

from math import asin, atan2, degrees, pi, sqrt
from typing import Tuple

from .airframe import Airframe, PidAxis, PidGains

# power_distribution_quadrotor.c: maxAllowedThrust = UINT16_MAX; roll/pitch halved.
_COUNTS_FULL_SCALE = 65535.0
# attitude_pid_controller.c: rate output passes saturateSignedInt16.
_INT16_SAT = 32767.0


class _Pid:
	"""One firmware PID channel. Mirrors pid.c exactly, including the D-on-measurement
	form and the two independent constrains (integral, then output)."""

	def __init__(self, axis: PidAxis, dt: float, output_limit: float):
		self.axis = axis
		self.dt = dt
		self.output_limit = output_limit
		self.integ = 0.0
		self.prev_measured = 0.0
		self.first = True

	def reset(self) -> None:
		self.integ = 0.0
		self.prev_measured = 0.0
		self.first = True

	def update(self, setpoint: float, measured: float, wrap: bool) -> float:
		"""pid.c pidUpdate: integrate, clamp, derive on measurement, clamp output.

		`wrap` mirrors pidUpdate's `updateError`/shouldWrap argument — the firmware
		wraps the yaw error into (-180, 180] and leaves roll/pitch unwrapped.
		"""
		error = setpoint - measured
		if wrap:
			error = _wrap_deg(error)
		self.integ += error * self.dt
		if self.axis.i_limit != 0.0:
			self.integ = _clamp(self.integ, -self.axis.i_limit, self.axis.i_limit)
		deriv = 0.0 if self.first else -(measured - self.prev_measured) / self.dt
		self.prev_measured = measured
		self.first = False
		out = self.axis.kp * error + self.axis.ki * self.integ + self.axis.kd * deriv
		if self.output_limit != 0.0:
			out = _clamp(out, -self.output_limit, self.output_limit)
		return out


class AttitudePidFirmware:
	"""Cascaded attitude+rate PID with firmware-sourced gains, emitting the 4 PWMs
	`AttitudeSim` expects.

	`hover_force` is the normalized collective force that holds hover on this airframe
	(m*g / (4*THRUST_MAX)). The sim has no translational state, so it only sets the
	operating point the differential rides on — but it is the airframe's real one, not
	a hardcoded 0.5.
	"""

	def __init__(
		self,
		airframe: Airframe,
		gains: PidGains,
		main_loop_hz: int = 1000,
		attitude_hz: int = 500,
	):
		if gains.rate is None:
			raise ValueError(
				f"gains for {gains.airframe!r} have no rate loop; the firmware "
				"controller is a cascade and cannot run single-loop gains")
		if gains.airframe != airframe.name:
			raise ValueError(
				f"gains were tuned for {gains.airframe!r}, not {airframe.name!r}")
		self.airframe = airframe
		self.gains = gains
		self.decimation = main_loop_hz // attitude_hz
		dt = 1.0 / attitude_hz
		# Attitude loop output is a rate setpoint in deg/s — the firmware leaves it
		# unlimited (no outputLimit in attitude_pid_controller.c), so 0.0 = off.
		self.att = [_Pid(a, dt, 0.0) for a in gains.attitude]
		self.rate = [_Pid(a, dt, _INT16_SAT) for a in gains.rate]
		self.hover_force = airframe.mass * airframe.gravity / (
			4.0 * airframe.k_thrust)
		self.tick = 0
		self.held = (0.0, 0.0, 0.0)

	def reset(self) -> None:
		for p in self.att + self.rate:
			p.reset()
		self.tick = 0
		self.held = (0.0, 0.0, 0.0)

	def step(
		self,
		q: Tuple[float, float, float, float],
		gyro: Tuple[float, float, float],
		target_rpy: Tuple[float, float, float],
	) -> Tuple[float, float, float, float]:
		"""One 1 kHz sim tick. q and gyro are SI (rad, rad/s) as the sim provides;
		conversion to the firmware's degrees happens here. Returns 4 PWMs in [0,1]."""
		if self.tick % self.decimation == 0:
			self.held = self._update_cascade(q, gyro, target_rpy)
		self.tick += 1
		return self._mix(self.held)

	def _update_cascade(
		self,
		q: Tuple[float, float, float, float],
		gyro: Tuple[float, float, float],
		target_rpy: Tuple[float, float, float],
	) -> Tuple[float, float, float]:
		"""Angle error (deg) -> rate setpoint (deg/s) -> actuator counts, per axis."""
		rpy = _quat_to_euler(q)
		out = []
		for i in range(3):
			# Faithful to attitude_pid_controller.c: the attitude PID is handed the
			# ACTUAL angle as its measurement (so its D-term differentiates the angle
			# — this matters for yaw, whose kd is 0.35, and only for yaw), with the
			# error wrapped for yaw alone (pidUpdate's shouldWrap = true for yaw).
			is_yaw = i == 2
			rate_sp = self.att[i].update(
				degrees(target_rpy[i]), degrees(rpy[i]), is_yaw)
			out.append(self.rate[i].update(rate_sp, degrees(gyro[i]), False))
		return (out[0], out[1], out[2])

	def _mix(self, counts: Tuple[float, float, float]) -> Tuple[float, ...]:
		"""Counts -> normalized force -> PWM, in the sim's '+' motor order.

		Sim convention (controller.rs body_torque): M0 front(+x), M1 right(-y),
		M2 rear(-x), M3 left(+y); roll = L*(-t1+t3), pitch = L*(-t0+t2),
		yaw = k*(t0-t1+t2-t3). Firmware halves roll/pitch before mixing.
		"""
		r = counts[0] / 2.0 / _COUNTS_FULL_SCALE
		p = counts[1] / 2.0 / _COUNTS_FULL_SCALE
		y = counts[2] / _COUNTS_FULL_SCALE
		h = self.hover_force
		# +roll must raise the LEFT motor (M3) and drop the RIGHT (M1); +pitch must
		# raise the REAR (M2) and drop the FRONT (M0) — matching body_torque's signs.
		forces = (h - p + y, h - r - y, h + p + y, h + r - y)
		return tuple(sqrt(_clamp(f, 0.0, 1.0)) for f in forces)


def _clamp(v: float, lo: float, hi: float) -> float:
	return lo if v < lo else (hi if v > hi else v)


def _wrap_deg(a: float) -> float:
	"""Wrap degrees into (-180, 180]."""
	while a > 180.0:
		a -= 360.0
	while a <= -180.0:
		a += 360.0
	return a


def _quat_to_euler(q: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
	"""Body-to-world unit quaternion -> (roll, pitch, yaw) rad. Same form as pid.py's
	so the two controllers cannot disagree about what attitude they were handed."""
	w, x, y, z = q
	roll = atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
	sinp = 2.0 * (w * y - z * x)
	pitch = pi / 2.0 if sinp >= 1.0 else (-pi / 2.0 if sinp <= -1.0 else asin(sinp))
	yaw = atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
	return (roll, pitch, yaw)
