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

SI ONLY, PAST THE BOUNDARY. The firmware's gains arrive in degrees, deg/s and int16
actuator counts. They are converted to SI EXACTLY ONCE, in `_SiGains.from_firmware`,
and nothing downstream ever sees a degree or a count: angles are rad, rates rad/s,
the rate loop's output is NEWTONS per motor. This is deliberate — mixed units inside a
control loop is how a sign or a factor hides for months (Luiz, 05/08/2026: "we load
whatever the world sends then we transform to SI then everything else uses SI").

The conversion is small and each line is forced by the source:
  - attitude kp [deg/s per deg] and kd [dimensionless] are RATIO gains — numerically
    unchanged in rad. ki [deg/s per deg*s] likewise. Only its i_limit is an angle*time,
    so that converts: rad = deg * pi/180.
  - rate kp/ki/kd are [counts per (deg/s)] etc: multiply by 180/pi to get per (rad/s),
    then by THRUST_MAX/65535 to turn counts into newtons.
  - the rate loop's outputLimit is int16 saturation, 32767 counts -> newtons.

MIXER NOTE. The firmware mixes X-config (all four rotors on both axes, roll/pitch
halved); `AttitudeSim::body_torque` mixes '+'-config (roll from motors 1/3, pitch from
0/2). We emit the '+' form the sim wants, and the geometry is reconciled in
`Airframe.arm_length` via `axis_arm_from_radius` (L = 2a = radius*sqrt(2)). Feeding the
raw published radius there under-models roll/pitch authority by 0.7071 — that was a real
bug, fixed 05/08/2026; see docs/disturbance_param_sources.md "MOTOR GEOMETRY".
"""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, atan2, pi, radians, sqrt
from typing import Tuple

from .airframe import Airframe, PidAxis, PidGains

# power_distribution_quadrotor.c: maxAllowedThrust = UINT16_MAX; roll/pitch halved.
_COUNTS_FULL_SCALE = 65535.0
# attitude_pid_controller.c: rate output passes saturateSignedInt16.
_INT16_SAT = 32767.0
_DEG_PER_RAD = 180.0 / pi


@dataclass(frozen=True)
class _SiGains:
	"""The firmware's gains, converted to SI once. Attitude: rad -> rad/s. Rate:
	rad/s -> NEWTONS per motor. `rate_output_limit_n` is int16 saturation in newtons."""

	attitude: tuple          # (roll, pitch, yaw) PidAxis, i_limit in rad*s
	rate: tuple              # (roll, pitch, yaw) PidAxis, gains in N per (rad/s)
	rate_output_limit_n: float

	@staticmethod
	def from_firmware(gains: PidGains, thrust_max: float) -> "_SiGains":
		"""THE ONLY PLACE foreign units exist. Everything after this is SI."""
		n_per_count = thrust_max / _COUNTS_FULL_SCALE
		att = tuple(
			PidAxis(kp=a.kp, ki=a.ki, kd=a.kd, i_limit=radians(a.i_limit))
			for a in gains.attitude
		)
		rate = tuple(
			PidAxis(
				kp=a.kp * _DEG_PER_RAD * n_per_count,
				ki=a.ki * _DEG_PER_RAD * n_per_count,
				kd=a.kd * _DEG_PER_RAD * n_per_count,
				i_limit=radians(a.i_limit),
			)
			for a in gains.rate
		)
		return _SiGains(att, rate, _INT16_SAT * n_per_count)


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
		self.si = _SiGains.from_firmware(gains, airframe.k_thrust)
		self.decimation = main_loop_hz // attitude_hz
		dt = 1.0 / attitude_hz
		# Attitude loop output is a rate setpoint in rad/s — the firmware applies no
		# outputLimit to it (attitude_pid_controller.c), so 0.0 = off.
		self.att = [_Pid(a, dt, 0.0) for a in self.si.attitude]
		self.rate = [_Pid(a, dt, self.si.rate_output_limit_n)
		             for a in self.si.rate]
		# Newtons per motor that hold hover. The sim has no translational state, so
		# this only sets the operating point the differential rides on — but it is the
		# airframe's real one, not a hardcoded 0.5 PWM.
		self.hover_thrust_n = airframe.mass * airframe.gravity / 4.0
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
		"""Angle (rad) -> rate setpoint (rad/s) -> per-motor force offset (N), per axis.

		All SI: the gains were converted once in `_SiGains.from_firmware`.
		"""
		rpy = _quat_to_euler(q)
		out = []
		for i in range(3):
			# Faithful to attitude_pid_controller.c: the attitude PID is handed the
			# ACTUAL angle as its measurement (so its D-term differentiates the angle
			# — this matters for yaw, whose kd is 0.35, and only for yaw), with the
			# error wrapped for yaw alone (pidUpdate's shouldWrap = true for yaw).
			is_yaw = i == 2
			rate_sp = self.att[i].update(target_rpy[i], rpy[i], is_yaw)
			out.append(self.rate[i].update(rate_sp, gyro[i], False))
		return (out[0], out[1], out[2])

	def _mix(self, axis_n: Tuple[float, float, float]) -> Tuple[float, ...]:
		"""Per-axis force offset (N) -> per-motor thrust (N) -> PWM, '+' motor order.

		Sim convention (controller.rs body_torque): M0 front(+x), M1 right(-y),
		M2 rear(-x), M3 left(+y); roll = L*(-t1+t3), pitch = L*(-t0+t2),
		yaw = k*(t0-t1+t2-t3). Firmware halves roll/pitch before mixing; yaw is not
		halved (power_distribution_quadrotor.c). thrust = k_thrust*pwm^2, so the
		inverse is pwm = sqrt(thrust/k_thrust).
		"""
		r = axis_n[0] / 2.0
		p = axis_n[1] / 2.0
		y = axis_n[2]
		h, kt = self.hover_thrust_n, self.airframe.k_thrust
		# +roll must raise the LEFT motor (M3) and drop the RIGHT (M1); +pitch must
		# raise the REAR (M2) and drop the FRONT (M0) — matching body_torque's signs.
		thrusts = (h - p + y, h - r - y, h + p + y, h + r - y)
		return tuple(sqrt(_clamp(t, 0.0, kt) / kt) for t in thrusts)


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
