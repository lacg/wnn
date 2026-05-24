"""Classical attitude PID controller.

Serves two roles:
  1. The teacher for WNN BPTT/EDRA training (DAGGER-style imitation
     learning where the WNN learns to match PID's per-step action).
  2. The baseline that the WNN must beat or match in the paper.

Design choices:
  - Per-axis (roll, pitch, yaw) independent PID loops. Drone attitude
    is approximately decoupled at small angles, so this works well in
    practice and matches what most commercial flight controllers do.
  - I-term has anti-windup via clamp; D-term operates on the gyro rate
    (not the differentiated error) to avoid derivative-of-target spikes.
  - Outputs base hover + roll/pitch/yaw mixing → 4 motor PWMs in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import asin, atan2, pi
from typing import Tuple


# Quaternion → euler angles helpers (right-hand frame, body z-up). Used to
# convert (q, target_q) error into per-axis (roll, pitch, yaw) terms.
def _quat_to_euler(q: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
	"""Body-to-world unit quaternion → (roll, pitch, yaw) in radians."""
	w, x, y, z = q
	# Roll (x-axis rotation)
	sinr_cosp = 2.0 * (w * x + y * z)
	cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
	roll = atan2(sinr_cosp, cosr_cosp)
	# Pitch (y-axis rotation)
	sinp = 2.0 * (w * y - z * x)
	if sinp >= 1.0:
		pitch = pi / 2.0
	elif sinp <= -1.0:
		pitch = -pi / 2.0
	else:
		pitch = asin(sinp)
	# Yaw (z-axis rotation)
	siny_cosp = 2.0 * (w * z + x * y)
	cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
	yaw = atan2(siny_cosp, cosy_cosp)
	return (roll, pitch, yaw)


def _wrap_angle(a: float) -> float:
	"""Wrap radians into (-pi, pi]."""
	while a > pi:
		a -= 2.0 * pi
	while a <= -pi:
		a += 2.0 * pi
	return a


@dataclass
class PIDGains:
	"""Per-axis PID gains. Roll/pitch typically use the same gains by
	symmetry; yaw is decoupled (no gravity restoring force) so usually
	weaker."""
	kp: float
	ki: float
	kd: float
	i_clamp: float = 0.5  # anti-windup limit on integral term


@dataclass
class AttitudePIDConfig:
	"""Gains + mixing config. Tuned by hand against AttitudeSim's default
	physics (arm_length=0.075, k_thrust=2.4, inertia diag [0.0023, 0.0023,
	0.0046]). The plant has a very high control authority — small kp values
	work well; large kp drives oscillation.

	The D-term operates on the gyro directly (not de/dt) which is critical
	for stability: it provides damping proportional to angular velocity.
	Without strong D, the high control authority causes wild oscillation."""
	roll: PIDGains = field(default_factory=lambda: PIDGains(kp=1.2, ki=0.05, kd=0.30))
	pitch: PIDGains = field(default_factory=lambda: PIDGains(kp=1.2, ki=0.05, kd=0.30))
	yaw: PIDGains = field(default_factory=lambda: PIDGains(kp=0.6, ki=0.02, kd=0.20))

	# Base throttle that maintains hover under default sim physics
	# (k_thrust=2.4, mass implied by inertia, gravity 9.81). Per-motor
	# thrust at hover = m*g/4. With k_thrust = 2.4 N per pwm² and
	# implied mass ~0.25 kg → per-motor thrust 0.613 N → pwm² = 0.255
	# → pwm ≈ 0.505. Round to 0.5 as the neutral hover throttle.
	hover_throttle: float = 0.5

	# Authority limits: how much each axis can swing the per-motor PWM
	# away from base hover. Keeps PID outputs in safe [0, 1] range.
	max_axis_authority: float = 0.4

	# Integration timestep (s). Must match the sim dt for I-term to be
	# accurate in physical units; 1 ms = 1 kHz control rate.
	dt: float = 0.001


class AttitudePID:
	"""3-axis (roll, pitch, yaw) attitude PID over body-frame gyro and
	an attitude error derived from the current quaternion vs the target
	euler attitude.

	State (mutable across step()):
	  - integral_roll, integral_pitch, integral_yaw: I-term accumulators.

	Usage:
	    pid = AttitudePID(AttitudePIDConfig())
	    pid.reset()
	    while sim.time < T:
	        gyro, accel = sim.read_imu()
	        # estimate q from the sim — for paper #1 we read it directly
	        q = sim.quaternion
	        pwm = pid.step(q, gyro, target_rpy=(0.0, 0.0, 0.0))
	        sim.step(pwm)
	"""

	def __init__(self, config: AttitudePIDConfig | None = None):
		self.cfg = config if config is not None else AttitudePIDConfig()
		self.integral_roll: float = 0.0
		self.integral_pitch: float = 0.0
		self.integral_yaw: float = 0.0

	def reset(self) -> None:
		self.integral_roll = 0.0
		self.integral_pitch = 0.0
		self.integral_yaw = 0.0

	def step(
		self,
		q: Tuple[float, float, float, float],
		gyro: Tuple[float, float, float],
		target_rpy: Tuple[float, float, float],
	) -> Tuple[float, float, float, float]:
		"""One PID cycle. Returns 4 motor PWMs in [0, 1].

		Args:
			q:           current body-to-world unit quaternion (w, x, y, z) from sim.
			gyro:        body-frame angular velocity (rad/s) from IMU.
			target_rpy:  target (roll, pitch, yaw) attitude in radians.

		Returns:
			(pwm_front, pwm_right, pwm_rear, pwm_left) for the '+' quadcopter
			layout matching AttitudeSim's motor convention.
		"""
		roll, pitch, yaw = _quat_to_euler(q)
		t_roll, t_pitch, t_yaw = target_rpy

		# Per-axis error (wrapped to handle yaw wraparound at ±pi).
		err_roll = _wrap_angle(t_roll - roll)
		err_pitch = _wrap_angle(t_pitch - pitch)
		err_yaw = _wrap_angle(t_yaw - yaw)

		# I-term update with anti-windup clamp.
		self.integral_roll = _clamp(
			self.integral_roll + err_roll * self.cfg.dt,
			-self.cfg.roll.i_clamp, self.cfg.roll.i_clamp,
		)
		self.integral_pitch = _clamp(
			self.integral_pitch + err_pitch * self.cfg.dt,
			-self.cfg.pitch.i_clamp, self.cfg.pitch.i_clamp,
		)
		self.integral_yaw = _clamp(
			self.integral_yaw + err_yaw * self.cfg.dt,
			-self.cfg.yaw.i_clamp, self.cfg.yaw.i_clamp,
		)

		# D-term operates on gyro directly (proportional to -rate, since
		# we want to dampen the rotation). This avoids derivative-of-target
		# spikes that hurt closed-loop stability under abrupt setpoint
		# changes.
		gx, gy, gz = gyro

		# PID outputs per axis.
		u_roll = (
			self.cfg.roll.kp * err_roll
			+ self.cfg.roll.ki * self.integral_roll
			- self.cfg.roll.kd * gx
		)
		u_pitch = (
			self.cfg.pitch.kp * err_pitch
			+ self.cfg.pitch.ki * self.integral_pitch
			- self.cfg.pitch.kd * gy
		)
		u_yaw = (
			self.cfg.yaw.kp * err_yaw
			+ self.cfg.yaw.ki * self.integral_yaw
			- self.cfg.yaw.kd * gz
		)

		# Clamp axis authority before mixing to keep per-motor PWM in range.
		a = self.cfg.max_axis_authority
		u_roll = _clamp(u_roll, -a, a)
		u_pitch = _clamp(u_pitch, -a, a)
		u_yaw = _clamp(u_yaw, -a, a)

		# Mix to '+' quadcopter PWMs.
		# Motor layout (from controller.rs):
		#   M0 = front  ( +x): roll-neutral, pitch-down command increases M0 thrust
		#   M1 = right  ( -y): roll-left command increases M1 thrust
		#   M2 = rear   ( -x): pitch-up command increases M2 thrust
		#   M3 = left   ( +y): roll-right command increases M3 thrust
		# Yaw torque convention (from body_torque): +k*(T0 - T1 + T2 - T3).
		# So increasing yaw command → increase M0+M2, decrease M1+M3.
		#
		# But pid u_roll is defined as kp*(t_roll - roll), so positive u_roll
		# means "need more positive roll torque" which means... torque about
		# +x. body_torque produces +x torque from T3-T1 (left minus right).
		# So u_roll > 0 → boost M3 (left), reduce M1 (right).
		base = self.cfg.hover_throttle
		pwm = (
			_clamp(base - u_pitch + u_yaw, 0.0, 1.0),  # M0 front
			_clamp(base - u_roll  - u_yaw, 0.0, 1.0),  # M1 right
			_clamp(base + u_pitch + u_yaw, 0.0, 1.0),  # M2 rear
			_clamp(base + u_roll  - u_yaw, 0.0, 1.0),  # M3 left
		)
		return pwm


def _clamp(v: float, lo: float, hi: float) -> float:
	if v < lo:
		return lo
	if v > hi:
		return hi
	return v


__all__ = ["AttitudePID", "AttitudePIDConfig", "PIDGains"]
