"""Mahony complementary attitude estimator — the PYTHON REFERENCE.

The estimator-fed teacher rule (13/08/2026): comparison-row teachers never read
the true quaternion; they read THIS filter's output, computed from the same
noisy IMU stream the WNN reads. Training (DAgger) teachers stay oracle.

This file is the reference implementation, same pattern as pid_firmware.py:
the Rust twin (ram_controller estimator.rs) must match it step-for-step via a
golden-trajectory parity test. Change one, change both.

Algorithm (Mahony, Nonlinear Complementary Filters on SO(3), IEEE TAC 2008 —
the MahonyAHRS formulation): predict the body-frame "up" direction from the
current estimate, take the cross-product error against the accelerometer's
measured "up" (specific force ≈ support ≈ −gravity in body frame at hover),
and feed kp·e (+ ki·∫e for gyro-bias tracking) back into the gyro integration.

Conventions match AttitudeSim: quaternion (w, x, y, z), body→world; the
accelerometer at rest with q = identity reads (0, 0, +g) — the support force.
Yaw is unobservable from gyro+accel (no magnetometer on cf21_brushless), so
yaw is gyro-dead-reckoned from the warm-start value — the same anchor the WNN
gets at reset.
"""
from __future__ import annotations

import math


class MahonyEstimator:
	"""One estimator instance per vehicle/episode. reset() then update() per step."""

	def __init__(self, dt: float, kp: float = 2.0, ki: float = 0.1):
		if dt <= 0.0:
			raise ValueError(f"dt must be > 0, got {dt}")
		self.dt = float(dt)
		self.kp = float(kp)
		self.ki = float(ki)
		self.q = [1.0, 0.0, 0.0, 0.0]
		self.integral = [0.0, 0.0, 0.0]

	def reset(self, q0=None) -> None:
		"""Warm-start from q0 (the converged-filter assumption: a firmware filter
		has been running since before takeoff, so at episode start it tracks the
		true attitude — disclosed). None = identity."""
		q = q0 if q0 is not None else (1.0, 0.0, 0.0, 0.0)
		n = math.sqrt(sum(float(v) * float(v) for v in q)) or 1.0
		self.q = [float(v) / n for v in q]
		self.integral = [0.0, 0.0, 0.0]

	def update(self, gyro, accel) -> list[float]:
		"""One 1 kHz fusion step: (measured gyro rad/s, measured accel m/s²) →
		quaternion estimate (w, x, y, z). Pure function of the sensor stream —
		no true state is ever read."""
		w, x, y, z = self.q
		gx, gy, gz = (float(v) for v in gyro)
		ax, ay, az = (float(v) for v in accel)

		norm = math.sqrt(ax * ax + ay * ay + az * az)
		if norm > 1e-9:
			ax, ay, az = ax / norm, ay / norm, az / norm
			# Predicted body-frame "up" = world +z rotated into the body frame
			# (third row of R(q)ᵀ… — the MahonyAHRS v-vector).
			vx = 2.0 * (x * z - w * y)
			vy = 2.0 * (w * x + y * z)
			vz = w * w - x * x - y * y + z * z
			# Error = measured × predicted; zero when they agree.
			ex = ay * vz - az * vy
			ey = az * vx - ax * vz
			ez = ax * vy - ay * vx
			if self.ki > 0.0:
				self.integral[0] += self.ki * ex * self.dt
				self.integral[1] += self.ki * ey * self.dt
				self.integral[2] += self.ki * ez * self.dt
			gx += self.kp * ex + self.integral[0]
			gy += self.kp * ey + self.integral[1]
			gz += self.kp * ez + self.integral[2]

		# q̇ = ½ q ⊗ (0, ω_corrected); first-order step, then renormalize.
		half_dt = 0.5 * self.dt
		dw = (-x * gx - y * gy - z * gz) * half_dt
		dx = (w * gx + y * gz - z * gy) * half_dt
		dy = (w * gy - x * gz + z * gx) * half_dt
		dz = (w * gz + x * gy - y * gx) * half_dt
		w, x, y, z = w + dw, x + dx, y + dy, z + dz
		n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
		self.q = [w / n, x / n, y / n, z / n]
		return list(self.q)
