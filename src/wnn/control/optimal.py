"""Optimal-control teachers for residual-DAGGER: LQR and MPC.

These replace PID+ as the DAGGER *expert* — the WNN learns clamp(expert − baseline).
To beat a hand-tuned PID+ we need a stronger teacher; LQR (optimal linear feedback)
and MPC (constrained receding-horizon optimization) are the classic candidates.

The plant model is FREE: AttitudeSim's dynamics are known, so near hover the
attitude channel linearizes to a 6-state double-integrator-per-axis
    x = [roll, pitch, yaw,  p, q, r]      (attitude error + body rates; target=0)
    ẋ = A x + B u,   A = [[0₃, I₃],[0₃, 0₃]],   B = [[0₃],[diag(b_roll,b_pitch,b_yaw)]]
where u ∈ [−authority, +authority] is the SAME normalized per-axis control the PID
mixing consumes, and the b's map normalized control → angular acceleration. We
CALIBRATE the b's by stepping the sim once per axis (robust to the sim's exact
drag/mixing) rather than deriving them analytically.

Both controllers expose the AttitudePID interface — step(q, gyro, target_rpy) → 4
motor PWMs, reset() — so they drop into make_pid_action_fn and the dagger expert
slot unchanged. They are used ONLY to generate DAGGER labels during training (not
the deployed hot path — the WNN is the deployed artifact), so Python is correct here.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from wnn.control.pid import _quat_to_euler


def _clip(v: float, lo: float, hi: float) -> float:
	return lo if v < lo else hi if v > hi else v


def mix_to_motors(hover: float, u_roll: float, u_pitch: float, u_yaw: float) -> list[float]:
	"""'+' quad mixing — bit-identical to AttitudePID / AttitudePidRs. Motors clamp [0,1]."""
	return [
		_clip(hover - u_pitch + u_yaw, 0.0, 1.0),  # M0 front
		_clip(hover - u_roll  - u_yaw, 0.0, 1.0),  # M1 right
		_clip(hover + u_pitch + u_yaw, 0.0, 1.0),  # M2 rear
		_clip(hover + u_roll  - u_yaw, 0.0, 1.0),  # M3 left
	]


def calibrate_control_gains(hover: float = 0.5, dt: float = 0.001, u_probe: float = 0.05) -> np.ndarray:
	"""Map normalized per-axis control → angular acceleration, by stepping a clean
	AttitudeSim once per axis from rest and reading the induced body rate. Returns
	[b_roll, b_pitch, b_yaw] (rad/s² per unit control), with the SIGN the sim uses."""
	from wnn.control._accel import AttitudeSim
	b = np.zeros(3)
	for axis in range(3):
		sim = AttitudeSim()
		sim.reset(q=[1.0, 0.0, 0.0, 0.0], omega=[0.0, 0.0, 0.0])
		u = [0.0, 0.0, 0.0]; u[axis] = u_probe
		sim.step(mix_to_motors(hover, u[0], u[1], u[2]))
		omega = sim.angular_velocity          # [p, q, r] after one step from rest
		b[axis] = (omega[axis] / dt) / u_probe  # ω̇ / u
	return b


def attitude_linear_model(dt: float = 0.001, hover: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Continuous (A, B) for the 6-state attitude double-integrator, plus the
	calibrated per-axis gains b. A: anglė=rate, ratė=b·u."""
	b = calibrate_control_gains(hover=hover, dt=dt)
	A = np.zeros((6, 6))
	A[0, 3] = A[1, 4] = A[2, 5] = 1.0      # anglė = rate
	B = np.zeros((6, 3))
	B[3, 0], B[4, 1], B[5, 2] = b[0], b[1], b[2]  # ratė = b·u
	return A, B, b


class LQRController:
	"""Optimal linear feedback u = clamp(−Kx, ±authority), K from the continuous
	Riccati equation. Iso-authority with PID+ via the same ±0.4 clamp. AttitudePID
	interface (step/reset). Stateless (no integral) — LQR needs none, the model does."""

	def __init__(self, q_att: float = 12.0, q_rate: float = 1.0, r_ctrl: float = 1.0,
	             hover: float = 0.5, authority: float = 0.4, dt: float = 0.001):
		self.hover, self.authority = hover, authority
		A, B, self.b = attitude_linear_model(dt=dt, hover=hover)
		Q = np.diag([q_att, q_att, q_att, q_rate, q_rate, q_rate])
		R = np.diag([r_ctrl, r_ctrl, r_ctrl])
		P = sla.solve_continuous_are(A, B, Q, R)
		self.K = np.linalg.solve(R, B.T @ P)   # 3×6

	def reset(self) -> None:
		pass   # memoryless

	def step(self, q, gyro, target_rpy) -> list[float]:
		roll, pitch, yaw = _quat_to_euler(q)
		tr, tp, ty = target_rpy
		x = np.array([roll - tr, pitch - tp, yaw - ty, gyro[0], gyro[1], gyro[2]])
		u = -self.K @ x                                    # [u_roll, u_pitch, u_yaw]
		a = self.authority
		return mix_to_motors(self.hover,
			_clip(float(u[0]), -a, a), _clip(float(u[1]), -a, a), _clip(float(u[2]), -a, a))


class MPCController:
	"""Constrained receding-horizon MPC. Each step solves a QP: minimize the
	horizon cost Σ xₖᵀQxₖ + uₖᵀRuₖ (+ terminal Qf) s.t. the linear model and the
	hard authority box uₖ ∈ [−authority, +authority] (iso-authority with PID+/LQR),
	apply u₀, recede. Prediction runs at a coarser dt_mpc for a meaningful horizon;
	the controller is re-solved every sim step. AttitudePID interface.

	NOTE: a QP per call — DAGGER training with this teacher is SLOW (the whole point
	is that the WNN then imitates it at RAM-lookup cost). The cvxpy problem is built
	ONCE and re-solved via a parameter (OSQP, warm-started)."""

	def __init__(self, horizon: int = 25, dt_mpc: float = 0.001,
	             q_att: float = 12.0, q_rate: float = 1.0, r_ctrl: float = 1.0,
	             hover: float = 0.5, authority: float = 0.4, dt: float = 0.001):
		import cvxpy as cp
		self.cp = cp
		self.hover, self.authority, self.N = hover, authority, horizon
		A, B, self.b = attitude_linear_model(dt=dt, hover=hover)
		Ad = np.eye(6) + A * dt_mpc          # forward-Euler discretization at dt_mpc
		Bd = B * dt_mpc
		Q = np.diag([q_att, q_att, q_att, q_rate, q_rate, q_rate])
		R = np.diag([r_ctrl, r_ctrl, r_ctrl])
		# Terminal cost = discrete-LQR cost-to-go (solve_discrete_are on the SAME
		# discretized model). This is what makes a short-horizon MPC approximate the
		# infinite-horizon optimum instead of being myopic — without it MPC is worse
		# than LQR. Symmetrize for cvxpy PSD acceptance.
		Qf = sla.solve_discrete_are(Ad, Bd, Q, R)
		Qf = 0.5 * (Qf + Qf.T)
		# Build the parametrized QP once; step() just sets x0 and re-solves.
		self._x0 = cp.Parameter(6)
		X = cp.Variable((6, horizon + 1))
		U = cp.Variable((3, horizon))
		cost, cons = 0, [X[:, 0] == self._x0]
		for k in range(horizon):
			cost += cp.quad_form(X[:, k], Q) + cp.quad_form(U[:, k], R)
			cons += [X[:, k + 1] == Ad @ X[:, k] + Bd @ U[:, k],
			         U[:, k] <= authority, U[:, k] >= -authority]
		cost += cp.quad_form(X[:, horizon], cp.psd_wrap(Qf))   # LQR cost-to-go terminal
		self._prob = cp.Problem(cp.Minimize(cost), cons)
		self._U = U

	def reset(self) -> None:
		pass

	def step(self, q, gyro, target_rpy) -> list[float]:
		roll, pitch, yaw = _quat_to_euler(q)
		tr, tp, ty = target_rpy
		self._x0.value = np.array([roll - tr, pitch - tp, yaw - ty, gyro[0], gyro[1], gyro[2]])
		self._prob.solve(solver=self.cp.OSQP, warm_start=True, verbose=False)
		u = self._U.value
		a = self.authority
		if u is None:   # QP failed to solve → hover (safe fallback)
			return mix_to_motors(self.hover, 0.0, 0.0, 0.0)
		return mix_to_motors(self.hover,
			_clip(float(u[0, 0]), -a, a), _clip(float(u[1, 0]), -a, a), _clip(float(u[2, 0]), -a, a))
