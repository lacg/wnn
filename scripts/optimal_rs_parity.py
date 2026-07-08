"""Parity: Rust AttitudeLqrRs/AttitudeMpcRs vs Python optimal.py teachers.

The Rust LQR uses the DISCRETE Riccati at dt (Python uses continuous CARE), so K
differs slightly — we verify the CONTROL OUTPUT agrees across random states (the
teacher is faithful if it commands the same motor PWMs). MPC compares the same way
(both solve the box-QP; Rust FISTA vs Python OSQP → statistical, not bit-exact).
"""
import numpy as np

from wnn.control.optimal import LQRController, MPCController
import ram_controller as rc


def rand_state(rng):
	# small random attitude + body rate near hover (the regime the teacher sees).
	rpy = rng.uniform(-np.radians(8), np.radians(8), size=3)
	cr, sr = np.cos(rpy[0] / 2), np.sin(rpy[0] / 2)
	cp, sp = np.cos(rpy[1] / 2), np.sin(rpy[1] / 2)
	cy, sy = np.cos(rpy[2] / 2), np.sin(rpy[2] / 2)
	q = [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
	     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]
	gyro = list(rng.uniform(-0.5, 0.5, size=3))
	return [float(x) for x in q], [float(x) for x in gyro], [0.0, 0.0, 0.0]


def compare(name, py_ctrl, rs_ctrl, n=500):
	rng = np.random.default_rng(12345)
	max_abs, sum_abs = 0.0, 0.0
	for _ in range(n):
		q, gyro, tgt = rand_state(rng)
		py_ctrl.reset(); rs_ctrl.reset()
		a = np.array(py_ctrl.step(q, gyro, tgt))
		b = np.array(rs_ctrl.step(q, gyro, tgt))
		d = float(np.max(np.abs(a - b)))
		max_abs = max(max_abs, d); sum_abs += d
	print(f"[{name}] over {n} states: max|Δpwm|={max_abs:.5f}  mean|Δpwm|={sum_abs/n:.5f}")
	return max_abs


def main():
	print("=== LQR gains (continuous vs discrete Riccati) ===")
	py = LQRController()
	rs = rc.AttitudeLqrRs()
	print("py K row0:", np.array2string(py.K[0], precision=3))
	print("rs K row0:", np.array2string(np.array(rs.gain()[0:6]), precision=3))
	lqr_max = compare("LQR", py, rs)

	print("\n=== MPC (OSQP vs FISTA box-QP) ===")
	pm = MPCController()
	rm = rc.AttitudeMpcRs()
	mpc_max = compare("MPC", pm, rm)

	print("\n=== VERDICT ===")
	print(f"LQR max|Δpwm| = {lqr_max:.5f}  ({'OK' if lqr_max < 0.02 else 'CHECK'})")
	print(f"MPC max|Δpwm| = {mpc_max:.5f}  ({'OK' if mpc_max < 0.03 else 'CHECK'})")


if __name__ == "__main__":
	main()
