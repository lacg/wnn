"""Quantify what the hand-tuned AttitudePIDConfig gains DO on each registered airframe.

The sim integrates tau = I*omega_dot only, and thrust ∝ pwm², so around the hover
operating point p the small-signal roll/pitch loop gain is

    G = d(omega_dot)/d(u) = 4 * arm * k_thrust * p / Ixx        [rad/s² per unit u]

(the 4*p factor is d(pwm²)/d(pwm) summed over the differential pair). The PID emits
u = kp*err - kd*rate (D on the gyro), so the closed-loop second-order form is

    theta'' + G*kd*theta' + G*kp*theta = 0
    omega_n = sqrt(G*kp)          zeta = kd*sqrt(G) / (2*sqrt(kp))

Yaw is driven through the drag coefficient instead of the arm:
    G_yaw = 4 * k_drag * k_thrust * p / Izz
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wnn.control.airframe import Airframe
from wnn.control.pid import AttitudePIDConfig

CFG = AttitudePIDConfig()
HOVER = CFG.hover_throttle  # 0.5, hardcoded — the operating point the loop is flown at

# The plant pid.py's docstring says the gains were hand-tuned against.
LEGACY = dict(arm=0.075, kt=2.4, kd_coef=0.05, I=(0.0023, 0.0023, 0.0046))


def loop(gain, kp, kd):
	"""(omega_n, zeta) for theta'' + G*kd*theta' + G*kp*theta = 0."""
	wn = (gain * kp) ** 0.5
	zeta = kd * gain**0.5 / (2.0 * kp**0.5)
	return wn, zeta


def report(name, arm, kt, kd_coef, inertia):
	g_rp = 4.0 * arm * kt * HOVER / inertia[0]
	g_yaw = 4.0 * kd_coef * kt * HOVER / inertia[2]
	wn_rp, z_rp = loop(g_rp, CFG.roll.kp, CFG.roll.kd)
	wn_y, z_yaw = loop(g_yaw, CFG.yaw.kp, CFG.yaw.kd)
	# Error at which the P-term alone saturates the axis-authority clamp.
	sat_err = CFG.max_axis_authority / CFG.roll.kp
	# Rate at which the D-term alone saturates it.
	sat_rate = CFG.max_axis_authority / CFG.roll.kd
	print(
		f"{name:16s} G_rp={g_rp:8.1f}  roll/pitch: wn={wn_rp:6.2f} rad/s zeta={z_rp:5.2f}"
		f"   |  G_yaw={g_yaw:7.1f}  yaw: wn={wn_y:5.2f} zeta={z_yaw:5.2f}"
	)
	return g_rp, wn_rp, z_rp, sat_err, sat_rate


print(f"PID gains (hand-tuned, NOT airframe-derived): roll/pitch kp={CFG.roll.kp} "
      f"ki={CFG.roll.ki} kd={CFG.roll.kd} | yaw kp={CFG.yaw.kp} ki={CFG.yaw.ki} "
      f"kd={CFG.yaw.kd}")
print(f"hover_throttle={HOVER} (hardcoded), max_axis_authority={CFG.max_axis_authority}\n")

base = report("LEGACY(tuned-on)", LEGACY["arm"], LEGACY["kt"], LEGACY["kd_coef"],
              LEGACY["I"])
print()
for n in Airframe.names():
	af = Airframe.preset(n)
	cur = report(n, af.arm_length, af.k_thrust, af.k_drag, af.inertia)
	print(f"{'':16s}   -> roll/pitch loop gain is {cur[0] / base[0]:.2f}x the plant the "
	      f"gains were tuned on; zeta {base[2]:.2f} -> {cur[2]:.2f}, "
	      f"wn {base[1]:.1f} -> {cur[1]:.1f} rad/s")
	print(f"{'':16s}      physical hover PWM for this airframe = {af.hover_pwm:.3f} "
	      f"(loop is flown at {HOVER})")

g, wn, z, sat_err, sat_rate = base
print(f"\nP-term saturates the +-{CFG.max_axis_authority} axis clamp at "
      f"err={sat_err:.3f} rad = {sat_err * 57.2958:.1f} deg (airframe-independent).")
print(f"D-term saturates it at rate={sat_rate:.3f} rad/s = "
      f"{sat_rate * 57.2958:.1f} deg/s (airframe-independent).")
# --- What is scale-invariant and what is not -------------------------------
# The I-term's trim authority in TORQUE is
#     tau_trim = ki * i_clamp * G * Ixx = ki * i_clamp * 4*arm*k_thrust*hover
# Ixx CANCELS. The L4C motor asymmetry induces a persistent torque of roughly
#     tau_asym ~ asym_frac * arm*k_thrust*hover^2
# which also scales with arm*k_thrust. So their RATIO is airframe-independent:
# ki is scale-invariant in normalized-PWM terms and is NOT part of the defect.
# What is NOT invariant is anything that touches inertia — i.e. the transient
# shaping (bandwidth omega_n and damping zeta), which is where the mismatch lives.
ASYM = 0.20  # L4C motor_asym_mag
print("\n--- invariance check: which gains actually go stale ---")
print(f"{'airframe':16s} {'tau_trim (N.m)':>15s} {'tau_asym (N.m)':>15s} {'ratio':>8s}")
rows = [("LEGACY(tuned-on)", LEGACY["arm"], LEGACY["kt"])] + [
	(n, Airframe.preset(n).arm_length, Airframe.preset(n).k_thrust)
	for n in Airframe.names()
]
for name, arm, kt in rows:
	tau_trim = CFG.roll.ki * CFG.roll.i_clamp * 4.0 * arm * kt * HOVER
	tau_asym = ASYM * arm * kt * HOVER**2
	print(f"{name:16s} {tau_trim:15.3e} {tau_asym:15.3e} {tau_trim / tau_asym:8.2f}")
print("\nRatio constant across airframes => ki/i_clamp/authority are SCALE-INVARIANT")
print("(they track arm*k_thrust, as does the disturbance). Only the inertia-coupled")
print("transient shaping (omega_n, zeta) goes stale. That is the whole defect.")
