"""Is the WNN's ~1.2° floor the OUTPUT ALPHABET rather than learning?

THE QUESTION. Three teachers spanning 0.69-0.93 deg of classical quality (MPCOF 0.69,
LQI 0.81, LQR 0.93) all produced students at 1.2-1.3 deg on cf21_brushless/L4C. A floor
that ignores teacher quality is a substrate property, not a teaching property.

THE HYPOTHESIS. The BINARY antagonist decode is
      pwm = 0.5 + (SumE - SumI) / levels
so with levels=16 a motor's ENTIRE vocabulary is 17 values in steps of 1/16 = 0.0625.
Two consequences on this airframe:
  (1) one step at hover is worth ~1.23e-3 N.m of roll torque = ~41 rad/s^2, while the
      L4C motor-asymmetry trim it must hold is ~1.36e-3 N.m. The smallest available
      action is ~0.9x the entire correction needed, so the loop cannot settle finely —
      it can only alternate one-step-under / one-step-over.
  (2) the decode centres on 0.5 but this airframe hovers at 0.6942, i.e. 3.1 steps away,
      so ~4 of the 8 one-sided levels are spent just reaching hover. Hover itself is not
      representable (nearest 0.6875).

THE TEST, and why it is decisive. Take a controller that ALREADY achieves the teacher
number, quantize ONLY its output through the WNN's alphabet, and re-fly it. No learning,
no GA, no cells — the sole difference is the output vocabulary. If the quantized teacher
degrades to ~1.2 deg, the ceiling is PROVEN to be the decode. If it stays near its
un-quantized error, the decode is innocent and the loss is in learning/generalization,
and we should stop blaming resolution.

Teacher used here is the firmware PID cascade (wnn.control.pid_firmware) because it is
the one teacher with a pure-Python implementation; the LQ/MPC family lives in Rust. That
is fine for this question: the claim under test is about the ALPHABET, and the alphabet
does not care which controller feeds it. The absolute numbers here are from a simplified
Euler harness (the production scorer is Rust/RK4 with disturbances), so read the DELTAS
between arms, never the absolute degrees.

Arms:
  full            un-quantized baseline for this harness
  L=16 c=0.5      production today
  L=16 c=hover    lever A — re-centre the decode on the airframe's hover (FREE: no
                  extra neurons, recovers the ~4 wasted levels)
  L=32 c=0.5      lever B — finer step, costs 2x output neurons (fits the current
                  --max-output-neurons 128)
  L=32 c=hover    both
  L=64 c=hover    both, needs the neuron cap raised

Usage:  python3 scripts/decode_quantization_ceiling.py [--tilt 5] [--steps 3000]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wnn.control.airframe import Airframe
from wnn.control.pid_firmware import AttitudePidFirmware


def quantize(pwm: float, levels: int, centre: float) -> float:
	"""Round `pwm` onto the antagonist decode's alphabet.

	The decode emits centre + k/levels for integer k in [-levels/2, +levels/2] (E and I
	banks of levels/2 neurons each), clamped to [0, 1). Nearest-neighbour rounding is the
	BEST case for the alphabet — a real WNN also has to *learn* which level to fire, so
	any degradation measured here is a LOWER BOUND on the true cost.
	"""
	half = levels // 2
	k = round((pwm - centre) * levels)
	k = max(-half, min(half, k))
	return min(1.0, max(0.0, centre + k / levels))


def rollout(af: Airframe, levels: int | None, centre: float, tilt_deg: float,
            steps: int, dt: float = 1e-3) -> dict:
	"""Fly the firmware cascade, optionally through the decode alphabet.

	Returns mean |roll| error, the steady-state tail mean (last 20%, matching the
	production `steady` metric), and how much the quantizer actually moved the command.
	"""
	kt, L, kd = af.k_thrust, af.arm_length, af.k_drag
	I = af.inertia
	pid = AttitudePidFirmware(af, af.gains())
	pid.reset()
	a = math.radians(tilt_deg) / 2.0
	q = [math.cos(a), math.sin(a), 0.0, 0.0]
	w = [0.0, 0.0, 0.0]
	errs: list[float] = []
	quant_deltas: list[float] = []
	for _ in range(steps):
		pwm = list(pid.step(tuple(q), tuple(w), (0.0, 0.0, 0.0)))
		if levels is not None:
			raw = list(pwm)
			pwm = [quantize(p, levels, centre) for p in pwm]
			quant_deltas.append(max(abs(a - b) for a, b in zip(raw, pwm)))
		t = [kt * p * p for p in pwm]
		tau = (L * (-t[1] + t[3]), L * (-t[0] + t[2]), kd * (t[0] - t[1] + t[2] - t[3]))
		w = [w[i] + tau[i] / I[i] * dt for i in range(3)]
		wx, wy, wz = w
		qw, qx, qy, qz = q
		dq = (-0.5 * (qx * wx + qy * wy + qz * wz),
		      0.5 * (qw * wx + qy * wz - qz * wy),
		      0.5 * (qw * wy - qx * wz + qz * wx),
		      0.5 * (qw * wz + qx * wy - qy * wx))
		q = [q[i] + dq[i] * dt for i in range(4)]
		n = math.sqrt(sum(v * v for v in q))
		q = [v / n for v in q]
		errs.append(abs(math.degrees(math.atan2(
			2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)))))
	tail = errs[int(len(errs) * 0.8):]
	return {
		"mean": sum(errs) / len(errs),
		"steady": sum(tail) / len(tail),
		"final": errs[-1],
		"quant_delta": (sum(quant_deltas) / len(quant_deltas)) if quant_deltas else 0.0,
	}


def alphabet_report(af: Airframe, levels: int, centre: float) -> None:
	"""What the alphabet can express, and how it compares to the trim it must hold."""
	step = 1.0 / levels
	hover = af.hover_pwm
	tau_step = af.arm_length * 2 * af.k_thrust * hover * step
	tau_trim = 0.20 * af.arm_length * af.k_thrust * hover ** 2   # L4C motor_asym 0.20
	off = abs(quantize(hover, levels, centre) - hover)
	print(f"  levels={levels:<4d} centre={centre:.4f}  step={step:.4f}  "
	      f"hover repr err={off:.4f}  tau_step={tau_step:.3e}  "
	      f"tau_step/trim={tau_step / tau_trim:.2f}  out_neurons={4 * levels}")


def main() -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("--airframe", default="cf21_brushless")
	ap.add_argument("--tilt", type=float, default=5.0)
	ap.add_argument("--steps", type=int, default=3000)
	a = ap.parse_args()
	af = Airframe.preset(a.airframe)
	hover = af.hover_pwm

	print(f"airframe {af.name}: hover_pwm={hover:.4f}  arm={af.arm_length:.6f}  "
	      f"k_thrust={af.k_thrust}  Ixx={af.inertia[0]:.4e}")
	print("\nALPHABET (what one action step is worth vs the trim it must hold):")
	for lv, c in ((16, 0.5), (16, hover), (32, 0.5), (32, hover), (64, hover)):
		alphabet_report(af, lv, c)

	print(f"\nCLOSED LOOP (tilt {a.tilt}deg, {a.steps} steps, firmware cascade teacher)")
	print("read the DELTAS, not the absolute degrees — simplified Euler harness\n")
	base = rollout(af, None, 0.5, a.tilt, a.steps)
	print(f"  {'full (un-quantized)':<24s} mean={base['mean']:.3f}  "
	      f"steady={base['steady']:.3f}  final={base['final']:.4f}")
	arms = ((16, 0.5, "L=16 c=0.5  [TODAY]"), (16, hover, "L=16 c=hover [lever A]"),
	        (32, 0.5, "L=32 c=0.5   [lever B]"), (32, hover, "L=32 c=hover [A+B]"),
	        (64, hover, "L=64 c=hover [A+B++]"))
	for lv, c, label in arms:
		r = rollout(af, lv, c, a.tilt, a.steps)
		d_mean = r["mean"] - base["mean"]
		d_steady = r["steady"] - base["steady"]
		print(f"  {label:<24s} mean={r['mean']:.3f} ({d_mean:+.3f})  "
		      f"steady={r['steady']:.3f} ({d_steady:+.3f})  "
		      f"final={r['final']:.4f}  avg|quant|={r['quant_delta']:.4f}")

	print("\nINTERPRETATION GUIDE (state which one the numbers support; do not assume):")
	print("  * L=16 c=0.5 degrades a LOT vs full  -> the alphabet IS the ceiling.")
	print("  * c=hover recovers most of it        -> the 0.5 centre is the bug, and the")
	print("                                          fix costs zero extra neurons.")
	print("  * only L=32/64 recovers it           -> genuine resolution limit; pay the")
	print("                                          neurons (L=32 fits the 128 cap).")
	print("  * L=16 c=0.5 barely degrades         -> HYPOTHESIS REFUTED. The decode is")
	print("                                          innocent; the 1.2deg floor is in")
	print("                                          learning/generalization, and we")
	print("                                          must stop blaming resolution.")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
