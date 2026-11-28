"""CTRL-83 stress panel: THE pre-registered grid (docs/ctrl83_stress_panel_prereg.md §2).

ONE source for every cell. A cell is a pure function nominal EpisodeConfig -> stressed
EpisodeConfig; it changes ONE plant parameter, never the controller. The controller is
frozen: the published genome AND its thermometer thresholds, which are always fitted on
the NOMINAL config (score_cell.py), so a stressed cell never re-quantizes the inputs.

The stressed config is handed to BOTH scorers: ControllerEvaluator.score_genomes (WNN)
and classical_baseline.teacher_metrics (rivals). score_cell.py gates that the plant
kwargs the two Rust calls receive are identical (plant_identity_gate).

Do not add, drop or re-level a cell after the first scored result: the prereg fixes the
grid, the primary cells and the read (§4).
"""
import dataclasses as dc

# Training values of the adopted base (l2b, argv of every *_l2b_full_s{1..5}.out):
#   --disturbance L4C  -> motor_asym_mag 0.20, torque_scale_jitter 0.20, ADIS16448 noise
#   --mass-jitter default 0.15, --motor-lag-s 0.20 (2% settling time T; tau = T/4)
NOMINAL = dict(wear=0.20, mass=0.15, lag_T=0.20, noise=1.0)


def _wear(mag):
	def f(ec):
		return dc.replace(ec, disturbance=dc.replace(ec.disturbance, motor_asym_mag=float(mag)))
	return f


def _mass(jit):
	def f(ec):
		return dc.replace(ec, mass_jitter=float(jit))
	return f


def _lag(T):
	def f(ec):
		if float(getattr(ec, "motor_lag_down_s", 0.0)) != 0.0 or float(getattr(ec, "pwm_min", 0.0)) != 0.0:
			raise SystemExit("ctrl83: lag cell on an asymmetric/floored actuator — not the pre-registered plant")
		return dc.replace(ec, motor_lag_s=float(T))
	return f


def _noise(k):
	def f(ec):
		d = ec.disturbance
		return dc.replace(ec, disturbance=dc.replace(
			d, gyro_sigma=d.gyro_sigma * k, gyro_bias_walk=d.gyro_bias_walk * k, accel_sigma=d.accel_sigma * k))
	return f


# (cell id, axis, level label, primary?, transform). Order = the interleaving order.
CELLS = (
	("nom", "anchor", "training values", True, lambda ec: ec),
	("wear30", "wear", "motor_asym_mag 0.30", True, _wear(0.30)),
	("mass25", "mass", "mass_jitter 0.25", True, _mass(0.25)),
	("lag30", "lag", "motor_lag_s T 0.30", True, _lag(0.30)),
	("lag10", "lag", "motor_lag_s T 0.10", True, _lag(0.10)),
	("noise2", "noise", "sensor sigma x2", True, _noise(2.0)),
	("wear25", "wear", "motor_asym_mag 0.25", False, _wear(0.25)),
	("wear40", "wear", "motor_asym_mag 0.40", False, _wear(0.40)),
	("mass35", "mass", "mass_jitter 0.35", False, _mass(0.35)),
	("lag40", "lag", "motor_lag_s T 0.40", False, _lag(0.40)),
	("noise3", "noise", "sensor sigma x3", False, _noise(3.0)),
)
CELL = {c[0]: c for c in CELLS}

WNN_TAG = "C-S2-d_wposx2_a63_rf_c200_dc_biasin_ms01_P_both_L1_yld_gt_l2b_full_s{s}"
WNN_SEEDS = (1, 2, 3, 4, 5)
# Rivals (classical_baseline.teacher_metrics): (id, teacher_id, ctbr_outer_hold, est-fed, label).
# hold None = the run's --action-repeat (10). Row B (eq-info MPCOF, THE comparator) is not
# reachable from Python: rowb.sh scores it through the cargo probe on a per-cell fixture.
RIVALS = (
	("mpcofA", 4, None, True, "MPCOF>rate est-fed, no bias comp (row A)"),
	("mpcofC", 4, None, False, "MPCOF>rate oracle-fed (row C, bound / label law)"),
	("pid", 0, 0, True, "PID firmware cascade, est-fed (row F)"),
	("lqi", 3, 0, True, "LQI motor-level, est-fed"),
)
CONTROLLERS = tuple(f"wnn_s{s}" for s in WNN_SEEDS) + tuple(r[0] for r in RIVALS)


def apply(cell_id, ec):
	return CELL[cell_id][4](ec)
