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

CELL `all` (ABI 69, Luiz 28/11/2026 — prereg Amendment 2): THE single joint-OOD headline,
"the default standard golden value to pursue". Every episode draws ALL the stresses at once:
wear ±30 %, mass spread ±25 %, a per-episode plant lag T_e ~ U(0.15, 0.30) s, sensor noise
x2; everything else is the TEST preset (35 CF2.1 units x 100 episodes). The lag is drawn in
Rust (train_lag_range::scorer_episode_lags, keyed on each episode's weather seed) by ALL
three scorers — the WNN CPU/Metal scorers and score_classical_baseline (rivals, row-B probe) —
from ONE kwarg, train_motor_lag_range, carried by EpisodeConfig.stress_motor_lag_range via
stress_kwargs(). Every model (the WNN twin, the rivals' designs) stays at the nominal 0.20 (D1).
From 28/11 `all` is the ONLY robustness number reported; the one-factor cells below are
DIAGNOSTIC (kept, scorable, never reported). `nom` stays as the in-range reference + G4 gate.

CTRL-83 fairness fixes (ABI 66, Luiz 28/11 "fix it properly"):
  D1  a lag cell changes the PLANT's lag only. Every controller's design-time model stays
      at the nominal T = 0.20 (EpisodeConfig.model_motor_lag_s): MPCOF / LQI are no longer
      re-derived for the stressed aircraft, and the WNN's actuator twin is held at 0.20
      too. Audited: no rival re-derives from wear, noise, rotor drag or the mass draw (hover
      and altitude PD are built on the nominal mass), so only the lag cells carry it. At
      T = 0.20 the override is the plant's own value — bit-identical (ctrl83_flag_off_pins).
      Lag cells scored before ABI 66 were matched-model and must be re-scored.
  D3  payload cells (SECONDARY, added after the prereg on Luiz's 28/11 brief; no rank
      claim): the plant mass becomes the episode's draw x scale at t_step — the identical
      event on every scorer; no controller is told, every model stays nominal.
  D2  the collective anchor keeps the lineage's TRUE-mass anchor (EpisodeConfig.anchor_mass
      'true'): the published WNN was trained on it, so switching it here would be a
      distribution shift, not a stress (see the ABI-66 D2 note).
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
		# D1: the plant flies T; every controller's MODEL stays at the nominal design lag.
		return dc.replace(ec, motor_lag_s=float(T), model_motor_lag_s=float(NOMINAL["lag_T"]))
	return f


# CELL `all` levels (Amendment 2): the four one-factor PRIMARY levels, flown jointly.
ALL_LEVELS = dict(wear=0.30, mass=0.25, lag_range=(0.15, 0.30), noise=2.0)


def _all(ec):
	if float(getattr(ec, "motor_lag_down_s", 0.0)) != 0.0 or float(getattr(ec, "pwm_min", 0.0)) != 0.0:
		raise SystemExit("ctrl83: cell all on an asymmetric/floored actuator — not the amended plant")
	if float(ec.motor_lag_s) != float(NOMINAL["lag_T"]):
		raise SystemExit(f"ctrl83: cell all needs the nominal plant lag {NOMINAL['lag_T']} (got {ec.motor_lag_s})")
	ec = _noise(ALL_LEVELS["noise"])(_mass(ALL_LEVELS["mass"])(_wear(ALL_LEVELS["wear"])(ec)))
	# The plant lag is drawn PER EPISODE (Rust); every MODEL stays at the nominal design lag (D1).
	return dc.replace(ec, model_motor_lag_s=float(NOMINAL["lag_T"]),
	                  stress_motor_lag_range=tuple(float(x) for x in ALL_LEVELS["lag_range"]))


def _payload(scale, t_s):
	def f(ec):
		return dc.replace(ec, payload_step_s=float(t_s), payload_mass_scale=float(scale))
	return f


def _noise(k):
	def f(ec):
		d = ec.disturbance
		return dc.replace(ec, disturbance=dc.replace(
			d, gyro_sigma=d.gyro_sigma * k, gyro_bias_walk=d.gyro_bias_walk * k, accel_sigma=d.accel_sigma * k))
	return f


# (cell id, axis, level label, primary?, transform). Order = the interleaving order.
# primary? (Amendment 2, 28/11): True ONLY for the reported cells — `all` (THE headline) and `nom`
# (in-range reference + G4). The former primary one-factor cells (wear30 .. noise2) are diagnostic.
CELLS = (
	("nom", "anchor", "training values", True, lambda ec: ec),
	("all", "joint", "wear ±30% + mass ±25% + lag T_e~U(0.15,0.30) s + noise x2", True, _all),
	("wear30", "wear", "motor_asym_mag 0.30", False, _wear(0.30)),
	("mass25", "mass", "mass_jitter 0.25", False, _mass(0.25)),
	("lag30", "lag", "motor_lag_s T 0.30", False, _lag(0.30)),
	("lag10", "lag", "motor_lag_s T 0.10", False, _lag(0.10)),
	("noise2", "noise", "sensor sigma x2", False, _noise(2.0)),
	("wear25", "wear", "motor_asym_mag 0.25", False, _wear(0.25)),
	("wear40", "wear", "motor_asym_mag 0.40", False, _wear(0.40)),
	("mass35", "mass", "mass_jitter 0.35", False, _mass(0.35)),
	("lag40", "lag", "motor_lag_s T 0.40", False, _lag(0.40)),
	("noise3", "noise", "sensor sigma x3", False, _noise(3.0)),
	# D3 (ABI 66), secondary: a payload added / dropped mid-episode (t = 1.0 s of the 2.0 s episode).
	("pay25", "payload", "mass x1.25 at t = 1.0 s", False, _payload(1.25, 1.0)),
	("pay50", "payload", "mass x1.50 at t = 1.0 s", False, _payload(1.50, 1.0)),
	("drop20", "payload", "mass x0.80 at t = 1.0 s", False, _payload(0.80, 1.0)),
)
CELL = {c[0]: c for c in CELLS}
PRIMARY = tuple(c[0] for c in CELLS if c[3])         # ("nom", "all") — the reported cells
DIAGNOSTIC = tuple(c[0] for c in CELLS if not c[3])  # one-factor cells: diagnostic-only, not reported

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
# CTRL-87 (28/11): EXTRA WNN controllers re-scored on this grid after each END (report-only readout,
# docs/ctrl87_nominal_mass_wide_spread_prereg.md). id = "<arm>_s<N>". NOT in the default CONTROLLERS (their
# winners land one END at a time): name them with run_grid.py --controllers. The grid itself is unchanged.
EXTRA_WNN = {
	"l2bnm": "C-S2-d_wposx2_a66_rf_c200_dc_biasin_ms01_P_both_L1_yld_gt_l2bnm_full_s{s}",
	"l2bnmw": "C-S2-d_wposx2_a68_rf_c200_dc_biasin_ms01_P_both_L1_yld_gt_l2bnmw_full_s{s}",
}
EXTRA_CONTROLLERS = tuple(f"{arm}_s{s}" for arm in EXTRA_WNN for s in WNN_SEEDS)
ALL_CONTROLLERS = CONTROLLERS + EXTRA_CONTROLLERS


def is_wnn(k):
	return k.startswith("wnn_s") or k in EXTRA_CONTROLLERS


def wnn_tag(k):
	"""Run tag of a WNN controller id (wnn_sN = the l2b reference; <arm>_sN = an EXTRA_WNN arm)."""
	if k.startswith("wnn_s"):
		return WNN_TAG.format(s=int(k[5:]))
	arm, s = k.rsplit("_s", 1)
	return EXTRA_WNN[arm].format(s=int(s))


def apply(cell_id, ec):
	return CELL[cell_id][4](ec)
