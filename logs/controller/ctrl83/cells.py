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

CELL `all` RE-LEVELLED (Luiz 28/11/2026 — prereg Amendment 3, "Zhang rule on the l2b reference ranges"):
Zhang 2022 Table I rule — every test bound = training lower x0.8 / training upper x1.2 — applied to ONE FIXED
reference, the training ranges of the ORIGINAL base l2b (never an arm's own: the yardstick must not move with the
arm). l2b trained wear factor 0.80-1.20, mass 0.85-1.15, lag T 0.20 fixed, noise x1  ->  `all` = wear factor
U(0.64, 1.44) per motor (EXACT: fixed base 1.04 x (1 + U(-0.3846, 0.3846)), the multiplier composition
apply_motor_fault already uses), mass 1 + U(-0.38, 0.38) (SYMMETRIC FALLBACK, 0.62-1.38: the mass draw has no base
factor, so the exact 0.68-1.38 needs code; harsher at the low end), lag T_e ~ U(0.16, 0.24) s, noise x1.2 FIXED
(the training noise is a point, so the Zhang interval is [x0.8, x1.2]; less noise is not a stress, the scorer has no
per-episode noise draw, x1.2 = the range's worst case). Inertia +-20 % and latency 8-12 ms are PENDING (ABI 70,
branch ctrl89-latency-inertia, not installed): recorded in ALL_LEVELS, NOT applied, never scored on ABI 69. When
they join, the ABI-69 `all` caches are renamed like all_v1 was and the cell re-scored.
The Amendment-2 cell (wear +-30 %, mass +-25 %, lag U(0.15, 0.30), noise x2) is kept, scorable, as `all_v1`
(diagnostic): every number quoted as "all" before this amendment is an all_v1 number (caches renamed
data/<k>__all_v1.npz; their meta still says cell "all" — that is the v1 definition).

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


# The FIXED Zhang reference (Amendment 3): the ORIGINAL base l2b's training ranges. Never an arm's own.
ZHANG_REF = dict(wear_factor=(0.80, 1.20), mass_factor=(0.85, 1.15), lag_T=(0.20, 0.20), noise=(1.0, 1.0))
ZHANG_LO, ZHANG_HI = 0.8, 1.2   # Zhang 2022 Table I: test = train lower x0.8, train upper x1.2

# CELL `all_v1` levels (Amendment 2, scored 28/11 on ABI 69 — superseded as the headline by Amendment 3).
ALL_LEVELS_V1 = dict(wear=0.30, mass=0.25, lag_range=(0.15, 0.30), noise=2.0)
# CELL `all` levels (Amendment 3, Luiz 28/11): Zhang on ZHANG_REF. wear_range = the per-motor thrust factor range
# (exact); mass = symmetric jitter (fallback for 0.68-1.38, see the module doc); lag_range s; noise = fixed multiplier.
# PENDING (requires ABI 70, branch ctrl89-latency-inertia dc222d3d; NOT applied, NOT scored on ABI 69):
#   inertia    -> EpisodeConfig.inertia_jitter 0.20 (diag scale U(0.8, 1.2); l2b trained the nominal inertia)
#   latency_ms -> EpisodeConfig.control_latency_ms (8.0, 12.0) (Zhang's 10 ms observation latency x0.8 / x1.2)
ALL_LEVELS = dict(wear_range=(0.64, 1.44), mass=0.38, lag_range=(0.16, 0.24), noise=1.2,
                  inertia=0.20, latency_ms=(8.0, 12.0))
ALL_PENDING_ABI70 = ("inertia", "latency_ms")


def _wear_range(lo, hi):
	"""Per-motor thrust factor U(lo, hi), exact: fixed base b = (lo+hi)/2 times the scorer's own (1 + U(-m, m)),
	m = (hi-lo)/(hi+lo). Composes like apply_motor_fault (the base multiplier flows to every scorer's resolved draw)."""
	def f(ec):
		d = ec.disturbance
		if any(abs(float(x) - 1.0) > 1e-12 for x in d.motor_asym):
			raise SystemExit(f"ctrl83: wear range on a faulted base multiplier {d.motor_asym} — not the amended plant")
		b, m = 0.5 * (lo + hi), (hi - lo) / (hi + lo)
		return dc.replace(ec, disturbance=dc.replace(d, motor_asym=(b, b, b, b), motor_asym_mag=float(m)))
	return f


def _check_joint_plant(ec, cell):
	if float(getattr(ec, "motor_lag_down_s", 0.0)) != 0.0 or float(getattr(ec, "pwm_min", 0.0)) != 0.0:
		raise SystemExit(f"ctrl83: cell {cell} on an asymmetric/floored actuator — not the amended plant")
	if float(ec.motor_lag_s) != float(NOMINAL["lag_T"]):
		raise SystemExit(f"ctrl83: cell {cell} needs the nominal plant lag {NOMINAL['lag_T']} (got {ec.motor_lag_s})")


def _per_episode_lag(ec, rng):
	# The plant lag is drawn PER EPISODE (Rust); every MODEL stays at the nominal design lag (D1).
	return dc.replace(ec, model_motor_lag_s=float(NOMINAL["lag_T"]), stress_motor_lag_range=tuple(float(x) for x in rng))


def _all_v1(ec):
	_check_joint_plant(ec, "all_v1")
	L = ALL_LEVELS_V1
	return _per_episode_lag(_noise(L["noise"])(_mass(L["mass"])(_wear(L["wear"])(ec))), L["lag_range"])


# CTRL-90 SECONDARY readout (docs/ctrl90_zhang_trained_prereg.md): Zhang x0.8/x1.2 applied to the l2bnmwz arm's OWN
# training ranges = the honest OOD read for an arm trained on `all`. Arm trains wear ±0.44 (0.56-1.44; the symmetric
# fallback, no asymmetric training flag), lag U(0.16, 0.24), mass ±0.15 and noise x1 (no training-only flags) ->
# wear 0.448-1.728 (exact, base x mag), lag U(0.128, 0.288), mass ±0.38 (fallback for 0.68-1.38), noise x1.2.
# Report-only for l2bnmwz (+ its l2bnmw pair); NOT a headline. Inertia/latency pending ABI 70 as in `all`.
ALLZ90_LEVELS = dict(wear_range=(0.448, 1.728), mass=0.38, lag_range=(0.128, 0.288), noise=1.2)


def _allz90(ec):
	_check_joint_plant(ec, "allz90")
	L = ALLZ90_LEVELS
	return _per_episode_lag(_noise(L["noise"])(_mass(L["mass"])(_wear_range(*L["wear_range"])(ec))), L["lag_range"])


def _all(ec):
	_check_joint_plant(ec, "all")
	L = ALL_LEVELS
	ec = _noise(L["noise"])(_mass(L["mass"])(_wear_range(*L["wear_range"])(ec)))
	# ALL_PENDING_ABI70 (inertia, latency_ms) deliberately NOT applied: ABI 69 has no such plant fields.
	return _per_episode_lag(ec, L["lag_range"])


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
	("all", "joint", "Zhang on l2b: wear U(0.64,1.44) + mass ±38% + lag T_e~U(0.16,0.24) s + noise x1.2"
	                 " [inertia ±20% + latency 8-12 ms pending ABI 70]", True, _all),
	# Amendment 2's joint cell, superseded by Amendment 3 (kept scorable; every pre-amendment "all" number).
	("all_v1", "joint", "wear ±30% + mass ±25% + lag T_e~U(0.15,0.30) s + noise x2", False, _all_v1),
	("allz90", "joint", "Zhang on l2bnmwz's own training: wear U(0.448,1.728) + mass ±38% + lag U(0.128,0.288) s"
	                    " + noise x1.2 (CTRL-90 secondary)", False, _allz90),
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
	# CTRL-90 (prereg docs/ctrl90_zhang_trained_prereg.md): l2bnmw trained ON the Zhang `all` ranges.
	"l2bnmwz": "C-S2-d_wposx2_a69_rf_c200_dc_biasin_ms01_P_both_L1_yld_gt_l2bnmwz_full_s{s}",
}
# The lag range each joint cell draws per episode (score_cell.py G3).
CELL_LAG_RANGE = {"all": ALL_LEVELS["lag_range"], "all_v1": ALL_LEVELS_V1["lag_range"],
                  "allz90": ALLZ90_LEVELS["lag_range"]}
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
